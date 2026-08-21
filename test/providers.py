"""Resumable OpenRouter JSON calls with complete cache provenance."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProviderError(RuntimeError):
    pass


@dataclass
class ProviderResponse:
    data: dict[str, Any]
    usage: dict[str, Any]
    cache_hit: bool
    request_hash: str
    model: str
    provider: str
    actual_model: str | None = None
    route_provider: str | None = None


class RateLimiter:
    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / max(float(requests_per_minute), 0.01)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            if delay:
                time.sleep(delay)
            self._next = time.monotonic() + self.interval


def _extract_json(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ProviderError(f"Model geçerli JSON döndürmedi: {exc}") from exc
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as inner:
            raise ProviderError(f"Model JSON'u ayrıştırılamadı: {inner}") from inner
    if not isinstance(value, dict):
        raise ProviderError("Model çıktısının kökü JSON object olmalı")
    return value


class OpenRouterProvider:
    def __init__(self, spec: dict[str, Any], cache_dir: Path, run_metadata: dict[str, Any]):
        self.spec = dict(spec)
        self.model = self.spec["model"]
        self.provider = "openrouter"
        self.cache_dir = cache_dir / self.provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_metadata = dict(run_metadata)
        self.limiter = RateLimiter(self.spec.get("requests_per_minute", 20))
        self._write_lock = threading.Lock()

    def _identity(self, system: str, prompt: str, schema: dict, purpose: str) -> tuple[str, dict]:
        identity = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.spec.get("base_url"),
            "purpose": purpose,
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "temperature": self.spec.get("temperature", 0.1),
            "max_tokens": self.spec.get("max_tokens", 5000),
            "provider_preferences": self.spec.get("provider_preferences", {}),
            "pipeline": self.run_metadata,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), identity

    def call_json(self, system: str, prompt: str, schema: dict, purpose: str) -> ProviderResponse:
        request_hash, identity = self._identity(system, prompt, schema, purpose)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ProviderResponse(
                cached["data"], cached.get("usage", {}), True, request_hash, self.model,
                self.provider, cached.get("response_model"), cached.get("route_provider"),
            )

        key = os.getenv(self.spec["api_key_env"])
        if not key:
            raise ProviderError(f"Eksik API anahtarı: {self.spec['api_key_env']}")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.spec.get("temperature", 0.1),
            "max_tokens": self.spec.get("max_tokens", 5000),
            "response_format": {"type": "json_schema", "json_schema": schema},
            "provider": {
                "require_parameters": True,
                **self.spec.get("provider_preferences", {}),
            },
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Bur8300/turkish-morph-retrieval",
            "X-Title": "Turkish Morph Retrieval Test Builder",
        }

        last_error: Exception | None = None
        for attempt in range(5):
            self.limiter.wait()
            req = urllib.request.Request(self.spec["base_url"], data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=float(self.spec.get("timeout_seconds", 180))) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                content = raw["choices"][0]["message"]["content"]
                data = _extract_json(content)
                record = {
                    "identity": identity,
                    "data": data,
                    "usage": raw.get("usage", {}),
                    "response_model": raw.get("model", self.model),
                    "route_provider": raw.get("provider"),
                    "created_at_unix": int(time.time()),
                }
                tmp = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
                with self._write_lock:
                    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                    tmp.replace(cache_path)
                return ProviderResponse(
                    data, record["usage"], False, request_hash, self.model, self.provider,
                    record["response_model"], record["route_provider"],
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = ProviderError(f"OpenRouter HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ProviderError) as exc:
                last_error = exc
            if attempt < 4:
                time.sleep(min(30.0, (2**attempt) + random.random()))
        raise ProviderError(f"OpenRouter çağrısı başarısız: {last_error}")


class CodexCliProvider:
    """Structured-output calls through the locally authenticated Codex CLI.

    This provider is intended for preview data only. It uses the user's ChatGPT/Codex login,
    not an API key, and therefore cannot serve as an independent judge for its own generations.
    """

    def __init__(self, spec: dict[str, Any], cache_dir: Path, run_metadata: dict[str, Any]):
        self.spec = dict(spec)
        self.model = self.spec.get("model", "gpt-5.6-sol")
        self.provider = "codex_cli"
        self.cache_dir = cache_dir / self.provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_metadata = dict(run_metadata)
        self._write_lock = threading.Lock()
        self.executable = self.spec.get("executable") or shutil.which("codex")
        if not self.executable:
            raise ProviderError("Codex CLI bulunamadı")

    def _identity(self, system: str, prompt: str, schema: dict, purpose: str) -> tuple[str, dict]:
        identity = {
            "provider": self.provider,
            "model": self.model,
            "reasoning_effort": self.spec.get("reasoning_effort", "medium"),
            "purpose": purpose,
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "pipeline": self.run_metadata,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), identity

    def call_json(self, system: str, prompt: str, schema: dict, purpose: str) -> ProviderResponse:
        request_hash, identity = self._identity(system, prompt, schema, purpose)
        cache_path = self.cache_dir / f"{request_hash}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return ProviderResponse(
                cached["data"], cached.get("usage", {}), True, request_hash,
                self.model, self.provider, cached.get("response_model", self.model),
                cached.get("route_provider", "codex_cli"),
            )
        if self.spec.get("cache_only"):
            raise ProviderError(f"Codex cache-only modunda yanıt bulunamadı: {request_hash}")

        request_dir = self.cache_dir / request_hash
        request_dir.mkdir(parents=True, exist_ok=True)
        schema_path = request_dir / "output_schema.json"
        output_path = request_dir / "last_message.json"
        schema_path.write_text(
            json.dumps(schema.get("schema", schema), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = [
            str(self.executable), "exec",
            "--model", str(self.model),
            "--sandbox", "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--config", f'model_reasoning_effort="{self.spec.get("reasoning_effort", "medium")}"',
            "--output-schema", str(schema_path),
            "--output-last-message", str(output_path),
            "--cd", str(Path(self.spec.get("workdir", Path.cwd())).resolve()),
            "-",
        ]
        combined_prompt = (
            "SYSTEM INSTRUCTIONS\n" + system.strip() +
            "\n\nUSER TASK\n" + prompt.strip() +
            "\n\nReturn only the JSON value required by the output schema."
        )
        started = time.monotonic()
        try:
            process = subprocess.run(
                command,
                input=combined_prompt,
                text=True,
                capture_output=True,
                timeout=float(self.spec.get("timeout_seconds", 900)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError(f"Codex CLI zaman aşımı: {exc}") from exc
        if process.returncode != 0 or not output_path.exists():
            detail = (process.stderr or process.stdout)[-4000:]
            raise ProviderError(
                f"Codex CLI başarısız (exit={process.returncode}): {detail}"
            )
        data = _extract_json(output_path.read_text(encoding="utf-8"))
        usage = {
            "wall_seconds": round(time.monotonic() - started, 3),
            "cli_stdout_tail": process.stdout[-1000:],
        }
        record = {
            "identity": identity,
            "data": data,
            "usage": usage,
            "created_at_unix": int(time.time()),
        }
        tmp = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
        with self._write_lock:
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(cache_path)
        return ProviderResponse(
            data, usage, False, request_hash, self.model, self.provider,
            self.model, "codex_cli",
        )


def make_provider(spec: dict[str, Any], cache_dir: Path, run_metadata: dict[str, Any]):
    provider = spec.get("provider")
    if provider == "openrouter":
        return OpenRouterProvider(spec, cache_dir, run_metadata)
    if provider == "codex_cli":
        return CodexCliProvider(spec, cache_dir, run_metadata)
    raise ProviderError(f"Desteklenmeyen test provider'ı: {provider}")
