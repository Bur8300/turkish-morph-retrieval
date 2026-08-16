"""Preview-only generation through the locally authenticated Codex CLI."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .planner import build_plan, plan_hash, plan_statistics
from .prompts import GENERATOR_SYSTEM, PROMPT_VERSION, build_generation_prompt, build_repair_prompt
from .providers import ProviderError, make_provider
from .schema import GENERATION_SCHEMA
from .selection import selection_statistics
from .validators import artifact_report, normalize_family, validate_family


HERE = Path(__file__).resolve().parent


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def _batch_schema(size: int) -> dict[str, Any]:
    return {
        "name": "turkish_morph_preview_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "families": {
                    "type": "array",
                    "minItems": size,
                    "maxItems": size,
                    "items": GENERATION_SCHEMA["schema"],
                }
            },
            "required": ["families"],
        },
    }


def _batch_prompt(slots: list[dict]) -> str:
    tasks = []
    for number, slot in enumerate(slots, start=1):
        tasks.append(f"\n===== TASK {number}/{len(slots)} =====\n{build_generation_prompt(slot)}")
    return f"""\
Bu çağrı yalnız PREVIEW veri üretimidir. Aşağıdaki {len(slots)} görevi
birbirinden bağımsız tamamla ve sonuçları verilen sırayı koruyarak `families` dizisine koy.
Her görev kendi SLOT kimliklerini ve bütün kalite kurallarını eksiksiz karşılamalıdır.
Dosya okuma, araç kullanma veya açıklama yazma; yalnız şemadaki JSON'u üret.
{''.join(tasks)}
"""


def _blind_item(item: dict) -> dict[str, Any]:
    return {
        "family_id": item["family_id"],
        "query": item["query"],
        "candidates": [{"id": row["id"], "text": row["text"]} for row in item["candidates"]],
    }


def _accept(raw: dict, slot: dict, cfg: dict, provenance: dict) -> tuple[dict | None, list[str]]:
    try:
        family = normalize_family(raw, slot)
        problems = validate_family(family, slot, cfg)
    except Exception as exc:
        return None, [f"normalizasyon hatası: {type(exc).__name__}: {exc}"]
    if problems:
        return family, problems
    family["generator_id"] = "codex_preview"
    family["source_type"] = "codex_cli_preview_unjudged"
    family["preview_only"] = True
    family["provenance"] = provenance
    family["qc"] = {
        "deterministic": "pass",
        "independent_judge": "not_run",
    }
    return family, []


def _process_batch(batch, provider, cfg, batch_number: int, batch_total: int, cache_only: bool):
    print(f"Codex preview batch {batch_number}/{batch_total}: {len(batch)} family", flush=True)
    response = provider.call_json(
        GENERATOR_SYSTEM,
        _batch_prompt(batch),
        _batch_schema(len(batch)),
        f"codex_preview_batch_{batch[0]['index']:04d}",
    )
    raw_families = response.data.get("families", [])
    by_frame = {
        row.get("semantic_frame_id"): row
        for row in raw_families if isinstance(row, dict)
    }
    accepted, rejected = [], []
    for slot in batch:
        raw = by_frame.get(slot["semantic_frame_id"])
        if raw is None:
            rejected.append({
                "slot_id": slot["slot_id"],
                "problems": ["batch çıktısında semantic_frame_id eksik"],
            })
            continue
        provenance = {
            "provider": response.provider,
            "model": response.model,
            "request_hash": response.request_hash,
            "cache_hit": response.cache_hit,
            "prompt_version": PROMPT_VERSION,
            "independent_judge": False,
        }
        family, problems = _accept(raw, slot, cfg, provenance)
        if problems:
            try:
                repair = provider.call_json(
                    GENERATOR_SYSTEM,
                    build_repair_prompt(slot, raw, problems),
                    GENERATION_SCHEMA,
                    f"codex_preview_repair_{slot['slot_id']}",
                )
            except ProviderError:
                if not cache_only:
                    raise
            else:
                provenance["repair_request_hash"] = repair.request_hash
                family, problems = _accept(repair.data, slot, cfg, provenance)
        refill_hashes = []
        for refill_round in range(1, 3):
            if not problems:
                break
            prompt_slot = {**slot, "preview_refill_round": refill_round}
            try:
                refill = provider.call_json(
                    GENERATOR_SYSTEM,
                    build_generation_prompt(prompt_slot),
                    GENERATION_SCHEMA,
                    f"codex_preview_refill_{slot['slot_id']}_{refill_round}",
                )
            except ProviderError:
                if not cache_only:
                    raise
                break
            refill_hashes.append(refill.request_hash)
            provenance["refill_request_hashes"] = list(refill_hashes)
            family, problems = _accept(refill.data, slot, cfg, provenance)
        if problems:
            rejected.append({
                "slot_id": slot["slot_id"],
                "problems": problems,
                "last_family": family,
            })
        else:
            accepted.append(family)
    print(
        f"Codex preview batch {batch_number}/{batch_total} tamamlandı: "
        f"accepted={len(accepted)}, rejected={len(rejected)}",
        flush=True,
    )
    return accepted, rejected


def generate_codex_preview(
    run_id: str,
    count: int = 60,
    batch_size: int = 5,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "medium",
    config_path: str | None = None,
    workers: int = 1,
    reserve_slots: int = 0,
    cache_only: bool = False,
) -> dict[str, Any]:
    if count < 1 or count > 60:
        raise ValueError("Preview count 1–60 arasında olmalı")
    if batch_size < 1 or batch_size > 10:
        raise ValueError("Preview batch-size 1–10 arasında olmalı")
    if workers < 1 or workers > 6:
        raise ValueError("Preview workers 1–6 arasında olmalı")
    if reserve_slots < 0 or count + reserve_slots > 60:
        raise ValueError("Preview count + reserve-slots toplamı 1–60 arasında olmalı")
    cfg = load_config(config_path, runtime=False)
    slots = build_plan(cfg, size=count + reserve_slots)
    root = HERE / "previews" / run_id
    cache = root / "cache"
    isolated_workdir = root / "isolated_workdir"
    root.mkdir(parents=True, exist_ok=True)
    isolated_workdir.mkdir(parents=True, exist_ok=True)
    spec = {
        "provider": "codex_cli",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": 1800,
        # Do not expose repository JSON/train examples as the model's working context.
        "workdir": str(isolated_workdir),
        "cache_only": cache_only,
    }
    metadata = {
        "dataset_version": cfg["version"],
        "prompt_version": PROMPT_VERSION,
        "preview_only": True,
    }
    provider = make_provider(spec, cache, metadata)
    accepted: list[dict] = []
    rejected: list[dict] = []
    started = datetime.now(timezone.utc)

    batches = [slots[offset : offset + batch_size] for offset in range(0, len(slots), batch_size)]
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as executor:
        futures = {
            executor.submit(
                _process_batch, batch, provider, cfg, number, len(batches), cache_only
            ): number
            for number, batch in enumerate(batches, start=1)
        }
        for future in as_completed(futures):
            batch_accepted, batch_rejected = future.result()
            accepted.extend(batch_accepted)
            rejected.extend(batch_rejected)
            accepted.sort(key=lambda item: item["slot_id"])
            rejected.sort(key=lambda item: item["slot_id"])
            _write_jsonl(root / "accepted.jsonl", accepted)
            _write_jsonl(root / "rejected.jsonl", rejected)

    accepted_reserve = accepted[count:]
    accepted = accepted[:count]
    _write_jsonl(root / "accepted.jsonl", accepted)
    _write_jsonl(root / "accepted_reserve.jsonl", accepted_reserve)

    internal = {
        "dataset_name": cfg["dataset_name"],
        "version": cfg["version"],
        "split": "codex_preview_unjudged",
        "warning": (
            "Preview only: generated by Codex CLI and deterministic QC passed, but no independent "
            "model judge was run. Do not report as benchmark data."
        ),
        "statistics": selection_statistics(accepted),
        "items": accepted,
    }
    blind = {
        "dataset_name": cfg["dataset_name"],
        "version": cfg["version"],
        "split": "codex_preview_blind",
        "warning": internal["warning"],
        "items": [_blind_item(item) for item in accepted],
    }
    _write_json(root / "preview_internal.json", internal)
    _write_json(root / "preview_blind.json", blind)
    artifacts = artifact_report(accepted)
    _write_json(root / "artifact_audit.json", artifacts)
    manifest = {
        "run_id": run_id,
        "created_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provider": "codex_cli_chatgpt_login",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "workers": workers,
        "api_key_used": False,
        "isolated_workdir": True,
        "few_shot_dataset_examples_used": False,
        "cache_only": cache_only,
        "independent_judge_run": False,
        "requested": count,
        "planned_slots": len(slots),
        "accepted": len(accepted),
        "accepted_reserve": len(accepted_reserve),
        "rejected": len(rejected),
        "plan_sha256": plan_hash(slots),
        "plan_statistics": {
            **plan_statistics(slots),
            "generator_id": {"codex_preview": len(slots)},
        },
        "accepted_role_counts": dict(Counter(
            candidate["role"] for family in accepted for candidate in family["candidates"]
        )),
        "artifact_audit": artifacts,
        "files": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "accepted.jsonl", "rejected.jsonl", "preview_internal.json",
                "preview_blind.json", "artifact_audit.json", "accepted_reserve.jsonl",
            )
        },
        "warning": internal["warning"],
    }
    _write_json(root / "manifest.json", manifest)
    return manifest
