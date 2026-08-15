"""Preview-only generation through the locally authenticated Codex CLI."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .planner import build_plan, plan_hash, plan_statistics
from .prompts import GENERATOR_SYSTEM, PROMPT_VERSION, build_generation_prompt, build_repair_prompt
from .providers import make_provider
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
Bu çağrı yalnız insanın inceleyeceği PREVIEW veri üretimidir. Aşağıdaki {len(slots)} görevi
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


def _write_review_markdown(path: Path, items: list[dict]) -> None:
    lines = [
        "# Codex preview — kör insan inceleme görünümü",
        "",
        "> Preview only: bağımsız LLM judge ve insan onayı henüz yoktur. "
        "Aşağıda gold/role/subtype gösterilmez.",
        "",
    ]
    for number, item in enumerate(items, start=1):
        lines.extend([
            f"## {number}. {item['family_id']}",
            "",
            f"- Hedef: `{item['target_feature']}` — {item['target_feature_label']}",
            f"- Objective/layer: `{item['objective']}` / `{item['layer']}`",
            f"- Query: {item['query']}",
            "",
            "Adaylar:",
            "",
        ])
        for candidate in item["candidates"]:
            lines.append(f"- `{candidate['id']}` — {candidate['text']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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
        "human_review": "pending",
    }
    return family, []


def generate_codex_preview(
    run_id: str,
    count: int = 60,
    batch_size: int = 5,
    model: str = "gpt-5.6-sol",
    reasoning_effort: str = "medium",
    config_path: str | None = None,
) -> dict[str, Any]:
    if count < 1 or count > 60:
        raise ValueError("Preview count 1–60 arasında olmalı")
    if batch_size < 1 or batch_size > 10:
        raise ValueError("Preview batch-size 1–10 arasında olmalı")
    cfg = load_config(config_path, runtime=False)
    slots = build_plan(cfg, size=count)
    root = HERE / "previews" / run_id
    cache = root / "cache"
    root.mkdir(parents=True, exist_ok=True)
    spec = {
        "provider": "codex_cli",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": 1800,
        "workdir": str(HERE.parent),
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

    for offset in range(0, len(slots), batch_size):
        batch = slots[offset : offset + batch_size]
        print(
            f"Codex preview batch {offset // batch_size + 1}/"
            f"{(len(slots) + batch_size - 1) // batch_size}: {len(batch)} family",
            flush=True,
        )
        response = provider.call_json(
            GENERATOR_SYSTEM,
            _batch_prompt(batch),
            _batch_schema(len(batch)),
            f"codex_preview_batch_{offset:04d}",
        )
        raw_families = response.data.get("families", [])
        by_frame = {
            row.get("semantic_frame_id"): row
            for row in raw_families if isinstance(row, dict)
        }
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
                repair = provider.call_json(
                    GENERATOR_SYSTEM,
                    build_repair_prompt(slot, raw, problems),
                    GENERATION_SCHEMA,
                    f"codex_preview_repair_{slot['slot_id']}",
                )
                provenance["repair_request_hash"] = repair.request_hash
                family, problems = _accept(repair.data, slot, cfg, provenance)
            if problems:
                rejected.append({
                    "slot_id": slot["slot_id"],
                    "problems": problems,
                    "last_family": family,
                })
            else:
                accepted.append(family)
        _write_jsonl(root / "accepted.jsonl", accepted)
        _write_jsonl(root / "rejected.jsonl", rejected)

    internal = {
        "dataset_name": cfg["dataset_name"],
        "version": cfg["version"],
        "split": "codex_preview_unjudged",
        "warning": (
            "Preview only: generated by Codex CLI, deterministic QC passed, but no independent "
            "model judge or human review was run. Do not report as benchmark data."
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
    _write_review_markdown(root / "preview_review.md", accepted)
    artifacts = artifact_report(accepted)
    _write_json(root / "artifact_audit.json", artifacts)
    manifest = {
        "run_id": run_id,
        "created_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "provider": "codex_cli_chatgpt_login",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "api_key_used": False,
        "independent_judge_run": False,
        "requested": count,
        "accepted": len(accepted),
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
                "preview_blind.json", "preview_review.md", "artifact_audit.json",
            )
        },
        "warning": internal["warning"],
    }
    _write_json(root / "manifest.json", manifest)
    return manifest
