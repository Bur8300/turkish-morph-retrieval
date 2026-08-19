"""Generate, validate, audit, and save curated_preview_20_v37 dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from new_items_definitions import RAW_ITEMS
from test.config import load_config
from test.evaluation import (
    EVALUATION_API_VERSION,
    artifact_baseline_summaries,
    closed_qrels,
    evaluate_run,
    load_items,
)
from test.planner import build_plan, plan_hash, plan_statistics
from test.selection import selection_statistics
from test.validators import artifact_report, corpus_problems, normalize_family, validate_family

HERE = Path(__file__).resolve().parent

def _blind_item(item: dict) -> dict[str, Any]:
    return {
        "family_id": item["family_id"],
        "query": item["query"],
        "candidates": [{"id": row["id"], "text": row["text"]} for row in item["candidates"]],
    }

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

def main() -> None:
    cfg = load_config()
    # Build deterministic plan for 50 slots and select the 20 slots [26:46]
    # This guarantees 100% disjoint frame and slot IDs from curated_preview_20_v36
    all_slots = build_plan(cfg, size=50)
    slots = all_slots[26:46]
    assert len(slots) == 20, f"Expected 20 slots, got {len(slots)}"
    assert len(RAW_ITEMS) == 20, f"Expected 20 raw items, got {len(RAW_ITEMS)}"

    accepted: list[dict] = []
    validation_problems: dict[str, list[str]] = {}

    for idx, (slot, raw) in enumerate(zip(slots, RAW_ITEMS), start=1):
        family = normalize_family(raw, slot)
        family["generator_id"] = "curated_expert"
        family["source_type"] = "curated_preview_unjudged"
        family["preview_only"] = True
        family["provenance"] = {
            "source": "curated_expert_generation",
            "dataset_version": cfg["version"],
            "slot_index": slot["index"],
            "slot_id": slot["slot_id"],
            "independent_judge": False,
        }
        family["qc"] = {
            "deterministic": "pass",
            "independent_judge": "not_run",
        }
        probs = validate_family(family, slot, cfg)
        if probs:
            validation_problems[family["family_id"]] = probs
            print(f"Validation failure on slot {slot['index']} ({family['family_id']}): {probs}")
        accepted.append(family)

    if validation_problems:
        raise ValueError(f"Family validation failed for {len(validation_problems)} items: {validation_problems}")

    c_probs = corpus_problems(accepted, cfg)
    if c_probs:
        raise ValueError(f"Corpus problems found: {c_probs}")

    artifacts = artifact_report(accepted)
    print("Artifact audit summary:")
    print("Tie-aware recall@1:", artifacts["recall_at_1_tie_aware"])
    print("Lexical balance:", artifacts["lexical_balance"])

    # Check tie-aware freeze thresholds
    quality = cfg["quality"]
    word_r1 = artifacts["recall_at_1_tie_aware"]["word_overlap"]
    char_r1 = artifacts["recall_at_1_tie_aware"]["character_3gram"]
    bm25_r1 = artifacts["recall_at_1_tie_aware"]["bm25"]
    max_word = float(quality["freeze_tie_aware_word_overlap_recall_at_1_max"])
    max_char = float(quality["freeze_tie_aware_character_3gram_recall_at_1_max"])
    max_bm25 = float(quality["freeze_tie_aware_bm25_recall_at_1_max"])

    assert word_r1 <= max_word, f"Word overlap R@1 {word_r1} > max {max_word}"
    assert char_r1 <= max_char, f"Char 3-gram R@1 {char_r1} > max {max_char}"
    assert bm25_r1 <= max_bm25, f"BM25 R@1 {bm25_r1} > max {max_bm25}"

    run_id = "curated_preview_20_v37"
    out_dir = HERE / "test" / "previews" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    internal = {
        "dataset_name": cfg["dataset_name"],
        "version": cfg["version"],
        "split": "curated_preview_unjudged",
        "warning": (
            "Curated preview; generated with deterministic QC passed, but no independent "
            "model judge was run. Do not report as benchmark data."
        ),
        "statistics": selection_statistics(accepted),
        "items": accepted,
    }
    blind = {
        "dataset_name": cfg["dataset_name"],
        "version": cfg["version"],
        "split": "curated_preview_blind",
        "warning": internal["warning"],
        "items": [_blind_item(item) for item in accepted],
    }

    _write_json(out_dir / "preview_internal.json", internal)
    _write_json(out_dir / "preview_blind.json", blind)
    _write_json(out_dir / "artifact_audit.json", artifacts)
    _write_jsonl(out_dir / "accepted.jsonl", accepted)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "curated_expert_preview_v37 (slots 26-45)",
        "count": len(accepted),
        "family_modes": dict(Counter(item["family_mode"] for item in accepted)),
        "query_expression": dict(Counter(item["query_expression"] for item in accepted)),
        "lexical_bands": dict(Counter(item["query_gold_lexical_band"] for item in accepted)),
        "validation_problems": {},
        "artifact_audit": artifacts,
        "corpus_problems": [],
        "files": {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in ("accepted.jsonl", "preview_internal.json", "preview_blind.json", "artifact_audit.json")
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(f"\nSuccessfully created and validated {len(accepted)} test families in {out_dir}")

if __name__ == "__main__":
    main()
