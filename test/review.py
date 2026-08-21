"""Blind human-review export and consensus application for judge disagreements/audits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .dataset_memory import DatasetMemory, family_memory_tags
from .pipeline import paths_for, read_jsonl
from .prompts import _blind_candidates


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def unresolved_reviews(run_id: str) -> list[dict[str, Any]]:
    paths = paths_for(run_id)
    memory = DatasetMemory(paths.memory)
    latest = {}
    for record in read_jsonl(paths.needs_review):
        if record.get("slot_id"):
            latest[record["slot_id"]] = record
    return [
        latest[slot_id] for slot_id in sorted(latest)
        if memory.slot_status(slot_id) == "needs_review"
    ]


def export_human_review(run_id: str) -> dict[str, Any]:
    paths = paths_for(run_id)
    records = unresolved_reviews(run_id)
    semantic_items = []
    morphology_items = []
    for record in records:
        family = record["family"]
        base = {
            "slot_id": record["slot_id"],
            "family_id": family["family_id"],
            "review_kind": record["review_kind"],
            "reviewers_required": record["reviewers_required"],
            "query": family["query"],
            "candidates": _blind_candidates(family, "human_review"),
        }
        semantic_items.append(base)
        morphology_items.append({
            **base,
            "target_feature": family["target_feature"],
            "target_feature_label": family["target_feature_label"],
            "objective": family["objective"],
            "layer": family["layer"],
        })
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instructions": (
            "Önce semantic_items target_feature bilgisi görülmeden doldurulmalı; sonra "
            "morphology_items incelenmeli. Reviewer gold/role/subtype görmez."
        ),
        "decision_schema": {
            "slot_id": "string",
            "reviewer_id": "string",
            "answers_query": ["candidate_id"],
            "morphology_valid": "boolean",
            "naturalness_valid": "boolean",
            "decision": "accept|reject",
            "notes": "string",
        },
        "semantic_items": semantic_items,
        "morphology_items": morphology_items,
    }
    output = paths.root / "human_review_manifest.json"
    _write_json(output, manifest)
    return {"run_id": run_id, "pending": len(records), "manifest": str(output)}


def _load_decisions(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        return [
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("decisions", [])
    if not isinstance(value, list):
        raise ValueError("Human review girdisi JSON list/decisions veya JSONL olmalı")
    return value


def _validate_decision(row: dict[str, Any]) -> None:
    required = {
        "slot_id", "reviewer_id", "answers_query", "morphology_valid",
        "naturalness_valid", "decision", "notes",
    }
    if set(row) != required:
        raise ValueError(f"Human review alanları eksik/fazla: {sorted(set(row) ^ required)}")
    if not row["slot_id"] or not row["reviewer_id"]:
        raise ValueError("slot_id ve reviewer_id boş olamaz")
    if row["decision"] not in {"accept", "reject"}:
        raise ValueError("Human review decision accept veya reject olmalı")
    if not isinstance(row["answers_query"], list):
        raise ValueError("answers_query list olmalı")
    if not isinstance(row["morphology_valid"], bool) or not isinstance(row["naturalness_valid"], bool):
        raise ValueError("morphology_valid ve naturalness_valid boolean olmalı")


def apply_human_reviews(
    run_id: str, decision_path: str | Path, config_path: str | None = None
) -> dict[str, Any]:
    """Apply blind reviews only after the configured independent-reviewer consensus."""
    load_config(config_path, runtime=False)
    paths = paths_for(run_id)
    memory = DatasetMemory(paths.memory)
    pending = {row["slot_id"]: row for row in unresolved_reviews(run_id)}
    decisions_path = paths.root / "human_review_decisions.jsonl"
    existing = read_jsonl(decisions_path)
    seen = {(row.get("slot_id"), row.get("reviewer_id")) for row in existing}
    appended = 0
    for row in _load_decisions(decision_path):
        _validate_decision(row)
        key = (row["slot_id"], row["reviewer_id"])
        if row["slot_id"] not in pending:
            raise ValueError(f"Bekleyen review slotu yok: {row['slot_id']}")
        if key in seen:
            continue
        stored = {**row, "recorded_at": datetime.now(timezone.utc).isoformat()}
        _append_jsonl(decisions_path, stored)
        existing.append(stored)
        seen.add(key)
        appended += 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing:
        if row.get("slot_id") in pending:
            grouped[row["slot_id"]].append(row)

    accepted_ids = {row.get("slot_id") for row in read_jsonl(paths.accepted)}
    resolved = Counter()
    unresolved = []
    for slot_id, record in pending.items():
        reviews = grouped.get(slot_id, [])
        required = int(record["reviewers_required"])
        if len({row["reviewer_id"] for row in reviews}) < required:
            unresolved.append(slot_id)
            continue
        family = record["family"]
        gold = family["gold_id"]
        votes = []
        for review in reviews:
            valid_accept = (
                review["decision"] == "accept"
                and set(review["answers_query"]) == {gold}
                and review["morphology_valid"]
                and review["naturalness_valid"]
            )
            votes.append("accept" if valid_accept else "reject")
        counts = Counter(votes)
        decision, count = counts.most_common(1)[0]
        if count <= len(votes) / 2:
            unresolved.append(slot_id)
            continue

        if decision == "accept":
            accepted = deepcopy(family)
            accepted.setdefault("qc", {})["human_review"] = {
                "status": "pass",
                "review_kind": record["review_kind"],
                "reviewer_count": len(reviews),
                "reviewer_id_sha256": sorted({
                    hashlib.sha256(row["reviewer_id"].encode()).hexdigest()
                    for row in reviews
                }),
            }
            accepted["source_type"] = "llm_generated_cascade_and_human_verified"
            accepted["memory_tags"] = family_memory_tags(accepted)
            if slot_id not in accepted_ids:
                _append_jsonl(paths.accepted, accepted)
            memory.record_outcome(slot_id, "accepted", accepted, actor="human_review_consensus")
            resolved["accepted"] += 1
        else:
            rejected = {
                "slot_id": slot_id,
                "stage": "human_review",
                "problems": ["human review consensus rejected candidate family"],
                "last_family": family,
                "next_refill_round": int(
                    family.get("provenance", {}).get("refill_round", 0)
                ) + 1,
                "human_reviews": reviews,
            }
            _append_jsonl(paths.rejected, rejected)
            memory.record_outcome(slot_id, "rejected", rejected, actor="human_review_consensus")
            resolved["rejected"] += 1

    return {
        "run_id": run_id,
        "new_decisions": appended,
        "resolved": dict(resolved),
        "unresolved": len(unresolved),
        "unresolved_slot_ids": unresolved,
        "decisions_file": str(decisions_path),
    }
