"""Calibration and escalation diagnostics for the cascade judge."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .dataset_memory import DatasetMemory
from .pipeline import paths_for, read_jsonl


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def judge_calibration_report(run_id: str) -> dict[str, Any]:
    paths = paths_for(run_id)
    memory = DatasetMemory(paths.memory)
    families: dict[str, dict[str, Any]] = {}
    review_records = read_jsonl(paths.needs_review)
    for record in review_records:
        if record.get("slot_id") and record.get("family"):
            families[record["slot_id"]] = record["family"]
    for family in read_jsonl(paths.accepted):
        if family.get("slot_id"):
            families[family["slot_id"]] = family

    semantic_pass_total = 0
    semantic_unique_gold = 0
    semantic_confidences = []
    order_stable = 0
    morphology_unique_gold = 0
    morphology_confidences = []
    candidate_naturalness_pass = 0
    for family in families.values():
        judging = family.get("qc", {}).get("judging", {})
        semantic = judging.get("semantic", {})
        passes = semantic.get("passes", [])
        semantic_pass_total += len(passes)
        semantic_unique_gold += sum(
            set(row.get("answers_query", [])) == {family.get("gold_id")} for row in passes
        )
        semantic_confidences.extend(
            row["confidence"] for row in passes if isinstance(row.get("confidence"), int)
        )
        order_stable += int(bool(semantic.get("order_stable")))
        candidate_naturalness_pass += int(
            int(semantic.get("candidate_naturalness_min", 0)) >= 4
        )
        morphology = judging.get("morphology", {})
        morphology_unique_gold += int(
            set(morphology.get("answers_query", [])) == {family.get("gold_id")}
        )
        if isinstance(morphology.get("confidence"), int):
            morphology_confidences.append(morphology["confidence"])

    decisions_path = paths.root / "human_review_decisions.jsonl"
    decisions = read_jsonl(decisions_path)
    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if row.get("slot_id"):
            by_slot[row["slot_id"]].append(row)
    human_consensus = 0
    human_disagreement = 0
    semantic_human_answer_agreement = []
    for slot_id, rows in by_slot.items():
        votes = Counter(row.get("decision") for row in rows)
        if votes and votes.most_common(1)[0][1] > len(rows) / 2:
            human_consensus += 1
        else:
            human_disagreement += 1
        answer_votes = Counter(tuple(sorted(row.get("answers_query", []))) for row in rows)
        if slot_id in families and answer_votes:
            human_answers = set(answer_votes.most_common(1)[0][0])
            semantic_passes = (
                families[slot_id].get("qc", {}).get("judging", {})
                .get("semantic", {}).get("passes", [])
            )
            semantic_human_answer_agreement.extend(
                set(row.get("answers_query", [])) == human_answers for row in semantic_passes
            )

    pending_slots = {
        row.get("slot_id") for row in review_records
        if row.get("slot_id") and memory.slot_status(row["slot_id"]) == "needs_review"
    }
    denominator = max(1, len(families))
    report = {
        "run_id": run_id,
        "judged_families": len(families),
        "semantic_passes": semantic_pass_total,
        "semantic_unique_gold_rate": semantic_unique_gold / max(1, semantic_pass_total),
        "semantic_order_stability_rate": order_stable / denominator,
        "semantic_confidence_mean": mean(semantic_confidences) if semantic_confidences else None,
        "candidate_naturalness_gate_rate": candidate_naturalness_pass / denominator,
        "morphology_unique_gold_rate": morphology_unique_gold / denominator,
        "morphology_confidence_mean": mean(morphology_confidences) if morphology_confidences else None,
        "review_escalation_rate": len({row.get("slot_id") for row in review_records}) / denominator,
        "pending_human_review": len(pending_slots),
        "human_reviewed_slots": len(by_slot),
        "human_consensus_slots": human_consensus,
        "human_disagreement_slots": human_disagreement,
        "semantic_human_answer_agreement_rate": (
            mean(semantic_human_answer_agreement) if semantic_human_answer_agreement else None
        ),
        "note": (
            "Threshold/model seçimi yalnız development ve blind human-review calibration sonuçlarıyla "
            "yapılmalı; sealed_test üzerinde eşik ayarlanmaz."
        ),
    }
    output = paths.root / "judge_calibration_report.json"
    _write_json(output, report)
    report["report_path"] = str(output)
    return report
