"""Five-reviewer blinded assignment, agreement tracking and adjudication preparation."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import paths_for, read_jsonl
from .selection import select_balanced, selection_statistics


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def review_file(path: str | Path) -> dict[str, Any]:
    """Dependency-free interactive reviewer UI; checkpoints after every completed family."""
    assignment = Path(path)
    rows = read_jsonl(assignment)
    completed_before = sum(row.get("review", {}).get("recommendation") in {"approve", "reject"} for row in rows)

    def ask_ids(prompt: str, allowed: set[str]) -> list[str]:
        while True:
            raw = input(prompt).strip()
            values = [] if not raw else [value.strip() for value in raw.split(",") if value.strip()]
            unknown = set(values) - allowed
            if not unknown:
                return values
            print("Bilinmeyen candidate id:", ", ".join(sorted(unknown)))

    for index, row in enumerate(rows):
        if row.get("review", {}).get("recommendation") in {"approve", "reject"}:
            continue
        print("\n" + "=" * 88)
        print(f"{index + 1}/{len(rows)}  {row['family_id']}")
        print("Hedef:", row["target_feature_label"], "|", row["objective"], "|", row["layer"])
        print("QUERY:", row["query"])
        allowed = set()
        for candidate in row["candidates"]:
            allowed.add(candidate["id"])
            print(f"\n[{candidate['id']}] {candidate['text']}")
        print("\nKomut: q = güvenli çıkış, s = bu family'yi şimdilik atla")
        command = input("Devam [Enter/q/s]: ").strip().lower()
        if command == "q":
            break
        if command == "s":
            continue
        selected = ask_ids("Tam doğru/relevant id(ler), virgülle: ", allowed)
        unnatural = ask_ids("Doğal olmayan id(ler), yoksa Enter: ", allowed)
        morphology = ask_ids("Biçimbilim sorunu olan id(ler), yoksa Enter: ", allowed)
        while True:
            try:
                naturalness = int(input("Family doğallığı [1-5]: ").strip())
                if 1 <= naturalness <= 5:
                    break
            except ValueError:
                pass
            print("1 ile 5 arasında tam sayı girin.")
        while True:
            artifact_raw = input("Uzunluk/üslup artefaktı var mı? [y/n]: ").strip().lower()
            if artifact_raw in {"y", "yes", "e", "evet", "n", "no", "h", "hayır", "hayir"}:
                artifact = artifact_raw in {"y", "yes", "e", "evet"}
                break
            print("y veya n girin.")
        recommendation = input("Öneri [approve/reject]: ").strip().lower()
        while recommendation not in {"approve", "reject"}:
            recommendation = input("Yalnız approve veya reject: ").strip().lower()
        notes = input("Not (opsiyonel): ").strip()
        row["review"] = {
            "recommendation": recommendation,
            "selected_relevant_ids": selected,
            "unnatural_candidate_ids": unnatural,
            "morphology_problem_ids": morphology,
            "family_naturalness": naturalness,
            "length_or_style_artifact": artifact,
            "notes": notes,
        }
        _write_jsonl(assignment, rows)
        print("Kaydedildi.")
    completed_after = sum(row.get("review", {}).get("recommendation") in {"approve", "reject"} for row in rows)
    return {
        "assignment": str(assignment),
        "completed_before": completed_before,
        "completed_after": completed_after,
        "total": len(rows),
    }


def _visible_family(item: dict[str, Any], reviewer_id: str) -> dict[str, Any]:
    candidates = [{"id": c["id"], "text": c["text"]} for c in item["candidates"]]
    seed = int(hashlib.sha256(f"{item['family_id']}|{reviewer_id}".encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(candidates)
    return {
        "family_id": item["family_id"],
        "query": item["query"],
        "target_feature": item["target_feature"],
        "target_feature_label": item["target_feature_label"],
        "objective": item["objective"],
        "layer": item["layer"],
        "candidates": candidates,
        "review": {
            "recommendation": None,
            "selected_relevant_ids": [],
            "unnatural_candidate_ids": [],
            "morphology_problem_ids": [],
            "family_naturalness": None,
            "length_or_style_artifact": None,
            "notes": "",
        },
    }


def prepare_review(run_id: str, config_path: str | None = None, force: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=False)
    run = paths_for(run_id)
    accepted = read_jsonl(run.accepted)
    factor = float(cfg["targets"]["review_pool_factor"])
    split_targets = {
        "development": int(round(cfg["targets"]["development"] * factor)),
        "sealed_test": int(round(cfg["targets"]["sealed_test"] * factor)),
    }
    selected = select_balanced(accepted, cfg, split_targets)
    review_dir = run.root / "review"
    private_path = review_dir / "review_pool_private.jsonl"
    if private_path.exists() and not force:
        raise FileExistsError(f"{private_path} zaten var; reviewer çalışmalarını korumak için --force gerekli")
    _write_jsonl(private_path, selected)

    reviewers = list(cfg["review"]["reviewer_ids"])
    calibration_count = int(round(len(selected) * float(cfg["review"]["calibration_fraction"])))
    ordered = sorted(selected, key=lambda item: hashlib.sha256(item["family_id"].encode()).hexdigest())
    calibration_ids = {item["family_id"] for item in ordered[:calibration_count]}
    pairs = list(itertools.combinations(reviewers, int(cfg["review"]["reviews_per_family"])))
    assignments = defaultdict(list)
    pair_index = 0
    for item in ordered:
        assigned = reviewers if item["family_id"] in calibration_ids else list(pairs[pair_index % len(pairs)])
        if item["family_id"] not in calibration_ids:
            pair_index += 1
        for reviewer_id in assigned:
            assignments[reviewer_id].append(_visible_family(item, reviewer_id))

    assignment_dir = review_dir / "assignments"
    for reviewer_id in reviewers:
        _write_jsonl(assignment_dir / f"{reviewer_id}.jsonl", assignments[reviewer_id])
    manifest = {
        "run_id": run_id,
        "review_pool_size": len(selected),
        "split_targets": split_targets,
        "reviewers": reviewers,
        "calibration_count": calibration_count,
        "calibration_ids": sorted(calibration_ids),
        "assignment_counts": {key: len(value) for key, value in assignments.items()},
        "selection_statistics": selection_statistics(selected),
        "instructions": {
            "recommendation": "approve veya reject",
            "selected_relevant_ids": "Sorguyu bütünüyle doğru yanıtlayan bütün aday kimlikleri",
            "family_naturalness": "1–5; kabul için en az 4",
            "length_or_style_artifact": "true/false",
        },
    }
    _write_json(review_dir / "review_manifest.json", manifest)
    return manifest


def _assessment_pass(row: dict, gold_id: str, naturalness_min: int) -> tuple[bool | None, str]:
    review = row.get("review", {})
    recommendation = review.get("recommendation")
    if recommendation not in {"approve", "reject"}:
        return None, "pending"
    selected = set(review.get("selected_relevant_ids") or [])
    automatic_pass = (
        recommendation == "approve"
        and selected == {gold_id}
        and not (review.get("unnatural_candidate_ids") or [])
        and not (review.get("morphology_problem_ids") or [])
        and isinstance(review.get("family_naturalness"), int)
        and review["family_naturalness"] >= naturalness_min
        and review.get("length_or_style_artifact") is False
    )
    return automatic_pass, "approve" if automatic_pass else "reject"


def merge_reviews(run_id: str, config_path: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=False)
    run = paths_for(run_id)
    review_dir = run.root / "review"
    private = {item["family_id"]: item for item in read_jsonl(review_dir / "review_pool_private.jsonl")}
    assignment_dir = review_dir / "assignments"
    assessments = defaultdict(list)
    for path in sorted(assignment_dir.glob("*.jsonl")):
        reviewer_id = path.stem
        for row in read_jsonl(path):
            assessments[row["family_id"]].append((reviewer_id, row))

    existing_adjudication_path = review_dir / "adjudication.jsonl"
    existing_adjudication = {
        row["family_id"]: row for row in read_jsonl(existing_adjudication_path)
    }
    review_manifest = json.loads((review_dir / "review_manifest.json").read_text(encoding="utf-8"))
    calibration_ids = set(review_manifest["calibration_ids"])
    statuses = []
    adjudication_rows = []
    naturalness_min = int(cfg["quality"]["judge_naturalness_min"])
    for family_id, item in private.items():
        decisions = []
        summaries = []
        adjudicated = False
        for reviewer_id, row in assessments.get(family_id, []):
            passed, label = _assessment_pass(row, item["gold_id"], naturalness_min)
            decisions.append(passed)
            summaries.append({"reviewer_id": reviewer_id, "status": label, "review": row.get("review", {})})
        completed = [decision for decision in decisions if decision is not None]
        expected_count = len(cfg["review"]["reviewer_ids"]) if family_id in calibration_ids else int(cfg["review"]["reviews_per_family"])
        if len(decisions) != expected_count or len(completed) != expected_count:
            status = "pending"
        elif all(completed):
            status = "approved"
        elif not any(completed):
            status = "rejected"
        else:
            status = "needs_adjudication"

        if status == "needs_adjudication":
            adjudication = existing_adjudication.get(family_id)
            if adjudication:
                passed, _ = _assessment_pass(adjudication, item["gold_id"], naturalness_min)
                if passed is not None:
                    adjudicated = True
                    summaries.append({
                        "reviewer_id": "adjudicator",
                        "status": "approve" if passed else "reject",
                        "review": adjudication.get("review", {}),
                    })
                if passed is True:
                    status = "approved"
                elif passed is False:
                    status = "rejected"
                else:
                    adjudication_rows.append(adjudication)
            else:
                visible = _visible_family(item, "adjudicator")
                visible["reviewer_summaries"] = summaries
                adjudication_rows.append(visible)
        statuses.append({
            "family_id": family_id,
            "status": status,
            "n_assigned": len(decisions),
            "n_completed": len(completed) + int(adjudicated),
            "adjudicated": adjudicated,
            "reviewer_summaries": summaries,
        })

    _write_jsonl(review_dir / "review_status.jsonl", statuses)
    _write_jsonl(existing_adjudication_path, adjudication_rows)
    report = {
        "run_id": run_id,
        "status_counts": dict(Counter(row["status"] for row in statuses)),
        "adjudication_remaining": len(adjudication_rows),
    }
    _write_json(review_dir / "review_report.json", report)
    return report
