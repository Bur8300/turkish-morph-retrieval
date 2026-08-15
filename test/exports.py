"""Freeze the automatically verified 100/500 split and export private/public/BEIR views."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import paths_for, read_jsonl
from .selection import select_balanced, selection_statistics
from .validators import artifact_report, corpus_problems, tr_lower


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent, text=True).strip()
    except Exception:
        return None


def _envelope(cfg: dict, split: str, items: list[dict]) -> dict[str, Any]:
    return {
        "dataset_name": cfg["dataset_name"],
        "version": cfg["version"],
        "split": split,
        "family_definition": "1 query + 1 positive + 8 hard negatives + 2 easy negatives",
        "statistics": selection_statistics(items),
        "items": items,
    }


def _blind_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "family_id": item["family_id"],
        "split": item["split"],
        "generalization_bucket": item["generalization_bucket"],
        "query_sentence_count": item["query_sentence_count"],
        "passage_sentence_count": item["passage_sentence_count"],
        "critical_sentence_position": item["critical_sentence_position"],
        "layer": item["layer"],
        "objective": item["objective"],
        "macro_phenomenon": item["macro_phenomenon"],
        "target_feature": item["target_feature"],
        "domain": item["domain"],
        "register": item["register"],
        "query": item["query"],
        "candidates": [{"id": c["id"], "text": c["text"]} for c in item["candidates"]],
    }


def _export_beir(base: Path, items: list[dict], include_qrels: bool) -> list[Path]:
    base.mkdir(parents=True, exist_ok=True)
    corpus = [
        {"_id": candidate["id"], "title": "", "text": candidate["text"]}
        for item in items for candidate in item["candidates"]
    ]
    queries = [{"_id": item["family_id"], "text": item["query"]} for item in items]
    pool = {item["family_id"]: [candidate["id"] for candidate in item["candidates"]] for item in items}
    files = [base / "corpus.jsonl", base / "queries.jsonl", base / "candidate_pool.json"]
    _write_jsonl(files[0], corpus)
    _write_jsonl(files[1], queries)
    _write_json(files[2], pool)
    if include_qrels:
        qrels = base / "qrels" / "test.tsv"
        qrels.parent.mkdir(parents=True, exist_ok=True)
        qrels.write_text(
            "query-id\tcorpus-id\tscore\n" + "".join(
                f"{item['family_id']}\t{candidate['id']}\t{candidate['relevance']}\n"
                for item in items for candidate in item["candidates"]
            ),
            encoding="utf-8",
        )
        files.append(qrels)
    return files


def _leakage_checks(dev: list[dict], test: list[dict]) -> list[str]:
    problems = []
    dev_lemmas = {tr_lower(item["critical_lemma"]).strip() for item in dev}
    test_holdout_lemmas = {
        tr_lower(item["critical_lemma"]).strip()
        for item in test if item["generalization_bucket"] == "lemma_holdout"
    }
    if dev_lemmas & test_holdout_lemmas:
        problems.append(f"lemma_holdout dev sızıntısı: {sorted(dev_lemmas & test_holdout_lemmas)}")
    dev_templates = {item["template_id"] for item in dev}
    test_templates = {item["template_id"] for item in test if item["generalization_bucket"] == "template_holdout"}
    if dev_templates & test_templates:
        problems.append(f"template_holdout dev sızıntısı: {sorted(dev_templates & test_templates)}")
    dev_chains = {item["target_feature"] for item in dev if item["layer"] == "chain"}
    test_chains = {item["target_feature"] for item in test if item["generalization_bucket"] == "composition_holdout"}
    if dev_chains & test_chains:
        problems.append(f"composition_holdout dev sızıntısı: {sorted(dev_chains & test_chains)}")
    dev_domain_register = {(item["domain"], item["register"]) for item in dev}
    test_shift_pairs = {
        (item["domain"], item["register"])
        for item in test if "domain_shift" in item.get("generalization_tags", [])
    }
    if dev_domain_register & test_shift_pairs:
        problems.append(f"domain_shift dev sızıntısı: {sorted(dev_domain_register & test_shift_pairs)}")
    return problems


def _critical_position_audit(items: list[dict]) -> tuple[dict[str, dict[str, int]], list[str]]:
    grouped = defaultdict(Counter)
    for item in items:
        passage_count = int(item["passage_sentence_count"])
        grouped[(item.get("split", item.get("target_split")), passage_count)][
            int(item["critical_sentence_position"])
        ] += 1
    report = {}
    problems = []
    for (split, passage_count), counts in sorted(grouped.items()):
        key = f"{split}|{passage_count}_sentences"
        complete = {position: counts[position] for position in range(1, passage_count + 1)}
        report[key] = {str(position): count for position, count in complete.items()}
        tolerance = max(2, math.ceil(sum(complete.values()) * 0.05))
        if max(complete.values()) - min(complete.values()) > tolerance:
            problems.append(f"kritik-cümle konumu dengesiz: {key} {complete}")
    return report, problems


def finalize(run_id: str, config_path: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=False)
    run = paths_for(run_id)
    accepted = read_jsonl(run.accepted)
    selected = select_balanced(
        accepted,
        cfg,
        {"development": cfg["targets"]["development"], "sealed_test": cfg["targets"]["sealed_test"]},
    )
    for item in selected:
        item["split"] = item.pop("target_split")
        item["source_type"] = "llm_generated_auto_verified"
    dev = [item for item in selected if item["split"] == "development"]
    sealed = [item for item in selected if item["split"] == "sealed_test"]
    position_audit, position_problems = _critical_position_audit(selected)
    problems = corpus_problems(selected, cfg) + _leakage_checks(dev, sealed) + position_problems
    artifacts = artifact_report(selected)
    if artifacts["longest_gold_rate"] >= 0.15:
        problems.append(f"longest-gold oranı freeze sınırını aşıyor: {artifacts['longest_gold_rate']:.3f}")
    max_position_rate = max(artifacts["gold_position_counts"].values(), default=0) / max(1, len(selected))
    if max_position_rate >= 0.15:
        problems.append(f"candidate-position dağılımı dengesiz: {max_position_rate:.3f}")
    if problems:
        _write_json(run.root / "freeze_blockers.json", {"problems": problems, "artifacts": artifacts})
        raise ValueError(f"Test dondurulamadı; {len(problems)} blocker var. freeze_blockers.json'a bak")

    release = run.root / "release"
    private = run.root / "private"
    dev_path = release / f"morph_dev_v{cfg['version']}.json"
    blind_path = release / f"morph_test_blind_v{cfg['version']}.json"
    internal_path = private / f"morph_test_internal_v{cfg['version']}.json"
    qrels_path = private / "private_qrels.jsonl"
    _write_json(dev_path, _envelope(cfg, "development", dev))
    _write_json(blind_path, _envelope(cfg, "sealed_test_blind", [_blind_item(item) for item in sealed]))
    _write_json(internal_path, _envelope(cfg, "sealed_test", sealed))
    _write_jsonl(qrels_path, [
        {
            "query_id": item["family_id"], "corpus_id": candidate["id"],
            "relevance": candidate["relevance"],
        }
        for item in sealed for candidate in item["candidates"]
    ])

    beir_files = []
    beir_files += _export_beir(release / "beir" / "dev", dev, include_qrels=True)
    beir_files += _export_beir(release / "beir" / "test", sealed, include_qrels=False)
    private_beir_qrels = private / "beir_test_qrels.tsv"
    private_beir_qrels.parent.mkdir(parents=True, exist_ok=True)
    private_beir_qrels.write_text(
        "query-id\tcorpus-id\tscore\n" + "".join(
            f"{item['family_id']}\t{candidate['id']}\t{candidate['relevance']}\n"
            for item in sealed for candidate in item["candidates"]
        ), encoding="utf-8"
    )

    holdout_manifest = {
        "critical_lemmas": sorted({tr_lower(item["critical_lemma"]).strip() for item in sealed if item["generalization_bucket"] == "lemma_holdout"}),
        "template_ids": sorted({item["template_id"] for item in sealed if item["generalization_bucket"] == "template_holdout"}),
        "composition_chains": sorted({item["target_feature"] for item in sealed if item["generalization_bucket"] == "composition_holdout"}),
        "domain_register_pairs": sorted({
            f"{item['domain']}|{item['register']}"
            for item in sealed if "domain_shift" in item.get("generalization_tags", [])
        }),
        "query_sha256": {item["family_id"]: hashlib.sha256(item["query"].encode()).hexdigest() for item in sealed},
        "candidate_text_sha256": sorted({hashlib.sha256(candidate["text"].encode()).hexdigest() for item in sealed for candidate in item["candidates"]}),
    }
    holdout_path = private / "train_exclusion_holdouts.json"
    _write_json(holdout_path, holdout_manifest)
    artifact_path = release / "artifact_audit.json"
    _write_json(artifact_path, artifacts)

    hashed_files = [dev_path, blind_path, internal_path, qrels_path, private_beir_qrels, holdout_path, artifact_path, *beir_files]
    freeze_manifest = {
        "dataset_name": cfg["dataset_name"],
        "version": cfg["version"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "counts": {"development": len(dev), "sealed_test": len(sealed), "candidates_per_family": 11},
        "statistics": selection_statistics(selected),
        "critical_sentence_position_audit": position_audit,
        "artifact_audit": artifacts,
        "files": {str(path.relative_to(run.root)): _sha256(path) for path in hashed_files},
        "qrels_definition": (
            "Her query için kendi family'sindeki gold=1 ve 10 generated negative=0. "
            "Başka family belgeleri etiketlenmez; full-corpus sonuç yalnız tanısaldır."
        ),
    }
    freeze_path = release / "freeze_manifest.json"
    _write_json(freeze_path, freeze_manifest)
    return freeze_manifest
