"""Generation pipeline: plan -> generate -> deterministic QC -> blind independent judge."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .planner import build_plan, plan_hash, plan_statistics
from .prompts import (
    GENERATOR_SYSTEM,
    JUDGE_SYSTEM,
    PROMPT_VERSION,
    build_generation_prompt,
    build_judge_prompt,
    build_repair_prompt,
)
from .providers import make_provider
from .schema import GENERATION_SCHEMA, JUDGE_SCHEMA
from .validators import interpret_judge, normalize_family, quality_score, validate_family


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunPaths:
    root: Path
    plan: Path
    manifest: Path
    accepted: Path
    rejected: Path
    failures: Path
    report: Path
    cache: Path


def paths_for(run_id: str) -> RunPaths:
    root = HERE / "runs" / run_id
    return RunPaths(
        root=root,
        plan=root / "plan.json",
        manifest=root / "run_manifest.json",
        accepted=root / "accepted.jsonl",
        rejected=root / "rejected.jsonl",
        failures=root / "failures.jsonl",
        report=root / "generation_report.json",
        cache=root / "cache",
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _pipeline_source_hashes() -> dict[str, str]:
    names = ("config.py", "planner.py", "prompts.py", "schema.py", "taxonomy.py", "validators.py")
    return {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in names
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, value: Any, lock: threading.Lock) -> None:
    line = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} bozuk JSONL: {exc}") from exc
    return rows


def initialise_run(run_id: str, cfg: dict[str, Any], slots: list[dict[str, Any]]) -> RunPaths:
    paths = paths_for(run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.cache.mkdir(parents=True, exist_ok=True)
    current_hash = plan_hash(slots)
    if paths.plan.exists():
        previous = json.loads(paths.plan.read_text(encoding="utf-8"))
        if plan_hash(previous) != current_hash:
            raise ValueError(f"{run_id} mevcut planı yeni config ile uyuşmuyor; yeni run-id kullan")
    else:
        _write_json(paths.plan, slots)

    config_sha256 = hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest()
    source_hashes = _pipeline_source_hashes()
    manifest = {
        "run_id": run_id,
        "dataset_name": cfg["dataset_name"],
        "dataset_version": cfg["version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_start": _git_commit(),
        "prompt_version": PROMPT_VERSION,
        "pipeline_source_sha256": source_hashes,
        "config_path": cfg["_config_path"],
        "config_sha256": config_sha256,
        "plan_sha256": current_hash,
        "plan_size": len(slots),
        "plan_statistics": plan_statistics(slots),
        "generator": {
            "provider": cfg["generation"]["generator"]["provider"],
            "model": cfg["generation"]["generator"]["model"],
        },
        "judge": {
            "provider": cfg["generation"]["judge"]["provider"],
            "model": cfg["generation"]["judge"]["model"],
        },
    }
    if paths.manifest.exists():
        previous = json.loads(paths.manifest.read_text(encoding="utf-8"))
        invariant_keys = (
            "dataset_version", "prompt_version", "pipeline_source_sha256", "config_sha256",
            "plan_sha256", "generator", "judge"
        )
        changed = [key for key in invariant_keys if previous.get(key) != manifest.get(key)]
        if changed:
            raise ValueError(
                f"{run_id} model/config/prompt karışımını engellemek için devam ettirilmedi; "
                f"değişen alanlar: {changed}. Yeni run-id kullan."
            )
    else:
        _write_json(paths.manifest, manifest)
    return paths


def _request_provenance(response) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "request_hash": response.request_hash,
        "cache_hit": response.cache_hit,
        "usage": response.usage,
    }


def _process_slot(slot, cfg, generator, judge) -> tuple[str, dict[str, Any]]:
    generation_provenance = []
    previous = None
    validation_problems: list[str] = []
    max_attempts = int(cfg["generation"]["max_generation_attempts"])
    family = None
    for attempt in range(max_attempts):
        prompt = (
            build_generation_prompt(slot)
            if attempt == 0
            else build_repair_prompt(slot, previous or {}, validation_problems)
        )
        response = generator.call_json(GENERATOR_SYSTEM, prompt, GENERATION_SCHEMA, "generate_test_family")
        generation_provenance.append(_request_provenance(response))
        previous = response.data
        try:
            family = normalize_family(response.data, slot)
            validation_problems = validate_family(family, slot, cfg)
        except Exception as exc:
            validation_problems = [f"normalizasyon hatası: {type(exc).__name__}: {exc}"]
            family = None
        if not validation_problems:
            break
    if family is None or validation_problems:
        return "rejected", {
            "slot_id": slot["slot_id"],
            "stage": "deterministic_validation",
            "problems": validation_problems,
            "last_raw": previous,
            "provenance": {"generator_attempts": generation_provenance},
        }

    judge_response = judge.call_json(JUDGE_SYSTEM, build_judge_prompt(family), JUDGE_SCHEMA, "blind_judge")
    judge_problems, judge_metadata = interpret_judge(family, judge_response.data, cfg)
    family["provenance"] = {
        "generator_attempts": generation_provenance,
        "judge": _request_provenance(judge_response),
        "prompt_version": PROMPT_VERSION,
    }
    family["qc"] = {
        "deterministic": "pass",
        "judge": judge_metadata,
        "quality_score": None,
    }
    family["qc"]["quality_score"] = round(quality_score(family), 5)
    if judge_problems:
        return "rejected", {
            "slot_id": slot["slot_id"],
            "stage": "blind_judge",
            "problems": judge_problems,
            "family": family,
            "judge_raw": judge_response.data,
        }
    return "accepted", family


def generate(run_id: str, config_path: str | None = None, limit: int | None = None, workers: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=True)
    all_slots = build_plan(cfg)
    slots = all_slots[:limit] if limit is not None else all_slots
    paths = initialise_run(run_id, cfg, all_slots)
    processed = {
        row.get("slot_id") for row in read_jsonl(paths.accepted) + read_jsonl(paths.rejected) if row.get("slot_id")
    }
    pending = [slot for slot in slots if slot["slot_id"] not in processed]
    metadata = {"dataset_version": cfg["version"], "prompt_version": PROMPT_VERSION}
    generator = make_provider(cfg["generation"]["generator"], paths.cache / "generator", metadata)
    judge = make_provider(cfg["generation"]["judge"], paths.cache / "judge", metadata)
    write_lock = threading.Lock()
    counts = Counter()
    errors = []
    started = time.time()

    worker_count = int(workers or cfg["generation"].get("workers", 4))
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        future_to_slot = {
            executor.submit(_process_slot, slot, cfg, generator, judge): slot for slot in pending
        }
        for future in as_completed(future_to_slot):
            slot = future_to_slot[future]
            try:
                status, record = future.result()
            except Exception as exc:  # transient/provider failures are retryable on the next resume
                status = "failed"
                record = {
                    "slot_id": slot["slot_id"],
                    "stage": "exception",
                    "problems": [f"{type(exc).__name__}: {exc}"],
                }
                errors.append(record)
            target_path = {
                "accepted": paths.accepted,
                "rejected": paths.rejected,
                "failed": paths.failures,
            }[status]
            _append_jsonl(target_path, record, write_lock)
            counts[status] += 1

    accepted = read_jsonl(paths.accepted)
    rejected = read_jsonl(paths.rejected)
    failures = read_jsonl(paths.failures)
    report = {
        "run_id": run_id,
        "elapsed_seconds_this_call": round(time.time() - started, 2),
        "pending_processed_this_call": len(pending),
        "outcomes_this_call": dict(counts),
        "accepted_total": len(accepted),
        "rejected_total": len(rejected),
        "retryable_failures_total": len(failures),
        "accepted_by_bucket": dict(Counter(item["generalization_bucket"] for item in accepted)),
        "accepted_by_query_sentence_count": dict(
            Counter(str(item["query_sentence_count"]) for item in accepted)
        ),
        "accepted_by_passage_sentence_count": dict(
            Counter(str(item["passage_sentence_count"]) for item in accepted)
        ),
        "rejection_stages": dict(Counter(item.get("stage", "unknown") for item in rejected)),
        "uncaught_errors_this_call": errors,
    }
    _write_json(paths.report, report)
    return report


def write_plan(run_id: str, config_path: str | None = None, size: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=False)
    slots = build_plan(cfg, size=size)
    paths = paths_for(run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    _write_json(paths.plan, slots)
    report = {"run_id": run_id, "size": len(slots), "sha256": plan_hash(slots), "statistics": plan_statistics(slots)}
    _write_json(paths.root / "plan_report.json", report)
    return report


def default_run_id() -> str:
    return datetime.now().strftime("test_%Y%m%d_%H%M%S")
