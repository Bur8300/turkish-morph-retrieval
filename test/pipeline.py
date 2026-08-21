"""Generation pipeline: plan -> generate -> QC -> cascade judges -> human-review gate."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .dataset_memory import DatasetMemory, family_memory_tags
from .planner import build_plan, plan_hash, plan_statistics
from .prompts import (
    ADJUDICATOR_SYSTEM,
    GENERATOR_SYSTEM,
    MORPHOLOGY_JUDGE_SYSTEM,
    PROMPT_VERSION,
    SEMANTIC_JUDGE_SYSTEM,
    build_adjudicator_prompt,
    build_generation_prompt,
    build_morphology_judge_prompt,
    build_repair_prompt,
    build_semantic_judge_prompt,
)
from .providers import make_provider
from .schema import ADJUDICATOR_SCHEMA, GENERATION_SCHEMA, MORPHOLOGY_JUDGE_SCHEMA, SEMANTIC_JUDGE_SCHEMA
from .validators import (
    interpret_morphology_judge,
    interpret_semantic_judges,
    normalize_family,
    quality_score,
    validate_family,
)


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunPaths:
    root: Path
    plan: Path
    manifest: Path
    accepted: Path
    needs_review: Path
    rejected: Path
    failures: Path
    report: Path
    cache: Path
    memory: Path


def paths_for(run_id: str) -> RunPaths:
    root = HERE / "runs" / run_id
    return RunPaths(
        root=root,
        plan=root / "plan.json",
        manifest=root / "run_manifest.json",
        accepted=root / "accepted.jsonl",
        needs_review=root / "needs_review.jsonl",
        rejected=root / "rejected.jsonl",
        failures=root / "failures.jsonl",
        report=root / "generation_report.json",
        cache=root / "cache",
        memory=root / "dataset_memory.sqlite3",
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _pipeline_source_hashes() -> dict[str, str]:
    names = (
        "config.py", "pipeline.py", "planner.py", "prompts.py", "schema.py", "taxonomy.py",
        "validators.py", "dataset_memory.py", "review.py", "judge_report.py",
    )
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
        "generators": [
            {"id": spec["id"], "provider": spec["provider"], "model": spec["model"]}
            for spec in cfg["generation"]["generators"]
        ],
        "judges": {
            name: {
                "enabled": spec.get("enabled", True),
                "provider": spec["provider"],
                "model": spec.get("model", ""),
                "provider_preferences": spec.get("provider_preferences", {}),
            }
            for name, spec in cfg["generation"]["judges"].items()
        },
        "human_review": cfg["generation"]["human_review"],
    }
    if paths.manifest.exists():
        previous = json.loads(paths.manifest.read_text(encoding="utf-8"))
        invariant_keys = (
            "dataset_version", "prompt_version", "pipeline_source_sha256", "config_sha256",
            "plan_sha256", "generators", "judges", "human_review"
        )
        changed = [key for key in invariant_keys if previous.get(key) != manifest.get(key)]
        if changed:
            raise ValueError(
                f"{run_id} model/config/prompt karışımını engellemek için devam ettirilmedi; "
                f"değişen alanlar: {changed}. Yeni run-id kullan."
            )
    else:
        _write_json(paths.manifest, manifest)
    DatasetMemory(paths.memory).sync_plan(slots)
    return paths


def _request_provenance(response) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "request_hash": response.request_hash,
        "cache_hit": response.cache_hit,
        "usage": response.usage,
        "actual_model": response.actual_model,
        "route_provider": response.route_provider,
    }


def human_audit_slots(slots: list[dict[str, Any]], cfg: dict[str, Any]) -> set[str]:
    """Select a deterministic stratified audit sample by split and holdout bucket."""
    rate = float(cfg["generation"]["human_review"]["audit_rate"])
    base_seed = int(cfg["generation"]["human_review"].get("seed", cfg.get("seed", 0)))
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for slot in slots:
        groups[(slot["target_split"], slot["generalization_bucket"])].append(slot["slot_id"])
    selected: set[str] = set()
    for stratum, slot_ids in sorted(groups.items()):
        count = round(len(slot_ids) * rate)
        material = f"{base_seed}|{stratum[0]}|{stratum[1]}"
        seed = int(hashlib.sha256(material.encode()).hexdigest()[:16], 16)
        selected.update(random.Random(seed).sample(sorted(slot_ids), count))
    return selected


def _process_slot(
    slot, cfg, generators, judges, start_refill_round: int = 0, stage_callback=None,
    memory_validator=None,
) -> tuple[str, dict[str, Any]]:
    generator = generators[slot["generator_id"]]
    max_attempts = int(cfg["generation"]["max_generation_attempts"])
    refill_count = int(cfg["generation"]["refill_rounds_per_call"])
    refill_history: list[dict[str, Any]] = []
    last_raw = None
    last_family = None
    last_stage = "deterministic_validation"
    last_problems: list[str] = []

    for refill_round in range(start_refill_round, start_refill_round + refill_count):
        # The nonce makes a replacement a fresh cached request while every balancing attribute
        # (feature, split, generator, lengths and holdout bucket) remains fixed to the slot.
        prompt_slot = {**slot, "refill_round": refill_round}
        generation_provenance = []
        previous = None
        validation_problems: list[str] = []
        family = None
        for attempt in range(max_attempts):
            prompt = (
                build_generation_prompt(prompt_slot)
                if attempt == 0
                else build_repair_prompt(prompt_slot, previous or {}, validation_problems)
            )
            response = generator.call_json(
                GENERATOR_SYSTEM, prompt, GENERATION_SCHEMA, "generate_test_family"
            )
            if stage_callback:
                stage_callback("generated", {"refill_round": refill_round, "attempt": attempt})
            generation_provenance.append(_request_provenance(response))
            previous = response.data
            try:
                family = normalize_family(response.data, slot)
                validation_problems = validate_family(family, slot, cfg)
            except Exception as exc:
                validation_problems = [f"normalizasyon hatası: {type(exc).__name__}: {exc}"]
                family = None
            if not validation_problems:
                if stage_callback:
                    stage_callback("deterministic_validated", {"refill_round": refill_round})
                break

        last_raw = previous
        last_family = family
        if family is None or validation_problems:
            last_stage = "deterministic_validation"
            last_problems = validation_problems
            refill_history.append({
                "refill_round": refill_round,
                "stage": last_stage,
                "problems": validation_problems,
                "generator_attempts": generation_provenance,
            })
            continue

        memory_problems = memory_validator(family) if memory_validator else []
        if memory_problems:
            last_stage = "dataset_memory"
            last_problems = memory_problems
            refill_history.append({
                "refill_round": refill_round,
                "stage": last_stage,
                "problems": memory_problems,
                "generator_attempts": generation_provenance,
            })
            if stage_callback:
                stage_callback("dataset_memory_rejected", {"refill_round": refill_round})
            continue

        semantic_responses = []
        semantic_verdicts = []
        for permutation in cfg["generation"]["judges"]["semantic"]["permutations"]:
            response = judges["semantic"].call_json(
                SEMANTIC_JUDGE_SYSTEM,
                build_semantic_judge_prompt(family, permutation),
                SEMANTIC_JUDGE_SCHEMA,
                f"semantic_judge_{permutation}",
            )
            semantic_responses.append(_request_provenance(response))
            semantic_verdicts.append(response.data)
        semantic_problems, semantic_metadata = interpret_semantic_judges(
            family, semantic_verdicts, cfg
        )

        morphology_response = judges["morphology"].call_json(
            MORPHOLOGY_JUDGE_SYSTEM,
            build_morphology_judge_prompt(family),
            MORPHOLOGY_JUDGE_SCHEMA,
            "morphology_judge",
        )
        morphology_problems, morphology_metadata = interpret_morphology_judge(
            family, morphology_response.data, cfg
        )
        judge_problems = sorted(set(semantic_problems + morphology_problems))
        judging_provenance = {
            "semantic": semantic_responses,
            "morphology": _request_provenance(morphology_response),
        }
        judging_metadata = {
            "semantic": semantic_metadata,
            "morphology": morphology_metadata,
        }
        if stage_callback:
            stage_callback(
                "cascade_judges_completed",
                {"refill_round": refill_round, "accepted": not bool(judge_problems)},
            )

        adjudicator_record = None
        if judge_problems and judges.get("adjudicator") is not None:
            adjudicator_response = judges["adjudicator"].call_json(
                ADJUDICATOR_SYSTEM,
                build_adjudicator_prompt(
                    family, semantic_verdicts, morphology_response.data, judge_problems
                ),
                ADJUDICATOR_SCHEMA,
                "judge_disagreement_advisory",
            )
            adjudicator_record = {
                "verdict": adjudicator_response.data,
                "provenance": _request_provenance(adjudicator_response),
            }
            judging_metadata["adjudicator"] = adjudicator_response.data
            judging_provenance["adjudicator"] = adjudicator_record["provenance"]

        family["provenance"] = {
            "generator_id": slot["generator_id"],
            "refill_round": refill_round,
            "rejected_replacements_before_review": refill_history,
            "generator_attempts": generation_provenance,
            "judges": judging_provenance,
            "prompt_version": PROMPT_VERSION,
        }
        family["qc"] = {
            "deterministic": "pass",
            "judging": judging_metadata,
        }
        family["qc"]["quality_score"] = round(quality_score(family), 5)
        family["memory_tags"] = family_memory_tags(family)

        review_reasons = list(judge_problems)
        review_kind = "judge_conflict"
        if not review_reasons and slot.get("human_review_audit"):
            review_reasons = ["deterministic stratified human audit sample"]
            review_kind = "stratified_audit"
        if review_reasons:
            reviewers_key = (
                "audit_reviewers_required"
                if review_kind == "stratified_audit"
                else "conflict_reviewers_required"
            )
            family["source_type"] = "llm_generated_pending_human_review"
            record = {
                "slot_id": slot["slot_id"],
                "family_id": family["family_id"],
                "review_kind": review_kind,
                "review_reasons": review_reasons,
                "reviewers_required": int(
                    cfg["generation"]["human_review"][reviewers_key]
                ),
                "adjudicator": adjudicator_record,
                "family": family,
            }
            if stage_callback:
                stage_callback(
                    "needs_human_review",
                    {"refill_round": refill_round, "review_kind": review_kind},
                )
            return "needs_review", record

        family["source_type"] = "llm_generated_cascade_judged"
        return "accepted", family

    return "rejected", {
        "slot_id": slot["slot_id"],
        "stage": last_stage,
        "problems": last_problems,
        "last_raw": last_raw,
        "last_family": last_family,
        "next_refill_round": start_refill_round + refill_count,
        "refill_history": refill_history,
    }


def _resume_refill_rounds(rejected: list[dict[str, Any]]) -> dict[str, int]:
    rounds: dict[str, int] = {}
    for row in rejected:
        slot_id = row.get("slot_id")
        if slot_id:
            rounds[slot_id] = max(rounds.get(slot_id, 0), int(row.get("next_refill_round", 0)))
    return rounds


def generate(run_id: str, config_path: str | None = None, limit: int | None = None, workers: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=True)
    all_slots = build_plan(cfg)
    slots = all_slots[:limit] if limit is not None else all_slots
    paths = initialise_run(run_id, cfg, all_slots)
    memory = DatasetMemory(paths.memory)
    accepted_before = read_jsonl(paths.accepted)
    needs_review_before = read_jsonl(paths.needs_review)
    rejected_before = read_jsonl(paths.rejected)
    completed = {row.get("slot_id") for row in accepted_before if row.get("slot_id")}
    refill_rounds = _resume_refill_rounds(rejected_before)
    for item in accepted_before:
        if item.get("slot_id"):
            memory.record_outcome(item["slot_id"], "accepted", item, actor="jsonl_reconcile")
    for item in needs_review_before:
        slot_id = item.get("slot_id")
        if slot_id and memory.slot_status(slot_id) not in {"accepted", "rejected"}:
            memory.record_outcome(slot_id, "needs_review", item, actor="jsonl_reconcile")
    review_pending = {
        slot["slot_id"] for slot in slots if memory.slot_status(slot["slot_id"]) == "needs_review"
    }
    audit_slot_ids = human_audit_slots(all_slots, cfg)
    pending = [
        {**slot, "human_review_audit": slot["slot_id"] in audit_slot_ids}
        for slot in slots
        if slot["slot_id"] not in completed and slot["slot_id"] not in review_pending
    ]
    metadata = {"dataset_version": cfg["version"], "prompt_version": PROMPT_VERSION}
    generators = {
        spec["id"]: make_provider(spec, paths.cache / spec["id"], metadata)
        for spec in cfg["generation"]["generators"]
    }
    judge_specs = cfg["generation"]["judges"]
    judges = {
        "semantic": make_provider(judge_specs["semantic"], paths.cache / "semantic_judge", metadata),
        "morphology": make_provider(
            judge_specs["morphology"], paths.cache / "morphology_judge", metadata
        ),
    }
    if judge_specs["adjudicator"].get("enabled", False):
        judges["adjudicator"] = make_provider(
            judge_specs["adjudicator"], paths.cache / "adjudicator", metadata
        )
    write_lock = threading.Lock()
    counts = Counter()
    errors = []
    started = time.time()

    worker_count = int(workers or cfg["generation"].get("workers", 4))
    submitted = 0
    reservation_skips = 0
    pending_iter = iter(pending)
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        future_to_slot: dict[Any, tuple[dict[str, Any], str]] = {}

        def fill_workers() -> None:
            nonlocal submitted, reservation_skips
            while len(future_to_slot) < max(1, worker_count):
                try:
                    slot = next(pending_iter)
                except StopIteration:
                    return
                owner = f"{run_id}:{os.getpid()}:{slot['slot_id']}"
                if not memory.reserve_slot(slot["slot_id"], owner):
                    reservation_skips += 1
                    continue
                prompt_slot = {
                    **slot,
                    "dataset_memory": memory.generation_context(slot),
                }

                def stage_callback(stage, payload, *, _slot=slot, _owner=owner):
                    memory.record_stage(_slot["slot_id"], stage, _owner, payload)

                future = executor.submit(
                    _process_slot, prompt_slot, cfg, generators, judges,
                    refill_rounds.get(slot["slot_id"], 0), stage_callback,
                    memory.conflicts_for,
                )
                future_to_slot[future] = (slot, owner)
                submitted += 1

        fill_workers()
        while future_to_slot:
            done, _ = wait(set(future_to_slot), return_when=FIRST_COMPLETED)
            for future in done:
                slot, owner = future_to_slot.pop(future)
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
                    "needs_review": paths.needs_review,
                    "rejected": paths.rejected,
                    "failed": paths.failures,
                }[status]
                _append_jsonl(target_path, record, write_lock)
                memory.record_outcome(slot["slot_id"], status, record, actor=owner)
                counts[status] += 1
            fill_workers()

    accepted = read_jsonl(paths.accepted)
    needs_review = read_jsonl(paths.needs_review)
    rejected = read_jsonl(paths.rejected)
    failures = read_jsonl(paths.failures)
    accepted_slot_ids = {item.get("slot_id") for item in accepted}
    target_count = len(slots)
    report = {
        "run_id": run_id,
        "elapsed_seconds_this_call": round(time.time() - started, 2),
        "pending_processed_this_call": submitted,
        "reservation_skips_this_call": reservation_skips,
        "outcomes_this_call": dict(counts),
        "accepted_total": len(accepted),
        "needs_review_total": sum(
            memory.slot_status(slot["slot_id"]) == "needs_review" for slot in slots
        ),
        "target_total": target_count,
        "unfilled_slots": sum(slot["slot_id"] not in accepted_slot_ids for slot in slots),
        "complete": all(slot["slot_id"] in accepted_slot_ids for slot in slots),
        "rejected_attempt_batches_total": len(rejected),
        "rejected_slots_ever": len({item.get("slot_id") for item in rejected}),
        "retryable_failures_total": len(failures),
        "accepted_by_bucket": dict(Counter(item["generalization_bucket"] for item in accepted)),
        "accepted_by_query_sentence_count": dict(
            Counter(str(item["query_sentence_count"]) for item in accepted)
        ),
        "accepted_by_passage_sentence_count": dict(
            Counter(str(item["passage_sentence_count"]) for item in accepted)
        ),
        "accepted_by_generator": dict(Counter(item["generator_id"] for item in accepted)),
        "accepted_strict_minimal_pairs": sum(bool(item["strict_minimal_pair"]) for item in accepted),
        "rejection_stages": dict(Counter(item.get("stage", "unknown") for item in rejected)),
        "human_review_reasons": dict(Counter(
            reason
            for item in needs_review
            if memory.slot_status(item.get("slot_id", "")) == "needs_review"
            for reason in item.get("review_reasons", [])
        )),
        "uncaught_errors_this_call": errors,
        "dataset_memory": memory.report(),
    }
    _write_json(paths.report, report)
    return report


def write_plan(run_id: str, config_path: str | None = None, size: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=False)
    slots = build_plan(cfg, size=size)
    paths = paths_for(run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    _write_json(paths.plan, slots)
    DatasetMemory(paths.memory).sync_plan(slots)
    report = {"run_id": run_id, "size": len(slots), "sha256": plan_hash(slots), "statistics": plan_statistics(slots)}
    _write_json(paths.root / "plan_report.json", report)
    return report


def default_run_id() -> str:
    return datetime.now().strftime("test_%Y%m%d_%H%M%S")
