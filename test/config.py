"""Configuration loading and validation for the test-set pipeline."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
_ENV = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


class ConfigError(ValueError):
    pass


def _resolve_env(value: Any, required: bool) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env(v, required) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v, required) for v in value]
    if isinstance(value, str):
        match = _ENV.match(value)
        if match:
            name = match.group(1)
            resolved = os.getenv(name)
            if required and not resolved:
                raise ConfigError(f"Eksik ortam değişkeni: {name}")
            return resolved or value
    return value


def _check_distribution(name: str, values: dict[str, float]) -> None:
    total = sum(float(v) for v in values.values())
    if abs(total - 1.0) > 1e-9:
        raise ConfigError(f"{name} toplamı 1.0 olmalı; bulunan: {total}")
    if any(float(v) <= 0 for v in values.values()):
        raise ConfigError(f"{name} oranlarının tamamı pozitif olmalı")


# A vendor reaches OpenRouter under several route prefixes, and a bare id carries no prefix at
# all. Treating the literal prefix as the family let `google-vertex/gemini-2.5-pro` and
# `gemini-2.5-pro` past the legacy-train ban that `google/gemini-2.5-pro` is caught by — and let
# two routes to the same vendor count as two "distinct" families. Both checks read this function,
# so canonicalising here closes both holes at once.
_FAMILY_ALIASES = {
    "google-vertex": "google",
    "vertex": "google",
    "gemini": "google",
    "gemma": "google",
    "palm": "google",
}


def model_family(model_id: str) -> str:
    """OpenRouter model IDs use `provider/model`; provider is our family boundary."""
    prefix = (model_id.split("/", 1)[0] if "/" in model_id else model_id).strip().lower()
    if prefix in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[prefix]
    # No usable prefix (or an unknown one): fall back to the first vendor marker anywhere in the
    # id, so an unprefixed `gemini-2.5-pro` still resolves to the google family.
    for token in re.split(r"[^a-z0-9]+", model_id.strip().lower()):
        if token in _FAMILY_ALIASES:
            return _FAMILY_ALIASES[token]
    return prefix


def validate_config(cfg: dict[str, Any], runtime: bool = False) -> None:
    counts = cfg["candidate_counts"]
    if counts != {"positive": 1, "hard_negative": 8, "easy_negative": 2}:
        raise ConfigError("Test family yapısı tam olarak 1 positive + 8 hard + 2 easy olmalı")
    if sum(counts.values()) != 11:
        raise ConfigError("Her family tam olarak 11 aday içermeli")
    targets = cfg["targets"]
    if targets["development"] != 100 or targets["sealed_test"] != 500:
        raise ConfigError("Dondurulmuş plan 100 development + 500 sealed test olmalı")
    if "oversample_factor" in targets:
        raise ConfigError("Statik oversampling kaldırıldı; plan doğrudan 600 slot olmalı")
    _check_distribution("query_sentence_distribution", cfg["query_sentence_distribution"])
    _check_distribution("passage_sentence_distribution", cfg["passage_sentence_distribution"])
    _check_distribution("family_mode_distribution", cfg["family_mode_distribution"])
    _check_distribution("query_expression_distribution", cfg["query_expression_distribution"])
    _check_distribution("query_gold_lexical_distribution", cfg["query_gold_lexical_distribution"])
    if cfg["family_mode_distribution"] != {
        "strict_minimal": 0.25,
        "controlled_diverse": 0.45,
        "natural_retrieval": 0.30,
    }:
        raise ConfigError("Family modları tam %25 strict + %45 controlled + %30 natural olmalı")
    if set(cfg["query_sentence_distribution"]) != {"1", "2"}:
        raise ConfigError("Query uzunluk katmanları yalnız 1 ve 2 cümle olmalı")
    if set(cfg["passage_sentence_distribution"]) != {"1", "2", "3", "4"}:
        raise ConfigError("Pasaj uzunluk katmanları 1, 2, 3 ve 4 cümle olmalı")
    _check_distribution("generalization_distribution", cfg["generalization_distribution"])
    quality = cfg["quality"]
    for name in (
        "hard_query_content_recall_min",
        "freeze_tie_aware_word_overlap_recall_at_1_max",
        "freeze_tie_aware_character_3gram_recall_at_1_max",
        "freeze_tie_aware_bm25_recall_at_1_max",
    ):
        if not 0 <= float(quality[name]) <= 1:
            raise ConfigError(f"{name} 0–1 arasında olmalı")
    if set(quality["query_gold_lexical_bands"]) != {"high", "medium", "low"}:
        raise ConfigError("Query–gold lexical bandları high/medium/low olmalı")
    if set(quality["family_mode_lexical_gates"]) != set(cfg["family_mode_distribution"]):
        raise ConfigError("Her family modu için lexical gate tanımlanmalı")
    for name in ("semantic_judge_confidence_min", "morphology_judge_confidence_min"):
        if not 0 <= int(quality[name]) <= 100:
            raise ConfigError(f"{name} 0–100 arasında olmalı")
    for name in ("judge_naturalness_min", "candidate_naturalness_min"):
        if not 1 <= int(quality[name]) <= 5:
            raise ConfigError(f"{name} 1–5 arasında olmalı")

    generation = cfg["generation"]
    judges = generation.get("judges", {})
    if set(judges) != {"semantic", "morphology", "adjudicator"}:
        raise ConfigError("judges semantic + morphology + adjudicator sözleşmesini taşımalı")
    permutations = judges["semantic"].get("permutations", [])
    if len(permutations) != 2 or len(set(permutations)) != 2:
        raise ConfigError("Semantic judge tam iki farklı candidate permütasyonu kullanmalı")
    human = generation.get("human_review", {})
    if not 0 <= float(human.get("audit_rate", -1)) <= 1:
        raise ConfigError("human_review.audit_rate 0–1 arasında olmalı")
    for name in ("audit_reviewers_required", "conflict_reviewers_required"):
        if int(human.get(name, 0)) < 1:
            raise ConfigError(f"human_review.{name} en az 1 olmalı")

    def _judge_specs() -> list[tuple[str, dict[str, Any]]]:
        rows = [("semantic_judge", judges["semantic"]), ("morphology_judge", judges["morphology"])]
        if judges["adjudicator"].get("enabled", False):
            rows.append(("adjudicator", judges["adjudicator"]))
        return rows

    if runtime:
        generators = generation.get("generators", [])
        if len(generators) != 2 or {row.get("id") for row in generators} != {"generator_a", "generator_b"}:
            raise ConfigError("Ana test üretimi generator_a + generator_b olmak üzere iki generator ister")
        judge_specs = _judge_specs()
        role_specs = [*((row["id"], row) for row in generators), *judge_specs]
        for label, spec in role_specs:
            if not spec.get("model") or str(spec["model"]).startswith("${"):
                raise ConfigError(f"{label} model kimliği ayarlanmamış")
            if not os.getenv(spec["api_key_env"]):
                raise ConfigError(f"{label} için {spec['api_key_env']} tanımlı değil")
            preferences = spec.get("provider_preferences", {})
            if preferences.get("require_parameters") is not True:
                raise ConfigError(f"{label} structured-output routing'i zorunlu tutmalı")
            if preferences.get("data_collection") != "deny" or preferences.get("zdr") is not True:
                raise ConfigError(f"{label} test verisi için data_collection=deny ve zdr=true kullanmalı")
        if generation.get("require_distinct_model_families", True):
            families = [model_family(spec["model"]) for _, spec in role_specs]
            if len(families) != len(set(families)):
                raise ConfigError(
                    "Generator ve bağımsız judge rolleri farklı model ailelerinden olmalı: "
                    + ", ".join(spec["model"] for _, spec in role_specs)
                )
        forbidden = {
            str(value).lower()
            for value in generation.get("forbidden_model_families_for_test", [])
        }
        used = {model_family(spec["model"]) for _, spec in role_specs}
        if forbidden & used:
            raise ConfigError(
                "Test generator/judge, legacy train model ailesinden bağımsız olmalı; "
                f"yasak aile kullanıldı: {sorted(forbidden & used)}"
            )
    if int(generation.get("max_generation_attempts", 0)) < 1:
        raise ConfigError("max_generation_attempts en az 1 olmalı")
    if int(generation.get("refill_rounds_per_call", 0)) < 1:
        raise ConfigError("refill_rounds_per_call en az 1 olmalı")


def load_config(path: str | Path | None = None, runtime: bool = False) -> dict[str, Any]:
    source = Path(path) if path else DEFAULT_CONFIG
    cfg = json.loads(source.read_text(encoding="utf-8"))
    cfg = _resolve_env(deepcopy(cfg), required=runtime)
    validate_config(cfg, runtime=runtime)
    cfg["_config_path"] = str(source.resolve())
    return cfg


def final_target(cfg: dict[str, Any]) -> int:
    return int(cfg["targets"]["development"] + cfg["targets"]["sealed_test"])
