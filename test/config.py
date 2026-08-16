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


def model_family(model_id: str) -> str:
    """OpenRouter model IDs use `provider/model`; provider is our family boundary."""
    return model_id.split("/", 1)[0].strip().lower() if "/" in model_id else model_id.lower()


def validate_config(cfg: dict[str, Any], runtime: bool = False) -> None:
    counts = cfg["candidate_counts"]
    if counts != {"positive": 1, "hard_negative": 8, "easy_negative": 2}:
        raise ConfigError("Test family yapısı tam olarak 1 positive + 8 hard + 2 easy olmalı")
    if sum(counts.values()) != 11:
        raise ConfigError("Her family tam olarak 11 aday içermeli")
    targets = cfg["targets"]
    if targets["development"] != 100 or targets["sealed_test"] != 500:
        raise ConfigError("Dondurulmuş plan 100 development + 500 sealed test olmalı")
    _check_distribution("query_sentence_distribution", cfg["query_sentence_distribution"])
    _check_distribution("passage_sentence_distribution", cfg["passage_sentence_distribution"])
    if set(cfg["query_sentence_distribution"]) != {"1", "2"}:
        raise ConfigError("Query uzunluk katmanları yalnız 1 ve 2 cümle olmalı")
    if set(cfg["passage_sentence_distribution"]) != {"1", "2", "3", "4"}:
        raise ConfigError("Pasaj uzunluk katmanları 1, 2, 3 ve 4 cümle olmalı")
    _check_distribution("generalization_distribution", cfg["generalization_distribution"])
    if float(cfg.get("strict_minimal_pair_fraction", 0)) != 0.25:
        raise ConfigError("Final plan tam olarak %25 strict minimal-pair (150/600) olmalı")
    quality = cfg["quality"]
    if not 1 <= int(quality["lexical_hards_at_or_above_gold_min"]) <= 8:
        raise ConfigError("lexical_hards_at_or_above_gold_min 1–8 arasında olmalı")
    if not 1 <= int(quality["content_preserving_hards_min"]) <= 8:
        raise ConfigError("content_preserving_hards_min 1–8 arasında olmalı")
    for name in (
        "gold_hard_overlap_median_gap_max", "hard_query_content_recall_min",
        "freeze_tie_aware_word_overlap_recall_at_1_max",
        "freeze_tie_aware_character_3gram_recall_at_1_max",
        "freeze_tie_aware_bm25_recall_at_1_max",
    ):
        if not 0 <= float(quality[name]) <= 1:
            raise ConfigError(f"{name} 0–1 arasında olmalı")
    if runtime:
        generators = cfg["generation"].get("generators", [])
        if len(generators) != 2 or {row.get("id") for row in generators} != {"generator_a", "generator_b"}:
            raise ConfigError("Ana test üretimi generator_a + generator_b olmak üzere iki generator ister")
        judge = cfg["generation"]["judge"]
        for label, spec in (*((row["id"], row) for row in generators), ("judge", judge)):
            if not spec.get("model") or str(spec["model"]).startswith("${"):
                raise ConfigError(f"{label} model kimliği ayarlanmamış")
            if not os.getenv(spec["api_key_env"]):
                raise ConfigError(f"{label} için {spec['api_key_env']} tanımlı değil")
        if cfg["generation"].get("require_distinct_model_families", True):
            families = [model_family(row["model"]) for row in generators] + [model_family(judge["model"])]
            if len(families) != len(set(families)):
                raise ConfigError(
                    "İki generator ve judge üç farklı model ailesinden olmalı: "
                    + ", ".join(row["model"] for row in [*generators, judge])
                )
        forbidden = {
            str(value).lower()
            for value in cfg["generation"].get("forbidden_model_families_for_test", [])
        }
        used = {model_family(row["model"]) for row in [*generators, judge]}
        if forbidden & used:
            raise ConfigError(
                "Test generator/judge, legacy train model ailesinden bağımsız olmalı; "
                f"yasak aile kullanıldı: {sorted(forbidden & used)}"
            )


def load_config(path: str | Path | None = None, runtime: bool = False) -> dict[str, Any]:
    source = Path(path) if path else DEFAULT_CONFIG
    cfg = json.loads(source.read_text(encoding="utf-8"))
    cfg = _resolve_env(deepcopy(cfg), required=runtime)
    validate_config(cfg, runtime=runtime)
    cfg["_config_path"] = str(source.resolve())
    return cfg


def final_target(cfg: dict[str, Any]) -> int:
    return int(cfg["targets"]["development"] + cfg["targets"]["sealed_test"])


def raw_target(cfg: dict[str, Any]) -> int:
    return int(round(final_target(cfg) * float(cfg["targets"]["oversample_factor"])))
