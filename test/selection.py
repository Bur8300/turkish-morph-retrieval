"""Quota checks for the frozen 600-family release; generation already owns the slots."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _expected(total: int, distribution: dict[str, float]) -> dict[str, int]:
    counts = {key: int(round(total * float(share))) for key, share in distribution.items()}
    if sum(counts.values()) != total:
        raise ValueError(f"Dağılım {total} kayıt için tam sayıya ayrılamıyor: {counts}")
    return counts


def _check_counts(label: str, actual: Counter, expected: dict[Any, int]) -> None:
    if dict(actual) != expected:
        raise ValueError(f"{label} kotası yanlış: {dict(actual)} != {expected}")


def select_balanced(
    items: list[dict[str, Any]],
    cfg: dict[str, Any],
    split_targets: dict[str, int],
) -> list[dict[str, Any]]:
    """Require one accepted family for every planned quota slot; never cherry-pick quality."""
    wanted_total = sum(int(value) for value in split_targets.values())
    if len(items) != wanted_total:
        raise ValueError(f"Freeze için {wanted_total} accepted family gerekir; bulunan: {len(items)}")
    family_ids = [item["family_id"] for item in items]
    slot_ids = [item["slot_id"] for item in items]
    if len(set(family_ids)) != wanted_total or len(set(slot_ids)) != wanted_total:
        raise ValueError("Accepted family_id/slot_id değerleri benzersiz değil")

    _check_counts(
        "target_split", Counter(item["target_split"] for item in items),
        {key: int(value) for key, value in split_targets.items()},
    )
    generator_ids = [row["id"] for row in cfg["generation"]["generators"]]
    if len(generator_ids) != 2:
        raise ValueError("Freeze iki generator bekliyor")

    for split, target_value in split_targets.items():
        target = int(target_value)
        subset = [item for item in items if item["target_split"] == split]
        _check_counts(
            f"{split} generalization",
            Counter(item["generalization_bucket"] for item in subset),
            _expected(target, cfg["generalization_distribution"]),
        )
        _check_counts(
            f"{split} query_sentence_count",
            Counter(str(item["query_sentence_count"]) for item in subset),
            _expected(target, cfg["query_sentence_distribution"]),
        )
        _check_counts(
            f"{split} passage_sentence_count",
            Counter(str(item["passage_sentence_count"]) for item in subset),
            _expected(target, cfg["passage_sentence_distribution"]),
        )
        strict_target = int(round(target * float(cfg["strict_minimal_pair_fraction"])))
        _check_counts(
            f"{split} strict_minimal_pair",
            Counter(bool(item["strict_minimal_pair"]) for item in subset),
            {False: target - strict_target, True: strict_target},
        )
        _check_counts(
            f"{split} generator_id",
            Counter(item["generator_id"] for item in subset),
            {generator_ids[0]: target // 2, generator_ids[1]: target - target // 2},
        )

    return sorted(items, key=lambda item: (int(item.get("index", 0)), item["family_id"]))


def selection_statistics(items: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "target_split", "generalization_bucket", "query_sentence_count",
        "passage_sentence_count", "critical_sentence_position", "layer", "objective",
        "macro_phenomenon", "target_feature", "domain", "register",
        "strict_minimal_pair", "generator_id",
    )
    statistics = {}
    for field in fields:
        values = []
        for item in items:
            value = item.get(field)
            if field == "target_split" and value is None:
                value = item.get("split")
            if value is not None:
                values.append(str(value))
        statistics[field] = dict(sorted(Counter(values).items()))
    return statistics
