"""Deterministic balanced selection for the review pool and frozen 600-family release."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any


def allocate_cells(total: int, cfg: dict[str, Any]) -> dict[tuple[str, int, int], int]:
    """Allocate bucket × passage cells, then preserve the exact global query marginal."""
    raw = []
    for bucket, bucket_share in cfg["generalization_distribution"].items():
        for sentence, sentence_share in cfg["passage_sentence_distribution"].items():
            exact = total * float(bucket_share) * float(sentence_share)
            raw.append(((bucket, int(sentence)), math.floor(exact), exact - math.floor(exact)))
    base_counts = {cell: floor for cell, floor, _ in raw}
    remainder = total - sum(base_counts.values())
    for cell, _, _ in sorted(raw, key=lambda row: (-row[2], row[0]))[:remainder]:
        base_counts[cell] += 1

    query_distribution = {
        int(sentence): float(share)
        for sentence, share in cfg["query_sentence_distribution"].items()
    }
    if set(query_distribution) != {1, 2}:
        raise ValueError("Dengeli seçim şu anda 1/2-cümle query tasarımı bekliyor")
    exact_query_one = total * query_distribution[1]
    target_query_one = math.floor(exact_query_one)
    if exact_query_one - target_query_one >= 0.5:
        target_query_one += 1
    query_one = {
        cell: math.floor(count * query_distribution[1]) for cell, count in base_counts.items()
    }
    add_one = target_query_one - sum(query_one.values())
    priority = sorted(
        base_counts,
        key=lambda cell: (
            -(base_counts[cell] * query_distribution[1] - query_one[cell]), cell
        ),
    )
    for cell in priority[:add_one]:
        query_one[cell] += 1

    counts = {}
    for (bucket, passage_sentences), count in base_counts.items():
        counts[(bucket, passage_sentences, 1)] = query_one[(bucket, passage_sentences)]
        counts[(bucket, passage_sentences, 2)] = count - query_one[(bucket, passage_sentences)]
    return counts


def _diverse_take(
    items: list[dict], count: int, position_counts: Counter | None = None
) -> list[dict]:
    remaining = list(items)
    chosen = []
    macros = Counter()
    features = Counter()
    positions = position_counts if position_counts is not None else Counter()
    while remaining and len(chosen) < count:
        def score(item):
            quality = float(item.get("qc", {}).get("quality_score", 0.0))
            macro_bonus = 0.30 / (1 + macros[item["macro_phenomenon"]])
            feature_bonus = 0.15 / (1 + features[item["target_feature"]])
            position = int(item["critical_sentence_position"])
            return (
                -positions[position],
                quality + macro_bonus + feature_bonus,
                item["family_id"],
            )

        best = max(remaining, key=score)
        remaining.remove(best)
        chosen.append(best)
        macros[best["macro_phenomenon"]] += 1
        features[best["target_feature"]] += 1
        positions[int(best["critical_sentence_position"])] += 1
    return chosen


def _fit_query_availability(
    desired: dict[tuple[str, int, int], int],
    by_cell: dict[tuple[str, int, int], list[dict]],
) -> dict[tuple[str, int, int], int]:
    """Keep bucket/passage totals and global 75/25 marginal despite sparse raw subcells."""
    base_needed = Counter()
    for (bucket, passage, _query), count in desired.items():
        base_needed[(bucket, passage)] += count
    target_query_one = sum(
        count for (_bucket, _passage, query), count in desired.items() if query == 1
    )
    bounds = {}
    for base, needed in base_needed.items():
        available_one = len(by_cell.get((*base, 1), []))
        available_two = len(by_cell.get((*base, 2), []))
        lower = max(0, needed - available_two)
        upper = min(needed, available_one)
        if lower > upper:
            raise ValueError(
                f"{base[0]}|p{base[1]} için {needed} kayıt yok: "
                f"q1={available_one}, q2={available_two}"
            )
        preferred = desired.get((*base, 1), 0)
        bounds[base] = (lower, upper, preferred)
    if not sum(row[0] for row in bounds.values()) <= target_query_one <= sum(
        row[1] for row in bounds.values()
    ):
        raise ValueError("Global query uzunluk marjinali mevcut havuzla sağlanamıyor")

    query_one = {base: lower for base, (lower, _upper, _preferred) in bounds.items()}
    remaining = target_query_one - sum(query_one.values())
    while remaining:
        choices = [base for base, value in query_one.items() if value < bounds[base][1]]
        best = max(
            choices,
            key=lambda base: (bounds[base][2] - query_one[base], base),
        )
        query_one[best] += 1
        remaining -= 1
    fitted = {}
    for base, needed in base_needed.items():
        fitted[(*base, 1)] = query_one[base]
        fitted[(*base, 2)] = needed - query_one[base]
    return fitted


def select_balanced(
    items: list[dict[str, Any]],
    cfg: dict[str, Any],
    split_targets: dict[str, int],
) -> list[dict[str, Any]]:
    """Select exact bucket × passage × query-length cells or fail loudly."""
    selected: list[dict[str, Any]] = []
    selected_dev_lemmas: set[str] = set()
    for split in ("development", "sealed_test"):
        target = int(split_targets[split])
        cells = allocate_cells(target, cfg)
        by_cell = defaultdict(list)
        for item in items:
            if item.get("target_split") != split:
                continue
            if split == "sealed_test" and item.get("generalization_bucket") == "lemma_holdout":
                lemma = str(item.get("critical_lemma", "")).strip().lower()
                if lemma in selected_dev_lemmas:
                    continue
            by_cell[(
                item["generalization_bucket"],
                int(item["passage_sentence_count"]),
                int(item["query_sentence_count"]),
            )].append(item)
        cells = _fit_query_availability(cells, by_cell)

        shortages = {}
        split_selected = []
        positions_by_passage = defaultdict(Counter)
        for cell, needed in sorted(cells.items()):
            available = by_cell.get(cell, [])
            if len(available) < needed:
                shortages[f"{cell[0]}|p{cell[1]}|q{cell[2]}"] = {
                    "needed": needed, "available": len(available)
                }
                continue
            split_selected.extend(
                _diverse_take(available, needed, positions_by_passage[cell[1]])
            )
        if shortages:
            raise ValueError(f"{split} dengeli seçim için yetersiz hücreler: {shortages}")
        if len(split_selected) != target:
            raise AssertionError(f"{split} seçim sayısı {len(split_selected)} != {target}")
        if split == "development":
            selected_dev_lemmas = {
                str(item.get("critical_lemma", "")).strip().lower() for item in split_selected
            }
        selected.extend(split_selected)
    return selected


def selection_statistics(items: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "target_split", "generalization_bucket", "query_sentence_count",
        "passage_sentence_count", "critical_sentence_position", "layer", "objective",
        "macro_phenomenon", "target_feature", "domain", "register",
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
