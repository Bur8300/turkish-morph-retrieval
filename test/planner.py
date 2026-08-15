"""Prefix-stable deterministic coverage planner for raw test-family generation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any

from .config import raw_target
from .taxonomy import DOMAINS, FEATURES, MACROS, REGISTERS, TEMPLATES, hard_profile


def _seed(base: int, *parts: object) -> int:
    raw = "|".join([str(base), *(str(part) for part in parts)])
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _weighted_value(
    distribution: dict[str, float], index: int, cycle_size: int, seed: int, salt: str
) -> str:
    """Pick from independently shuffled fixed-size cycles; prefixes never change with target size."""
    counts = {key: int(round(float(share) * cycle_size)) for key, share in distribution.items()}
    delta = cycle_size - sum(counts.values())
    order = sorted(distribution, key=lambda key: (-distribution[key], key))
    for offset in range(abs(delta)):
        key = order[offset % len(order)]
        counts[key] += 1 if delta > 0 else -1
    values = [key for key, count in counts.items() for _ in range(count)]
    cycle = index // cycle_size
    random.Random(_seed(seed, salt, cycle)).shuffle(values)
    return values[index % cycle_size]


def _round_robin(values: tuple | list, index: int, seed: int, salt: str) -> Any:
    block = list(values)
    cycle = index // len(block)
    random.Random(_seed(seed, salt, cycle)).shuffle(block)
    return block[index % len(block)]


DEV_COMPOSITION_KEYS = {"NEG.AOR", "PLUPRF", "PST.PROG", "FUT.PST"}
SEALED_DOMAIN_REGISTER_PAIRS = (
    ("health", "conversational"),
    ("finance", "news_report"),
    ("law_public_services", "everyday"),
    ("ecommerce", "formal_record"),
)


def _feature_for(
    index: int,
    layer: str,
    macro: str,
    seed: int,
    target_split: str,
    bucket: str,
):
    pool = [feature for feature in FEATURES if feature.layer == layer and feature.macro == macro]
    if layer == "chain" and target_split == "development":
        pool = [feature for feature in pool if feature.key in DEV_COMPOSITION_KEYS]
    elif layer == "chain" and bucket == "composition_holdout":
        pool = [feature for feature in pool if feature.key not in DEV_COMPOSITION_KEYS]
    if not pool:
        pool = [feature for feature in FEATURES if feature.layer == layer]
    return _round_robin(pool, index, seed, f"feature:{layer}:{macro}")


def make_slot(index: int, cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg["seed"])
    bucket = _weighted_value(cfg["generalization_distribution"], index, 10, seed, "bucket")
    query_sentence_count = int(
        _weighted_value(cfg["query_sentence_distribution"], index, 4, seed, "query_sentences")
    )
    passage_sentence_count = int(
        _weighted_value(cfg["passage_sentence_distribution"], index, 10, seed, "passage_sentences")
    )
    critical_sentence_position = 1 + (
        _seed(seed, "critical_position", index) % passage_sentence_count
    )
    target_split = _weighted_value(
        {"development": 1 / 6, "sealed_test": 5 / 6}, index, 6, seed, "split"
    )
    strict_minimal_pair = _weighted_value(
        {"no": 1.0 - float(cfg["strict_minimal_pair_fraction"]),
         "yes": float(cfg["strict_minimal_pair_fraction"])},
        index, 4, seed, "strict_minimal_pair",
    ) == "yes"
    generator_ids = [row["id"] for row in cfg["generation"]["generators"]]
    generator_id = _round_robin(generator_ids, index, seed, "generator")

    if bucket == "composition_holdout":
        layer = "chain"
    else:
        chain_rate = float(cfg["non_composition_chain_rate"])
        layer = _weighted_value(
            {"single": 1.0 - chain_rate, "chain": chain_rate}, index, 4, seed, "layer"
        )

    available_macros = [macro for macro in MACROS if (macro == "suffix_chain_composition") == (layer == "chain")]
    force_allomorph = False
    if layer == "single":
        force_allomorph = _weighted_value(
            {"no": 1.0 - float(cfg["single_allomorph_rate"]), "yes": float(cfg["single_allomorph_rate"])},
            index,
            6,
            seed,
            "allomorph",
        ) == "yes"
    if force_allomorph:
        macro = "derivation_nonfinite_allomorphy"
        allomorphs = [feature for feature in FEATURES if feature.objective == "allomorph_invariance"]
        feature = _round_robin(allomorphs, index, seed, "allomorph_feature")
    else:
        macro = _round_robin(available_macros, index, seed, f"macro:{layer}")
        feature = _feature_for(index, layer, macro, seed, target_split, bucket)
        if feature.objective == "allomorph_invariance":
            non_allomorph = [
                candidate for candidate in FEATURES
                if candidate.layer == "single" and candidate.macro == macro
                and candidate.objective != "allomorph_invariance"
            ]
            feature = _round_robin(non_allomorph, index, seed, "non_allomorph_feature")
    domain = _round_robin(DOMAINS, index, seed, "domain")
    register = _round_robin(REGISTERS, index, seed, "register")

    # Template holdout uses a dedicated abstract-template subset. The future train generator must
    # consume the frozen holdout manifest and exclude these IDs, not infer disjointness from prose.
    if bucket == "template_holdout":
        template_pool = TEMPLATES[-4:-2] if target_split == "development" else TEMPLATES[-2:]
    else:
        template_pool = TEMPLATES[:-4]
    template = _round_robin(template_pool, index, seed, f"template:{bucket}")
    domain_shift = _weighted_value(
        {"no": 1.0 - float(cfg["domain_shift_fraction"]), "yes": float(cfg["domain_shift_fraction"])},
        index,
        5,
        seed,
        "domain_shift",
    ) == "yes"
    if target_split == "sealed_test" and domain_shift:
        domain, register = _round_robin(
            SEALED_DOMAIN_REGISTER_PAIRS, index, seed, "sealed_domain_register_pair"
        )
    elif target_split == "development" and (domain, register) in SEALED_DOMAIN_REGISTER_PAIRS:
        register = REGISTERS[(REGISTERS.index(register) + 1) % len(REGISTERS)]

    fingerprint = hashlib.sha256(
        f"{cfg['version']}|{index}|{feature.key}|{bucket}|{query_sentence_count}|"
        f"{passage_sentence_count}|{critical_sentence_position}|{domain}|{template['id']}".encode()
    ).hexdigest()[:10]
    slot_id = f"raw_{index:05d}_{fingerprint}"
    tags = [bucket]
    if domain_shift:
        tags.append("domain_shift")

    return {
        "index": index,
        "slot_id": slot_id,
        "target_split": target_split,
        "generalization_bucket": bucket,
        "generalization_tags": tags,
        "objective": feature.objective,
        "layer": layer,
        "macro_phenomenon": feature.macro,
        "feature": feature.to_dict(),
        "domain": domain,
        "register": register,
        "query_sentence_count": query_sentence_count,
        "passage_sentence_count": passage_sentence_count,
        "critical_sentence_position": critical_sentence_position,
        "hard_profile": hard_profile(feature),
        "strict_minimal_pair": strict_minimal_pair,
        "generator_id": generator_id,
        "template": dict(template),
        "semantic_frame_id": f"frame_{index:05d}",
        "lemma_policy": "test_only_critical_lemma" if bucket == "lemma_holdout" else "open_critical_lemma",
        "composition_policy": (
            "components_seen_chain_unseen" if bucket == "composition_holdout" else "open_composition"
        ),
    }


def build_plan(cfg: dict[str, Any], size: int | None = None) -> list[dict[str, Any]]:
    return [make_slot(index, cfg) for index in range(size if size is not None else raw_target(cfg))]


def plan_statistics(slots: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = (
        "target_split", "generalization_bucket", "objective", "layer", "macro_phenomenon",
        "domain", "register", "query_sentence_count", "passage_sentence_count",
        "critical_sentence_position",
        "strict_minimal_pair", "generator_id",
    )
    return {
        field: dict(sorted(Counter(str(slot[field]) for slot in slots).items())) for field in fields
    }


def plan_hash(slots: list[dict[str, Any]]) -> str:
    payload = json.dumps(slots, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
