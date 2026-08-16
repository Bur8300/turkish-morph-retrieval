"""Dependency-free regression tests for the new test-set package."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from .config import load_config
from .evaluation import evaluate_run, validate_binary_qrels
from .planner import build_plan, make_slot, plan_statistics
from .selection import select_balanced
from .taxonomy import FEATURE_BY_KEY, hard_profile
from .validators import interpret_judge, normalize_family, validate_family


def _fixture_slot(cfg):
    slot = make_slot(0, cfg)
    slot.update({
        "slot_id": "fixture_neg_0000",
        "target_split": "development",
        "generalization_bucket": "standard",
        "generalization_tags": ["standard"],
        "objective": "morpheme_sensitivity",
        "layer": "single",
        "macro_phenomenon": FEATURE_BY_KEY["NEG"].macro,
        "feature": FEATURE_BY_KEY["NEG"].to_dict(),
        "domain": "daily_life",
        "register": "everyday",
        "query_sentence_count": 1,
        "passage_sentence_count": 1,
        "critical_sentence_position": 1,
        "hard_profile": hard_profile(FEATURE_BY_KEY["NEG"]),
        "template": {"id": "event_report", "description": "doğal bir olay bildirimi"},
        "semantic_frame_id": "frame_fixture",
        "strict_minimal_pair": False,
        "generator_id": "generator_a",
    })
    return slot


def _fixture_raw():
    candidates = [
        ("positive_01", "positive", "equivalence_positive", "Ece raporu toplantıdan önce bitirmedi.", "bitirmedi", "target_preserved"),
        ("hard_01", "hard_negative", "minimal_morph_negative", "Ece raporu toplantıdan önce tamamladı.", "tamamladı", "feature_changed"),
        ("hard_02", "hard_negative", "same_lemma_wrong_inflection", "Ece raporu toplantıdan önce tamamlayacak.", "tamamlayacak", "wrong_inflection"),
        ("hard_03", "hard_negative", "related_feature_negative", "Ece raporu toplantıdan sonra tamamlamadı.", "tamamlamadı", "wrong_inflection"),
        ("hard_04", "hard_negative", "same_morph_wrong_content", "Ece sunumu toplantıdan önce tamamlamadı.", "tamamlamadı", "same_feature_wrong_content"),
        ("hard_05", "hard_negative", "state_participant_time_trap", "Mert raporu toplantıdan önce tamamlamadı.", "tamamlamadı", "target_preserved"),
        ("hard_06", "hard_negative", "close_paraphrase_wrong_meaning", "Ece raporu toplantıdan önce başlatmadı.", "başlatmadı", "feature_changed"),
        ("hard_07", "hard_negative", "argument_role_reversal", "Rapor Ece'yi toplantıdan önce tamamlamadı.", "tamamlamadı", "target_preserved"),
        ("hard_08", "hard_negative", "morph_distractor", "Ece toplantıyı bitirmedi ama rapor tamamlandı.", "bitirmedi", "feature_changed"),
        ("easy_01", "easy_negative", "easy_negative", "Mert otobüs biletini sabah erkenden aldı.", "aldı", "unrelated"),
        ("easy_02", "easy_negative", "easy_negative", "Derya bahçedeki çiçekleri akşam suladı.", "suladı", "unrelated"),
    ]
    return {
        "semantic_frame_id": "frame_fixture",
        "template_id": "event_report",
        "critical_lemma": "tamamlamak",
        "critical_word_query": "tamamlamadı",
        "critical_word_positive": "bitirmedi",
        "feature_delta": "NEG korunur; minimal negatifte NEG kaldırılır",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Ece raporu toplantıdan önce tamamlamadı.",
        "context_sentences": [],
        "candidates": [
            {"candidate_slot": candidate_slot, "role": role, "subtype": subtype,
             "critical_sentence": text, "critical_word": word,
             "morph_relation": relation, "reason": subtype}
            for candidate_slot, role, subtype, text, word, relation in candidates
        ],
        "generation_notes": "fixture",
    }


def run() -> list[str]:
    failures = []
    metric_summary, metric_rows = evaluate_run(
        {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}},
        {"q1": ["x", "d1"], "q2": ["d2", "x"]},
    )
    if metric_summary.get("recall@1") != 0.5 or metric_summary.get("mrr@10") != 0.75:
        failures.append(f"closed retrieval metrikleri yanlış: {metric_summary}")
    if "map@10" in metric_summary or "bpref" in metric_summary:
        failures.append("tek-gold düzende gereksiz MAP@10/bpref hâlâ raporlanıyor")
    if metric_summary.get("mean_rank") != 1.5 or len(metric_rows) != 2:
        failures.append(f"rank özeti yanlış: {metric_summary}")
    condensed, condensed_rows = evaluate_run(
        {"q1": {"d1": 1.0, "n1": 0.0}}, {"q1": ["unjudged", "d1", "n1"]},
        unjudged_policy="condensed",
    )
    if condensed.get("recall@1") != 1.0 or condensed_rows[0].get("judged@1") != 0.0:
        failures.append(f"unjudged condensed evaluation yanlış: {condensed} / {condensed_rows}")
    try:
        validate_binary_qrels({"q1": {"d1": 2.0, "n1": 0.0}})
        failures.append("binary qrels relevance=2 değerini reddetmedi")
    except ValueError:
        pass

    cfg = load_config(runtime=False)
    slots_600 = build_plan(cfg, 600)
    stats = plan_statistics(slots_600)
    expected = {
        "target_split": {"development": 100, "sealed_test": 500},
        "query_sentence_count": {"1": 450, "2": 150},
        "passage_sentence_count": {"1": 180, "2": 180, "3": 180, "4": 60},
        "generalization_bucket": {
            "composition_holdout": 120, "lemma_holdout": 120,
            "standard": 240, "template_holdout": 120,
        },
        "strict_minimal_pair": {"False": 450, "True": 150},
        "generator_id": {"generator_a": 300, "generator_b": 300},
    }
    for field, wanted in expected.items():
        if stats[field] != wanted:
            failures.append(f"plan {field}: {stats[field]} != {wanted}")
    if build_plan(cfg, 50) != build_plan(cfg, 600)[:50]:
        failures.append("planner prefix-stable değil")
    dev_templates = {slot["template"]["id"] for slot in slots_600 if slot["target_split"] == "development"}
    test_holdout_templates = {
        slot["template"]["id"] for slot in slots_600
        if slot["target_split"] == "sealed_test" and slot["generalization_bucket"] == "template_holdout"
    }
    if dev_templates & test_holdout_templates:
        failures.append("sealed template holdout development planına sızdı")
    dev_chains = {
        slot["feature"]["key"] for slot in slots_600
        if slot["target_split"] == "development" and slot["layer"] == "chain"
    }
    test_holdout_chains = {
        slot["feature"]["key"] for slot in slots_600
        if slot["target_split"] == "sealed_test" and slot["generalization_bucket"] == "composition_holdout"
    }
    if dev_chains & test_holdout_chains:
        failures.append("sealed composition holdout development planına sızdı")
    dev_domain_pairs = {
        (slot["domain"], slot["register"]) for slot in slots_600
        if slot["target_split"] == "development"
    }
    test_shift_pairs = {
        (slot["domain"], slot["register"]) for slot in slots_600
        if slot["target_split"] == "sealed_test" and "domain_shift" in slot["generalization_tags"]
    }
    if dev_domain_pairs & test_shift_pairs:
        failures.append("sealed domain/register shift development planına sızdı")

    fake_items = []
    for slot_item in build_plan(cfg):
        fake_items.append({
            "family_id": f"fake_{slot_item['slot_id']}",
            "target_split": slot_item["target_split"],
            "generalization_bucket": slot_item["generalization_bucket"],
            "query_sentence_count": slot_item["query_sentence_count"],
            "passage_sentence_count": slot_item["passage_sentence_count"],
            "critical_sentence_position": slot_item["critical_sentence_position"],
            "macro_phenomenon": slot_item["macro_phenomenon"],
            "target_feature": slot_item["feature"]["key"],
            "critical_lemma": f"lemma_{slot_item['index']}",
            "strict_minimal_pair": slot_item["strict_minimal_pair"],
            "generator_id": slot_item["generator_id"],
            "qc": {"quality_score": 1.0},
        })
    try:
        final_selection = select_balanced(
            fake_items, cfg, {"development": 100, "sealed_test": 500}
        )
        if len(final_selection) != 600:
            failures.append("600-family final balanced selection sayısı yanlış")
        for split, query_expected, passage_expected in (
            ("development", {1: 75, 2: 25}, {1: 30, 2: 30, 3: 30, 4: 10}),
            ("sealed_test", {1: 375, 2: 125}, {1: 150, 2: 150, 3: 150, 4: 50}),
        ):
            split_items = [item for item in final_selection if item["target_split"] == split]
            query_actual = dict(Counter(item["query_sentence_count"] for item in split_items))
            passage_actual = dict(Counter(item["passage_sentence_count"] for item in split_items))
            if query_actual != query_expected:
                failures.append(f"{split} query dağılımı yanlış: {query_actual}")
            if passage_actual != passage_expected:
                failures.append(f"{split} passage dağılımı yanlış: {passage_actual}")
            strict_expected = 25 if split == "development" else 125
            if sum(item["strict_minimal_pair"] for item in split_items) != strict_expected:
                failures.append(f"{split} strict minimal-pair kotası yanlış")
            generator_expected = (
                {"generator_a": 50, "generator_b": 50} if split == "development"
                else {"generator_a": 250, "generator_b": 250}
            )
            if dict(Counter(item["generator_id"] for item in split_items)) != generator_expected:
                failures.append(f"{split} generator dengesi yanlış")
            for passage_count in (2, 3, 4):
                position_counts = Counter(
                    item["critical_sentence_position"] for item in split_items
                    if item["passage_sentence_count"] == passage_count
                )
                if len(position_counts) != passage_count or (
                    max(position_counts.values()) - min(position_counts.values()) > 2
                ):
                    failures.append(
                        f"{split} p{passage_count} kritik konum dağılımı dengesiz: "
                        f"{dict(position_counts)}"
                    )
    except Exception as exc:
        failures.append(f"balanced selection başarısız: {type(exc).__name__}: {exc}")

    slot = _fixture_slot(cfg)
    family = normalize_family(_fixture_raw(), slot)
    problems = validate_family(family, slot, cfg)
    if problems:
        failures.append(f"geçerli fixture reddedildi: {problems}")
    hard = {
        candidate["candidate_slot"]: candidate["subtype"]
        for candidate in family["candidates"] if candidate["role"] == "hard_negative"
    }
    expected_hard = {item["slot"]: item["subtype"] for item in slot["hard_profile"]}
    if hard != expected_hard:
        failures.append("fixture uyarlanabilir sekiz hard slotu kapsamıyor")
    if Counter(family["qrels"].values()) != {0: 10, 1: 1}:
        failures.append("family oluşturulurken 1 gold + 10 negative qrels üretilmedi")

    strict_slot = deepcopy(slot)
    strict_slot["strict_minimal_pair"] = True
    strict_raw = _fixture_raw()
    strict_raw["critical_word_positive"] = "bitirmedi"
    positive_raw = next(row for row in strict_raw["candidates"] if row["candidate_slot"] == "positive_01")
    minimal_raw = next(row for row in strict_raw["candidates"] if row["candidate_slot"] == "hard_01")
    positive_raw.update({
        "critical_sentence": "Ece raporu toplantıdan önce bitirmedi.",
        "critical_word": "bitirmedi",
    })
    minimal_raw.update({
        "critical_sentence": "Ece raporu toplantıdan önce bitirdi.",
        "critical_word": "bitirdi",
    })
    strict_raw["edit_script"] = {
        "applies": True, "positive_form": "bitirmedi",
        "minimal_negative_form": "bitirdi", "operation": "NEG ekini kaldır",
        "changed_feature": "NEG", "invariants": ["lemma", "token_order", "event"],
    }
    strict_family = normalize_family(strict_raw, strict_slot)
    strict_problems = validate_family(strict_family, strict_slot, cfg)
    if strict_problems:
        failures.append(f"geçerli strict minimal pair reddedildi: {strict_problems}")

    long_slot = deepcopy(slot)
    long_slot.update({
        "query_sentence_count": 2,
        "passage_sentence_count": 4,
        "critical_sentence_position": 3,
    })
    long_raw = _fixture_raw()
    long_raw["query"] = (
        "Ece raporu toplantıdan önce tamamlamadı. Kayıt, teslimin hâlâ eksik olduğunu belirtiyor."
    )
    long_raw["context_sentences"] = [
        "Ekip sabah erkenden ofiste toplandı.",
        "Toplantı için bütün hazırlıklar gözden geçirildi.",
        "Yönetici günün sonunda kısa bir değerlendirme yaptı.",
    ]
    long_family = normalize_family(long_raw, long_slot)
    long_problems = validate_family(long_family, long_slot, cfg)
    if long_problems:
        failures.append(f"dört cümleli fixture reddedildi: {long_problems}")
    elif any(candidate["text"] != " ".join([
        *long_raw["context_sentences"][:2],
        candidate["critical_sentence"],
        *long_raw["context_sentences"][2:],
    ]) for candidate in long_family["candidates"]):
        failures.append("kritik cümle planlanan üçüncü konuma yerleşmedi")

    assessments = []
    for candidate in family["candidates"]:
        intended = "positive" if candidate["role"] == "positive" else (
            "easy_negative" if candidate["role"] == "easy_negative" else candidate["subtype"]
        )
        assessments.append({
            "id": candidate["id"], "relevance": "relevant" if candidate["role"] == "positive" else "not_relevant",
            "naturalness": 5, "inferred_type": intended, "morphology_ok": True, "reason": "fixture",
        })
    judge = {
        "answers_query": [family["gold_id"]], "candidate_assessments": assessments,
        "length_or_style_artifact": False, "allomorph_treated_as_wrong": False,
        "family_naturalness": 5, "notes": "fixture",
    }
    judge_problems, _ = interpret_judge(family, judge, cfg)
    if judge_problems:
        failures.append(f"geçerli judge fixture reddedildi: {judge_problems}")

    allomorph_slot = deepcopy(slot)
    allomorph_slot["feature"] = FEATURE_BY_KEY["ALLO.LOC"].to_dict()
    allomorph_slot["objective"] = "allomorph_invariance"
    allomorph_slot["hard_profile"] = hard_profile(FEATURE_BY_KEY["ALLO.LOC"])
    allomorph_family = normalize_family(_fixture_raw(), allomorph_slot)
    positive = next(c for c in allomorph_family["candidates"] if c["role"] == "positive")
    positive["morph_relation"] = "allomorph_equivalent"
    negative = next(c for c in allomorph_family["candidates"] if c["role"] == "hard_negative")
    negative["morph_relation"] = "allomorph_equivalent"
    if not any("geçerli allomorph negatif" in p for p in validate_family(allomorph_family, allomorph_slot, cfg)):
        failures.append("allomorph-negatif koruması çalışmadı")
    return failures


if __name__ == "__main__":
    errors = run()
    if errors:
        for error in errors:
            print("FAIL:", error)
        raise SystemExit(1)
    print("OK: test pipeline self-test")
