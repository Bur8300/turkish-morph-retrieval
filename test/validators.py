"""Deterministic family/corpus gates and blind-judge interpretation."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+", re.I)
_META_PATTERNS = ("kaydı bul", "kaydi bul", "arıyorum", "ariyorum", "hangi kayıt", "hangi kayit")
_ABBREVIATIONS = ("Dr.", "Prof.", "Doç.", "Sn.", "vb.", "vs.", "örn.", "T.C.")


def tr_lower(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(tr_lower(text))


def sentence_count(text: str) -> int:
    clean = text.strip()
    for abbreviation in _ABBREVIATIONS:
        clean = clean.replace(abbreviation, abbreviation.replace(".", "∯"))
    return len(re.findall(r"[.!?]+(?:[\"'”’)]*)?(?=\s|$)", clean))


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = " ".join(tokens(text))
    return {normalized[i : i + n] for i in range(max(0, len(normalized) - n + 1))}


def jaccard(left: set, right: set) -> float:
    return len(left & right) / max(1, len(left | right))


def _contains_word(text: str, word: str) -> bool:
    return tr_lower(word).strip(".,;:!?()[]{}\"'") in tr_lower(text)


def normalize_family(raw: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    """Assemble controlled passages, then inject trusted metadata and neutral IDs."""
    if not isinstance(raw, dict):
        raise ValueError("Generator çıktısı object olmalı")
    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates listesi yok")
    context_sentences = raw.get("context_sentences", [])
    if not isinstance(context_sentences, list):
        raise ValueError("context_sentences liste olmalı")
    candidates = [dict(candidate) for candidate in raw_candidates]
    insert_at = int(slot["critical_sentence_position"]) - 1
    for candidate in candidates:
        sentences = [str(sentence).strip() for sentence in context_sentences]
        sentences.insert(insert_at, str(candidate.get("critical_sentence", "")).strip())
        candidate["text"] = " ".join(sentence for sentence in sentences if sentence)
    shuffle_seed = int(hashlib.sha256(slot["slot_id"].encode()).hexdigest()[:16], 16)
    random.Random(shuffle_seed).shuffle(candidates)
    family_id = f"family_{slot['slot_id']}"
    for index, candidate in enumerate(candidates, start=1):
        candidate["id"] = f"{family_id}_c{index:02d}"

    positives = [candidate for candidate in candidates if candidate.get("role") == "positive"]
    gold_id = positives[0]["id"] if len(positives) == 1 else None
    return {
        "family_id": family_id,
        "slot_id": slot["slot_id"],
        "target_split": slot["target_split"],
        "generalization_bucket": slot["generalization_bucket"],
        "generalization_tags": list(slot["generalization_tags"]),
        "objective": slot["objective"],
        "macro_phenomenon": slot["macro_phenomenon"],
        "target_feature": slot["feature"]["key"],
        "target_feature_label": slot["feature"]["label"],
        "surface_forms": list(slot["feature"]["surface_forms"]),
        "layer": slot["layer"],
        "domain": slot["domain"],
        "register": slot["register"],
        "query_sentence_count": slot["query_sentence_count"],
        "passage_sentence_count": slot["passage_sentence_count"],
        "critical_sentence_position": slot["critical_sentence_position"],
        "hard_profile": list(slot["hard_profile"]),
        "semantic_frame_id": raw.get("semantic_frame_id"),
        "template_id": raw.get("template_id"),
        "critical_lemma": raw.get("critical_lemma"),
        "critical_word_query": raw.get("critical_word_query"),
        "critical_word_positive": raw.get("critical_word_positive"),
        "feature_delta": raw.get("feature_delta"),
        "query": raw.get("query"),
        "context_sentences": context_sentences,
        "candidates": candidates,
        "gold_id": gold_id,
        "source_type": "llm_generated_pending_human_review",
        "generation_notes": raw.get("generation_notes", ""),
    }


def validate_family(family: dict[str, Any], slot: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    query = family.get("query")
    candidates = family.get("candidates")
    if not isinstance(query, str) or not query.strip():
        problems.append("query boş veya string değil")
        return problems
    if not isinstance(candidates, list):
        problems.append("candidates listesi yok")
        return problems
    if len(candidates) != 11:
        problems.append(f"11 aday gerekli; bulunan {len(candidates)}")

    roles = Counter(candidate.get("role") for candidate in candidates)
    expected_roles = cfg["candidate_counts"]
    if dict(roles) != expected_roles:
        problems.append(f"rol sayıları yanlış: {dict(roles)} != {expected_roles}")

    hard_expected = {
        item["slot"]: item["subtype"] for item in slot["hard_profile"]
    }
    hard_actual = {
        candidate.get("candidate_slot"): candidate.get("subtype")
        for candidate in candidates if candidate.get("role") == "hard_negative"
    }
    if len(hard_actual) != 8 or hard_actual != hard_expected:
        problems.append(f"uyarlanabilir sekiz hard slot yanlış: {hard_actual} != {hard_expected}")
    positives = [candidate for candidate in candidates if candidate.get("role") == "positive"]
    if len(positives) == 1 and positives[0].get("subtype") != "equivalence_positive":
        problems.append("positive subtype equivalence_positive olmalı")
    if len(positives) == 1 and positives[0].get("candidate_slot") != "positive_01":
        problems.append("positive candidate_slot positive_01 olmalı")
    easies = [candidate for candidate in candidates if candidate.get("role") == "easy_negative"]
    if any(candidate.get("subtype") != "easy_negative" for candidate in easies):
        problems.append("iki easy adayın subtype'ı easy_negative olmalı")
    if {candidate.get("candidate_slot") for candidate in easies} != {"easy_01", "easy_02"}:
        problems.append("easy candidate_slot değerleri easy_01/easy_02 olmalı")
    expected_slots = {"positive_01", "easy_01", "easy_02", *hard_expected}
    actual_slots = [candidate.get("candidate_slot") for candidate in candidates]
    if len(set(actual_slots)) != 11 or set(actual_slots) != expected_slots:
        problems.append(f"candidate_slot kapsamı yanlış: {actual_slots}")

    ids = [candidate.get("id") for candidate in candidates]
    if len(ids) != len(set(ids)) or any(not candidate_id for candidate_id in ids):
        problems.append("candidate id'leri eksik veya benzersiz değil")
    texts = [candidate.get("text") for candidate in candidates]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        problems.append("boş/non-string candidate text var")
    elif len({tr_lower(text).strip() for text in texts}) != len(texts):
        problems.append("candidate metinleri arasında birebir tekrar var")

    wanted_query_sentences = int(family["query_sentence_count"])
    wanted_passage_sentences = int(family["passage_sentence_count"])
    if sentence_count(query) != wanted_query_sentences:
        problems.append(
            f"query {wanted_query_sentences} yerine {sentence_count(query)} cümle"
        )
    context_sentences = family.get("context_sentences")
    if not isinstance(context_sentences, list):
        problems.append("context_sentences liste değil")
        context_sentences = []
    if len(context_sentences) != wanted_passage_sentences - 1:
        problems.append(
            f"ortak bağlam {wanted_passage_sentences - 1} yerine {len(context_sentences)} cümle"
        )
    for index, context in enumerate(context_sentences, start=1):
        if not isinstance(context, str) or sentence_count(context) != 1:
            problems.append(f"context_sentences[{index}] tek tam cümle değil")
    for candidate in candidates:
        text = candidate.get("text", "")
        critical_sentence = candidate.get("critical_sentence", "")
        if not isinstance(critical_sentence, str) or sentence_count(critical_sentence) != 1:
            problems.append(f"{candidate.get('id')} critical_sentence tek tam cümle değil")
        if isinstance(text, str) and sentence_count(text) != wanted_passage_sentences:
            problems.append(
                f"{candidate.get('id')} {wanted_passage_sentences} yerine "
                f"{sentence_count(text)} cümle"
            )

    if query.rstrip().endswith("?"):
        problems.append("query soru cümlesi olamaz")
    lowered_query = tr_lower(query)
    if any(pattern in lowered_query for pattern in _META_PATTERNS):
        problems.append("query meta-arama dili içeriyor")

    if family.get("semantic_frame_id") != slot["semantic_frame_id"]:
        problems.append("semantic_frame_id SLOT ile uyuşmuyor")
    if family.get("template_id") != slot["template"]["id"]:
        problems.append("template_id SLOT ile uyuşmuyor")
    for name in ("critical_lemma", "critical_word_query", "critical_word_positive", "feature_delta"):
        if not isinstance(family.get(name), str) or not family[name].strip():
            problems.append(f"{name} eksik")
    if isinstance(family.get("critical_word_query"), str) and not _contains_word(query, family["critical_word_query"]):
        problems.append("critical_word_query query içinde yok")
    if positives and isinstance(family.get("critical_word_positive"), str):
        if not _contains_word(positives[0].get("text", ""), family["critical_word_positive"]):
            problems.append("critical_word_positive positive metninde yok")

    for candidate in candidates:
        critical = candidate.get("critical_word")
        if not isinstance(critical, str) or not critical.strip():
            problems.append(f"{candidate.get('id')} critical_word eksik")
        elif not _contains_word(candidate.get("critical_sentence", ""), critical):
            problems.append(f"{candidate.get('id')} critical_word kritik cümlede yok")

    if family["objective"] == "allomorph_invariance":
        if positives and positives[0].get("morph_relation") != "allomorph_equivalent":
            problems.append("allomorph family positive ilişkisi allomorph_equivalent olmalı")
        bad = [candidate.get("id") for candidate in candidates if candidate.get("role") != "positive" and candidate.get("morph_relation") == "allomorph_equivalent"]
        if bad:
            problems.append(f"geçerli allomorph negatif yapılmış: {bad}")
    elif positives and positives[0].get("morph_relation") != "target_preserved":
        problems.append("semantic/composition positive ilişkisi target_preserved olmalı")

    lengths = [len(tokens(text)) for text in texts if isinstance(text, str)]
    if len(lengths) == 11 and min(lengths) > 0:
        ratio = max(lengths) / min(lengths)
        if ratio > float(cfg["quality"]["candidate_token_ratio_max"]):
            problems.append(f"candidate token uzunluk oranı yüksek: {ratio:.2f}")
        if positives:
            gold_len = len(tokens(positives[0]["text"]))
            median = sorted(lengths)[len(lengths) // 2]
            if gold_len / max(1, median) > float(cfg["quality"]["gold_to_median_token_ratio_max"]):
                problems.append(f"gold uzunluk bias'ı: gold/median={gold_len / max(1, median):.2f}")

    return sorted(set(problems))


def interpret_judge(
    family: dict[str, Any], verdict: dict[str, Any], cfg: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    candidates = {candidate["id"]: candidate for candidate in family["candidates"]}
    answers = verdict.get("answers_query", [])
    if not isinstance(answers, list) or set(answers) != {family["gold_id"]}:
        problems.append(f"judge tek gold üzerinde anlaşmadı: {answers}")
    assessments = verdict.get("candidate_assessments", [])
    if not isinstance(assessments, list) or {row.get("id") for row in assessments} != set(candidates):
        problems.append("judge assessment candidate id kapsamı eksik/fazla")
        assessments = []

    agreement = 0
    naturalness_values: list[int] = []
    morphology_failures = []
    for row in assessments:
        candidate = candidates[row["id"]]
        intended = "positive" if candidate["role"] == "positive" else (
            "easy_negative" if candidate["role"] == "easy_negative" else candidate["subtype"]
        )
        agreement += int(row.get("inferred_type") == intended)
        if isinstance(row.get("naturalness"), int):
            naturalness_values.append(row["naturalness"])
        if not row.get("morphology_ok", False):
            morphology_failures.append(row["id"])
    agreement_rate = agreement / max(1, len(candidates))
    if agreement_rate < float(cfg["quality"]["judge_subtype_agreement_min"]):
        problems.append(f"judge subtype uyumu düşük: {agreement_rate:.3f}")
    if morphology_failures:
        problems.append(f"judge bozuk biçimbilim işaretledi: {morphology_failures}")
    if verdict.get("length_or_style_artifact"):
        problems.append("judge uzunluk/üslup artefaktı buldu")
    if verdict.get("allomorph_treated_as_wrong"):
        problems.append("judge geçerli allomorphun yanlış kullanıldığını buldu")
    family_naturalness = verdict.get("family_naturalness", 0)
    if not isinstance(family_naturalness, int) or family_naturalness < int(cfg["quality"]["judge_naturalness_min"]):
        problems.append(f"judge family naturalness düşük: {family_naturalness}")

    metadata = {
        "answers_query": answers,
        "subtype_agreement": round(agreement_rate, 4),
        "family_naturalness": family_naturalness,
        "candidate_naturalness_min": min(naturalness_values, default=0),
        "length_or_style_artifact": bool(verdict.get("length_or_style_artifact")),
        "allomorph_treated_as_wrong": bool(verdict.get("allomorph_treated_as_wrong")),
        "notes": verdict.get("notes", ""),
    }
    return sorted(set(problems)), metadata


def quality_score(family: dict[str, Any]) -> float:
    judge = family.get("qc", {}).get("judge", {})
    lengths = [len(tokens(candidate["text"])) for candidate in family["candidates"]]
    balance = min(lengths) / max(lengths) if lengths and max(lengths) else 0.0
    return (
        float(judge.get("family_naturalness", 0))
        + 2.0 * float(judge.get("subtype_agreement", 0))
        + balance
    )


def corpus_problems(items: list[dict[str, Any]], cfg: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    ids = [item["family_id"] for item in items]
    if len(ids) != len(set(ids)):
        problems.append("family_id tekrarı var")
    query_grams = [(item["family_id"], char_ngrams(item["query"])) for item in items]
    threshold = float(cfg["quality"]["near_duplicate_jaccard_max"])
    for left in range(len(query_grams)):
        for right in range(left + 1, len(query_grams)):
            score = jaccard(query_grams[left][1], query_grams[right][1])
            if score > threshold:
                problems.append(
                    f"yakın query kopyası: {query_grams[left][0]} / {query_grams[right][0]} ({score:.3f})"
                )
                if len(problems) >= 100:
                    return problems
    return problems


def _bm25_scores(query: str, docs: list[str]) -> list[float]:
    doc_tokens = [tokens(doc) for doc in docs]
    query_tokens = tokens(query)
    n_docs = len(docs)
    avgdl = sum(map(len, doc_tokens)) / max(1, n_docs)
    df = Counter(token for token in set(query_tokens) for doc in doc_tokens if token in set(doc))
    scores = []
    for doc in doc_tokens:
        tf = Counter(doc)
        score = 0.0
        for term in query_tokens:
            freq = tf[term]
            if not freq:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * len(doc) / max(1, avgdl)))
        scores.append(score)
    return scores


def artifact_report(items: list[dict[str, Any]]) -> dict[str, Any]:
    wins = Counter()
    gold_positions = Counter()
    for item in items:
        candidates = item["candidates"]
        gold_index = next(index for index, candidate in enumerate(candidates) if candidate["id"] == item["gold_id"])
        gold_positions[gold_index + 1] += 1
        lengths = [len(candidate["text"]) for candidate in candidates]
        token_lengths = [len(tokens(candidate["text"])) for candidate in candidates]
        query_words = set(tokens(item["query"]))
        overlap = [len(query_words & set(tokens(candidate["text"]))) / max(1, len(query_words)) for candidate in candidates]
        query_grams = char_ngrams(item["query"])
        trigram = [jaccard(query_grams, char_ngrams(candidate["text"])) for candidate in candidates]
        bm25 = _bm25_scores(item["query"], [candidate["text"] for candidate in candidates])
        wins["longest_candidate"] += int(gold_index == max(range(11), key=lengths.__getitem__))
        wins["most_tokens"] += int(gold_index == max(range(11), key=token_lengths.__getitem__))
        wins["word_overlap"] += int(gold_index == max(range(11), key=overlap.__getitem__))
        wins["character_3gram"] += int(gold_index == max(range(11), key=trigram.__getitem__))
        wins["bm25"] += int(gold_index == max(range(11), key=bm25.__getitem__))
    n = max(1, len(items))
    return {
        "n_families": len(items),
        "chance_recall_at_1": round(1 / 11, 4),
        "recall_at_1": {name: round(value / n, 4) for name, value in sorted(wins.items())},
        "gold_position_counts": dict(sorted(gold_positions.items())),
        "longest_gold_rate": round(wins["longest_candidate"] / n, 4),
    }
