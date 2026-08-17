"""Optional Stanza-based Turkish lemma/UFeats audit; diagnostic, never a silent gold gate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evaluation import load_items
from .validators import tr_lower


EXPECTED_UD_FEATURES = {
    "ACC": ("Case=Acc",), "DAT": ("Case=Dat",), "LOC": ("Case=Loc",),
    "ABL": ("Case=Abl",), "INS": ("Case=Ins",), "GEN": ("Case=Gen",),
    "EQU": ("Case=Equ",), "PL": ("Number=Plur",),
    "PST": ("Tense=Past",), "FUT": ("Tense=Fut",),
    "PRS.PROG": ("Aspect=Prog",), "PRF.EVID": ("Evident=Nfh",),
    "PASS": ("Voice=Pass",), "CAUS": ("Voice=Cau",),
    "V.AGR": ("Person=", "Number="), "NEG": ("Polarity=Neg",),
    "NEG.AOR": ("Polarity=Neg",),
}

EXPECTED_POSSESSIVE_FEATURES = {
    "POSS.1SG": ("Person[psor]=1", "Number[psor]=Sing"),
    "POSS.2SG": ("Person[psor]=2", "Number[psor]=Sing"),
    "POSS.3SG": ("Person[psor]=3", "Number[psor]=Sing"),
    "POSS.1PL": ("Person[psor]=1", "Number[psor]=Plur"),
    "POSS.2PL": ("Person[psor]=2", "Number[psor]=Plur"),
    "POSS.3PL": ("Person[psor]=3", "Number[psor]=Plur"),
}


def _lemma_key(value: str) -> str:
    value = tr_lower(value).strip(".,;:!?()[]{}\"'")
    if len(value) > 4 and value.endswith(("mak", "mek")):
        value = value[:-3]
    return value


def _token_record(word) -> dict[str, Any]:
    return {
        "text": word.text,
        "lemma": word.lemma,
        "upos": word.upos,
        "ufeats": word.feats or "",
    }


def _find_word(document, critical_word: str) -> dict[str, Any] | None:
    wanted = tr_lower(critical_word).strip(".,;:!?()[]{}\"'")
    words = [word for sentence in document.sentences for word in sentence.words]
    exact = [word for word in words if tr_lower(word.text).strip(".,;:!?()[]{}\"'") == wanted]
    if exact:
        return _token_record(exact[0])
    contained = [
        word for word in words
        if tr_lower(word.text) in wanted or wanted in tr_lower(word.text)
    ]
    return _token_record(contained[0]) if contained else None


def _expected_feature_check(target_feature: str, token: dict[str, Any] | None) -> bool | None:
    if token is None:
        return False
    normalized_feature = target_feature.removeprefix("ALLO.")
    expected = EXPECTED_POSSESSIVE_FEATURES.get(target_feature)
    if expected is None:
        expected = EXPECTED_UD_FEATURES.get(normalized_feature)
    if expected is None:
        return None
    ufeats = str(token.get("ufeats", ""))
    return all(fragment in ufeats for fragment in expected)


def run_morphology_audit(
    input_path: str | Path, output_path: str | Path, download_model: bool = False,
    use_gpu: bool = False,
) -> dict[str, Any]:
    try:
        import stanza
    except ImportError as exc:
        raise RuntimeError("Morph audit için `pip install stanza` çalıştırın") from exc
    if download_model:
        stanza.download("tr", processors="tokenize,mwt,pos,lemma", verbose=False)
    try:
        nlp = stanza.Pipeline(
            "tr", processors="tokenize,mwt,pos,lemma", use_gpu=use_gpu, verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Türkçe Stanza modeli bulunamadı. Bir kez `python -m test morph-audit ... "
            "--download-model` çalıştırın."
        ) from exc

    source = Path(input_path)
    items = (
        [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if source.suffix == ".jsonl" else load_items(source)
    )
    family_rows = []
    counts = Counter()
    for item in items:
        query_doc = nlp(item["query"])
        query_token = _find_word(query_doc, item.get("critical_word_query", ""))
        query_feature_ok = _expected_feature_check(
            str(item.get("target_feature", "")), query_token
        )
        candidate_tokens = {}
        for candidate in item["candidates"]:
            document = nlp(candidate.get("critical_sentence", candidate["text"]))
            candidate_tokens[candidate["id"]] = _find_word(document, candidate.get("critical_word", ""))
        positive = next(row for row in item["candidates"] if row["id"] == item["gold_id"])
        minimal = next(
            (row for row in item["candidates"] if row.get("subtype") == "minimal_morph_negative"),
            None,
        )
        positive_token = candidate_tokens.get(positive["id"])
        minimal_token = candidate_tokens.get(minimal["id"]) if minimal else None
        expected_lemma = _lemma_key(str(item.get("critical_lemma", "")))
        lemma_ok = bool(
            positive_token and _lemma_key(str(positive_token.get("lemma", ""))) == expected_lemma
        )
        ud_feature_ok = _expected_feature_check(str(item.get("target_feature", "")), positive_token)
        strict_pair_ok = None
        if item.get("strict_minimal_pair"):
            strict_pair_ok = bool(
                positive_token and minimal_token
                and _lemma_key(str(positive_token.get("lemma", "")))
                == _lemma_key(str(minimal_token.get("lemma", "")))
                and positive_token.get("ufeats") != minimal_token.get("ufeats")
            )
        explicit_query_ok = (
            query_feature_ok is not False
            if item.get("query_expression") == "morph_explicit"
            else True
        )
        status = "pass" if (
            lemma_ok and strict_pair_ok is not False
            and ud_feature_ok is not False and explicit_query_ok
        ) else "warning"
        counts[status] += 1
        family_rows.append({
            "family_id": item["family_id"],
            "target_feature": item.get("target_feature"),
            "strict_minimal_pair": bool(item.get("strict_minimal_pair")),
            "status": status,
            "critical_lemma_expected": item.get("critical_lemma"),
            "query_token": query_token,
            "positive_token": positive_token,
            "minimal_negative_token": minimal_token,
            "lemma_match": lemma_ok,
            "query_expression": item.get("query_expression"),
            "query_expected_ud_feature_present": query_feature_ok,
            "expected_ud_feature_present": ud_feature_ok,
            "strict_same_lemma_different_ufeats": strict_pair_ok,
            "candidate_tokens": candidate_tokens,
        })
    report = {
        "backend": "stanza:tokenize,mwt,pos,lemma",
        "note": (
            "Bu audit lemma ve UD morphological features (UFeats) kontrolüdür; gerçek morfem "
            "segmentasyonu veya morfem sayısı olduğunu iddia etmez. Uyarılar audit raporunda kalır."
        ),
        "counts": dict(counts),
        "families": family_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output.resolve()), "counts": dict(counts), "families": len(family_rows)}
