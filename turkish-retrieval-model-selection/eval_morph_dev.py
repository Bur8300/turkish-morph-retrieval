#!/usr/bin/env python
"""Score encoders on a morphological split, to check the data measures what it claims to.

    conda run -n dl_hw1 python eval_morph_dev.py                      # dev split, default models
    conda run -n dl_hw1 python eval_morph_dev.py --split test         # the held-out v1.3.1 set
    conda run -n dl_hw1 python eval_morph_dev.py --models BAAI/bge-m3
    conda run -n dl_hw1 python eval_morph_dev.py --variant deascii    # diacritic-robustness slice
    conda run -n dl_hw1 python eval_morph_dev.py --pool beir          # pooled BEIR corpus, not
                                                                       # just each query's own 11

Reports two things per model, because neither alone is interpretable:

**Pairwise accuracy per negative type** (NevIR's metric). For each (gold, typed negative) pair,
is the gold ranked above the negative? Random is 50%. This is far more sensitive than nDCG over
11 candidates, where a model can be wrong about the one candidate that matters and still score
well because the five easy negatives were easy. Reporting it per type is the point: a model can
be fine on `same_feature_wrong_content` (lexical) and at chance on `morph_counterfactual`
(morphological), and that gap is the whole thesis of the project.

**A sparse char-3gram baseline**, which needs no model at all. If it matches the dense encoders,
the split is being solved lexically and is not measuring morphology — the cheap-baseline
discipline from the classical Turkish IR literature, where naive 5-character prefix truncation
was statistically indistinguishable from a real lemmatiser.

nDCG@10 over the full candidate set is also reported, reusing `evaluate_run` from
`eval_semantic_encoders.py` so the numbers are comparable with the rest of the project.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import morph_annotate as A
import morph_validators as V
from eval_semantic_encoders import encode, evaluate_run, free, load_model

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data_morph_v2"
V1_PATH = HERE.parent / "morph_eval_set_v1.3.1_review_reviewer_C_fixed.json"

DEFAULT_MODELS = [
    dict(id="newmindai/Mursit-Base-TR-Retrieval", short="Mursit-Base",
         query_prompt="", doc_prompt="", trust_remote_code=False),
    dict(id="intfloat/multilingual-e5-large", short="mE5-large",
         query_prompt="query: ", doc_prompt="passage: ", trust_remote_code=False),
]
TYPE_ORDER = ["morph_counterfactual", "partial_trap", "same_feature_wrong_content",
              "state_variant", "easy_negative"]


def load_split(split):
    if split == "test":
        return json.loads(V1_PATH.read_text(encoding="utf-8"))["items"]
    path = next((q for q in sorted(DATA_DIR.glob(f"morph_{split}_v*.json"),
                                   reverse=True)), DATA_DIR / f"morph_{split}.json")
    if not path.exists():
        raise SystemExit(f"{path} yok — önce gen_morph_dataset.py çalıştırın")
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def apply_variant(items, variant):
    """Return a transformed COPY of `items` for `--variant`; `none` returns `items` unchanged.

    `deascii`: strips diacritics and Turkish-lowercases the query and every candidate (reusing
    `morph_annotate.deascii`, the same transform used for the dataset's own `variants` field), to
    score how much a model degrades when a user types without diacritics. Well-defined to run: no
    generated item has two candidates that collapse to the same string under this transform (the
    self-test's deascii-invariant check), so every perturbed item still has a unique right answer.
    """
    if variant == "none":
        return items
    out = []
    for it in items:
        out.append({**it, "query": A.deascii(it["query"]),
                   "candidates": [{**c, "text": A.deascii(c["text"])} for c in it["candidates"]]})
    return out


def sparse_scores(item):
    q = item["query"]
    return {c["id"]: V.text_sim(q, c["text"]) for c in item["candidates"]}


def dense_scores(model, cand, items):
    """One encode pass over every query and candidate in the split; each query scored only
    against its OWN 11 candidates — the closed-set framing this project reports elsewhere."""
    q_ids = [it["query_id"] for it in items]
    q_emb = encode(model, [it["query"] for it in items], cand["query_prompt"])
    flat = [(it["query_id"], c["id"], c["text"]) for it in items for c in it["candidates"]]
    d_emb = encode(model, [t for _, _, t in flat], cand["doc_prompt"])

    q_index = {qid: i for i, qid in enumerate(q_ids)}
    out = defaultdict(dict)
    for (qid, cid, _), vec in zip(flat, d_emb):
        out[qid][cid] = float(np.dot(q_emb[q_index[qid]], vec))
    return out


def pool_corpus(items):
    """Every candidate across every item, flattened — the pooled BEIR corpus `morph_beir.py`
    exports, and a genuinely harder task than the closed-set default: a foreign query's candidate
    can now outrank this query's own gold (measured on dev: 12/90 queries, see morph_beir.py)."""
    return [(c["id"], c["text"]) for it in items for c in it["candidates"]]


def sparse_scores_pooled(item, corpus):
    q = item["query"]
    return {cid: V.text_sim(q, text) for cid, text in corpus}


def dense_scores_pooled(model, cand, items, corpus):
    """Full query x corpus similarity matrix — every query scored against EVERY candidate in the
    split, not just its own. One matmul; trivial at these dataset sizes (dev: 90 x 990)."""
    q_ids = [it["query_id"] for it in items]
    q_emb = encode(model, [it["query"] for it in items], cand["query_prompt"])
    doc_ids = [cid for cid, _ in corpus]
    d_emb = encode(model, [t for _, t in corpus], cand["doc_prompt"])
    sims = q_emb @ d_emb.T
    return {qid: dict(zip(doc_ids, (float(x) for x in sims[i]))) for i, qid in enumerate(q_ids)}


def score_split(items, scores_by_query):
    """Pairwise accuracy per negative type + nDCG@10 over the 11 candidates."""
    pair_hits, pair_tot = defaultdict(int), defaultdict(int)
    qrels, run = {}, {}
    for it in items:
        s = scores_by_query[it["query_id"]]
        gold = it["gold_id"]
        for c in it["candidates"]:
            if c["id"] == gold:
                continue
            key = c.get("subtype") or c["role"]
            pair_tot[key] += 1
            pair_hits[key] += int(s[gold] > s[c["id"]])
        qrels[it["query_id"]] = {gold: 1.0}
        run[it["query_id"]] = [cid for cid, _ in
                               sorted(s.items(), key=lambda kv: -kv[1])]
    res = {t: round(100 * pair_hits[t] / pair_tot[t], 1) for t in pair_tot}
    res["ALL_pairwise"] = round(100 * sum(pair_hits.values()) / sum(pair_tot.values()), 1)
    res.update(evaluate_run(qrels, run))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "train", "test"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--variant", default="none", choices=["none", "deascii"],
                    help="'deascii' scores the diacritic-stripped robustness slice")
    ap.add_argument("--pool", default="closed", choices=["closed", "beir"],
                    help="'closed' (default): each query vs its own 11 candidates, matching the "
                         "rest of this project. 'beir': every query vs the full pooled corpus, "
                         "a harder and DIFFERENT task — see morph_beir.py's README.")
    args = ap.parse_args()

    items = apply_variant(load_split(args.split), args.variant)
    print(f"{args.split} (variant={args.variant}, pool={args.pool}): {len(items)} sorgu, "
          f"{sum(len(i['candidates']) for i in items)} aday\n")

    cands = DEFAULT_MODELS if not args.models else [
        dict(id=m, short=m.split("/")[-1], query_prompt="", doc_prompt="",
             trust_remote_code=True) for m in args.models]

    if args.pool == "closed":
        rows = {"sparse-char3gram": score_split(
            items, {it["query_id"]: sparse_scores(it) for it in items})}
    else:
        corpus = pool_corpus(items)
        rows = {"sparse-char3gram": score_split(
            items, {it["query_id"]: sparse_scores_pooled(it, corpus) for it in items})}
    for cand in cands:
        try:
            model = load_model(cand)
        except Exception as e:
            print(f"[atlandı] {cand['id']}: {type(e).__name__}: {str(e)[:120]}")
            continue
        try:
            if args.pool == "closed":
                rows[cand["short"]] = score_split(items, dense_scores(model, cand, items))
            else:
                rows[cand["short"]] = score_split(
                    items, dense_scores_pooled(model, cand, items, corpus))
        finally:
            free(model)

    cols = [t for t in TYPE_ORDER if any(t in r for r in rows.values())]
    head = ["model"] + cols + ["ALL_pairwise", "nDCG@10"]
    print("\n=== ikili doğruluk (%), rastgele = 50 ===")
    print(" | ".join(f"{h[:22]:>22s}" for h in head))
    for name, r in rows.items():
        print(" | ".join([f"{name[:22]:>22s}"] +
                         [f"{r.get(c, float('nan')):>22.1f}" for c in cols] +
                         [f"{r['ALL_pairwise']:>22.1f}", f"{r['nDCG@10']:>22.1f}"]))
    print("\nOkuma notu: `morph_counterfactual` sütunu asıl ölçümdür. Seyrek temel çizgi burada "
          "50'ye yakın olmalı — değilse öğeler biçimbilimi değil sözcük örtüşmesini ölçüyordur.")


if __name__ == "__main__":
    main()
