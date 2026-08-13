#!/usr/bin/env python
"""Export a split to standard BEIR format, plus two sidecars the format can't express.

    conda run -n dl_hw1 python morph_beir.py --split train --split dev --split test

Writes, per split, exactly the layout `beir.datasets.data_loader.GenericDataLoader` (and this
repo's own `eval_semantic_encoders.load_beir_dataset`) expects:

    data_morph_v2/beir/<split>/corpus.jsonl    {"_id", "title": "", "text"}
    data_morph_v2/beir/<split>/queries.jsonl   {"_id", "text"}
    data_morph_v2/beir/<split>/qrels/test.tsv  query-id \\t corpus-id \\t score   (binary: 1)

Binary qrels only: a typed hard negative is a WRONG answer to its query, so scoring it above an
easy negative in a graded qrel would misrepresent the task. `role`/`subtype` typing — the actual
point of this dataset — has no BEIR field to live in, so it ships as a sidecar instead of being
discarded:

    data_morph_v2/beir/<split>/candidate_pool.json   {query_id: [11 candidate ids]}
    data_morph_v2/beir/<split>/hard_negatives.jsonl  {query_id, gold_id, negatives:[{id,role,subtype}]}

candidate_pool.json is what keeps the CLOSED-SET evaluation this dataset was designed for
reproducible: `corpus.jsonl` pools every candidate from every query in the split into one corpus
(that is what "BEIR format" means), and pooling is not free — measured on dev, 11 of 90 queries
have a candidate from a DIFFERENT query outranking their own gold under plain sparse similarity
(`morph_validators.text_sim`). Restricting retrieval to `candidate_pool[query_id]` reproduces the
number the rest of this project reports; scoring against the full pooled corpus reproduces the
standard BEIR number, which is a different, harder task and should be reported as such, not
conflated with the first. See `beir/<split>/README.md` (written per export) for the measured gap.
"""
import argparse
import json
from pathlib import Path

import morph_validators as V

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "data_morph_v2" / "beir"


def pooling_contamination(items):
    """How often a FOREIGN query's candidate outranks this query's own gold in the pooled corpus.

    Cheap sparse-similarity check, not a claim about any particular dense model — it lower-bounds
    the contamination a real retriever would also hit, since a dense encoder that is at least as
    topically aware as character trigrams will not do better on this axis.
    """
    corpus = [(c["id"], c["text"]) for it in items for c in it["candidates"]]
    contaminated = []
    for it in items:
        own = {c["id"] for c in it["candidates"]}
        gold = it["gold_id"]
        gold_text = next(c["text"] for c in it["candidates"] if c["id"] == gold)
        gold_score = V.text_sim(it["query"], gold_text)
        beaten_by = sum(1 for cid, text in corpus
                        if cid not in own and V.text_sim(it["query"], text) > gold_score)
        if beaten_by:
            contaminated.append((it["query_id"], beaten_by))
    return contaminated


def export_split(items, split, out_root=OUT_ROOT):
    out_dir = out_root / split
    (out_dir / "qrels").mkdir(parents=True, exist_ok=True)

    seen_docs = {}
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            for c in it["candidates"]:
                if c["id"] in seen_docs:
                    continue                      # candidate ids are globally unique already,
                seen_docs[c["id"]] = True          # but guard the invariant rather than assume it
                fh.write(json.dumps({"_id": c["id"], "title": "", "text": c["text"]},
                                    ensure_ascii=False) + "\n")

    with (out_dir / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({"_id": it["query_id"], "text": it["query"]},
                                ensure_ascii=False) + "\n")

    with (out_dir / "qrels" / "test.tsv").open("w", encoding="utf-8") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        for it in items:
            fh.write(f"{it['query_id']}\t{it['gold_id']}\t1\n")

    pool = {it["query_id"]: [c["id"] for c in it["candidates"]] for it in items}
    (out_dir / "candidate_pool.json").write_text(
        json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")

    with (out_dir / "hard_negatives.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            negs = [{"id": c["id"], "role": c["role"], "subtype": c.get("subtype")}
                    for c in it["candidates"] if c["id"] != it["gold_id"]]
            fh.write(json.dumps({"query_id": it["query_id"], "gold_id": it["gold_id"],
                                 "negatives": negs}, ensure_ascii=False) + "\n")

    contaminated = pooling_contamination(items)
    readme = f"""\
# {split} — BEIR export

Standard BEIR layout: `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` (binary relevance).
Loadable by `beir.datasets.data_loader.GenericDataLoader` or this repo's own
`eval_semantic_encoders.load_beir_dataset`-style reader.

**{len(items)} queries, {len(seen_docs)} pooled candidates.**

## Two ways to score this, and they are not the same task

1. **Closed-set (what this project reports elsewhere)**: for query Q, rank only the 11 ids in
   `candidate_pool.json[Q]`. This is what `eval_morph_dev.py` does and what every nDCG@10 /
   pairwise-accuracy number in this repo's READMEs refers to.
2. **Pooled BEIR (the standard task this export also supports)**: rank the FULL `corpus.jsonl`
   ({len(seen_docs)} docs) against every query. Harder and not equivalent: pooling candidates
   from every query into one corpus means a query can legitimately be answered by another
   query's candidate. Measured here with a model-free sparse baseline (`morph_validators.text_sim`,
   a lower bound — any real retriever with topical awareness will not do better on this axis):
   **{len(contaminated)}/{len(items)} queries** have at least one foreign candidate that
   out-scores their own gold.

Report which of the two you are using. `hard_negatives.jsonl` carries the typed structure
(`morph_counterfactual`, `same_feature_wrong_content`, `partial_trap`, `state_variant`) that BEIR's
binary qrels cannot express — the actual contribution of this dataset over a generic BEIR set.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return {"n_queries": len(items), "n_corpus": len(seen_docs),
           "pooling_contaminated": len(contaminated), "out_dir": str(out_dir)}


def main():
    # Deferred: eval_morph_dev imports eval_semantic_encoders, which imports torch and
    # sentence_transformers at module level. export_split() itself needs none of that (it takes
    # already-loaded items); only the CLI, which actually loads a split from disk, does — so
    # `import morph_beir` alone (as gen_morph_dataset.py and morph_selftest.py both do) stays
    # dependency-light, and the heavy chain only loads when this CLI actually runs.
    from eval_morph_dev import load_split

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", default=None,
                    choices=["train", "dev", "test"], help="tekrarlanabilir")
    args = ap.parse_args()
    for split in (args.split or ["train", "dev", "test"]):
        items = load_split(split)
        stats = export_split(items, split)
        print(f"{split}: {stats['n_queries']} sorgu, {stats['n_corpus']} aday -> "
              f"{stats['out_dir']}  (havuzlama kirliliği: {stats['pooling_contaminated']} sorgu)")


if __name__ == "__main__":
    main()
