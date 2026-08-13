# train — BEIR export

Standard BEIR layout: `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` (binary relevance).
Loadable by `beir.datasets.data_loader.GenericDataLoader` or this repo's own
`eval_semantic_encoders.load_beir_dataset`-style reader.

**510 queries, 5610 pooled candidates.**

## Two ways to score this, and they are not the same task

1. **Closed-set (what this project reports elsewhere)**: for query Q, rank only the 11 ids in
   `candidate_pool.json[Q]`. This is what `eval_morph_dev.py` does and what every nDCG@10 /
   pairwise-accuracy number in this repo's READMEs refers to.
2. **Pooled BEIR (the standard task this export also supports)**: rank the FULL `corpus.jsonl`
   (5610 docs) against every query. Harder and not equivalent: pooling candidates
   from every query into one corpus means a query can legitimately be answered by another
   query's candidate. Measured here with a model-free sparse baseline (`morph_validators.text_sim`,
   a lower bound — any real retriever with topical awareness will not do better on this axis):
   **144/510 queries** have at least one foreign candidate that
   out-scores their own gold.

Report which of the two you are using. `hard_negatives.jsonl` carries the typed structure
(`morph_counterfactual`, `same_feature_wrong_content`, `partial_trap`, `state_variant`) that BEIR's
binary qrels cannot express — the actual contribution of this dataset over a generic BEIR set.
