# Legacy train/dev and model-selection system

> Bu dizin geçmiş v2 train/dev üretimini ve model-selection provenance'ını korur. Yeni
> 100-development + 500-sealed test benchmark'ı burada üretilmez; güncel test sistemi
> [`../test/`](../test/) altındadır. Aşağıdaki “test” ifadeleri yalnız eski 50-family v1.3.1
> referans setini anlatır.

Benchmark-backed model comparison for the **Morphology-Aware Contrastive Fine-Tuning for
Turkish Retrieval** project (inzva AI Projects #10). One notebook, three decisions:

1. **Semantic encoder** base model (LoRA fine-tuning target)
2. **Baselines** for the report
3. **Morphological-channel encoder** (lightweight, for the dual-encoder ablation)

## Contents

- `model_selection_colab.ipynb` — the comparison notebook (Sections: A published scores ·
  B our Turkish-BEIR retrieval runs · C morphological suffix probe · D efficiency ·
  E weighted selection matrix)
- `docs/literature_review.md` — how prior work has handled morphology in representation learning
  and IR for agglutinative languages, and how meaning-flipping minimal contrasts have been
  attacked elsewhere. 81 sources, each tagged **[F]** full text read / **[A]** abstract only /
  **[S]** search snippet — nothing is presented at higher confidence than its tag.
  Raw survey output kept alongside as `docs/literature_survey_raw.json`.
- **v2.0 dataset generation** (see below) — `morph_taxonomy.py`, `morph_prompts.py`,
  `morph_validators.py`, `morph_selftest.py`, `gen_morph_dataset.py`, output in `data_morph_v2/`.
- **v2.3 additions** (see below) — `morph_annotate.py` (morphological annotation + variant
  groups, zero API calls) and `morph_beir.py` (standard BEIR export).

## v2.0 morphological train/dev set

`legacy_test_data/morph_eval_set_v1.3.1_review_reviewer_C_fixed.json` (50 queries, hand-QC'd) stays the
**held-out test set**. `gen_morph_dataset.py` synthesises a much larger **train + dev** set with
`gemini-3.5-flash-lite` and never writes to the test set.

```bash
# validators only, no API calls — run this first
conda run -n dl_hw1 python gen_morph_dataset.py --self-test

# smoke run (~30 calls), then the full run (resumable)
conda run -n dl_hw1 python gen_morph_dataset.py --target 16 --overproduce 1.0
conda run -n dl_hw1 python gen_morph_dataset.py --target 600
```

Pipeline: **generate** (1 call/slot) → **zero-API gates** → **blind judge** (1 call/item) →
**repair** → **corpus audit** → **train/dev split**. Every response is cached under
`data_morph_v2/_cache/` keyed by slot *and prompt hash*, so a resumed run continues where it
stopped and a prompt edit correctly invalidates stale responses. Per-key request budget
(15 RPM / 500 RPD, pooled across however many `API_KEY_N` entries are in `.env` — the v2.2 run
used 5, for a 2500/day pool) is tracked in `data_morph_v2/_usage.json`, keyed by a hash of each
key so a rotated key gets a fresh counter instead of inheriting the old one's spent budget; when
the daily pool is exhausted the run checkpoints and exits cleanly instead of burning retries.

Three design choices worth knowing, all from `docs/literature_review.md`:

- **The LLM writes the semantics; rules check the morphology.** LLMs are measurably weak at
  Turkish morphological productivity, so `morph_validators.py` verifies vowel harmony (both
  backness and rounding), consonant assimilation, vowel hiatus and consonant softening on the
  contrasting word pair — which is *derived from the passages*, not taken from the model's own
  metadata.
- **The judge never sees the labels.** It classifies every candidate blind; the comparison against
  what the generator intended happens in Python. This gives a real single-gold probe rather than
  an LLM approving its own family's output.
- **The set is audited for query-blind artifacts.** v1.3.1 has a measured 50% "just pick the
  longest candidate" baseline against a 9% chance level — positives there are systematically
  longer. v2.0 is generated and gated against that; `generation_report.md` reports the numbers
  either way.

Outputs in `data_morph_v2/`: `morph_{train,dev}_v2.0.json` (v1.3.1 shape, so existing tooling
reads them unchanged), `*_pairs_*.jsonl` (MNRL-ready, negatives grouped per query with difficulty
rank and graded scores), `*_paired_*.jsonl` (NevIR-style gold-vs-negative rows for pairwise
accuracy), `rejected.jsonl`, and `generation_report.md`.

### v2.2 (current) — how the candidates are built

Every candidate is assembled **in code** from a shared frame plus a per-candidate core:

```
text = frame_before + core + frame_after      # frame identical across all 11 candidates
```

The model emits the frame once and one short core per candidate; `assemble_item` concatenates.
Asking a model to reproduce the same context in 11 candidates is exactly the instruction it drifts
on, and that drift was the length artifact. Concatenating in Python makes the frame byte-identical
by construction.

Three properties follow, and each is enforced somewhere:

| property | how |
|---|---|
| hard negatives inherit the query's surface | their cores are minimal edits of the query (ADIM 4) |
| the positive does not | its core is a re-telling, written last to match the negatives' length |
| the query is as long as a core | ADIM 1 gives an explicit character band, so a one-word edit of the query comes out the same length as the positive |

`minimal` tier inverts the first two — there the counterfactual is a one-word edit of the
*positive*, because that is what a minimal pair means. Each tier is shown only its own recipe
(`_STEP4_QUERY_ANCHORED` / `_STEP4_POSITIVE_ANCHORED`); a conditional in one shared prompt was
dropped by the model half the time.

The judge prompt states up front that the correct answer is *deliberately* worded differently from
the query and the wrong ones are near-copies. Without that, it rejected correct positives for
"different vocabulary" — fixing it took yield from 29% to 58% on identical generations.

### v2.0 → v2.1: what the prompt fix changed

v2.0 was lexically solvable in a way the human-built v1.3.1 is not. Root cause: v2.0 built each
hard negative as a minimal edit of the **positive**, so the positive inherited the query's surface
and a bag of character trigrams could find it. v1.3.1 anchors the negatives on the **query** and
lets the positive be an independent re-telling, which makes lexical overlap point at a *wrong*
candidate. v2.1 adopts that anchoring (`morph_prompts.py`, ADIM 4).

Sparse char-3gram pairwise accuracy, random = 50, **lower is better** (below 50 means overlap
actively misleads):

| negative type | v2.0 | v2.1 | **v2.2** | v1.3.1 (human) |
|---|---|---|---|---|
| `morph_counterfactual` | 65.8 | 16.4 | **2.2** | 23.5 |
| `same_feature_wrong_content` | 88.3 | 38.3 | **19.8** | 25.5 |
| `state_variant` | 63.4 | 26.2 | **9.6** | 34.3 |
| positive is sparse top-1 | 31.4% | 8.2% | **1.3%** | 12% |

A bag of character trigrams picks the gold 1.3% of the time against a 9.1% chance rate — worse than
guessing, because lexical overlap points at the counterfactual instead. On every hard-negative type
v2.2 is more adversarial than the human-built set.

**Length balance — much improved, target narrowly missed.** `blind_longest_is_gold` is reported two
ways, because the bare argmax overstates the artifact once a shared frame clusters all candidate
lengths together:

| | v1.3.1 | v2.0 | v2.1 | v2.2 |
|---|---|---|---|---|
| positive is longest (argmax) | 42% | 19% | 33% | 49.5% |
| **longest by >5% (exploitable)** | 36% | 5% | 11% | **20.3%** |

The second row is the honest one: 20.3% against a 9.1% chance level and a ≤15% target. Better than
v1.3.1's 36%, worse than v2.0's 5%. The residual comes from the positive being a re-telling, which
runs slightly long even when the query is length-matched. `LONGEST_MEDIAN_RATIO` (1.25, measured on
assembled text against the median of the other candidates) bounds it; tightening it trades yield,
and because it is a validator constant rather than a prompt, re-scoring the cached responses at
another value costs nothing.

That gate is deliberately calibrated *stricter* than the human data, so it is excluded from the
"v1.3.1 must pass" sweep in `morph_selftest.py` and gets a directional test instead: it must fire
on v1.3.1 (which has the artifact) and stay quiet on v2.0 (which does not).

Reproduce any of this without an API key:

```bash
conda run -n dl_hw1 python eval_morph_dev.py --split dev
```

## v2.3 — morphological annotation, variant groups, BEIR export

Pure post-processing over the v2.2 items: **zero API calls**, no regeneration. Run in place on the
existing dataset:

```bash
conda run -n dl_hw1 python morph_annotate.py --split train --split dev
conda run -n dl_hw1 python morph_beir.py --split train --split dev --split test
```

`gen_morph_dataset.py` also calls both automatically from now on, so a future full regeneration
ships them without a separate step.

**Morphological annotation** (`item["morphology"]`, `item["variants"]` in the output JSON).
Two tiers: a deterministic Tier 1 that recovers `stem`/`diff_query`/`diff_counterfactual` for the
critical-word pair via a windowed boundary search against an allomorph table *parsed from
`morph_taxonomy.py`'s own `ek_turu` strings* (not hand-maintained), and an optional Tier 2
(`zeyrek`) giving a full lemma + gold morpheme tags. The search runs against the FULL table, not
just the item's own declared feature — that is what turns it into a label audit rather than a
confirmation exercise.

| metric | train | dev |
|---|---|---|
| Tier 1 exact match | 59.8% | 70.0% |
| Tier 1 agrees with declared `target_feature` | 60.3% | 36.5% |
| no recoverable word-level pair (`no_pair`) | 1.8% | 1.1% |
| zeyrek parse rate | 91.4% | 87.8% |

Three things worth knowing before trusting these numbers at face value:

- **The train/dev agreement gap (60.3% vs 36.5%) is unexplained.** Both splits are annotated by
  the identical code; dev is smaller (90 vs 510 items) so some of the gap is plausibly sampling
  variance, but a gap this size hasn't been root-caused. Treat the disagreement rate as a QC
  signal to investigate per-split, not a single trustworthy number.
- **`no_pair` is a real, if small, QC finding, not just an edge case.** Every occurrence was
  inspected by hand during development. Most are legitimate multi-word contrasts a single-word
  matcher was never going to catch (`DISTR`'s `birer` vs `toplu halde`); a few are genuine
  generation defects that slipped past the entire v2.2 judge/validator pipeline (`priv_0528_40b1dd`'s
  counterfactual candidate is ungrammatical — `"... kira kontratı lidersi."`); a few are `PRIV`/`PROP`
  items where the model realised the contrast as the suppletive existential pair `var`/`yok`
  instead of the nominal `-lı`/`-sız` suffix it was asked for (happened 3 times independently).
  See `no_pair_examples` in each split's `statistics.morphology_annotation` for the full list.
- **A known, separate confound lowers the zeyrek-vs-label agreement specifically**, documented in
  `morph_annotate.py` next to `tier2_annotate`: for chain features whose reported critical word is
  a *phrase* (rejected by the single-word check), `morph_validators.resolve_critical_pair` falls
  back to whichever positive-vs-counterfactual word pair shares a stem FIRST — even an incidental
  one from the independently-reworded positive — rather than trying the query-vs-counterfactual
  fallback that would recover the true contrast. Confirmed case: an `EVID.COND.NEG` item's real
  contrast (`duyurmuş olsaydı` vs `duyurmuştu`, exactly as reported) gets bypassed in favour of an
  unrelated noun's case-marking difference (`biletinin`/`biletini`) elsewhere in the reworded
  positive. Not fixed here — it's existing, self-tested pipeline code with its own calibration
  against v1.3.1, and reordering its fallback needs its own measurement pass, not a change bundled
  into an annotation tool.

**Variant groups** (`item["variants"]`): Turkish-correct deasciified/lowercased forms of the query
and every candidate (`benimkiydi`→`benimkiydi`, `İlgili`→`ilgili` — not `i̇lgili`, the bug this
round fixed; see below), plus a `lemma_family` grouping (zeyrek lemma when available, else the
derived Tier-1 stem) for a root-family retrieval diagnostic. Verified on every generated item: no
two candidates within one item collapse to the same string after deasciification, so the
perturbation never destroys an item's unique correct answer — locked in by `morph_selftest.py`.
Score the diacritic-robustness slice with `eval_morph_dev.py --variant deascii`.

**BEIR export** (`data_morph_v2/beir/<split>/`): standard `corpus.jsonl` / `queries.jsonl` /
`qrels/test.tsv`, loadable by the `beir`/`mteb` libraries or this repo's own
`eval_semantic_encoders.load_beir_dataset`-style reader. Pooling is not free — standard BEIR
merges every candidate from every query into one corpus, and a foreign query's candidate can
legitimately outrank this query's own gold: measured at **144/510 train, 12/90 dev, 22/50 test**
queries. Two sidecars carry what pooled BEIR can't express: `candidate_pool.json` (each query's
own 11 candidate ids, reproducing the closed-set numbers this project reports everywhere else) and
`hard_negatives.jsonl` (the typed structure — `morph_counterfactual`, `same_feature_wrong_content`,
`partial_trap`, `state_variant` — which is the dataset's actual contribution over a generic BEIR
set and has no field to live in otherwise). Score either framing explicitly:

```bash
conda run -n dl_hw1 python eval_morph_dev.py --split dev --pool closed   # this project's numbers
conda run -n dl_hw1 python eval_morph_dev.py --split dev --pool beir     # standard, harder, different task
```

**Also fixed this round**: `morph_validators.tokens()` mis-split any word containing a capital
`İ` (`İlgili` → `['i', 'lgili']`) because Python's `.lower()` emits a combining dot Turkish doesn't
use. Affected 23% of v2.2 items' candidates. Fixed via `tr_lower()` (`İ`→`i`, `I`→`ı`, then
`.lower()`); re-validating all 510 train items with the fix changed zero accept/reject verdicts.

## How to run the notebook

1. Upload the notebook to [Google Colab](https://colab.research.google.com) (or open via
   GitHub/Drive). Runtime → Change runtime type → **A100 or L4 GPU** (Colab Pro).
2. Optional but recommended: accept the license for
   [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) and run the
   `notebook_login()` cell — otherwise the gated model is skipped automatically.
3. Run all. Full run ≈ **1.5–3 h** (10 semantic + 7 small models, 4 datasets + probe).
   Results are cached in `cache_results/` — interrupted runs resume where they stopped.

### Quick smoke test

Set `SMOKE=1` in the environment (or flip the flag in the setup cell) to run 2 small models on a
capped NFCorpus-TR subset — verifies the full pipeline in minutes on CPU/T4.

## Benchmarks used (trusted sources)

| Source | Used for |
|---|---|
| [TR-MTEB (Findings of EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.471/) | published retrieval nDCG@10; benchmark conventions |
| [Mizan leaderboard (NewMind AI)](https://newmindai-mizan.hf.space) | published overall Turkish MTEB scores |
| Turkish-BEIR sets ([TurkColBERT blog](https://huggingface.co/blog/nmmursit/late-interaction-models)) | our own retrieval runs: SciFact-TR, NFCorpus-TR, ArguAna-TR, FiQA-TR |
| Project pilot (deck p.8) + extended 50-triplet set | morphological suffix probe |

## Notes

- Proprietary API models (e.g. `text-embedding-3-small`) appear as **cited reference rows only**;
  the notebook never calls paid APIs.
- Prompts matter: e5/Qwen3/Gemma-style models are encoded with their trained query/passage
  prefixes (see the model registry cell) — comparing without them silently sandbagged models in
  several published comparisons.
- `morph_probe_triplets.json` is written on first run for team review of the probe set.
