# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This repo holds the **model-selection phase** of *Morphology-Aware Contrastive Fine-Tuning for
Turkish Retrieval* (inzva AI Projects #10). The end goal is a dual-encoder retrieval system for
Turkish with two encoders:

- a **semantic encoder** (a general-purpose embedding model, later LoRA fine-tuned)
- a **morphological channel encoder** (lightweight, tuned to Turkish suffix/agglutination signal)

Everything currently in this repo covers two finished phases — *picking* the base models for both
encoders, and *building the training/eval data* that will judge the fine-tuned result later —
there is no training code yet.

This is a git repository (`main` branch); commit history is authoritative for what changed and
why — check it before assuming something is undocumented.

## Repository layout

- `turkish-retrieval-model-selection/` — the active project (see its
  [README](turkish-retrieval-model-selection/README.md) for the fuller picture)
  - `model_selection_colab.ipynb` — the canonical comparison notebook (Colab, A100/L4). Sections:
    A published scores → B our Turkish-BEIR retrieval runs → C morphological suffix probe →
    D efficiency → E weighted selection matrix → Decision.
  - `semantic_encoder_eval_a100.ipynb` — a narrower, semantic-encoder-only version of the same
    retrieval eval (no morph probe, no gated models), meant for a clean A100 Colab run.
  - `eval_semantic_encoders.py` — a standalone local port of notebook Section B, for running the
    same retrieval benchmark outside Colab (see Commands below).
  - `results_local/` — cached per-model results (`retr_<slug>.json`), the combined CSV/PNG, and
    run logs from `eval_semantic_encoders.py`. Deleting a model's cache file re-runs just that
    model.
  - `gen_morph_dataset.py`, `morph_prompts.py`, `morph_validators.py`, `morph_taxonomy.py`,
    `morph_selftest.py`, `eval_morph_dev.py` — the **v2 dataset-generation pipeline**. Synthesises
    a large morphological retrieval train/dev set with Gemini, gated by rule-based Turkish
    phonology checks and a blind LLM judge. See "Dataset-generation pipeline" below and the
    subproject README for the full design rationale.
  - `docs/literature_review.md` — literature survey backing the pipeline's design choices (81
    sources, each tagged by verification confidence).
  - `data_morph_v2/` — pipeline output: `morph_{train,dev}_v2.2.json` (current), `archive_v2.0/`
    and `archive_v2.1/` (earlier iterations, kept as the evidence trail for what each prompt fix
    changed — see the subproject README's comparison tables), `generation_report.md` (QC report),
    `rejected.jsonl` (every rejected item with its reason, never silently dropped).
    `_cache/` and `_usage.json` are local run state, gitignored.
- `model_selection_colab_runned_version.ipynb` (repo root) — an **executed snapshot** of
  `model_selection_colab.ipynb` kept for its outputs/results. It can drift slightly from the
  live notebook in the subfolder; treat the subfolder copy as the one to edit.
- `morph_eval_set_v1.3.1_review_reviewer_C_fixed.json` (repo root) — the **held-out test set**:
  two human review passes over 50 queries × 11 candidates each (one gold positive, four typed
  hard negatives — `morph_counterfactual`, `same_feature_wrong_content`, `partial_trap`,
  `state_variant` — six easy negatives). This is the real morphology benchmark; the v2 pipeline
  reads it for a few few-shot exemplars and its `fix_log` (an empirical taxonomy of what a human
  reviewer had to fix) but **never writes to it**. `morph_eval_set_v1.3_review_reviewer_C.json`
  is the pre-fix reviewer pass, kept for provenance only — nothing reads it programmatically.
- `.env` (repo root, gitignored) — `API_KEY_1`, `API_KEY_2`, ... Gemini API keys (Google AI
  Studio) for the dataset-generation pipeline. Not required to read/run the model-selection
  notebooks or score existing data (`eval_morph_dev.py`), only to regenerate the dataset.
- `.agents/skills/` — generic subagent persona definitions (nlp-engineer, data-engineer,
  prompt-engineer); not project-specific configuration.

## Commands

Colab notebooks install their own pinned deps in-notebook (`%pip install ...`) — no local
environment setup needed to read or run them there.

For the local scripts (conda env `dl_hw1`):

```bash
# semantic-encoder retrieval benchmark — full run (hours on Apple Silicon MPS / CPU)
conda run -n dl_hw1 python turkish-retrieval-model-selection/eval_semantic_encoders.py

# quick pipeline smoke test (2 small models, capped NFCorpus-TR subset)
SMOKE=1 conda run -n dl_hw1 python turkish-retrieval-model-selection/eval_semantic_encoders.py

# dataset-generation pipeline: validators + phonology self-test, no API calls
conda run -n dl_hw1 python turkish-retrieval-model-selection/gen_morph_dataset.py --self-test

# regenerate the morphological train/dev set (needs .env; resumable; see subproject README)
conda run -n dl_hw1 python turkish-retrieval-model-selection/gen_morph_dataset.py --target 600

# score any split's lexical solvability with no model download
conda run -n dl_hw1 python turkish-retrieval-model-selection/eval_morph_dev.py --split dev
```

There is no `requirements.txt`/`environment.yml` in the repo — the `dl_hw1` conda env is assumed
to already have `torch`, `sentence-transformers`, `transformers`, `datasets`, `pandas`,
`huggingface_hub`, `matplotlib`, `google-genai`, `requests` installed. There are no linters
configured; `morph_selftest.py` (invoked via `--self-test` above) is the only test suite.

## Dataset-generation pipeline gotchas

- **The LLM writes semantics, rules check morphology.** `morph_validators.py` verifies Turkish
  vowel harmony (backness *and* rounding), consonant assimilation/softening, and vowel hiatus on
  the contrasting word pair, derived from the actual candidate texts rather than trusted from the
  model's self-reported metadata — LLMs are measurably unreliable at Turkish morphological
  productivity, so nothing about the suffix contrast is taken on faith.
- **The judge never sees the generator's labels.** It classifies each candidate blind; agreement
  is computed in Python. Showing an LLM its own labels and asking "is this right?" is the known
  failure mode this avoids.
- **Every candidate is assembled from a shared frame + a per-candidate core**, concatenated in
  Python (`text = frame_before + core + frame_after`), not written whole by the model. Asking a
  model to reproduce identical surrounding context across 11 candidates is exactly where it drifts
  — that drift was the project's length artifact, and it disappeared once the frame became
  Python-concatenated rather than model-authored.
- **Cache keys include a hash of the prompt, not just the slot id**, and specifically exclude the
  rolling ban-list line — including it made every cache key depend on generation history, so a
  single differently-classified item silently invalidated hundreds of cached responses on resume.
- **The daily request ledger is keyed by a hash of each API key**, not by its `.env` position —
  rotating `API_KEY_1` to a new value must not inherit the old key's spent quota.
- Before trusting any change to the validators, run `--self-test`: it checks that all 50 hand-QC'd
  v1.3.1 items pass every gate (with genuine v1.3.1 defects listed explicitly, not silenced) and
  that deliberately corrupted copies fail on the specific gate meant to catch them.

## Architecture notes

**One registry drives each notebook/script.** Candidate models are plain dicts with `id` (HF
repo), `role` (`semantic` / `morph` / `reference` in the full notebook; `finished` / `raw` in the
local script), the exact **query/passage prompt strings the model was trained with**, and
`trust_remote_code`. Getting the prompt wrong is called out repeatedly as the classic silent bug
in embedding comparisons (e5/Qwen3/Gemma-style models need their prefixes or they're silently
sandbagged) — when adding a candidate, copy its prompt convention from the model card exactly.

**Retrieval eval pipeline** (shared shape across notebook Section B and `eval_semantic_encoders.py`):
1. Load 4 `trmteb`-org BEIR-format Turkish datasets (`scifact-tr`, `nfcorpus-tr`, `arguana-tr`,
   `fiqa-tr`) directly via `requests` + `hf_hub_url` on individual parquet files — **not**
   `datasets.load_dataset()`, because the HF `xet` transfer backend is flaky/hangs on Colab and
   GCP (`HF_HUB_DISABLE_XET=1` is set for the same reason).
2. Encode corpus/queries with each model's own prompts, L2-normalized.
3. Brute-force top-k cosine search in torch (CPU/CUDA; MPS is skipped for the matmul step since
   it gains nothing there).
4. Score nDCG@10 / MRR@10 / Recall@{10,100} against qrels.
5. Cache one JSON per model in `results_local/`, keyed by a sanitized model-id slug — this is
   what makes long runs resumable after interruption.

`fiqa-tr`'s 57K-doc corpus dominates runtime on constrained hardware; it's capped (keeping every
gold/relevant doc, subsampling filler negatives) via `FIQA_CORPUS_CAP` in the local script — the
A100 notebook runs it uncapped.

**Morphological probe** (notebook Section C) is separate from the retrieval eval: small
anchor/distractor triplet sets testing whether a model's embedding space is sensitive to a single
Turkish suffix change (negation, tense, case, etc.). It's a fast pilot-scale sanity check, not the
project's real morphological benchmark — that's the held-out set in
`morph_eval_set_v1.3.1_review_reviewer_C_fixed.json` (see above), now also the test set the v2
dataset-generation pipeline is built and gated against.

**Selection matrix** (notebook Section E) is a min-max-normalized weighted score:
`own_ndcg10 (0.40) + morph_acc (0.30) + mizan (0.20) + speed (0.10)` — adjust the `WEIGHTS` dict
in-notebook rather than hand-computing rankings.

**Model roles matter for interpretation, not just filtering**: `raw`/foundation models
(TabiBERT, TURKCELL-roberta) get auto mean-pooled by `SentenceTransformer` and are *expected* to
score low zero-shot — they're fine-tuning bases, not finished retrievers. Don't read a low score
on those as a bug. `reference`-role entries in the full notebook (e.g. proprietary API embedders)
are cited-only rows — the notebook never calls paid APIs.

**Published-score provenance matters**: `trmteb_retr` (TR-MTEB paper, Table 2) and `mizan`
(Mizan leaderboard) are different snapshots from different dates — never average them into one
number; keep them as separate cited columns.
