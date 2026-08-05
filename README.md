# Morphology-Aware Contrastive Fine-Tuning for Turkish Retrieval

inzva AI Projects #10. Building a dual-encoder retrieval system for Turkish — a general-purpose
**semantic encoder** (LoRA fine-tuned) paired with a lightweight **morphological-channel encoder**
tuned to Turkish suffix/agglutination signal, on the premise that a single pooled vector washes out
meaning carried by a single suffix (`katıldı` vs `katılmadı` vs `katılamadı`).

This repo currently covers two finished phases — picking the base models, and building the
training/eval data that will judge the fine-tuned result — with no training code yet.

## Repository layout

| Path | What it is |
|---|---|
| [`turkish-retrieval-model-selection/`](turkish-retrieval-model-selection/) | Active project: model-selection notebooks, the local retrieval benchmark, the literature review, and the full v2 dataset-generation pipeline. **See its [README](turkish-retrieval-model-selection/README.md) for the full picture.** |
| [`morph_eval_set_v1.3.1_review_reviewer_C_fixed.json`](morph_eval_set_v1.3.1_review_reviewer_C_fixed.json) | The held-out, hand-QC'd **test set** — 50 queries × 11 candidates, two human review passes. Never touched by the generation pipeline. |
| [`morph_eval_set_v1.3_review_reviewer_C.json`](morph_eval_set_v1.3_review_reviewer_C.json) | Earlier reviewer pass, superseded by the file above; kept for provenance. |
| [`CLAUDE.md`](CLAUDE.md) | Guidance for Claude Code when working in this repo — architecture notes, gotchas, conventions. |
| `model_selection_colab_runned_version.ipynb` | Executed snapshot of the model-selection notebook, kept for its outputs. |

## Headline result

A synthetic training set generated with Gemini, validated against Turkish phonology rules and a
blind LLM judge, and audited for the lexical- and length-artifacts that make this kind of dataset
easy to get wrong. Sparse char-3gram pairwise accuracy — can a model-free bag-of-trigrams scorer
tell gold from a hard negative? Random is 50%; **lower is better**, since a hard negative should be
lexically indistinguishable from the gold and only separable by morphology:

| negative type | v1.3.1 (human-built test set) | v2.2 (generated) |
|---|---|---|
| `morph_counterfactual` — differs by one suffix, meaning reverses | 23.5% | **2.2%** |
| `same_feature_wrong_content` — same suffix, wrong referent | 25.5% | **19.8%** |
| `state_variant` — different person/time/state | 34.3% | **9.6%** |

On every hard-negative type, the 600-item generated set (75/75 target morphological features
covered, 510 train / 90 dev) is *more* lexically adversarial than the hand-built test set — a bag
of trigrams does worse than guessing. Full methodology, the v1.3.1→v2.0→v2.1→v2.2 iteration that
got there, and the honest remaining gap (a length artifact, quantified and not hidden) are in
[`turkish-retrieval-model-selection/README.md`](turkish-retrieval-model-selection/README.md).

## Quickstart

```bash
# validators + phonology self-test, no API calls
conda run -n dl_hw1 python turkish-retrieval-model-selection/gen_morph_dataset.py --self-test

# score the generated dev split against the held-out test set, no API calls
conda run -n dl_hw1 python turkish-retrieval-model-selection/eval_morph_dev.py --split dev
```

Regenerating the dataset from scratch needs a `.env` file at the repo root with one or more
`API_KEY_N="..."` lines (Google AI Studio / Gemini keys) — see the subproject README for the full
`gen_morph_dataset.py` pipeline and budget model. `.env` is gitignored and not included here.

Everything else (`conda run -n dl_hw1 python ...`) assumes a conda env named `dl_hw1` with `torch`,
`sentence-transformers`, `transformers`, `pandas`, `huggingface_hub`, `google-genai`, and
`requests` installed — there is no `requirements.txt` in the repo yet.

## Further reading

- [`docs/literature_review.md`](turkish-retrieval-model-selection/docs/literature_review.md) — how
  prior work has handled morphology in representation learning and IR for agglutinative languages;
  81 sources, each tagged by how well it was verified.
- [`data_morph_v2/generation_report.md`](turkish-retrieval-model-selection/data_morph_v2/generation_report.md) —
  full QC report for the current dataset: rejection reasons, coverage, artifact audits.

## License

MIT — see [LICENSE](LICENSE).
