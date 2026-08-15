# Morphology-Aware Retrieval for Turkish — Literature Review

**Project:** *Morphology-Aware Contrastive Fine-Tuning for Turkish Retrieval* (inzva AI Projects #10)
**Scope:** how prior work has handled morphology in representation learning and IR for
agglutinative languages; how meaning-flipping minimal contrasts have been attacked elsewhere; and
how evaluation sets that isolate a single linguistic feature are built and validated.
**Compiled:** 2026-08-01 · 81 unique sources.

## How to read the citations in this document

Every source carries a verification tag, because a large share of the Turkish literature here is
2026 preprints that could only be reached as abstracts:

| Tag | Meaning |
|---|---|
| **[F]** | Full text (or full HTML) was fetched and read. Numbers quoted are read off the page. |
| **[A]** | Only the abstract / landing page was read. Numbers are the authors' abstract claims. |
| **[S]** | Search snippet only. **Treat every number as unverified.** |

Nothing in this document should be cited in the report at a confidence higher than its tag. One
survey agent explicitly flagged a "results table" extracted from a PDF that appeared to be
model-fabricated rather than present in the source; that paper (Learning Robust Negation Text
Representations) is included below **without** numbers for exactly that reason.

---

## 0. Executive summary

Four things fall out of the literature, and all four bear directly on this project.

**First, the artifact this project has already built appears to be novel.** Six independent survey
passes converged on the same gap: there is no published minimal-pair *retrieval* benchmark for
Turkish morphology. NevIR **[A]** is English and negation-only; TurBLiMP **[F]** is a
log-probability acceptability benchmark over sentence pairs, not a ranking set; Thunder-KoNUBench
**[F]** is Korean and multiple-choice; Morpheus **[A]** measures root-family clustering, which is the
*opposite* axis (same root, different suffix → should cluster) from suffix semantics (same root,
different suffix → must separate). A 50-query set where distractors differ by one morpheme and the
correct answer changes has no direct predecessor. That is a publishable contribution, not an
internal eval — and it also means there is no external baseline to calibrate against.

**Second, the dual-encoder architecture is the weakest architecture for this exact task.** On NevIR,
where documents differ only by negation, reproduced pairwise accuracies **[F]** are DPR 6.5,
msmarco-bert-base-dot-v5 6.9, all-mpnet-base-v2 8.1, multi-qa-mpnet-base-dot-v1 11.1 — against a
random baseline of 25. Cross-encoders reach 27.7–50.6 and listwise LLM rerankers 46.3–64.1. The
ordering cross-encoder > late-interaction > bi-encoder is stable across the original and the
reproduction. A single pooled vector is structurally poor at "same topic, opposite polarity", and
Turkish suffix contrasts are that problem with a smaller surface delta. Budget for a ceiling.

**Third, the mechanism that most reliably improves morphological handling is *tokenization*, and the
evidence for it is genuinely contradictory.** Toraman et al. **[A]** — the only matched-conditions
study, pretraining five tokenizers under one budget — found morphological-level tokenization merely
*competitive* with WordPiece/BPE on Turkish downstream tasks. MorphScore v2 **[S]** reports that
morphological alignment does not significantly correlate with downstream performance across 70
languages. Against that, Morpheus **[A]**, Tokens with Meaning **[F]** and MorphBPE **[A]** all
report clear gains. The discrepancy is unresolved in the literature and should be stated as such.

**Fourth, LLM-synthesized morphological data is a real risk, and the literature names the specific
failure modes.** LLMs lack systematicity in morphological composition **[A]**; TurBLiMP deliberately
used a masked LM plus manual verification rather than an instruction-tuned generator **[F]**;
CausalNeg measured 24% of LLM-generated negatives forming clusters separable by generation origin
rather than relevance **[F]**. Every one of these has a named mitigation, and Section 6 lists them.

---

## 1. How Turkish morphology has been handled: the analyzer tradition

Turkish morphological processing has a 30-year finite-state tradition that remains the substrate for
everything else.

**Oflazer (1994), *Two-level Description of Turkish Morphology* [S]** — Koskenniemi-style two-level
morphotactics plus morphophonemic rules. Cited across the newer literature as the ancestor of
essentially every rule-based Turkish analyzer since.

**Çöltekin (2010), *TRmorph* [S]** ([L10-1068](https://aclanthology.org/L10-1068/)) — a freely
licensed two-level FST analyzer built with SFST: lexicon, morphotactic FSA, two-level phonological
rules. `coltekin/TRmorph` on GitHub, with a `trmorph2` branch.

**Akın & Akın (2007), *Zemberek* [S]** ([github](https://github.com/ahmetaa/zemberek-nlp)) — the
open-source Java framework: analysis, ambiguity resolution, **and word generation**. The generation
direction is the underused half: given root + feature bundle it synthesizes the surface form. That
is precisely the operation needed to build a guaranteed-correct morphological counterfactual.

**Sak, Güngör & Saraçlar (2008) [S]** — stochastic FST parser plus an averaged-perceptron reranker
over n-best parses, trained on the Yuret & Türe semi-automatically disambiguated ~1M-token corpus
with a ~20K-token manually annotated test set. Disambiguation accuracies of 96.80% and 97.81% appear
in search snippets; **not verified against the paper**.

**Türk et al. (2021), *BOUN Treebank* [S]** — manually annotated UD treebank; morphological features
and UPOS produced automatically by the Sak et al. parser, converted to UD, then hand-corrected. A
follow-up re-annotation handles null morphemes, heavy agglutination, and syncretic morphemes such as
the copula and `-ki`. This is the gold-standard source for morphological probing (Section 3).

**Ozen & Can (2017), *Building Morphological Chains for Agglutinative Languages* [S]** — extends the
log-linear MorphoChains model so the candidate space expands recursively rather than by binary split,
letting a word segment into an arbitrarily long suffix chain — the correct inductive bias for
Turkish. Reported +12% over prior state of the art to F-measure 72% for Turkish (snippet-sourced).

> **Takeaway for this project.** The analyzer tradition gives a deterministic path from
> *root + feature bundle → surface form*. Wherever a morphological mutation needs to be exact, a
> generator is strictly better than an LLM. This is the single most actionable finding in the review
> and it is why the v2.0 pipeline validates the LLM's suffix edits mechanically rather than trusting
> them.

---

## 2. Tokenization: the dominant intervention, and an unresolved dispute

Almost all Turkish-specific work that claims morphological improvement operates on the tokenizer.

### 2.1 The morphology-aware tokenizers

**Tokens with Meaning (Bayram et al., 2026) [F]** ([arXiv 2508.14292](https://arxiv.org/html/2508.14292))
— a three-part TurkishTokenizer: dictionary-driven root/affix segmentation (22,231 roots mapped to
20,000 canonical IDs; **72 affix IDs covering 177 allomorphic surface forms**), phonological
normalization collapsing allomorphs onto shared IDs so `-den/-dan/-ten/-tan` become one identifier,
and a controlled subword fallback for OOV. Reported: TR-MMLU Turkish token percentage 90.29%, pure
token percentage 85.80%, round-trip exact-match reconstruction 99.48%. With randomly initialized
models: STSb-TR Pearson 51.44% vs Tabi 43.01 / CosmosGPT2 43.09 / Mursit 46.63; TR-MTEB overall
average 39.57% vs 34.92 / 35.65 / 35.92.

**Morpheus (Şakar, 2026) [A]** ([arXiv 2606.18717](https://arxiv.org/abs/2606.18717)) — joint neural
tokenizer and word embedder. Per-character morpheme-boundary probabilities become soft segment
assignments during training via a differentiable Poisson-binomial dynamic program, and hard segments
at inference; round-trip tokenization is lossless by construction. Abstract reports 1.425
bits-per-character (lowest among reversible tokenizers compared), MorphScore macro-F1 0.61 vs ~0.32
for the subword family, ~19% less GPU memory than 64K-vocab subword models, **root-family retrieval
MAP 0.85 and same-root verification ROC-AUC 1.00**, beating BGE-M3 and BERTurk on those probes.

**MorphBPE (Asgari et al., 2025) [A]** ([arXiv 2502.00894](https://arxiv.org/abs/2502.00894)) —
constrains BPE merges to respect morpheme boundaries, keeping statistical efficiency while adding
linguistic structure. Introduces Morphological Consistency F1 and Morphological Edit Distance.
Claims reduced cross-entropy and faster convergence at 300M and 1B scales. No retrieval numbers.

**HeceTokenizer (Gulgonul, 2026) [A]** ([arXiv 2604.10665](https://arxiv.org/abs/2604.10665)) —
phonology- rather than morphology-driven: Turkish's six deterministic syllable patterns give a
closed, OOV-free vocabulary of ~8,000 syllable types. A 1.5M-parameter BERT-tiny trained from
scratch on Turkish Wikipedia reaches 50.3% Recall@5 on TQuAD retrieval vs 46.92% for a
morphology-driven baseline the abstract describes as ~200× larger.

**MorphPiece (Jabbar, 2023) [S]**, **MorphTE (Gan et al., 2022) [S]** (morpheme-indexed tensorized
embeddings, ~20× compression claimed), and **Korean morpheme-aware subword tokenization (2023) [S]**
(analyzer at training time, analyzer-free at inference — an operationally important pattern) round
out the family.

### 2.2 The counter-evidence

**Toraman et al. (2022/2023), *Impact of Tokenization on Language Models: An Analysis for Turkish*
[A]** ([arXiv 2204.08832](https://arxiv.org/abs/2204.08832)) — the strongest matched-conditions study
available. Five tokenizers (character, word, morphological-level, WordPiece, BPE), matched
medium-size RoBERTa models pretrained on the Turkish OSCAR split, six downstream tasks, statistical
testing. Findings: the morphological-level tokenizer is **competitive but not superior** to
WordPiece/BPE; BPE ≈ WordPiece; word- and character-level are drastically worse downstream; and
**morphological tokenizers benefit far more from larger vocabulary** — the paper suggests ~20%
vocabulary-to-parameter ratio for WordPiece/BPE vs ~40% for the morphological tokenizer.

**MorphScore v2 (Arnett, Hudspeth & O'Connor, 2025) [S]**
([arXiv 2507.06378](https://arxiv.org/abs/2507.06378)) — expands morphological-alignment scoring from
22 to 70 languages and correlates alignment with downstream performance for five pretrained LMs
across seven tasks. Headline (snippet-sourced, **not verified**): morphological alignment does
**not** significantly correlate with model performance and does not explain much variance.

**Optimal Turkish Subword Strategies at Scale (Altinok, 2026) [F]**
([arXiv 2602.06942](https://arxiv.org/html/2602.06942)) — jointly sweeps vocabulary size and
tokenizer-training-corpus size up to ~80GB under matched parameter budgets. Character-level reaches
91.56 POS accuracy, 65.19 UAS, 57.15 LAS on BOUN and morphological micro-accuracy 96.19 — near
ceiling on several categories — but shows large gaps on STS-B and CoLA. Word-level collapses
(~60 POS, ~19 LAS, ~12 morphology F1). Introduces a morphology-aware diagnostic suite: boundary
micro/macro-F1 against gold morpheme boundaries, lemma boundary hit rate, over-/under-segmentation
indices, affix-type coverage, continuation rate, fertility.

**Truong et al. (2024), *Revisiting subword tokenization: affixal negation* [S]**
([2024.naacl-long.284](https://aclanthology.org/2024.naacl-long.284/)) — directly relevant negative
result: there are mismatches between tokenization accuracy and negation-detection performance, and
on the whole models *do* reliably recognize the meaning of affixal negation. Morphologically
implausible tokenization is therefore **not automatically** the cause of a morphological failure.

> **Takeaway.** The character-level result is the interesting one: near-ceiling morphology,
> collapsed semantics. That profile — strong morphology, weak semantics — is the empirical
> justification for a *two-channel* architecture rather than one model asked to do both. It also
> warns against the obvious shortcut of simply swapping the semantic encoder's tokenizer.

### 2.3 Adapting an existing multilingual encoder instead

**Cross-Lingual Tokenizer Surgery + Offline Distillation (Bayram, Diri & Yıldırım, 2026) [F]**
([arXiv 2605.29992](https://arxiv.org/html/2605.29992)) — build a 128K Turkish-optimized vocabulary
by pruning redundant teacher tokens and re-adding frequency-selected multilingual tokens; clone the
teacher keeping transformer weights; remap the embedding matrix by **mean-composition** of the old
subword pieces; then distill offline against precomputed teacher embeddings with a cosine loss.
EmbeddingGemma-300M teacher → 200M student. Reported: STSb-TR Pearson **77.55 vs teacher 73.84**,
Spearman 77.45 vs 72.92; TR-MTEB 26-task average 63.9 (7th of 26) vs teacher 65.2; +4.6% STS,
+7.3% NLI relative to teacher; ~98% of teacher performance with 33% fewer parameters; **$5–20 and
~4 GPU-hours**. Gains attributed to reduced fragmentation of Turkish surface forms.

This is a cheap, strong ablation arm: it isolates *"Turkish fragmentation is the problem"* from
*"the model lacks morphological representation."* If surgery alone closes most of the morphological
gap, a second encoder is not earned.

---

## 3. Morphology inside the representation, not the tokenizer

The older, language-general tradition injects morphology into the embedding itself.

- **Luong, Socher & Manning (2013) [S]** — recursive NN over morphemic decomposition; rare/complex
  words composed rather than memorized.
- **Kim et al. (2016), *Character-Aware Neural LMs* [S]** — char-CNN + highway + word LSTM, no word
  embeddings and no morphological supervision. Reported parity with SOTA on English PTB at 60% fewer
  parameters, and **outperforming both word-level and morpheme-level LSTM baselines on
  morphologically rich languages**. The most deployable analyzer-free morph channel.
- **Bojanowski et al. (2017), fastText [S]** — word = bag of character n-grams; compositional,
  OOV-free, inherently clusters inflected forms of one root. The mandatory cheap baseline.
- **Cotterell & Schütze (2015) [S]** — semi-supervised log-bilinear model with a term over
  morphological annotation, pulling morphologically similar words together. The ancestor of a
  morphological-tag auxiliary loss.
- **Hofmann, Pierrehumbert & Schütze (2021), *Superbizarre Is Not Superb* [S]** — DelBERT feeds
  derivational (morpheme-respecting) segmentation instead of WordPiece and frames PLMs as serial
  dual-route models: frequent complex words stored whole, novel ones composed from morphemes. This
  is the theoretical justification for a two-channel split, and it suggests a routing/gating design
  rather than plain concatenation.
- **Akdemir, Shibuya & Güngör (2020) [A]** — hierarchical multi-task learning over dependency parsing
  and NER for Turkish with subword contextual embeddings carrying morphology implicitly rather than
  via an external analyzer. Reported +18.86% F1 on dependency parsing, +4.61% on NER.

### Probing: find out what the encoder already knows before adding a channel

- **LINSPECTOR (Şahin et al., 2020) [S]** ([2020.cl-2.4](https://aclanthology.org/2020.cl-2.4/)) —
  multilingual morpho-syntactic/morpho-semantic probing suite. Reported that **Case, POS, Person,
  Tense and TagCount probes correlate relatively highly with downstream performance in Finnish,
  Turkish, German and Russian**. Ships Turkish tasks and data.
- **Morphosyntactic probing of multilingual BERT [S]** — morphological information is layer-wise
  recoverable from mBERT-family representations; no numbers verified, year/authors unconfirmed.

> **Takeaway.** Run layerwise morphological probes on the chosen semantic encoder *before*
> committing to a second encoder. If case/tense/person/negation are already linearly decodable from
> mid layers, the cheaper design is a morphology-aware pooling or objective change, and the probes
> also tell you which layers LoRA should target.

---

## 4. Retrieval for morphologically rich languages

### 4.1 The classical Turkish IR result that constrains all claims

**Can et al. (2008), *Information Retrieval on Turkish Texts* [S]** — 408,305 documents, 72 ad hoc
queries (Milliyet). Compared no stemming, fixed-length prefix truncation (notably **first-5**),
corpus-statistics-driven truncation, and an elaborate lemmatizer-based stemmer. The repeated headline
claim: **all three give similar retrieval effectiveness**, with 5-character truncation an effective
indexing choice. Companion work: Ekmekçioğlu et al. (1996) **[S]** on stemming vs n-gram conflation;
a probabilistic lexicon-free Turkish stemmer (2003) **[S]** reporting 95.8% test success;
Ozturkmenoglu & Alpkocak **[S]** comparing lemmatization approaches.

This sets a hard credibility bar: **first-5-character-truncated BM25 is nearly free to compute**, and
if it matches a fine-tuned dense retriever on general Turkish retrieval, the fine-tuning has not
earned its cost. On a morphological minimal-pair set it should, by construction, score near chance —
which makes it the ideal artifact detector.

### 4.2 Modern Turkish retrieval

- **TR-MTEB (Baysan & Güngör, 2025) [A]**
  ([2025.findings-emnlp.471](https://aclanthology.org/2025.findings-emnlp.471/)) — 26 datasets,
  six task families; releases a 34.2M weakly-supervised Turkish sentence-pair corpus and two
  embedding models trained by contrastive pretraining + supervised fine-tuning.
- **TurkColBERT (Ezerceli et al., 2025) [F]** ([arXiv 2511.16528](https://arxiv.org/abs/2511.16528))
  — two-stage adaptation: fine-tune English/multilingual encoders on Turkish NLI/STS, then convert to
  ColBERT-style late-interaction retrievers with PyLate on MS MARCO-TR. Ten models over five Turkish
  BEIR ports. Reported: `colbert-hash-nano-tr` (1.0M params) retains **>71% of the average mAP of
  `turkish-e5-large` (600M)**; `ColmmBERT-base-TR` up to **+13.8% mAP** on domain-specific tasks;
  MUVERA+Rerank +1.7% relative mAP over PLAID at 3.33× the speed; 0.54 ms query times.
  **Important caveat:** TurkColBERT motivates late interaction with Turkish agglutination but runs no
  morphology-specific analysis — no fertility study, no suffix probe, no per-phenomenon breakdown.
  *"Late interaction preserves Turkish suffix signal"* is currently an architectural intuition, not a
  measured result.
- **TurkEmbed4Retrieval (Ezerceli et al., 2025) [S]** — GTE-multilingual-base with early layers
  frozen, fine-tuned on MS MARCO-TR with Matryoshka + a tailored MNRL.

### 4.3 The Amharic evidence — the cleanest analogue

- **Mekonnen, Alemneh & de Rijke (2025) [A]** — RoBERTa-Base-Amharic-Embed (110M) achieves
  **+17.6% relative MRR@10 and +9.86% Recall@10** over Arctic Embed 2.0 (568M); a 42M variant stays
  competitive at >13× smaller; the ColBERT late-interaction variant attains the highest MRR@10.
- **The Multilingual Curse at the Retrieval Layer (Alemneh et al., 2026) [F]** — 68K query-passage
  Amharic benchmark across four retrieval paradigms. The best zero-shot multilingual retriever trails
  the best monolingual model by **~23% relative MRR@10 (0.653 vs 0.803)**; Amharic-specific
  fine-tuning gives 32–60% relative gains, yet fine-tuned multilingual models **still stay below
  monolingual ones despite ~2.5× more parameters**.

> **Takeaway.** For a morphologically complex language, a small monolingual encoder beats a large
> multilingual one, and fine-tuning does not close the gap. That is a direct argument for the small
> Turkish-specific morphological channel.

### 4.4 Pooling — the mechanism behind the project's own hypothesis

**Gao, Xu, Mei & Metaxas (2026), *Pooling and Semantic Shift* [F]**
([arXiv 2603.21437](https://arxiv.org/html/2603.21437)) — two theorems with empirical validation on
ArXiv and a literary corpus using bge-large. **Theorem 1 (semantic dilution):** the discrepancy
between a pooled text embedding and its constituent sentences strictly increases with semantic
diversity. **Theorem 2 (pooling-induced collapse):** contextual aggregation strictly reduces Mean
Pairwise Distance of the vector space.

This is the formal backing for *"does mean pooling wash out a single suffix in a long passage?"* —
but note carefully what it does and does not say. It proves dilution grows with *pooling scale and
semantic diversity*. **It does not measure morphology.** No paper found runs the passage-length ×
morphological-accuracy experiment in any language. That is an open, publishable slot, and it is why
the v2.0 dataset makes passage length a first-class stratification axis.

### 4.5 Two complementary intrinsic diagnostics

Morpheus's own reported profile is instructive: root-family MAP 0.85 and same-root ROC-AUC 1.00 —
outstanding at clustering forms of one root — while remaining weak on context-dependent semantics.
A morphological channel must do **both**: cluster forms of the same root *and* separate minimal
suffix pairs. Measuring only the first lets a degenerate encoder look excellent — a pure stemmer aces
root-family MAP and fails every suffix contrast.

---

## 5. Fine-grained meaning-flipping contrasts: the closest published analogue

No morphological retrieval benchmark exists, but the **negation-in-IR** literature is structurally
the same problem: two documents on the same topic, one answers the query and one does not, and the
difference is a single small marker.

### 5.1 The benchmarks and the numbers

**NevIR (Weller et al., 2023) [A]** ([arXiv 2305.07614](https://arxiv.org/abs/2305.07614)) —
crowdsourced contrastive document pairs; a model is correct only if it ranks **both** queries
correctly, so **random = 25%**. Most IR models perform at or below random. Ordering:
cross-encoders > late-interaction > bi-encoders/sparse.

**Reproducing NevIR (2025) [F]** ([arXiv 2502.13506](https://arxiv.org/html/2502.13506)) — the
numbers, read off the page:

| Model | Pairwise acc. | | Model | Pairwise acc. |
|---|---|---|---|---|
| *random* | **25.0** | | MonoT5-small | 27.7 |
| DPR | 6.5 | | MonoT5-base | 34.9 |
| msmarco-bert-base-dot-v5 | 6.9 | | MonoT5-3B | 50.6 |
| all-mpnet-base-v2 | 8.1 | | Mistral-7B-Instruct-v0.3 (listwise) | 46.3 |
| multi-qa-mpnet-base-dot-v1 | 11.1 | | GPT-4o-mini (listwise) | 64.1 |

Two further findings from this paper matter operationally. **(a) Checkpoint selection on the
fine-grained metric alone catastrophically overfits**: the same model scores 0.06 MRR@10 on MS MARCO
under NevIR-only selection versus 0.20 under a trade-off criterion. **(b) Training on one contrast
type does not transfer**: bi-encoders and listwise rerankers improved only on the dataset they were
trained on; only the cross-encoder generalized.

**ExcluIR (2025) [A]** — exclusionary queries built on HotpotQA; 3,452 manually annotated eval
queries and 70,293 training queries. Same story: models struggle, training helps, human gap remains.

**CONDAQA (Ravichander et al., 2022) [S]** — 14,182 QA pairs, 200+ negation cues, **three edit types
per passage (paraphrase / scope change / polarity reversal)** forming contrast clusters, scored with
an all-or-nothing consistency metric. Best model 42% vs 81% human (snippet-sourced).

**Thunder-KoNUBench (Jung et al., 2026) [F]** — the closest *typological* analogue. Korean, four-way
multiple choice with **typed distractors** (standard negation / LOCAL negation / contradiction /
paraphrase). Type distribution set by corpus analysis: 29,476 sentences yielded 3,160 negative
instances (10.7% prevalence). Zero-shot over 47 models: Korean-specialized 62.8%, non-Korean 62.7% —
**language-specific pretraining bought essentially nothing** — against a 97.6% human baseline. Over
90% of cloze errors concentrated on the LOCAL-negation distractor.

### 5.2 The fixes that worked

- **NegCLIP / ARO (Yuksekgonul et al., 2023) [S]** — the key recipe: place the minimally-perturbed
  counterfactual **in the same batch** as the positive, so the InfoNCE denominator is dominated by a
  document differing by one edit. Reported >10% improvement on 11 of 16 cases without significant
  downstream loss.
- **CANNOT / NegBLEURT (Anschütz et al., 2023) [S]** — built ~77K contrast pairs with a **rule-based**
  sentence negation tool, not an LLM: the edit type is known by construction, minimality is
  guaranteed, and there is no generator style drift between positives and negatives.
- **Rezaei & Blanco (2025), *Making LMs Robust Against Negation* [S]** — Next Sentence Polarity
  Prediction: given only the first of two consecutive sentences, predict whether the next contains a
  negation cue. Self-supervised, no labels. Reported 1.8–9.1% gains on CondaQA.
- **InF-IR (2025) [S]** — 38K+ (instruction, query, document) triplets; hard negatives made by
  *poisoning* one component, then **validated by a strong reasoning model** that must confirm each is
  semantically plausible yet genuinely non-satisfying.
- **FollowIR (Weller et al., 2024) [A]** — training data synthesized with GPT-3.5-Turbo, FollowIR-7B
  fine-tuned from Mistral-7B-Instruct **with LoRA**; existing retrieval models treat instructions as
  keywords.

> **Takeaway.** The Turkish analogue of the CANNOT recipe is far more tractable than the English one:
> a morphological generator can deterministically flip negation, tense, aspect, case, possessive and
> privative `-sIz`. Turkish is *better* suited to rule-based counterfactual generation than English
> is — which is a project-level argument, not just an implementation detail.

---

## 6. Building and validating a set that isolates one feature

### 6.1 The construction protocols

**BLiMP (Warstadt et al., 2020) [F]** — 67 paradigms × 1000 pairs over 12 phenomena, fully synthetic
from linguist-crafted grammar templates, with lexical items drawn from a hand-annotated >3,000-item
vocabulary carrying morphological/syntactic/semantic features to enforce selectional restrictions
(11 verb subcategorization frames). Pairs matched for length. Human validation: 20 validators × 5
pairs per paradigm = 6,700 forced-choice judgments, gold by majority vote. **Aggregate human
agreement 96.4%, individual 88.6%**; a paradigm is retained only if a majority of validators agree
with the label.

**TurBLiMP (Başar, Padovani, Jumelet & Bisazza, 2025) [F]**
([2025.emnlp-main.834](https://aclanthology.org/2025.emnlp-main.834/)) — the direct Turkish
predecessor and the protocol worth copying. **Three stages:** (1) 10 items per phenomenon written
entirely by hand to fix the guidelines; (2) semi-automatic augmentation where masked BERTurk proposes
lexical replacements, **every one verified and adjusted manually**, to 100 validated pairs per
phenomenon; (3) fully automatic augmentation to 1000 once the schema is proven. 16 phenomena × 1000
pairs = 16,000. Human validation: 30 native speakers (17 linguistics students, 13 non-linguists),
7-point Likert, 216 sentences each. Finding: cutting-edge LLMs still struggle with phenomena easy for
humans and show different sensitivities to word order and morphological complexity than humans do.

Crucially, TurBLiMP deliberately used a **masked** LM plus a rule-based analyzer with human
verification — **not** an instruction-tuned generator.

**A Morphology-Aware Evaluation of Turkish Syntax in LLMs (Başar & Bisazza, 2026) [A]**
([2026.sigturk-1.9](https://aclanthology.org/2026.sigturk-1.9/)) — analyses how morpheme count,
subword count and sentence length relate to LM performance on Turkish minimal pairs. States that
surface factors have limited predictive power **but may act as a systematic source of bias**, that
morphological alignment corresponds with performance, and that **morpheme-level imbalances in a
benchmark may significantly influence evaluation results.** This is the direct mandate for a
confound-balance audit.

**Evaluating Morphological Compositional Generalization in LLMs (Ismayilzada et al., 2025) [A]**
([2025.naacl-long.59](https://aclanthology.org/2025.naacl-long.59/)) — treats morphemes as
compositional primitives and probes productivity (apply a known suffix to a novel root) and
systematicity. Evaluates instruction-tuned multilingual LLMs including GPT-4 and Gemini. Finding:
**LLMs struggle with morphological compositional generalization, particularly on novel roots, with
performance declining sharply as complexity increases**; they identify individual combinations above
chance but lack systematicity.

**CheckList (Ribeiro et al., 2020) [S]** — the capability × test-type matrix (MFT / INV / DIR). This
gives principled *pass conditions* for typed negatives: `morph_counterfactual` is a **DIR** item
(the gold must lose), `state_variant` is arguably **INV**, `partial_trap` is an **MFT**.

**Contrast sets (Gardner et al., 2020) [S]** and **counterfactually-augmented data (Kaushik et al.,
2020) [S]** — dataset authors, not crowdworkers, perturb test instances minimally in ways that change
the gold label; models drop substantially (reported up to ~25 points).

**MultiBLiMP 1.0 (Jumelet et al., 2026) [S]** — fully automated construction combining UD treebanks
with UniMorph inflection tables; 101 languages, >128,000 minimal pairs, no per-language grammar
engineering. Proof that treebank + inflection-table generation scales.

### 6.2 LLM-synthesized retrieval data

**InPars-v2 (Jeronymo et al., 2023) [A]**, **Promptagator (Dai et al., 2022) [S]**, **E5-mistral
(Wang et al., 2024) [S]**, **Gecko (2024) [S]**, **SWIM-IR (Thakur et al., 2024) [S]**. The
mechanisms that transfer:

- **Consistency / round-trip filtering** (Promptagator): train an initial retriever on the raw
  synthetic data, feed each query back, keep the item only if its intended gold returns at rank 1.
  Reported to help 8 of 11 datasets for ~2.5 points average. **This is the cheapest automated
  double-gold detector available.**
- **Relabeling rather than assuming** (Gecko): do not assume the seeding passage is the best
  positive — retrieve a candidate pool and have the LLM relabel which is genuinely gold and which are
  hard negatives.
- **Brainstorm-then-instantiate with explicit variation axes** (E5-mistral): generate a pool of task
  templates first, then instances conditioned on one template each, so surface form is not drawn
  from a single degenerate mode.
- **Summarize-then-ask** (SWIM-IR): give the model the passage plus a summary so it localizes the
  relevant span before asking.

### 6.3 The failure modes, and their detectors

**When Hard Negatives Hurt / CausalNeg (Zhang et al., 2026) [F]**
([arXiv 2606.01304v2](https://arxiv.org/html/2606.01304v2)) — the most important paper in this
section for a synthesis project. Diagnoses two failure modes of LLM-synthesized hard negatives:
**discriminative-agnostic generation** (measured: **24% of generated negatives form "pure clusters"**
isolated from other document types) and **source-dependent shortcuts** (the model learns to
discriminate by generation origin rather than relevance). Vanilla generation yields near-zero
retrieval gain. The proposed fix is chain-of-thought-guided counterfactual perturbation: decompose
*why* a document satisfies a query into explicit information requirements, then violate one
requirement at a time — plus Query-View Entropy Maximization to suppress the shortcut.

**SugarCrepe (Hsieh et al., 2023) [A]** — the diagnostic that matters: **"blind" models with no access
to the query beat SOTA models** on prior compositionality benchmarks, meaning previously reported
improvements were "hugely overestimated." The remedy is LLM-regenerated negatives plus adversarial
refinement that maximally reduces the remaining bias.

**Misleading Failures of Partial-input Baselines (2019) [S]** — the necessary caveat: a *failing*
partial-input baseline does not prove a dataset is artifact-free. Treat it as hypothesis testing, not
certification.

**Silencer (Yuan et al., 2025) [A]** — measures inflation of a model's score on benchmarks it
generated itself. Mitigation via multiple heterogeneous generators aggregated at sample and benchmark
level. Reported Pearson correlation with a human-annotated benchmark rising 0.655 → 0.833.

**LLM-judge surface-overlap bias** — LLM judges are documented to over-mark passages as relevant when
they share surface terms with the query. For this project that bias falls precisely on
`same_feature_wrong_content` and `partial_trap`, the two categories built to share surface terms.
An LLM judge therefore cannot be the final gate.

**Contamination** — the eval set is held out and will judge a LoRA fine-tuned model, so any synthetic
*training* data sharing prompts, seeds or exemplars with it is a contamination vector even without
literal string overlap. Generate train and test through separate pipelines and account for exemplars
explicitly.

---

## 7. Contrastive fine-tuning: what transfers to the LoRA stage

### 7.1 The recipes

**SimCSE (2021) [S]** — dropout-as-augmentation unsupervised; NLI entailment positives with
contradiction hard negatives supervised; alignment/uniformity framing.
**E5 (2022) [S]** and **multilingual-E5 (2024) [S]** — two-stage: (1) weakly-supervised contrastive
pretraining on a huge noisy pair corpus, InfoNCE with **in-batch negatives only**, batch 32,768 for
30k steps (snippet-sourced); (2) short supervised fine-tuning with mined hard negatives and
cross-encoder distillation. Origin of the `query:` / `passage:` convention.
**GTE (2023) [S]**, **BGE-M3 (2024) [S]** (dense + sparse + multi-vector in one model, self-knowledge
distilled from their ensemble), **NV-Embed (2024) [S]** (bidirectional attention, latent-attention
pooling).
**Qwen3-Embedding (2025) [F]** ([arXiv 2506.05176v2](https://arxiv.org/html/2506.05176v2)) — three
stages: ~150M synthetic pairs from Qwen3-32B; SFT on ~7M labeled + ~12M synthetic filtered to cosine
> 0.7; **slerp model merging** over SFT checkpoints. Extended InfoNCE with an explicit
**false-negative mask: m_ij = 0 when s_ij > s(q_i, d_i+) + 0.1**, or when the candidate is the
positive itself. (Mask rule, 0.1 margin, data sizes and 0.7 filter read directly off the page;
temperature and batch size were not stated in the section read.)

**e5-mistral-7b-instruct (2024) [S]** — LLM-generated (task, query, positive, hard negative) tuples
across ~100 languages, then **LoRA rank 16, <1k total steps**, no weak-supervision stage at all.

### 7.2 Negatives — where the real leverage is

**NV-Retriever (Moreira et al., 2024) [F]** ([arXiv 2407.15831v1](https://arxiv.org/html/2407.15831v1))
— the systematic ablation, read off the page. Best settings: **TopK-PercPos at 95% of the positive
score**, TopK-MarginPos at margin 0.05, TopK-Abs at 0.7, Top-K-shifted at N=10. On
e5-large-unsupervised across NQ/HotpotQA/FiQA: naive Top-K **0.5407** avg nDCG@10 vs TopK-PercPos
**0.5856**. Also: **mining-model quality dominates mining-method cleverness** —
e5-mistral-7b-instruct as miner gives 0.5810 vs BM25's 0.5002, while a 4-model ensemble adds almost
nothing (0.5825). Reported that on MS MARCO roughly 70% of the passages most similar to a query
should arguably be labelled positive.

**GISTEmbed (Solatorio, 2024) [A]** — a stronger *guide* model scores every in-batch pairing at
training time and dynamically masks items it judges too related, replacing the equal-utility
assumption of in-batch negatives. A one-line swap from `MultipleNegativesRankingLoss`.

**Conan-embedding (2024) [S]** — dynamic hard-negative re-mining *during* training as the model
improves, rather than fixed preprocessing.

**Negative Sampling Techniques in IR: A Survey (Wischounig et al., 2026) [A]** — six families:
in-batch, cross-batch, BM25/ANCE-mined, denoised/false-negative-filtered, curriculum/dynamic, and
LLM-generated. The survey explicitly frames negative count, temperature and false-negative rate as
**dataset-dependent calibration tasks**, committing to no universal thresholds.

> **A direct conflict this project must resolve deliberately.** Every published false-negative filter
> (NV-Retriever's 95%-of-positive rule, Qwen3's +0.1 margin mask) would **delete a morphological
> counterfactual**, because a one-suffix edit is lexically near-identical to the gold and will score
> above any such threshold. These filters are correct for *mined* negatives and catastrophic for
> *curated typed* negatives. The pipeline must tag curated negatives and hard-exempt them from
> positive-anchored filtering — and log how often the mask would have fired.

### 7.3 Loss and optimization mechanics

- **Understanding the Behaviour of Contrastive Loss (Wang & Liu, 2021) [S]** — InfoNCE is
  hardness-aware; temperature τ controls how sharply the penalty concentrates on the hardest
  negatives, with a uniformity–tolerance dilemma. **The two channels want opposite settings**: the
  morph channel must aggressively separate surface-similar/morphologically-different pairs (low τ);
  the semantic channel must tolerate paraphrase (higher τ).
- **GradCache (Gao et al., 2021) [S]** — decouples the contrastive backward pass from the encoder
  backward pass at near-constant memory; ~20% runtime overhead, 8×V100 → 1 GPU. This is what makes a
  serious batch size feasible on a single Colab A100/L4.
- **Matryoshka Representation Learning (Kusupati et al., 2022) [S]** — nested-prefix-valid embeddings
  at no inference cost. Lets a small morph vector (128–256 dims) be concatenated onto a full semantic
  vector, with the ratio tunable post-hoc.
- **AnglE / CoSENT [S]** — graded-similarity losses consuming `(text_a, text_b, float_score)`. Worth
  it only if graded targets are assigned per negative type (e.g. `state_variant` 0.6,
  `partial_trap` 0.4, `morph_counterfactual` 0.1).
- **Curriculum over negative difficulty** — natural ordering here: random in-batch → BM25/dense-mined
  topical → `same_feature_wrong_content` → `partial_trap` → `morph_counterfactual`. The existing
  typed taxonomy *is* a difficulty ladder.
- **NV-Embed's rule [S]:** disable in-batch negatives when mixing heterogeneous task types, because
  cross-task in-batch pairings are not meaningful negatives.
- **Anti-forgetting:** short runs, low LR, adapters over full fine-tuning, optional checkpoint
  merging — and **measure it**: report Δ nDCG@10 on the four trmteb BEIR sets alongside any
  morphological gain. A morph gain reported without the general-retrieval delta is not interpretable.
- **Prompt-prefix consistency:** whatever prefix the LoRA is trained with becomes load-bearing at
  eval time. `CLAUDE.md` already flags this for zero-shot comparison; it gets worse after tuning.

---

## 8. Design implications for this project

| # | Implication | Grounded in | Action |
|---|---|---|---|
| 1 | Do not let an LLM be the sole author of a morphological mutation | Ismayilzada 2025 [A]; TurBLiMP [F]; CANNOT [S] | LLM writes the semantics; a deterministic gate (vowel harmony, consonant assimilation, buffer consonants; optionally Zemberek/`zeyrek`) validates the suffix edit. Items failing are rejected, not repaired |
| 2 | Generate positive + all negatives in one call under shared style constraints | CausalNeg [F] (24% pure clusters) | Single-call generation; then a query-blind separability audit |
| 3 | Frame negative construction as requirement violation | CausalNeg [F] | Prompt asks the model to enumerate why the positive answers the query, then violate exactly one requirement per typed negative |
| 4 | Run a query-blind / partial-input audit before trusting the set | SugarCrepe [A]; Feng 2019 [S] | Check golds and hard negatives are not separable without the query (length, sentence count, position). Report it as hypothesis testing, not certification |
| 5 | Balance and report morpheme-count and length confounds | Başar & Bisazza 2026 [A] | Tolerance check on gold vs `morph_counterfactual`; per-feature imbalance table in the report |
| 6 | Use round-trip retrieval consistency as the double-gold detector | Promptagator [S]; Gecko [S] | Two dissimilar retrievers over each item's candidates; a hard negative beating the gold under both → quarantine |
| 7 | Reject anything a sparse/prefix-truncation baseline solves | Can et al. 2008 [S] | First-5 truncated BM25 must be near chance on `morph_counterfactual` pairs; if it ranks the gold first the item is lexical, not morphological |
| 8 | The LLM judge cannot be the last gate | LLM-judge overlap bias; Silencer [A] | Judge is one signal among the deterministic gates and the retriever ensemble; note the generator/judge family coupling as a stated limitation |
| 9 | Stratify training data across **every** feature | NevIR reproduction [F] (no cross-type transfer for bi-encoders) | ~60 target features with balanced quotas, not negation-heavy |
| 10 | Make passage length a first-class axis | Gao et al. 2026 [F]; project's own open question | short/medium/long quotas so the length × morphological-accuracy curve is runnable — no published equivalent exists |
| 11 | Export the typed taxonomy as a curriculum and as graded targets | Curriculum literature; CoSENT/AnglE [S] | `difficulty_rank` and graded score per negative in the training export |
| 12 | Guarantee the counterfactual is in-batch | NegCLIP [S] | Group negatives per query in the MNRL export so the trainer can place them in one batch |
| 13 | Hard-exempt curated typed negatives from positive-anchored filtering | NV-Retriever [F]; Qwen3 mask [F] | Tag curated vs mined; apply the 95%/+0.1 rules only to mined; log how often they would have fired |
| 14 | Per-channel temperature | Wang & Liu 2021 [S] | Low τ for the morph channel, higher τ for the semantic channel — an ablation worth claiming |
| 15 | Score with pairwise accuracy per negative type, not only nDCG@10 | NevIR [A/F] | Export a NevIR-style paired view; state the random baseline explicitly; report below-random results as findings |
| 16 | Select checkpoints jointly, never on the morph metric alone | NevIR reproduction [F] (0.06 vs 0.20 MRR@10) | Report Δ on the four trmteb BEIR sets alongside every morph gain |
| 17 | Add cheap intrinsic probes on both axes | Morpheus [A]; LINSPECTOR [S] | Root-family MAP *and* suffix-contrast accuracy — a stemmer aces the first and fails the second |
| 18 | Try tokenizer surgery as an ablation arm before assuming a second encoder is needed | Bayram et al. 2026 [F] ($5–20, ~4 GPU-h) | If surgery alone closes the morph gap, the second channel is not earned |
| 19 | Account for few-shot exemplar leakage | Contamination hygiene | Record which v1.3.1 items were used as exemplars; exclude them when reporting test scores; ban their content words |
| 20 | Treat translated Turkish-BEIR ports as the secondary layer | translationese; project's own setup | Native morphological set is primary evidence; say so in limitations |

---

## 9. Open gaps (opportunities)

1. **No two-channel (semantic + morphological) dual encoder for retrieval exists in any agglutinative
   language.** No published fusion recipe to copy — concatenation vs gating vs score interpolation vs
   learned routing is an open design question.
2. **No minimal-pair morphological retrieval benchmark for Turkish.** This project's artifact appears
   to be first of its kind; no external baseline exists to calibrate against.
3. **"Late interaction preserves Turkish suffix signal" is untested.** TurkColBERT asserts it
   architecturally and never measures it. Running the project's morph set against a Turkish ColBERT
   would settle it.
4. **Passage-length × morphological-sensitivity has never been measured.** Pooling dilution is proven
   in general (Gao et al. 2026) but never for morphology, in any language.
5. **No validation that a generative LLM can produce well-formed Turkish morphological
   counterfactuals at scale.** TurBLiMP chose a masked LM plus manual verification precisely to avoid
   this question. The v2.0 pipeline's morphology-gate rejection rate will be the first datapoint —
   worth reporting regardless of which way it comes out.
6. **The Turkish tokenization evidence is contradictory** (Toraman/MorphScore vs
   Morpheus/MorphBPE/Tokens-with-Meaning) and remains unresolved under matched conditions.
7. **Affixal (morphological) negation in retrieval is essentially unstudied.** Truong et al. (2024)
   covers affixal negation for LLM *detection* only, never for ranking.

---

## 10. Provenance and limitations of this review

Compiled from a six-topic parallel literature sweep (Turkish morphology in NLP; morphology-aware
representations; IR for morphologically rich languages; fine-grained/negation failures in dense
retrieval; contrastive fine-tuning recipes; eval-set design and LLM synthesis). Each topic ran a
broad search pass with 6–10 distinct queries and fetched primary sources where possible.

**The planned deep-read verification, adversarial fact-check, and cross-topic synthesis passes did
not run** — they failed on a session limit. This document is therefore written from the first-pass
survey output. The practical consequence is that **[S]**-tagged entries have not had a second pass,
and several **[A]** entries would likely be upgraded or corrected by one. Specific known weaknesses:

- Several key Turkish papers are 2026 preprints (Morpheus, HeceTokenizer, Altinok, Bayram et al.,
  Başar & Bisazza) reachable only as abstracts. Their numbers are authors' claims.
- One source (*Learning Robust Negation Text Representations*, arXiv 2507.12782) is included without
  numbers because the extracted results table could not be corroborated and appeared possibly
  model-generated.
- Disambiguation accuracies for Sak et al. (2008), the Ozen & Can F-measure, the CONDAQA human/model
  gap, the Promptagator and Gecko numbers, and the SimCSE/E5/GTE headline figures are all
  snippet-sourced.
- No claim here about *this project's* own results is made; everything is prior work.

Raw survey output (81 unique papers, 82 techniques, per-paper verification tags) is retained
alongside this document for anyone who wants to re-verify a specific entry.
