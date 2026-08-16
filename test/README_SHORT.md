# Test benchmark — kısa sürüm

- 600 family: 100 development + 500 final test.
- Her family: 1 query + 1 gold + 8 hard negative + 2 easy negative.
- 150 strict minimal pair.
- İki generator: 300 + 300; ayrı model ailesinden bir blind LLM judge.
- Query: `%75` 1 cümle, `%25` 2 cümle.
- Pasaj: `%30/%30/%30/%10` oranında 1/2/3/4 cümle.
- 65 fenomen, 6 macro grup; morph-hard ve semantic-hard ayrı raporlanır.

Tek akış:

```text
1.800 slot → iki LLM generator → deterministic QC → blind LLM judge
→ otomatik 100/500 seçim → duplicate/leakage kontrolü → freeze
```

Lexical artefakt kapısı gold ile hard overlap'ını dengeler, en az dört içerik-koruyan hard ister
ve word-overlap/char-3gram/BM25 sonuçlarını tie-aware hesaplar.

Qrels family oluşturulurken hazırdır:

```text
gold candidate = 1
diğer 10 generated negative = 0
```

Kontrollü 11-aday metrikleri `Recall@1/3`, MRR@10 ve nDCG@10'dur. Full-corpus retrieval, her
sealed query için farklı semantic frame'lerdeki 5.500 belgenin tamamını sıralar ve
`Recall@1/3/10/50`, MRR@10, nDCG@10 raporlar.

Komutlar:

```bash
python3 -m test self-test
python3 -m test plan --run-id test_v33

export OPENROUTER_API_KEY="..."
export TEST_GENERATOR_MODEL_A="provider-a/model-a"
export TEST_GENERATOR_MODEL_B="provider-b/model-b"
export TEST_JUDGE_MODEL="provider-c/model-c"
python3 -m test generate --run-id test_v33
python3 -m test finalize --run-id test_v33
```

Detaylar: [`README.md`](README.md). Final Colab:
[`notebooks/morph_baseline_eval_600_colab.ipynb`](notebooks/morph_baseline_eval_600_colab.ipynb).
