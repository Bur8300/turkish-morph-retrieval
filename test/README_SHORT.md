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
600 dengeli slot → iki LLM generator (300 + 300) → deterministic QC → blind LLM judge
→ kalan slot için taze replacement → duplicate/leakage kontrolü → freeze
```

Bir family kontrolden kalırsa fenomeni, split'i, generator'ı ve uzunluk kotası değişmez; yalnız
örnek yeniden üretilir. Üç refill turu yetmezse aynı komut yeniden çalıştırıldığında kaldığı yerden
devam eder. Amaç fazladan 1.800 örnek yazmak değil, tam 600 kabul edilmiş family elde etmektir.

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

Paper'ın iki ana sonucu birlikte sunulur: kontrollü contrast retrieval morfolojik ayrımı,
full-corpus retrieval ise aynı gold'u büyük ortak corpus içinde bulma başarısını ölçer. İkincisi
yalnız tanısal bir tablo değildir.

Komutlar:

```bash
python3 -m test self-test
python3 -m test plan --run-id test_v34

export OPENROUTER_API_KEY="..."
export TEST_GENERATOR_MODEL_A="provider-a/model-a"
export TEST_GENERATOR_MODEL_B="provider-b/model-b"
export TEST_JUDGE_MODEL="provider-c/model-c"
python3 -m test generate --run-id test_v34
python3 -m test finalize --run-id test_v34
```

Detaylar: [`README.md`](README.md). Final Colab:
[`notebooks/morph_baseline_eval_600_colab.ipynb`](notebooks/morph_baseline_eval_600_colab.ipynb).
