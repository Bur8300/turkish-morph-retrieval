# Test benchmark — kısa sürüm

- 600 family: 100 development + 500 final test.
- Her family: 1 query + 1 gold + 8 hard negative + 2 easy negative.
- Easy'ler aynı domain/ortamda farklı olayı anlatan `same_domain_off_intent` adaylardır; rastgele
  konu dışı değildir ve başka bir family'nin gold'uyla exact/fuzzy çakışamaz.
- Family modları: 150 strict minimal + 270 controlled diverse + 180 natural retrieval.
- Query anlatımı: 300 morph-explicit + 300 semantic-paraphrase.
- Query–gold lexical bandı: 180 high + 240 medium + 180 low.
- İki generator: 300 + 300; ayrı model ailesinden etiket/konum-kör, feature-aware bir LLM judge.
- Query: `%75` 1 cümle, `%25` 2 cümle.
- Pasaj: `%30/%30/%30/%10` oranında 1/2/3/4 cümle.
- 71 fenomen, 6 macro grup; morph-hard ve semantic-hard ayrı raporlanır.
- Yeni altılı: `COP.NEG`, `COP.TAM`, `Q.PART.SCOPE`, `NMLZ.MA_VS_DIK`,
  `REL.GEN.POSS`, `ANAPHOR.AGR`.

Tek akış:

```text
600 dengeli slot → iki LLM generator (300 + 300) → deterministic QC → blind LLM judge
→ kalan slot için taze replacement → duplicate/leakage kontrolü → freeze
```

Bir family kontrolden kalırsa fenomeni, split'i, generator'ı ve uzunluk kotası değişmez; yalnız
örnek yeniden üretilir. Üç refill turu yetmezse aynı komut yeniden çalıştırıldığında kaldığı yerden
devam eder. Amaç fazladan 1.800 örnek yazmak değil, tam 600 kabul edilmiş family elde etmektir.

Çoklu generator koordinasyonu `dataset_memory.sqlite3` üzerinden yapılır. Registry slotları atomik
rezerve eder; kabul edilen fenomen/lemma/anlatı metadata'sından aggregate coverage ve kaçınılacak
etiketler üretir. Önceki test cümleleri generator promptuna verilmez. Ayrıntı:
[`DATASET_MEMORY.md`](DATASET_MEMORY.md).

Strict modda gold–`hard_01` yalnız hedef biçimde ayrışır. Controlled modda gold ve hard farklı
doğal sözdizimi kullanabilir. Natural modda query, gold ve negatifler bağımsız yazılabilir; gold
bilgi ihtiyacını karşılayan tek passage'dır. Lexical kapılar moda göre 4/3/2 içerik-koruyan hard
ister ve word-overlap/char-3gram/BM25 sonuçlarını tie-aware hesaplar.

LLM küçük bir JSON üretir: query, ortak context, kritik lemma/sözcük ve aday başına yalnız
`candidate_slot + critical_sentence + critical_word`. Rol, subtype, morph relation, qrels,
edit script ve kimlikler Python tarafından eklenir.

Judge tek gold'u, query'yi yanlışlıkla destekleyen negatifleri, adayların iç tutarlılığını,
doğallığı ve hedef biçimbilimi kontrol eder; başarısız family aynı kotada yeniden üretilir.

Pilot provenance: `v36` Codex CLI / `gpt-5.6-sol`; `v37` deneysel Google /
`gemini-2.5-flash` üretimidir. İkisinde de independent judge çalışmadığından paper sonucu olarak
raporlanmaz.

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
python3 -m test plan --run-id test_v36
python3 -m test memory-report --run-id test_v36

export OPENROUTER_API_KEY="..."
export TEST_GENERATOR_MODEL_A="provider-a/model-a"
export TEST_GENERATOR_MODEL_B="provider-b/model-b"
export TEST_JUDGE_MODEL="provider-c/model-c"
python3 -m test generate --run-id test_v36
python3 -m test finalize --run-id test_v36
```

Detaylar: [`README.md`](README.md). Final Colab:
[`notebooks/morph_baseline_eval_600_colab.ipynb`](notebooks/morph_baseline_eval_600_colab.ipynb).
