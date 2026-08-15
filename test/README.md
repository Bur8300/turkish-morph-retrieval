# Turkish Morph Retrieval Test — v3.2.1

Bu dizin Türkçe encoder'ların küçük fakat anlam değiştiren morfolojik farkları ayırt edip
etmediğini ölçen benchmark'ı üretir ve değerlendirir. Train sistemi [`../train/`](../train/)
altında ayrıdır; test üretiminde eski JSON cümleleri few-shot örneği olarak kullanılmaz.

## Veri tasarımı

| Özellik | Development | Final test | Toplam |
|---|---:|---:|---:|
| Contrast family/query | 100 | 500 | 600 |
| Aday/family | 11 | 11 | 11 |
| Aday pasaj | 1.100 | 5.500 | 6.600 |
| Strict minimal pair | 25 | 125 | 150 |
| Generator A/B | 50/50 | 250/250 | 300/300 |

Her family tam olarak şunları içerir:

- 1 query
- 1 positive/gold
- 8 hard negative
- 2 easy negative
- Otomatik binary qrels: gold `1`, diğer 10 aday `0`

Uzunluk dağılımı:

- Query: `%75` bir cümle, `%25` iki cümle.
- Pasaj: `%30/%30/%30/%10` oranında 1/2/3/4 cümle.
- Ortak bağlam cümleleri family içindeki 11 adayda aynıdır.
- Yalnız kritik cümle değişir; kritik cümlenin pasaj konumu dengelenir.

Generalizasyon dağılımı:

- `%40 standard`
- `%20 lemma_holdout`
- `%20 template_holdout`
- `%20 composition_holdout`
- Yaklaşık `%20` ek `domain_shift` etiketi

`development` model/ayar seçimi içindir. `sealed_test`, kararlar tamamlandıktan sonra yalnız final
sonuç için kullanılır; kapalı tutulması gereken şey veri metni değil, model seçerken final gold
sonuçlarına tekrar tekrar bakmamaktır.

## Tek üretim akışı

```text
config + taxonomy
        ↓
1.800 deterministic slot
        ↓
iki farklı LLM generator (900 + 900)
        ↓
strict JSON + deterministic QC
        ↓
generator'lardan farklı model ailesinden blind LLM judge
        ↓
geçen kayıtlardan dengeli 100 dev + 500 final seçim
        ↓
duplicate/leakage/artefakt kontrolü + otomatik qrels + freeze
```

İki generator ve judge üç farklı model ailesinden olmalıdır. Legacy train Gemini ile üretildiği
için `google/*` test generator/judge rollerinde varsayılan olarak yasaktır. Model, prompt, config,
request hash, kullanım ve git commit bilgileri manifestte saklanır.

## Strict minimal pairs

600 family'sinin 150'sinde positive ile `hard_01`:

- aynı kritik lemmayı,
- aynı sözdizimsel şablonu,
- aynı token sırasını ve olayı korur,
- yalnız hedef eki veya ek zincirini taşıyan kritik sözcükte ayrışır.

`edit_script`, positive biçimi, minimal-negative biçimi, değişen feature ve korunan alanları
kaydeder. Validator iki kritik cümlenin sözcük iskeletini otomatik karşılaştırır.

## Fenomenler ve hard negatifler

Kod 6 macro grup altında 65 hedef taşır: 51 single ve 14 composition/chain.

- Hâl, konum ve yön
- Çoğul, iyelik ve fiilde kişi-sayı uyumu (`V.AGR`)
- Zaman, görünüş, kip ve kanıtsallık
- Olumsuzluk, çatı ve valency
- Ulaç, sıfat-fiil, türetim ve allomorph
- Ek-zinciri composition

Her family'deki altı çekirdek hard:

1. `minimal_morph_negative`
2. `same_lemma_wrong_inflection`
3. `related_feature_negative`
4. `same_morph_wrong_content`
5. `state_participant_time_trap`
6. `close_paraphrase_wrong_meaning`

Son iki hard fenomene göre seçilir:

- Hâl/fiil uyumu/çatı: `argument_role_reversal`, `morph_distractor`
- Ad çoğulluğu/iyelik: `noun_possessor_number_trap`, `morph_distractor`
- TAM/türetim: `scope_attachment_trap`, `morph_distractor`
- Composition: `partial_chain_negative`, `scope_attachment_trap`
- Allomorph: `allomorph_form_function_trap`, `morph_distractor`

Geçerli allomorph negative değildir. Örneğin `-da/-de/-ta/-te` aynı bulunma işlevinin yüzey
biçimleri olabilir; `şubesinde` ile `şubesinden` ise LOC/ABL anlam karşıtlığıdır.

## Kayıt ve qrels yapısı

Sadeleştirilmiş internal family:

```json
{
  "family_id": "family_raw_00042_xxx",
  "split": "sealed_test",
  "query": "Ekip raporu zamanında tamamlamadı.",
  "gold_id": "family_raw_00042_xxx_c03",
  "qrels": {
    "family_raw_00042_xxx_c01": 0,
    "family_raw_00042_xxx_c02": 0,
    "family_raw_00042_xxx_c03": 1
  },
  "candidates": [
    {
      "id": "family_raw_00042_xxx_c03",
      "role": "positive",
      "relevance": 1,
      "text": "Ekip raporu vaktinde bitirmedi."
    }
  ]
}
```

Qrels, retrieval cevap anahtarıdır: `query_id + candidate_id + relevance`.

- `1`: query'yi karşılayan tek gold.
- `0`: aynı family için üretilmiş ve LLM judge tarafından yanlış olduğu doğrulanmış negatif.

Bu qrels yalnız query'nin kendi 11 adayını etiketler. Başka family'lerin belgeleri otomatik `0`
sayılmaz; tesadüfen ilgili olabilirler. Bu nedenle ortak 5.500-belge gold sırası notebook'ta
korunur fakat yalnız tanısal stres testi olarak raporlanır. Ana paper sonucu kontrollü 11-aday
contrast deneyidir.

## Otomatik kalite kontrolleri

Family düzeyi:

- Tam `1 positive + 8 hard + 2 easy`
- Eksiksiz ve benzersiz candidate slot/ID'leri
- Query/pasaj için planlanan cümle sayısı
- Kritik sözcüğün gerçekten metinde bulunması
- Strict minimal-pair iskeleti ve `edit_script` uyumu
- Allomorph/function ayrımı
- Candidate uzunluk dengesi ve gold-length bias
- Tek ve doğru gold qrels
- Blind LLM judge ile benzersiz positive, doğallık, morfoloji ve subtype kontrolü

Corpus/freeze düzeyi:

- Exact ve fuzzy query tekrarları
- Cross-family candidate tekrarları
- Cross-family query–candidate yakın kopyaları
- Aşırı kullanılan soyut cümle şablonları
- Generator'a göre tekrarlanan başlangıç kalıpları
- Lemma/template/composition/domain leakage
- Candidate ve kritik-cümle konum bias'ı
- Train üretildikten sonra exact/fuzzy train–test leakage

Opsiyonel [Stanza](https://stanfordnlp.github.io/stanza/pipeline.html) audit'i kritik lemma ve UD
`UFeats` bilgisini kontrol eder. Bu gerçek morfem segmentasyonu değildir; audit uyarıları otomatik
gold değiştirmez.

## Evaluation

Ana metrikler:

- `Recall@1/5/10`, `MRR@10`, `nDCG@10`, `MAP@10`
- `hard_only_recall@1/5`, `hard_only_mrr@10`, `hard_only_ndcg@10`
- `pairwise_hard_accuracy`
- `pairwise_morph_hard_accuracy`
- `pairwise_semantic_hard_accuracy`
- `all_hard_family_consistency`
- `minimal_margin`, `hardest_hard_margin`, `hardest_negative_margin`

Artefakt kontrolleri:

- Longest candidate / most tokens / candidate position
- Character 3-gram / word overlap / BM25
- Query'siz candidate-only char-TFIDF
- Kritik sözcük silme
- `prefix5` suffix-reduction kontrolü; gerçek lemma/kök analizi değildir

İstatistikler query-level bootstrap `%95 CI`, paired bootstrap, approximate randomization,
McNemar, Holm düzeltmesi ve slice sonuçlarını içerir. Tekil 65 fenomen küçük örnekli tanısal
tablodur; ana rapor macro/layer/objective ve morph-hard/semantic-hard düzeyindedir.

## Komutlar

```bash
# API'siz regresyon testi ve plan
python3 -m test self-test
python3 -m test plan --run-id test_v32

# İki generator + bağımsız blind judge
export OPENROUTER_API_KEY="..."
export TEST_GENERATOR_MODEL_A="provider-a/model-a"
export TEST_GENERATOR_MODEL_B="provider-b/model-b"
export TEST_JUDGE_MODEL="provider-c/model-c"
python3 -m test generate --run-id test_v32

# Geçen kayıtlardan otomatik 100/500 freeze + qrels export
python3 -m test finalize --run-id test_v32

# Train üretildikten sonra leakage audit
python3 -m test audit-leakage --test TEST.json --train TRAIN.json

# Opsiyonel lemma/UFeats audit
python3 -m test morph-audit --input TEST.json --output morph_audit.json --download-model

# API key'siz yalnız preview
python3 -m test preview-codex --run-id sol_preview_20_v32 --count 20 \
  --batch-size 10 --model gpt-5.6-sol --reasoning-effort medium
```

Codex preview deterministic QC ve otomatik qrels içerir fakat bağımsız LLM judge çalıştırmaz;
paper verisine dondurulmaz.

## Ana dosyalar

| Dosya | Görev |
|---|---|
| `config.json` | Sayılar, dağılımlar, eşikler ve modeller |
| `taxonomy.py` | Fenomen ve hard-negative kataloğu |
| `planner.py` | 1.800 dengeli slot ve iki-generator ataması |
| `schema.py`, `prompts.py` | Strict üretim/judge sözleşmesi |
| `pipeline.py` | Generate → QC → LLM judge → checkpoint |
| `validators.py` | Family/corpus/duplicate/leakage kontrolleri ve qrels |
| `selection.py` | Otomatik 100/500 dengeli seçim |
| `exports.py` | Freeze, blind/internal JSON, BEIR ve qrels |
| `morphology.py` | Opsiyonel Stanza audit'i |
| `evaluation.py` | Metrikler, baseline, ablation ve istatistik |
| `notebooks/morph_baseline_eval_preview20_colab.ipynb` | 20-family hızlı test |
| `notebooks/morph_baseline_eval_600_colab.ipynb` | 100 dev + 500 final paper değerlendirmesi |

Freeze çıktıları:

```text
test/runs/<run_id>/
├── plan.json
├── accepted.jsonl
├── rejected.jsonl
├── generation_report.json
├── release/
│   ├── morph_dev_v3.2.1.json
│   ├── morph_test_blind_v3.2.1.json
│   ├── artifact_audit.json
│   └── freeze_manifest.json
└── private/
    ├── morph_test_internal_v3.2.1.json
    ├── private_qrels.jsonl
    ├── beir_test_qrels.tsv
    └── train_exclusion_holdouts.json
```
