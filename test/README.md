# Turkish Morph Retrieval Test Pipeline — ayrıntılı açıklama

Bu dizin, Türkçe retrieval modellerinin eklerin taşıdığı anlamı gerçekten ayırt edip etmediğini
ölçen, insan denetimli test benchmark'ını üretir. Legacy train üretimi ve eski JSON verileri
[`../train/`](../train/) altında kalır. Test generator'ı eski test cümlelerini few-shot örneği
olarak kullanmaz.

Kısa kullanım ve veri özeti için [`README_SHORT.md`](README_SHORT.md) dosyasına bakın.

## 1. Benchmark neyi ölçüyor?

Her **contrast family** tek bir query ve aynı bağlamda yazılmış 11 aday pasaj içerir:

- 1 positive: query'nin anlamını ve hedef morfolojik özelliği doğru karşılar.
- 8 hard negative: konu ve sözcük bakımından query'ye yakındır; küçük ama anlamlı bir
  morfolojik/semantik fark nedeniyle yanlıştır.
- 2 easy negative: açıkça ilgisizdir; temel retrieval davranışının da ölçülmesini sağlar.

Ana iddia, modelin yalnız sözcük benzerliğine değil Türkçe morfolojik anlama duyarlı olup
olmadığıdır. Benchmark tek başına genel amaçlı veya long-context retrieval benchmark'ı değildir.
Genel retrieval yeteneği ayrıca pooled full-corpus ve harici Turkish-BEIR/TR-MTEB deneyleriyle
raporlanmalıdır.

## 2. Dondurulmuş veri tasarımı

### Split ve aday sayıları

| Bölüm | Development | Final/sealed test | Toplam |
|---|---:|---:|---:|
| Contrast family/query | 100 | 500 | 600 |
| Aday/family | 11 | 11 | 11 |
| Aday pasaj | 1.100 | 5.500 | 6.600 |

Her family tam olarak `1 positive + 8 hard_negative + 2 easy_negative` içerir.

### Query ve pasaj uzunluğu

Query ile aday pasaj uzunluğu ayrı planlanır:

| Uzunluk | Development | Final test | Toplam |
|---|---:|---:|---:|
| 1 cümle query | 75 | 375 | 450 |
| 2 cümle query | 25 | 125 | 150 |
| 1 cümle pasaj | 30 | 150 | 180 |
| 2 cümle pasaj | 30 | 150 | 180 |
| 3 cümle pasaj | 30 | 150 | 180 |
| 4 cümle pasaj | 10 | 50 | 60 |

Query iki cümle olduğunda ikinci cümle yeni bir bilgi ihtiyacı açmaz; ilk önermeyi doğal biçimde
sınırlar. Dört cümlelik pasaj dilimi morfolojik sinyalin bağlam içinde seyrelmesine karşı bir
robustness dilimidir; “long-context retrieval” iddiası için yeterli değildir.

### Generalizasyon dağılımı

| Grup | Development | Final test | Toplam | Amaç |
|---|---:|---:|---:|---|
| `standard` | 40 | 200 | 240 | Normal test dağılımı |
| `lemma_holdout` | 20 | 100 | 120 | Kritik lemma development/train'den ayrılır |
| `template_holdout` | 20 | 100 | 120 | Soyut cümle kalıbı ayrılır |
| `composition_holdout` | 20 | 100 | 120 | Ekler tek tek görülür, tam zincir testte tutulur |

`domain_shift`, bu dört gruba rakip beşinci bir split değildir. Yaklaşık `%20` family'ye eklenen
ortogonal bir analiz etiketidir. Domain/register birleşimlerinin sonuçları ayrıca raporlanabilir.

## 3. Development ve sealed test neden ayrı?

`development` model seçmek, prompt/QC eşiklerini ayarlamak, değerlendirme kodunu hata ayıklamak ve
ablation kararlarını vermek içindir. `sealed_test`, bu kararlar bittikten sonra yalnız nihai paper
sonuçlarını ölçmek içindir.

Sealed şu anlama gelmez: “Hiçbir insan veriyi görmeyecek.” Beş insan reviewer, family'leri gold ve
rol etiketlerini görmeden kontrol edebilir. Sealed'ın anlamı şudur:

- Model/prompt/eşik geliştiren kişi final-test gold sonuçlarına bakarak karar vermez.
- İnsan düzeltmeleri tamamlanınca 500 family hash/manifest ile dondurulur.
- Final test üzerinde tekrar tekrar model veya prompt seçilmez.
- Yeni karar gerekiyorsa 100-family development kullanılır.

Tüm 600 family'yi tek test olarak kullanmak ilk skoru büyütür; fakat sonuç görüldükten sonra yapılan
her model/prompt değişikliği test setine dolaylı tuning olur. Paper hedefi nedeniyle `100 dev + 500
final` ayrımı korunur. İsim kafa karıştırıyorsa raporda `development` ve `final_test` denebilir;
kodda `sealed_test` etiketi, yanlışlıkla tuning yapılmasını önleyen protokol adıdır.

## 4. Bir family nasıl oluşturuluyor?

Generator 11 tam pasajı bağımsız yazmaz. Önce şu parçaları üretir:

- `query`: 1 veya 2 cümle.
- `context_sentences`: pasaj uzunluğuna göre 0–3 ortak tam cümle.
- Her aday için tek bir `critical_sentence`.

Kod, kritik cümleyi planlanan `critical_sentence_position` konumuna yerleştirir. Ortak bağlam
cümleleri bütün 11 adayda byte-identical kalır. Böylece doğru aday:

- daha uzun olduğu için,
- daha ayrıntılı olduğu için,
- kritik bilgi hep ilk/son cümlede olduğu için

kolayca seçilemez. Adaylar arasındaki karar mümkün olduğunca yerel kritik cümlede kalır.

Örnek sadeleştirilmiş internal kayıt:

```json
{
  "family_id": "family_raw_00042_xxxxxxxxxx",
  "target_split": "sealed_test",
  "generalization_bucket": "composition_holdout",
  "objective": "composition",
  "target_feature": "POSS.PL.ABL",
  "query_sentence_count": 1,
  "passage_sentence_count": 3,
  "critical_sentence_position": 2,
  "query": "Arkadaşlarımızdan biri yanlış otobüse bindi.",
  "context_sentences": [
    "Grup sabah havaalanında toplandı.",
    "Rehber yolcuları çıkışta yeniden saydı."
  ],
  "candidates": [
    {
      "id": "family_raw_00042_xxxxxxxxxx_c01",
      "candidate_slot": "positive_01",
      "role": "positive",
      "subtype": "equivalence_positive",
      "critical_sentence": "Bizim gruptaki arkadaşlarımızdan biri yanlış otobüse bindi.",
      "text": "...kod tarafından birleştirilmiş tam pasaj...",
      "critical_word": "arkadaşlarımızdan",
      "morph_relation": "target_preserved",
      "reason": "Hedef çoğul + iyelik + ayrılma zinciri korunuyor."
    }
  ],
  "gold_id": "family_raw_00042_xxxxxxxxxx_c01",
  "qc": {},
  "provenance": {}
}
```

`role`, `subtype`, `gold_id`, kritik sözcükler ve generator gerekçeleri private/internal görünümde
kalır. Blind final-test görünümünde model yalnız query, aday ID'leri, aday metinleri ve izin verilen
analiz metadata'sını görür.

## 5. Hard-negative taksonomisi

Sekiz, family başına hard aday **sayısıdır**. Her morfolojik fenomene zorla aynı sekiz hata türü
uygulanmaz. Taksonomide toplam 11 olası hard senaryo vardır.

Her family'deki altı çekirdek senaryo:

1. `minimal_morph_negative`: yalnız hedef morfolojik karşıtlık yanlış.
2. `same_lemma_wrong_inflection`: aynı kritik lemma, ilgili fakat yanlış çekim.
3. `related_feature_negative`: hedefe komşu ikinci morfolojik özellik yanlış.
4. `same_morph_wrong_content`: hedef morfoloji doğru, olay/nesne/referans yanlış.
5. `state_participant_time_trap`: durum, katılımcı, kişi veya zaman yanlış.
6. `close_paraphrase_wrong_meaning`: sözcüksel olarak yakın, temel önerme yanlış.

Fenomene göre eklenen iki senaryo:

| Family türü | Ek iki hard senaryo |
|---|---|
| Hâl, iyelik, kişi/sayı, çatı | `argument_role_reversal`, `morph_distractor` |
| TAM ve türetim | `scope_attachment_trap`, `morph_distractor` |
| Ek zinciri/composition | `partial_chain_negative`, `scope_attachment_trap` |
| Allomorph invariance | `allomorph_form_function_trap`, `morph_distractor` |

Bozuk Türkçe, imkânsız ek dizisi veya anlamsız cümle hard negative değildir; bunlar ucuz
dilbilgisi ipucu olur ve family reddedilir.

### Allomorph ile anlam karşıtlığı

`-da/-de/-ta/-te` gibi aynı işlevin geçerli yüzey biçimleri anlamı koruyabilir ve negative
olamaz. Buna karşılık `şubesinde` (LOC) ile `şubesinden` (ABL), birbirinin allomorphu değildir;
farklı hâl ve anlam taşır. Kod `allomorph_invariance` ile `morpheme_sensitivity` objective'lerini
ayrı tutar ve geçerli allomorphu negatif yapan family'yi engeller.

### Tam morfolojik fenomen kataloğu

Kodda toplam **64 hedef fenomen** bulunur: 50 single-layer fenomen ve 14 composition/chain.

- **Olumsuzluk, çatı ve valency (7):** `NEG` olumsuzluk, `ABIL` yeterlilik, `NEG.ABIL`
  yetersizlik, `CAUS` ettirgen, `PASS` edilgen, `REFL` dönüşlü, `RECP` işteş.
- **Zaman, görünüş, kip ve kanıtsallık (10):** `PST` görülen geçmiş, `PRF.EVID` duyulan
  geçmiş/kanıtsallık, `PRS.PROG` şimdiki zaman, `FUT` gelecek, `AOR` geniş zaman, `NEC`
  gereklilik, `OPT` istek, `IMP.3` üçüncü kişi emir, `COND` koşul, `PRSM` tahmin/kesinlik.
- **Hâl, konum ve yön (7):** `ACC` belirtme, `DAT` yönelme, `LOC` bulunma, `ABL` ayrılma,
  `INS` vasıta/birliktelik, `GEN` ilgi, `EQU` eşitlik hâli.
- **Çoğul, kişi ve iyelik (7):** `PL`, `POSS.1SG`, `POSS.2SG`, `POSS.3SG`, `POSS.1PL`,
  `POSS.2PL`, `POSS.3PL`.
- **Ulaç, sıfat-fiil, türetim ve allomorph (19):** `CVB.WHEN` -ince, `CVB.BY` -erek,
  `CVB.AND` -ip, `CVB.WITHOUT` -meden, `CVB.ASLONG` -dikçe, `CVB.WHILE` -ken,
  `NMLZ.MEK` -mek adlaştırması, `PTCP.SUBJ` özne sıfat-fiili, `PTCP.FUT` gelecek
  sıfat-fiili, `PRIV` yoksunluk, `PROP` varlık/bulunma, `REL.KI` aitlik -ki, `AGT`
  meslek/uğraş, `ABST` soyutlama, `DISTR` üleştirme, `ALLO.LOC`, `ALLO.ABL`, `ALLO.DAT`,
  `ALLO.ACC`.
- **Ek-zinciri composition (14):** `NEG.AOR`, `PLUPRF`, `PST.PROG`, `FUT.PST`, `CNTR`,
  `NMLZ.DIK`, `PL.POSS.CASE`, `CAUS.PASS.NEG`, `EVID.COND.NEG`, `NMLZ.CASE.CNTR`,
  `TENSE.PERS.NEG`, `EVID.POSS.NEG`, `RECP.CAUS`, `POSS.PL.ABL`.

Objective dağılımı katalog düzeyinde 46 `morpheme_sensitivity`, 4 `allomorph_invariance` ve
14 `composition` seçeneğidir.

## 6. Uçtan uca ana üretim akışı

Ana paper pipeline şu sırayla çalışır:

1. **Config doğrulama:** Hedef sayılar, dağılımlar, model aileleri ve reviewer sayısı kontrol edilir.
2. **Deterministic plan:** Seed ile 1.800 ham slot hazırlanır. Prefix-stable plan aynı config/seed ile
   yeniden üretilebilir.
3. **LLM generation:** Generator query, ortak bağlam ve 11 kritik cümleyi strict JSON schema ile
   üretir.
4. **Normalize/assemble:** Kod aday ID'lerini nötr biçimde karıştırır ve tam pasajları birleştirir.
5. **Deterministic QC:** Rol/slot sayıları, cümle sayısı, kritik sözcük, allomorph, uzunluk bias'ı,
   tekrar ve şema hataları kontrol edilir. Başarısız family en fazla bir kez yeniden yazdırılır.
6. **Blind LLM judge:** Generator'dan farklı model ailesi, etiketleri görmeden family'yi değerlendirir.
7. **Oversampling/select:** Geçen ham havuzdan dengeli 750 family insan review'una seçilir.
8. **Blind human review:** Normal family'yi iki reviewer; `%10` kalibrasyon family'sini beş reviewer
   değerlendirir. Anlaşmazlıklar adjudication'a gider.
9. **Final selection:** İnsan onaylı havuzdan dağılımı koruyan 100 dev + 500 final seçilir.
10. **Freeze/export:** Leakage, yakın-kopya, aday pozisyonu, kritik-cümle konumu ve artefakt
    kontrolleri geçerse JSON/BEIR/qrels dosyaları hash manifestiyle dondurulur.

Ana üretimde generator ile judge aynı model ailesinden olamaz. Legacy train Gemini ile üretildiği
için varsayılan test config'i `google/*` model ailesini de generator/judge için reddeder.

## 7. LLM-as-a-judge tam olarak neye karar veriyor?

Judge'a gösterilenler:

- Query.
- Hedef morfolojik feature, objective ve layer.
- Nötr sırada yalnız aday ID + tam aday metni.

Judge'a **gösterilmeyenler**:

- Hangi adayın positive/hard/easy olduğu.
- Gold ID.
- Generator'ın subtype etiketi, gerekçesi ve candidate slotu.

Judge şu kararları verir:

1. Query'yi bütünüyle karşılayan bütün aday ID'leri hangileri?
2. Her aday `fully`, `partially` veya `not_relevant` mı?
3. Her aday doğal Türkçe mi (`naturalness` 1–5)?
4. Biçimbilim ve ek zinciri dilbilgisel olarak geçerli mi?
5. Adayın gözlenen hata türü intended subtype ile uyumlu mu?
6. Geçerli allomorph yanlışlıkla negatif yapılmış mı?
7. Uzunluk, üslup veya ayrıntı gold'u ele veriyor mu?
8. Family'nin genel doğallığı yeterli mi?

Otomatik kabul için mevcut kod:

- Judge'ın relevant kümesi tam olarak tek gold olmalı.
- Subtype agreement en az `0.90` olmalı.
- `morphology_ok=false` aday bulunmamalı.
- Allomorph veya uzunluk/üslup artefaktı uyarısı olmamalı.
- Family naturalness en az `4/5` olmalı.

Dolayısıyla judge yalnız hard/easy etiketini kontrol etmez; positive'ın benzersizliğini, bütün
negatiflerin gerçekten negatif olmasını, doğallığı, biçimbilimi ve artefaktları birlikte denetler.

### Birden fazla LLM judge kullanmak

Paper sürümü için birden fazla judge faydalıdır; fakat aynı modelin üç kez çalıştırılması bağımsız
kanıt sayılmaz. Önerilen gelecek tasarım:

- Generator'dan farklı en az iki judge model ailesi.
- Judge A: relevance ve tek-gold kararı.
- Judge B: Türkçe biçimbilim, allomorph ve ek-zinciri denetimi.
- İstenirse Judge C: doğallık ve artefakt denetimi.
- Tüm judge'lar aynı blind aday görünümünü alır; birbirlerinin kararını görmez.
- Hepsi tek gold ve geçerli morphology üzerinde anlaşırsa family insan havuzuna geçer.
- Herhangi bir anlaşmazlık otomatik çoğunlukla bastırılmaz; insan review/adjudication'a gider.
- Model kimliği, prompt sürümü, request hash'i ve ham kararlar saklanır.

Bu repository'de şu anda **tek bağımsız LLM judge + insan reviewer sistemi** uygulanmıştır.
Multi-judge ensemble yukarıdaki biçimde planlanan sonraki geliştirmedir.

## 8. İnsan review protokolü

Beş reviewer tanımlıdır. Normal family başına iki bağımsız review atanır; toplam review havuzunun
`%10` kalibrasyon dilimini beş reviewer da görür. Reviewer aday rollerini ve gold'u görmez; aday
sırası reviewer'a özel deterministik olarak karıştırılır.

Reviewer şunları işaretler:

- Tam doğru/relevant aday ID'leri.
- Doğal olmayan adaylar.
- Biçimbilim sorunu olan adaylar.
- Family doğallığı 1–5.
- Uzunluk/üslup artefaktı.
- `approve` veya `reject` ve opsiyonel not.

Family ancak reviewer'ın seçtiği relevant kümesi tek gold ise, doğallık en az 4 ise, bozuk aday ve
artefakt yoksa otomatik review-pass alır. İki reviewer anlaşmazsa üçüncü adjudication kararı gerekir.

## 9. Provider'lar ve iki ayrı üretim modu

### Paper ana üretimi: OpenRouter

Ana `generate` komutu iki farklı model ailesi kullanır:

```bash
export OPENROUTER_API_KEY="..."
export TEST_GENERATOR_MODEL="provider-a/model-a"
export TEST_JUDGE_MODEL="provider-b/model-b"
python3 -m test generate --run-id test_v31
```

Bu mod deterministic QC + bağımsız judge çalıştırır ve paper veri hattına girebilir.

### API key'siz Codex preview

Yerelde ChatGPT ile oturum açmış Codex CLI varsa GPT-5.6 Sol ile küçük preview üretilebilir:

```bash
codex login status
python3 -m test preview-codex \
  --run-id sol_preview_20_v31 \
  --count 20 \
  --batch-size 10 \
  --model gpt-5.6-sol \
  --reasoning-effort medium
```

Bu yol API key istemez; ChatGPT/Codex kullanım kotasını kullanır. Çıktı strict schema ve bütün
deterministic QC kapılarından geçer. Ancak generator ve değerlendiren aynı Codex sistemi olacağı
için bağımsız LLM judge çalıştırılmaz. Bu nedenle çıktılar `preview_only=true` ve
`codex_cli_preview_unjudged` olarak işaretlenir; doğrudan benchmark'a dondurulamaz.

## 10. Komutlar

Repo kökünden çalıştırın:

```bash
# API kullanmadan config/plan/QC regression testleri
python3 -m test self-test

# API çağırmadan 1.800-slot planı yaz
python3 -m test plan --run-id test_v31

# Küçük OpenRouter pilotu
python3 -m test generate --run-id test_v31_pilot --limit 30

# Tam ham üretim; JSONL checkpoint ve cache ile devam edebilir
python3 -m test generate --run-id test_v31

# 750 family'yi beş reviewera dağıt
python3 -m test prepare-review --run-id test_v31

# Bir reviewer dosyasını interaktif doldur
python3 -m test review-file \
  --path test/runs/test_v31/review/assignments/reviewer_1.jsonl

# Agreement ve adjudication durumu
python3 -m test merge-reviews --run-id test_v31

# Bütün anlaşmazlıklar çözüldükten sonra freeze
python3 -m test finalize --run-id test_v31
```

Run kimliğiyle yeniden başlatma; aynı plan/config/prompt/model hash'leri değişmediyse mevcut JSONL
checkpoint ve request cache'inden devam eder. Aynı run ID altında model/config karıştırılması
engellenir.

## 11. Modüller ne yapıyor?

| Dosya | Sorumluluk |
|---|---|
| `config.json` | Sayılar, dağılımlar, kalite eşikleri, reviewer ve provider ayarları |
| `config.py` | Config/env çözümleme ve güvenlik doğrulaması |
| `taxonomy.py` | Türkçe feature kataloğu, macro/layer yapısı, domain/template ve hard profilleri |
| `planner.py` | Seed'li, prefix-stable ham slot planı ve holdout politikaları |
| `schema.py` | Generator ve judge strict JSON Schema sözleşmeleri |
| `prompts.py` | Zero-shot generator, repair ve blind judge promptları |
| `providers.py` | OpenRouter JSON çağrısı ve API key'siz yerel Codex CLI preview provider'ı |
| `pipeline.py` | Plan → generate → repair → deterministic QC → judge → checkpoint akışı |
| `preview.py` | GPT-5.6 Sol ile preview-only batch üretimi ve blind/internal export |
| `validators.py` | Cümle, rol, slot, allomorph, uzunluk, duplicate, judge ve artefakt kapıları |
| `selection.py` | Bucket × query/pasaj dağılımını ve kritik-cümle konumunu dengeli seçme |
| `review.py` | Beş reviewer assignment'ı, interaktif checkpoint, agreement ve adjudication |
| `exports.py` | 100/500 freeze, leakage/artefakt kontrolü, private/public/BEIR/qrels export |
| `evaluation.py` | Retrieval metrikleri, ucuz baseline'lar, ablation ve istatistiksel testler |
| `selftest.py` | API'siz regression testleri ve kasıtlı hata fixture'ları |
| `notebooks/morph_baseline_eval_colab_v2.ipynb` | Güncel Colab: 20-preview/frozen modları, encoder cache'i, tam closed-family metrikleri, CI/test/slice/ablation/hata analizi |
| `notebooks/morph_baseline_eval_colab.ipynb` | Önceki encoder/baseline notebook'u; karşılaştırma için korunur |

## 12. Deterministic kalite kapıları

Başlıca family-level kontroller:

- Tam 11 aday ve doğru `1/8/2` rol dağılımı.
- `positive_01`, `hard_01..08`, `easy_01..02` slotlarının eksiksiz ve benzersiz olması.
- Family feature'ına atanmış uyarlanabilir sekiz hard subtype'ın birebir bulunması.
- Query ve pasajların planlanan tam cümle sayısına uyması.
- Her `critical_sentence` ve ortak context parçasının tek tam cümle olması.
- Kritik sözcüklerin gerçekten query/critical sentence içinde bulunması.
- Positive'ın doğru morph relation taşıması.
- Geçerli allomorphun hiçbir negatifte eşdeğer diye kullanılmaması.
- Adayların birebir tekrar etmemesi.
- Candidate token uzunluk oranı ve gold/median uzunluk bias sınırları.
- Soru/meta-arama dili (`kaydı bul`, `arıyorum`) kullanılmaması.

Corpus/freeze düzeyinde ayrıca yakın query kopyası, lemma/template/composition/domain leakage,
candidate-position bias, kritik-cümle-position bias ve longest-gold oranı kontrol edilir.

## 13. Çıktı dizinleri

Ana run:

```text
test/runs/<run_id>/
├── plan.json
├── run_manifest.json
├── accepted.jsonl
├── rejected.jsonl
├── failures.jsonl
├── cache/
├── review/
│   ├── review_pool_private.jsonl
│   ├── assignments/
│   ├── review_status.jsonl
│   └── adjudication.jsonl
├── release/
│   ├── morph_dev_v3.1.0.json
│   ├── morph_test_blind_v3.1.0.json
│   ├── artifact_audit.json
│   ├── freeze_manifest.json
│   └── beir/
└── private/
    ├── morph_test_internal_v3.1.0.json
    ├── private_qrels.jsonl
    ├── beir_test_qrels.tsv
    └── train_exclusion_holdouts.json
```

Codex preview:

```text
test/previews/<run_id>/
├── accepted.jsonl
├── rejected.jsonl
├── preview_internal.json
├── preview_blind.json
├── preview_review.md
├── artifact_audit.json
├── manifest.json
└── cache/
```

`preview_internal.json` roller/gold/subtype içerir; `preview_blind.json` insanın etiketsiz biçimde
örneklere bakması içindir. `preview_review.md`, aynı kör görünümün doğrudan okunabilen Markdown
sürümüdür.

## 14. Evaluation katmanları

### Closed-family contrast

Her query yalnız kendi 11 adayı arasında değerlendirilir. Ana sonuçlar:

- `hard_only_recall@1/5`, `hard_only_mrr@10`, `hard_only_ndcg@10`: gold, 8 hard arasındaki sıralamada nerede?
- `pairwise_hard_accuracy`: gold, sekiz hard negatifin ne kadarını skor olarak geçiyor?
- `contrast_consistency`: gold özellikle minimal morfolojik karşıtın üstünde mi?
- `minimal_margin`, `hardest_hard_margin`, `hardest_negative_margin`: karar güvenliği ne kadar?
- `Recall@1/5/10/100`, `MRR@10`, `nDCG@10`, `MAP@10` ve ortalama/medyan gold sırası.

Tek relevant belge bulunan closed-family düzende `MAP@10`, `MRR@10` ile aynıdır. `Recall@100`
11 adaylık deneyde doygundur; asıl full-corpus raporunda anlam taşır.

### Full-corpus retrieval

Final testte 500 query ve 5.500 aday doküman ortak corpus'a konabilir. Fakat başka family'nin
dokümanı tesadüfen relevant olabilir. Bu nedenle yalnız own-family gold qrels'iyle full-corpus
metrik raporlanmaz. BM25, char n-gram, dense encoder ve reranker top sonuçları pool edilir; yabancı
query-document çiftleri insanlarca yargılanır. Yargılanmamış doküman otomatik negatif değildir.

V2 notebook ortak corpus sıralamasını encoder embedding'leri üretilirken aynı geçişte hesaplar.
İnsan qrels'i yokken yalnız açıkça etiketlenmiş `known-gold diagnostic` tabloyu ve BM25 + character
3-gram + word overlap + dense encoder top-20 birleşiminden kör `pooling_judgment_template.jsonl`
dosyasını üretir. Pooled insan qrels'i eklendiğinde resmi full-corpus Recall/MRR/nDCG/MAP tablosu
aynı notebook'ta otomatik açılır.

### Artefakt ve ablation

`evaluation.py` ve notebook şu ucuz baseline/ablation'ları destekler:

- Longest candidate ve most tokens.
- Candidate position.
- Character 3-gram, word overlap ve BM25.
- Query'siz candidate-only char-TFIDF classifier.
- Kritik sözcüğü silme.
- Her tokenı ilk beş karaktere indirgeme (`prefix5`); bu gerçek kök/lemma analizi değil,
  ek bilgisini azaltan ucuz bir kontrol deneyidir.

İstatistiksel raporlama query-level bootstrap `%95 CI`, paired bootstrap, approximate randomization,
McNemar, Holm düzeltmesi ve query/pasaj/layer/objective/generalization slice sonuçlarını içerir.

## 15. Reproducibility ve paper kuralları

Her run manifestinde dataset/prompt sürümü, config ve plan SHA-256, kaynak dosya hash'leri, model
kimlikleri, request hash'leri, token/çalışma bilgisi ve başlangıç git commit'i tutulur. Freeze
manifesti yayımlanan her dosyanın SHA-256 değerini kaydeder.

Paper için zorunlu ayrımlar:

- Codex preview ile bağımsız-judge + insan-onaylı final veri karıştırılmaz.
- Development üzerinde yapılan bütün kararlar kaydedilir.
- Final test yalnız kararlar dondurulduktan sonra çalıştırılır.
- LLM judge insan review'un yerine geçmez; yalnız pahalı insan havuzuna girecek family'leri süzer.
- Train generator, `train_exclusion_holdouts.json` içindeki final-test lemma/template/zincir/domain
  ve metin hash'lerini dışlar.
- Closed-family ve pooled full-corpus sonuçları iki ayrı deney olarak raporlanır.
