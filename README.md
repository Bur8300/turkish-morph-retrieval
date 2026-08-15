# Morphology-Aware Contrastive Fine-Tuning for Turkish Retrieval

Türkçe eklerin taşıdığı anlamı retrieval embedding'lerinde korumayı hedefleyen dual-encoder
araştırma projesi. Repo artık veri yaşam döngüsünü iki bağımsız parçaya ayırır:

| Dizin | Amaç |
|---|---|
| [`train/`](train/) | Eski Gemini train/dev generator'ı, model-selection araçları ve v2.0–v2.2 JSON geçmişi. Bu bölüm korunmuş legacy sistemdir. |
| [`test/`](test/) | Yeni paper-grade test generator'ı: 100 development + 500 sealed test, farklı-model judge, otomatik QC, beş kişilik kör review ve freeze/export. |
| [`test/notebooks/morph_baseline_eval_colab_v2.ipynb`](test/notebooks/morph_baseline_eval_colab_v2.ipynb) | 20-family preview ve final freeze için güncel encoder/artefakt/istatistik değerlendirmesi. |
| [`test/notebooks/morph_baseline_eval_colab.ipynb`](test/notebooks/morph_baseline_eval_colab.ipynb) | Önceki baseline notebook'u; karşılaştırma amacıyla korunur. |

Eski, iki insan turundan geçmiş 50-family JSON ve önceki reviewer sürümü
[`train/legacy_test_data/`](train/legacy_test_data/) altında provenance amacıyla saklanır. Yeni test
generator'ı bu metinleri few-shot olarak kullanmaz.

## Yeni test kararı

- Toplam **600 bağımsız contrast family**: 100 development + 500 final test.
- Her family: 1 query, **1 positive + 8 hard negative + 2 easy negative**.
- Query uzunluğu: **%75 bir, %25 iki cümle**; tek ve açık bir bilgi ihtiyacı.
- Aday pasaj uzunluğu: **%30/%30/%30/%10 oranında 1/2/3/4 cümle**. Kritik cümle konumu
  dengelenir; aynı family'deki diğer tam bağlam cümleleri bütün adaylarda aynıdır.
- Standard, lemma-holdout, template-holdout ve compositional-holdout dilimleri.
- Allomorph invariance ile anlam değiştiren morfem karşıtlığı ayrı objective'lerdir.
- 3× ham üretim → deterministic QC → farklı model ailesinden kör judge → 750-family insan review
  havuzu → 100/500 dengeli seçim → hash/manifest ile freeze.
- Beş reviewer; normal family başına iki bağımsız karar, %10 ortak kalibrasyon ve anlaşmazlıklarda
  adjudication.
- Test qrels'i private kalır. Full-corpus evaluation için yabancı family dokümanları ayrıca
  pool edilip insanlarca yargılanmadan “negatif” kabul edilmez.

Ayrıntılı şema, uyarlanabilir hard-negative taksonomisi, review formatı ve komutlar:
[`test/README.md`](test/README.md). Daha kısa başlangıç özeti:
[`test/README_SHORT.md`](test/README_SHORT.md).

## Hızlı doğrulama

Yeni test kodu planlama/QC tarafında yalnız Python standart kütüphanesini kullanır:

```bash
python3 -m test self-test
python3 -m test plan --run-id test_v31
```

API üretimi için:

```bash
export OPENROUTER_API_KEY="..."
export TEST_GENERATOR_MODEL="provider-a/model-a"
export TEST_JUDGE_MODEL="provider-b/model-b"
python3 -m test generate --run-id test_v3_pilot --limit 30
```

Generator ve judge aynı OpenRouter model ailesindeyse kod çalışmayı reddeder. Model kimlikleri,
prompt/config sürümü, request hash'leri, token kullanımı ve git commit'i run manifestinde tutulur.

Legacy train sistemi için önce `cd train`, ardından [`train/README.md`](train/README.md) içindeki
komutları kullanın.

## Paper değerlendirme katmanları

1. Her query'nin kendi 11 adayı üzerindeki kontrollü morfolojik contrast sonucu.
2. İnsan pooling qrels'i tamamlandıktan sonra ortak 5.500 test dokümanı üzerinde full-corpus
   retrieval.
3. Genel retrieval yeteneğinin korunması için harici Turkish-BEIR/TR-MTEB sonuçları.

## Lisans

MIT — [`LICENSE`](LICENSE).
