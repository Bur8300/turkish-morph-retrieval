# Test benchmark — kısa sürüm

Bu dizin Türkçe retrieval modellerinin eklerin taşıdığı anlamı ayırt edip etmediğini ölçen test
benchmark'ını üretir. Ayrıntılar için [`README.md`](README.md) dosyasına bakın.

## Veri yapısı

- Toplam: **600 contrast family**.
- Split: **100 development + 500 final/sealed test**.
- Her family: **1 query + 1 positive + 8 hard negative + 2 easy negative**.
- Toplam: 600 query ve 6.600 aday pasaj.

Uzunluk:

- Query: `%75` bir cümle, `%25` iki cümle.
- Pasaj: `%30/%30/%30/%10` oranında 1/2/3/4 cümle.
- Aynı family'deki ortak bağlam cümleleri bütün adaylarda aynıdır.
- Yalnız kritik cümle değişir; kritik cümlenin pasaj içindeki konumu dengelenir.

Generalizasyon:

- `%40 standard`
- `%20 lemma_holdout`
- `%20 template_holdout`
- `%20 composition_holdout`
- Yaklaşık `%20` ayrıca `domain_shift` etiketi alır.

Sekiz hard negative, her family'de altı çekirdek hata + fenomene uygun iki özel hata olarak
planlanır. Kod 6 macro grup altında toplam 64 hedef morfolojik fenomen içerir: 50 single ve 14
composition/chain. Geçerli allomorph hiçbir zaman negative değildir.

## Kodun mantığı

```text
config + taxonomy
        ↓
1.800 deterministic ham slot
        ↓
LLM generator → strict JSON
        ↓
pasajları kodla birleştir + deterministic QC
        ↓
generator'dan farklı blind LLM judge
        ↓
750 family insan review havuzu
        ↓
5 reviewer: normalde 2 karar, %10 family'de 5 karar
        ↓
100 dev + 500 final seç → leakage/artefakt kontrolü → freeze
```

Judge etiketleri görmeden tek doğru positive'ı, negatiflerin gerçekten yanlış olmasını, doğallığı,
biçimbilimi, allomorphları ve uzunluk/üslup artefaktlarını kontrol eder. Mevcut kod tek bağımsız LLM
judge kullanır; paper'ın sonraki aşamasında farklı ailelerden 2–3 judge ve anlaşmazlıkları insana
gönderme sistemi önerilir.

Development ayar/model/evaluation kararları içindir. Final test insanlarca kör kontrol edilebilir;
“sealed” demek gold sonuçlarının model veya prompt seçmek için kullanılmaması demektir.

## En önemli dosyalar

| Dosya | İş |
|---|---|
| `config.json` | Sayılar, dağılımlar, eşikler ve provider ayarları |
| `taxonomy.py` | Morfolojik feature ve hard-negative sınıfları |
| `planner.py` | Seed'li veri planı |
| `prompts.py` / `schema.py` | Generator/judge görevleri ve JSON sözleşmesi |
| `pipeline.py` | Ana generate → QC → judge akışı |
| `validators.py` | Otomatik kalite kapıları |
| `review.py` | Beş kişilik kör insan review'u |
| `exports.py` | 100/500 freeze ve BEIR/qrels export |
| `evaluation.py` | Retrieval, baseline, ablation ve istatistik |
| `preview.py` | API key'siz GPT-5.6 Sol preview üretimi |

## Hızlı komutlar

```bash
# Offline kontrol
python3 -m test self-test

# Yalnız plan
python3 -m test plan --run-id test_v31

# Ana paper üretimi: OpenRouter generator + farklı-family judge
export OPENROUTER_API_KEY="..."
export TEST_GENERATOR_MODEL="provider-a/model-a"
export TEST_JUDGE_MODEL="provider-b/model-b"
python3 -m test generate --run-id test_v31

# API key'siz, ChatGPT oturumuyla 20-family Sol preview
python3 -m test preview-codex \
  --run-id sol_preview_20_v31 \
  --count 20 --batch-size 10 \
  --model gpt-5.6-sol --reasoning-effort medium
```

Codex preview deterministic QC'den geçer fakat bağımsız LLM judge ve insan review'u içermez;
paper verisi değil, yalnız tasarımı görmek içindir. Üretilen `preview_review.md` dosyası örnekleri
gold/role etiketleri olmadan hızlıca okumak içindir.
