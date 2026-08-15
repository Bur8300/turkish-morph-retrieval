# v2.1 üretim raporu

- model: `gemini-3.5-flash-lite`
- süre: 26.6 dk
- API çağrısı: 1333 (üretim 653, jüri 308, onarım 372); önbellekten: 531
- kabul 269 · red 644 · verim 29.5%
- train 228 · dev 41 (sözcük dağarcığı örtüşmesi nedeniyle 0 öğe dev'den train'e taşındı)

## Kota kullanımı

| anahtar | kullanılan | kalan |
|---|---|---|
| API_KEY_1 | 500 | 0 |
| API_KEY_2 | 500 | 0 |
| API_KEY_3 | 500 | 0 |
| API_KEY_4 | 500 | 0 |

## Red gerekçeleri (aşama)

| aşama | adet |
|---|---|
| validate | 481 |
| judge | 163 |

### Kural tabanlı kapılar

| kapı | tetiklenme |
|---|---|
| lexical | 425 |
| morphology | 132 |
| tier | 12 |
| repair | 9 |
| structure | 6 |

### En sık gerekçeler

| gerekçe | adet |
|---|---|
| lexical | 852 |
| morphology | 180 |
| jüri pozitifi | 142 |
| çift altın | 37 |
| tier | 21 |
| jüri Türkçe biçimbilim hatası bildirdi | 15 |
| repair | 9 |
| structure | 6 |
| sorgu bir durumu iddia etmiyor | 1 |

## Kör (query-blind) artefakt denetimi

SugarCrepe'in kör-model tanısının sıralama kümesine uyarlanmışı. Sorguyu hiç okumadan doğru adayı seçebilen bir ölçüt, veri kümesinin biçimbilimi değil üretim artefaktını ölçtüğü anlamına gelir. Şans düzeyi 9.1%.

| ölçüt | değer | şans |
|---|---|---|
| blind_longest_is_gold | 32.0% | 9.1% |
| blind_most_tokens_is_gold | 34.6% | 9.1% |
| sparse_char3gram_top1_is_gold | 8.2% | 9.1% |

Seyrek temel çizgi pozitifi ilk sıraya koyduğunda aradaki fark: medyan **0.024**, en yüksek 0.1133. Oranın tek başına anlamı yoktur: 0.005 farkla kazanmak ile 0.30 farkla kazanmak farklı şeylerdir; medyan farkın sıfıra yakın olması, seyrek ölçütün neredeyse berabere kalan adaylar arasında tahmin yürüttüğünü gösterir.

Kritik sözcük çiftinin kaynağı: `{'reported': 224, 'derived': 40, 'derived_from_query': 5}` (`reported` = üreticinin hedeflediği karşıtlık doğrulandı; `derived` = çift metinlerden türetildi, hedef özellikle birebir aynı olmayabilir).


### Seyrek (char-3gram) temel çizgi — alt tür bazında ikili doğruluk

Modelsiz bir sözcük-örtüşmesi ölçütü, altın adayı her bir olumsuz adayın üstüne koyabiliyor mu? Rastgele = %50.

| alt tür | v2.1 | v2.0 (istem düzeltmesi öncesi) | v1.3.1 (test) | okuma |
|---|---|---|---|---|
| morph_counterfactual | 16.4 | 65.8 | 23.5 | **çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)** |
| partial_trap | 37.2 | 60.5 | 66.7 | **çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)** |
| same_feature_wrong_content | 38.3 | 88.3 | 25.5 | **çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)** |
| state_variant | 26.2 | 63.4 | 34.3 | **çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)** |
| easy_negative | 99.5 | 99.5 | 97.6 | tasarımı gereği kolay |

> **Bilinen zayıflık.** v2.0'da `same_feature_wrong_content` seyrek ölçütle neredeyse tamamen çözülebiliyor, `morph_counterfactual` ise rastgelenin üzerinde. v1.3.1'de her ikisi de rastgelenin ALTINDA, yani sözcük örtüşmesi yanlış adayı işaret ediyor. Nedeni: v1.3.1'de pozitif, sorgunun bağımsız bir yeniden ifadesidir; v2.0'da ise sorgunun içerik sözcüklerini aynı sırada koruyan daha yakın bir yeniden yazımdır, dolayısıyla yüzey benzerliği doğru adayı ele veriyor. Bu, v2.0'ın bir EĞİTİM kümesi olarak kullanılabilirliğini ortadan kaldırmaz, ama tek başına bir ölçüt (benchmark) olarak kullanılmasını engeller — ölçüt v1.3.1'dir.


Ortalama karakter uzunlukları:

| rol/alt tür | ort. uzunluk |
|---|---|
| positive | 128.6 |
| hard_negative | 124.0 |
| easy_negative | 110.3 |
| morph_counterfactual | 122.1 |
| same_feature_wrong_content | 122.1 |
| partial_trap | 129.6 |
| state_variant | 123.2 |

> Karşılaştırma: v1.3.1'de `blind_longest_is_gold` **%50** (şans %9). Yani insan denetiminden geçmiş test kümesinde pozitifler sistematik olarak daha uzun. Bu, v2.0 için düzeltilmesi hedeflenen bilinen bir artefakttır ve üretim isteminde açıkça yasaklanmıştır.
> Uyarı (Feng vd. 2019): kör ölçütün BAŞARISIZ olması, veri kümesinin artefaktsız olduğunu KANITLAMAZ. Bu sayılar bir hipotez testidir, temiz kâğıt değil.

## Kapsam

- hedef biçimbirim özelliği: 73/75
- eksik özellikler: CVB.ABOUTTO, EVID.COND.NEG

| özellik | adet |
|---|---|
| PL.POSS.CASE | 10 |
| PTCP.OBJ | 9 |
| ALLO.DAT | 7 |
| CVB.BY | 7 |
| NMLZ.MEK | 6 |
| POSS.2PL | 6 |
| HAB.PST | 6 |
| POSS.PL.ABL | 6 |
| EVID.POSS.NEG | 6 |
| RECP | 6 |
| ALLO.ABL | 6 |
| PST | 6 |
| PLUPRF | 5 |
| NMLZ.ECEK | 5 |
| OPT | 5 |
| NMLZ.ME | 5 |
| POSS.3SG | 5 |
| ABL | 5 |
| DAT | 5 |
| POSS.3PL | 4 |
| CVB.NEGINS | 4 |
| TENSE.PERS.NEG | 4 |
| CAUS.PASS.NEG | 4 |
| ABIL.COND | 4 |
| CNTR | 4 |
| INS | 4 |
| ABST | 4 |
| NMLZ.DIK | 4 |
| REL.KI | 4 |
| POSS.1SG | 4 |
| CVB.ASLONG | 4 |
| NMLZ.CASE.CNTR | 4 |
| AOR | 4 |
| NEC | 4 |
| POSS.1PL | 4 |
| PST.PROG | 3 |
| IMP.3 | 3 |
| PASS | 3 |
| PRS.PROG | 3 |
| CVB.WHILE | 3 |
| COND | 3 |
| AGT | 3 |
| DESID | 3 |
| CAUS | 3 |
| PRIV | 3 |
| FUT | 3 |
| RECP.CAUS | 3 |
| PRF.EVID | 3 |
| DISTR | 3 |
| LOC | 3 |
| PL | 3 |
| CAUS.CAUS | 3 |
| NEG.ABIL | 3 |
| REFL | 3 |
| PRSM | 3 |
| PRIV.VS.NEG | 2 |
| POSS.2SG | 2 |
| CVB.AND | 2 |
| FUT.PST | 2 |
| PTCP.FUT | 2 |
| PTCP.SUBJ | 2 |
| ACC | 2 |
| VBLZ | 2 |
| ABIL | 2 |
| NEG | 2 |
| CVB.SINCE | 2 |
| CVB.WHEN | 1 |
| CAUS.REFL.NEG | 1 |
| CVB.WITHOUT | 1 |
| PROP | 1 |
| GEN | 1 |
| EQU | 1 |
| NEG.AOR | 1 |

## Sızıntı muhasebesi

- üreticiye örnek olarak gösterilen v1.3.1 öğeleri: `q03_negation`, `q103_dilution_person`, `q108_evid_counterfactual`, `q113_min_person_number`
- **test kümesi skorları raporlanırken bu 4 öğe hariç tutulmalıdır** (50 öğelik test kümesinin %8'si)
- üretilen sorguların v1.3.1'e en yüksek benzerliği: 0.286 (eşik 0.60)
