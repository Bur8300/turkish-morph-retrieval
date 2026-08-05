# v2.2 üretim raporu

- model: `gemini-3.5-flash-lite`
- süre: 36.3 dk
- API çağrısı: 2048 (üretim 972, jüri 681, onarım 395); önbellekten: 48
- kabul 600 · red 396 · verim 60.2%
- train 510 · dev 90 (sözcük dağarcığı örtüşmesi nedeniyle 0 öğe dev'den train'e taşındı)

## Kota kullanımı

| anahtar | kullanılan | kalan |
|---|---|---|
| API_KEY_1 | 436 | 64 |
| API_KEY_2 | 435 | 65 |
| API_KEY_3 | 435 | 65 |
| API_KEY_4 | 435 | 65 |
| API_KEY_5 | 435 | 65 |

## Red gerekçeleri (aşama)

| aşama | adet |
|---|---|
| validate | 299 |
| judge | 97 |

### Kural tabanlı kapılar

| kapı | tetiklenme |
|---|---|
| lexical | 278 |
| morphology | 20 |
| tier | 18 |
| repair | 9 |

### En sık gerekçeler

| gerekçe | adet |
|---|---|
| lexical | 483 |
| çift altın | 62 |
| jüri pozitifi | 29 |
| tier | 29 |
| morphology | 23 |
| jüri Türkçe biçimbilim hatası bildirdi | 16 |
| repair | 9 |
| sorgu bir durumu iddia etmiyor | 4 |
| pozitif üslup/uzunluk bakımından aykırı | 4 |
| jüri adayların yalnızca 0/11 tanesini değerlendirdi | 1 |
| jüri adayların yalnızca 7/11 tanesini değerlendirdi | 1 |
| jüri adayların yalnızca 1/11 tanesini değerlendirdi | 1 |

## Kör (query-blind) artefakt denetimi

SugarCrepe'in kör-model tanısının sıralama kümesine uyarlanmışı. Sorguyu hiç okumadan doğru adayı seçebilen bir ölçüt, veri kümesinin biçimbilimi değil üretim artefaktını ölçtüğü anlamına gelir. Şans düzeyi 9.1%.

| ölçüt | değer | şans |
|---|---|---|
| blind_longest_is_gold | 49.5% | 9.1% |
| blind_longest_decisive | 20.3% | 9.1% |
| blind_most_tokens_is_gold | 44.5% | 9.1% |
| sparse_char3gram_top1_is_gold | 1.3% | 9.1% |

Seyrek temel çizgi pozitifi ilk sıraya koyduğunda aradaki fark: medyan **0.0336**, en yüksek 0.1138. Oranın tek başına anlamı yoktur: 0.005 farkla kazanmak ile 0.30 farkla kazanmak farklı şeylerdir; medyan farkın sıfıra yakın olması, seyrek ölçütün neredeyse berabere kalan adaylar arasında tahmin yürüttüğünü gösterir.

Kritik sözcük çiftinin kaynağı: `{'derived_from_query': 10, 'reported': 511, 'derived': 69, 'none': 10}` (`reported` = üreticinin hedeflediği karşıtlık doğrulandı; `derived` = çift metinlerden türetildi, hedef özellikle birebir aynı olmayabilir).


### Seyrek (char-3gram) temel çizgi — alt tür bazında ikili doğruluk

Modelsiz bir sözcük-örtüşmesi ölçütü, altın adayı her bir olumsuz adayın üstüne koyabiliyor mu? Rastgele = %50.

| alt tür | v2.2 | v2.0 (istem düzeltmesi öncesi) | v1.3.1 (test) | okuma |
|---|---|---|---|---|
| morph_counterfactual | 2.2 | 65.8 | 23.5 | **çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)** |
| partial_trap | 35.2 | 60.5 | 66.7 | **çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)** |
| same_feature_wrong_content | 19.8 | 88.3 | 25.5 | **çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)** |
| state_variant | 9.6 | 63.4 | 34.3 | **çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)** |
| easy_negative | 99.1 | 99.5 | 97.6 | tasarımı gereği kolay |

> **Bilinen zayıflık.** v2.0'da `same_feature_wrong_content` seyrek ölçütle neredeyse tamamen çözülebiliyor, `morph_counterfactual` ise rastgelenin üzerinde. v1.3.1'de her ikisi de rastgelenin ALTINDA, yani sözcük örtüşmesi yanlış adayı işaret ediyor. Nedeni: v1.3.1'de pozitif, sorgunun bağımsız bir yeniden ifadesidir; v2.0'da ise sorgunun içerik sözcüklerini aynı sırada koruyan daha yakın bir yeniden yazımdır, dolayısıyla yüzey benzerliği doğru adayı ele veriyor. Bu, v2.0'ın bir EĞİTİM kümesi olarak kullanılabilirliğini ortadan kaldırmaz, ama tek başına bir ölçüt (benchmark) olarak kullanılmasını engeller — ölçüt v1.3.1'dir.


Ortalama karakter uzunlukları:

| rol/alt tür | ort. uzunluk |
|---|---|
| positive | 154.3 |
| hard_negative | 144.2 |
| easy_negative | 136.2 |
| morph_counterfactual | 140.3 |
| same_feature_wrong_content | 141.6 |
| partial_trap | 153.4 |
| state_variant | 142.7 |

> Karşılaştırma: v1.3.1'de `blind_longest_is_gold` **%50** (şans %9). Yani insan denetiminden geçmiş test kümesinde pozitifler sistematik olarak daha uzun. Bu, v2.0 için düzeltilmesi hedeflenen bilinen bir artefakttır ve üretim isteminde açıkça yasaklanmıştır.
> Uyarı (Feng vd. 2019): kör ölçütün BAŞARISIZ olması, veri kümesinin artefaktsız olduğunu KANITLAMAZ. Bu sayılar bir hipotez testidir, temiz kâğıt değil.

## Kapsam

- hedef biçimbirim özelliği: 75/75
- eksik özellikler: yok

| özellik | adet |
|---|---|
| PTCP.OBJ | 17 |
| DESID | 16 |
| NMLZ.DIK | 16 |
| ABIL.COND | 14 |
| PLUPRF | 14 |
| TENSE.PERS.NEG | 14 |
| CNTR | 14 |
| NMLZ.ME | 13 |
| EVID.COND.NEG | 13 |
| PST.PROG | 13 |
| POSS.PL.ABL | 13 |
| CAUS.PASS.NEG | 12 |
| EVID.POSS.NEG | 12 |
| PL.POSS.CASE | 12 |
| NMLZ.ECEK | 12 |
| PRIV.VS.NEG | 12 |
| RECP.CAUS | 11 |
| CAUS.REFL.NEG | 11 |
| CAUS.CAUS | 11 |
| NMLZ.CASE.CNTR | 10 |
| FUT.PST | 9 |
| CVB.BY | 9 |
| OPT | 9 |
| PTCP.SUBJ | 9 |
| AGT | 9 |
| CAUS | 9 |
| HAB.PST | 9 |
| PASS | 8 |
| NEG.ABIL | 8 |
| CVB.ASLONG | 8 |
| CVB.ABOUTTO | 8 |
| ABST | 7 |
| CVB.NEGINS | 7 |
| PRSM | 7 |
| INS | 7 |
| POSS.2PL | 7 |
| POSS.3SG | 7 |
| ACC | 7 |
| PTCP.FUT | 7 |
| IMP.3 | 7 |
| NEG.AOR | 7 |
| NEC | 7 |
| REL.KI | 7 |
| POSS.1PL | 6 |
| ABIL | 6 |
| REFL | 6 |
| CVB.WHEN | 6 |
| PRF.EVID | 6 |
| PRIV | 6 |
| DAT | 6 |
| DISTR | 6 |
| CVB.WHILE | 6 |
| POSS.1SG | 6 |
| ABL | 6 |
| VBLZ | 6 |
| POSS.2SG | 6 |
| CVB.SINCE | 6 |
| AOR | 6 |
| CVB.WITHOUT | 5 |
| RECP | 5 |
| POSS.3PL | 5 |
| PRS.PROG | 5 |
| NMLZ.MEK | 5 |
| PROP | 5 |
| CVB.AND | 5 |
| COND | 5 |
| NEG | 4 |
| EQU | 4 |
| FUT | 4 |
| GEN | 4 |
| ALLO.DAT | 4 |
| LOC | 3 |
| PL | 3 |
| PST | 3 |
| ALLO.ABL | 2 |

## Sızıntı muhasebesi

- üreticiye örnek olarak gösterilen v1.3.1 öğeleri: `q03_negation`, `q103_dilution_person`, `q108_evid_counterfactual`, `q113_min_person_number`
- **test kümesi skorları raporlanırken bu 4 öğe hariç tutulmalıdır** (50 öğelik test kümesinin %8'si)
- üretilen sorguların v1.3.1'e en yüksek benzerliği: 0.261 (eşik 0.60)
