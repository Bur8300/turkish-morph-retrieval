# v2.0 üretim raporu

- model: `gemini-3.5-flash-lite`
- süre: 0.0 dk
- API çağrısı: 0 (üretim 0, jüri 0, onarım 0); önbellekten: 1444
- kabul 475 · red 258 · verim 64.8%
- train 403 · dev 72 (sözcük dağarcığı örtüşmesi nedeniyle 0 öğe dev'den train'e taşındı)

## Kota kullanımı

| anahtar | kullanılan | kalan |
|---|---|---|
| API_KEY_1 | 500 | 0 |
| API_KEY_2 | 500 | 0 |
| API_KEY_3 | 500 | 0 |

## Red gerekçeleri (aşama)

| aşama | adet |
|---|---|
| judge | 139 |
| validate | 119 |

### Kural tabanlı kapılar

| kapı | tetiklenme |
|---|---|
| lexical | 81 |
| morphology | 47 |
| tier | 2 |

### En sık gerekçeler

| gerekçe | adet |
|---|---|
| jüri pozitifi | 107 |
| lexical | 88 |
| morphology | 55 |
| çift altın | 49 |
| jüri Türkçe biçimbilim hatası bildirdi | 19 |
| tier | 3 |
| pozitif üslup/uzunluk bakımından aykırı | 3 |
| jüri adayların yalnızca 0/11 tanesini değerlendirdi | 2 |

## Kör (query-blind) artefakt denetimi

SugarCrepe'in kör-model tanısının sıralama kümesine uyarlanmışı. Sorguyu hiç okumadan doğru adayı seçebilen bir ölçüt, veri kümesinin biçimbilimi değil üretim artefaktını ölçtüğü anlamına gelir. Şans düzeyi 9.1%.

| ölçüt | değer | şans |
|---|---|---|
| blind_longest_is_gold | 18.1% | 9.1% |
| blind_most_tokens_is_gold | 30.7% | 9.1% |
| sparse_char3gram_top1_is_gold | 31.4% | 9.1% |

Seyrek temel çizgi pozitifi ilk sıraya koyduğunda aradaki fark: medyan **0.0174**, en yüksek 0.1463. Oranın tek başına anlamı yoktur: 0.005 farkla kazanmak ile 0.30 farkla kazanmak farklı şeylerdir; medyan farkın sıfıra yakın olması, seyrek ölçütün neredeyse berabere kalan adaylar arasında tahmin yürüttüğünü gösterir.

Kritik sözcük çiftinin kaynağı: `{'derived': 140, 'reported': 302, 'none': 33}` (`reported` = üreticinin hedeflediği karşıtlık doğrulandı; `derived` = çift metinlerden türetildi, hedef özellikle birebir aynı olmayabilir).


### Seyrek (char-3gram) temel çizgi — alt tür bazında ikili doğruluk

Modelsiz bir sözcük-örtüşmesi ölçütü, altın adayı her bir olumsuz adayın üstüne koyabiliyor mu? Rastgele = %50.

| alt tür | v2.0 | v1.3.1 (test) | okuma |
|---|---|---|---|
| morph_counterfactual | 64.6 | 23.5 | kısmen sözcüksel olarak ayrılabilir |
| partial_trap | 59.4 | 66.7 | rastgeleye yakın: sözcüksel sinyal yok (iyi) |
| same_feature_wrong_content | 89.7 | 25.5 | **büyük ölçüde sözcüksel olarak çözülebilir (zayıf)** |
| state_variant | 63.8 | 34.3 | kısmen sözcüksel olarak ayrılabilir |
| easy_negative | 99.5 | 97.6 | tasarımı gereği kolay |

> **Bilinen zayıflık.** v2.0'da `same_feature_wrong_content` seyrek ölçütle neredeyse tamamen çözülebiliyor, `morph_counterfactual` ise rastgelenin üzerinde. v1.3.1'de her ikisi de rastgelenin ALTINDA, yani sözcük örtüşmesi yanlış adayı işaret ediyor. Nedeni: v1.3.1'de pozitif, sorgunun bağımsız bir yeniden ifadesidir; v2.0'da ise sorgunun içerik sözcüklerini aynı sırada koruyan daha yakın bir yeniden yazımdır, dolayısıyla yüzey benzerliği doğru adayı ele veriyor. Bu, v2.0'ın bir EĞİTİM kümesi olarak kullanılabilirliğini ortadan kaldırmaz, ama tek başına bir ölçüt (benchmark) olarak kullanılmasını engeller — ölçüt v1.3.1'dir.


Ortalama karakter uzunlukları:

| rol/alt tür | ort. uzunluk |
|---|---|
| positive | 131.7 |
| hard_negative | 128.9 |
| easy_negative | 112.2 |
| morph_counterfactual | 128.6 |
| same_feature_wrong_content | 128.5 |
| partial_trap | 131.9 |
| state_variant | 127.8 |

> Karşılaştırma: v1.3.1'de `blind_longest_is_gold` **%50** (şans %9). Yani insan denetiminden geçmiş test kümesinde pozitifler sistematik olarak daha uzun. Bu, v2.0 için düzeltilmesi hedeflenen bilinen bir artefakttır ve üretim isteminde açıkça yasaklanmıştır.
> Uyarı (Feng vd. 2019): kör ölçütün BAŞARISIZ olması, veri kümesinin artefaktsız olduğunu KANITLAMAZ. Bu sayılar bir hipotez testidir, temiz kâğıt değil.

## Kapsam

- hedef biçimbirim özelliği: 75/75
- eksik özellikler: yok

| özellik | adet |
|---|---|
| TENSE.PERS.NEG | 13 |
| DESID | 12 |
| PTCP.OBJ | 12 |
| PLUPRF | 12 |
| NMLZ.ECEK | 12 |
| NMLZ.ME | 11 |
| EVID.COND.NEG | 10 |
| RECP.CAUS | 10 |
| POSS.PL.ABL | 9 |
| NMLZ.DIK | 9 |
| EVID.POSS.NEG | 9 |
| POSS.1SG | 8 |
| PST.PROG | 8 |
| PTCP.FUT | 8 |
| COND | 8 |
| PL.POSS.CASE | 8 |
| CVB.SINCE | 8 |
| CAUS.CAUS | 8 |
| CVB.NEGINS | 7 |
| CAUS.REFL.NEG | 7 |
| PRIV.VS.NEG | 7 |
| CVB.WHEN | 7 |
| ALLO.ABL | 7 |
| POSS.3SG | 7 |
| PTCP.SUBJ | 7 |
| HAB.PST | 7 |
| FUT.PST | 7 |
| CVB.ASLONG | 7 |
| PL | 7 |
| CAUS.PASS.NEG | 6 |
| DAT | 6 |
| AOR | 6 |
| EQU | 6 |
| CVB.ABOUTTO | 6 |
| INS | 6 |
| LOC | 6 |
| PST | 6 |
| ACC | 6 |
| FUT | 6 |
| AGT | 6 |
| NMLZ.CASE.CNTR | 6 |
| POSS.2PL | 6 |
| REL.KI | 5 |
| NEC | 5 |
| POSS.1PL | 5 |
| ABST | 5 |
| REFL | 5 |
| PASS | 5 |
| NEG.AOR | 5 |
| PRIV | 5 |
| CVB.WHILE | 5 |
| ALLO.DAT | 5 |
| NMLZ.MEK | 5 |
| POSS.3PL | 5 |
| PRS.PROG | 5 |
| GEN | 5 |
| CVB.AND | 5 |
| POSS.2SG | 5 |
| CVB.BY | 5 |
| ABL | 5 |
| PROP | 4 |
| PRSM | 4 |
| IMP.3 | 4 |
| CVB.WITHOUT | 4 |
| ABIL.COND | 4 |
| CAUS | 4 |
| CNTR | 4 |
| PRF.EVID | 4 |
| ABIL | 4 |
| OPT | 4 |
| NEG.ABIL | 4 |
| RECP | 4 |
| VBLZ | 3 |
| NEG | 2 |
| DISTR | 2 |

## Sızıntı muhasebesi

- üreticiye örnek olarak gösterilen v1.3.1 öğeleri: `q03_negation`, `q103_dilution_person`, `q108_evid_counterfactual`, `q113_min_person_number`
- **test kümesi skorları raporlanırken bu 4 öğe hariç tutulmalıdır** (50 öğelik test kümesinin %8'si)
- üretilen sorguların v1.3.1'e en yüksek benzerliği: 0.3 (eşik 0.60)
