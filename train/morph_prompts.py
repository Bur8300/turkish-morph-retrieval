#!/usr/bin/env python
"""Prompts and response schemas for the v2.0 Turkish morphological dataset generation.

Three prompts, each with a strict Gemini `responseSchema` so a malformed response is impossible:

    build_generation_prompt  -> one complete item (query + 11 candidates)
    build_judge_prompt       -> a BLIND review of one item
    build_repair_prompt      -> minimal-edit fix for a flagged item

Two design choices are worth reading before editing anything here.

**The conventions are copied verbatim from v1.3.1, not paraphrased.** That block survived two
human QC rounds; it is the specification. `EXEMPLAR_IDS` likewise pulls real items from v1.3.1 as
few-shot examples rather than inventing new ones — but see `EXEMPLAR_IDS` for the leakage
accounting that requires.

**The judge never sees the intended labels.** It is asked to decide, for each candidate
independently, whether that candidate answers the query and — if not — *why not*, choosing from a
list of reasons that happen to map onto our subtypes. The comparison against what the generator
intended is then done in Python. This gives a genuine blind single-gold probe and a subtype
audit from one API call, with no anchoring on the generator's own labels. The alternative (show
the labels, ask "are these right?") is what the literature calls out as the standard way an
LLM judge rubber-stamps its own family's output.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "legacy_test_data" / "morph_eval_set_v1.3.1_review_reviewer_C_fixed.json"

# v1.3.1 items used as few-shot exemplars. LEAKAGE ACCOUNTING: these are test-set items shown to
# the generator, so they must be excluded when reporting held-out test scores, and their content
# words go on the permanent ban list. Kept to FOUR (8% of the 50-item test set) and fixed, so the
# exclusion is a known reportable constant. Four is enough because the exemplars teach structure,
# which is identical across tiers — only the difficulty of the contrast differs, and that is
# specified directly in the slot rather than learned from examples.
EXEMPLAR_IDS = {
    "minimal": ["q113_min_person_number"],
    "hard": ["q108_evid_counterfactual"],
    "standard": ["q03_negation"],
    "long": ["q103_dilution_person"],
}
ALL_EXEMPLAR_IDS = sorted({q for ids in EXEMPLAR_IDS.values() for q in ids})

_V1_CACHE = {}


def load_v1():
    if "data" not in _V1_CACHE:
        _V1_CACHE["data"] = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return _V1_CACHE["data"]


def v1_conventions_block():
    """The v1.3.1 conventions, verbatim. This is the spec — do not rewrite it here."""
    return json.dumps(load_v1()["conventions"], ensure_ascii=False, indent=2)


def _item_by_id(qid):
    for it in load_v1()["items"]:
        if it["query_id"] == qid:
            return it
    raise KeyError(qid)


def _format_exemplar(item):
    """Render a v1.3.1 item as a compact example in the same shape the model must produce."""
    cands = []
    for c in item["candidates"]:
        cands.append({k: v for k, v in
                      (("role", c["role"]), ("subtype", c.get("subtype", "")),
                       ("text", c["text"]), ("note", c["note"])) if v})
    return json.dumps({"query": item["query"], "candidates": cands},
                      ensure_ascii=False, indent=1)


def exemplars_for(tier, passage_length):
    """Two exemplars: the tier-matched one, plus the long-passage one when length is `long`
    (that item is the only one showing the mid-passage-dilution layout) else the standard one."""
    ids = list(EXEMPLAR_IDS.get(tier, EXEMPLAR_IDS["standard"]))
    ids += EXEMPLAR_IDS["long"] if passage_length == "long" else EXEMPLAR_IDS["standard"]
    seen, out = set(), []
    for q in ids:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return [_item_by_id(q) for q in out[:2]]


# --------------------------------------------------------------------------- error taxonomy
# Derived from v1.3.1's own `fix_log`: every entry there is a defect a careful human reviewer
# actually had to repair, which makes it an empirically grounded list of what goes wrong rather
# than a guess. Fed back to the generator as hard prohibitions.
FIX_LOG_PROHIBITIONS = """\
AŞAĞIDAKİ HATALAR YASAKTIR. Bunların her biri, bu veri kümesinin önceki sürümünde bir insan
denetçinin elle düzeltmek zorunda kaldığı gerçek hatalardır:

1. YASAK — pozitifte içerik sözcüğünü eşanlamlısıyla değiştirmek.
   Sorgu "Sözleşmeyi ... göndermiştim" diyorsa pozitif de "sözleşme" demelidir; "belge" demek
   ölçüme sözcük-eşleşmesi sorunu enjekte eder ve ek duyarlılığı ölçümünü bozar.
   Aynı şekilde kritik içerik sözcüğü daha zayıf/gevşek bir yakın anlamlıya indirgenemez.

2. YASAK — sorgunun olumsuz/olumlu iddiasını neredeyse birebir tekrar eden ikinci bir aday.
   Bu, ikinci bir doğru cevap (çift-altın) yaratır ve soruyu ölçülemez hâle getirir.

3. YASAK — "partial_trap" adayının ters bir sonuç içermemesi.
   partial_trap KISMEN örtüşen ama SONUCU TERS olan bir olay anlatmalıdır. Sadece konuyla ilgili
   olup cevap vermeyen bir cümle partial_trap değil, easy_negative'dir.

4. YASAK — sayı, katılımcı veya zaman değişikliğini "morph_counterfactual" diye etiketlemek.
   morph_counterfactual YALNIZCA hedef ekin kendisi değiştiği için anlamı ters dönen adaydır.
   Katılımcı/sayı/olay değişimi "state_variant"tır.

5. YASAK — birden fazla adayın birincil minimal karşıtlık olması.
   Tam olarak BİR aday morph_counterfactual olmalıdır; diğer yakın adaylar state_variant'tır.

6. YASAK — easy_negative'lerin aslında sorguyu yanıtlaması. easy_negative konu komşusu olmalı ama
   sorgunun sorduğu bilgiyi HİÇ taşımamalıdır.
"""

# WHY THE HARD NEGATIVES ARE ANCHORED ON THE QUERY (v2.1)
#
# v2.0 built every hard negative as a minimal edit of the POSITIVE. v1.3.1 — the human-built set —
# anchors them on the QUERY instead, and that one difference is what makes it hard. Median
# char-3gram similarity to the query:
#
#                                  v1.3.1 (human)   v2.0 (this generator, before the fix)
#     positive                          0.198              0.386
#     morph_counterfactual              0.333              0.380
#     same_feature_wrong_content        0.398              0.279
#
# In v1.3.1 the positive is the LEAST query-like of the serious candidates, so lexical overlap
# points at a wrong answer and a bag-of-trigrams scorer lands at 23.5 / 25.5 pairwise accuracy
# against a 50 random baseline. v2.0 inverted the ordering and scored 64.6 / 89.7 — the string
# matcher was doing the encoder's job.
#
# So: negatives are minimal edits of the QUERY (they inherit its surface), and the positive is an
# independent re-telling (it does not). One exception, which v1.3.1 also makes: at `minimal` tier
# the counterfactual must be a one-word edit of the POSITIVE, because that is what a minimal pair
# means there — measured sim(positive, cf) is 0.41-0.92 at that tier versus 0.13-0.39 for
# sim(query, cf).
# ADIM 4 has two mutually exclusive recipes, and only one is ever shown to the model.
#
# The first v2.1 attempt put both in the prompt behind an "if tier == minimal" branch. Half of all
# minimal slots followed the standard/hard recipe anyway and failed the minimal-pair check
# (2/8 passing in a targeted probe). A conditional buried in a long instruction is the kind of
# thing a model drops; a prompt that only contains the applicable recipe has nothing to drop.
_STEP4_QUERY_ANCHORED = """\
ADIM 4 — Sert olumsuzların ÇEKİRDEKLERİNİ (`core`) SORGUDAN TÜRET, pozitiften değil. Ortak
  çerçeveyi çekirdeğe yazma. Her biri requirements listesinden TAM OLARAK BİR maddeyi ihlal etsin
  ve hangisini ihlal ettiğini `violated_requirement` alanına yaz:
  - morph_counterfactual (TAM 1 adet): çekirdek olarak SORGUYU aynen kopyala, YALNIZCA hedef eki
    değiştir, diğer bütün sözcükleri olduğu gibi bırak. Anlam sorgunun tersi olsun.
    ÖNCELİK SIRASI — bu sırayı bozma:
      (1) cümle DİLBİLGİSEL olarak kusursuz Türkçe olacak,
      (2) anlam sorgunun tersi olacak,
      (3) sorgudan farkı en az olacak.
    Eki değiştirmek cümlenin başka bir yerini dilbilgisel olarak bozuyorsa (örneğin "-seydi"
    koşul cümlesi "-irdi" ile eşlenir; "esnetilebilseydi yapamadı" BOZUKTUR, doğrusu
    "esnetilebilseydi yapamazdı"), o kısmı da en az düzeyde düzelt. Bozuk Türkçe bir aday,
    yakın bir aday olmaktan daha kötüdür: model onu dilbilgisinden eler, ekten değil.
  - same_feature_wrong_content (1 adet): çekirdek olarak SORGUYU al, betimlenen NESNE/OLAY adını değiştir
    (örn. "sipariş formları" -> "vize randevuları"), hedef eki ve diğer bütün sözcükleri aynen
    bırak. KİŞİ veya YER adını değiştirme — katılımcı/yer değişimi `state_variant`tır, bu değil.
  - partial_trap (1 adet): kısmen örtüşen bir olay ama SONUÇ TERS. Bu adayı sorgudan türetmen
    gerekmez; kendi cümlesi olabilir.
  - state_variant (2 adet): sorgunun katılımcısını, zamanını veya durumunu değiştir."""

_STEP4_POSITIVE_ANCHORED = """\
ADIM 4 — Bu öğe `minimal` katmandadır. Bu katmanda ÖNCE pozitifin çekirdeğini yaz (ADIM 5'teki
  kurallara göre), SONRA aşağıdakileri ondan türet. Burada ölçülen şey şudur: POZİTİF ile
  MORPH_COUNTERFACTUAL, TEK BİR SÖZCÜK dışında birebir aynı iki cümledir ve o tek sözcükte de
  yalnızca hedef ek değişmiştir. Sorgu ise ikisinden de farklı sözcüklerle kurulmuş, bağımsız
  bir ifadedir. Sırayla:
  - morph_counterfactual (TAM 1 adet): POZİTİFİN ÇEKİRDEĞİNİ AYNEN KOPYALA, sonra
    içindeki TEK bir sözcüğün hedef ekini değiştir. Başka HİÇBİR sözcüğe dokunma — sözcük
    sırasını, noktalama işaretlerini, bağlaçları, hiçbirini değiştirme.
    Doğru: pozitif "O evi kendi birikimimle aldım." -> counterfactual "O evi kendi
    birikimimle aldın."   (tek sözcük: aldım/aldın)
    Yanlış: counterfactual'ı SORGUDAN türetmek. Bu katmanda sorgu çapa DEĞİLDİR.
    Yanlış: cümleyi yeniden yazmak, sözcük eklemek/çıkarmak.
    Ek değişimi cümlenin başka bir yerini dilbilgisel olarak bozuyorsa, pozitifi baştan öyle kur
    ki tek sözcüklük değişim dilbilgisel kalsın.
  - same_feature_wrong_content (1 adet): yine pozitifin çekirdeğini kopyala, ama bu kez betimlenen
    NESNE/OLAY adını değiştir; hedef ek aynı kalsın. KİŞİ/YER değiştirme.
  - partial_trap (1 adet): kısmen örtüşen bir olay ama SONUÇ TERS. Kendi cümlesi olabilir.
  - state_variant (2 adet): katılımcıyı, zamanı veya durumu değiştir.
  Her biri requirements listesinden TAM OLARAK BİR maddeyi ihlal etsin; hangisini ihlal ettiğini
  `violated_requirement` alanına yaz."""

# How much shared context each passage length gets. The frame is what makes every candidate the
# same length, so it scales with the target passage size; at `short` there is nothing to pad and an
# empty frame is correct.
FRAME_RULE = {
    "short": ("Bu öğede çerçeve KULLANMA: `frame_before` ve `frame_after` boş string olsun. "
              "Adaylar tek cümlelik olacak."),
    "medium": ("`frame_before` olarak olayın geçtiği ortamı anlatan TEK bir cümle yaz; "
               "`frame_after` boş kalsın."),
    "long": ("`frame_before` olarak BİR, `frame_after` olarak BİR cümle yaz. Kritik ek böylece "
             "uzun bir pasajın ortasına gömülmüş olur — bu öğenin ölçmek istediği şey tam olarak "
             "budur: tek bir ekin sinyali dolgu metnin içinde kayboluyor mu?"),
}

# Restated at the point of use, next to the slot's tier, so the recipe and the slot metadata agree.
TIER_RULE = {
    "standard": ("morph_counterfactual'ı SORGUDAN türet: sorgunun kendisi, yalnızca hedef ek "
                 "değişmiş hâli. Pozitif ise sorgunun bağımsız yeniden anlatımı olsun."),
    "hard": ("morph_counterfactual'ı SORGUDAN türet: sorgunun kendisi, yalnızca hedef ek değişmiş "
             "hâli. Pozitif ise sorgunun bağımsız yeniden anlatımı olsun."),
    "minimal": ("morph_counterfactual'ı POZİTİFTEN türet: pozitiften yalnızca TEK SÖZCÜK farklı "
                "olsun, o sözcükte de yalnızca hedef ek değişsin. Sorgu ikisinden de farklı "
                "sözcüklerle kurulmuş bağımsız bir ifade olsun."),
}

SYSTEM_INSTRUCTION = """\
Sen Türkçe biçimbilim (morfoloji) uzmanı bir hesaplamalı dilbilimcisin. Türkçe için bir
BİÇİMBİRİM DUYARLILIĞI ERİŞİM (retrieval) veri kümesi üretiyorsun.

Bu veri kümesinin tek amacı şudur: bir erişim modelinin, iki metin arasındaki farkın YALNIZCA bir
ek olduğu durumda doğru olanı seçip seçemediğini ölçmek. Bu yüzden ürettiğin her öğede ayırt edici
sinyal SÖZCÜK DAĞARCIĞINDA DEĞİL, EKTE olmalıdır.

Türkçen kusursuz olmalı: ünlü uyumu, ünsüz benzeşmesi/yumuşaması ve kaynaştırma ünsüzleri
(-y-, -n-, -s-) doğru uygulanmalı. Ürettiğin biçimler gerçek, kullanılan Türkçe biçimler olmalı;
uydurma ek dizilimi kullanma.

Yalnızca JSON döndür. Açıklama, giriş cümlesi, markdown kod bloğu ekleme.
"""

# --------------------------------------------------------------------------- generation
GENERATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "requirements": {
            "type": "ARRAY",
            "description": "Pozitifin sorguyu neden yanıtladığını oluşturan ayrı ayrı bilgi koşulları.",
            "items": {"type": "STRING"},
        },
        "query": {"type": "STRING"},
        "frame_before": {
            "type": "STRING",
            "description": ("11 adayın HEPSİNİN başına aynen eklenecek ortak bağlam. Kısa pasajlarda "
                            "boş bırak. Bir kez yaz — kod her adayın başına kendisi ekleyecek."),
        },
        "frame_after": {
            "type": "STRING",
            "description": "11 adayın HEPSİNİN sonuna aynen eklenecek ortak bağlam. Boş olabilir.",
        },
        "critical_word_query": {
            "type": "STRING",
            "description": "SORGUDA hedef eki taşıyan TEK sözcük, sorguda geçtiği hâliyle.",
        },
        "critical_word_counterfactual": {
            "type": "STRING",
            "description": "morph_counterfactual çekirdeğindeki karşıt biçim, orada geçtiği hâliyle.",
        },
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "role": {"type": "STRING",
                             "enum": ["positive", "hard_negative", "easy_negative"]},
                    "subtype": {"type": "STRING",
                                "enum": ["none", "morph_counterfactual", "same_feature_wrong_content",
                                         "partial_trap", "state_variant"]},
                    "core": {
                        "type": "STRING",
                        "description": ("YALNIZCA bu adaya özgü cümle. Ortak çerçeveyi (frame_before/"
                                        "frame_after) BURAYA TEKRAR YAZMA — kod ekleyecek."),
                    },
                    "note": {"type": "STRING",
                             "description": "Türkçe, tek satır: bu adayın neden bu rolde olduğu."},
                    "violated_requirement": {
                        "type": "STRING",
                        "description": ("hard_negative için: yukarıdaki requirements listesinden "
                                        "ihlal edilen TEK koşul. positive/easy_negative için boş."),
                    },
                },
                "required": ["role", "subtype", "core", "note", "violated_requirement"],
                "propertyOrdering": ["role", "subtype", "core", "note", "violated_requirement"],
            },
        },
        "self_check": {
            "type": "OBJECT",
            "properties": {
                "query_asserts_state": {"type": "BOOLEAN"},
                "positive_keeps_content_words": {"type": "BOOLEAN"},
                "counterfactual_is_single_suffix_contrast": {"type": "BOOLEAN"},
                "single_gold": {"type": "BOOLEAN"},
                "all_subtypes_present": {"type": "BOOLEAN"},
                "turkish_morphology_correct": {"type": "BOOLEAN"},
                "notes": {"type": "STRING"},
            },
            "required": ["query_asserts_state", "positive_keeps_content_words",
                         "counterfactual_is_single_suffix_contrast", "single_gold",
                         "all_subtypes_present", "turkish_morphology_correct", "notes"],
            "propertyOrdering": ["query_asserts_state", "positive_keeps_content_words",
                                 "counterfactual_is_single_suffix_contrast", "single_gold",
                                 "all_subtypes_present", "turkish_morphology_correct", "notes"],
        },
    },
    # propertyOrdering doubles as the generation order the model works in: query first, then the
    # shared frame, then the candidate cores. The positive's core is written last among the
    # candidates (enforced in the prompt text) because its length is the free variable — the
    # negatives' lengths are already fixed by being edits of the query.
    "required": ["requirements", "query", "frame_before", "frame_after", "critical_word_query",
                 "critical_word_counterfactual", "candidates", "self_check"],
    "propertyOrdering": ["requirements", "query", "frame_before", "frame_after",
                         "critical_word_query", "critical_word_counterfactual",
                         "candidates", "self_check"],
}


def build_generation_prompt(slot, ban_words=(), length_spec=None):
    """One slot -> one prompt string.

    `ban_words` is the rolling set of content nouns already saturated in accepted items. This is
    the main defence against the distributional narrowness that LLM synthesis is known for: without
    it the model converges on a handful of favourite scenarios within ~50 items.
    """
    ex = exemplars_for(slot["tier"], slot["passage_length"])
    ex_block = "\n\n".join(f"ÖRNEK {i+1} (tier={e['tier']}, layer={e['layer']}, "
                           f"ek={e['ek_turu']}):\n{_format_exemplar(e)}"
                           for i, e in enumerate(ex))
    ban = ", ".join(sorted(ban_words)[:60])
    ent = slot["seed_entities"]
    ls = length_spec or {}
    lo, hi = ls.get("chars", (40, 220))
    step4 = (_STEP4_POSITIVE_ANCHORED if slot["tier"] == "minimal"
             else _STEP4_QUERY_ANCHORED)
    frame_rule = FRAME_RULE[slot["passage_length"]]
    # The query and every core must live in the same band, so that a negative (a one-word edit of
    # the query) and the positive (a re-telling of it) come out the same length. The frame is added
    # to all 11 candidates alike, so it is excluded from the band the model is asked to hit.
    frame_budget = {"short": 0, "medium": 60, "long": 120}[slot["passage_length"]]
    core_lo, core_hi = max(35, lo - frame_budget), max(60, hi - frame_budget)

    return f"""{FIX_LOG_PROHIBITIONS}

=== SÖZLEŞME (bu veri kümesinin kuralları — birebir uyulacak) ===
{v1_conventions_block()}

=== ÖRNEKLER (yapıyı göster; içeriklerini KOPYALAMA) ===
{ex_block}

=== ŞİMDİ ÜRETECEĞİN ÖĞENİN ÖZELLİKLERİ ===
- Hedef biçimbirim özelliği : {slot['target_feature']}
- Ek türü                    : {slot['ek_turu']}
- Ölçülecek karşıtlık        : {slot['contrast']}
- Katman (layer)             : {slot['layer']}  ("single" = tek ek, "chain" = iki+ ekin bileşimi)
- Zorluk (tier)              : {slot['tier']}
  >> {TIER_RULE[slot['tier']]}
- Alan                       : {slot['domain_label']} — {slot['domain_desc']}
- Bakış açısı                : {slot['perspective']}
- Kişi                       : {slot['person']}
- Pasaj uzunluğu             : {ls.get('label', slot['passage_length'])}
  {ls.get('instruction', '')}

Şu tohum ögelerden EN AZ İKİSİNİ kullan (aynen ya da çekimli hâliyle):
  kişiler: {', '.join(ent['names'])} | yer: {ent['place']} | kurum: {ent['org']}
  nesneler: {', '.join(ent['objects'])}

{f"KULLANMA (bu içerik sözcükleri fazlaca kullanıldı, yenilerini seç): {ban}" if ban else ""}

=== ÜRETİM YÖNTEMİ (sırayla uygula) ===
ADIM 1 — Sorguyu yaz. Bir durumu İDDİA eden, doğal, gündelik bir anımsama olsun. Evet/hayır
  sorusu yazma, "kaydı arıyorum" / "bul" gibi arama diline girme. Hedef ek sorguda mutlaka
  bulunsun.
  SORGU UZUNLUĞU — {core_lo}-{core_hi} KARAKTER. Bu bir üslup tercihi değil, ölçümün temelidir:
  sert olumsuzlar sorgunun tek bir ekini/adını değiştirerek türetilecek, yani uzunlukları
  sorgununkiyle aynı olacak. Pozitif ise aynı olayın yeniden anlatımıdır ve o da bu uzunlukta
  olmalıdır. Sorguyu kısa bir tek cümle olarak yazarsan olumsuzlar kısa, pozitif uzun kalır;
  o zaman sorguyu hiç okumayan bir model "en uzun adayı seç" diyerek doğru cevabı bulur ve öğe
  biçimbilimi değil uzunluğu ölçer. Gerekiyorsa sorguya ikinci bir yan cümle ekleyerek bu
  uzunluğa çıkar — ama sorgu tek bir olayı anlatmaya devam etsin.

ADIM 2 — `requirements`: Pozitifin bu sorguyu yanıtlaması için sağlaması gereken bilgi koşullarını
  ayrı ayrı, madde madde yaz (genelde 3-5 madde). En az bir madde HEDEF EKİN taşıdığı anlamı
  açıkça ifade etmeli. Örneğin: "eylemin gerçekleşmediği bilgisi", "eylemi yapanın konuşmacı
  olduğu bilgisi", "olayın dünkü toplantı olduğu bilgisi".

ADIM 2b — ORTAK ÇERÇEVE: `frame_before` ve `frame_after`.
  {frame_rule}
  Bu iki metin 11 adayın HEPSİNE aynen eklenecek — sen bir kez yazıyorsun, kodu ekleme işini
  kendisi yapıyor. Bu yüzden çerçeveyi adayların `core` alanına TEKRAR YAZMA.
  Çerçeve, sorunun cevabını içermemeli: hiçbir adayı doğru ya da yanlış yapmayan, olayın geçtiği
  ortamı anlatan nötr cümleler olsun. (Ortak çerçeve sayesinde bütün adaylar aynı uzunlukta olur
  ve sorguyu hiç okumayan bir model onları uzunluklarından ayırt edemez.)

{step4}

ADIM 4b — YÜZEY BENZERLİĞİ SIRALAMASI. Bu kural veri kümesinin bütün amacıdır:

    morph_counterfactual ≈ same_feature_wrong_content  >  state_variant  >  POZİTİF  >  easy_negative
                        (sorguya en benzer)                                        (en benzemez)

  Yani POZİTİF, sorguya sözcük olarak EN ÇOK BENZEYEN aday OLMAMALIDIR. Sert olumsuzlar sorgunun
  yüzeyini devraldığı için, sözcük örtüşmesi YANLIŞ adayı işaret eder; doğru cevabı yalnızca eki
  okuyan bir model bulabilir. Pozitif en benzer aday olursa öğe, biçimbilime hiç bakmayan bir
  sözcük eşleştiricisi tarafından çözülür ve ölçüm değerini yitirir.

  Bu sıralama `standard` ve `hard` katmanları içindir. `minimal` katmanında sert olumsuzlar
  pozitiften türediği için sorguya benzerlik sıralaması serbesttir; orada aranan tek şey
  pozitif ile counterfactual'ın TEK SÖZCÜK farkla ayrılan iki cümle olmasıdır.

  ÖRNEK (v1.3.1'den, q03_negation — istenen davranış):
    sorgu   : "Derya dünkü toplantıya katılmamıştı."
    pozitif : "Derya dünkü toplantıda yoktu; yoklamada adı okununca yanıt veren olmadı."
              (bağımsız anlatım: "katılmamıştı" yerine "yoktu ... yanıt veren olmadı")
    morph_c.: "Derya dünkü toplantıya katıldı."          <- sorgunun kendisi, ek değişti
    same_f. : "Derya dünkü seminere katılmadı."          <- sorgunun kendisi, ad değişti
  Görüldüğü gibi iki sert olumsuz sorgunun sözcüklerini taşır, pozitif taşımaz.

ADIM 5 — POZİTİFİN ÇEKİRDEĞİNİ EN SON YAZ. Bu sıra bilinçlidir: sert olumsuzların uzunluğu
  sorgudan türedikleri için zaten bellidir, dolayısıyla uyum sağlaması gereken taraf pozitiftir.
  * Önce yazdığın morph_counterfactual çekirdeğinin karakter sayısını say. Pozitifin çekirdeği
    o sayının %90-%110'u arasında olsun. Daha uzun yazma.
  * Sorgunun içerik sözcüklerini (kişi, nesne, yer) AYNEN KORU.
  * Ama sorgunun CÜMLE YAPISINI VE FİİL KURULUŞUNU KOPYALAMA. Sorgudan 4 sözcükten uzun hiçbir
    dizi pozitifte birebir tekrar etmesin. Sorguyu alıp sonuna cümle eklemek YASAKTIR.
  * Hedef ekin taşıdığı anlamı BAŞKA bir dilbilgisel yolla ver. Şu dönüşümlerden EN AZ BİRİNİ
    uygula — hepsi aynı uzunlukta kalmanı sağlar, sadece kuruluşu değiştirir:
      - eylemi adlaştır:      "çıkarılmıştı"        -> "çıkarıldığı görüldü"
      - çatıyı çevir:         "Derya raporu yolladı" -> "rapor Derya tarafından yollandı"
      - yan cümleyi öne al:   "X yapınca Y oldu"     -> "Y'nin nedeni X'ti"
      - olayı sonucundan anlat: "katılmamıştı"       -> "yoklamada adı okununca yanıt gelmedi"
      - eş anlamlı FİİL kullan (içerik ADLARINI değil!)
  * Sorgudaki ANA FİİLİ aynı çekimle tekrar kullanma; en az bir dönüşüm görünür olsun.
  * Tüm requirements maddelerini karşılasın.

ADIM 6 — 5 adet easy_negative çekirdeği yaz: aynı konu evreninden, doğal, ama sorgunun sorduğu
  bilgiyi taşımayan cümleler. Uzunlukları diğer çekirdeklerle aynı olsun.

ADIM 7 — `critical_word_query`: SORGUDA hedef eki taşıyan TEK sözcük, sorguda geçtiği hâliyle.
  `critical_word_counterfactual`: morph_counterfactual adayında bu sözcüğün karşılığı, o metinde
  geçtiği hâliyle. Bu ikisi AYNI KÖKTEN olmalı ve yalnızca hedef ek bakımından ayrılmalıdır
  (örn. "katılmamıştı" / "katıldı"). Sözcükleri uydurma — ikisi de ilgili metinde AYNEN geçmeli.

ADIM 8 — `self_check`: kendi ürettiğini yukarıdaki kurallara göre denetle. Bir madde sağlanmıyorsa
  önce öğeyi DÜZELT, sonra true yaz. Yanlış bilinen bir öğeyi true işaretleme.

=== BİÇİM ===
Tam olarak 11 aday: 1 positive + 5 hard_negative + 5 easy_negative.

ÇEKİRDEK UZUNLUKLARI — BU KURAL ZORUNLUDUR:
Ortak çerçeve 11 adayın hepsine kod tarafından ekleneceği için uzunluk farkı YALNIZCA
çekirdeklerden gelir. Bu yüzden 11 çekirdeğin karakter sayısı birbirine yakın olmalı; özellikle
pozitifin çekirdeği en uzun olan OLMAMALIDIR. Aşağıdaki aralık tek bir adayın TAMAMI için
(çerçeve + çekirdek) geçerlidir:
11 adayın HEPSİ {lo}-{hi} karakter aralığında olsun. Özellikle positive adayı diğerlerinden
UZUN OLMAMALIDIR; easy_negative'ler de kısa kesilmemelidir.
Neden: bu veri kümesinin önceki sürümünde pozitif adaylar sistematik olarak daha uzundu ve
sorguyu hiç okumadan "en uzun adayı seç" diyen kör bir model öğelerin %50'sini doğru biliyordu
(rastgele oran %9). Bu, ek duyarlılığını değil uzunluk artefaktını ölçmek demektir. Yazdıktan
sonra adayların uzunluklarını gözden geçir ve gerekiyorsa easy_negative'leri uzat, pozitifi kısalt.
"""


# --------------------------------------------------------------------------- judge
WHY_NOT_ENUM = ["none", "minimal_suffix_contrast", "same_pattern_wrong_content",
                "partial_overlap_opposite_outcome", "different_state_or_participant", "unrelated"]

# judge's inferred reason -> our subtype vocabulary
# "none" is deliberately absent: the schema uses it as the sentinel for "this candidate answers
# the query", so a .get() miss is the correct behaviour.
WHY_NOT_TO_SUBTYPE = {
    "minimal_suffix_contrast": "morph_counterfactual",
    "same_pattern_wrong_content": "same_feature_wrong_content",
    "partial_overlap_opposite_outcome": "partial_trap",
    "different_state_or_participant": "state_variant",
    "unrelated": "easy_negative",
}

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "answers_query": {"type": "BOOLEAN"},
                    "why_not": {"type": "STRING", "enum": WHY_NOT_ENUM},
                    "reason": {"type": "STRING"},
                },
                "required": ["id", "answers_query", "why_not", "reason"],
                "propertyOrdering": ["id", "answers_query", "why_not", "reason"],
            },
        },
        "query_is_assertion": {"type": "BOOLEAN"},
        "turkish_is_correct": {"type": "BOOLEAN"},
        "morphology_errors": {
            "type": "ARRAY",
            "description": "Ünlü uyumu / ek dizilimi hatası taşıyan aday id'leri.",
            "items": {"type": "STRING"},
        },
        "fluency_outliers": {
            "type": "ARRAY",
            "description": "Üslup ya da uzunluk bakımından diğerlerinden ayrışan aday id'leri.",
            "items": {"type": "STRING"},
        },
        "notes": {"type": "STRING"},
    },
    "required": ["candidates", "query_is_assertion", "turkish_is_correct",
                 "morphology_errors", "fluency_outliers", "notes"],
    "propertyOrdering": ["candidates", "query_is_assertion", "turkish_is_correct",
                         "morphology_errors", "fluency_outliers", "notes"],
}


def build_judge_prompt(item, seed=0):
    """Blind review. Candidates are shuffled and stripped of role, subtype and generator notes.

    The judge is never told which candidate is the gold, so `answers_query` is a real
    single-gold probe, and `why_not` is an independent subtype opinion that Python then compares
    against what the generator intended.
    """
    cands = list(item["candidates"])
    random.Random(seed).shuffle(cands)
    listing = "\n".join(f"[{c['id']}] {c['text']}" for c in cands)

    return f"""Aşağıda Türkçe bir SORGU ve numaralandırılmış ADAY metinler var. Adayların hangisinin
doğru cevap olduğu sana SÖYLENMEDİ; buna kendin karar vereceksin.

SORGU: {item['query']}

ADAYLAR:
{listing}

ÖNCE ŞUNU BİL — bu veri kümesi bilerek TERS kurulmuştur:
  * DOĞRU cevap, sorgudan FARKLI SÖZCÜKLERLE yazılmıştır. Aynı olayı başka bir cümle kuruluşuyla,
    başka fiillerle anlatır. Sözcükleri sorgudan farklı diye bir adayı ELEME — bu, doğru cevabın
    tanımıdır.
  * YANLIŞ adayların çoğu sorguya neredeyse BİREBİR benzer; aralarındaki tek fark bir ektir ve o
    ek anlamı tersine çevirir.
Yani sözcük benzerliği bu görevde YANILTICIDIR. Kararını yalnızca şu soruya göre ver:
"Bu aday, sorgunun iddia ettiği durumun GERÇEKLEŞTİĞİNİ söylüyor mu?"
Sözcükler tutuyor ama ek yüzünden anlam ters dönmüşse cevap HAYIR'dır.
Sözcükler hiç tutmuyor ama aynı durumu anlatıyorsa cevap EVET'tir.

Her aday için ayrı ayrı karar ver:

1. `answers_query`: Dikkatli bir okuyucu bu adayı sorgunun ifade ettiği duruma ait geçerli bir
   kayıt olarak KABUL EDER Mİ? Sorgu bir durumu iddia ediyor; aday o durumu doğruluyorsa true.
   Titiz ol: yalnızca konuyla ilgili olmak yeterli DEĞİLDİR; aday sorgunun iddia ettiği bilgiyi
   taşımalıdır. Emin değilsen false yaz.
   ANCAK "kelime dağarcığı farklı", "ifade biçimi değişik", "cümle yapısı farklı" GEÇERLİ BİR RET
   GEREKÇESİ DEĞİLDİR. Anlam aynıysa true yaz.

2. `answers_query` false ise `why_not` alanına en uygun tek nedeni seç:
   - minimal_suffix_contrast          : neredeyse aynı cümle, ama bir EK farkı anlamı tersine
                                        çevirmiş (olumsuzluk, kişi, zaman, hâl, yeterlilik vb.)
   - same_pattern_wrong_content       : aynı dilbilgisel yapı, ama farklı nesne/olay/kişi hakkında
   - partial_overlap_opposite_outcome : olayın bir kısmı örtüşüyor ama sonuç ters
   - different_state_or_participant   : farklı bir durum, zaman ya da katılımcı anlatılıyor
   - unrelated                        : aynı konu evreninde ama sorgunun bilgisini hiç taşımıyor
   `answers_query` true ise `why_not` alanına "none" yaz.

3. `morphology_errors`: Türkçesi hatalı olan (ünlü uyumu bozuk, ek dizilimi yanlış, var olmayan
   biçim kullanılmış) adayların id'leri.

4. `fluency_outliers`: Diğerlerinden belirgin biçimde ayrışan — çok daha uzun/kısa, farklı üslupta
   ya da "makine yazmış" izlenimi veren — adayların id'leri. Böyle bir aday yoksa boş liste.

5. `query_is_assertion`: Sorgu bir durumu İDDİA ediyor mu (true), yoksa soru mu soruyor (false)?
"""


# --------------------------------------------------------------------------- repair
REPAIR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "query": {"type": "STRING"},
        "frame_before": {"type": "STRING"},
        "frame_after": {"type": "STRING"},
        "critical_word_query": {"type": "STRING"},
        "critical_word_counterfactual": {"type": "STRING"},
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "role": {"type": "STRING",
                             "enum": ["positive", "hard_negative", "easy_negative"]},
                    "subtype": {"type": "STRING",
                                "enum": ["none", "morph_counterfactual", "same_feature_wrong_content",
                                         "partial_trap", "state_variant"]},
                    "core": {"type": "STRING",
                             "description": "Yalnızca bu adaya özgü cümle; ortak çerçeveyi yazma."},
                    "note": {"type": "STRING"},
                },
                "required": ["id", "role", "subtype", "core", "note"],
                "propertyOrdering": ["id", "role", "subtype", "core", "note"],
            },
        },
        "changes": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["query", "frame_before", "frame_after", "critical_word_query",
                 "critical_word_counterfactual", "candidates", "changes"],
    "propertyOrdering": ["query", "frame_before", "frame_after", "critical_word_query",
                         "critical_word_counterfactual", "candidates", "changes"],
}


def _repair_recipes(problems):
    """Turn each detected problem into the concrete edit that fixes it.

    "Fix these problems, minimally" is the wrong instruction for the two most common failures.
    Both are properties of the positive relative to everything else, and both need a REWRITE, not
    a tweak — in a 24-item probe the repair pass ran 14 times and fixed almost none of them,
    because it was being told to change as little as possible. Naming the required edit per
    problem type is what turns the repair call from a formality into a yield lever.
    """
    text = " ".join(problems)
    out = []
    if "birebir kopyalıyor" in text:
        out.append(
            "- 'birebir kopyalıyor' sorunu için POZİTİFİ BAŞTAN YAZ; burada az değişiklik YETMEZ. "
            "İçerik sözcüklerini (kişi/nesne/yer) koru ama cümleyi sıfırdan kur: eylemi adlaştır, "
            "edilgen/etken çevir, olayı sonucundan anlat, yan cümleyi ana cümle yap. Sorgudan "
            "4 sözcükten uzun hiçbir dizi pozitifte kalmasın.")
    if "en çok benzeyen aday" in text:
        out.append(
            "- 'sorguya en çok benzeyen aday' sorunu için POZİTİFİ sorgunun sözcüklerinden "
            "UZAKLAŞTIR (fiili ve cümle kuruluşunu değiştir). Sert olumsuzlara DOKUNMA — onların "
            "sorguya benzemesi gerekiyor, sorun onlarda değil.")
    if "sistematik olarak uzun" in text:
        out.append(
            "- 'sistematik olarak uzun' sorunu için POZİTİFİ KISALTMA yerine OLUMSUZ ADAYLARI "
            "UZAT: her birine, kendi anlamını değiştirmeden, aynı sahneden nötr bir cümle ekle. "
            "Hedef: hiçbir aday pozitifin %80'inden kısa olmasın.")
    if "minimal tier" in text:
        out.append(
            "- 'minimal tier' sorunu için morph_counterfactual'ı POZİTİFTEN yeniden türet: "
            "pozitifi aynen kopyala, yalnızca tek sözcüğün hedef ekini değiştir.")
    return "\n".join(out)


def build_repair_prompt(item, problems):
    """Minimal-edit repair. Ids must be preserved so the caller can diff what actually changed."""
    listing = json.dumps(
        {"query": item["query"],
         "frame_before": item.get("frame_before", ""),
         "frame_after": item.get("frame_after", ""),
         "candidates": [{"id": c["id"], "role": c["role"], "subtype": c.get("subtype", ""),
                         "core": c.get("core", c["text"])} for c in item["candidates"]]},
        ensure_ascii=False, indent=1)
    probs = "\n".join(f"- {p}" for p in problems)

    return f"""{FIX_LOG_PROHIBITIONS}

=== SÖZLEŞME ===
{v1_conventions_block()}

=== DÜZELTİLECEK ÖĞE ===
Hedef biçimbirim özelliği: {item.get('target_feature')} ({item.get('ek_turu')})
Ölçülecek karşıtlık: {item.get('contrast', '')}

{listing}

=== DENETİMDE BULUNAN SORUNLAR ===
{probs}

=== GÖREV ===
Yukarıdaki sorunları gider. Kurallar:
- EN AZ DEĞİŞİKLİKLE düzelt. Sorunsuz adaylara dokunma.
{_repair_recipes(problems)}
- Aday id'lerini AYNEN koru; aday sayısını ve rollerini değiştirme.
- Adaylar `core` (yalnızca o adaya özgü cümle) olarak verilmiştir; ortak çerçeve `frame_before` /
  `frame_after` alanlarındadır ve 11 adayın hepsine kod tarafından eklenir. Sen de `core` döndür,
  çerçeveyi çekirdeğin içine yazma.
- Bir adayı düzeltmek başka bir kuralı bozmamalı; özellikle içerik sözcüklerini eşanlamlısıyla
  değiştirme ve ikinci bir doğru cevap yaratma.
- `changes` alanına yaptığın her değişikliği tek satırda, "aday_id: ne değişti" biçiminde yaz.
- Sorunu düzeltmenin yolu adayı silmekse, onu aynı rolde YENİ bir metinle değiştir.
"""
