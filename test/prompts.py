"""Zero-shot generation, repair and blind-judge prompts.

No real development/test item is ever used as a few-shot exemplar.
"""

from __future__ import annotations

import hashlib
import json
import random

from .taxonomy import HARD_SUBTYPES


PROMPT_VERSION = "test-prompts-3.2.0"

GENERATOR_SYSTEM = """\
Sen Türkçe biçimbilim ve bilgi erişimi için contrast-set yazan uzman bir veri küratörüsün.
Görevin doğal, gündelik veya kurumsal Türkçe ile tek bir sorgu ve tam 11 adaydan oluşan bir family
üretmektir. Çıktı yalnızca istenen JSON şemasına uymalıdır. Meta-arama dili, soru cümlesi,
çeviri kokan ifade, yapay dolgu ve bozuk ek kullanımı yasaktır. Aynı family içindeki uzunluk,
üslup ve ayrıntı yoğunluğu dengeli olmalıdır.
"""

JUDGE_SYSTEM = """\
Sen generator'dan bağımsız, kör bir Türkçe retrieval ve biçimbilim hakemisin. Adayların gold,
hard/easy veya alt-tür etiketlerini görmeyeceksin. Önce sorguyu gerçekten yanıtlayan bütün adayları
bul; sonra her adayı anlam, doğal Türkçe ve biçimbilim açısından sınıflandır. Geçerli bir allomorf
yalnız yüzeyi değiştiği için yanlış sayılamaz. JSON dışında metin yazma.
"""


_HARD_DESCRIPTIONS = {
    "minimal_morph_negative": "Sorguya çok yakın; yalnız hedef morfolojik özellik değiştiği için anlam yanlış.",
    "same_lemma_wrong_inflection": "Pozitifle aynı kritik lemma; farklı ve ilgili bir çekim anlamı yanlış yapıyor.",
    "related_feature_negative": "Hedefe komşu ikinci bir morfolojik özellik yanlış; hedef karşıtlığın kopyası değil.",
    "same_morph_wrong_content": "Hedef morfoloji doğru; nesne/olay/referans yanlış.",
    "state_participant_time_trap": "Katılımcı, kişi, zaman veya gerçekleşme durumu yanlış.",
    "close_paraphrase_wrong_meaning": "Sözcüksel olarak yakın bir yeniden yazım; temel önerme yanlış veya çelişik.",
    "argument_role_reversal": "Kim-kime-ne yaptı rolleri ters veya yanlış bağlanmış.",
    "scope_attachment_trap": "Olumsuzluk, kip, iyelik veya ek zincirinin kapsamı yanlış yerde.",
    "morph_distractor": "Hedef biçimbilim metinde var ama yanlış yükleme/olaya bağlı; sorguyu yanıtlamıyor.",
    "partial_chain_negative": "Ek zincirinin yalnız bir bölümü doğru; zincirin tamamı sorgunun anlamını vermiyor.",
    "allomorph_form_function_trap": "Yüzeyce benzer ek/biçim var; fakat gramatik işlev geçerli allomorf eşdeğeri değil.",
    "noun_possessor_number_trap": "Adın/nesnenin çoğulluğu ile sahibin çoğulluğu karıştırılmış; örneğin çok nesne ile çok sahip aynı şey değildir.",
}


def _hard_rules(profile: list[dict]) -> str:
    return "\n".join(
        f"  - {item['slot']} -> {item['subtype']}: {_HARD_DESCRIPTIONS[item['subtype']]} "
        f"Odak: {item['focus']}."
        for item in profile
    )


def build_generation_prompt(slot: dict) -> str:
    feature = slot["feature"]
    allomorph_rule = (
        "Bu bir ALLOMORPH INVARIANCE family’sidir: pozitif, aynı gramatik işlevi farklı geçerli "
        "yüzey biçimiyle korumalıdır. Geçerli allomorf hard/easy negatif OLAMAZ. Negatifler başka "
        "bir morfem veya başka bir anlam yüzünden yanlış olmalıdır."
        if slot["objective"] == "allomorph_invariance"
        else "Bu family morfem duyarlılığını ölçer: pozitif hedef anlamı korur; minimal negatifte "
             "morfolojik değişim gerçekten anlamı değiştirmelidir."
    )
    bucket_rule = {
        "standard": "Train'e benzer doğal zorlukta, fakat hiçbir metnin kopyası olmayan bir örnek üret.",
        "lemma_holdout": "Nadir olmayan fakat üretim boyunca tekrar edilmeyecek doğal bir kritik lemma seç.",
        "template_holdout": "Verilen soyut sözdizimi kalıbını doğal ve özgün biçimde gerçekleştir.",
        "composition_holdout": "Bileşenleri anlaşılır, tam ek zinciri kapsam/sıra bakımından ayırt edici olsun.",
    }[slot["generalization_bucket"]]
    domain_rule = (
        "Bu family ayrıca domain_shift etiketli: domain + register birleşimini doğal biçimde "
        "gerçekleştir; bu birleşim gelecekteki train exclusion manifestine girecek."
        if "domain_shift" in slot["generalization_tags"]
        else "Domain/register doğal çeşitlilik içindir; yapay jargon ekleme."
    )
    strict_rule = (
        "Bu STRICT MINIMAL-PAIR family’sidir. Positive ile hard_01 aynı kritik lemmayı, aynı "
        "sözdizimsel şablonu ve aynı sözcük dizisini korumalı; yalnız kritik çekimli sözcüğün "
        "hedef eki/ek zinciri değişmelidir. Noktalama dahil diğer tokenlar aynı kalmalıdır. "
        "edit_script.applies=true olmalı; iki yüzey biçimini, tek işlemi, değişen feature'ı ve "
        "korunan invariants alanlarını açıkça kaydet."
        if slot["strict_minimal_pair"] else
        "Bu family strict minimal-pair slice'ında değildir. hard_01 yine yerel ve kontrollü olsun; "
        "edit_script.applies=false, diğer edit_script string/dizi alanları boş olabilir."
    )

    return f"""\
SLOT
{json.dumps(slot, ensure_ascii=False, indent=2)}

HEDEF
- Bir query ve tam 11 aday: 1 positive + 8 hard_negative + 2 easy_negative.
- Query tam {slot['query_sentence_count']} cümle olmalı ve tek bir bilgi ihtiyacını ifade etmeli.
- Query iki cümleyse ikinci cümle yeni bir olay/niyet açmamalı; ilk önermeyi doğal biçimde sınırlandırmalı.
- Her aday pasajı tam {slot['passage_sentence_count']} cümle olacak.
- `context_sentences` tam {slot['passage_sentence_count'] - 1} doğal TAM cümle içermeli.
- Her candidate yalnız bir TAM `critical_sentence` üretmeli. Kod bu cümleyi pasajın
  {slot['critical_sentence_position']}. konumuna yerleştirip aynı `context_sentences` cümlelerini
  diğer konumlara byte-identical olarak koyacak.
- Ortak bağlam tek başına sorguyu yanıtlamamalı ve adaylar arasındaki doğru/yanlış ayrımını
  değiştirmemeli; ayrım yalnız `critical_sentence` içinde yerel kalmalı.
- Adayların token uzunlukları ve ayrıntı yoğunluğu birbirine yakın olmalı. Gold sistematik olarak en uzun olamaz.
- Query doğal bir durumu İDDİA etmeli; evet/hayır sorusu ve “kaydı bul/arıyorum” dili kullanma.
- Positive sorgudaki aynı olayı/anlamı doğru biçimbilimle doğal bir paraphrase olarak vermeli.
- `equivalence_positive` tek positive alt-türüdür.
- İki easy_negative konu ve sözcük bakımından bariz farklıdır ama uzunluk/akıcılık açısından ucuz ipucu vermez.

SEKİZ HARD NEGATIVE — SLOT'a atanmış sekiz uyumlu senaryonun her birinden TAM BİR tane:
{_hard_rules(slot['hard_profile'])}

BİÇİMBİLİM
- Bir adayı yanlış yapan neden tek ve açıklanabilir olmalı.
- Bozuk Türkçe veya imkânsız ek dizisi hard negative üretmez; yalnız ucuz dilbilgisi ipucu üretir.
- Yüzey biçimi ile işlevi karıştırma. `şubesinden` (ABL) ve `şubesinde` (LOC) allomorf değil,
  anlam değiştiren farklı hâllerdir.
- {allomorph_rule}
- Yüzey biçimleri yalnız rehberdir: {feature['surface_forms']}.
- Anlam karşıtlığı: {feature['meaning_contrast']}.
- {strict_rule}

GENERALİZASYON
- {bucket_rule}
- {domain_rule}
- Domain: {slot['domain']}; register: {slot['register']}.
- Soyut template: {slot['template']['id']} — {slot['template']['description']}.
- Testteki gerçek örneklerden veya eski JSON cümlelerinden alıntı/kopya yapma.

ALANLAR
- `semantic_frame_id` ve `template_id` SLOT değerleriyle aynı olsun.
- `critical_lemma`, query/positive kritik sözcükleri ve gerçek `feature_delta` açıkça yazılsın.
- Positive `candidate_slot=positive_01`; hard slotları SLOT'taki `hard_01`…`hard_08` ile birebir
  eşleştir; easy slotları `easy_01` ve `easy_02` yap.
- Her candidate için tek cümlelik `critical_sentence`, `critical_word`, `morph_relation` ve kısa gerekçe ver.
- Positive morph_relation: allomorph family’de `allomorph_equivalent`, diğerlerinde `target_preserved`.
- Easy morph_relation `unrelated`; hard relation kendi hatasına uygun olsun.
"""


def build_repair_prompt(slot: dict, previous: dict, problems: list[str]) -> str:
    return f"""\
Önceki üretim otomatik kalite kapısından geçmedi. Bütün family’yi yeniden ve bağımsız yaz.
Kimlikleri veya hatalı metni korumaya çalışma. Temel SLOT ve bütün üretim kuralları geçerlidir.

SORUNLAR
{json.dumps(problems, ensure_ascii=False, indent=2)}

ÖNCEKİ ÇIKTI — yalnız hatayı anlamak için, kopyalama:
{json.dumps(previous, ensure_ascii=False, indent=2)}

ORİJİNAL GÖREV
{build_generation_prompt(slot)}
"""


def _blind_candidates(family: dict) -> list[dict]:
    candidates = [{"id": c["id"], "text": c["text"]} for c in family["candidates"]]
    seed = int(hashlib.sha256(family["family_id"].encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(candidates)
    return candidates


def build_judge_prompt(family: dict) -> str:
    allowed = sorted(HARD_SUBTYPES)
    visible = {
        "query": family["query"],
        "target_feature": family["target_feature"],
        "objective": family["objective"],
        "layer": family["layer"],
        "query_sentence_count": family["query_sentence_count"],
        "passage_sentence_count": family["passage_sentence_count"],
        "candidates": _blind_candidates(family),
    }
    return f"""\
Aşağıdaki family’yi etiketleri görmeden değerlendir.

{json.dumps(visible, ensure_ascii=False, indent=2)}

KURALLAR
1. `answers_query`: sorguyu bütünüyle doğru yanıtlayan bütün aday kimlikleri. Kısmi aday ekleme.
2. Her aday için relevance, 1–5 naturalness, morphology_ok ve inferred_type ver.
3. Olası inferred_type değerleri: positive, {', '.join(allowed)}, easy_negative, unclear.
4. Bir hard aday dilbilgisel olarak bozuksa morphology_ok=false; bozukluk zorluk sayılamaz.
5. Allomorph invariance hedefinde geçerli yüzey değişkesini yanlış sayma.
6. Uzunluk, üslup veya ayrıntı yalnız gold’u ele veriyorsa length_or_style_artifact=true.
7. Family doğal değilse veya birden fazla doğru aday varsa bunu açıkça kaydet.
"""
