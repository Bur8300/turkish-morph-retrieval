"""Curated, explicit taxonomy for the automatically verified test benchmark.

Unlike the legacy training generator, surface suffix forms are structured data.  No regex tries
to recover allomorphs from Turkish prose, so gloss words can never silently become suffixes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    macro: str
    layer: str
    meaning_contrast: str
    surface_forms: tuple[str, ...] = ()
    objective: str = "morpheme_sensitivity"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["surface_forms"] = list(self.surface_forms)
        return out


def F(key, label, macro, layer, contrast, forms=(), objective="morpheme_sensitivity"):
    return Feature(key, label, macro, layer, contrast, tuple(forms), objective)


CASE = "case_location_direction"
AGR = "possession_person_number"
TAM = "tense_mood_aspect_evidentiality"
VOICE = "negation_voice_valency"
DERIV = "derivation_nonfinite_allomorphy"
COMP = "suffix_chain_composition"


FEATURES = (
    # Negation, ability, voice, valency
    F("NEG", "olumsuzluk", VOICE, "single", "eylemin gerçekleşmesi ↔ gerçekleşmemesi", ("-ma", "-me")),
    F("COP.NEG", "isim cümlesinde olumsuzluk", VOICE, "single", "yüklemin özne için geçerli olması ↔ değil ile reddedilmesi", ("değil", "değildi", "değilmiş", "değilse")),
    F("ABIL", "yeterlilik", VOICE, "single", "yapabilme ↔ yapamama", ("-abil", "-ebil")),
    F("NEG.ABIL", "yetersizlik", VOICE, "single", "yapamama ↔ yapmama", ("-ama", "-eme")),
    F("CAUS", "ettirgen", VOICE, "single", "kendisi yapma ↔ başkasına yaptırma", ("-dır", "-dir", "-tır", "-tir", "-ir")),
    F("PASS", "edilgen", VOICE, "single", "öznenin yapması ↔ özneye yapılması", ("-il", "-ıl", "-in", "-ın", "-n")),
    F("REFL", "dönüşlü", VOICE, "single", "başkasına yapma ↔ kendine yapma", ("-in", "-ın", "-n")),
    F("RECP", "işteş", VOICE, "single", "tek taraflı ↔ karşılıklı/birlikte", ("-iş", "-ış", "-uş", "-üş", "-ş")),

    # Tense, aspect, mood, evidentiality
    F("PST", "görülen geçmiş", TAM, "single", "tanık olunan geçmiş ↔ şimdiki/gelecek", ("-dı", "-di", "-du", "-dü", "-tı", "-ti", "-tu", "-tü")),
    F("PRF.EVID", "duyulan geçmiş/kanıtsallık", TAM, "single", "doğrudan tanıklık ↔ aktarma/çıkarım", ("-mış", "-miş", "-muş", "-müş")),
    F("PRS.PROG", "şimdiki zaman", TAM, "single", "sürmekte olan ↔ tamamlanmış", ("-ıyor", "-iyor", "-uyor", "-üyor")),
    F("FUT", "gelecek zaman", TAM, "single", "henüz olacak ↔ olmuş", ("-acak", "-ecek")),
    F("AOR", "geniş zaman", TAM, "single", "alışkanlık/genel doğru ↔ tek olay", ("-ar", "-er", "-ır", "-ir", "-ur", "-ür")),
    F("NEC", "gereklilik", TAM, "single", "yapması gerekli ↔ yaptığı/istediği", ("-malı", "-meli")),
    F("OPT", "istek kipi", TAM, "single", "dilek/istek ↔ gerçekleşmiş olay", ("-a", "-e")),
    F("IMP.3", "üçüncü kişi emir", TAM, "single", "yapması isteniyor ↔ yaptığı bildiriliyor", ("-sın", "-sin", "-sun", "-sün")),
    F("COND", "koşul", TAM, "single", "koşullu ihtimal ↔ gerçekleşmiş neden-sonuç", ("-sa", "-se")),
    F("PRSM", "tahmin/kesinlik", TAM, "single", "çıkarım/tahmin ↔ doğrudan bilgi", ("-dır", "-dir", "-dur", "-dür")),
    F("COP.TAM", "ek-fiilde zaman/kanıtsallık/koşul", TAM, "single", "isim yüklemin geçmişte geçerli olması ↔ aktarılması/koşula bağlanması", ("-ydı", "-ydi", "-ymış", "-ymiş", "-ysa", "-yse", "idi", "imiş", "ise")),
    F("Q.PART.SCOPE", "soru parçacığında odak kapsamı", TAM, "single", "eyleyenin sorgulanması ↔ nesnenin/başka ögenin sorgulanması", ("mı", "mi", "mu", "mü")),

    # Nominal case, possession, number and agreement
    F("ACC", "belirtme hâli", CASE, "single", "belirli nesne ↔ belirsiz/genel nesne", ("-ı", "-i", "-u", "-ü")),
    F("DAT", "yönelme hâli", CASE, "single", "hedefe doğru ↔ kaynaktan", ("-a", "-e", "-ya", "-ye")),
    F("LOC", "bulunma hâli", CASE, "single", "bir yerde bulunma ↔ bir yerden ayrılma", ("-da", "-de", "-ta", "-te")),
    F("ABL", "ayrılma hâli", CASE, "single", "kaynaktan ayrılma ↔ hedefe yönelme", ("-dan", "-den", "-tan", "-ten")),
    F("INS", "vasıta/birliktelik hâli", CASE, "single", "birlikte/araçla ↔ hedefe", ("-la", "-le", "-yla", "-yle")),
    F("GEN", "ilgi hâli", CASE, "single", "aitlik/sahiplik ↔ yalın ad", ("-ın", "-in", "-un", "-ün", "-nın", "-nin")),
    F("EQU", "eşitlik hâli", CASE, "single", "ona göre/o biçimde ↔ başka ölçüt", ("-ca", "-ce", "-ça", "-çe")),
    F("CASE.ROLE.FRAME", "hâl işaretli argüman rolleri", CASE, "single", "aynı katılımcılarla kim, kimi, kime ve kimden rollerinin korunması ↔ rollerin ters/yanlış bağlanması"),
    F("PL", "çoğul", AGR, "single", "birden çok ↔ tek", ("-lar", "-ler")),
    F("V.AGR", "fiilde kişi-sayı uyumu", AGR, "single", "eylemi yapan kişi/sayı ↔ başka kişi/sayı", ("-m", "-ım", "-im", "-sın", "-sin", "-k", "-ız", "-iz", "-siniz", "-lar", "-ler")),
    F("ANAPHOR.AGR", "kendi zamirinde kişi-sayı ve gönderim", AGR, "single", "dönüşlü zamirin doğru kişiye/gruba gönderimi ↔ başka kişi/gruba gönderimi", ("kendim", "kendin", "kendi", "kendimiz", "kendiniz", "kendileri")),
    F("POSS.1SG", "1. tekil iyelik", AGR, "single", "benim ↔ başkasının", ("-ım", "-im", "-um", "-üm", "-m")),
    F("POSS.2SG", "2. tekil iyelik", AGR, "single", "senin ↔ başkasının", ("-ın", "-in", "-un", "-ün", "-n")),
    F("POSS.3SG", "3. tekil iyelik", AGR, "single", "onun ↔ benim/bizim", ("-ı", "-i", "-u", "-ü", "-sı", "-si")),
    F("POSS.1PL", "1. çoğul iyelik", AGR, "single", "bizim ↔ tek kişinin", ("-ımız", "-imiz", "-umuz", "-ümüz")),
    F("POSS.2PL", "2. çoğul iyelik", AGR, "single", "sizin ↔ senin", ("-ınız", "-iniz", "-unuz", "-ünüz")),
    F("POSS.3PL", "3. çoğul iyelik", AGR, "single", "onların ↔ onun", ("-ları", "-leri")),

    # Non-finite and derivational morphology
    F("CVB.WHEN", "-ince ulacı", DERIV, "single", "olay gerçekleşince ↔ olaydan bağımsız", ("-ınca", "-ince", "-unca", "-ünce")),
    F("CVB.BY", "-erek ulacı", DERIV, "single", "o yolla yaparak ↔ başka yolla", ("-arak", "-erek")),
    F("CVB.AND", "-ip ulacı", DERIV, "single", "ardışık iki olay ↔ yalnız biri", ("-ıp", "-ip", "-up", "-üp")),
    F("CVB.WITHOUT", "-meden ulacı", DERIV, "single", "yapmadan ↔ yaptıktan sonra", ("-madan", "-meden")),
    F("CVB.ASLONG", "-dikçe ulacı", DERIV, "single", "sürdükçe/orantılı ↔ tek sefer", ("-dıkça", "-dikçe", "-dukça", "-dükçe")),
    F("CVB.WHILE", "-ken ulacı", DERIV, "single", "eşzamanlı ↔ önce/sonra", ("-ken",)),
    F("NMLZ.MEK", "-mek adlaştırması", DERIV, "single", "eylemin kendisi ↔ gerçekleşmiş olay", ("-mak", "-mek")),
    F("NMLZ.MA_VS_DIK", "-mA ve -DIK adlaştırma karşıtlığı", DERIV, "single", "istenen/planlanan olay ↔ gerçekleşmiş olgu bilgisi", ("-ma", "-me", "-dığı", "-diği", "-duğu", "-düğü")),
    F("PTCP.SUBJ", "özne sıfat-fiili", DERIV, "single", "eylemi yapan ↔ maruz kalan", ("-an", "-en")),
    F("PTCP.FUT", "gelecek sıfat-fiili", DERIV, "single", "yapılacak ↔ yapılmış", ("-acak", "-ecek")),
    F("PRIV", "yoksunluk", DERIV, "single", "bir şeyden yoksun ↔ onu içeren", ("-sız", "-siz", "-suz", "-süz")),
    F("PROP", "varlık/bulunma", DERIV, "single", "bir şeyi içeren ↔ ondan yoksun", ("-lı", "-li", "-lu", "-lü")),
    F("REL.KI", "aitlik -ki", DERIV, "single", "belirtilen yere/zamana ait ↔ başka yere ait", ("-ki",)),
    F("AGT", "meslek/uğraş", DERIV, "single", "işi yapan kişi ↔ işin kendisi", ("-cı", "-ci", "-cu", "-cü", "-çı", "-çi")),
    F("ABST", "soyutlama", DERIV, "single", "soyut görev/nitelik ↔ somut varlık", ("-lık", "-lik", "-luk", "-lük")),
    F("DISTR", "üleştirme", DERIV, "single", "her birine ayrı ayrı ↔ toplam", ("-ar", "-er", "-şar", "-şer")),
    F("MORPH.CONTEXT_AMBIG", "bağlamsal morfolojik belirsizlik", DERIV, "single", "aynı yüzey biçiminin bağlamda doğru lemma/sözcük türü/çekim analizi ↔ bağlama uymayan alternatif analizi"),

    # Allomorph invariance: valid variants preserve function and can only support the positive.
    F("ALLO.LOC", "bulunma hâli allomorfları", DERIV, "single", "aynı bulunma işlevi, farklı uyum/ötümsüzleşme", ("-da", "-de", "-ta", "-te"), "allomorph_invariance"),
    F("ALLO.ABL", "ayrılma hâli allomorfları", DERIV, "single", "aynı ayrılma işlevi, farklı uyum/ötümsüzleşme", ("-dan", "-den", "-tan", "-ten"), "allomorph_invariance"),
    F("ALLO.DAT", "yönelme kaynaştırma allomorfları", DERIV, "single", "aynı yönelme işlevi, kaynaştırmalı yüzey", ("-a", "-e", "-ya", "-ye"), "allomorph_invariance"),
    F("ALLO.ACC", "belirtme hâli allomorfları", DERIV, "single", "aynı belirtme işlevi, ünlü uyumlu yüzey", ("-ı", "-i", "-u", "-ü", "-yı", "-yi"), "allomorph_invariance"),

    # Explicit suffix chains and scope/order generalisation
    F("NEG.AOR", "geniş zaman + olumsuzluk", COMP, "chain", "genel olumsuz alışkanlık ↔ anlık olumsuzluk", ("-maz", "-mez"), "composition"),
    F("PLUPRF", "kanıtsal geçmiş + geçmiş", COMP, "chain", "daha önce tamamlanmış ↔ o sırada sürüyor", ("-mıştı", "-mişti"), "composition"),
    F("PST.PROG", "şimdiki + geçmiş", COMP, "chain", "geçmişte sürüyordu ↔ daha önce bitmişti", ("-ıyordu", "-iyordu"), "composition"),
    F("FUT.PST", "gelecek + geçmiş", COMP, "chain", "gerçekleşmemiş niyet ↔ gerçekleşmiş olay", ("-acaktı", "-ecekti"), "composition"),
    F("CNTR", "karşı-olgusal koşul", COMP, "chain", "olmadı ama olsaydı ↔ gerçekten oldu", ("-saydı", "-seydi"), "composition"),
    F("NMLZ.DIK", "adlaştırma + iyelik", COMP, "chain", "gerçekleşmiş eylem bilgisi ↔ gelecek eylem", ("-dığı", "-diği", "-duğu", "-düğü"), "composition"),
    F("REL.GEN.POSS", "-DIK sıfat-fiilinde genitif–iyelik uyumu", COMP, "chain", "yan cümle öznesi ile sıfat-fiil iyeliğinin aynı kişi/sayıya bağlanması ↔ başka possessor/gönderim", ("-ın", "-in", "-un", "-ün", "-dığı", "-diği", "-dığım", "-diğim"), "composition"),
    F("PL.POSS.CASE", "çoğul + iyelik + hâl", COMP, "chain", "kaç tane, kimin ve yön/konum birlikte", (), "composition"),
    F("CAUS.PASS.NEG", "ettirgen + edilgen + olumsuzluk", COMP, "chain", "yaptıran, maruz kalan ve olumsuzluk kapsamı", (), "composition"),
    F("EVID.COND.NEG", "kanıtsallık + koşul + olumsuzluk", COMP, "chain", "bilgi kaynağı, ihtimal ve kutupluluk", (), "composition"),
    F("NMLZ.CASE.CNTR", "adlaştırma + hâl + karşı-olgusal", COMP, "chain", "adlaştırılmış olayın hâli ve gerçekliği", (), "composition"),
    F("TENSE.PERS.NEG", "zaman + kişi + olumsuzluk", COMP, "chain", "kim, ne zaman ve gerçekleşip gerçekleşmediği", (), "composition"),
    F("EVID.POSS.NEG", "kanıtsallık + iyelik + olumsuzluk", COMP, "chain", "kimin bilgisi, kaynağı ve kutupluluğu", (), "composition"),
    F("RECP.CAUS", "işteş + ettirgen", COMP, "chain", "karşılıklı yapma ↔ karşılıklı yaptırma", (), "composition"),
    F("POSS.PL.ABL", "iyelik + çoğul + ayrılma", COMP, "chain", "kimin kaç tanesinden ayrılma", (), "composition"),
    F("DERIV.IG_CHAIN", "türetim sınırları ve IG zinciri", COMP, "chain", "kök sözcük türü, türetim adımları ve son sözcük türünün korunması ↔ farklı türetim yolu veya sınırı", (), "composition"),
    F("SUSP.AFFIX", "koordinasyonda askıda ekleme", COMP, "chain", "son eşlenikteki çekimin koordineli ögelere doğru yayılması ↔ yalnız son ögeye bağlanması veya rollerin değişmesi", (), "composition"),
    F("MWE.MORPH", "çok sözcüklü ifadede morfoloji", COMP, "chain", "destek fiilli/kalıplaşmış/tekrarlı yapının morfolojik ve bütüncül anlamının korunması ↔ yalnız yüzey parçalarının eşleşmesi", (), "composition"),
)


FEATURE_BY_KEY = {feature.key: feature for feature in FEATURES}
MACROS = (CASE, AGR, TAM, VOICE, DERIV, COMP)

DOMAINS = (
    "daily_life", "health", "education", "finance", "law_public_services",
    "travel", "workplace", "ecommerce", "news", "culture_technology",
)

REGISTERS = ("everyday", "conversational", "formal_record", "news_report")

TEMPLATES = (
    {"id": "event_report", "description": "doğal bir olay/durum bildirimi"},
    {"id": "personal_recollection", "description": "birinci kişi gündelik hatırlama"},
    {"id": "reported_speech", "description": "başkasından aktarılan bilgi"},
    {"id": "relative_clause", "description": "sıfat-fiilli yan cümle"},
    {"id": "temporal_subordinate", "description": "zaman ilişkili yan cümle"},
    {"id": "causal_explanation", "description": "neden-sonuç açıklaması"},
    {"id": "conditional_scenario", "description": "koşullu veya karşı-olgusal senaryo"},
    {"id": "formal_record", "description": "kurumsal kayıt/rapor anlatımı"},
    {"id": "coordinated_events", "description": "iki bağlantılı olayın koordinasyonu"},
    {"id": "contrastive_statement", "description": "iki olasılığı karşılaştıran bildirim"},
)


HARD_SUBTYPES = (
    "minimal_morph_negative",
    "controlled_morph_negative",
    "same_lemma_wrong_inflection",
    "related_feature_negative",
    "same_morph_wrong_content",
    "state_participant_time_trap",
    "close_paraphrase_wrong_meaning",
    "argument_role_reversal",
    "scope_attachment_trap",
    "morph_distractor",
    "partial_chain_negative",
    "allomorph_form_function_trap",
    "noun_possessor_number_trap",
    "lexical_retrieval_trap",
    "semantic_retrieval_hard",
)

MORPH_HARD_SUBTYPES = frozenset({
    "minimal_morph_negative", "controlled_morph_negative", "same_lemma_wrong_inflection", "related_feature_negative",
    "scope_attachment_trap", "morph_distractor", "partial_chain_negative",
    "allomorph_form_function_trap", "noun_possessor_number_trap",
})
SEMANTIC_HARD_SUBTYPES = frozenset(set(HARD_SUBTYPES) - MORPH_HARD_SUBTYPES)


_CORE_HARD_PROFILE = (
    ("minimal_morph_negative", "yalnız hedef morfolojik karşıtlık"),
    ("same_lemma_wrong_inflection", "aynı kritik lemma, yanlış çekim"),
    ("related_feature_negative", "hedefe komşu ikinci morfolojik özellik yanlış"),
    ("same_morph_wrong_content", "hedef morfoloji doğru, önerme içeriği yanlış"),
    ("state_participant_time_trap", "durum, katılımcı, kişi veya zaman yanlış"),
    ("close_paraphrase_wrong_meaning", "yakın paraphrase, temel önerme yanlış"),
)


def hard_profile(feature: Feature | dict, family_mode: str = "strict_minimal") -> list[dict[str, str]]:
    """Select eight compatible slots instead of forcing eight universal error classes."""
    if isinstance(feature, dict):
        key = str(feature["key"])
        layer = str(feature["layer"])
        macro = str(feature["macro"])
        objective = str(feature["objective"])
    else:
        key, layer, macro, objective = feature.key, feature.layer, feature.macro, feature.objective

    if objective == "allomorph_invariance":
        extras = (
            ("allomorph_form_function_trap", "benzer yüzey, fakat farklı gramatik işlev"),
            ("morph_distractor", "geçerli hedef biçim yanlış olay veya sözcüğe bağlı"),
        )
    elif layer == "chain":
        extras = (
            ("partial_chain_negative", "ek zincirinin yalnız bir parçası doğru"),
            ("scope_attachment_trap", "ek zincirinin kapsamı veya bağlanması yanlış"),
        )
    elif macro == AGR and (key == "PL" or key.startswith("POSS.")):
        extras = (
            ("noun_possessor_number_trap", "nesnenin sayısı ile sahibin sayısı karıştırılmış"),
            ("morph_distractor", "hedef biçim yanlış olay veya sözcüğe bağlı"),
        )
    elif key in {"COP.NEG", "Q.PART.SCOPE", "NMLZ.MA_VS_DIK", "MORPH.CONTEXT_AMBIG"}:
        extras = (
            ("scope_attachment_trap", "olumsuzluk, soru odağı veya yan-cümle kapsamı yanlış"),
            ("morph_distractor", "hedef biçim yanlış yüklem, öge veya olaya bağlı"),
        )
    elif macro in {CASE, AGR, VOICE}:
        extras = (
            ("argument_role_reversal", "katılımcı veya sahiplik rolleri ters"),
            ("morph_distractor", "hedef biçim yanlış olay veya sözcüğe bağlı"),
        )
    else:
        extras = (
            ("scope_attachment_trap", "kip, olumsuzluk veya yan-cümle kapsamı yanlış"),
            ("morph_distractor", "hedef biçim yanlış olay veya sözcüğe bağlı"),
        )

    if family_mode == "strict_minimal":
        core = _CORE_HARD_PROFILE
    elif family_mode == "controlled_diverse":
        core = (
            ("controlled_morph_negative", "aynı olay ve hedef karşıtlık, farklı doğal sözdizimi"),
            *_CORE_HARD_PROFILE[1:],
        )
    elif family_mode == "natural_retrieval":
        core = (
            ("controlled_morph_negative", "hedef morfolojik anlam yanlış, doğal farklı anlatım"),
            ("related_feature_negative", "komşu morfolojik özellik yanlış"),
            ("same_morph_wrong_content", "morfoloji benzer, önerme içeriği yanlış"),
            ("state_participant_time_trap", "durum, katılımcı veya zaman yanlış"),
            ("close_paraphrase_wrong_meaning", "yakın anlam alanı, temel önerme yanlış"),
            ("lexical_retrieval_trap", "query sözcükleri çekici, fakat belge irrelevant"),
        )
        extras = (
            ("semantic_retrieval_hard", "doğal ve konuya yakın, fakat bilgi ihtiyacını karşılamıyor"),
            ("morph_distractor", "hedef biçim yanlış olay veya sözcüğe bağlı"),
        )
    else:
        raise ValueError(f"Bilinmeyen family_mode: {family_mode}")

    return [
        {"slot": f"hard_{index:02d}", "subtype": subtype, "focus": focus}
        for index, (subtype, focus) in enumerate((*core, *extras), start=1)
    ]
