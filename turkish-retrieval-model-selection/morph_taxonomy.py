#!/usr/bin/env python
"""Generation plan for the v2.0 Turkish morphological retrieval dataset.

Defines what gets generated, not how. A *slot* is one fully-specified combination

    (target_feature, layer, tier, domain, person, passage_length, seed_entities)

that `gen_morph_dataset.py` expands into exactly one dataset item. Slot planning is
deterministic given a seed, so a resumed run rebuilds the identical plan.

Design notes that are literature-driven (see docs/literature_review.md):

* Feature coverage is deliberately BROAD and BALANCED rather than corpus-frequency-aligned.
  The NevIR reproduction found bi-encoders do not transfer across contrast types, so a training
  set that over-weights negation buys nothing for case/tense/possessive sensitivity. Corpus
  alignment (Thunder-KoNUBench) is the right call for an *eval* set, balance for a *training* set.
* `passage_length` is a first-class axis. Pooling dilution is formally proven to grow with
  pooling scale (Gao et al. 2026) but has never been measured against morphology in any language;
  stratifying by length makes that experiment runnable from this data.
* Entity pools are disjoint from v1.3.1's, which is the held-out test set.
"""
import hashlib
import random

# --------------------------------------------------------------------------- feature inventory
# key          stable id, also written to the item as `target_feature`
# ek_turu      the Turkish label written into the item (matches v1.3.1's field of the same name)
# phenomenon   coarse grouping, matches v1.3.1's vocabulary where one already exists
# layer        "single" = one target suffix, "chain" = composition of two or more
# contrast     what the morph_counterfactual must flip. This string goes into the prompt verbatim,
#              so it must name BOTH poles of the contrast, not just the target form.
# tiers        which tiers this feature can legitimately be generated at
F = lambda key, ek_turu, phenomenon, layer, contrast, tiers: dict(
    key=key, ek_turu=ek_turu, phenomenon=phenomenon, layer=layer, contrast=contrast, tiers=tiers)

ALL = ("standard", "hard", "minimal")
STD = ("standard", "minimal")
STD_HARD = ("standard", "hard")
HARD = ("hard",)

TARGET_FEATURES = [
    # ---- verbal: polarity, ability, voice -----------------------------------------------
    F("NEG", "olumsuzluk (-ma/-me)", "olumsuzluk", "single",
      "olumlu eylem <-> olumsuz eylem (-ma/-me)", ALL),
    F("ABIL", "yeterlilik (-ebil/-abil)", "yeterlilik", "single",
      "eylemi yapabilme <-> yapma (yeterlilik eki var/yok)", ALL),
    F("NEG.ABIL", "yetersizlik (-ama/-eme)", "yeterlilik", "single",
      "yapamama (-ama, engel/imkânsızlık) <-> yapmama (-ma, tercih)", ALL),
    F("CAUS", "ettirgen (-tır/-dır/-t/-ir)", "ettirgen", "single",
      "eylemi kendisi yapma <-> başkasına yaptırma", ALL),
    F("CAUS.CAUS", "çift ettirgen (-tirt/-ttır)", "ettirgen", "chain",
      "yaptırma <-> yaptırtma (araya bir aracı daha girer)", STD_HARD),
    F("PASS", "edilgen (-il/-in/-n)", "cati_edilgen", "single",
      "etken (özne eylemi yapar) <-> edilgen (eylem özneye yapılır)", ALL),
    F("REFL", "dönüşlü (-in/-n)", "cati_donuslu", "single",
      "eylemi başkasına yapma <-> kendine yapma", STD),
    F("RECP", "işteş (-iş/-ş)", "cati_istes", "single",
      "tek taraflı eylem <-> karşılıklı/birlikte eylem", STD_HARD),
    # ---- verbal: tense, aspect, evidentiality --------------------------------------------
    F("PST", "görülen geçmiş (-di)", "zaman", "single",
      "geçmişte olmuş <-> şu an oluyor / gelecekte olacak", STD),
    F("PRF.EVID", "duyulan geçmiş / kanıtsallık (-miş)", "kanit_kipligi", "single",
      "tanık olunan bilgi (-di) <-> aktarılan/çıkarım yoluyla edinilen bilgi (-miş)", ALL),
    F("PRS.PROG", "şimdiki zaman (-iyor)", "gorunus", "single",
      "sürmekte olan eylem (-iyor) <-> tamamlanmış eylem (-di)", STD),
    F("FUT", "gelecek zaman (-ecek)", "zaman", "single",
      "henüz olmamış/olacak <-> çoktan olmuş", STD),
    F("AOR", "geniş zaman (-ir/-er)", "gorunus_genis_zaman", "single",
      "alışkanlık/genel doğru (geniş zaman) <-> tek seferlik olay (-di/-iyor)", STD_HARD),
    F("NEG.AOR", "geniş zaman olumsuz (-mem/-mez)", "gorunus_genis_zaman", "single",
      "hiç yapmama alışkanlığı (-mem) <-> şu an yapmıyor olma (-miyorum)", ALL),
    F("PLUPRF", "-mişti (öncelik/tamamlanmışlık)", "gorunus", "chain",
      "daha önce tamamlanmış (-mişti) <-> o sırada sürmekte olan (-iyordu)", STD_HARD),
    F("PST.PROG", "-iyordu (sürerlik, geçmişte)", "gorunus", "chain",
      "o sırada sürüyordu (-iyordu) <-> o andan önce bitmişti (-mişti)", STD_HARD),
    F("FUT.PST", "-ecekti (gerçekleşmemiş niyet)", "kip", "chain",
      "yapılacaktı ama olmadı (-ecekti) <-> yapıldı (-di)", STD_HARD),
    F("HAB.PST", "-irdi (geçmişte alışkanlık)", "gorunus", "chain",
      "eskiden düzenli yapardı (-irdi) <-> bir kez yaptı (-di)", STD_HARD),
    # ---- verbal: mood and modality --------------------------------------------------------
    F("NEC", "gereklilik (-meli/-malı)", "kip", "single",
      "yapması gerekiyor (-meli) <-> yaptı / yapmak istiyor", ALL),
    F("OPT", "istek kipi (-e/-a)", "kip_istek", "single",
      "dilek/istek <-> gerçekleşmiş eylem", STD_HARD),
    F("IMP.3", "üçüncü kişi emir/istek (-sin)", "kip_istek_ucuncu", "single",
      "birinden bir şey yapması isteniyor (-sin) <-> yaptığı bildiriliyor", ALL),
    F("COND", "koşul/şart (-se/-sa)", "kip_kosul", "single",
      "koşullu, henüz gerçekleşmemiş (-se) <-> gerçekleşmiş neden-sonuç", ALL),
    F("CNTR", "karşı-olgusal koşul (-seydi)", "kip_karsi_olgusal", "chain",
      "olmadı ama olsaydı (-seydi) <-> gerçekten oldu", HARD),
    F("PRSM", "kesinlik/tahmin (-dir)", "kip_tahmin", "single",
      "tahmin/çıkarım (-dir) <-> doğrudan tanıklık (-di)", STD_HARD),
    F("DESID", "dilek-şart (-se) + yeterlilik", "kip_dilek", "chain",
      "keşke yapabilseydim <-> yaptım / yapamadım", HARD),
    # ---- converbs / ulaçlar ----------------------------------------------------------------
    F("CVB.WHEN", "-ince ulacı (zaman)", "ulac_zaman", "single",
      "olay gerçekleşince başlayan durum <-> olaydan bağımsız durum", STD),
    F("CVB.BY", "-erek ulacı (araç/tarz)", "ulac_tarz", "single",
      "eylemi o yolla yaparak <-> o eylemi yapmadan / başka yolla", STD_HARD),
    F("CVB.AND", "-ip ulacı (ardışıklık)", "ulac_ardisiklik", "single",
      "önce biri sonra öteki (-ip) <-> yalnızca biri", STD),
    F("CVB.WITHOUT", "-meden ulacı (olumsuz eşzamanlılık)", "ulac_olumsuz", "single",
      "o eylemi yapmadan (-meden) <-> yaptıktan sonra (-dikten sonra)", ALL),
    F("CVB.ASLONG", "-dikçe ulacı (orantı/sürerlik)", "ulac_orantililik", "single",
      "arttıkça artan bağıntı (-dikçe) <-> tek seferlik neden-sonuç", STD_HARD),
    F("CVB.SINCE", "-(y)alı ulacı (başlangıçtan bu yana)", "ulac_sure", "single",
      "o andan bu yana geçen süre (-alı) <-> o ana kadar geçen süre (-e kadar)", STD_HARD),
    F("CVB.WHILE", "-ken ulacı (eşzamanlılık)", "ulac_eszamanlilik", "single",
      "tam o sırada (-ken) <-> ondan önce/sonra", STD),
    F("CVB.ABOUTTO", "-mek üzere (yakınlık)", "gorunus_yakinlik", "single",
      "olmak üzere, henüz olmamış <-> çoktan olmuş (-dikten sonra)", STD_HARD),
    F("CVB.NEGINS", "-meksizin (hiç yapmaksızın)", "ulac_olumsuz", "single",
      "hiç yapmaksızın <-> yaparak", HARD),
    # ---- nominalisation and participles -----------------------------------------------------
    F("NMLZ.MEK", "adlaştırma (-mek/-mak)", "adlastirma", "single",
      "eylemin kendisi <-> eylemin gerçekleşmiş hâli", STD),
    F("NMLZ.ME", "adlaştırma (-me/-ma) + iyelik", "adlastirma", "chain",
      "kimin yaptığı bilgisi (-mesi/-mem) değişir", STD_HARD),
    F("NMLZ.DIK", "adlaştırma (-diğ- + iyelik)", "adlastirma", "chain",
      "yaptığı (gerçekleşmiş) <-> yapacağı (gerçekleşmemiş)", STD_HARD),
    F("NMLZ.ECEK", "adlaştırma (-eceğ- + iyelik)", "adlastirma", "chain",
      "yapacağı (gelecek) <-> yaptığı (geçmiş)", STD_HARD),
    F("PTCP.SUBJ", "sıfat-fiil (-en/-an, özne)", "sifat_fiil", "single",
      "eylemi yapan <-> eyleme maruz kalan", STD_HARD),
    F("PTCP.OBJ", "sıfat-fiil (-diği, nesne/tümleç)", "sifat_fiil", "chain",
      "kimin yaptığı nesne <-> başkasının yaptığı nesne", STD_HARD),
    F("PTCP.FUT", "sıfat-fiil (-ecek, gelecek)", "sifat_fiil", "single",
      "yapılacak olan <-> yapılmış olan (-miş)", STD),
    # ---- nominal: possession, number ---------------------------------------------------------
    F("POSS.1SG", "iyelik 1. tekil (-m)", "iyelik", "single",
      "benim <-> senin/onun", ALL),
    F("POSS.2SG", "iyelik 2. tekil (-n)", "iyelik", "single",
      "senin <-> benim/onun", ALL),
    F("POSS.3SG", "iyelik 3. tekil (-si/-i)", "iyelik", "single",
      "onun <-> benim/bizim", STD),
    F("POSS.1PL", "iyelik 1. çoğul (-miz)", "iyelik", "single",
      "bizim <-> benim (tek kişi)", ALL),
    F("POSS.2PL", "iyelik 2. çoğul (-niz)", "iyelik", "single",
      "sizin <-> senin", STD),
    F("POSS.3PL", "iyelik 3. çoğul (-leri)", "iyelik", "single",
      "onların <-> onun (tek kişi)", STD),
    F("PL", "çoğul (-ler/-lar)", "cogul", "single",
      "birden çok <-> tek", ALL),
    F("PL.POSS.CASE", "çoğul + iyelik + hal", "composition", "chain",
      "kimin, kaç tane ve hangi yönde ilişkilendiği birlikte değişir", STD_HARD),
    # ---- nominal: case -------------------------------------------------------------------------
    F("ACC", "belirtme hâli (-i/-ı/-u/-ü)", "hal_belirtme", "single",
      "belirli bir nesne (-i) <-> belirsiz/genel nesne (eksiz)", ALL),
    F("DAT", "yönelme hâli (-e/-a)", "hal_yonelme", "single",
      "hedefe doğru (-e) <-> kaynaktan (-den)", ALL),
    F("LOC", "bulunma hâli (-de/-da/-te/-ta)", "hal_bulunma", "single",
      "bir yerde bulunma (-de) <-> bir yerden ayrılma (-den)", ALL),
    F("ABL", "ayrılma hâli (-den/-dan/-ten/-tan)", "hal_ayrilma", "single",
      "kaynaktan (-den) <-> hedefe (-e)", ALL),
    F("INS", "vasıta hâli (-le/-la/-yle)", "hal_vasita", "single",
      "biriyle/bir şeyle birlikte (-le) <-> birine/bir şeye (-e)", ALL),
    F("GEN", "ilgi hâli (-in/-ın/-nin)", "hal_ilgi", "single",
      "aitlik/sahiplik (-in) <-> yalın ad", STD),
    F("EQU", "eşitlik hâli (-ce/-ca)", "hal_esitlik", "single",
      "o biçimde/ona göre (-ce) <-> o kadar/o yönde", STD_HARD),
    F("ALLO.ABL", "ayrılma hâli ünlü/ünsüz uyumu (-dan/-den/-tan/-ten)", "allomorf", "single",
      "aynı hâl ekinin uyum değişkesi; anlam aynı kalır, biçim değişir", ("minimal",)),
    F("ALLO.DAT", "yönelme hâli kaynaştırma ünsüzü (-ya/-ye)", "allomorf", "single",
      "ünlüyle biten adda kaynaştırma -y-; anlam aynı kalır", ("minimal",)),
    # ---- nominal: derivational ------------------------------------------------------------------
    F("PRIV", "yoksunluk (-siz/-sız)", "yoksunluk", "single",
      "o şey olmadan (-siz) <-> o şeyle birlikte (-li)", ALL),
    F("PROP", "varlık/bulunma (-li/-lı)", "yoksunluk_bulunma", "single",
      "o şeyi içeren (-li) <-> o şeyden yoksun (-siz)", ALL),
    F("REL.KI", "ilgi/aitlik eki (-ki)", "ilgi_zamiri", "single",
      "belirtilen yere/zamana ait olan (-ki) <-> ait olmayan / başka yere ait", ALL),
    F("AGT", "meslek/uğraş eki (-ci/-cı)", "turetim_meslek", "single",
      "işi yapan kişi (-ci) <-> işin kendisi", STD),
    F("ABST", "soyutlama eki (-lik/-lık)", "turetim_soyut", "single",
      "soyut nitelik/görev (-lik) <-> somut nesne/kişi", STD),
    F("VBLZ", "adtan eylem (-leş/-len)", "turetim_eylem", "single",
      "o duruma gelme (-leş) <-> o duruma sahip olma (-len)", STD_HARD),
    F("DISTR", "üleştirme (-şer/-er)", "uslestirme", "single",
      "her birine ayrı ayrı (-şer) <-> toplamda", STD_HARD),
    # ---- chains where scope order decides the meaning ---------------------------------------------
    F("CAUS.PASS.NEG", "ettirgen + edilgen + olumsuzluk", "composition", "chain",
      "kim yaptırdı, kime yaptırıldı ve hangi öge olumsuz — kapsam sırası anlamı belirler", STD_HARD),
    F("EVID.COND.NEG", "kanıtsallık + koşul + olumsuzluk", "composition", "chain",
      "duyulmuş olsaydı ne olurdu <-> duyulmuştu ve yine de oldu", HARD),
    F("NMLZ.CASE.CNTR", "adlaştırma + hal + karşı-olgusal", "composition", "chain",
      "adlaştırılmış eylemin hâli ve gerçekleşip gerçekleşmediği birlikte değişir", HARD),
    F("PRIV.VS.NEG", "yoksunluk (ad üzerinde) / olumsuzluk (fiil üzerinde)", "yoksunluk_olumsuzluk",
      "chain", "olumsuzluk hangi ögeye bağlanıyor: ada mı (-siz) fiile mi (-ma)", STD_HARD),
    F("TENSE.PERS.NEG", "zaman + kişi + olumsuzluk", "composition", "chain",
      "kim, ne zaman ve olumlu mu olumsuz mu — üçü birlikte", STD_HARD),
    F("ABIL.COND", "yeterlilik + koşul (karşı-olgusal)", "yeterlilik_kosul", "chain",
      "yapabilseydi <-> yapabildi / yapamadı", HARD),
    F("EVID.POSS.NEG", "kanıtsallık + iyelik + olumsuzluk", "composition", "chain",
      "kimin bilgisi, nereden edinildi ve olumsuz mu", HARD),
    F("RECP.CAUS", "işteş + ettirgen", "cati_istes_ettirgen", "chain",
      "karşılıklı eylem <-> birine karşılıklı eylem yaptırma", HARD),
    F("CAUS.REFL.NEG", "ettirgen / dönüşlü + olumsuzluk", "cati_ettirgen_donuslu", "chain",
      "kendi yaptı <-> başkasına yaptırdı, ve olumsuzluk hangisine düşüyor", HARD),
    F("POSS.PL.ABL", "iyelik + çoğul + ayrılma", "composition", "chain",
      "kimin, kaç tanesinden — üç ek birlikte", STD_HARD),
]

FEATURE_BY_KEY = {f["key"]: f for f in TARGET_FEATURES}

# --------------------------------------------------------------------------- domains
# `perspective` is passed to the prompt so the query/record person convention is explicit.
# agent_memory is retained deliberately: it is v1.3.1's register, so train and test stay
# comparable on at least one slice.
DOMAINS = [
    dict(key="agent_memory", label="kişisel asistan kaydı",
         desc="Bir yapay zekâ asistanının kullanıcı adına tuttuğu kısa hatırlatma/kayıt notları.",
         perspective="birinci veya üçüncü kişi anı kaydı"),
    dict(key="workplace", label="iş yeri / ekip iletişimi",
         desc="Ekip içi yazışma, toplantı notu, görev takibi, izin ve onay süreçleri.",
         perspective="birinci veya üçüncü kişi"),
    dict(key="news", label="haber metni",
         desc="Kısa haber bülteni cümleleri: belediye, ulaşım, hava durumu, yerel olaylar.",
         perspective="üçüncü kişi, tarafsız haber dili"),
    dict(key="health", label="sağlık / hasta kaydı",
         desc="Randevu, reçete, tahlil sonucu, tedavi süreci notları. Teşhis tavsiyesi verilmez.",
         perspective="birinci veya üçüncü kişi"),
    dict(key="legal", label="sözleşme / hukuk",
         desc="Sözleşme maddeleri, tebligat, itiraz süresi, imza ve fesih süreçleri.",
         perspective="üçüncü kişi resmî dil"),
    dict(key="ecommerce", label="e-ticaret / müşteri desteği",
         desc="Sipariş, kargo, iade, garanti, destek talebi yazışmaları.",
         perspective="birinci veya üçüncü kişi"),
    dict(key="education", label="eğitim",
         desc="Ders kaydı, sınav, ödev teslimi, devamsızlık, burs başvurusu.",
         perspective="birinci veya üçüncü kişi"),
    dict(key="finance", label="finans / bankacılık",
         desc="Ödeme, taksit, ekstre, limit, otomatik talimat, faiz bildirimi.",
         perspective="birinci veya üçüncü kişi"),
    dict(key="travel", label="seyahat / ulaşım",
         desc="Bilet, rezervasyon, aktarma, bagaj, konaklama, vize randevusu.",
         perspective="birinci veya üçüncü kişi"),
    dict(key="daily_chat", label="gündelik konuşma",
         desc="Arkadaşlar ve aile arasında gündelik mesajlaşma; samimi ama düzgün Türkçe.",
         perspective="birinci veya ikinci kişi"),
]

# --------------------------------------------------------------------------- surface variation axes
PERSONS = ["1. tekil (ben)", "2. tekil (sen)", "3. tekil (o / adı geçen kişi)",
           "1. çoğul (biz)", "3. çoğul (onlar)"]

# `chars` is the target length band for candidate passages; it is enforced softly in the prompt
# and reported (not enforced) by the validators, because a hard cut would bias the confound audit.
PASSAGE_LENGTHS = [
    dict(key="short", label="tek cümle", chars=(40, 110),
         instruction="Her aday tek, kısa bir cümle olsun."),
    dict(key="medium", label="iki cümle", chars=(110, 220),
         instruction="Her aday iki cümle olsun; ikinci cümle ilkini destekleyen bir ayrıntı versin."),
    dict(key="long", label="üç-dört cümle, dolgu içeren", chars=(220, 420),
         instruction=("Her aday üç-dört cümle olsun. Ayırt edici ek, pasajın BAŞINDA veya SONUNDA "
                      "değil ORTASINDA, bir yan cümlenin içinde geçsin; kalan cümleler konuyla "
                      "ilgili ama ayırt edici olmayan dolgu bilgisi taşısın. Dolgu cümleleri TÜM "
                      "adaylarda benzer uzunlukta olsun.")),
]

# --------------------------------------------------------------------------- entity pools
# Disjoint from v1.3.1 (which uses Ali, Ayşe, Cem, Deniz, Derya, Kerem, Selin).
BANNED_NAMES = {"Ali", "Ayşe", "Cem", "Deniz", "Derya", "Kerem", "Selin"}

NAMES = [
    "Bahar", "Burak", "Ceyda", "Doruk", "Ebru", "Emre", "Esra", "Fatih", "Gamze", "Görkem",
    "Hakan", "Hande", "İlker", "İpek", "Kaan", "Kıvanç", "Lale", "Levent", "Melis", "Merve",
    "Murat", "Nazlı", "Nihan", "Okan", "Onur", "Özge", "Pelin", "Rüya", "Sedef", "Serkan",
    "Sinem", "Tolga", "Tuğçe", "Ufuk", "Umut", "Yağmur", "Yiğit", "Zeynep", "Barış", "Çağla",
    "Efe", "Ferda", "Gizem", "Halil", "Irmak", "Koray", "Meltem", "Nadir", "Oya", "Sarp",
]

PLACES = [
    "Kadıköy", "Bornova", "Çankaya", "Nilüfer", "Konak", "Muratpaşa", "Odunpazarı", "Selçuklu",
    "Şahinbey", "Tepebaşı", "Karşıyaka", "Yenimahalle", "Balçova", "Osmangazi", "Meram",
    "Trabzon", "Eskişehir", "Gaziantep", "Samsun", "Denizli", "Kayseri", "Malatya", "Aydın",
]

ORGS = [
    "Aydın Lojistik", "Berrak Sigorta", "Civan Yazılım", "Defne Yayıncılık", "Ergene Tekstil",
    "Filiz Gıda", "Günebakan Kooperatifi", "Hisar Mimarlık", "Işıl Enerji", "Kavak Matbaa",
    "Lodos Denizcilik", "Meşe Eğitim Kurumları", "Narin Kimya", "Ova Tarım", "Poyraz Turizm",
    "Rüzgâr Ambalaj", "Söğüt Mobilya", "Tuna Makine", "Ulu Çelik", "Vadi Danışmanlık",
]

OBJECTS = [
    "fatura", "kargo kolisi", "abonelik sözleşmesi", "laboratuvar sonucu", "yıllık rapor",
    "servis anahtarı", "başvuru dilekçesi", "sunum dosyası", "kira kontratı", "tahlil raporu",
    "sipariş formu", "bakım çizelgesi", "eğitim sertifikası", "toplantı tutanağı", "kredi ekstresi",
    "uçak bileti", "otel voucherı", "ders programı", "burs belgesi", "garanti belgesi",
    "iade talebi", "izin formu", "vize randevusu", "sağlık raporu", "denetim listesi",
    "stok sayımı", "tedarik siparişi", "sunum taslağı", "arıza kaydı", "teslimat fişi",
]

# --------------------------------------------------------------------------- quotas
LAYER_QUOTA = {"single": 0.60, "chain": 0.40}
# Tier quota is conditional on layer. A `minimal` item is a one-token, one-suffix contrast, which
# a multi-suffix chain almost never is — drawing minimal uniformly across both layers just gets
# re-routed to hard by the feature constraint and silently inflates the hard share.
TIER_QUOTA_BY_LAYER = {
    "single": {"standard": 0.50, "hard": 0.25, "minimal": 0.25},
    "chain": {"standard": 0.70, "hard": 0.30, "minimal": 0.00},
}
# Realised marginal, measured not assumed: ~52% standard / ~33% hard / ~15% minimal. `hard` runs
# above its nominal 25% because 8 of the 21 chain features (counterfactual conditional, evidential
# +conditional+negation, reciprocal+causative, ...) are hard-ONLY constructions — the inventory,
# not the sampler, decides that. `minimal` lands on target, which is the one that matters: minimal
# pairs are the highest-signal contrastive training data and v1.3.1 has only 6 of them.
TIER_QUOTA = {"standard": 0.52, "hard": 0.33, "minimal": 0.15}
LENGTH_QUOTA = {"short": 0.45, "medium": 0.35, "long": 0.20}

N_CANDIDATES = 11          # 1 positive + 5 hard negatives + 5 easy negatives, as in v1.3.1
REQUIRED_SUBTYPES = ("morph_counterfactual", "same_feature_wrong_content",
                     "partial_trap", "state_variant")


def _weighted_cycle(quota, n, rng):
    """Deterministic multiset honouring `quota` proportions, then shuffled."""
    out = []
    for key, share in quota.items():
        out += [key] * round(share * n)
    while len(out) < n:
        out.append(max(quota, key=quota.get))
    out = out[:n]
    rng.shuffle(out)
    return out


def slot_id(slot):
    """Stable id: same slot spec always maps to the same cache file across runs."""
    raw = f"{slot['target_feature']}|{slot['tier']}|{slot['layer']}|{slot['domain']}|" \
          f"{slot['passage_length']}|{slot['index']}"
    return f"{slot['target_feature'].lower().replace('.', '_')}_{slot['index']:04d}_" \
           f"{hashlib.sha1(raw.encode()).hexdigest()[:6]}"


def _round_robin(pool, n, rng):
    """n items from `pool`, touching every element before repeating any."""
    out = []
    while len(out) < n:
        block = list(pool)
        rng.shuffle(block)
        out += block
    return out[:n]


PLAN_SIZE = 1200   # canonical plan length; any `n_slots` is a PREFIX of this


def plan_slots(n_slots, seed=20260801):
    """Build the generation plan. Returns the first `n_slots` of a fixed PLAN_SIZE plan.

    Prefix-stability is the point: `--target 250` today and `--target 600` tomorrow must share
    their first 250 slots, or the smaller run's API spend is thrown away. Because every quota
    cycle is uniformly shuffled before truncation, a prefix stays quota-correct in expectation
    and covers features evenly.

    `layer` is drawn from its quota FIRST and the feature is then picked from the matching pool,
    because layer is a property of the feature — drawing the feature first would let the pool's
    single/chain ratio (which is an accident of the inventory) decide the split.

    `tier` is drawn from its quota but intersected with what the feature actually supports: a
    vowel-harmony allomorph pair only makes sense at `minimal`, a counterfactual conditional only
    at `hard`. When the drawn tier is unsupported the fallback re-draws from the feature's own
    tiers *weighted by the global quota*, so unsupported draws do not silently bias toward `hard`.
    """
    total = max(n_slots, PLAN_SIZE)
    rng = random.Random(seed)
    lengths = _weighted_cycle(LENGTH_QUOTA, total, rng)
    layers = _weighted_cycle(LAYER_QUOTA, total, rng)

    pools = {lay: [f for f in TARGET_FEATURES if f["layer"] == lay] for lay in ("single", "chain")}
    queues = {lay: _round_robin(pool, total, rng) for lay, pool in pools.items()}
    tier_queues = {lay: _weighted_cycle({t: w for t, w in q.items() if w > 0}, total, rng)
                   for lay, q in TIER_QUOTA_BY_LAYER.items()}
    domain_queue = _round_robin(DOMAINS, total, rng)   # round-robin, not sampling: even coverage
    cursors = {"single": 0, "chain": 0}

    slots = []
    for i in range(total):
        lay = layers[i]
        cur = cursors[lay]
        feat = queues[lay][cur]
        tier = tier_queues[lay][cur]
        cursors[lay] += 1

        if tier not in feat["tiers"]:
            allowed = list(feat["tiers"])
            weights = [TIER_QUOTA_BY_LAYER[lay].get(t, 0.01) or 0.01 for t in allowed]
            tier = rng.choices(allowed, weights=weights)[0]
        length = lengths[i]
        if tier == "minimal" and length == "long":
            length = "short"          # a minimal pair buried in filler tests two things at once
        domain = domain_queue[i]
        slots.append(dict(
            index=i,
            target_feature=feat["key"],
            ek_turu=feat["ek_turu"],
            phenomenon=feat["phenomenon"],
            contrast=feat["contrast"],
            layer=feat["layer"],
            tier=tier,
            domain=domain["key"],
            domain_label=domain["label"],
            domain_desc=domain["desc"],
            perspective=domain["perspective"],
            person=rng.choice(PERSONS),
            passage_length=length,
            seed_entities=dict(
                names=rng.sample(NAMES, 2),
                place=rng.choice(PLACES),
                org=rng.choice(ORGS),
                objects=rng.sample(OBJECTS, 2),
            ),
        ))
    for s in slots:
        s["slot_id"] = slot_id(s)
    return slots[:n_slots]


if __name__ == "__main__":
    import collections
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    slots = plan_slots(n)
    print(f"{len(slots)} slots over {len(TARGET_FEATURES)} features, {len(DOMAINS)} domains\n")
    for axis in ("tier", "layer", "domain", "passage_length"):
        c = collections.Counter(s[axis] for s in slots)
        print(f"{axis:15s} {dict(c.most_common())}")
    feats = collections.Counter(s["target_feature"] for s in slots)
    print(f"\nfeature coverage: min={min(feats.values())} max={max(feats.values())} "
          f"features={len(feats)}")
    print(f"\nexample slot:\n{slots[0]}")
