#!/usr/bin/env python
"""Self-test for the zero-API validation gates. Run before spending any API quota:

    conda run -n dl_hw1 python gen_morph_dataset.py --self-test

Three parts, in increasing order of what they prove:

1. **Phonology unit tests.** Known-good Turkish suffix pairs must pass and known-bad ones must
   fail, with the failure attributed to the right rule (harmony / assimilation / hiatus).
2. **No false rejections on human data.** All 50 hand-QC'd v1.3.1 items must clear the gates.
   Any item that does not is either a validator bug or a genuine v1.3.1 defect, and the two are
   distinguished explicitly via KNOWN_V1_DEFECTS rather than by loosening a threshold until the
   count reaches zero.
3. **No false acceptances.** Deliberately corrupted copies of real v1.3.1 items must fail, each
   on the specific gate that is supposed to catch it. This is the half that actually matters: a
   validator that accepts everything also passes part 2.

Reference distributions measured over v1.3.1 (these are what the thresholds in morph_validators
are calibrated against, and re-measuring them is how you would re-calibrate):

    positive is char-3gram top-1                  12% of items
    top1-top2 margin when positive is top-1       0.015 .. 0.055
    minimal-tier sim(positive, counterfactual)    0.741 .. 0.919 for sound items
    query->positive content overlap (prefix-4)    p05 0.20, median 0.50
    |len(positive) - median len| / median         median 0.53  <- positives ARE long by design,
                                                     which is why per-item length balance is a
                                                     corpus-level report, not a gate
"""
import copy
import json
from pathlib import Path

import morph_validators as V
from morph_taxonomy import N_CANDIDATES, REQUIRED_SUBTYPES

V1_PATH = Path(__file__).resolve().parent.parent / "morph_eval_set_v1.3.1_review_reviewer_C_fixed.json"

# Genuine defects in v1.3.1 that the gates correctly surface. Recorded here rather than silenced,
# because they are real findings about the test set: q18 and q53 are missing a partial_trap, and
# q53 carries two candidates labelled morph_counterfactual (its own fix_log flags the same class
# of problem elsewhere but never reached these two).
KNOWN_V1_DEFECTS = {
    "q18_comp_plural_case": ["eksik subtype: partial_trap"],
    "q53_min_poss": ["eksik subtype: partial_trap", "morph_counterfactual sayısı 2"],
    # The overlap-ordering gate's measured cost on human data: 3 of 44 standard/hard items (7%)
    # have a positive that out-scores every negative on query similarity, so a lexical matcher
    # solves them. Listed rather than tuned away — the gate is calibrated to catch v2.0's 29% rate,
    # and pretending it has no false-positive cost would be dishonest about the tradeoff.
    "q09_causative": ["pozitif sorguya en çok benzeyen aday"],
    "q40_modality2": ["pozitif sorguya en çok benzeyen aday"],
    "q43_poss_case": ["pozitif sorguya en çok benzeyen aday"],
}

# The one gate deliberately calibrated STRICTER than the human data, so it is excluded from the
# "v1.3.1 must pass" sweep and gets a directional test of its own instead (test_length_gate).
#
# Every other threshold here is set so v1.3.1 clears it, on the principle that a rule the
# human-QC'd set fails is more likely wrong than the humans are. Length is the exception: v1.3.1
# has a measured length artifact — a blind "pick the longest candidate" scorer hits 50% on it
# against a 9% chance level — so agreeing with v1.3.1 here would mean reproducing the defect.
# 42 of its 50 items fail this gate, and that is the point, not a false-positive rate.
LENGTH_GATE_MARKER = "sistematik olarak uzun"

# (positive_form, counterfactual_form) pairs that are well-formed Turkish
GOOD_PAIRS = [
    ("katılmadım", "katılamadım"),     # negation vs inability
    ("gelmedi", "gelemedi"),
    ("yazdım", "yazdın"),              # person
    ("evim", "evimiz"),                # possessive number
    ("imzasız", "imzalı"),             # privative vs proprietive
    ("kitapta", "kitaptan"),           # locative vs ablative, both assimilated
    ("evde", "evden"),
    ("okulda", "okuldan"),
    ("arabayı", "arabaya"),            # buffer -y- kept in both
    ("gözlerim", "gözlerimiz"),
    ("yaptı", "yapmış"),               # evidentiality
    ("geliyor", "gelmiyor"),           # invariant -yor must not trip harmony
    ("geliyordu", "gelmiyordu"),
    ("çocuğa", "çocukta"),             # softening in one member only
    ("açıp", "açtırdım"),              # 2-letter root: must not be rejected as "no shared stem"
    ("kalbe", "kalpten"),              # loanword taking a front suffix after a back vowel
    ("gidiyorken", "gitmişken"),       # invariant -ken
    ("imzalayabilmişti", "imzalamıştı"),   # -abil never harmonises: front i after back a
    ("hazırlayabilmişti", "hazırlamıştı"),
    ("yapabildim", "yapamadım"),           # ability vs inability, both well formed
    ("okuyabilir", "okuyamaz"),
    ("masadaki", "masadan"),               # invariant -ki (relative) vs ablative, shared stem
    ("gördüm", "gördün"),                  # rounding harmony satisfied after 'ö'
    ("okudum", "okudun"),                  # rounding harmony satisfied after 'u'
]

# (positive_form, counterfactual_form, substring expected in the failure message)
BAD_PAIRS = [
    ("kitapta", "kitapda", "ünsüz benzeşmesi"),        # d after voiceless p
    ("çocukta", "çocukda", "ünsüz benzeşmesi"),
    ("evde", "evlerdan", "büyük ünlü uyumu"),          # back suffix after front stem
    ("kitaplarda", "kitaplarden", "büyük ünlü uyumu"), # front suffix after back stem
    ("gözde", "gözda", "büyük ünlü uyumu"),
    ("okulda", "okulde", "büyük ünlü uyumu"),
    ("aldım", "aldun", "küçük ünlü uyumu"),            # rounding: 'u' after unrounded 'a'
    ("kitabı", "kitabu", "küçük ünlü uyumu"),
    # 'i' is front like 'ö', so backness is satisfied and only rounding can catch this one
    ("gördüm", "gördim", "küçük ünlü uyumu"),
    ("arabayı", "arabaı", "ünlü çakışması"),           # missing buffer consonant
    ("geldi", "gitti", "ortak kök yok"),               # different verb, not a suffix contrast
    ("evim", "evim", "aynı"),                          # no contrast at all
]


def _fail(msg, bucket):
    bucket.append(msg)


def test_phonology(errors):
    for a, b in GOOD_PAIRS:
        probs = V.check_critical_pair(a, b)
        if probs:
            _fail(f"[phonology] '{a}'/'{b}' geçerli olmalıydı, reddedildi: {probs}", errors)
    for a, b, expect in BAD_PAIRS:
        probs = V.check_critical_pair(a, b)
        if not probs:
            _fail(f"[phonology] '{a}'/'{b}' reddedilmeliydi, kabul edildi", errors)
        elif not any(expect in p for p in probs):
            _fail(f"[phonology] '{a}'/'{b}' yanlış gerekçeyle reddedildi: "
                  f"beklenen '{expect}', gelen {probs}", errors)


def _load_v1():
    return json.loads(V1_PATH.read_text(encoding="utf-8"))["items"]


def test_no_false_rejections(errors):
    """Every v1.3.1 item must pass, except the defects we have explicitly acknowledged."""
    for it in _load_v1():
        res = V.validate_item(it, N_CANDIDATES, REQUIRED_SUBTYPES)
        res.pop("morphology", None)      # v1.3.1 predates the critical_word fields
        flat = [p for probs in res.values() for p in probs
                if LENGTH_GATE_MARKER not in p]       # see LENGTH_GATE_MARKER
        allowed = KNOWN_V1_DEFECTS.get(it["query_id"], [])
        unexpected = [p for p in flat if not any(a in p for a in allowed)]
        if unexpected:
            _fail(f"[false-reject] {it['query_id']}: {unexpected}", errors)
    # and the acknowledged defects must still be detected — if a threshold change silences them,
    # that is a regression in the other direction
    for qid, expected in KNOWN_V1_DEFECTS.items():
        it = next(i for i in _load_v1() if i["query_id"] == qid)
        flat = [p for probs in V.validate_item(it, N_CANDIDATES, REQUIRED_SUBTYPES).values()
                for p in probs]
        for exp in expected:
            if not any(exp in p for p in flat):
                _fail(f"[regression] {qid}: bilinen kusur '{exp}' artık yakalanmıyor", errors)


def _base_item():
    """A sound v1.3.1 item, augmented with the critical_word fields v2.0 items carry."""
    it = copy.deepcopy(next(i for i in _load_v1() if i["query_id"] == "q100_min_person"))
    pos = next(c for c in it["candidates"] if c["role"] == "positive")
    cf = next(c for c in it["candidates"] if c.get("subtype") == "morph_counterfactual")
    # locate the single differing token pair to use as the critical words
    a, b = V.tokens(pos["text"]), V.tokens(cf["text"])
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    it["critical_word_query"], it["critical_word_counterfactual"] = diff[0]
    return it


def _without_length_gate(res):
    """Drop the length finding: the base item is a real v1.3.1 item and v1.3.1 has that artifact
    by construction, so it would mask every corruption test built on top of it."""
    out = {g: [p for p in ps if LENGTH_GATE_MARKER not in p] for g, ps in res.items()}
    return {g: ps for g, ps in out.items() if ps}


def test_no_false_acceptances(errors):
    base = _base_item()
    clean = _without_length_gate(V.validate_item(base, N_CANDIDATES, REQUIRED_SUBTYPES))
    if clean:
        _fail(f"[setup] temel öğe zaten başarısız: {clean}", errors)

    def expect(name, mutate, gate, needle):
        it = copy.deepcopy(base)
        mutate(it)
        res = V.validate_item(it, N_CANDIDATES, REQUIRED_SUBTYPES)
        probs = res.get(gate, [])
        if not any(needle in p for p in probs):
            _fail(f"[false-accept] '{name}' {gate} kapısında yakalanmalıydı "
                  f"('{needle}'); dönen: {res}", errors)

    def _pos(it):
        return next(c for c in it["candidates"] if c["role"] == "positive")

    def _cf(it):
        return next(c for c in it["candidates"] if c.get("subtype") == "morph_counterfactual")

    # 1. content-noun substitution in the positive — fix_log defect #1
    expect("içerik sözcüğü değişimi",
           lambda it: _pos(it).update(text="Tamamen alakasız bir konu hakkında kısa bir not."),
           "lexical", "içerik sözcüklerini korumuyor")

    # 2. a second gold: duplicate the positive into an easy_negative slot
    expect("çift altın (iki positive)",
           lambda it: it["candidates"][-1].update(role="positive"),
           "structure", "positive sayısı")

    # 3. two candidates labelled morph_counterfactual — fix_log defect #5
    expect("iki morph_counterfactual",
           lambda it: next(c for c in it["candidates"]
                           if c.get("subtype") == "state_variant").update(
                               subtype="morph_counterfactual"),
           "structure", "morph_counterfactual sayısı")

    # 4. missing required subtype
    expect("eksik partial_trap",
           lambda it: next(c for c in it["candidates"]
                           if c.get("subtype") == "partial_trap").update(subtype="state_variant"),
           "structure", "eksik subtype: partial_trap")

    # The next three probe the morphology gate through the TEXTS, not the metadata: the gate
    # resolves its pair from the passages and falls back to deriving it, so corrupting only the
    # reported critical words is (correctly) not a defect. See test 7.

    # 5. vowel-harmony violation in the counterfactual passage itself
    expect("ünlü uyumu bozuk counterfactual",
           lambda it: _cf(it).update(text=_cf(it)["text"].replace("aldın", "aldun")),
           "morphology", "ünlü uyumu")

    # 6. counterfactual shares no stem with the positive: not a minimal pair at all
    expect("minimal tier'da ek karşıtlığı yok",
           lambda it: _cf(it).update(
               text="Buzdolabı geçen sene peşin ödendi; fiş bir yerlerde kayboldu."),
           "morphology", "sözcük çifti bulunamadı")

    # 7. REGRESSION GUARD (positive test): bogus reported critical words over sound passages must
    # still pass, because the gate derives the real pair from the texts.
    bogus = copy.deepcopy(base)
    bogus["critical_word_query"] = "uydurulmusbirsozcuk"
    bogus["critical_word_counterfactual"] = "bambaskabirsey"
    res = _without_length_gate(V.validate_item(bogus, N_CANDIDATES, REQUIRED_SUBTYPES))
    if res:
        _fail(f"[false-reject] hatalı bildirilen kritik sözcükler metinlerden türetilerek "
              f"kurtarılmalıydı; dönen: {res}", errors)
    pair = V.resolve_critical_pair(bogus)
    if not pair or pair[3] != "derived":
        _fail(f"[false-reject] geri düşüş türetmesi çalışmadı: {pair}", errors)

    # 8. minimal tier where the counterfactual is a different sentence
    expect("minimal tier'da minimal olmayan çift",
           lambda it: _cf(it).update(
               text="Bambaşka bir konuda, tamamen ilgisiz ve çok daha uzun bir cümle kurdum."),
           "tier", "minimal bir çift değil")

    # 9. lexically solvable: make the positive a near-copy of the query
    def _shortcut(it):
        _pos(it)["text"] = it["query"] + " " + it["query"]
    expect("yüzeysel çözülebilirlik", _shortcut, "lexical", "yüzeysel olarak çözülebilir")

    # 9b. the LLM's characteristic failure: query copied verbatim with padding bolted on
    def _copy_pad(it):
        _pos(it)["text"] = it["query"] + " Bu konuda ayrıca kısa bir not daha düştüm."
    expect("kopyala-doldur pozitif", _copy_pad, "lexical", "birebir kopyalıyor")

    # 9c. v2.0's defect: a positive that re-words the query just enough to dodge the verbatim-run
    # gate while still out-scoring every negative on surface overlap. Uses a standard-tier item,
    # since the ordering gate deliberately exempts minimal tier.
    std = copy.deepcopy(next(i for i in _load_v1() if i["query_id"] == "q06_aspect"))
    std_pos = next(c for c in std["candidates"] if c["role"] == "positive")
    words = V.tokens(std["query"])
    std_pos["text"] = " ".join(words[:2] + ["ayrıca"] + words[2:]) + " ve bu böyle kaldı."
    res = V.validate_item(std, N_CANDIDATES, REQUIRED_SUBTYPES)
    if not any("en çok benzeyen aday" in p for p in res.get("lexical", [])):
        _fail(f"[false-accept] 'sorguya yakın yeniden yazım' örtüşme-sıralaması kapısında "
              f"yakalanmalıydı; dönen: {res}", errors)

    # 10. English leakage
    expect("İngilizce sızıntı",
           lambda it: it["candidates"][-1].update(
               text="The user said that this document should have been sent."),
           "text", "İngilizce")

    # 11. unfilled placeholder
    expect("yer tutucu",
           lambda it: it["candidates"][-1].update(text="[isim] dün [nesne] teslim etti."),
           "text", "yer tutucu")


def test_length_gate(errors):
    """Directional test for the one gate stricter than the human data.

    Asserting "v1.3.1 passes" is the wrong contract here, so assert the intended discrimination
    instead: the gate must fire on the set that HAS the length artifact (v1.3.1, 50% blind-longest)
    and stay quiet on the set that does not (v2.0, 19%). A gate that fires on both is broken in one
    direction; a gate that fires on neither is broken in the other.
    """
    def rate(items):
        n = sum(1 for it in items
                if any(LENGTH_GATE_MARKER in p for p in V.check_lexical(it)))
        return 100.0 * n / max(len(items), 1)

    v1_rate = rate(_load_v1())
    if v1_rate < 60:
        _fail(f"[length-gate] v1.3.1'de yalnızca %{v1_rate:.0f} tetiklendi; bu kümede uzunluk "
              f"artefaktı ölçülmüş durumda (kör en-uzun %50), kapı çok gevşek", errors)

    prev = Path(__file__).resolve().parent / "data_morph_v2/archive_v2.0/morph_train_v2.0.json"
    if prev.exists():
        v20 = json.loads(prev.read_text(encoding="utf-8"))["items"]
        v20_rate = rate(v20)
        if v20_rate > 20:
            _fail(f"[length-gate] uzunluk bakımından dengeli v2.0'da %{v20_rate:.0f} tetiklendi; "
                  f"kapı çok sıkı", errors)
        return v1_rate, v20_rate
    return v1_rate, None


def test_prompt_contract(errors):
    """Three regression tests for bugs that already cost real API budget.

    Each one was found by hand, after the fact, by reading rejected items — which is the expensive
    way to find them. All three are properties of the code, checkable for free.
    """
    from morph_prompts import build_generation_prompt
    from morph_taxonomy import PASSAGE_LENGTHS, plan_slots
    import gen_morph_dataset as G

    ls = {l["key"]: l for l in PASSAGE_LENGTHS}
    slots = plan_slots(1200)

    # (1) Tier recipes are mutually exclusive. A conditional buried in a long prompt gets dropped:
    # half of all `minimal` slots followed the standard recipe and failed the minimal-pair check.
    marks = {"minimal": "POZİTİFİN ÇEKİRDEĞİNİ AYNEN KOPYALA",
             "standard": "çekirdek olarak SORGUYU aynen kopyala"}
    for tier in ("minimal", "standard", "hard"):
        slot = next(s for s in slots if s["tier"] == tier)
        prompt = build_generation_prompt(slot, set(), ls[slot["passage_length"]])
        want = marks["minimal"] if tier == "minimal" else marks["standard"]
        other = marks["standard"] if tier == "minimal" else marks["minimal"]
        if want not in prompt:
            _fail(f"[prompt] {tier}: kendi ADIM 4 tarifi istemde yok", errors)
        if other in prompt:
            _fail(f"[prompt] {tier}: DİĞER katmanın ADIM 4 tarifi de istemde — model yanlış "
                  f"olanı izleyebilir", errors)

    # (2) Cache keys must ignore the rolling ban list. When they did not, a resumed run fell off
    # its own cache after 216 of 718 slots because one differently-classified item shifted every
    # subsequent hash.
    slot = slots[0]
    base = build_generation_prompt(slot, set(), ls[slot["passage_length"]])
    banned = build_generation_prompt(slot, {"sözleşme", "toplantı", "rapor"},
                                     ls[slot["passage_length"]])
    if G.cache_key(base) != G.cache_key(banned):
        _fail("[cache] yasak liste önbellek anahtarını değiştiriyor; sürdürülen çalıştırma "
              "kendi önbelleğini ıskalar", errors)
    if base == banned:
        _fail("[cache] yasak liste isteme hiç yansımıyor", errors)

    # (3) Frame assembly: every candidate must carry the shared frame, byte-identical. This is what
    # makes length balance a property of the code rather than of the model's compliance.
    raw = {
        "query": "Derya dünkü toplantıya katılmamıştı.",
        "frame_before": "Ofiste sabah toplantısı hazırlıkları sürerken,",
        "frame_after": "Gün sonunda tutanak paylaşıldı.",
        "critical_word_query": "katılmamıştı",
        "critical_word_counterfactual": "katıldı",
        "candidates": [{"role": "positive", "subtype": "none", "core": "Derya toplantıda yoktu.",
                        "note": "", "violated_requirement": ""},
                       {"role": "hard_negative", "subtype": "morph_counterfactual",
                        "core": "Derya dünkü toplantıya katıldı.", "note": "",
                        "violated_requirement": ""}],
    }
    item = G.assemble_item(slots[0], raw)
    for c in item["candidates"]:
        if not c["text"].startswith(raw["frame_before"]):
            _fail(f"[frame] {c['id']}: ortak çerçeve başa eklenmemiş", errors)
        if not c["text"].endswith(raw["frame_after"]):
            _fail(f"[frame] {c['id']}: ortak çerçeve sona eklenmemiş", errors)
        if not c.get("core"):
            _fail(f"[frame] {c['id']}: `core` saklanmamış; doğrulayıcılar çerçeveyi de ölçer",
                  errors)
    lens = {len(c["text"]) - len(c["core"]) for c in item["candidates"]}
    if len(lens) != 1:
        _fail(f"[frame] çerçeve katkısı adaylar arasında değişiyor: {lens}", errors)


def test_corpus_gates(errors):
    items = _load_v1()
    dups = V.find_near_duplicates(items)
    if dups:
        _fail(f"[corpus] v1.3.1 kendi içinde yakın kopya içeriyor: {dups[:3]}", errors)
    # v1.3.1 against itself must be maximally leaky — proves the leakage detector actually fires
    hits = V.check_test_leakage(items)
    if len(hits) < len(items):
        _fail(f"[corpus] sızıntı denetimi v1.3.1'i kendisine karşı yakalayamadı "
              f"({len(hits)}/{len(items)})", errors)
    rep = V.confound_report(items)
    if rep["sparse_char3gram_top1_is_gold"] > 0.35:
        _fail(f"[corpus] confound raporu beklenmedik: {rep}", errors)
    return rep


def run(verbose=True):
    errors = []
    test_phonology(errors)
    test_no_false_rejections(errors)
    test_no_false_acceptances(errors)
    lg = test_length_gate(errors)
    test_prompt_contract(errors)
    rep = test_corpus_gates(errors)

    if verbose:
        print("=" * 78)
        print("DOĞRULAYICI ÖZ-TESTİ")
        print("=" * 78)
        print(f"  faz 1  fonoloji birim testleri : {len(GOOD_PAIRS)} geçerli + "
              f"{len(BAD_PAIRS)} geçersiz çift")
        print(f"  faz 2  v1.3.1 yanlış-red       : 50 insan-onaylı öğe, "
              f"{len(KNOWN_V1_DEFECTS)} bilinen kusur hariç")
        print("  faz 3  yanlış-kabul            : 11 kasıtlı bozma")
        print(f"  faz 4  uzunluk kapısı          : v1.3.1 %{lg[0]:.0f} tetiklendi "
              f"(artefakt var, beklenen), v2.0 %{(lg[1] if lg[1] is not None else -1):.0f} "
              f"(dengeli, beklenen)")
        print("  faz 5  istem sözleşmesi        : katman tarifi, önbellek anahtarı, "
              "çerçeve montajı")
        print(f"  faz 6  külliyat kapıları       : yakın-kopya, sızıntı, confound")
        print()
        print("  v1.3.1 referans confound raporu:")
        for k, v in rep.items():
            print(f"    {k:32s} {v}")
        print()
        if errors:
            print(f"  BAŞARISIZ — {len(errors)} sorun:")
            for e in errors:
                print(f"    - {e}")
        else:
            print("  TÜM TESTLER GEÇTİ")
        print("=" * 78)
    return errors


if __name__ == "__main__":
    import sys
    sys.exit(1 if run() else 0)
