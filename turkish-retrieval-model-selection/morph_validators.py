#!/usr/bin/env python
"""Zero-API validation gates for generated Turkish morphological retrieval items.

Everything here runs locally and costs no API quota, so it runs BEFORE the LLM judge and rejects
malformed items without spending requests on them.

Gate groups
-----------
`check_structure`   schema, role counts, single gold, required subtypes, id uniqueness
`check_morphology`  Turkish phonology of the critical-word pair — the literature-driven gate
`check_lexical`     content-word preservation, lexical-shortcut solvability, confound balance
`check_tier`        minimal-tier one-token/one-suffix requirement
`check_text_sanity` no English leakage, markdown, or unfilled placeholders

Why the morphology gate exists
------------------------------
LLMs are measurably weak at Turkish morphological *productivity* (Ismayilzada et al., NAACL 2025);
TurBLiMP deliberately used a masked LM plus a rule-based analyzer with human verification rather
than an instruction-tuned generator. So the LLM is trusted for semantics and audited for
morphology. Turkish phonotactics are deterministic enough that vowel harmony, consonant
assimilation and vowel hiatus can be checked with rules and no analyzer dependency.

Precision/recall stance: assimilation, hiatus, stem-sharing and word-in-text failures are HARD
rejections — Turkish admits no exceptions to those at a suffix boundary. Vowel-harmony failures
are also rejections, but they carry a known false-positive class (Arabic/French loanwords such as
`kalp -> kalbe`, `rol -> role` take front suffixes after a back root vowel). `LOAN_FRONT_STEMS`
covers the common ones and every harmony rejection is logged with its word, so the report can show
the actual false-positive rate rather than hiding it. A false rejection costs one generation call;
a false acceptance corrupts the dataset. The asymmetry justifies the strictness.
"""
import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_PATH = HERE.parent / "morph_eval_set_v1.3.1_review_reviewer_C_fixed.json"

# --------------------------------------------------------------------------- Turkish phonology
VOWELS = set("aeıioöuüâîû")
BACK = set("aıouâû")
FRONT = set("eiöüî")
HIGH = set("ıiuü")                   # suffix vowels subject to rounding harmony
ROUNDED = set("oöuü")
VOICELESS = set("fstkçşhp")          # "fıstıkçı şahap"

# Suffixes whose vowel does not harmonise with the stem. After each of these, harmony continues
# from the suffix's OWN vowel (geliyor -> geliyordu, not *geliyordü).
#
# `-abil/-ebil` matters most here and is easy to miss: it is a compound of the verb `bilmek`, so
# its `i` stays front regardless of the stem — `imzalayabilmişti`, `okuyabildim`, `yapabilir` all
# put a front `i` after back vowels, and everything after it harmonises with that `i`. Without
# this entry the gate rejects every well-formed abilitative in the dataset.
INVARIANT_SUFFIXES = [("iyor", "o"), ("ıyor", "o"), ("uyor", "o"), ("üyor", "o"), ("yor", "o"),
                      ("yabil", "i"), ("yebil", "i"), ("abil", "i"), ("ebil", "i"), ("bil", "i"),
                      ("ıver", "i"), ("iver", "i"), ("uver", "i"), ("üver", "i"),
                      ("ken", "e"), ("leyin", "i"), ("layın", "ı"), ("ki", "i"),
                      ("daş", "a"), ("taş", "a"), ("imtırak", "a"), ("gil", "i")]
# Longest first, so `yabil` wins over `abil` and `abil` over `bil` when the shared prefix happens
# to cut before, at, or after the linking vowel. Which of those it does depends on the other member
# of the pair (`yapabildim`/`yapamadım` shares `yapa`, so the region starts at `bil`).
INVARIANT_SUFFIXES.sort(key=lambda kv: -len(kv[0]))

# Loanword stems that take FRONT suffixes despite a back root vowel. Not exhaustive — Turkish
# has no closed list — but it covers the frequent ones. Anything missed shows up as a logged
# harmony rejection, not as silent corruption.
LOAN_FRONT_STEMS = {
    "kalp", "kalb", "hal", "hâl", "harf", "rol", "gol", "alkol", "petrol", "kontrol", "protokol",
    "konsol", "futbol", "ampul", "mol", "usul", "mahsul", "meşgul", "kabul", "sual", "misal",
    "ihtimal", "saat", "hakikat", "dikkat", "şefkat", "sıhhat", "kanaat", "menfaat", "sanat",
    "istikbal", "ideal", "lokal", "normal", "sinyal", "kristal", "metal", "moral", "final",
    "santral", "terminal", "festival", "kanal", "hastal",
}

STOPWORDS = {
    "ve", "ile", "ama", "fakat", "ancak", "çünkü", "ya", "da", "de", "ki", "mi", "mı", "mu", "mü",
    "bir", "bu", "şu", "o", "ben", "sen", "biz", "siz", "onlar", "için", "gibi", "kadar", "daha",
    "en", "çok", "az", "her", "hiç", "bazı", "tüm", "sonra", "önce", "şimdi", "artık", "yine",
    "hem", "ise", "değil", "var", "yok", "olarak", "üzere", "diye", "göre", "beri", "dolayı",
    "rağmen", "ne", "nasıl", "neden", "niye", "kim", "hangi", "kez", "defa", "yani", "zaten",
}

_WORD_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+")


def tokens(text):
    return _WORD_RE.findall(text.lower())


def _last_vowel(s):
    for ch in reversed(s):
        if ch in VOWELS:
            return ch
    return None


def _harmony_class(v):
    return "back" if v in BACK else "front" if v in FRONT else None


def lcp_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# Turkish consonant softening (ünsüz yumuşaması): a root-final p/ç/t/k voices to b/c/d/ğ before a
# vowel-initial suffix. `git-` surfaces as `gid-` in `gidiyor`, `kitap` as `kitab-` in `kitabı`.
# Collapsing the pair lets the shared-stem test see through the alternation, so `gidiyorken` and
# `gitmişken` are correctly recognised as one verb while `geldi` / `gitti` still are not.
_SOFTEN = str.maketrans("bcdğ", "pçtk")


def soft_lcp_len(a, b):
    return lcp_len(a.translate(_SOFTEN), b.translate(_SOFTEN))


def common_suffix_len(a, b):
    return lcp_len(a[::-1], b[::-1])


def _shares_stem(a, b, strict=False):
    """`strict` is for pairs we DERIVE rather than pairs the generator reported.

    The permissive bar exists for two-letter Turkish roots (`aç-` in açıp/açtırdım), but applied
    to an unconstrained cross-product of every differing token it also matches noise: `belgesi`
    and `peşin` share `be` after softening and would be handed to the phonology checker as if they
    were a morphological contrast. A reported pair is a claim about a specific word, so it earns
    the benefit of the doubt; a derived pair has to prove itself.
    """
    n = soft_lcp_len(a, b)
    if strict:
        return n >= 3 and n >= 0.35 * min(len(a), len(b))
    return n >= 2 and n >= 0.3 * min(len(a), len(b))


def _shares_ending(a, b):
    """Same construction, different stem — `benimkiydi` / `seninkiydi`, `bende` / `sende`.

    Turkish person contrasts on pronouns surface at the FRONT of the word, so the shared-stem test
    cannot see them; what they share is a long common ending. Treated as a valid contrast, but no
    phonology is run on it: the two endings are byte-identical, so if one is well formed so is the
    other, and the differing region is a root, where disharmony is legal.
    """
    n = common_suffix_len(a, b)
    return n >= 3 and n >= 0.4 * min(len(a), len(b))


def derive_critical_pair(pos_text, cf_text):
    """Recover the contrasting word pair from the texts themselves.

    Preferred over the generator's own `critical_word_*` fields, which in practice are often
    reported in a citation form that never appears in the passage (`okumadan` written down while
    the text says `okunmadan`). Rejecting a sound item over a metadata slip wastes a generation
    call; reading the answer off the data does not.

    Returns (word_from_positive, word_from_counterfactual, mode) or None.
    """
    a, b = tokens(pos_text), tokens(cf_text)
    only_a = [t for t in a if t not in set(b)]
    only_b = [t for t in b if t not in set(a)]
    best = None
    for x in only_a:
        for y in only_b:
            if _shares_stem(x, y, strict=True):
                score = (2, soft_lcp_len(x, y) / max(len(x), len(y)))
                mode = "stem"
            elif _shares_ending(x, y):
                score = (1, common_suffix_len(x, y) / max(len(x), len(y)))
                mode = "ending"
            else:
                continue
            if best is None or score > best[0]:
                best = (score, x, y, mode)
    return (best[1], best[2], best[3]) if best else None


def levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def check_suffix_region(word, boundary):
    """Phonology of `word[boundary:]` attached to `word[:boundary]`.

    Returns a list of problem strings (empty = well formed). Only the region after `boundary` is
    inspected, because that is where the generator's edit landed; checking the whole word would
    false-positive on roots, which in Turkish may legitimately be disharmonic.
    """
    stem, suffix = word[:boundary], word[boundary:]
    if not suffix:
        return []
    problems = []

    prev_char = stem[-1] if stem else ""
    # consonant assimilation: a suffix-initial d/c/g after a voiceless consonant must devoice
    if suffix[0] in "dcg" and prev_char in VOICELESS:
        problems.append(f"ünsüz benzeşmesi: '{prev_char}' sonrası '{suffix[0]}' → "
                        f"'{ {'d': 't', 'c': 'ç', 'g': 'k'}[suffix[0]] }' olmalı ({word})")
    # vowel hiatus: Turkish inserts a buffer consonant instead of allowing V+V at a boundary
    if suffix[0] in VOWELS and prev_char in VOWELS:
        problems.append(f"ünlü çakışması: kaynaştırma ünsüzü eksik ({word})")

    # vowel harmony, walking the suffix region and carrying the running vowel forward
    running = _last_vowel(stem)
    is_loan = any(word.lower().startswith(s) for s in LOAN_FRONT_STEMS)
    if running is not None and not is_loan:
        i = 0
        low = suffix.lower()
        while i < len(low):
            hit = next((s for s, v in INVARIANT_SUFFIXES if low.startswith(s, i)), None)
            if hit:
                running = dict(INVARIANT_SUFFIXES)[hit]
                i += len(hit)
                continue
            ch = low[i]
            if ch in VOWELS:
                want, got = _harmony_class(running), _harmony_class(ch)
                if want and got and want != got:
                    problems.append(f"büyük ünlü uyumu: '{running}' sonrası '{ch}' ({word})")
                    break          # one report per word; the rest cascade
                # küçük ünlü uyumu: a HIGH suffix vowel must match the previous vowel's rounding,
                # so `aldım` is well formed and `aldun` is not — a violation that backness alone
                # cannot see, since 'a' and 'u' are both back.
                if ch in HIGH and (running in ROUNDED) != (ch in ROUNDED):
                    problems.append(f"küçük ünlü uyumu (düzlük-yuvarlaklık): '{running}' sonrası "
                                    f"'{ch}' ({word})")
                    break
                running = ch
            i += 1
    return problems


def check_critical_pair(w_pos, w_cf):
    """The core morphology gate: are these two forms the same stem differing by a suffix?"""
    problems = []
    a, b = (w_pos or "").strip().lower(), (w_cf or "").strip().lower()
    if not a or not b:
        return ["kritik sözcük boş"]
    if a == b:
        return ["kritik sözcükler aynı: karşıtlık yok"]
    if " " in a or " " in b:
        problems.append("kritik sözcük tek sözcük değil")

    # Two different boundaries, deliberately:
    #
    #   `shared` (softening-tolerant) answers "is this the same root?" — it must see through
    #      git-/gid- and kitap/kitab- so a real suffix contrast is not rejected as a word swap.
    #   `edit` (raw) answers "where did the edit land?" — it must NOT see through those
    #      alternations, because collapsing d/t is exactly what would hide an assimilation error
    #      like kitapda: under the tolerant boundary the two words look identical and no suffix
    #      region is left to inspect.
    #
    # Thresholds calibrated against v1.3.1: Turkish verb roots are often two letters ("aç-",
    # "ye-"), so requiring a 3-character shared prefix rejects legitimate pairs like
    # açıp/açtırdım. `gel-` / `git-` still fails, which is correct — those are different verbs.
    shared = soft_lcp_len(a, b)
    if shared < 2 or shared < 0.3 * min(len(a), len(b)):
        problems.append(f"ortak kök yok ('{a}' / '{b}', ortak ön ek {shared} harf): "
                        f"bu bir ek karşıtlığı değil, sözcük değişimi")
        return problems                      # downstream checks are meaningless without a stem
    if max(len(a), len(b)) - shared > 12:
        problems.append(f"fark ek bölgesi için fazla uzun ('{a[shared:]}' / '{b[shared:]}')")

    edit = lcp_len(a, b)
    problems += check_suffix_region(a, edit)
    problems += check_suffix_region(b, edit)
    return problems


# --------------------------------------------------------------------------- similarity
def char_ngrams(text, n=3):
    s = " " + re.sub(r"\s+", " ", text.lower().strip()) + " "
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def text_sim(a, b):
    return jaccard(char_ngrams(a), char_ngrams(b))


def f5(word):
    """First-5-character truncation. Can et al. (2008) found this statistically indistinguishable
    from a full lemmatiser for Turkish ad hoc retrieval, which makes it a defensible zero-dependency
    stemmer and — as a BM25 indexing unit — the cheapest possible lexical-shortcut detector."""
    return word[:5]


def content_words(text):
    # len > 1, not len > 2: Turkish has plenty of two-letter content nouns (et, ev, su, iş, yıl)
    # and dropping them makes `Ben et yemem` -> `ete elimi sürmem` look like total content drift.
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1 and not t.isdigit()]


def content_stems(text):
    return {f5(t) for t in content_words(text)}


def content_overlap(query, positive, k=4):
    """Fraction of the query's content words that survive into the positive.

    Bidirectional prefix matching rather than exact stem equality: `kurstan` and `kursa` are the
    same content word under different case suffixes, and any fixed-length truncation splits them
    (f5 gives `kurst` vs `kursa`). Since this gate exists to catch content-word *substitution*,
    matching on a 4-character prefix in either direction is the behaviour we actually want.
    """
    q = content_words(query)
    if not q:
        return 1.0
    p = tokens(positive)
    hit = sum(1 for a in q if any(b.startswith(a[:k]) or a.startswith(b[:k]) for b in p))
    return hit / len(q)


# --------------------------------------------------------------------------- gates
def _by_role(item):
    pos = [c for c in item["candidates"] if c["role"] == "positive"]
    hard = [c for c in item["candidates"] if c["role"] == "hard_negative"]
    easy = [c for c in item["candidates"] if c["role"] == "easy_negative"]
    return pos, hard, easy


def check_structure(item, n_candidates=11, required_subtypes=()):
    p = []
    cands = item.get("candidates") or []
    if len(cands) != n_candidates:
        p.append(f"aday sayısı {len(cands)}, beklenen {n_candidates}")
    ids = [c.get("id") for c in cands]
    if len(set(ids)) != len(ids):
        p.append("aday id'leri benzersiz değil")
    pos, hard, easy = _by_role(item)
    if len(pos) != 1:
        p.append(f"positive sayısı {len(pos)}, tam olarak 1 olmalı")
    if item.get("gold_id") and pos and item["gold_id"] != pos[0]["id"]:
        p.append("gold_id positive adayı göstermiyor")
    if not hard:
        p.append("hard_negative yok")
    subs = {c.get("subtype") for c in hard}
    for want in required_subtypes:
        if want not in subs:
            p.append(f"eksik subtype: {want}")
    n_cf = sum(1 for c in hard if c.get("subtype") == "morph_counterfactual")
    if n_cf != 1:
        p.append(f"morph_counterfactual sayısı {n_cf}, tam olarak 1 olmalı")
    for c in cands:
        if not (c.get("text") or "").strip():
            p.append(f"boş metin: {c.get('id')}")
    # Two candidates with the same text make the item unanswerable regardless of which is gold.
    # Nothing else catches it: the similarity gates saturate near 1.0 well before they hit exact
    # equality, and the judge sees the duplicate as two separately plausible passages.
    texts = [(c.get("text") or "").strip().lower() for c in cands]
    dupes = {t for t in texts if texts.count(t) > 1 and t}
    if dupes:
        p.append(f"{len(dupes)} aday metni birden fazla kez geçiyor")
    if not (item.get("query") or "").strip():
        p.append("sorgu boş")
    return p


def resolve_critical_pair(item):
    """Pick the word pair the morphology gate should inspect. Returns (a, b, mode, source).

    Three attempts, in decreasing order of how much they tell us:

    1. The generator's reported pair, but only if `critical_word_query` really occurs in the query
       and `critical_word_counterfactual` in the counterfactual. That is the *target* contrast.
    2. Derived from positive vs counterfactual (v1.3.1: recoverable for 45/50 items).
    3. Derived from query vs counterfactual (v1.3.1: 44/50). Since v2.1 builds the counterfactual
       as a minimal edit of the query and the positive as an independent re-telling, this is the
       anchor most likely to survive when attempt 2 comes up empty.
    """
    pos, hard, _ = _by_role(item)
    cf = next((c for c in hard if c.get("subtype") == "morph_counterfactual"), None)
    if not (pos and cf):
        return None
    query = item["query"]
    # Prefer cores over assembled texts: the shared frame is identical in both, so it contributes
    # nothing but noise to the diff that finds the contrasting pair.
    pos_text = pos[0].get("core") or pos[0]["text"]
    cf_text = cf.get("core") or cf["text"]

    w_q = (item.get("critical_word_query") or "").strip().lower()
    w_cf = (item.get("critical_word_counterfactual") or "").strip().lower()
    # The reported pair is a HINT, accepted only when it checks out against the actual texts and
    # is genuinely a single-word suffix contrast. It is never a reason to reject: the model
    # routinely reports a phrase where a word was asked for, and rejecting on that discarded 118
    # otherwise-sound items in v2.1 even though derivation recovers the pair from the texts.
    if (w_q and w_cf and w_q != w_cf and " " not in w_q and " " not in w_cf
            and w_q in query.lower() and w_cf in cf_text.lower()
            and (_shares_stem(w_q, w_cf) or _shares_ending(w_q, w_cf))):
        mode = "stem" if _shares_stem(w_q, w_cf) else "ending"
        return w_q, w_cf, mode, "reported"

    for base, src in ((pos_text, "derived"), (query, "derived_from_query")):
        derived = derive_critical_pair(base, cf_text)
        if derived:
            return derived[0], derived[1], derived[2], src
    return None


def check_morphology(item):
    """Phonology of the contrast between the positive and its counterfactual.

    Tier-conditional by design. A `minimal` item is definitionally a surface-minimal pair, so
    failing to find one is a defect. At `standard` and `hard` the counterfactual is allowed to
    re-realise the meaning with different words — 4 of v1.3.1's 50 human-approved items do exactly
    that (`tamamlamayı başardım` / `bitiremedim`) — so absence of a surface pair is not evidence of
    anything and must not be treated as a failure.
    """
    resolved = resolve_critical_pair(item)
    if resolved is None:
        if item.get("tier") == "minimal":
            return ["minimal tier: pozitif ile counterfactual arasında ek karşıtlığı taşıyan "
                    "sözcük çifti bulunamadı"]
        return []
    a, b, mode, _src = resolved
    if mode == "ending":
        return []          # differing region is a root; disharmony there is legal Turkish
    return check_critical_pair(a, b)


# Calibrated on v1.3.1's 50 hand-QC'd items so that the gates accept human-approved data. The
# measured reference distributions are in the module docstring of morph_selftest.py; the headline
# numbers are: the positive is char-3gram top-1 in only 12% of v1.3.1 items and, when it is, the
# top1-top2 margin never exceeds 0.055; minimal-tier sim(positive, counterfactual) runs 0.74-0.92
# for sound items; query->positive content overlap has a 5th percentile of 0.20.
MIN_CONTENT_OVERLAP = 0.20      # below this the positive has drifted off the query's content words
MAX_SHORTCUT_MARGIN = 0.15      # v1.3.1 max is 0.055, so this only fires on genuinely degenerate items
MAX_VERBATIM_RUN = 5            # v1.3.1 max is 4 tokens; see verbatim_run()
MINIMAL_SIM_FLOOR = 0.55        # v1.3.1 sound minimal items: 0.741 .. 0.919
MINIMAL_MAX_TOKEN_DIFF = 6      # v1.3.1 sound minimal items: 2 .. 5
# Compared against the MEDIAN of the other candidates, not the longest of them. Against the
# runner-up the gate almost never fired (one near-length candidate is enough to satisfy it) while
# the positive still beat the typical candidate every time, leaving a 45% blind-longest rate.
# Measured cost at 1.25: v2.0 10%, v2.1 27%, v1.3.1 84% — the human set fails it badly, which is
# consistent with its own 50% blind-longest rate and is the artifact this is meant to beat.
LONGEST_MEDIAN_RATIO = 1.25


def verbatim_run(query, positive):
    """Longest contiguous run of query tokens reproduced verbatim in the positive.

    `positive_rule` demands an *independent paraphrase* that keeps the content words — not a copy
    of the query with padding bolted on. The distinction is measurable and the two populations are
    far apart: across v1.3.1's 50 human-QC'd items the longest run has a median of 1 token and a
    maximum of 4, whereas an unconstrained LLM reproduces the entire query (median ratio 1.00),
    which turns part of the retrieval task back into string matching.
    """
    q, p = tokens(query), tokens(positive)
    best = 0
    for i in range(len(q)):
        for j in range(len(p)):
            k = 0
            while i + k < len(q) and j + k < len(p) and q[i + k] == p[j + k]:
                k += 1
            best = max(best, k)
    return best


def check_lexical(item):
    p = []
    pos, hard, easy = _by_role(item)
    if not pos:
        return ["positive yok"]
    query, positive = item["query"], pos[0]["text"]

    # positive_rule: the positive must re-realise the morphology while KEEPING the content words
    overlap = content_overlap(query, positive)
    if overlap < MIN_CONTENT_OVERLAP:
        p.append(f"pozitif sorgunun içerik sözcüklerini korumuyor (örtüşme {overlap:.2f} "
                 f"< {MIN_CONTENT_OVERLAP})")

    run = verbatim_run(query, positive)
    n_q = len(tokens(query))
    if run >= MAX_VERBATIM_RUN and n_q and run / n_q >= 0.7:
        p.append(f"pozitif sorguyu yeniden ifade etmiyor, birebir kopyalıyor "
                 f"({run}/{n_q} sözcüklük dizi aynen tekrar ediyor)")

    # The overlap-ordering rule: the positive must not be the most query-similar candidate.
    #
    # This is the whole design. The hard negatives are minimal edits of the query, so they inherit
    # its surface; the positive is an independent re-telling and should not. When that ordering
    # holds, lexical similarity points at a WRONG candidate and only a model that reads the suffix
    # can win. When it inverts, a bag of character trigrams solves the item.
    #
    # Restricted to standard/hard, and calibrated: v1.3.1 violates it in 3/44 standard+hard items
    # (7%) but in 3/6 minimal ones (50%) — at minimal tier the positive and counterfactual are
    # near-identical strings by definition, so which of them edges ahead on query overlap is close
    # to a coin flip and cannot be designed away. v2.0 (before the prompt fix) violated it in
    # 101/345 standard+hard items, which is exactly what this gate is here to stop.
    if item.get("tier") in ("standard", "hard"):
        ranked = sorted(((text_sim(query, c["text"]), c["id"]) for c in item["candidates"]),
                        reverse=True)
        if ranked and ranked[0][1] == pos[0]["id"]:
            p.append(f"pozitif sorguya en çok benzeyen aday ({ranked[0][0]:.3f}); sert olumsuzlar "
                     f"sorgunun yüzeyini devralmalı, pozitif devralmamalı — bu hâliyle öğe sözcük "
                     f"eşleştirmesiyle çözülebilir")

    # Length artifact, the other half of "solvable without reading the query".
    #
    # Fixing lexical overlap created this one: once the hard negatives became short edits of the
    # query while the positive stayed a full re-telling, the positive became the longest candidate
    # in 61% of items (v2.0: 19%, chance: 9%), so "pick the longest" beats the intended task. Only
    # a CLEAR win is blocked — being longest by a few characters tells a blind model nothing, and
    # gating on the bare argmax would reject 61% of otherwise sound items. At a 20% margin the cost
    # is 14% on v1.3.1 and 1% on v2.0, so this targets the regression rather than length itself.
    # Measured on the ASSEMBLED text, not the core. The blind baseline this gate exists to defeat
    # reads whole candidates, so the shared frame's dilution of the difference is real and should
    # count in the gate's favour — measuring cores instead made the gate stricter than the artifact
    # warrants and rejected sound items whose assembled lengths were fine.
    others = [len(c["text"]) for c in item["candidates"] if c["id"] != pos[0]["id"]]
    if others:
        med = statistics.median(others)
        own = len(positive)
        if med and own > med * LONGEST_MEDIAN_RATIO:
            p.append(f"pozitif diğer adaylardan sistematik olarak uzun ({own} vs "
                     f"medyan {med:.0f} karakter); sorguyu hiç okumayan bir model onu "
                     f"uzunluğundan seçebilir")

    # Degenerate item: one an f5/char-ngram bag-of-words scorer already solves, i.e. it does not
    # require morphology at all. Note we do NOT require the counterfactual to out-score the
    # positive on query similarity — in v1.3.1 the positive is an independent paraphrase that keeps
    # the query's content words, so it legitimately scores high; what matters is the MARGIN.
    scored = sorted(((text_sim(query, c["text"]), c["id"]) for c in item["candidates"]),
                    reverse=True)
    if len(scored) > 1 and scored[0][1] == pos[0]["id"] \
            and scored[0][0] - scored[1][0] > MAX_SHORTCUT_MARGIN:
        p.append(f"yüzeysel olarak çözülebilir: pozitif char-3gram sıralamasında "
                 f"{scored[0][0]:.3f} vs {scored[1][0]:.3f} ile açık ara önde")
    return p


def check_tier(item):
    """Minimal tier: positive and counterfactual must be near-identical strings.

    Deliberately NOT an exact one-token rule. A single morphological feature can surface on more
    than one word (1sg -> 1pl changes both `birikimimle` and `aldım`) and voice alternation changes
    token count outright (`bizim tarafımızdan arandı` / `bizi aradı`), both of which are sound
    minimal pairs in v1.3.1. Similarity plus a bounded token difference captures "same sentence,
    one feature flipped" without over-fitting to token arithmetic.
    """
    p = []
    if item.get("tier") != "minimal":
        return p
    pos, hard, _ = _by_role(item)
    cf = next((c for c in hard if c.get("subtype") == "morph_counterfactual"), None)
    if not (pos and cf):
        return p
    sim = text_sim(pos[0]["text"], cf["text"])
    if sim < MINIMAL_SIM_FLOOR:
        p.append(f"minimal tier: pozitif ile counterfactual yeterince benzer değil "
                 f"({sim:.3f} < {MINIMAL_SIM_FLOOR}); bu minimal bir çift değil")
    a, b = set(tokens(pos[0]["text"])), set(tokens(cf["text"]))
    if len(a ^ b) > MINIMAL_MAX_TOKEN_DIFF:
        p.append(f"minimal tier: {len(a ^ b)} sözcük farklı (üst sınır "
                 f"{MINIMAL_MAX_TOKEN_DIFF}): {sorted(a ^ b)[:6]}")
    return p


_MD_RE = re.compile(r"(\*\*|__|```|^\s*[-*]\s|\|\s*---)", re.M)
_PLACEHOLDER_RE = re.compile(r"[\[<]\s*(isim|ad|kişi|nesne|yer|x|placeholder|name)\s*[\]>]", re.I)
_ENGLISH_RE = re.compile(r"\b(the|and|with|that|this|which|from|have|been|would|should)\b", re.I)


def check_text_sanity(item):
    p = []
    for c in [{"id": "query", "text": item["query"]}] + list(item["candidates"]):
        t = c["text"]
        if _MD_RE.search(t):
            p.append(f"{c['id']}: markdown biçimlendirme içeriyor")
        if _PLACEHOLDER_RE.search(t):
            p.append(f"{c['id']}: doldurulmamış yer tutucu içeriyor")
        if len(_ENGLISH_RE.findall(t)) >= 2:
            p.append(f"{c['id']}: İngilizce sızıntısı")
        if unicodedata.normalize("NFC", t) != t:
            p.append(f"{c['id']}: Unicode NFC normalize değil")
    return p


def validate_item(item, n_candidates=11, required_subtypes=()):
    """All zero-API gates. Returns {gate: [problems]} containing only gates that found something."""
    out = {}
    for name, fn in (("structure", lambda: check_structure(item, n_candidates, required_subtypes)),
                     ("text", lambda: check_text_sanity(item)),
                     ("morphology", lambda: check_morphology(item)),
                     ("lexical", lambda: check_lexical(item)),
                     ("tier", lambda: check_tier(item))):
        try:
            probs = fn()
        except Exception as e:                      # a malformed item must not kill the run
            probs = [f"{name} denetimi çalışamadı: {type(e).__name__}: {e}"]
        if probs:
            out[name] = probs
    return out


# --------------------------------------------------------------------------- corpus-level gates
def find_near_duplicates(items, threshold=0.80):
    """Query-level near duplicates within the generated pool."""
    grams = [(it["query_id"], char_ngrams(it["query"])) for it in items]
    dups = []
    for i in range(len(grams)):
        for j in range(i + 1, len(grams)):
            s = jaccard(grams[i][1], grams[j][1])
            if s > threshold:
                dups.append((grams[i][0], grams[j][0], round(s, 3)))
    return dups


def load_v1_queries():
    data = json.loads(V1_PATH.read_text(encoding="utf-8"))
    return [(it["query_id"], it["query"]) for it in data["items"]]


def check_test_leakage(items, threshold=0.60):
    """No generated query may be a near-copy of a held-out v1.3.1 test query."""
    v1 = [(qid, char_ngrams(q)) for qid, q in load_v1_queries()]
    hits = []
    for it in items:
        g = char_ngrams(it["query"])
        for qid, vg in v1:
            s = jaccard(g, vg)
            if s > threshold:
                hits.append((it["query_id"], qid, round(s, 3)))
    return hits


def sparse_pairwise_by_type(items):
    """Per-negative-type pairwise accuracy of a char-3gram scorer. Random = 50%.

    The single most diagnostic number about a morphological contrast set, and the reason it is
    computed here rather than left to the eval script: it needs no model, so there is no excuse
    for shipping a dataset without it.

    How to read it. A type near 50% is doing its job — surface overlap carries no signal, so a
    model must use morphology. A type well ABOVE 50% is separable by string matching alone and is
    not a hard negative in any meaningful sense. A type well BELOW 50% is adversarial: lexical
    similarity actively points at the wrong candidate, which is the strongest possible form of the
    negative and what `morph_counterfactual` should look like.
    """
    hits, tot = Counter(), Counter()
    for it in items:
        scores = {c["id"]: text_sim(it["query"], c["text"]) for c in it["candidates"]}
        gold = it.get("gold_id")
        if gold not in scores:
            continue
        for c in it["candidates"]:
            if c["id"] == gold:
                continue
            key = c.get("subtype") or c["role"]
            tot[key] += 1
            hits[key] += int(scores[gold] > scores[c["id"]])
    return {k: round(100 * hits[k] / tot[k], 1) for k in sorted(tot)}


def confound_report(items):
    """Query-blind artifact audit, adapted from SugarCrepe's "blind model" diagnostic.

    If the gold can be picked out without reading the query, the set measures generation artifacts
    rather than morphology. Three blind signals, each with its chance level: a balanced set puts
    the positive at 1/n_candidates on each.

    Interpretation caveat (Feng et al. 2019): a blind baseline that FAILS does not certify the set
    as artifact-free. These numbers are hypothesis tests, not a clean bill of health.
    """
    import statistics as st

    n = len(items) or 1
    longest, decisive = 0, 0
    for it in items:
        lens = [(len(c["text"]), c["id"]) for c in it["candidates"]]
        top = max(lens)
        gold = it.get("gold_id")
        if top[1] != gold:
            continue
        longest += 1
        runner_up = max((l for l, cid in lens if cid != gold), default=0)
        # Winning by a hair is not an exploitable artifact — with candidates built from a shared
        # frame nearly all lengths cluster, so the bare argmax overstates what a blind model could
        # actually use. `decisive` is the number that matters; argmax is kept for continuity with
        # the earlier versions' reported figures.
        if runner_up and top[0] > runner_up * 1.05:
            decisive += 1
    most_tokens = sum(1 for it in items
                      if max(it["candidates"],
                             key=lambda c: len(tokens(c["text"])))["role"] == "positive")
    sparse_top1, margins = 0, []
    for it in items:
        ranked = sorted(((text_sim(it["query"], c["text"]), c["role"]) for c in it["candidates"]),
                        reverse=True)
        if ranked[0][1] == "positive":
            sparse_top1 += 1
            margins.append(ranked[0][0] - ranked[1][0])

    def lens(role):
        return [len(c["text"]) for it in items for c in it["candidates"] if c["role"] == role]

    def sub_lens(sub):
        return [len(c["text"]) for it in items for c in it["candidates"]
                if c.get("subtype") == sub]

    chance = 1.0 / max(len(items[0]["candidates"]), 1) if items else 0.0
    out = {
        "n_items": len(items),
        "chance_level": round(chance, 4),
        "blind_longest_is_gold": round(longest / n, 4),
        "blind_longest_decisive": round(decisive / n, 4),
        "blind_most_tokens_is_gold": round(most_tokens / n, 4),
        "sparse_char3gram_top1_is_gold": round(sparse_top1 / n, 4),
        # Top-1 alone is not interpretable: winning by 0.005 and winning by 0.30 are different
        # artifacts. A high top-1 rate with a near-zero margin means the sparse scorer is
        # essentially guessing among near-ties, which is what a well-built minimal-contrast set
        # should look like.
        "sparse_top1_margin_median": round(st.median(margins), 4) if margins else None,
        "sparse_top1_margin_max": round(max(margins), 4) if margins else None,
        "critical_pair_source": {},
        "mean_len": {},
    }
    src = Counter()
    for it in items:
        prov = (it.get("provenance") or {}).get("critical_pair")
        src[prov["source"] if prov else "none"] += 1
    out["critical_pair_source"] = dict(src)
    out["sparse_pairwise_by_type"] = sparse_pairwise_by_type(items)
    for role in ("positive", "hard_negative", "easy_negative"):
        v = lens(role)
        out["mean_len"][role] = round(st.mean(v), 1) if v else None
    for sub in ("morph_counterfactual", "same_feature_wrong_content",
                "partial_trap", "state_variant"):
        v = sub_lens(sub)
        out["mean_len"][sub] = round(st.mean(v), 1) if v else None
    return out


def max_leakage_similarity(items):
    v1 = [char_ngrams(q) for _, q in load_v1_queries()]
    best = 0.0
    for it in items:
        g = char_ngrams(it["query"])
        best = max(best, max((jaccard(g, vg) for vg in v1), default=0.0))
    return round(best, 3)
