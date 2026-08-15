#!/usr/bin/env python
"""Generate the v2.2 Turkish morphological retrieval train/dev set with Gemini.

    conda run -n dl_hw1 python gen_morph_dataset.py --self-test      # validators, no API calls
    conda run -n dl_hw1 python gen_morph_dataset.py --target 12 -v   # smoke run, ~30 calls
    conda run -n dl_hw1 python gen_morph_dataset.py --target 600     # full run, resumable

v1.3.1 stays the held-out TEST set and is never written to. This script produces train + dev only.

Pipeline
--------
  A generate    1 API call per slot            -> shared frame + one core per candidate,
                                                  assembled into texts here (see assemble_item)
  B validate    0 API calls (morph_validators) -> reject malformed before spending judge quota
  C judge       1 API call per surviving item  -> blind review, no labels shown
  D repair      1 API call per flagged item    -> minimal edit, then back through B and C once
  E audit       0 API calls                    -> corpus-level confounds, near-dups, leakage
  F split       train/dev, stratified, vocabulary-disjoint

Quota
-----
15 RPM / 250k TPM / 500 RPD per key, one counter per key. RPD is the binding constraint, so every call is
counted against a persisted per-key daily ledger and the run checkpoints and exits cleanly when
the pool is exhausted rather than burning retries against a wall. All responses are cached by
slot id, so re-running resumes instead of regenerating — the same contract as
`eval_semantic_encoders.py`'s `results_local/`.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import morph_annotate as A
import morph_beir as B
import morph_validators as V
from morph_prompts import (ALL_EXEMPLAR_IDS, GENERATION_SCHEMA, JUDGE_SCHEMA, REPAIR_SCHEMA,
                           SYSTEM_INSTRUCTION, WHY_NOT_TO_SUBTYPE, build_generation_prompt,
                           build_judge_prompt, build_repair_prompt, load_v1)
from morph_taxonomy import (N_CANDIDATES, PASSAGE_LENGTHS, REQUIRED_SUBTYPES, TARGET_FEATURES,
                            plan_slots)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data_morph_v2"
CACHE_DIR = OUT_DIR / "_cache"
USAGE_PATH = OUT_DIR / "_usage.json"
ENV_PATH = HERE.parent / ".env"

MODEL = "gemini-3.5-flash-lite"
RPM_PER_KEY = 15
RPD_PER_KEY = 500
MIN_INTERVAL = 60.0 / RPM_PER_KEY + 0.2      # 4.2s, a little margin under the 15/min ceiling
VERSION = "2.2"
LENGTH_SPEC = {l["key"]: l for l in PASSAGE_LENGTHS}

# Difficulty ordering for the contrastive curriculum: easy/random negatives first, the minimal
# suffix contrast last. Also used as the graded target for CoSENT/AnglE-style losses.
NEGATIVE_RANK = {"easy_negative": 0, "state_variant": 1, "same_feature_wrong_content": 2,
                 "partial_trap": 3, "morph_counterfactual": 4}
GRADED_SCORE = {"positive": 1.0, "easy_negative": 0.05, "state_variant": 0.55,
                "same_feature_wrong_content": 0.35, "partial_trap": 0.40,
                "morph_counterfactual": 0.10}


# --------------------------------------------------------------------------- API key pool
class BudgetExhausted(Exception):
    pass


class KeyPool:
    """Round-robin over the API keys with a per-key RPM bucket and a persisted daily RPD ledger.

    The ledger is what makes an interrupted run safe to resume: without it a restart would happily
    spend a second 500-request budget against a key that is already at its daily cap, and every
    call would come back 429 after the rate limiter had already slept for it.
    """

    def __init__(self, keys, rpd=RPD_PER_KEY):
        self.keys = keys
        self.rpd = rpd
        self.lock = threading.Lock()
        self.next_ok = {k: 0.0 for k in keys}
        self.cooldown = {k: 0.0 for k in keys}
        self.today = date.today().isoformat()
        self.used = self._load()

    def _load(self):
        """Ledger is keyed by a HASH of the key, never by the key itself or its position.

        Not by the key: this file lives in the repo. Not by position (`API_KEY_1`…): rotating a
        key leaves the name pointing at different credentials, so a fresh key would inherit the
        old one's spent counter and the run would refuse to start against a full 500-request
        budget. Hashing gives each credential its own counter and makes rotation self-resetting.
        """
        if USAGE_PATH.exists():
            data = json.loads(USAGE_PATH.read_text())
            if data.get("date") == self.today:
                used = data.get("used", {})
                return Counter({k: used.get(self._id(k), {}).get("used", 0) for k in self.keys})
        return Counter()

    def _save(self):
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        USAGE_PATH.write_text(json.dumps(
            {"date": self.today,
             "used": {self._id(k): {"name": self._name(k), "used": self.used[k],
                                    "remaining": max(0, self.rpd - self.used[k])}
                      for k in self.keys}},
            indent=1))

    def _id(self, key):
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]

    def _name(self, key):
        return f"API_KEY_{self.keys.index(key) + 1}"

    def remaining(self):
        return sum(max(0, self.rpd - self.used[k]) for k in self.keys)

    def acquire(self):
        """Block until a key is free, then reserve one request against it."""
        while True:
            with self.lock:
                now = time.time()
                if self.remaining() <= 0:
                    raise BudgetExhausted("günlük istek bütçesi tükendi (tüm anahtarlar)")
                ready = [k for k in self.keys
                         if self.used[k] < self.rpd and max(self.next_ok[k], self.cooldown[k]) <= now]
                if ready:
                    k = min(ready, key=lambda x: self.used[x])
                    self.next_ok[k] = now + MIN_INTERVAL
                    self.used[k] += 1
                    self._save()
                    return k
                waits = [max(self.next_ok[k], self.cooldown[k]) - now
                         for k in self.keys if self.used[k] < self.rpd]
                sleep_for = min(waits) if waits else 1.0
            time.sleep(max(0.05, min(sleep_for, 30.0)))

    def penalise(self, key, seconds):
        with self.lock:
            self.cooldown[key] = time.time() + seconds

    def exhaust(self, key):
        """A 429 that is a daily-quota error, not a rate error: retire the key for today."""
        with self.lock:
            self.used[key] = self.rpd
            self._save()

    def report(self):
        return {self._name(k): {"used": self.used[k], "remaining": max(0, self.rpd - self.used[k])}
                for k in self.keys}


def load_keys(preflight=True):
    """Parse API_KEY_N from .env, then verify each one before the run commits work to it.

    Preflight matters because a 401 is permanent, not transient: without it a mistyped key stays
    in the rotation and silently eats one slot per acquisition for the whole run (a bad key in a
    4-key pool cost 7 of 16 slots on the first v2.1 smoke run). One cheap ListModels call per key
    turns that into a startup warning.

    The `AQ.AQ.` repair is a narrow fix for one observed paste artifact — the prefix duplicated on
    copy — and it is applied only when the doubled form is present, announced when it fires, and
    still has to pass preflight like any other key. Nothing is written back to .env.
    """
    if not ENV_PATH.exists():
        sys.exit(f"{ENV_PATH} bulunamadı")
    named = [(m.group(1), m.group(2)) for m in
             (re.match(r'\s*(API_KEY_\d+)\s*=\s*["\']?([^"\'\s]+)', line)
              for line in ENV_PATH.read_text().splitlines()) if m]
    if not named:
        sys.exit(f"{ENV_PATH} içinde API_KEY_N bulunamadı")

    repaired = []
    for i, (name, key) in enumerate(named):
        if key.startswith("AQ.AQ."):
            named[i] = (name, key[3:])
            repaired.append(name)
    if repaired:
        print(f"  [uyarı] {', '.join(repaired)}: 'AQ.' öneki iki kez yapıştırılmış, "
              f"kopya önek kaldırıldı (.env dosyası değiştirilmedi — orada da düzeltin)")

    seen, keys = set(), []
    for name, key in named:
        if key in seen:
            print(f"  [uyarı] {name} başka bir anahtarla aynı, yok sayıldı")
            continue
        seen.add(key)
        keys.append((name, key))

    if not preflight:
        return [k for _, k in keys]

    import requests
    live = []
    for name, key in keys:
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                             params={"key": key, "pageSize": 1}, timeout=20)
            if r.status_code == 200:
                live.append(key)
            else:
                print(f"  [uyarı] {name} geçersiz (HTTP {r.status_code}), havuzdan çıkarıldı")
        except Exception as e:                                   # network hiccup: keep the key
            print(f"  [uyarı] {name} doğrulanamadı ({type(e).__name__}), yine de kullanılacak")
            live.append(key)
    if not live:
        sys.exit("Hiçbir API anahtarı doğrulanamadı.")
    return live


# --------------------------------------------------------------------------- Gemini call
_clients = {}
_clients_lock = threading.Lock()


def _client(key):
    with _clients_lock:
        if key not in _clients:
            from google import genai
            _clients[key] = genai.Client(api_key=key)
        return _clients[key]


def call_gemini(pool, prompt, schema, temperature, max_attempts=4, verbose=False):
    """One structured-output call, with key rotation on quota errors and backoff on transient ones."""
    from google.genai import types

    last = None
    for attempt in range(max_attempts):
        key = pool.acquire()
        try:
            resp = _client(key).models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            text = (resp.text or "").strip()
            if not text:
                raise ValueError(f"boş yanıt (finish_reason={getattr(resp, 'candidates', None)})")
            return json.loads(text)
        except BudgetExhausted:
            raise
        except Exception as e:                                  # noqa: BLE001 - classify by message
            last = e
            msg = str(e)
            transient = any(s in msg for s in ("503", "500", "502", "UNAVAILABLE", "INTERNAL",
                                               "DEADLINE", "timeout", "Timeout"))
            quota = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "quota"))
            if quota:
                # per-minute vs per-day is not always distinguishable from the message; treat a
                # repeated 429 on the same key as a daily cap and retire it rather than looping
                if attempt >= 1:
                    pool.exhaust(key)
                else:
                    pool.penalise(key, 65)
            elif transient:
                time.sleep(2 ** attempt)
            else:
                if isinstance(e, json.JSONDecodeError):
                    time.sleep(1)
                else:
                    raise
            if verbose:
                print(f"    [retry {attempt + 1}/{max_attempts - 1}] {type(e).__name__}: {msg[:120]}")
    raise RuntimeError(f"{max_attempts} denemede başarısız: {last}")


_BAN_LINE_RE = re.compile(r"^KULLANMA \(.*$", re.M)


def cache_key(prompt):
    """Hash of the prompt with the rolling ban-list line removed.

    The prompt hash is part of the cache key on purpose: keying on the slot alone means editing a
    prompt silently keeps serving responses generated by the previous one, so a prompt fix looks
    like it did nothing on a resumed run.

    But the ban list must be excluded, because it is a function of which items were accepted
    *earlier in the same run*. Including it makes every cache key depend on the entire preceding
    history, so a single differently-classified item shifts the hash of every slot after it and a
    resumed run silently falls off its cache — which is exactly what happened on the first full
    run: replaying it re-derived only 216 of 718 slots before missing.
    """
    return hashlib.sha1(_BAN_LINE_RE.sub("", prompt).encode("utf-8")).hexdigest()[:8]


ADOPT_LEGACY_CACHE = False      # set by --adopt-legacy-cache


def cached_call(pool, stage, slot_id, prompt, schema, temperature, force=False, verbose=False):
    """Cache-and-resume wrapper: one JSON file per (stage, slot, prompt)."""
    rev = cache_key(prompt)
    path = CACHE_DIR / f"{stage}_{slot_id}_{rev}.json"
    if not path.exists() and ADOPT_LEGACY_CACHE and not force:
        # Migration path for responses cached under the old history-dependent key. Adopts only
        # when exactly one response exists for this (stage, slot), so there is nothing to choose
        # between. Off by default: with the ban list excluded from the key, a mismatch now means a
        # genuine prompt change, and silently reusing the old answer would hide it.
        legacy = [f for f in CACHE_DIR.glob(f"{stage}_{slot_id}_*.json")]
        if len(legacy) == 1:
            legacy[0].rename(path)
    if path.exists() and not force:
        try:
            return json.loads(path.read_text(encoding="utf-8")), True
        except json.JSONDecodeError:
            path.unlink()
    data = call_gemini(pool, prompt, schema, temperature, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data, False


# --------------------------------------------------------------------------- item assembly
def assemble_item(slot, raw):
    """Turn a raw generation response into a v1.3.1-shaped item.

    Ids are assigned here rather than asked for, because an id the model invents is one more thing
    that can collide or go missing, and nothing downstream needs the model's choice.
    """
    qid = f"{slot['slot_id']}"
    # The shared frame is concatenated HERE, not by the model. That is the whole point: asking a
    # model to reproduce the same two sentences in 11 candidates is exactly the instruction it
    # drifts on, and that drift is what made the positive systematically ~8% longer than the median
    # candidate (blind "pick the longest" scored 32% against a 9% chance rate). Concatenating in
    # Python makes the frame byte-identical by construction, so the only length difference left is
    # core length, which the validators can check directly.
    frame_before = (raw.get("frame_before") or "").strip()
    frame_after = (raw.get("frame_after") or "").strip()

    cands = []
    counters = Counter()
    for c in raw.get("candidates", []):
        role = c.get("role", "")
        counters[role] += 1
        prefix = {"positive": "p", "hard_negative": "h", "easy_negative": "e"}.get(role, "x")
        sub = c.get("subtype")
        core = (c.get("core") or "").strip()
        cands.append({
            "id": f"{qid}_{prefix}{counters[role]}",
            "role": role,
            # "none" is the schema's sentinel for "no subtype" — Gemini rejects "" as an enum member
            **({"subtype": sub} if sub and sub != "none" else {}),
            "text": " ".join(p for p in (frame_before, core, frame_after) if p),
            "core": core,
            "note": (c.get("note") or "").strip(),
            **({"violated_requirement": c["violated_requirement"]}
               if c.get("violated_requirement") else {}),
        })
    gold = next((c["id"] for c in cands if c["role"] == "positive"), None)
    feature = next((f for f in TARGET_FEATURES if f["key"] == slot["target_feature"]), {})
    item = {
        "query_id": qid,
        "target_feature": slot["target_feature"],
        "phenomenon": slot["phenomenon"],
        "layer": slot["layer"],
        "tier": slot["tier"],
        "ek_turu": slot["ek_turu"],
        "contrast": slot["contrast"],
        "domain": slot["domain"],
        "passage_length": slot["passage_length"],
        "person": slot["person"],
        "query": (raw.get("query") or "").strip(),
        "query_form": "doğal_ifade",
        "gold_id": gold,
        "frame_before": frame_before,
        "frame_after": frame_after,
        "critical_word_query": (raw.get("critical_word_query") or "").strip(),
        "critical_word_counterfactual": (raw.get("critical_word_counterfactual") or "").strip(),
        "requirements": raw.get("requirements", []),
        "candidates": cands,
        "provenance": {
            "generator": MODEL,
            "prompt_version": VERSION,
            "slot_index": slot["index"],
            "self_check": raw.get("self_check", {}),
        },
    }
    if feature.get("layer") == "chain":
        item["chain"] = [p.strip() for p in re.split(r"[+/]", slot["ek_turu"]) if p.strip()]
    return item


def apply_repair(item, fixed):
    """Overlay a repair response onto the item, keeping ids, roles and the shared frame stable."""
    by_id = {c["id"]: c for c in item["candidates"]}
    changed = []
    # The repair may rewrite the frame, but it must stay shared — so it is applied to every
    # candidate here rather than trusted per-candidate.
    frame_before = (fixed.get("frame_before") or item.get("frame_before") or "").strip()
    frame_after = (fixed.get("frame_after") or item.get("frame_after") or "").strip()
    item["frame_before"], item["frame_after"] = frame_before, frame_after

    for c in fixed.get("candidates", []):
        tgt = by_id.get(c.get("id"))
        if not tgt:
            continue
        new_core = (c.get("core") or "").strip()
        if new_core and new_core != tgt.get("core"):
            changed.append(tgt["id"])
            tgt["core"] = new_core
        if c.get("subtype") is not None and c.get("role") == tgt["role"]:
            if c["subtype"] and c["subtype"] != "none":
                tgt["subtype"] = c["subtype"]
            else:
                tgt.pop("subtype", None)
        if c.get("note"):
            tgt["note"] = c["note"]
    # Rebuild every text from the (possibly new) shared frame, so a repair can never desynchronise
    # the frame across candidates.
    for c in item["candidates"]:
        c["text"] = " ".join(p for p in (frame_before, c.get("core", ""), frame_after) if p)

    if (fixed.get("query") or "").strip():
        item["query"] = fixed["query"].strip()
    for k in ("critical_word_query", "critical_word_counterfactual"):
        if (fixed.get(k) or "").strip():
            item[k] = fixed[k].strip()
    item.setdefault("provenance", {})["repair"] = {
        "changes": fixed.get("changes", []), "changed_ids": changed}
    return item


# --------------------------------------------------------------------------- judge interpretation
def interpret_judge(item, verdict):
    """Compare the judge's blind, independent read against what the generator intended.

    The judge never saw the labels, so agreement here is real evidence rather than assent.
    """
    problems = []
    by_id = {c["id"]: c for c in item["candidates"]}
    rows = {r["id"]: r for r in verdict.get("candidates", []) if r.get("id") in by_id}
    if len(rows) < len(by_id) * 0.8:
        problems.append(f"jüri adayların yalnızca {len(rows)}/{len(by_id)} tanesini değerlendirdi")

    answerable = {i for i, r in rows.items() if r.get("answers_query")}
    gold = item["gold_id"]
    extra = answerable - {gold}
    if extra:
        problems.append("çift altın: jüri şu adayları da geçerli cevap saydı: "
                        + ", ".join(f"{i} ({rows[i].get('reason', '')[:70]})" for i in sorted(extra)))
    if gold not in answerable:
        problems.append(f"jüri pozitifi ({gold}) geçerli cevap saymadı: "
                        f"{rows.get(gold, {}).get('reason', '')[:120]}")

    # subtype audit: only the counterfactual is worth blocking on. The literature documents LLM
    # judges over-marking surface-overlapping text as relevant, which is exactly the
    # same_feature_wrong_content / partial_trap categories — so disagreement there is recorded but
    # does not reject the item.
    subtype_mismatches = []
    for cid, c in by_id.items():
        want = c.get("subtype") or ("easy_negative" if c["role"] == "easy_negative" else None)
        got = WHY_NOT_TO_SUBTYPE.get(rows.get(cid, {}).get("why_not", ""))
        if want and got and want != got:
            subtype_mismatches.append({"id": cid, "intended": want, "judge": got})
    # On the counterfactual specifically, only a judgement of `unrelated` blocks. Telling
    # `morph_counterfactual` from `same_feature_wrong_content` is a fine-grained call that one
    # sample from the generator's own family is weak evidence on — and it is precisely the
    # surface-overlapping category where LLM judges are documented to misfire. `unrelated` is
    # different in kind: it says the candidate is not a hard negative at all.
    cf = next((c["id"] for c in item["candidates"]
               if c.get("subtype") == "morph_counterfactual"), None)
    if cf and rows.get(cf, {}).get("why_not") == "unrelated":
        problems.append(f"morph_counterfactual jüriye göre sorguyla ilgisiz: "
                        f"{rows[cf].get('reason', '')[:110]}")

    if verdict.get("query_is_assertion") is False:
        problems.append("sorgu bir durumu iddia etmiyor (soru biçiminde)")
    if verdict.get("morphology_errors"):
        problems.append("jüri Türkçe biçimbilim hatası bildirdi: "
                        + ", ".join(verdict["morphology_errors"][:4]))
    outliers = [o for o in verdict.get("fluency_outliers", []) if o in by_id]
    if gold in outliers:
        problems.append(f"pozitif üslup/uzunluk bakımından aykırı ({gold}) — kör model onu "
                        f"sorguyu okumadan seçebilir")
    return problems, {"subtype_mismatches": subtype_mismatches,
                      "fluency_outliers": outliers,
                      "judge_notes": verdict.get("notes", "")}


# --------------------------------------------------------------------------- splitting
def split_train_dev(items, dev_frac=0.15, seed=7):
    """Stratified by tier x layer x passage_length, then made vocabulary-disjoint.

    Deliberately NOT stratified on phenomenon as well. With ~40 phenomena the four-way key
    produces mostly singleton strata, every one of which is too small to yield a dev item, and the
    split silently collapses (it produced 23 dev items out of 475 before this was coarsened). The
    three-way key gives 18 strata that are each large enough to sample from, and feature coverage
    in dev is then a reporting concern rather than a constraint.

    Vocabulary disjointness matters more than the exact ratio: without it a dev item can be a
    paraphrase of a train item over the same entities, and dev stops being a held-out measurement.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for it in items:
        strata[(it["tier"], it["layer"], it["passage_length"])].append(it)

    dev, train = [], []
    for _, group in sorted(strata.items()):
        rng.shuffle(group)
        n_dev = max(1, round(len(group) * dev_frac)) if len(group) >= 4 else 0
        dev += group[:n_dev]
        train += group[n_dev:]

    # Send a dev item back to train only if some train query is a near-paraphrase of it.
    #
    # An earlier version demanded that 25% of a dev query's content words be absent from train
    # entirely. That is the wrong target: with 10 shared domains and a shared entity pool, almost
    # no query clears it, and dev collapsed to 27 of 475 items. What actually breaks a held-out
    # split is a dev query that restates a train query, and similarity measures that directly.
    kept, moved = [], 0
    train_grams = [V.char_ngrams(it["query"]) for it in train]
    for it in dev:
        g = V.char_ngrams(it["query"])
        if max((V.jaccard(g, t) for t in train_grams), default=0.0) > 0.50:
            train.append(it)
            moved += 1
        else:
            kept.append(it)
    for it in train:
        it["split"] = "train"
    for it in kept:
        it["split"] = "dev"
    return train, kept, moved


# --------------------------------------------------------------------------- writers
def dataset_envelope(items, split, extra_stats=None):
    v1 = load_v1()
    return {
        "dataset_name": "turkish_morph_retrieval_v2",
        "version": VERSION,
        "split": split,
        "language": "tr",
        "note": ("Gemini ile sentezlenmiş, kural tabanlı biçimbilim denetiminden ve kör jüri "
                 "incelemesinden geçmiş EĞİTİM/GELİŞTİRME kümesi. Tutulan test kümesi "
                 "morph_eval_set_v1.3.1'dir ve bu dosyaya dahil değildir."),
        "conventions": v1["conventions"],
        "held_out_test_set": "legacy_test_data/morph_eval_set_v1.3.1_review_reviewer_C_fixed.json",
        "exemplars_shown_to_generator": ALL_EXEMPLAR_IDS,
        "statistics": {
            "n_queries": len(items),
            "n_candidates": sum(len(i["candidates"]) for i in items),
            "by_tier": dict(Counter(i["tier"] for i in items)),
            "by_layer": dict(Counter(i["layer"] for i in items)),
            "by_domain": dict(Counter(i["domain"] for i in items)),
            "by_passage_length": dict(Counter(i["passage_length"] for i in items)),
            "n_target_features": len({i["target_feature"] for i in items}),
            **(extra_stats or {}),
        },
        "items": items,
    }


def write_pair_views(items, path_pairs, path_paired):
    """Two training/eval views of the same data.

    `path_pairs`  MNRL-ready: negatives grouped per query so the trainer can guarantee the
                  counterfactual sits in the same batch as its positive (the NegCLIP condition).
    `path_paired` NevIR-style: one row per (gold, typed negative) so evaluation can report
                  pairwise accuracy per negative type against an explicit 50% baseline, instead
                  of only nDCG@10 over 11 candidates.
    """
    with path_pairs.open("w", encoding="utf-8") as fh:
        for it in items:
            pos = next(c for c in it["candidates"] if c["role"] == "positive")
            negs = [c for c in it["candidates"] if c["role"] != "positive"]
            negs.sort(key=lambda c: NEGATIVE_RANK.get(c.get("subtype") or c["role"], 0))
            fh.write(json.dumps({
                "query_id": it["query_id"],
                "query": it["query"],
                "positive": pos["text"],
                "negatives": [c["text"] for c in negs],
                "negative_types": [c.get("subtype") or c["role"] for c in negs],
                "difficulty_rank": [NEGATIVE_RANK.get(c.get("subtype") or c["role"], 0)
                                    for c in negs],
                "graded_scores": [GRADED_SCORE.get(c.get("subtype") or c["role"], 0.0)
                                  for c in negs],
                "target_feature": it["target_feature"],
                "tier": it["tier"],
                "layer": it["layer"],
                "domain": it["domain"],
                "passage_length": it["passage_length"],
                "curated": True,   # exempt from positive-anchored false-negative filtering
            }, ensure_ascii=False) + "\n")

    with path_paired.open("w", encoding="utf-8") as fh:
        for it in items:
            pos = next(c for c in it["candidates"] if c["role"] == "positive")
            for c in it["candidates"]:
                if c["role"] == "positive":
                    continue
                fh.write(json.dumps({
                    "query_id": it["query_id"],
                    "query": it["query"],
                    "gold": pos["text"],
                    "negative": c["text"],
                    "negative_type": c.get("subtype") or c["role"],
                    "target_feature": it["target_feature"],
                    "tier": it["tier"],
                    "passage_length": it["passage_length"],
                }, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- pipeline
def generate_one(pool, slot, ban_words, args, stats, lock):
    """Stages A-D for a single slot. Returns (item, reject_record) with exactly one non-None."""
    sid = slot["slot_id"]
    try:
        raw, hit = cached_call(pool, "gen", sid,
                               build_generation_prompt(slot, ban_words,
                                                       LENGTH_SPEC[slot["passage_length"]]),
                               GENERATION_SCHEMA, args.gen_temp, verbose=args.verbose)
    except BudgetExhausted:
        return None, {"slot_id": sid, "stage": "budget", "problems": ["bütçe tükendi"]}
    except Exception as e:
        return None, {"slot_id": sid, "stage": "generate", "problems": [f"{type(e).__name__}: {e}"]}
    with lock:
        stats["gen_cache_hits" if hit else "gen_calls"] += 1

    item = assemble_item(slot, raw)
    probs = V.validate_item(item, N_CANDIDATES, REQUIRED_SUBTYPES)

    # Stage D can only fix content; a broken suffix is not a QC issue, it is a broken item.
    if probs and "morphology" not in probs:
        flat = [f"{g}: {p}" for g, ps in probs.items() for p in ps]
        try:
            fixed, hit = cached_call(pool, "repair", sid, build_repair_prompt(item, flat),
                                     REPAIR_SCHEMA, args.judge_temp, verbose=args.verbose)
            with lock:
                stats["repair_cache_hits" if hit else "repair_calls"] += 1
            import copy as _copy
            before = _copy.deepcopy(item)
            item = apply_repair(item, fixed)
            probs = V.validate_item(item, N_CANDIDATES, REQUIRED_SUBTYPES)
            # A repair must not make the item worse. Observed doing exactly that: asked to fix a
            # length imbalance, the model rewrote a sound positive (query similarity 0.405) as the
            # query verbatim (1.000), trading one gate failure for three. Keeping whichever version
            # has fewer distinct failures makes the repair call strictly non-destructive.
            if len(probs) > len(before_probs := V.validate_item(before, N_CANDIDATES,
                                                               REQUIRED_SUBTYPES)):
                item, probs = before, before_probs
                with lock:
                    stats["repair_reverted"] += 1
            with lock:
                stats["repaired_ok" if not probs else "repair_failed"] += 1
        except BudgetExhausted:
            return None, {"slot_id": sid, "stage": "budget", "problems": ["bütçe tükendi"]}
        except Exception as e:
            probs.setdefault("repair", []).append(f"{type(e).__name__}: {e}")
    if probs:
        with lock:
            for gate in probs:
                stats["gate_" + gate] += 1
        return None, {"slot_id": sid, "stage": "validate", "problems":
                      [f"{g}: {p}" for g, ps in probs.items() for p in ps], "item": item}

    # Stage C: blind judge
    try:
        verdict, hit = cached_call(pool, "judge", sid, build_judge_prompt(item, seed=slot["index"]),
                                   JUDGE_SCHEMA, args.judge_temp, verbose=args.verbose)
        with lock:
            stats["judge_cache_hits" if hit else "judge_calls"] += 1
    except BudgetExhausted:
        return None, {"slot_id": sid, "stage": "budget", "problems": ["bütçe tükendi"]}
    except Exception as e:
        return None, {"slot_id": sid, "stage": "judge", "problems": [f"{type(e).__name__}: {e}"],
                      "item": item}

    jprobs, jmeta = interpret_judge(item, verdict)
    item["provenance"]["judge"] = jmeta
    resolved = V.resolve_critical_pair(item)
    if resolved:
        a, b, mode, src = resolved
        item["provenance"]["critical_pair"] = {"positive": a, "counterfactual": b,
                                               "mode": mode, "source": src}
    if jprobs:
        with lock:
            stats["judge_rejected"] += 1
        return None, {"slot_id": sid, "stage": "judge", "problems": jprobs, "item": item}

    with lock:
        stats["accepted"] += 1
    return item, None


def run_generation(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pool = KeyPool(load_keys())
    slots = plan_slots(round(args.target * args.overproduce))

    print(f"model={MODEL}  hedef={args.target} kabul  planlanan={len(slots)} yuva  "
          f"anahtar={len(pool.keys)}  günlük kalan≈{pool.remaining()} istek")
    print(f"çıktı: {OUT_DIR}")

    stats = Counter()
    lock = threading.Lock()
    accepted, rejected = [], []
    # Ban list grows as items are accepted, so later slots are pushed away from vocabulary the
    # model has already leaned on. Seeded with the exemplars' content words, which the generator
    # sees every single call and would otherwise echo.
    ban = set()
    for qid in ALL_EXEMPLAR_IDS:
        it = next(i for i in load_v1()["items"] if i["query_id"] == qid)
        ban |= set(V.content_words(it["query"]))

    t0 = time.time()
    if True:
        n_workers = args.workers or len(pool.keys)
        with ThreadPoolExecutor(max_workers=min(n_workers, len(pool.keys))) as ex:
            for start in range(0, len(slots), args.batch):
                if len(accepted) >= args.target:
                    break
                chunk = slots[start:start + args.batch]
                snapshot = set(ban)
                futures = [ex.submit(generate_one, pool, s, snapshot, args, stats, lock)
                           for s in chunk]
                for fut in futures:
                    item, rej = fut.result()
                    if item:
                        accepted.append(item)
                        ban |= {w for w in V.content_words(item["query"]) if len(w) > 3}
                    else:
                        rejected.append(rej)
                done, tot = len(accepted), len(accepted) + len(rejected)
                print(f"  [{tot:4d}/{len(slots)}] kabul={done:4d} red={len(rejected):4d} "
                      f"({done / max(tot, 1):.0%} verim)  {time.time() - t0:.0f}s  "
                      f"kalan istek≈{pool.remaining()}")
    deferred = [r for r in rejected if r["stage"] == "budget"]
    rejected = [r for r in rejected if r["stage"] != "budget"]
    if deferred:
        print(f"\n  [BÜTÇE] {len(deferred)} yuva bugünkü istek bütçesi bittiği için üretilmedi.")
        print("  Aynı komut yarın çalıştırıldığında önbellekteki her şey korunur ve yalnızca "
              "bu yuvalar üretilir.")

    accepted = accepted[:args.target]
    finalize(accepted, rejected, stats, pool, args, time.time() - t0)


def _write_rejects(rejected):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "rejected.jsonl").open("w", encoding="utf-8") as fh:
        for r in rejected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def finalize(accepted, rejected, stats, pool, args, elapsed):
    # Written first and unconditionally: when yield is 0 the rejects are the only diagnostic there
    # is, and that is exactly the run where you most need them.
    _write_rejects(rejected)
    if not accepted:
        print("\nKabul edilen öğe yok.")
        print(f"Red gerekçeleri: {OUT_DIR / 'rejected.jsonl'}")
        for stage, n in Counter(r["stage"] for r in rejected).most_common():
            print(f"  {stage}: {n}")
        for r in rejected[:3]:
            print(f"  örnek [{r['stage']}] {r['problems'][:2]}")
        return

    # Stage E: corpus-level gates
    dups = V.find_near_duplicates(accepted)
    dup_ids = {b for _, b, _ in dups}
    if dup_ids:
        rejected += [{"slot_id": i, "stage": "near_duplicate", "problems": ["yakın kopya"]}
                     for i in dup_ids]
        accepted = [i for i in accepted if i["query_id"] not in dup_ids]
    leaks = V.check_test_leakage(accepted)
    leak_ids = {a for a, _, _ in leaks}
    if leak_ids:
        rejected += [{"slot_id": i, "stage": "test_leakage", "problems": ["v1.3.1 ile örtüşme"]}
                     for i in leak_ids]
        accepted = [i for i in accepted if i["query_id"] not in leak_ids]

    confound = V.confound_report(accepted)
    train, dev, moved = split_train_dev(accepted, args.dev_frac)

    # Morphological annotation + variant groups, in place, before serialising — so the shipped
    # train/dev JSON already carries `item["morphology"]`/`item["variants"]` rather than needing a
    # separate post-processing pass every time the dataset regenerates.
    train, ann_train = A.annotate_dataset(train, use_zeyrek=True)
    dev, ann_dev = A.annotate_dataset(dev, use_zeyrek=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extra_train = {"confound_audit": confound, "morphology_annotation": ann_train}
    extra_dev = {"confound_audit": confound, "morphology_annotation": ann_dev}
    (OUT_DIR / f"morph_train_v{VERSION}.json").write_text(
        json.dumps(dataset_envelope(train, "train", extra_train), ensure_ascii=False, indent=1),
        encoding="utf-8")
    (OUT_DIR / f"morph_dev_v{VERSION}.json").write_text(
        json.dumps(dataset_envelope(dev, "dev", extra_dev), ensure_ascii=False, indent=1),
        encoding="utf-8")
    write_pair_views(train, OUT_DIR / f"morph_train_pairs_v{VERSION}.jsonl",
                     OUT_DIR / f"morph_train_paired_v{VERSION}.jsonl")
    write_pair_views(dev, OUT_DIR / f"morph_dev_pairs_v{VERSION}.jsonl",
                     OUT_DIR / f"morph_dev_paired_v{VERSION}.jsonl")
    _write_rejects(rejected)      # rewrite: stage E added near-duplicate and leakage rejections

    beir_train = B.export_split(train, "train")
    beir_dev = B.export_split(dev, "dev")

    write_report(accepted, train, dev, rejected, stats, pool, confound, moved, elapsed,
                ann_train, ann_dev, beir_train, beir_dev)
    print(f"\nyazıldı: {OUT_DIR}")
    print(f"  train {len(train)} · dev {len(dev)} · red {len(rejected)}")
    print(f"  biçimbilim eşleşme: train %{ann_train['tier1_coverage']['exact_pct']:.0f} · "
          f"dev %{ann_dev['tier1_coverage']['exact_pct']:.0f}")
    print(f"  BEIR: {beir_train['out_dir']}, {beir_dev['out_dir']}")
    print(f"  rapor: {OUT_DIR / 'generation_report.md'}")


def write_report(accepted, train, dev, rejected, stats, pool, confound, moved, elapsed,
                 ann_train=None, ann_dev=None, beir_train=None, beir_dev=None):
    by_stage = Counter(r["stage"] for r in rejected)
    gate_counts = {k[5:]: v for k, v in stats.items() if k.startswith("gate_")}
    reasons = Counter()
    for r in rejected:
        for p in r["problems"]:
            reasons[re.split(r"[:(]", p)[0].strip()[:70]] += 1
    feat_cov = Counter(i["target_feature"] for i in accepted)
    missing = sorted({f["key"] for f in TARGET_FEATURES} - set(feat_cov))
    calls = stats["gen_calls"] + stats["judge_calls"] + stats["repair_calls"]

    lines = [
        f"# v{VERSION} üretim raporu", "",
        f"- model: `{MODEL}`", f"- süre: {elapsed / 60:.1f} dk",
        f"- API çağrısı: {calls} (üretim {stats['gen_calls']}, jüri {stats['judge_calls']}, "
        f"onarım {stats['repair_calls']}); önbellekten: "
        f"{stats['gen_cache_hits'] + stats['judge_cache_hits'] + stats['repair_cache_hits']}",
        f"- kabul {len(accepted)} · red {len(rejected)} · verim "
        f"{len(accepted) / max(len(accepted) + len(rejected), 1):.1%}",
        f"- train {len(train)} · dev {len(dev)} (sözcük dağarcığı örtüşmesi nedeniyle "
        f"{moved} öğe dev'den train'e taşındı)", "",
        "## Kota kullanımı", "",
        "| anahtar | kullanılan | kalan |", "|---|---|---|",
    ]
    for name, u in pool.report().items():
        lines.append(f"| {name} | {u['used']} | {u['remaining']} |")

    lines += ["", "## Red gerekçeleri (aşama)", "", "| aşama | adet |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in by_stage.most_common()]
    if gate_counts:
        lines += ["", "### Kural tabanlı kapılar", "", "| kapı | tetiklenme |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(gate_counts.items(), key=lambda x: -x[1])]
    lines += ["", "### En sık gerekçeler", "", "| gerekçe | adet |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in reasons.most_common(18)]

    lines += ["", "## Kör (query-blind) artefakt denetimi", "",
              "SugarCrepe'in kör-model tanısının sıralama kümesine uyarlanmışı. Sorguyu hiç "
              "okumadan doğru adayı seçebilen bir ölçüt, veri kümesinin biçimbilimi değil üretim "
              "artefaktını ölçtüğü anlamına gelir. Şans düzeyi "
              f"{confound['chance_level']:.1%}.", "",
              "| ölçüt | değer | şans |", "|---|---|---|"]
    for k in ("blind_longest_is_gold", "blind_longest_decisive", "blind_most_tokens_is_gold",
              "sparse_char3gram_top1_is_gold"):
        lines.append(f"| {k} | {confound[k]:.1%} | {confound['chance_level']:.1%} |")
    lines += ["",
              f"Seyrek temel çizgi pozitifi ilk sıraya koyduğunda aradaki fark: medyan "
              f"**{confound['sparse_top1_margin_median']}**, en yüksek "
              f"{confound['sparse_top1_margin_max']}. Oranın tek başına anlamı yoktur: 0.005 farkla "
              f"kazanmak ile 0.30 farkla kazanmak farklı şeylerdir; medyan farkın sıfıra yakın "
              f"olması, seyrek ölçütün neredeyse berabere kalan adaylar arasında tahmin yürüttüğünü "
              f"gösterir.", "",
              f"Kritik sözcük çiftinin kaynağı: `{confound['critical_pair_source']}` "
              f"(`reported` = üreticinin hedeflediği karşıtlık doğrulandı; `derived` = çift "
              f"metinlerden türetildi, hedef özellikle birebir aynı olmayabilir).", ""]
    sp = confound["sparse_pairwise_by_type"]
    lines += ["", "### Seyrek (char-3gram) temel çizgi — alt tür bazında ikili doğruluk", "",
              "Modelsiz bir sözcük-örtüşmesi ölçütü, altın adayı her bir olumsuz adayın üstüne "
              "koyabiliyor mu? Rastgele = %50.", "",
              f"| alt tür | v{VERSION} | v2.0 (istem düzeltmesi öncesi) | v1.3.1 (test) | okuma |",
              "|---|---|---|---|---|"]
    v1_ref = V.sparse_pairwise_by_type(load_v1()["items"])
    v20_ref = {}
    prev = OUT_DIR / "archive_v2.0" / "morph_train_v2.0.json"
    if prev.exists():
        v20_ref = V.sparse_pairwise_by_type(
            json.loads(prev.read_text(encoding="utf-8"))["items"])
    def verdict(v):
        if v < 45:
            return "**çekişmeli — sözcük örtüşmesi YANLIŞ adayı gösteriyor (en güçlü)**"
        if v <= 60:
            return "rastgeleye yakın: sözcüksel sinyal yok (iyi)"
        if v <= 75:
            return "kısmen sözcüksel olarak ayrılabilir"
        return "**büyük ölçüde sözcüksel olarak çözülebilir (zayıf)**"

    for k in ("morph_counterfactual", "partial_trap", "same_feature_wrong_content",
              "state_variant", "easy_negative"):
        if k not in sp:
            continue
        # easy_negative is SUPPOSED to be trivially separable; that is its job.
        note = "tasarımı gereği kolay" if k == "easy_negative" else verdict(sp[k])
        lines.append(f"| {k} | {sp[k]} | {v20_ref.get(k, '—')} | {v1_ref.get(k, '—')} | {note} |")
    lines += ["",
              "> **Bilinen zayıflık.** v2.0'da `same_feature_wrong_content` seyrek ölçütle "
              "neredeyse tamamen çözülebiliyor, `morph_counterfactual` ise rastgelenin üzerinde. "
              "v1.3.1'de her ikisi de rastgelenin ALTINDA, yani sözcük örtüşmesi yanlış adayı "
              "işaret ediyor. Nedeni: v1.3.1'de pozitif, sorgunun bağımsız bir yeniden ifadesidir; "
              "v2.0'da ise sorgunun içerik sözcüklerini aynı sırada koruyan daha yakın bir "
              "yeniden yazımdır, dolayısıyla yüzey benzerliği doğru adayı ele veriyor. "
              "Bu, v2.0'ın bir EĞİTİM kümesi olarak kullanılabilirliğini ortadan kaldırmaz, ama "
              "tek başına bir ölçüt (benchmark) olarak kullanılmasını engeller — ölçüt v1.3.1'dir.",
              ""]
    lines += ["", "Ortalama karakter uzunlukları:", "", "| rol/alt tür | ort. uzunluk |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in confound["mean_len"].items() if v]
    lines += ["",
              "> Karşılaştırma: v1.3.1'de `blind_longest_is_gold` **%50** (şans %9). Yani insan "
              "denetiminden geçmiş test kümesinde pozitifler sistematik olarak daha uzun. "
              "Bu, v2.0 için düzeltilmesi hedeflenen bilinen bir artefakttır ve üretim isteminde "
              "açıkça yasaklanmıştır.",
              "> Uyarı (Feng vd. 2019): kör ölçütün BAŞARISIZ olması, veri kümesinin artefaktsız "
              "olduğunu KANITLAMAZ. Bu sayılar bir hipotez testidir, temiz kâğıt değil.", ""]

    lines += ["## Kapsam", "",
              f"- hedef biçimbirim özelliği: {len(feat_cov)}/{len(TARGET_FEATURES)}",
              f"- eksik özellikler: {', '.join(missing) if missing else 'yok'}", "",
              "| özellik | adet |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in feat_cov.most_common()]

    if ann_train and ann_dev:
        lines += ["", "## Biçimbilim ek anlamlandırması (morph_annotate.py)", "",
                  "Kural tabanlı Katman 1 (`morph_taxonomy.py`'den ayrıştırılan ek tablosu, sınır "
                  "araması) + isteğe bağlı Katman 2 (`zeyrek`). Her öğeye `morphology`/`variants` "
                  "eklenmiş olarak yazıldı — API çağrısı yok, tamamen yerel.", "",
                  "| ölçüt | train | dev |", "|---|---|---|",
                  f"| tam eşleşme (`exact`) | %{ann_train['tier1_coverage']['exact_pct']:.1f} | "
                  f"%{ann_dev['tier1_coverage']['exact_pct']:.1f} |",
                  f"| çift bulunamadı (`no_pair`) | %{ann_train['tier1_coverage']['no_pair_pct']:.1f} | "
                  f"%{ann_dev['tier1_coverage']['no_pair_pct']:.1f} |",
                  f"| bildirilen özellikle uyum | %{ann_train['tier1_target_feature_agreement']['agree_pct']:.1f} | "
                  f"%{ann_dev['tier1_target_feature_agreement']['agree_pct']:.1f} |",
                  f"| zeyrek ayrıştırma oranı | %{ann_train['zeyrek_parse_rate_pct']:.1f} | "
                  f"%{ann_dev['zeyrek_parse_rate_pct']:.1f} |", "",
                  "`no_pair` öğeleri — kritik sözcük çifti hiçbir şekilde türetilemedi; bu ya "
                  "üretimde gerçek bir kusur (bkz. örnekler) ya da yalnızca çok sözcüklü bir "
                  "karşıtlıktır (tek-sözcük eşleştirici bunu yakalayamaz, bu beklenen bir sınırdır):",
                  ""]
        for ex in ann_train.get("no_pair_examples", [])[:10]:
            lines.append(f"- `{ex['query_id']}` ({ex['target_feature']}): "
                         f"{ex['reported_query']!r} / {ex['reported_counterfactual']!r}")
        lines += ["", "`target_feature` ile uyuşmayan tam eşleşmeler — Katman 1'in bulduğu gerçek "
                  "ek, öğenin kendi etiketiyle örtüşmüyor (etiket gürültüsü denetimi):", ""]
        for ex in ann_train.get("disagreement_examples", [])[:10]:
            lines.append(f"- `{ex['query_id']}`: bildirilen **{ex['declared']}**, bulunan "
                         f"**{ex['found']}** (fark {ex['diff']})")

    if beir_train and beir_dev:
        lines += ["", "## BEIR dışa aktarımı (morph_beir.py)", "",
                  f"| bölüm | sorgu | havuzlanmış aday | havuzlama kirliliği |",
                  "|---|---|---|---|",
                  f"| train | {beir_train['n_queries']} | {beir_train['n_corpus']} | "
                  f"{beir_train['pooling_contaminated']} sorgu |",
                  f"| dev | {beir_dev['n_queries']} | {beir_dev['n_corpus']} | "
                  f"{beir_dev['pooling_contaminated']} sorgu |", "",
                  "\"Havuzlama kirliliği\": standart BEIR'de tüm adaylar tek bir külliyatta "
                  "birleştirilir; bu sayı, başka bir sorgunun adayının kendi altınını geride "
                  "bıraktığı sorgu sayısıdır. Bu projenin geri kalanındaki tüm sayılar "
                  "`candidate_pool.json` ile sınırlı KAPALI KÜME değerlendirmesidir, havuzlanmış "
                  "BEIR değil — ayrıntı için `beir/<bölüm>/README.md`.", ""]

    lines += ["", "## Sızıntı muhasebesi", "",
              f"- üreticiye örnek olarak gösterilen v1.3.1 öğeleri: "
              f"`{'`, `'.join(ALL_EXEMPLAR_IDS)}`",
              f"- **test kümesi skorları raporlanırken bu {len(ALL_EXEMPLAR_IDS)} öğe hariç "
              f"tutulmalıdır** (50 öğelik test kümesinin %"
              f"{100 * len(ALL_EXEMPLAR_IDS) // 50}'si)",
              f"- üretilen sorguların v1.3.1'e en yüksek benzerliği: "
              f"{V.max_leakage_similarity(accepted)} (eşik 0.60)", ""]

    (OUT_DIR / "generation_report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=600, help="kabul edilecek öğe sayısı")
    ap.add_argument("--overproduce", type=float, default=1.35,
                    help="hedefin kaç katı yuva planlansın (red payı)")
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0,
                    help="eşzamanlı işçi sayısı; 0 = anahtar sayısı kadar")
    ap.add_argument("--batch", type=int, default=12, help="ilerleme raporu ve yasak listesi adımı")
    ap.add_argument("--gen-temp", type=float, default=1.0)
    ap.add_argument("--judge-temp", type=float, default=0.1)
    ap.add_argument("--self-test", action="store_true", help="doğrulayıcı öz-testi, API kullanmaz")
    ap.add_argument("--adopt-legacy-cache", action="store_true",
                    help="eski (geçmişe bağlı) anahtarla önbelleğe alınmış "
                         "yanıtları kanonik anahtara taşı; tek seferlik geçiş")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        import morph_selftest
        sys.exit(1 if morph_selftest.run() else 0)

    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    global ADOPT_LEGACY_CACHE
    ADOPT_LEGACY_CACHE = args.adopt_legacy_cache
    run_generation(args)


if __name__ == "__main__":
    main()
