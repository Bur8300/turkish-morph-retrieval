#!/usr/bin/env python
"""Local semantic-encoder retrieval benchmark for the Turkish dual-encoder project.

Ranks candidate semantic encoders on 4 trmteb Turkish-BEIR datasets (nDCG@10 / Recall / MRR).
Run in the dl_hw1 conda env:

    conda run -n dl_hw1 python eval_semantic_encoders.py            # full run (hours on MPS)
    SMOKE=1 conda run -n dl_hw1 python eval_semantic_encoders.py    # quick pipeline check

Reuses the proven BEIR loader + metrics from model_selection_colab.ipynb. Results/plots go to
results_local/; each model's result is cached there, so an interrupted run resumes.
"""
import os
# Must be set before torch is imported: lets any MPS-unsupported op fall back to CPU
# instead of raising; and keep HF's flaky xet transfer backend out of the way.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import gc
import io
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.constants import HF_HUB_CACHE
from sentence_transformers import SentenceTransformer

SMOKE = os.environ.get("SMOKE", "0") == "1"
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cpu")
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "results_local"
CACHE_DIR.mkdir(exist_ok=True)


def slug(model_id):
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_id)


# --------------------------------------------------------------------------- registry
# role "finished" = retrieval/STS-tuned embedder, run zero-shot as-is.
# role "raw"      = MLM foundation model, no pooling head -> SentenceTransformer auto-wraps
#                   with mean pooling; expect low zero-shot scores (it's a base to fine-tune).
CANDIDATES = [
    dict(id="intfloat/multilingual-e5-large", short="mE5-large", role="finished",
         query_prompt="query: ", doc_prompt="passage: ", trust_remote_code=False),
    dict(id="newmindai/TurkEmbed4STS", short="TurkEmbed4STS", role="finished",
         query_prompt="", doc_prompt="", trust_remote_code=True),
    dict(id="newmindai/TurkEmbed4Retrieval", short="TurkEmbed4Retrieval", role="finished",
         query_prompt="", doc_prompt="", trust_remote_code=True),
    dict(id="ytu-ce-cosmos/turkish-e5-large", short="turkish-e5-large", role="finished",
         query_prompt="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
         doc_prompt="", trust_remote_code=False),
    dict(id="BAAI/bge-m3", short="BGE-M3", role="finished",
         query_prompt="", doc_prompt="", trust_remote_code=False),
    dict(id="Alibaba-NLP/gte-multilingual-base", short="GTE-mult-base", role="finished",
         query_prompt="", doc_prompt="", trust_remote_code=True),
    dict(id="newmindai/Mursit-Base-TR-Retrieval", short="Mursit-Base", role="finished",
         query_prompt="", doc_prompt="", trust_remote_code=False),
    dict(id="trmteb/turkish-embedding-model", short="trmteb-pretrain", role="finished",
         query_prompt="", doc_prompt="", trust_remote_code=False),
    dict(id="boun-tabilab/TabiBERT", short="TabiBERT", role="raw",
         query_prompt="", doc_prompt="", trust_remote_code=False),
    dict(id="TURKCELL/roberta-base-turkish-uncased", short="TURKCELL-roberta", role="raw",
         query_prompt="", doc_prompt="", trust_remote_code=False),
]
SMOKE_MODELS = {"newmindai/Mursit-Base-TR-Retrieval", "trmteb/turkish-embedding-model"}


def active_candidates():
    if SMOKE:
        return [c for c in CANDIDATES if c["id"] in SMOKE_MODELS]
    return CANDIDATES


# --------------------------------------------------------------------------- BEIR loader
# All 4 datasets are trmteb-style: `corpus` + `queries` configs, and qrels under `data/<split>`.
# Fetch parquet with plain requests (HfApi lists paths; hf_hub_url builds the resolve URL) --
# this transport is dataset-repo-agnostic and avoids the datasets/xet flakiness entirely.
_hf_api = HfApi()
_parquet_cache = {}
BEIR_DATASETS = {
    "scifact-tr": "trmteb/scifact-tr",
    "nfcorpus-tr": "trmteb/nfcorpus-tr",
    "arguana-tr": "trmteb/arguana-tr",
    "fiqa-tr": "trmteb/fiqa-tr",
}
EVAL_DATASETS = ["scifact-tr"] if SMOKE else list(BEIR_DATASETS)
# fiqa's 57K corpus dominates MPS runtime (76% of total encode cost). Cap it to keep the run
# tractable: retrieval metrics stay valid because we keep EVERY relevant (gold) document and
# only subsample the non-relevant filler. The other 3 datasets run at full size.
FIQA_CORPUS_CAP = 12000


def _list_parquet(repo):
    if repo not in _parquet_cache:
        _parquet_cache[repo] = [f for f in _hf_api.list_repo_files(repo, repo_type="dataset")
                                if f.endswith(".parquet")]
    return _parquet_cache[repo]


def _read(repo, folder, split=None):
    files = _list_parquet(repo)
    term = split or folder
    paths = sorted(f for f in files if f.startswith(f"{folder}/") and term in f)
    frames = []
    for p in paths:
        r = requests.get(hf_hub_url(repo, p, repo_type="dataset"), timeout=180)
        r.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(r.content)))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _doc_text(row):
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def load_beir_dataset(name):
    repo = BEIR_DATASETS[name]
    corpus_rows = _read(repo, "corpus").to_dict("records")
    query_rows = _read(repo, "queries").to_dict("records")
    qrels_rows = _read(repo, "data", "test").to_dict("records")

    corpus = {str(r["_id"]): _doc_text(r) for r in corpus_rows}
    queries = {str(r["_id"]): r["text"] for r in query_rows}
    qrels = {}
    for r in qrels_rows:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = float(r["score"])
    queries = {qid: q for qid, q in queries.items() if qid in qrels}  # BEIR convention

    if not SMOKE and name == "fiqa-tr" and len(corpus) > FIQA_CORPUS_CAP:
        gold = {d for rels in qrels.values() for d in rels}          # every relevant doc: kept
        filler = [d for d in corpus if d not in gold][:FIQA_CORPUS_CAP - len(gold)]
        corpus = {d: corpus[d] for d in list(gold) + filler}

    if SMOKE:  # tiny but valid subset: gold docs + a filler sample
        qids = list(queries)[:16]
        queries = {q: queries[q] for q in qids}
        qrels = {q: qrels[q] for q in qids}
        gold = {d for q in qids for d in qrels[q]}
        filler = [d for d in corpus if d not in gold][:1200]
        corpus = {d: corpus[d] for d in list(gold) + filler}
    return corpus, queries, qrels


# --------------------------------------------------------------------------- metrics
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def evaluate_run(qrels, run, k_ndcg=10, k_mrr=10, ks_recall=(10, 100)):
    ndcgs, mrrs = [], []
    recalls = {k: [] for k in ks_recall}
    for qid, ranked in run.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        ideal = dcg(sorted(rel.values(), reverse=True)[:k_ndcg])
        ndcgs.append(dcg([rel.get(d, 0.0) for d in ranked[:k_ndcg]]) / ideal if ideal > 0 else 0.0)
        mrrs.append(next((1.0 / (i + 1) for i, d in enumerate(ranked[:k_mrr]) if rel.get(d, 0) > 0), 0.0))
        n_rel = sum(1 for v in rel.values() if v > 0)
        for k in ks_recall:
            recalls[k].append(sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / n_rel if n_rel else 0.0)
    out = {"nDCG@10": np.mean(ndcgs), "MRR@10": np.mean(mrrs)}
    out.update({f"Recall@{k}": np.mean(recalls[k]) for k in ks_recall})
    return {m: round(float(v) * 100, 2) for m, v in out.items()}


def search(query_emb, doc_emb, doc_ids, query_ids, top_k=100, remove_self=False):
    dev = "cuda" if DEVICE == "cuda" else "cpu"  # top-k on CPU; MPS gains nothing for this matmul
    D = torch.from_numpy(doc_emb).to(dev)
    Q = torch.from_numpy(query_emb).to(dev)
    run = {}
    for start in range(0, len(Q), 256):
        chunk = Q[start:start + 256]
        scores = chunk @ D.T
        k = min(top_k + (1 if remove_self else 0), len(doc_ids))
        top = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        for row, qi in zip(top, range(start, start + len(chunk))):
            qid = query_ids[qi]
            ranked = [doc_ids[j] for j in row]
            if remove_self:
                ranked = [d for d in ranked if d != qid]
            run[qid] = ranked[:top_k]
    del D, Q
    return run


# --------------------------------------------------------------------------- model I/O
def _sanity_check(model, cand):
    v = model.encode(["kontrol cümlesi"], convert_to_numpy=True)
    if not np.isfinite(v).all():
        raise RuntimeError(f"{cand['id']}: NaN/inf embeddings — model loaded incorrectly")


def _clear_hf_cache(repo_id):
    d = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def load_model(cand, attempts=3):
    last_err = None
    for i in range(attempts):
        try:
            model = SentenceTransformer(cand["id"], device=DEVICE,
                                        trust_remote_code=cand["trust_remote_code"])
            model.max_seq_length = min(getattr(model, "max_seq_length", 512) or 512, 512)
            _sanity_check(model, cand)
            return model
        except Exception as e:
            last_err = e
            _clear_hf_cache(cand["id"])  # a partial download poisons the next attempt
            if i < attempts - 1:
                print(f"    [retry {i+1}/{attempts-1}] {cand['id']}: {type(e).__name__}: {str(e)[:150]}")
                time.sleep(2 ** i)
    raise last_err


def encode(model, texts, prompt="", batch_size=64):
    if prompt:
        texts = [prompt + t for t in texts]
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True)


def free(model):
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- main
def main():
    print(f"device={DEVICE}  smoke={SMOKE}  torch={torch.__version__}")
    data = {name: load_beir_dataset(name) for name in EVAL_DATASETS}
    for name, (c, q, r) in data.items():
        print(f"  {name}: corpus={len(c)}  queries={len(q)}  qrels={len(r)}")

    rows = []
    for cand in active_candidates():
        res_path = CACHE_DIR / f"retr_{slug(cand['id'])}{'_smoke' if SMOKE else ''}.json"
        if res_path.exists():
            rows += json.loads(res_path.read_text())
            print(f"[cache] {cand['short']}")
            continue
        print(f"\n=== {cand['id']} ({cand['role']}) ===")
        try:
            model = load_model(cand)
        except Exception as e:
            print(f"  [skip] could not load: {type(e).__name__}: {str(e)[:160]}")
            continue
        try:
            model_rows = []
            for name in EVAL_DATASETS:
                corpus, queries, qrels = data[name]
                doc_ids, query_ids = list(corpus), list(queries)
                t0 = time.time()
                doc_emb = encode(model, [corpus[d] for d in doc_ids], cand["doc_prompt"])
                query_emb = encode(model, [queries[q] for q in query_ids], cand["query_prompt"])
                run = search(query_emb, doc_emb, doc_ids, query_ids, remove_self=(name == "arguana-tr"))
                metrics = evaluate_run(qrels, run)
                dt = time.time() - t0
                model_rows.append(dict(model=cand["short"], role=cand["role"], dataset=name,
                                       **metrics, encode_sec=round(dt, 1)))
                print(f"  {name}: {metrics}  ({dt:.0f}s)")
            res_path.write_text(json.dumps(model_rows))
            rows += model_rows
        except Exception as e:
            print(f"  [skip] error evaluating: {type(e).__name__}: {str(e)[:160]}")
        finally:
            free(model)

    if not rows:
        print("\nNo results produced.")
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["model", "role"], columns="dataset", values="nDCG@10")
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).round(2)
    print("\n===== mean nDCG@10 (higher = better) =====")
    print(pivot.to_string())

    csv_path = CACHE_DIR / f"semantic_retrieval{'_smoke' if SMOKE else ''}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = pivot["mean"].sort_values()
        colors = ["#9ca3af" if role == "raw" else "#2563eb" for _, role in d.index]
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.2))
        bars = ax.barh([m for m, _ in d.index], d.values, color=colors, height=0.62)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_title(f"Semantic-encoder candidates — mean nDCG@10 across {len(EVAL_DATASETS)} "
                     f"trmteb datasets\n(gray = raw MLM, zero-shot floor)", fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
        ax.set_axisbelow(True)
        plt.tight_layout()
        png_path = CACHE_DIR / f"semantic_retrieval{'_smoke' if SMOKE else ''}.png"
        plt.savefig(png_path, dpi=140)
        print(f"wrote {png_path}")
    except Exception as e:
        print(f"[plot skipped] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
