"""Binary full-corpus pooling utilities.

Only pooled rows receive judgments. Documents outside the pool remain unknown; they are never
silently exported as relevance=0.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evaluation import validate_binary_qrels


def load_pool_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pool_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = Counter(
        "unjudged" if row.get("relevance") is None else str(row.get("relevance"))
        for row in rows
    )
    return {
        "rows": len(rows),
        "queries": len({str(row.get("query_id")) for row in rows}),
        "labels": dict(sorted(values.items())),
        "complete": values.get("unjudged", 0) == 0,
    }


def finalize_binary_qrels(pool_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows = load_pool_rows(pool_path)
    missing = [
        f"{row.get('query_id')}::{row.get('corpus_id')}"
        for row in rows if row.get("relevance") is None
    ]
    if missing:
        raise ValueError(
            f"Pool tamamlanmadı: {len(missing)} satır hâlâ null. "
            "Yalnız gerçekten yargılanan satıra 0 veya 1 verin; örnek: " + ", ".join(missing[:5])
        )
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        query_id, corpus_id = str(row["query_id"]), str(row["corpus_id"])
        if corpus_id in qrels[query_id]:
            raise ValueError(f"Pool içinde mükerrer query/document çifti: {query_id} / {corpus_id}")
        qrels[query_id][corpus_id] = float(row["relevance"])
    qrels = dict(qrels)
    validate_binary_qrels(qrels, require_nonrelevant=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query-id\tcorpus-id\tscore\n"]
    for query_id in sorted(qrels):
        for corpus_id, score in sorted(qrels[query_id].items()):
            lines.append(f"{query_id}\t{corpus_id}\t{int(score)}\n")
    output.write_text("".join(lines), encoding="utf-8")
    status = pool_status(rows)
    status.update({"output": str(output.resolve()), "judgments": sum(map(len, qrels.values()))})
    return status
