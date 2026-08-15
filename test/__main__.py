"""Command-line interface for the test benchmark pipeline."""

from __future__ import annotations

import argparse
import json

from .exports import finalize
from .pipeline import default_run_id, generate, paths_for, read_jsonl, write_plan
from .preview import generate_codex_preview
from .review import merge_reviews, prepare_review, review_file
from .selftest import run as run_selftest
from .validators import artifact_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="varsayılan: test/config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="API çağrısı yapmadan kapsam planı üret")
    plan.add_argument("--run-id", default="planned_test_v31")
    plan.add_argument("--size", type=int, default=None)

    generation = sub.add_parser("generate", help="üret + deterministic QC + kör judge")
    generation.add_argument("--run-id", default=None)
    generation.add_argument("--limit", type=int, default=None, help="pilot için plan prefix'i")
    generation.add_argument("--workers", type=int, default=None)

    preview = sub.add_parser(
        "preview-codex", help="API key olmadan yerel Codex oturumuyla preview üret"
    )
    preview.add_argument("--run-id", required=True)
    preview.add_argument("--count", type=int, default=60)
    preview.add_argument("--batch-size", type=int, default=5)
    preview.add_argument("--model", default="gpt-5.6-sol")
    preview.add_argument("--reasoning-effort", default="medium")

    review = sub.add_parser("prepare-review", help="beş reviewer için kör dosyalar hazırla")
    review.add_argument("--run-id", required=True)
    review.add_argument("--force", action="store_true")

    merge = sub.add_parser("merge-reviews", help="review agreement/adjudication durumunu çıkar")
    merge.add_argument("--run-id", required=True)

    reviewer = sub.add_parser("review-file", help="bir reviewer assignment dosyasını interaktif doldur")
    reviewer.add_argument("--path", required=True)

    freeze = sub.add_parser("finalize", help="onaylı 100 dev + 500 sealed testi dondur")
    freeze.add_argument("--run-id", required=True)

    audit = sub.add_parser("audit", help="accepted veya frozen internal veri üzerinde ucuz artefakt denetimi")
    audit.add_argument("--run-id", required=True)

    sub.add_parser("self-test", help="API kullanmadan regression testleri")
    args = parser.parse_args()

    if args.command == "plan":
        result = write_plan(args.run_id, args.config, args.size)
    elif args.command == "generate":
        run_id = args.run_id or default_run_id()
        result = generate(run_id, args.config, args.limit, args.workers)
        result["run_id"] = run_id
    elif args.command == "preview-codex":
        result = generate_codex_preview(
            args.run_id, args.count, args.batch_size, args.model,
            args.reasoning_effort, args.config,
        )
    elif args.command == "prepare-review":
        result = prepare_review(args.run_id, args.config, args.force)
    elif args.command == "merge-reviews":
        result = merge_reviews(args.run_id, args.config)
    elif args.command == "review-file":
        result = review_file(args.path)
    elif args.command == "finalize":
        result = finalize(args.run_id, args.config)
    elif args.command == "audit":
        run = paths_for(args.run_id)
        result = artifact_report(read_jsonl(run.accepted))
    else:
        failures = run_selftest()
        if failures:
            for failure in failures:
                print("FAIL:", failure)
            return 1
        result = {"status": "ok", "message": "test pipeline self-test passed"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
