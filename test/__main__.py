"""Command-line interface for the test benchmark pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluation import load_items
from .exports import finalize
from .pipeline import default_run_id, generate, paths_for, read_jsonl, write_plan
from .preview import generate_codex_preview
from .selftest import run as run_selftest
from .validators import artifact_report, train_test_leakage_problems


def _load_flexible(path: str) -> list[dict]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return load_items(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="varsayılan: test/config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="API çağrısı yapmadan kapsam planı üret")
    plan.add_argument("--run-id", default="planned_test_v36")
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
    preview.add_argument("--workers", type=int, default=1)
    preview.add_argument("--reserve-slots", type=int, default=0)
    preview.add_argument("--cache-only", action="store_true")

    freeze = sub.add_parser(
        "finalize",
        help="otomatik doğrulanmış 100 dev + 500 sealed testi dondur",
    )
    freeze.add_argument("--run-id", required=True)

    audit = sub.add_parser("audit", help="accepted veya frozen internal veri üzerinde ucuz artefakt denetimi")
    audit.add_argument("--run-id", required=True)

    leakage = sub.add_parser("audit-leakage", help="train ile test arasında exact/fuzzy kopya ara")
    leakage.add_argument("--test", required=True)
    leakage.add_argument("--train", required=True)

    morph = sub.add_parser("morph-audit", help="opsiyonel Stanza lemma/UFeats denetimi")
    morph.add_argument("--input", required=True)
    morph.add_argument("--output", required=True)
    morph.add_argument("--download-model", action="store_true")
    morph.add_argument("--use-gpu", action="store_true")

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
            args.reasoning_effort, args.config, args.workers, args.reserve_slots,
            args.cache_only,
        )
    elif args.command == "finalize":
        result = finalize(args.run_id, args.config)
    elif args.command == "audit":
        run = paths_for(args.run_id)
        result = artifact_report(read_jsonl(run.accepted))
    elif args.command == "audit-leakage":
        cfg = load_config(args.config, runtime=False)
        problems = train_test_leakage_problems(
            _load_flexible(args.test), _load_flexible(args.train), cfg
        )
        result = {"problem_count": len(problems), "problems": problems}
    elif args.command == "morph-audit":
        from .morphology import run_morphology_audit
        result = run_morphology_audit(
            args.input, args.output, args.download_model, args.use_gpu
        )
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
