from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .pipeline import (
    REPOSITORY_ROOT,
    build_extract_config,
    dual_camera_config_paths,
    extract,
    finalize_run_root,
    preflight,
    prepare_run_root,
    run,
    run_batch,
)
from .report import build_dual_report
from .safety import is_within, require_absolute_directory


def _add_raw_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imagecal", description="RAW exposure, white-balance, and neutral-axis normalization"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_raw_run_arguments(sub.add_parser("run", help="Process one camera validation set"))
    _add_raw_run_arguments(sub.add_parser("run-dual", help="Process both validation sets"))
    _add_raw_run_arguments(sub.add_parser("run-full", help="Process both full camera series"))

    extract_parser = sub.add_parser("extract", help="Re-extract metrics from an existing run")
    extract_parser.add_argument("--config", type=Path, required=True)
    extract_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def _configuration_paths(path: Path, *, dual: bool) -> list[Path]:
    return dual_camera_config_paths(path) if dual else [path.resolve()]


def _run_raw_command(args: argparse.Namespace) -> int:
    dual = args.command in {"run-dual", "run-full"}
    full = args.command == "run-full"
    config_paths = _configuration_paths(args.config, dual=dual)
    source_root, output_root, identity = prepare_run_root(
        command=args.command,
        config_paths=config_paths,
        source_root=args.source_root,
        output_root=args.output_dir,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        result = preflight(config_paths, source_root, output_root, full=full)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    results = []
    for config_path in config_paths:
        if full:
            config, rows, color_rows, acceptance = run_batch(
                config_path, source_root, output_root, resume=args.resume
            )
        else:
            config, rows, color_rows, acceptance = run(
                config_path, source_root, output_root, resume=args.resume
            )
        del color_rows
        results.append((config, rows, acceptance))

    if dual:
        build_dual_report(results, output_root)
    passed = all(item[2]["passed"] for item in results)
    finalize_run_root(output_root, identity, passed)
    print(
        json.dumps(
            {"processed": sum(len(item[1]) for item in results), "passed": passed},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 2


def _extract(args: argparse.Namespace) -> int:
    run_root = require_absolute_directory(args.run_dir, "--run-dir", must_exist=True)
    repository = REPOSITORY_ROOT.resolve()
    if is_within(run_root, repository) or is_within(repository, run_root):
        raise RuntimeError(f"Run directory must be separate from the repository: {run_root}")

    document = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")) or {}
    paths = dual_camera_config_paths(args.config) if "cameras" in document else [args.config.resolve()]
    count = 0
    for path in paths:
        config = build_extract_config(path, run_root)
        count += len(extract(config))
    print(json.dumps({"extracted": count, "run_dir": str(run_root)}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "extract":
        return _extract(args)
    return _run_raw_command(args)


if __name__ == "__main__":
    raise SystemExit(main())

