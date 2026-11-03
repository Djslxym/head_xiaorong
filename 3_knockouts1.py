#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HAWK: Qwen2.5-VL single Query Head zero ablation.

Modes:
  baseline: Run the unmodified model and save a verified baseline.
  ablation: Reuse the latest compatible verified baseline and run one ablation.
  all:      Run baseline first, verify it, then run one ablation.

In ablation mode, --baseline_run_name is optional.
If omitted, the script automatically discovers the latest compatible baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from knockout_utils import (
    discover_score_files,
    extract_realworldqa_accuracy,
    parse_gpu_ids,
    run_hawk_evaluation,
    write_experiment_manifest,
    write_sitecustomize,
)


# ============================================================
# Basic helpers
# ============================================================

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_baseline_name(task: str, limit: int | None) -> str:
    sample_tag = f"n{limit}" if limit is not None else "full"
    return f"{task}_baseline_{sample_tag}_{timestamp()}"


def default_ablation_name(task: str, layer: int, head: int) -> str:
    return f"{task}_L{layer:02d}_H{head:02d}_zero_{timestamp()}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read manifest: {path}\nReason: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid manifest format: {path}")

    return data


# ============================================================
# Result validation
# ============================================================

def find_prediction_csvs(result_dir: Path) -> list[Path]:
    """
    Find non-empty prediction CSV files.

    Score files such as *_acc.csv are excluded.
    This is a conservative structural check; it does not assume a
    particular VLMEvalKit prediction filename.
    """
    if not result_dir.is_dir():
        return []

    candidates: list[Path] = []

    for path in result_dir.rglob("*.csv"):
        name = path.name.lower()

        if "acc" in name or "score" in name:
            continue

        try:
            if path.stat().st_size == 0:
                continue

            with path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                reader = csv.reader(stream)
                rows = list(reader)

            # At least one header row and one data row.
            if len(rows) >= 2 and any(str(cell).strip() for cell in rows[0]):
                candidates.append(path)

        except (OSError, UnicodeDecodeError, csv.Error):
            continue

    return sorted(candidates)


def validate_baseline_result(
    result_dir: Path,
    *,
    task: str,
    limit: int | None,
) -> dict[str, Any]:
    """
    Validate that a baseline has actual prediction and native score artifacts.

    For RealWorldQA:
      - extract_realworldqa_accuracy must successfully parse Overall.
      - At least one non-empty prediction CSV must exist.
      - If --limit is specified, at least one prediction CSV must contain
        at least that many data rows.

    This catches common cases where VLMEvalKit catches an internal exception
    and the outer Python process nevertheless exits with code 0.
    """
    if not result_dir.is_dir():
        raise RuntimeError(
            f"Baseline result directory does not exist:\n{result_dir}"
        )

    score_files = discover_score_files(result_dir)

    if not score_files:
        raise RuntimeError(
            "Baseline validation failed: no native score/accuracy artifact "
            f"was found in:\n{result_dir}"
        )

    accuracy = extract_realworldqa_accuracy(result_dir)

    if accuracy is None:
        raise RuntimeError(
            "Baseline validation failed: the native RealWorldQA Accuracy "
            "file was found or expected, but its Overall value could not "
            f"be parsed.\nResult directory: {result_dir}"
        )

    prediction_files = find_prediction_csvs(result_dir)

    if not prediction_files:
        raise RuntimeError(
            "Baseline validation failed: no non-empty prediction CSV "
            f"was found in:\n{result_dir}"
        )

    prediction_row_counts: dict[str, int] = {}

    for path in prediction_files:
        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                rows = list(csv.reader(stream))

            prediction_row_counts[str(path.resolve())] = max(0, len(rows) - 1)

        except (OSError, UnicodeDecodeError, csv.Error):
            continue

    if not prediction_row_counts:
        raise RuntimeError(
            "Baseline validation failed: prediction CSV files could not "
            f"be read in:\n{result_dir}"
        )

    if limit is not None:
        enough_rows = any(
            count >= limit
            for count in prediction_row_counts.values()
        )

        if not enough_rows:
            details = "\n".join(
                f"  {path}: {count} rows"
                for path, count in prediction_row_counts.items()
            )
            raise RuntimeError(
                "Baseline validation failed: no prediction CSV contains "
                f"at least {limit} data rows.\n"
                f"Observed prediction files:\n{details}"
            )

    return {
        "accuracy_percent": accuracy,
        "native_score_files": score_files,
        "prediction_files": [
            str(path.resolve()) for path in prediction_files
        ],
        "prediction_row_counts": prediction_row_counts,
    }


# ============================================================
# Baseline manifest
# ============================================================

def baseline_manifest_path(results_root: Path, run_name: str) -> Path:
    return results_root / f"{run_name}_baseline_manifest.json"


def create_baseline_manifest(
    results_root: Path,
    *,
    run_name: str,
    task: str,
    model_path: Path,
    limit: int | None,
    validation: dict[str, Any],
) -> Path:
    result_dir = results_root / run_name

    payload = {
        "task": task,
        "model_path": str(model_path.resolve()),
        "pruning_ratio": 0.0,
        "keep_ratio": 1.0,
        "limit": limit,
        "baseline_run_name": run_name,
        "baseline_result_dir": str(result_dir.resolve()),
        "baseline_accuracy_percent": validation["accuracy_percent"],
        "native_score_files": validation["native_score_files"],
        "prediction_files": validation["prediction_files"],
        "prediction_row_counts": validation["prediction_row_counts"],
        "validation_status": "passed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    path = baseline_manifest_path(results_root, run_name)
    write_json(path, payload)
    return path


# ============================================================
# Baseline compatibility and discovery
# ============================================================

def check_manifest_compatibility(
    manifest: dict[str, Any],
    *,
    task: str,
    model_path: Path,
    limit: int | None,
) -> list[str]:
    mismatches: list[str] = []

    if manifest.get("task") != task:
        mismatches.append(
            f"task: baseline={manifest.get('task')} vs current={task}"
        )

    saved_model = manifest.get("model_path")

    if not saved_model:
        mismatches.append("manifest has no model_path")
    else:
        try:
            saved_model_path = Path(saved_model).expanduser().resolve()
            current_model_path = model_path.expanduser().resolve()

            if saved_model_path != current_model_path:
                mismatches.append(
                    "model_path: "
                    f"baseline={saved_model_path} "
                    f"vs current={current_model_path}"
                )
        except Exception:
            mismatches.append("manifest model_path is invalid")

    if manifest.get("limit") != limit:
        mismatches.append(
            f"limit: baseline={manifest.get('limit')} vs current={limit}"
        )

    if manifest.get("pruning_ratio") != 0.0:
        mismatches.append(
            "baseline pruning_ratio is not 0.0"
        )

    if manifest.get("validation_status") != "passed":
        mismatches.append(
            "baseline manifest does not have validation_status='passed'"
        )

    return mismatches


def verify_named_baseline(
    results_root: Path,
    run_name: str,
    *,
    task: str,
    model_path: Path,
    limit: int | None,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = baseline_manifest_path(results_root, run_name)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Baseline manifest not found:\n{manifest_path}\n\n"
            "The baseline cannot be safely reused without a manifest. "
            "Run --mode baseline to create a verified baseline."
        )

    manifest = read_json(manifest_path)

    mismatches = check_manifest_compatibility(
        manifest,
        task=task,
        model_path=model_path,
        limit=limit,
    )

    if mismatches:
        raise ValueError(
            "The selected baseline is incompatible:\n  "
            + "\n  ".join(mismatches)
        )

    result_dir = results_root / run_name

    # Re-check the actual result files instead of trusting the manifest alone.
    validation = validate_baseline_result(
        result_dir,
        task=task,
        limit=limit,
    )

    return result_dir, validation


def discover_latest_baseline(
    results_root: Path,
    *,
    task: str,
    model_path: Path,
    limit: int | None,
) -> tuple[str, Path, dict[str, Any]]:
    """
    Automatically find the newest compatible and valid baseline.

    Only manifests with validation_status='passed' are considered.
    Actual prediction and score files are checked again.
    """
    manifests = list(results_root.glob("*_baseline_manifest.json"))

    if not manifests:
        raise FileNotFoundError(
            "No baseline manifest was found in:\n"
            f"{results_root}\n\n"
            "Please run baseline mode first:\n"
            "python 3_knockouts.py --mode baseline "
            f"--task {task} --model_path {model_path} "
            f"--gpus 0 --num_processes 1"
        )

    candidates: list[tuple[datetime, str, Path, dict[str, Any]]] = []
    rejected: list[str] = []

    for manifest_path in manifests:
        try:
            manifest = read_json(manifest_path)

            run_name = manifest.get("baseline_run_name")

            if not run_name:
                # Infer the run name from the conventional manifest filename.
                suffix = "_baseline_manifest.json"
                if manifest_path.name.endswith(suffix):
                    run_name = manifest_path.name[:-len(suffix)]

            if not run_name:
                rejected.append(
                    f"{manifest_path.name}: missing baseline_run_name"
                )
                continue

            mismatches = check_manifest_compatibility(
                manifest,
                task=task,
                model_path=model_path,
                limit=limit,
            )

            if mismatches:
                rejected.append(
                    f"{run_name}: " + "; ".join(mismatches)
                )
                continue

            result_dir = results_root / run_name

            validation = validate_baseline_result(
                result_dir,
                task=task,
                limit=limit,
            )

            created_at_text = manifest.get("created_at")

            try:
                created_at = datetime.fromisoformat(created_at_text)
            except (TypeError, ValueError):
                created_at = datetime.fromtimestamp(
                    manifest_path.stat().st_mtime
                )

            candidates.append(
                (created_at, run_name, result_dir, validation)
            )

        except Exception as exc:
            rejected.append(
                f"{manifest_path.name}: {exc}"
            )

    if not candidates:
        details = "\n".join(f"  - {item}" for item in rejected)

        raise RuntimeError(
            "No compatible and valid baseline was found.\n"
            "The script will not silently run a new baseline.\n\n"
            "Rejected candidates:\n"
            f"{details}\n\n"
            "If none of these baselines is usable, run --mode baseline "
            "to create a new one."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)

    _, run_name, result_dir, validation = candidates[0]

    return run_name, result_dir, validation


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HAWK baseline and single Qwen2.5-VL Query Head "
            "zero ablation experiments."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "ablation", "all"],
        default="ablation",
        help="baseline: baseline only; ablation: reuse baseline; all: baseline then ablation",
    )

    parser.add_argument(
        "--task",
        choices=["realworldqa"],
        default="realworldqa",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Zero-based Transformer layer index.",
    )

    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Zero-based Query Head index.",
    )

    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/Qwen2.5-VL-7B-Instruct"),
    )

    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU ids, e.g. 0 or 0,1,2,3.",
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit.",
    )

    parser.add_argument(
        "--run_name",
        default=None,
        help=(
            "Output run name. In baseline mode it names the baseline; "
            "in ablation mode it names the ablation."
        ),
    )

    parser.add_argument(
        "--baseline_run_name",
        default=None,
        help=(
            "Optional existing baseline name. "
            "If omitted in ablation mode, automatically find the latest "
            "compatible verified baseline."
        ),
    )

    parser.add_argument(
        "--skip_data_download",
        action="store_true",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent

    required_files = [
        project_root / "scripts" / "evaluate.py",
        project_root / "scripts" / "evaluate_vlmeval.sh",
        project_root / "src" / "hawk" / "vlmeval_adapter.py",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Place 3_knockouts.py in the HAWK repository root. "
            "Missing required HAWK files:\n  "
            + "\n  ".join(missing)
        )

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    if args.mode in {"ablation", "all"}:
        if args.layer is None or args.head is None:
            raise ValueError(
                "--layer and --head are required for ablation/all mode"
            )

        if args.layer < 0 or args.head < 0:
            raise ValueError(
                "--layer and --head must be non-negative integers"
            )

        # Qwen2.5-VL-7B-Instruct: 28 layers, 28 Query Heads.
        if args.layer >= 28:
            raise ValueError("--layer must be in [0, 27]")

        if args.head >= 28:
            raise ValueError("--head must be in [0, 27]")

    gpu_ids = parse_gpu_ids(args.gpus)

    num_processes = args.num_processes

    if num_processes is not None:
        if not 1 <= num_processes <= len(gpu_ids):
            raise ValueError(
                "--num_processes must be between 1 and the number of --gpus"
            )

    model_path = args.model_path.expanduser()

    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()

    if not args.dry_run and not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    results_root = project_root / "vlmeval_results"
    results_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Mode 1: baseline only
    # --------------------------------------------------------

    if args.mode == "baseline":
        baseline_name = (
            args.run_name
            or default_baseline_name(args.task, args.limit)
        )

        baseline_dir = results_root / baseline_name

        if baseline_dir.exists() and any(baseline_dir.iterdir()):
            raise FileExistsError(
                f"Baseline output already exists and is not empty:\n"
                f"{baseline_dir}\n"
                "Choose another --run_name."
            )

        print("=" * 72)
        print("HAWK BASELINE")
        print(f"Task:       {args.task}")
        print(f"Model:      {model_path}")
        print(f"GPUs:       {','.join(gpu_ids)}")
        print("Pruning:    disabled (pruning_ratio=0.0)")
        print(f"Run name:   {baseline_name}")
        print(f"Limit:      {args.limit or 'full dataset'}")
        print("=" * 72, flush=True)

        run_hawk_evaluation(
            project_root,
            task=args.task,
            model_path=model_path,
            gpu_ids=gpu_ids,
            run_name=baseline_name,
            limit=args.limit,
            num_processes=num_processes,
            skip_data_download=args.skip_data_download,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            print(f"[dry-run] Baseline run name: {baseline_name}")
            return

        # Important: validate actual files before writing a success manifest.
        validation = validate_baseline_result(
            baseline_dir,
            task=args.task,
            limit=args.limit,
        )

        manifest_path = create_baseline_manifest(
            results_root,
            run_name=baseline_name,
            task=args.task,
            model_path=model_path,
            limit=args.limit,
            validation=validation,
        )

        print("\nBaseline validation passed.")
        print(f"Baseline run name: {baseline_name}")
        print(f"Baseline directory: {baseline_dir}")
        print(f"Baseline manifest:  {manifest_path}")
        print(
            f"Baseline Accuracy:  "
            f"{validation['accuracy_percent']:.2f}%"
        )
        print("Prediction CSV files:")

        for path in validation["prediction_files"]:
            print(f"  {path}")

        print("Native score files:")

        for path in validation["native_score_files"]:
            print(f"  {path}")

        return

    # --------------------------------------------------------
    # Mode 2: baseline then ablation
    # --------------------------------------------------------

    if args.mode == "all":
        baseline_name = (
            args.baseline_run_name
            or default_baseline_name(args.task, args.limit)
        )

        baseline_dir = results_root / baseline_name

        if baseline_dir.exists() and any(baseline_dir.iterdir()):
            raise FileExistsError(
                f"Baseline output already exists:\n{baseline_dir}"
            )

        print(f"[1/2] Running baseline: {baseline_name}", flush=True)

        run_hawk_evaluation(
            project_root,
            task=args.task,
            model_path=model_path,
            gpu_ids=gpu_ids,
            run_name=baseline_name,
            limit=args.limit,
            num_processes=num_processes,
            skip_data_download=args.skip_data_download,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            validation = validate_baseline_result(
                baseline_dir,
                task=args.task,
                limit=args.limit,
            )

            manifest_path = create_baseline_manifest(
                results_root,
                run_name=baseline_name,
                task=args.task,
                model_path=model_path,
                limit=args.limit,
                validation=validation,
            )

            print(
                f"Baseline validation passed: {manifest_path}",
                flush=True,
            )

        else:
            print(
                f"[dry-run] Baseline run name: {baseline_name}",
                flush=True,
            )

    # --------------------------------------------------------
    # Mode 3: ablation only; auto-discover baseline if needed
    # --------------------------------------------------------

    else:
        if args.baseline_run_name:
            baseline_name = args.baseline_run_name

            if args.dry_run:
                baseline_dir = results_root / baseline_name
                print(
                    f"[dry-run] Use specified baseline: {baseline_name}"
                )
            else:
                baseline_dir, validation = verify_named_baseline(
                    results_root,
                    baseline_name,
                    task=args.task,
                    model_path=model_path,
                    limit=args.limit,
                )

        elif args.dry_run:
            baseline_name = "<auto-discover-latest-compatible-baseline>"
            baseline_dir = results_root / baseline_name

            print(
                "[dry-run] Baseline will be automatically discovered "
                "when the real experiment runs."
            )

        else:
            print(
                "Searching for the latest compatible verified baseline...",
                flush=True,
            )

            baseline_name, baseline_dir, validation = (
                discover_latest_baseline(
                    results_root,
                    task=args.task,
                    model_path=model_path,
                    limit=args.limit,
                )
            )

            print("Automatically selected baseline:")
            print(f"  Run name: {baseline_name}")
            print(f"  Directory: {baseline_dir}")
            print(
                f"  Accuracy: {validation['accuracy_percent']:.2f}%"
            )

    # --------------------------------------------------------
    # Run one head ablation
    # --------------------------------------------------------

    ablated_name = (
        args.run_name
        or default_ablation_name(
            args.task,
            args.layer,
            args.head,
        )
    )

    ablated_dir = results_root / ablated_name

    if ablated_dir.exists() and any(ablated_dir.iterdir()):
        raise FileExistsError(
            f"Ablation output already exists:\n{ablated_dir}\n"
            "Choose another --run_name."
        )

    print("=" * 72)
    print("HAWK SINGLE QUERY HEAD ZERO ABLATION")
    print(f"Task:       {args.task}")
    print(f"Layer/Head: {args.layer}/{args.head}")
    print(f"Model:      {model_path}")
    print(f"GPUs:       {','.join(gpu_ids)}")
    print("Pruning:    disabled (pruning_ratio=0.0)")
    print(f"Baseline:   {baseline_name}")
    print(f"Ablation:   {ablated_name}")
    print("=" * 72, flush=True)

    with tempfile.TemporaryDirectory(
        prefix="hawk_head_ablation_"
    ) as temp_dir:

        patch_dir = Path(temp_dir)
        write_sitecustomize(patch_dir)

        run_hawk_evaluation(
            project_root,
            task=args.task,
            model_path=model_path,
            gpu_ids=gpu_ids,
            run_name=ablated_name,
            limit=args.limit,
            num_processes=num_processes,
            skip_data_download=True,
            dry_run=args.dry_run,
            ablation=(args.layer, args.head),
            patch_dir=patch_dir,
        )

    if args.dry_run:
        print(
            "[dry-run] Ablation command prepared. "
            "No baseline or ablation was executed."
        )
        return

    # --------------------------------------------------------
    # Save experiment record
    # --------------------------------------------------------

    manifest_path = results_root / f"{ablated_name}_experiment.json"

    write_experiment_manifest(
        manifest_path,
        task=args.task,
        layer=args.layer,
        head=args.head,
        model_path=model_path,
        baseline_dir=baseline_dir,
        ablated_dir=ablated_dir,
        limit=args.limit,
    )

    baseline_accuracy = extract_realworldqa_accuracy(baseline_dir)
    ablated_accuracy = extract_realworldqa_accuracy(ablated_dir)

    print("\nAblation evaluation command finished.")
    print(f"Baseline directory: {baseline_dir}")
    print(f"Ablated directory:  {ablated_dir}")
    print(f"Experiment manifest: {manifest_path}")

    if baseline_accuracy is not None:
        print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    else:
        print("Baseline Accuracy: unavailable")

    if ablated_accuracy is not None:
        print(f"Ablated Accuracy:  {ablated_accuracy:.2f}%")
    else:
        print("Ablated Accuracy:  unavailable")

    if (
        baseline_accuracy is not None
        and ablated_accuracy is not None
    ):
        print(
            "Accuracy drop:     "
            f"{baseline_accuracy - ablated_accuracy:+.2f} "
            "percentage points"
        )

    print(
        "\nNote: The native VLMEvalKit prediction and score files "
        "are retained in the result directories."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\n[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
