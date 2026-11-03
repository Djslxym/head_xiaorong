#!/usr/bin/env python3
"""Run HAWK baseline or a single Qwen2.5-VL Query Head zero ablation.

Examples:
  # Run baseline once
  python 3_knockouts.py --mode baseline --task realworldqa --gpus 0 --limit 5

  # Reuse that baseline for a head ablation (use the exact run name printed above)
  python 3_knockouts.py --mode ablation --task realworldqa --layer 0 --head 0 \
      --baseline_run_name realworldqa_baseline_n5_YYYYMMDD_HHMMSS --gpus 0 --limit 5
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from knockout_utils import (
    discover_score_files,
    extract_realworldqa_accuracy,
    parse_gpu_ids,
    run_hawk_evaluation,
    write_experiment_manifest,
    write_sitecustomize,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_baseline_name(task: str, limit: int | None) -> str:
    sample_tag = f"n{limit}" if limit is not None else "full"
    return f"{task}_baseline_{sample_tag}_{_timestamp()}"


def _default_ablation_name(task: str, layer: int, head: int) -> str:
    return f"{task}_L{layer:02d}_H{head:02d}_zero_{_timestamp()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HAWK baseline once, then reuse it for individual Query Head ablations."
    )
    parser.add_argument("--mode", choices=["baseline", "ablation", "all"], default="ablation",
                        help="baseline: baseline only; ablation: ablation only; all: baseline then ablation")
    parser.add_argument("--task", choices=["realworldqa"], default="realworldqa")
    parser.add_argument("--layer", type=int, default=None, help="0-based Transformer layer index (required for ablation/all)")
    parser.add_argument("--head", type=int, default=None, help="0-based Query Head index (required for ablation/all)")
    parser.add_argument("--model_path", type=Path, default=Path("models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0 or 0,1,2,3")
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests")
    parser.add_argument("--run_name", default=None,
                        help="Output run name. For baseline mode this names the baseline; for ablation mode it names the ablation run.")
    parser.add_argument("--baseline_run_name", default=None,
                        help="Existing baseline run name to reuse in ablation mode (directory: vlmeval_results/<name>)")
    parser.add_argument("--skip_data_download", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    required = [
        project_root / "scripts" / "evaluate.py",
        project_root / "scripts" / "evaluate_vlmeval.sh",
        project_root / "src" / "hawk" / "vlmeval_adapter.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Place 3_knockouts.py and knockout_utils.py in the HAWK repository root. "
            "Missing required HAWK files:\n  " + "\n  ".join(missing)
        )

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.mode in {"ablation", "all"}:
        if args.layer is None or args.head is None:
            raise ValueError("--layer and --head are required when --mode is ablation or all")
        if args.layer < 0 or args.head < 0:
            raise ValueError("--layer and --head must be zero-based non-negative integers")

    gpu_ids = parse_gpu_ids(args.gpus)
    num_processes = args.num_processes
    if num_processes is not None and not 1 <= num_processes <= len(gpu_ids):
        raise ValueError("--num_processes must be between 1 and the number of --gpus")

    model_path = args.model_path.expanduser()
    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()
    if not args.dry_run and not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    results_root = project_root / "vlmeval_results"
    results_root.mkdir(parents=True, exist_ok=True)

    # Baseline mode: execute only the unmodified model and print the run name
    # that must be supplied later through --baseline_run_name.
    if args.mode == "baseline":
        baseline_name = args.run_name or _default_baseline_name(args.task, args.limit)
        baseline_dir = results_root / baseline_name
        if baseline_dir.exists() and any(baseline_dir.iterdir()):
            raise FileExistsError(
                f"Baseline output already exists and is not empty: {baseline_dir}\n"
                "Choose a different --run_name to avoid overwriting results."
            )

        print("=" * 72)
        print("HAWK baseline only")
        print(f"Task:       {args.task}")
        print(f"Model:      {model_path}")
        print(f"GPUs:       {','.join(gpu_ids)}")
        print("Pruning:    disabled (pruning_ratio=0.0, keep_ratio=1.0)")
        print(f"Run name:   {baseline_name}")
        print(f"Limit:      {args.limit if args.limit is not None else 'full dataset'}")
        print("=" * 72)

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
            accuracy = extract_realworldqa_accuracy(baseline_dir)
            metadata = {
                "task": args.task,
                "model_path": str(model_path),
                "pruning_ratio": 0.0,
                "limit": args.limit,
                "baseline_run_name": baseline_name,
                "baseline_result_dir": str(baseline_dir),
                "baseline_accuracy_percent": accuracy,
                "native_score_files": discover_score_files(baseline_dir),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            metadata_path = results_root / f"{baseline_name}_baseline_manifest.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("\nBaseline finished.")
            print(f"Baseline run name: {baseline_name}")
            print(f"Baseline directory: {baseline_dir}")
            print(f"Baseline manifest:  {metadata_path}")
            if accuracy is not None:
                print(f"Baseline Accuracy:  {accuracy:.2f}%")
            else:
                print("Baseline score CSV was not automatically recognized; inspect the native VLMEvalKit score files.")
            print("Reuse this exact run name with --baseline_run_name for all later head ablations.")
        else:
            print(f"[dry-run] Baseline run name: {baseline_name}")
        return

    # Ablation mode (or all mode): reuse an existing baseline, unless all was
    # explicitly requested, in which case create the baseline first below.
    if args.mode == "all":
        baseline_name = args.baseline_run_name or _default_baseline_name(args.task, args.limit)
        baseline_dir = results_root / baseline_name
        if baseline_dir.exists() and any(baseline_dir.iterdir()):
            raise FileExistsError(f"Baseline output already exists: {baseline_dir}; choose another --baseline_run_name")
        print(f"[1/2] Running baseline: {baseline_name}", flush=True)
        run_hawk_evaluation(
            project_root, task=args.task, model_path=model_path, gpu_ids=gpu_ids,
            run_name=baseline_name, limit=args.limit, num_processes=num_processes,
            skip_data_download=args.skip_data_download, dry_run=args.dry_run,
        )
    else:
        if not args.baseline_run_name:
            raise ValueError(
                "Ablation mode requires --baseline_run_name. Run --mode baseline first, "
                "then pass the exact baseline run name printed by that run."
            )
        baseline_name = args.baseline_run_name
        baseline_dir = results_root / baseline_name
        if not args.dry_run and not baseline_dir.is_dir():
            raise FileNotFoundError(
                f"Baseline result directory not found: {baseline_dir}\n"
                "Check --baseline_run_name. Do not rerun baseline for every head; point to the existing baseline."
            )
        if not args.dry_run:
            baseline_manifest = results_root / f"{baseline_name}_baseline_manifest.json"
            if baseline_manifest.is_file():
                saved = json.loads(baseline_manifest.read_text(encoding="utf-8"))
                mismatches = []
                if saved.get("task") != args.task:
                    mismatches.append(f"task: baseline={saved.get('task')} vs current={args.task}")
                saved_model = str(Path(saved.get("model_path", "")).expanduser().resolve())
                if saved_model != str(model_path.resolve()):
                    mismatches.append(f"model_path: baseline={saved_model} vs current={model_path.resolve()}")
                if saved.get("limit") != args.limit:
                    mismatches.append(f"limit: baseline={saved.get('limit')} vs current={args.limit}")
                if mismatches:
                    raise ValueError(
                        "The selected baseline does not match this ablation configuration:\n  "
                        + "\n  ".join(mismatches)
                    )
            else:
                print(
                    "WARNING: no baseline manifest was found. The result directory exists, "
                    "but task/model/limit compatibility cannot be verified. Use a baseline "
                    "created by --mode baseline for safer comparisons.",
                    flush=True,
                )

    ablated_name = args.run_name or _default_ablation_name(args.task, args.layer, args.head)
    ablated_dir = results_root / ablated_name
    if ablated_dir.exists() and any(ablated_dir.iterdir()):
        raise FileExistsError(f"Ablation output already exists and is not empty: {ablated_dir}; choose another --run_name")

    print("=" * 72)
    print("HAWK single Query Head zero ablation")
    print(f"Task:       {args.task}")
    print(f"Layer/Head: {args.layer}/{args.head} (zero-based)")
    print(f"Model:      {model_path}")
    print(f"GPUs:       {','.join(gpu_ids)}")
    print("Pruning:    disabled (pruning_ratio=0.0, keep_ratio=1.0)")
    print(f"Baseline:   {baseline_name} (reused; not rerun)")
    print(f"Ablation:   {ablated_name}")
    print("=" * 72)

    with tempfile.TemporaryDirectory(prefix="hawk_head_ablation_") as temp_dir:
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

    if not args.dry_run:
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
        print("\nAblation finished.")
        print(f"Baseline directory: {baseline_dir}")
        print(f"Ablated directory:  {ablated_dir}")
        print(f"Experiment manifest: {manifest_path}")
        if baseline_accuracy is not None:
            print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
        else:
            print("Baseline Accuracy: not parsed; native score artifacts are retained.")
        if ablated_accuracy is not None:
            print(f"Ablated Accuracy:  {ablated_accuracy:.2f}%")
        else:
            print("Ablated Accuracy:  not parsed; native score artifacts are retained.")
        if baseline_accuracy is not None and ablated_accuracy is not None:
            print(f"Accuracy drop:     {baseline_accuracy - ablated_accuracy:+.2f} percentage points")
    else:
        print("[dry-run] Ablation evaluation command prepared; baseline will not be rerun in ablation mode.")


if __name__ == "__main__":
    main()
