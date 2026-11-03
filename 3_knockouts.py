#!/usr/bin/env python3
"""Run a RealWorldQA baseline and a single Qwen2.5-VL Query Head zero ablation."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from knockout_utils import (
    extract_realworldqa_accuracy,
    make_run_prefix,
    parse_gpu_ids,
    run_hawk_evaluation,
    write_experiment_manifest,
    write_sitecustomize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HAWK's native RealWorldQA/VLMEvalKit scoring twice: unablated "
            "baseline and one zeroed Qwen2.5-VL Query Head. HAWK visual-token "
            "pruning is disabled for both runs."
        )
    )
    parser.add_argument("--task", choices=["realworldqa"], default="realworldqa")
    parser.add_argument("--layer", type=int, required=True, help="0-based Transformer layer index")
    parser.add_argument("--head", type=int, required=True, help="0-based Query Head index")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/Qwen2.5-VL-7B-Instruct"),
    )
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0 or 0,1,2,3")
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for a smoke test, e.g. 5")
    parser.add_argument("--run_name", default=None, help="Optional common prefix for the two result directories")
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

    if args.layer < 0 or args.head < 0:
        raise ValueError("--layer and --head are zero-based non-negative integers")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    gpu_ids = parse_gpu_ids(args.gpus)
    num_processes = args.num_processes
    if num_processes is not None and not 1 <= num_processes <= len(gpu_ids):
        raise ValueError("--num_processes must be between 1 and the number of --gpus")

    model_path = args.model_path.expanduser()
    if not model_path.is_absolute():
        model_path = (project_root / model_path).resolve()
    if not args.dry_run and not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    prefix = make_run_prefix(args.task, args.layer, args.head, args.run_name)
    baseline_name = f"{prefix}_baseline"
    ablated_name = f"{prefix}_L{args.layer:02d}_H{args.head:02d}_zero"
    baseline_dir = project_root / "vlmeval_results" / baseline_name
    ablated_dir = project_root / "vlmeval_results" / ablated_name

    print("=" * 72)
    print("HAWK RealWorldQA single-head ablation")
    print(f"Task:       {args.task}")
    print(f"Layer/Head: {args.layer}/{args.head} (zero-based)")
    print(f"Model:      {model_path}")
    print(f"GPUs:       {','.join(gpu_ids)}")
    print("Pruning:    disabled (pruning_ratio=0.0, keep_ratio=1.0)")
    print("Runs:       baseline -> head ablation")
    if args.limit:
        print(f"Limit:      {args.limit} samples (smoke test)")
    print("=" * 72)

    # The temporary sitecustomize is imported by each torchrun worker before
    # run_vlmeval.py imports HawkQwen2VLChat. It patches integration.configure_model
    # and leaves the HAWK repository source files untouched.
    with tempfile.TemporaryDirectory(prefix="hawk_head_ablation_") as temp_dir:
        patch_dir = Path(temp_dir)
        write_sitecustomize(patch_dir)

        print("\n[1/2] Running unablated baseline...", flush=True)
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

        print("\n[2/2] Running zero-ablation evaluation...", flush=True)
        run_hawk_evaluation(
            project_root,
            task=args.task,
            model_path=model_path,
            gpu_ids=gpu_ids,
            run_name=ablated_name,
            limit=args.limit,
            num_processes=num_processes,
            skip_data_download=True,  # baseline already prepared the dataset
            dry_run=args.dry_run,
            ablation=(args.layer, args.head),
            patch_dir=patch_dir,
        )

    manifest_path = project_root / "vlmeval_results" / f"{prefix}_experiment.json"
    if not args.dry_run:
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

    print("\nFinished.")
    print(f"Baseline result directory: {baseline_dir}")
    print(f"Ablated result directory:  {ablated_dir}")
    if not args.dry_run:
        print(f"Experiment manifest:      {manifest_path}")
        baseline_accuracy = extract_realworldqa_accuracy(baseline_dir)
        ablated_accuracy = extract_realworldqa_accuracy(ablated_dir)
        if baseline_accuracy is not None and ablated_accuracy is not None:
            print(f"Baseline RealWorldQA Accuracy: {baseline_accuracy:.2f}%")
            print(f"Ablated  RealWorldQA Accuracy: {ablated_accuracy:.2f}%")
            print(
                "Accuracy drop:               "
                f"{baseline_accuracy - ablated_accuracy:+.2f} percentage points"
            )
        else:
            print("VLMEvalKit completed, but an *_acc.csv with an Overall column was not found.")
            print("Inspect the native score artifacts listed in the experiment manifest.")


if __name__ == "__main__":
    main()
