#!/usr/bin/env python3
"""Run the complete two-stage DLWM protocol for multiple random seeds."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: pathlib.Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--work-dir", default="./output/dlwm_multiseed")
    parser.add_argument("--deepspeed", default="deepspeed")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = pathlib.Path(args.work_dir)
    stage1_base = load_yaml(args.stage1_config)
    stage2_base = load_yaml(args.stage2_config)
    eval_base = load_yaml(args.eval_config)
    summaries = []

    for seed in args.seeds:
        seed_root = root / f"seed_{seed}"
        config_root = seed_root / "configs"
        stage1 = dict(stage1_base)
        stage2 = dict(stage2_base)
        evaluation = dict(eval_base)
        stage1_dir = seed_root / "stage1"
        stage2_dir = seed_root / "stage2"
        eval_dir = seed_root / "evaluation"
        stage1.update(
            seed=seed, output_dir=str(stage1_dir), run_name=f"dlwm-stage1-{seed}"
        )
        stage2.update(
            seed=seed,
            checkpoint_name=str(stage1_dir),
            output_dir=str(stage2_dir),
            run_name=f"dlwm-stage2-{seed}",
        )
        evaluation.update(
            seed=seed,
            checkpoint=str(stage2_dir),
            output_dir=str(eval_dir),
        )
        stage1_path = config_root / "stage1.yaml"
        stage2_path = config_root / "stage2.yaml"
        eval_path = config_root / "eval.yaml"
        save_yaml(stage1_path, stage1)
        save_yaml(stage2_path, stage2)
        save_yaml(eval_path, evaluation)
        run(
            [args.deepspeed, "src/train/train_dlwm.py", str(stage1_path)],
            args.dry_run,
        )
        run(
            [args.deepspeed, "src/train/train_dlwm.py", str(stage2_path)],
            args.dry_run,
        )
        run(
            ["python", "evaluation/evaluation.py", "--config", str(eval_path)],
            args.dry_run,
        )
        summary_path = eval_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as handle:
                summaries.append({"seed": seed, **json.load(handle)})

    if summaries:
        benchmark_values: Dict[str, list[float]] = {}
        for summary in summaries:
            for benchmark in summary["benchmarks"]:
                benchmark_values.setdefault(benchmark["benchmark"], []).append(
                    float(benchmark["score"])
                )
        aggregate = {
            "seeds": args.seeds,
            "runs": summaries,
            "benchmarks": {
                name: {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "values": values,
                }
                for name, values in benchmark_values.items()
            },
        }
        root.mkdir(parents=True, exist_ok=True)
        with (root / "mean_std.json").open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2)


if __name__ == "__main__":
    main()
