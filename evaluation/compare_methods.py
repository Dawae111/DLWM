#!/usr/bin/env python3
"""Create a fair-comparison table from DLWM and baseline summary files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.summary) != len(args.name):
        raise ValueError("--summary and --name counts must match")
    rows = []
    for name, path in zip(args.name, args.summary):
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle)
        for benchmark in summary["benchmarks"]:
            rows.append(
                {
                    "method": name,
                    "benchmark": benchmark["benchmark"],
                    "score": benchmark["score"],
                    "question_accuracy": benchmark.get("question_accuracy"),
                    "average_rollout_steps": benchmark.get("average_rollout_steps"),
                    "average_active_worlds": benchmark.get("average_active_worlds"),
                    "average_physically_packed_worlds": benchmark.get(
                        "average_physically_packed_worlds"
                    ),
                    "packed_world_reduction_fraction": benchmark.get(
                        "packed_world_reduction_fraction"
                    ),
                    "peak_gpu_memory_gb": benchmark.get("peak_gpu_memory_gb"),
                    "tokens_per_second": benchmark.get("tokens_per_second"),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
