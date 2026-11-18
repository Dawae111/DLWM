#!/usr/bin/env python3
"""Generate matched Stage-1/Stage-2/evaluation configs for paper ablations."""

from __future__ import annotations

import argparse
import pathlib
from copy import deepcopy
from typing import Any, Dict

import yaml


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save(path: pathlib.Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output-dir", default="./output/ablation_configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load(args.matrix)
    stage1_base = load(matrix["stage1_config"])
    stage2_base = load(matrix["stage2_config"])
    eval_base = load(matrix["eval_config"])
    output_root = pathlib.Path(args.output_dir)
    for experiment in matrix["experiments"]:
        name = experiment["name"]
        experiment_root = output_root / name
        stage1 = deepcopy(stage1_base)
        stage2 = deepcopy(stage2_base)
        evaluation = deepcopy(eval_base)
        overrides = experiment.get("overrides", {})
        for key, value in overrides.items():
            stage1[key] = value
            stage2[key] = value
            if key.startswith("dlwm_"):
                evaluation.setdefault("dlwm", {})[key.removeprefix("dlwm_")] = value
        stage1["output_dir"] = str(experiment_root / "stage1")
        stage2["checkpoint_name"] = stage1["output_dir"]
        stage2["output_dir"] = str(experiment_root / "stage2")
        evaluation["checkpoint"] = stage2["output_dir"]
        evaluation["output_dir"] = str(experiment_root / "evaluation")
        save(experiment_root / "stage1.yaml", stage1)
        save(experiment_root / "stage2.yaml", stage2)
        save(experiment_root / "eval.yaml", evaluation)
    print(output_root)


if __name__ == "__main__":
    main()
