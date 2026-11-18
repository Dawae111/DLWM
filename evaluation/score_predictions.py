#!/usr/bin/env python3
"""Score predictions from external baseline implementations with DLWM metrics."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from evaluation import (  # type: ignore
        ADAPTERS,
        DEFAULT_METRICS,
        mmvp_pair_accuracy,
        score_prediction,
    )
else:
    from .evaluation import (
        ADAPTERS,
        DEFAULT_METRICS,
        mmvp_pair_accuracy,
        score_prediction,
    )


def load_predictions(path: str) -> Dict[str, str]:
    prediction_path = pathlib.Path(path)
    if prediction_path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return {
            str(row.get("id", row.get("question_id"))): str(
                row.get("prediction", row.get("answer", row.get("text", "")))
            )
            for row in rows
        }
    payload: Any = json.loads(prediction_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    return {
        str(row.get("id", row.get("question_id"))): str(
            row.get("prediction", row.get("answer", row.get("text", "")))
        )
        for row in payload
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="BENCHMARK=path/to/predictions.json[l]",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    prediction_paths = dict(item.split("=", 1) for item in args.prediction)
    results = []
    for spec in config["benchmarks"]:
        name = spec["name"]
        if name not in prediction_paths:
            continue
        adapter_name = spec.get("adapter", name.lower())
        examples = ADAPTERS[adapter_name](spec)
        predictions = load_predictions(prediction_paths[name])
        metric = spec.get("metric", DEFAULT_METRICS[adapter_name])
        rows = []
        missing = []
        for example in examples:
            prediction = predictions.get(str(example["id"]))
            if prediction is None:
                missing.append(str(example["id"]))
                continue
            rows.append(
                {
                    "id": example["id"],
                    "prediction": prediction,
                    "score": score_prediction(
                        prediction,
                        example["answers"],
                        metric,
                    ),
                }
            )
        question_accuracy = sum(row["score"] for row in rows) / max(len(rows), 1)
        result = {
            "benchmark": name,
            "metric": metric,
            "count": len(rows),
            "missing": len(missing),
            "question_accuracy": question_accuracy,
            "score": question_accuracy,
        }
        if adapter_name == "mmvp":
            result["pair_accuracy"] = mmvp_pair_accuracy(rows)
            result["score"] = result["pair_accuracy"]
        results.append(result)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"benchmarks": results}, indent=2),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
