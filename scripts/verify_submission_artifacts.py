#!/usr/bin/env python3
"""Validate the result artifacts listed in the submission manifest."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
from typing import Any, Mapping

import yaml


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_path(value: Any, label: str) -> pathlib.Path:
    if not value:
        raise ValueError(f"{label} is missing")
    path = pathlib.Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts/submission_manifest.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest: Mapping[str, Any] = yaml.safe_load(handle)
    checkpoint = require_path(
        manifest["paper_result_checkpoint"].get("uri"),
        "paper_result_checkpoint.uri",
    )
    expected_hash = manifest["paper_result_checkpoint"].get("sha256")
    if checkpoint.is_file() and expected_hash:
        actual_hash = sha256(checkpoint)
        if actual_hash != expected_hash:
            raise ValueError(
                f"checkpoint SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
    for key in (
        "stage1_config",
        "stage2_config",
        "run_manifest",
        "trainer_state",
        "log_export",
    ):
        require_path(manifest["training"].get(key), f"training.{key}")
    require_path(manifest["evaluation"].get("config"), "evaluation.config")
    require_path(manifest["evaluation"].get("summary"), "evaluation.summary")
    predictions = manifest["evaluation"].get("prediction_files") or []
    if not predictions:
        raise ValueError("evaluation.prediction_files is empty")
    for index, value in enumerate(predictions):
        require_path(value, f"evaluation.prediction_files[{index}]")
    if not manifest["training"].get("git_commit"):
        raise ValueError("training.git_commit is missing")
    print("submission artifacts verified")


if __name__ == "__main__":
    main()
