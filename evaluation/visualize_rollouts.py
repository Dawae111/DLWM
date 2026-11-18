#!/usr/bin/env python3
"""Render controller actions and per-world token predictions from evaluation logs."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def flatten_actions(value):
    if isinstance(value, list):
        if value and isinstance(value[0], str):
            yield value
        else:
            for item in value:
                yield from flatten_actions(item)


def render(row: Dict[str, Any], output_dir: pathlib.Path) -> pathlib.Path:
    rollout = row["dlwm"]
    token_rollouts = rollout.get("generation_rollouts", [])
    actions = list(flatten_actions(rollout.get("actions", [])))
    figure, (action_axis, token_axis) = plt.subplots(
        2,
        1,
        figsize=(max(9, len(token_rollouts) * 0.8), 6),
        constrained_layout=True,
    )
    action_axis.axis("off")
    action_text = " -> ".join(
        action[0]
        + (
            f"({action[1]},{action[2]})"
            if action[0] == "merge"
            else f"({action[1]})" if action[0] in {"expand", "stop"} else ""
        )
        for action in actions
    )
    action_axis.text(
        0.01,
        0.5,
        action_text or "no controller actions recorded",
        fontsize=11,
        va="center",
        wrap=True,
    )
    action_axis.set_title(
        f"{row['id']} | prediction: {row['prediction']} | score: {row['score']}"
    )

    for token_index, token_rollout in enumerate(token_rollouts):
        tokens = token_rollout.get("per_world_top_tokens", [])
        indices = token_rollout.get("per_world_indices", [[]])
        if indices and isinstance(indices[0], list):
            indices = indices[0]
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        for compact_index, token in enumerate(tokens):
            world_index = (
                indices[compact_index]
                if compact_index < len(indices) and indices[compact_index] >= 0
                else compact_index
            )
            token_axis.text(
                token_index,
                world_index,
                str(token),
                ha="center",
                va="center",
                fontsize=8,
                rotation=45,
            )
    token_axis.set_xlabel("generated token step")
    token_axis.set_ylabel("world id")
    token_axis.set_yticks(
        range(
            max(
                (
                    len(item.get("per_world_top_tokens", [[]])[0])
                    for item in token_rollouts
                ),
                default=1,
            )
        )
    )
    token_axis.grid(alpha=0.25)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{row['id']}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = set(args.ids or [])
    rendered = 0
    for row in read_jsonl(args.predictions):
        if wanted and str(row["id"]) not in wanted:
            continue
        print(render(row, pathlib.Path(args.output_dir)))
        rendered += 1
        if not wanted and rendered >= args.limit:
            break


if __name__ == "__main__":
    main()
