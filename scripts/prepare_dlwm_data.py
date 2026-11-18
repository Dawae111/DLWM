#!/usr/bin/env python3
"""Build leakage-checked DLWM SFT manifests from official benchmark files."""

from __future__ import annotations

import argparse
import json
import pathlib
import string
from typing import Any, Dict, Iterable, List, Mapping

import yaml


TRAIN_SPLITS = {
    "gqa": {"train", "train_balanced"},
    "vqav2": {"train", "train2014"},
    "scienceqa": {"train"},
}


def read_json(path: pathlib.Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def records(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return (
            (
                {"id": key, **value}
                if isinstance(value, dict)
                else {"id": key, "value": value}
            )
            for key, value in payload.items()
        )
    raise ValueError("expected a JSON list or object")


def conversation_record(
    example_id: str,
    benchmark: str,
    split: str,
    image: str | None,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    prompt = f"<image>\n{question}" if image else question
    item: Dict[str, Any] = {
        "id": example_id,
        "benchmark": benchmark,
        "split": split,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
    }
    if image:
        item["image"] = image
    return item


def build_gqa(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    payload = read_json(pathlib.Path(spec["questions_file"]))
    output = []
    template = spec.get("image_template", "{image_id}.jpg")
    for row in records(payload):
        image_id = str(row.get("imageId", row.get("image_id", "")))
        image = str(
            pathlib.Path(spec["image_root"]) / template.format(image_id=image_id)
        )
        output.append(
            conversation_record(
                f"gqa:{row.get('questionId', row.get('id'))}",
                "gqa",
                str(spec["split"]),
                image,
                str(row["question"]),
                str(row["answer"]),
            )
        )
    return output


def build_vqav2(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    questions = read_json(pathlib.Path(spec["questions_file"]))["questions"]
    annotations = read_json(pathlib.Path(spec["annotations_file"]))["annotations"]
    annotations_by_id = {
        int(annotation["question_id"]): annotation for annotation in annotations
    }
    template = spec.get("image_template", "COCO_train2014_{image_id:012d}.jpg")
    output = []
    for row in questions:
        annotation = annotations_by_id[int(row["question_id"])]
        answer = annotation.get("multiple_choice_answer")
        if answer is None:
            answer = annotation["answers"][0]["answer"]
        image = str(
            pathlib.Path(spec["image_root"])
            / template.format(image_id=int(row["image_id"]))
        )
        output.append(
            conversation_record(
                f"vqav2:{row['question_id']}",
                "vqav2",
                str(spec["split"]),
                image,
                str(row["question"]),
                str(answer),
            )
        )
    return output


def build_scienceqa(
    spec: Mapping[str, Any],
    image_output: pathlib.Path,
) -> List[Dict[str, Any]]:
    if spec.get("source", "huggingface") == "huggingface":
        from datasets import load_dataset

        rows = load_dataset(
            spec.get("dataset", "derek-thomas/ScienceQA"),
            split=str(spec["split"]),
        )
    else:
        rows = records(read_json(pathlib.Path(spec["path"])))
    image_output.mkdir(parents=True, exist_ok=True)
    output = []
    for index, row in enumerate(rows):
        choices = list(row["choices"])
        question = (
            str(row["question"])
            + "\nOptions:\n"
            + "\n".join(
                f"{string.ascii_uppercase[choice_index]}. {choice}"
                for choice_index, choice in enumerate(choices)
            )
        )
        question += "\nAnswer with the option letter only."
        answer = row["answer"]
        if isinstance(answer, int):
            answer = string.ascii_uppercase[answer]
        image = row.get("image")
        image_path = None
        if image is not None:
            image_path = image_output / f"{row.get('id', index)}.png"
            if hasattr(image, "save"):
                image.save(image_path)
            else:
                image_path = pathlib.Path(str(image))
        output.append(
            conversation_record(
                f"scienceqa:{row.get('id', index)}",
                "scienceqa",
                str(spec["split"]),
                str(image_path) if image_path else None,
                question,
                str(answer),
            )
        )
    return output


def validate_spec(spec: Mapping[str, Any], purpose: str) -> None:
    name = str(spec["name"]).lower()
    split = str(spec["split"]).lower()
    if name == "mmvp" and purpose == "train":
        raise ValueError("MMVP is zero-shot only and cannot appear in training data")
    if purpose == "train":
        if name not in TRAIN_SPLITS:
            raise ValueError(f"unsupported DLWM training benchmark: {name}")
        if split not in TRAIN_SPLITS[name]:
            raise ValueError(
                f"{name} split {split!r} is not allowed for training; "
                f"expected one of {sorted(TRAIN_SPLITS[name])}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    purpose = str(config.get("purpose", "train")).lower()
    output_dir = pathlib.Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    split_manifest = {
        "purpose": purpose,
        "datasets": [],
        "mmvp_used_for_training": False,
    }
    for spec in config["datasets"]:
        validate_spec(spec, purpose)
        name = str(spec["name"]).lower()
        if name == "gqa":
            dataset = build_gqa(spec)
        elif name == "vqav2":
            dataset = build_vqav2(spec)
        elif name == "scienceqa":
            dataset = build_scienceqa(spec, output_dir / "scienceqa_images")
        else:
            raise ValueError(f"unsupported dataset builder: {name}")
        output_path = output_dir / f"{name}_{spec['split']}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(dataset, handle, ensure_ascii=False)
        meta.append({"data_path": str(output_path.resolve())})
        split_manifest["datasets"].append(
            {
                "name": name,
                "split": str(spec["split"]),
                "count": len(dataset),
                "path": str(output_path.resolve()),
            }
        )
    meta_path = output_dir / "train_dlwm.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)
    print(meta_path)


if __name__ == "__main__":
    main()
