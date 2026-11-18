"""Deterministic DLWM evaluation with controller-efficiency reporting.

The evaluator supports the paper benchmarks (GQA, VQAv2, ScienceQA and MMVP)
plus a configurable generic JSON/JSONL/Hugging Face adapter. It contains no
storage credentials or machine-specific paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import shlex
import string
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import yaml
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.model.qwen_dlwm_model import QwenWithDLWM


ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
VQA_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "id": "i'd",
    "ill": "i'll",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "lets": "let's",
    "mightnt": "mightn't",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "oclock": "o'clock",
    "shant": "shan't",
    "shed": "she'd",
    "shell": "she'll",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "shouldve": "should've",
    "somebodyd": "somebody'd",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someone'd": "someone'd",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "wed": "we'd",
    "well": "we'll",
    "were": "we're",
    "werent": "weren't",
    "weve": "we've",
    "whats": "what's",
    "wheres": "where's",
    "whod": "who'd",
    "wholl": "who'll",
    "whos": "who's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "wouldve": "would've",
    "yall": "y'all",
    "youd": "you'd",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
VQA_PUNCTUATION = (
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
)
VQA_PERIOD_STRIP = re.compile(r"(?<!\d)\.(?!\d)")
VQA_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def normalize_answer(answer: Any) -> str:
    """VQA-style normalization used by open-ended benchmark adapters."""
    text = str(answer).lower().strip()
    text = re.sub(r"(?<!\d)[.,!?;:/\"'()\[\]{}](?!\d)", " ", text)
    words = []
    for word in text.split():
        word = NUMBER_WORDS.get(word, word)
        if word not in ARTICLES:
            words.append(word)
    return " ".join(words)


def normalize_vqa_answer(answer: Any) -> str:
    """Port of the normalization rules used by the official VQA evaluator."""
    text = str(answer).replace("\n", " ").replace("\t", " ").strip().lower()
    for punctuation in VQA_PUNCTUATION:
        if (
            punctuation + " " in text
            or " " + punctuation in text
            or VQA_COMMA_STRIP.search(text)
        ):
            text = text.replace(punctuation, "")
        else:
            text = text.replace(punctuation, " ")
    text = VQA_PERIOD_STRIP.sub("", text)
    normalized = []
    for word in text.split():
        word = NUMBER_WORDS.get(word, word)
        if word in ARTICLES:
            continue
        normalized.append(VQA_CONTRACTIONS.get(word, word))
    return " ".join(normalized)


def extract_option(text: str) -> str:
    """Extract an A-Z multiple-choice answer without matching prose letters."""
    answer_section = re.search(
        r"<answer>\s*([A-Za-z])(?:\s*</answer>)?", text, flags=re.IGNORECASE
    )
    if answer_section:
        return answer_section.group(1).upper()
    patterns = (
        r"(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([A-Za-z])[\)\]]?",
        r"^\s*[\(\[]?([A-Za-z])[\)\].:]?(?:\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return normalize_answer(text)


def score_prediction(
    prediction: str,
    answers: Sequence[str],
    metric: str,
) -> float:
    if not answers:
        return 0.0
    if metric == "multiple_choice":
        predicted = extract_option(prediction)
        return float(predicted == extract_option(str(answers[0])))
    if metric == "vqa_soft":
        predicted = normalize_vqa_answer(prediction)
        normalized_answers = [normalize_vqa_answer(answer) for answer in answers]
        # Official VQA evaluation is leave-one-annotator-out: for each human
        # answer, agreement is counted among the other humans, then averaged.
        accuracies = []
        for index in range(len(normalized_answers)):
            matches = sum(
                predicted == answer
                for other_index, answer in enumerate(normalized_answers)
                if other_index != index
            )
            accuracies.append(min(1.0, matches / 3.0))
        return sum(accuracies) / len(accuracies)
    if metric == "gqa_accuracy":
        return float(str(prediction).strip().lower() == str(answers[0]).strip().lower())
    predicted = normalize_answer(prediction)
    normalized_answers = [normalize_answer(answer) for answer in answers]
    if metric == "exact_match":
        return float(predicted in normalized_answers)
    raise ValueError(f"unsupported metric: {metric}")


def _read_json_records(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if "data" in payload and isinstance(payload["data"], list):
        return payload["data"]
    if all(isinstance(value, dict) for value in payload.values()):
        return [{"id": key, **value} for key, value in payload.items()]
    raise ValueError(f"cannot infer records from {path}")


def _resolve_image(image: Any, image_root: Optional[str]) -> Any:
    if image is None or not isinstance(image, str) or not image_root:
        return image
    return image if os.path.isabs(image) else os.path.join(image_root, image)


def load_generic(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    source = spec.get("source", "json")
    if source == "huggingface":
        from datasets import load_dataset

        dataset = load_dataset(
            spec["dataset"],
            spec.get("subset"),
            split=spec.get("split", "test"),
        )
        rows: Iterable[Mapping[str, Any]] = dataset
    elif source in {"json", "jsonl"}:
        rows = _read_json_records(spec["path"])
    else:
        raise ValueError(f"unsupported generic source: {source}")

    columns = spec.get("columns", {})
    id_column = columns.get("id", "id")
    question_column = columns.get("question", "question")
    image_column = columns.get("image", "image")
    answers_column = columns.get("answers")
    answer_column = columns.get("answer", "answer")
    options_column = columns.get("options", "options")
    examples = []
    for index, row in enumerate(rows):
        question = str(row[question_column])
        options = row.get(options_column)
        if options:
            question += "\nOptions:\n" + "\n".join(
                f"{string.ascii_uppercase[i]}. {choice}"
                for i, choice in enumerate(options)
            )
        if answers_column and row.get(answers_column) is not None:
            raw_answers = row[answers_column]
            answers = raw_answers if isinstance(raw_answers, list) else [raw_answers]
        else:
            answers = [row[answer_column]]
        image = row.get(image_column)
        images = image if isinstance(image, list) else [image]
        examples.append(
            {
                "id": str(row.get(id_column, index)),
                "question": question,
                "images": [
                    _resolve_image(item, spec.get("image_root"))
                    for item in images
                    if item is not None
                ],
                "answers": [str(answer) for answer in answers],
                "category": row.get(columns.get("category", "category")),
            }
        )
    return examples


def load_gqa(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = _read_json_records(spec["questions_file"])
    examples = []
    for row in records:
        image_id = str(row.get("imageId", row.get("image_id", "")))
        image_name = spec.get("image_template", "{image_id}.jpg").format(
            image_id=image_id
        )
        examples.append(
            {
                "id": str(row.get("questionId", row.get("id"))),
                "question": row["question"],
                "images": [_resolve_image(image_name, spec.get("image_root"))],
                "answers": [str(row["answer"])],
                "category": (
                    row.get("types", {}).get("semantic")
                    if isinstance(row.get("types"), dict)
                    else row.get("category")
                ),
            }
        )
    return examples


def load_vqav2(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    with open(spec["questions_file"], encoding="utf-8") as handle:
        questions = json.load(handle)["questions"]
    with open(spec["annotations_file"], encoding="utf-8") as handle:
        annotations = json.load(handle)["annotations"]
    annotations_by_id = {
        str(annotation["question_id"]): annotation for annotation in annotations
    }
    template = spec.get("image_template", "COCO_val2014_{image_id:012d}.jpg")
    examples = []
    for row in questions:
        question_id = str(row["question_id"])
        annotation = annotations_by_id[question_id]
        answers = [
            item["answer"] if isinstance(item, dict) else item
            for item in annotation["answers"]
        ]
        image_name = template.format(image_id=int(row["image_id"]))
        examples.append(
            {
                "id": question_id,
                "question": row["question"],
                "images": [_resolve_image(image_name, spec.get("image_root"))],
                "answers": answers,
                "category": annotation.get("answer_type"),
            }
        )
    return examples


def load_scienceqa(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if spec.get("source", "huggingface") == "huggingface":
        from datasets import load_dataset

        rows = load_dataset(
            spec.get("dataset", "derek-thomas/ScienceQA"),
            split=spec.get("split", "test"),
        )
    else:
        rows = _read_json_records(spec["path"])
    examples = []
    for index, row in enumerate(rows):
        choices = row["choices"]
        question = (
            str(row["question"])
            + "\nOptions:\n"
            + "\n".join(
                f"{string.ascii_uppercase[i]}. {choice}"
                for i, choice in enumerate(choices)
            )
        )
        question += "\nAnswer with the option letter only."
        answer = row["answer"]
        if isinstance(answer, int):
            answer = string.ascii_uppercase[answer]
        image = row.get("image")
        images = (
            [] if image is None else [_resolve_image(image, spec.get("image_root"))]
        )
        examples.append(
            {
                "id": str(row.get("id", index)),
                "question": question,
                "images": images,
                "answers": [str(answer)],
                "category": row.get("subject"),
            }
        )
    return examples


def load_mmvp(spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    examples = []
    with open(spec["questions_file"], newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["Index"])
            question = (
                row["Question"]
                + "\nOptions:\n"
                + row["Options"].replace("(a)", "A.").replace("(b)", "B.")
                + "\nAnswer with the option letter only."
            )
            examples.append(
                {
                    "id": str(index),
                    "question": question,
                    "images": [_resolve_image(f"{index}.jpg", spec.get("image_root"))],
                    "answers": [extract_option(row["Correct Answer"])],
                    "category": None,
                }
            )
    return examples


ADAPTERS = {
    "generic": load_generic,
    "gqa": load_gqa,
    "vqav2": load_vqav2,
    "scienceqa": load_scienceqa,
    "mmvp": load_mmvp,
}

DEFAULT_METRICS = {
    "gqa": "gqa_accuracy",
    "vqav2": "vqa_soft",
    "scienceqa": "multiple_choice",
    "mmvp": "multiple_choice",
    "generic": "exact_match",
}


def create_messages(example: Mapping[str, Any]) -> List[Dict[str, Any]]:
    content = [{"type": "image", "image": image} for image in example["images"]]
    content.append({"type": "text", "text": example["question"]})
    return [{"role": "user", "content": content}]


class DLWMEvaluator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.checkpoint = config["checkpoint"]
        model_config = AutoConfig.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        for name, default in (
            ("lvr_head", False),
            ("lvr_head_type", "simple"),
            ("latent_end_token", False),
            ("enable_dlwm", True),
        ):
            if not hasattr(model_config, name):
                setattr(model_config, name, default)
        model_config.enable_dlwm = True
        overrides = config.get("dlwm", {})
        for name, value in overrides.items():
            setattr(model_config, f"dlwm_{name}", value)
        dtype_name = config.get("torch_dtype", "bfloat16")
        dtype = getattr(torch, dtype_name)
        self.model = QwenWithDLWM.from_pretrained(
            self.checkpoint,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=config.get("device_map", "auto"),
            attn_implementation=config.get("attn_implementation", "sdpa"),
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )

    @property
    def input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def cuda_devices(self) -> List[torch.device]:
        return sorted(
            {
                parameter.device
                for parameter in self.model.parameters()
                if parameter.device.type == "cuda"
            },
            key=lambda device: device.index or 0,
        )

    @torch.inference_mode()
    def predict(self, example: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise RuntimeError(
                "qwen-vl-utils is required for multimodal evaluation"
            ) from error

        method = self.config.get("method", {})
        method_type = method.get("type", "dlwm")
        prepared_example = dict(example)
        if method_type == "explicit_cot":
            prepared_example["question"] = (
                str(example["question"])
                + "\nThink step by step, then give only the final answer."
            )
        messages = create_messages(prepared_example)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs or None,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
        ).to(self.input_device)
        generation = self.config.get("generation", {})
        for device in self.cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
        for device in self.cuda_devices:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        generation_kwargs = {
            "max_new_tokens": generation.get("max_new_tokens", 128),
            "do_sample": False,
            "num_beams": int(generation.get("num_beams", 1)),
            "num_return_sequences": 1,
            "use_cache": True,
        }
        if method_type == "dlwm":
            generated = self.model.generate_with_dlwm(
                **inputs,
                **generation_kwargs,
                dlwm_max_steps=self.config.get("dlwm", {}).get("max_steps", 8),
            )
        elif method_type in {"direct", "explicit_cot"}:
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
                enable_dlwm=False,
            )
        else:
            raise ValueError("method.type must be dlwm, direct, or explicit_cot")
        for device in self.cuda_devices:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_memory_gb = max(
            (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                for device in self.cuda_devices
            ),
            default=0.0,
        )
        trimmed = [
            output[len(prompt_ids) :]
            for prompt_ids, output in zip(inputs.input_ids, generated)
        ]
        prediction = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        rollout = self.model.get_last_dlwm_statistics() if method_type == "dlwm" else {}
        if rollout.get("generation_rollouts"):
            for token_rollout in rollout["generation_rollouts"]:
                token_rollout["per_world_top_tokens"] = [
                    self.processor.tokenizer.convert_ids_to_tokens(token_ids)
                    for token_ids in token_rollout["per_world_top_token_ids"]
                ]
                token_rollout["aggregated_top_tokens"] = [
                    self.processor.tokenizer.convert_ids_to_tokens(token_id)
                    for token_id in token_rollout["aggregated_top_token_ids"]
                ]
        return {
            "prediction": prediction,
            "latency_seconds": elapsed,
            "peak_gpu_memory_gb": peak_memory_gb,
            "generated_tokens": int(trimmed[0].numel()),
            "dlwm": rollout,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (
        value != value or value in {float("inf"), float("-inf")}
    ):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def export_official_predictions(
    adapter_name: str,
    rows: Sequence[Mapping[str, Any]],
    benchmark_dir: str,
) -> str:
    """Write the native submission shape expected by each benchmark."""
    if adapter_name == "gqa":
        path = os.path.join(benchmark_dir, "official_predictions.json")
        payload = {str(row["id"]): str(row["prediction"]) for row in rows}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    elif adapter_name == "vqav2":
        path = os.path.join(benchmark_dir, "official_predictions.json")
        payload = [
            {
                "question_id": int(row["id"]),
                "answer": str(row["prediction"]),
            }
            for row in rows
        ]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    else:
        path = os.path.join(benchmark_dir, "official_predictions.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "prediction", "answer", "score"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "prediction": row["prediction"],
                        "answer": row["answers"][0] if row["answers"] else "",
                        "score": row["score"],
                    }
                )
    return path


def mmvp_pair_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    """Official MMVP: a pair is correct only when both questions are correct."""
    ordered = sorted(rows, key=lambda row: int(row["id"]))
    pairs = [
        ordered[index : index + 2]
        for index in range(0, len(ordered), 2)
        if len(ordered[index : index + 2]) == 2
    ]
    return sum(
        float(all(float(item["score"]) == 1.0 for item in pair)) for pair in pairs
    ) / max(len(pairs), 1)


def evaluate_benchmark(
    evaluator: DLWMEvaluator,
    spec: Mapping[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    adapter_name = spec.get("adapter", spec["name"].lower())
    if adapter_name not in ADAPTERS:
        raise ValueError(f"unknown adapter: {adapter_name}")
    examples = ADAPTERS[adapter_name](spec)
    limit = spec.get("limit")
    if limit is not None:
        examples = examples[: int(limit)]
    warmup_batches = int(
        spec.get("warmup_batches", evaluator.config.get("warmup_batches", 3))
    )
    for example in examples[:warmup_batches]:
        evaluator.predict(example)
    metric = spec.get("metric", DEFAULT_METRICS[adapter_name])
    benchmark_dir = os.path.join(output_dir, spec["name"])
    os.makedirs(benchmark_dir, exist_ok=True)
    predictions_path = os.path.join(benchmark_dir, "predictions.jsonl")
    completed: Dict[str, Dict[str, Any]] = {}
    if spec.get("resume", True) and os.path.exists(predictions_path):
        for row in _read_json_records(predictions_path):
            completed[str(row["id"])] = row

    rows = list(completed.values())
    file_mode = "a" if completed else "w"
    with open(predictions_path, file_mode, encoding="utf-8") as output:
        for example in tqdm(examples, desc=f"Evaluating {spec['name']}"):
            if example["id"] in completed:
                continue
            prediction = evaluator.predict(example)
            row = {
                "id": example["id"],
                "prediction": prediction["prediction"],
                "answers": example["answers"],
                "score": score_prediction(
                    prediction["prediction"], example["answers"], metric
                ),
                "category": example.get("category"),
                "latency_seconds": prediction["latency_seconds"],
                "peak_gpu_memory_gb": prediction["peak_gpu_memory_gb"],
                "generated_tokens": prediction["generated_tokens"],
                "dlwm": prediction["dlwm"],
            }
            row = _json_safe(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            rows.append(row)

    selected_ids = {example["id"] for example in examples}
    rows = [row for row in rows if str(row["id"]) in selected_ids]
    question_accuracy = sum(float(row["score"]) for row in rows) / max(len(rows), 1)
    score = question_accuracy
    pair_accuracy = None
    if adapter_name == "mmvp":
        pair_accuracy = mmvp_pair_accuracy(rows)
        score = pair_accuracy
    steps, worlds, packed_worlds = [], [], []

    def positive_numbers(value):
        if isinstance(value, list):
            for item in value:
                yield from positive_numbers(item)
        elif isinstance(value, (int, float)) and value > 0:
            yield float(value)

    for row in rows:
        rollout = row.get("dlwm") or {}
        if rollout.get("num_steps"):
            steps.extend(rollout["num_steps"])
        if rollout.get("avg_active_worlds"):
            worlds.extend(rollout["avg_active_worlds"])
        packed_worlds.extend(
            positive_numbers(rollout.get("packed_storage_history", []))
        )
    average_packed_worlds = sum(packed_worlds) / max(len(packed_worlds), 1)
    initial_worlds = int(getattr(evaluator.model.dlwm, "num_worlds", 1))
    metrics = {
        "benchmark": spec["name"],
        "metric": metric,
        "count": len(rows),
        "warmup_batches": warmup_batches,
        "score": score,
        "question_accuracy": question_accuracy,
        "average_rollout_steps": sum(steps) / max(len(steps), 1),
        "average_active_worlds": sum(worlds) / max(len(worlds), 1),
        "average_physically_packed_worlds": average_packed_worlds,
        "packed_world_reduction_fraction": (
            1.0 - average_packed_worlds / max(initial_worlds, 1)
            if packed_worlds
            else 0.0
        ),
        "average_latency_seconds": sum(float(row["latency_seconds"]) for row in rows)
        / max(len(rows), 1),
        "peak_gpu_memory_gb": max(
            (float(row.get("peak_gpu_memory_gb", 0.0)) for row in rows),
            default=0.0,
        ),
        "tokens_per_second": sum(int(row["generated_tokens"]) for row in rows)
        / max(sum(float(row["latency_seconds"]) for row in rows), 1e-12),
    }
    if pair_accuracy is not None:
        metrics["pair_accuracy"] = pair_accuracy
        metrics["pair_count"] = len(rows) // 2
    action_counts: Counter[str] = Counter()

    def action_names(value):
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], str):
                yield value[0]
            else:
                for item in value:
                    yield from action_names(item)

    for row in rows:
        for action_name in action_names((row.get("dlwm") or {}).get("actions", [])):
            action_counts[action_name] += 1
    if action_counts:
        total_actions = sum(action_counts.values())
        metrics["action_distribution"] = {
            action: {
                "count": count,
                "fraction": count / total_actions,
            }
            for action, count in sorted(action_counts.items())
        }
    category_scores: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("category") is not None:
            category_scores.setdefault(str(row["category"]), []).append(
                float(row["score"])
            )
    if category_scores:
        metrics["categories"] = {
            category: {
                "count": len(scores),
                "score": sum(scores) / len(scores),
            }
            for category, scores in sorted(category_scores.items())
        }
    with open(
        os.path.join(benchmark_dir, "metrics.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    official_path = export_official_predictions(
        adapter_name,
        rows,
        benchmark_dir,
    )
    official_command = spec.get("official_evaluator_command")
    if official_command:
        command = [
            part.format(predictions=official_path, benchmark_dir=benchmark_dir)
            for part in (
                shlex.split(official_command)
                if isinstance(official_command, str)
                else official_command
            )
        ]
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
        with open(
            os.path.join(benchmark_dir, "official_evaluator.log"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(completed.stdout)
            handle.write(completed.stderr)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to a DLWM evaluation YAML file"
    )
    parser.add_argument("--checkpoint", help="Override checkpoint in YAML")
    parser.add_argument(
        "--benchmark",
        action="append",
        help="Evaluate only this named benchmark (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.benchmark:
        wanted = {name.lower() for name in args.benchmark}
        config["benchmarks"] = [
            spec for spec in config["benchmarks"] if spec["name"].lower() in wanted
        ]
    torch.manual_seed(int(config.get("seed", 42)))
    output_dir = config.get("output_dir", "./evaluation/results/dlwm")
    os.makedirs(output_dir, exist_ok=True)
    evaluator = DLWMEvaluator(config)
    results = [
        evaluate_benchmark(evaluator, spec, output_dir) for spec in config["benchmarks"]
    ]
    summary = {
        "checkpoint": config["checkpoint"],
        "seed": int(config.get("seed", 42)),
        "method": config.get("method", {"type": "dlwm"}),
        "generation": config.get("generation", {}),
        "warmup_batches": int(config.get("warmup_batches", 3)),
        "dlwm": evaluator.model.get_dlwm_statistics(),
        "dlwm_parameter_count": sum(
            parameter.numel()
            for name, parameter in evaluator.model.named_parameters()
            if name.startswith("dlwm")
        ),
        "benchmarks": results,
    }
    with open(
        os.path.join(output_dir, "summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(summary), handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
