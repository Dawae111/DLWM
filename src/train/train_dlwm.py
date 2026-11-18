"""Train DLWM in supervised stage 1 or resource-aware RL stage 2.

Examples:
    deepspeed src/train/train_dlwm.py configs/train/dlwm_stage1_7b.yaml
    deepspeed src/train/train_dlwm.py configs/train/dlwm_stage2_rl_7b.yaml
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import string
import subprocess
import sys
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer, set_seed

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.model.qwen_dlwm_model import QwenWithDLWM
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.train_utils import safe_save_model_for_hf_trainer


def normalize_rule_answer(value: str) -> str:
    """Dataset-agnostic exact-match normalization used by Stage-2 rollouts."""
    text = str(value).lower().strip()
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if answer_match:
        text = answer_match.group(1)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


DLWM_CONFIG_FIELDS = (
    "dlwm_num_worlds",
    "dlwm_hidden_size",
    "dlwm_compute_budget",
    "dlwm_lambda_orth",
    "dlwm_lambda_steps",
    "dlwm_lambda_worlds",
    "dlwm_use_rl",
    "dlwm_max_steps",
    "dlwm_merge_mode",
    "dlwm_merge_fixed_beta",
    "dlwm_merge_temperature",
    "dlwm_merge_topk_pairs",
    "dlwm_expand_cost",
    "dlwm_stop_cost",
    "dlwm_merge_cost",
    "dlwm_answer_cost",
    "dlwm_dropout",
    "dlwm_reasoner_layers",
    "dlwm_reasoner_heads",
    "dlwm_reasoner_ffn_multiplier",
    "dlwm_world_generator_residual",
    "dlwm_diversity_regularizer",
    "dlwm_diversity_margin",
    "dlwm_enable_expand",
    "dlwm_enable_stop",
    "dlwm_enable_merge",
    "dlwm_prediction_chunk_size",
    "dlwm_aggregation",
)

RL_CONFIG_FIELDS = (
    "dlwm_stage",
    "dlwm_rl_lambda",
    "dlwm_rl_value_coef",
    "dlwm_rl_entropy_coef",
    "dlwm_rl_gamma",
    "dlwm_rl_baseline",
    "dlwm_rl_baseline_momentum",
    "dlwm_reward_mode",
    "dlwm_rollout_max_new_tokens",
)


def rank0_print(training_args: TrainingArguments, *values) -> None:
    if training_args.local_rank in (-1, 0):
        print(*values)


def write_run_manifest(
    model,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    parameter_summary: Dict[str, float],
) -> None:
    if training_args.local_rank not in (-1, 0):
        return
    output_dir = pathlib.Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    accelerators = (
        [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else []
    )
    payload = {
        "git_commit": git_commit,
        "model_arguments": asdict(model_args),
        "data_arguments": asdict(data_args),
        "training_arguments": training_args.to_sanitized_dict(),
        "resolved_model_config": model.config.to_dict(),
        "parameters": parameter_summary,
        "hardware": {
            "accelerators_visible": accelerators,
            "accelerator_count_visible": len(accelerators),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def parse_arguments() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) >= 2 and pathlib.Path(sys.argv[1]).suffix in {".yaml", ".yml"}:
        parsed = tuple(
            parser.parse_yaml_file(
                yaml_file=os.path.abspath(sys.argv[1]),
                allow_extra_keys=False,
            )
        )
        # DeepSpeed versions may append --local_rank to the script command.
        remaining = iter(sys.argv[2:])
        for argument in remaining:
            if argument.startswith("--local_rank="):
                parsed[2].local_rank = int(argument.split("=", 1)[1])
            elif argument == "--local_rank":
                parsed[2].local_rank = int(next(remaining))
            else:
                raise ValueError(
                    "A YAML config may only be followed by DeepSpeed's --local_rank"
                )
        return parsed
    if len(sys.argv) == 2 and pathlib.Path(sys.argv[1]).suffix == ".json":
        return tuple(
            parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1]),
                allow_extra_keys=False,
            )
        )
    return tuple(parser.parse_args_into_dataclasses())


def configure_model_config(
    config,
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> None:
    config.lvr_head = model_args.lvr_head
    config.lvr_head_type = model_args.lvr_head_type
    config.latent_end_token = model_args.latent_end_token
    config.enable_dlwm = model_args.enable_dlwm
    for name in DLWM_CONFIG_FIELDS:
        setattr(config, name, getattr(model_args, name))
    for name in RL_CONFIG_FIELDS:
        setattr(config, name, getattr(training_args, name))
    stage = training_args.dlwm_stage.lower()
    if stage not in {"stage1", "stage2", "rl", "stage2_rl"}:
        raise ValueError("dlwm_stage must be stage1 or stage2")
    if stage != "stage1":
        config.dlwm_use_rl = True


def configure_special_tokens(model, processor) -> None:
    tokens = ("<|lvr_start|>", "<|lvr|>", "<|lvr_latent_end|>", "<|lvr_end|>")
    processor.tokenizer.add_tokens(list(tokens), special_tokens=True)
    for token, config_name in zip(
        tokens,
        ("lvr_start_id", "lvr_id", "lvr_latent_end_id", "lvr_end_id"),
    ):
        setattr(
            model.config,
            config_name,
            processor.tokenizer.convert_tokens_to_ids(token),
        )
    if model.get_input_embeddings().num_embeddings < len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))


def configure_trainable_parameters(
    model: QwenWithDLWM,
    training_args: TrainingArguments,
) -> None:
    stage2 = training_args.dlwm_stage.lower() != "stage1"
    if stage2 and training_args.dlwm_stage2_controller_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        # Algorithm 1 updates only omega in stage 2. Beta uses the stage-1
        # trained world scorer utilities and is therefore frozen here.
        for parameter in model.dlwm.policy.parameters():
            parameter.requires_grad = True
        return

    if training_args.freeze_llm:
        for parameter in model.model.parameters():
            parameter.requires_grad = False
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = False
    if training_args.freeze_vision_tower:
        for parameter in model.visual.parameters():
            parameter.requires_grad = False
    elif training_args.freeze_merger and hasattr(model.visual, "merger"):
        for parameter in model.visual.merger.parameters():
            parameter.requires_grad = False


def trainable_parameter_summary(model: torch.nn.Module) -> Dict[str, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / max(total, 1),
    }


def build_data_module(
    model_args: ModelArguments,
    data_args: DataArguments,
    processor,
):
    if data_args.dataset_format == "sft":
        from src.dataset.sft_dataset import (
            DataCollatorForSupervisedDataset as Collator,
            SupervisedDataset as Dataset,
        )
    elif data_args.dataset_format == "lvr_bbox":
        from src.dataset.lvr_sft_dataset import (
            DataCollatorForSupervisedDatasetLVR as Collator,
            SupervisedDatasetLVR as Dataset,
        )
    else:
        raise ValueError("dataset_format must be sft or lvr_bbox")

    if not data_args.data_path:
        raise ValueError("data_path is required")
    train_dataset = Dataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_args.model_id,
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = Dataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            data_args=data_args,
            model_id=model_args.model_id,
        )
    collator = Collator(pad_token_id=processor.tokenizer.pad_token_id)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }


class DLWMTrainer(Trainer):
    """Trainer that switches from SFT to on-policy REINFORCE in stage 2."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_dlwm_metrics: Dict[str, float] = {}

    @property
    def is_stage2(self) -> bool:
        return self.args.dlwm_stage.lower() != "stage1"

    def _build_prompt_batch(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Remove supervised answer tokens while preserving multimodal tensors."""
        labels = inputs["labels"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prefixes = []
        lengths = []
        for row in range(input_ids.size(0)):
            supervised = labels[row].ne(-100)
            if supervised.any():
                length = int(supervised.to(torch.int64).argmax().item())
            else:
                length = int(attention_mask[row].sum().item())
            length = max(length, 1)
            prefixes.append(input_ids[row, :length])
            lengths.append(length)
        max_length = max(lengths)
        pad_id = int(self.processing_class.tokenizer.pad_token_id)
        prompt_ids = input_ids.new_full((input_ids.size(0), max_length), pad_id)
        prompt_mask = attention_mask.new_zeros((input_ids.size(0), max_length))
        # Left padding ensures the final prompt token is the generation position
        # for every row, including mixed-length batches.
        for row, prefix in enumerate(prefixes):
            prompt_ids[row, -prefix.numel() :] = prefix
            prompt_mask[row, -prefix.numel() :] = 1

        prompt_inputs = {
            key: value
            for key, value in inputs.items()
            if key
            not in {
                "labels",
                "input_ids",
                "attention_mask",
                "position_ids",
                "cache_position",
                "lvr_tokens",
            }
        }
        prompt_inputs["input_ids"] = prompt_ids
        prompt_inputs["attention_mask"] = prompt_mask
        return prompt_inputs, torch.tensor(
            lengths, dtype=torch.long, device=input_ids.device
        )

    @torch.no_grad()
    def _generate_stage2_rewards(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one controller trajectory, generate, and score final answers."""
        unwrapped = self.accelerator.unwrap_model(model)
        prompt_inputs, _ = self._build_prompt_batch(inputs)
        max_new_tokens = int(self.args.dlwm_rollout_max_new_tokens)
        generated = unwrapped.generate_with_dlwm(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            dlwm_max_steps=int(unwrapped.config.dlwm_max_steps),
            dlwm_sample_actions=bool(self.args.dlwm_sample_actions),
        )
        prompt_width = prompt_inputs["input_ids"].size(1)
        completions = generated[:, prompt_width:]
        predictions = self.processing_class.tokenizer.batch_decode(
            completions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        gold_ids = [row[row.ne(-100)] for row in inputs["labels"]]
        gold_answers = self.processing_class.tokenizer.batch_decode(
            gold_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        rewards = torch.tensor(
            [
                float(normalize_rule_answer(prediction) == normalize_rule_answer(gold))
                for prediction, gold in zip(predictions, gold_answers)
            ],
            dtype=torch.float32,
            device=inputs["input_ids"].device,
        )

        statistics = unwrapped.get_last_dlwm_statistics()
        sampled = statistics.get("action_indices")
        masks = statistics.get("trajectory_mask")
        if sampled is None or masks is None:
            raise RuntimeError("Stage-2 rollout did not expose controller actions")
        sampled_actions = torch.as_tensor(
            sampled,
            dtype=torch.long,
            device=inputs["input_ids"].device,
        )
        trajectory_mask = torch.as_tensor(
            masks,
            dtype=torch.bool,
            device=inputs["input_ids"].device,
        )
        if sampled_actions.ndim == 1:
            sampled_actions = sampled_actions.unsqueeze(0)
            trajectory_mask = trajectory_mask.unsqueeze(0)
        answer_index = unwrapped.dlwm.policy.answer_action_index
        max_steps = int(unwrapped.config.dlwm_max_steps)
        replay = torch.full(
            (sampled_actions.size(0), max_steps),
            answer_index,
            dtype=torch.long,
            device=sampled_actions.device,
        )
        width = min(sampled_actions.size(1), max_steps)
        replay[:, :width] = torch.where(
            trajectory_mask[:, :width],
            sampled_actions[:, :width],
            torch.full_like(sampled_actions[:, :width], answer_index),
        )
        return rewards, replay

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        del num_items_in_batch
        labels = inputs["labels"]
        model_inputs = dict(inputs)
        # Only the legacy bbox-LVR forward consumes this indexing metadata.
        model_inputs.pop("lvr_tokens", None)
        on_policy = (
            self.is_stage2
            and self.args.dlwm_reward_mode == "generated_exact_match"
            and model.training
        )
        unwrapped = self.accelerator.unwrap_model(model)
        was_training = unwrapped.training
        external_rewards = None
        replay_actions = None
        if on_policy:
            # Eval mode disables dropout for both rollout and differentiable
            # replay, so the sampled action probabilities are exactly replayed.
            unwrapped.eval()
            external_rewards, replay_actions = self._generate_stage2_rewards(
                model, model_inputs
            )
        try:
            outputs = model(
                **model_inputs,
                enable_dlwm=True,
                dlwm_use_rl=self.is_stage2,
                dlwm_sample_actions=(
                    self.args.dlwm_sample_actions and model.training
                    if self.is_stage2 and not on_policy
                    else False
                ),
                dlwm_action_sequence=replay_actions,
                dlwm_external_rewards=external_rewards,
                dlwm_update_baseline=on_policy,
                dlwm_stage=self.args.dlwm_stage,
            )
        finally:
            if was_training:
                unwrapped.train()
        loss_dict = outputs.get("dlwm_loss_dict")
        if loss_dict is None:
            loss_dict = model.compute_loss(
                outputs,
                labels,
                stage=self.args.dlwm_stage,
            )
        results = outputs.get("dlwm_results", {})
        self._last_dlwm_metrics = {
            "dlwm/task_loss": float(loss_dict["task_loss"].detach()),
            "dlwm/orth_loss": float(loss_dict["orth_loss"].detach()),
            "dlwm/policy_loss": float(loss_dict["policy_loss"].detach()),
            "dlwm/value_loss": float(loss_dict["value_loss"].detach()),
            "dlwm/entropy": float(loss_dict["entropy"].detach()),
            "dlwm/reward": float(loss_dict["reward"].detach()),
        }
        if results:
            self._last_dlwm_metrics.update(
                {
                    "dlwm/steps": float(results["num_steps"].float().mean().detach()),
                    "dlwm/active_worlds": float(
                        results["num_active_worlds"].float().mean().detach()
                    ),
                    "dlwm/remaining_budget": float(
                        results["remaining_budget"].float().mean().detach()
                    ),
                }
            )
        if return_outputs:
            return loss_dict["loss"], outputs
        return loss_dict["loss"]

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        logs.update(self._last_dlwm_metrics)
        super().log(logs, *args, **kwargs)


def main() -> None:
    model_args, data_args, training_args = parse_arguments()
    if not model_args.enable_dlwm:
        raise ValueError("train_dlwm.py requires enable_dlwm=true")
    set_seed(training_args.seed)
    model_path = training_args.checkpoint_name or model_args.model_id
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else torch.bfloat16 if training_args.bf16 else torch.float32
    )

    config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    configure_model_config(config, model_args, training_args)
    model = QwenWithDLWM.from_pretrained(
        model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation=(
            "sdpa" if training_args.disable_flash_attn2 else "flash_attention_2"
        ),
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )
    configure_special_tokens(model, processor)
    configure_trainable_parameters(model, training_args)
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # Multimodal fields are consumed by the model even though they do not all
    # appear in the static forward signature of upstream Transformers.
    training_args.remove_unused_columns = False

    summary = trainable_parameter_summary(model)
    write_run_manifest(model, model_args, data_args, training_args, summary)
    rank0_print(
        training_args,
        f"DLWM {training_args.dlwm_stage}: "
        f"{summary['trainable']:,}/{summary['total']:,} trainable "
        f"({summary['trainable_percent']:.2f}%)",
    )
    data_module = build_data_module(model_args, data_args, processor)
    trainer = DLWMTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )
    resume = bool(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
    if training_args.should_save:
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
