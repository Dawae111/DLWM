"""Qwen2.5-VL integration for Diverse Latent World Models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

from src.model.dlwm_modules import DLWMReasoningModule
from src.model.qwen_lvr_model import QwenWithLVR


class QwenWithDLWM(QwenWithLVR):
    """Qwen with an executable multi-world controller and output aggregation."""

    def __init__(self, config):
        # QwenWithLVR predates DLWM and accesses these attributes directly.
        for name, default in (
            ("lvr_head", False),
            ("lvr_head_type", "simple"),
            ("latent_end_token", False),
            ("enable_dlwm", False),
        ):
            if not hasattr(config, name):
                setattr(config, name, default)
        super().__init__(config)
        self._last_dlwm_statistics: Dict[str, Any] = {}
        self._dlwm_rollout_history = []
        self._collect_dlwm_rollouts = False
        self._dlwm_generation_cache: Optional[Dict[str, Any]] = None
        self._dlwm_generation_controller_calls = 0
        self._dlwm_generation_num_beams = 1
        self._dlwm_generation_row_to_world: Optional[torch.Tensor] = None
        self._dlwm_generation_beam_reorders = 0
        if config.enable_dlwm:
            self._init_dlwm(config)

    def _init_dlwm(self, config) -> None:
        """Initialize all DLWM modules and persist every architectural option."""
        if hasattr(self, "dlwm"):
            return
        options = {
            "num_worlds": getattr(config, "dlwm_num_worlds", 4),
            "compute_budget": getattr(config, "dlwm_compute_budget", 10.0),
            "lambda_orth": getattr(config, "dlwm_lambda_orth", 0.1),
            "lambda_steps": getattr(config, "dlwm_lambda_steps", 0.05),
            "lambda_worlds": getattr(config, "dlwm_lambda_worlds", 0.1),
            "merge_mode": getattr(config, "dlwm_merge_mode", "score_softmax"),
            "merge_fixed_beta": getattr(config, "dlwm_merge_fixed_beta", 0.5),
            "merge_temperature": getattr(config, "dlwm_merge_temperature", 1.0),
            "merge_topk_pairs": getattr(config, "dlwm_merge_topk_pairs", 1),
            "expand_cost": getattr(config, "dlwm_expand_cost", 1.0),
            "stop_cost": getattr(config, "dlwm_stop_cost", 0.1),
            "merge_cost": getattr(config, "dlwm_merge_cost", 0.5),
            "answer_cost": getattr(config, "dlwm_answer_cost", 0.0),
            "dropout": getattr(config, "dlwm_dropout", 0.1),
            "reasoner_layers": getattr(config, "dlwm_reasoner_layers", 6),
            "reasoner_heads": getattr(config, "dlwm_reasoner_heads", 8),
            "reasoner_ffn_multiplier": getattr(
                config, "dlwm_reasoner_ffn_multiplier", 4.0
            ),
            "world_generator_residual": getattr(
                config, "dlwm_world_generator_residual", False
            ),
            "diversity_regularizer": getattr(
                config, "dlwm_diversity_regularizer", "orthogonal"
            ),
            "diversity_margin": getattr(config, "dlwm_diversity_margin", 0.2),
            "enable_expand": getattr(config, "dlwm_enable_expand", True),
            "enable_stop": getattr(config, "dlwm_enable_stop", True),
            "enable_merge": getattr(config, "dlwm_enable_merge", True),
        }
        backbone_hidden_size = config.get_text_config().hidden_size
        self.dlwm = DLWMReasoningModule(
            hidden_size=(
                getattr(config, "dlwm_hidden_size", None) or backbone_hidden_size
            ),
            **options,
        )
        options["merge_mode"] = self.dlwm.world_merger.mode
        latent_hidden_size = self.dlwm.hidden_size
        self.dlwm_input_projection = nn.Linear(
            backbone_hidden_size,
            latent_hidden_size,
            bias=False,
        )
        self.dlwm_output_projection = nn.Linear(
            latent_hidden_size,
            backbone_hidden_size,
            bias=False,
        )
        self.dlwm_prediction_norm = nn.LayerNorm(backbone_hidden_size)
        self.register_buffer(
            "dlwm_reward_baseline", torch.zeros((), dtype=torch.float32)
        )
        self.register_buffer(
            "dlwm_reward_baseline_initialized", torch.zeros((), dtype=torch.bool)
        )
        config.enable_dlwm = True
        for name, value in options.items():
            setattr(config, f"dlwm_{name}", value)
        # Keep the historical spelling used by released configs.
        config.dlwm_num_worlds = options["num_worlds"]
        config.dlwm_compute_budget = options["compute_budget"]
        config.dlwm_hidden_size = latent_hidden_size
        config.dlwm_merge_beta_definition = self.dlwm.world_merger.beta_definition

    @staticmethod
    def _select_prompt_positions(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Locate the final prompt token without consuming supervised answers."""
        batch_size, sequence_length, _ = hidden_states.shape
        device = hidden_states.device
        positions = torch.full(
            (batch_size,), sequence_length - 1, dtype=torch.long, device=device
        )
        if labels is not None and labels.ndim == 2:
            supervised = labels.ne(-100)
            has_answer = supervised.any(dim=1)
            first_answer = supervised.to(torch.int64).argmax(dim=1)
            prompt_end = (first_answer - 1).clamp_min(0)
            positions = torch.where(has_answer, prompt_end, positions)
        elif (
            attention_mask is not None
            and attention_mask.ndim == 2
            and attention_mask.size(1) == sequence_length
        ):
            offsets = torch.arange(
                sequence_length, device=device, dtype=torch.long
            ).unsqueeze(0)
            positions = (
                torch.where(
                    attention_mask.to(device=device, dtype=torch.bool),
                    offsets,
                    torch.full_like(offsets, -1),
                )
                .max(dim=1)
                .values.clamp_min(0)
            )
        return positions.clamp(0, sequence_length - 1)

    # Backward-compatible public spelling; semantics are intentionally corrected.
    _select_reasoning_positions = _select_prompt_positions

    @staticmethod
    def _prediction_positions(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Return every state position whose next token is supervised.

        At inference there is exactly one position: the final non-padding state.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        device = hidden_states.device
        if labels is not None:
            if labels.shape[:2] != (batch_size, sequence_length):
                raise ValueError("labels must align with hidden states")
            supervised_next = labels[:, 1:].ne(-100)
            offsets = torch.arange(sequence_length - 1, device=device)
            return [offsets[supervised_next[row]] for row in range(batch_size)]
        if attention_mask is None or attention_mask.size(1) != sequence_length:
            return [
                torch.tensor([sequence_length - 1], dtype=torch.long, device=device)
                for _ in range(batch_size)
            ]
        offsets = torch.arange(sequence_length, device=device).unsqueeze(0)
        last = (
            torch.where(
                attention_mask.to(device=device, dtype=torch.bool),
                offsets,
                torch.full_like(offsets, -1),
            )
            .max(dim=1)
            .values.clamp_min(0)
        )
        return [last[row : row + 1] for row in range(batch_size)]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        enable_dlwm: Optional[bool] = None,
        dlwm_use_rl: Optional[bool] = None,
        dlwm_max_steps: Optional[int] = None,
        dlwm_sample_actions: Optional[bool] = None,
        dlwm_stage: Optional[str] = None,
        dlwm_action_sequence: Optional[torch.Tensor] = None,
        dlwm_external_rewards: Optional[torch.Tensor] = None,
        dlwm_update_baseline: Optional[bool] = None,
        lvr_mode_switch: Optional[torch.Tensor] = None,
        last_position_hidden_state: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        enable_dlwm = (
            getattr(self.config, "enable_dlwm", False)
            if enable_dlwm is None
            else enable_dlwm
        )
        dlwm_use_rl = (
            getattr(self.config, "dlwm_use_rl", False)
            if dlwm_use_rl is None
            else dlwm_use_rl
        )
        dlwm_max_steps = (
            getattr(self.config, "dlwm_max_steps", 8)
            if dlwm_max_steps is None
            else dlwm_max_steps
        )
        dlwm_stage = dlwm_stage or getattr(self.config, "dlwm_stage", "stage1")
        if enable_dlwm and not hasattr(self, "dlwm"):
            raise RuntimeError(
                "DLWM was requested at forward time but was not initialized. "
                "Set config.enable_dlwm=True before from_pretrained()."
            )

        parent_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            # DLWM recomputes CE after replacing the selected logits. Avoid
            # retaining the upstream loss graph over the pre-aggregation logits.
            labels=None if enable_dlwm else labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True if enable_dlwm else output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        # These arguments are accepted by the LVR monkey-patched forward, but
        # not by the stock Transformers Qwen forward.
        if lvr_mode_switch is not None:
            parent_kwargs["lvr_mode_switch"] = lvr_mode_switch
        if last_position_hidden_state is not None:
            parent_kwargs["last_position_hidden_state"] = last_position_hidden_state
        parent_kwargs.update(kwargs)
        outputs = super().forward(**parent_kwargs)

        if enable_dlwm:
            outputs = self._apply_dlwm(
                outputs,
                attention_mask=attention_mask,
                labels=labels,
                use_rl=dlwm_use_rl,
                max_steps=dlwm_max_steps,
                sample_actions=dlwm_sample_actions,
                action_sequence=dlwm_action_sequence,
            )
            if labels is not None:
                loss_dict = self.compute_loss(
                    outputs,
                    labels,
                    stage=dlwm_stage,
                    external_rewards=dlwm_external_rewards,
                    update_baseline=dlwm_update_baseline,
                )
                outputs.loss = loss_dict["loss"]
                outputs["dlwm_loss_dict"] = loss_dict

        if return_dict:
            return outputs
        return outputs.to_tuple()

    def _aggregate_world_logits(
        self,
        per_world_logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate compact active-world predictions without materializing K."""
        if per_world_logits.ndim != 3:
            raise ValueError(
                "per_world_logits must have shape [positions, worlds, vocab]"
            )
        if weights.ndim != 1 or weights.numel() != per_world_logits.size(1):
            raise ValueError("weights must align with compact worlds")
        aggregation = getattr(self.config, "dlwm_aggregation", "probability_mixture")
        if aggregation == "mean_pooling":
            weights = torch.full_like(weights, 1.0 / max(weights.numel(), 1))
            aggregation = "probability_mixture"
        elif aggregation == "max_pooling":
            selected = int(weights.argmax().item())
            log_probs = F.log_softmax(per_world_logits.float(), dim=-1)
            return per_world_logits[:, selected], log_probs
        elif aggregation == "last_world":
            log_probs = F.log_softmax(per_world_logits.float(), dim=-1)
            return per_world_logits[:, -1], log_probs

        log_probs = F.log_softmax(per_world_logits.float(), dim=-1)
        if aggregation == "probability_mixture":
            log_weights = weights.float().clamp_min(1e-12).log()
            aggregated = torch.logsumexp(
                log_probs + log_weights[None, :, None],
                dim=1,
            ).to(per_world_logits.dtype)
        elif aggregation == "logit_mixture":
            aggregated = torch.sum(
                per_world_logits * weights[None, :, None],
                dim=1,
            )
        else:
            raise ValueError(
                "dlwm_aggregation must be probability_mixture, logit_mixture, "
                "mean_pooling, max_pooling, or last_world"
            )
        return aggregated, log_probs

    def _predict_sample_positions(
        self,
        token_states: torch.Tensor,
        joint_repr: torch.Tensor,
        active_worlds: torch.Tensor,
        active_world_ids: torch.Tensor,
        active_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict all requested answer positions using only surviving worlds."""
        if not active_world_ids.numel():
            raise RuntimeError("DLWM must retain at least one answer world")
        if active_worlds.size(0) != active_world_ids.numel():
            raise ValueError("active worlds and stable IDs must align")
        if active_weights.shape != active_world_ids.shape:
            raise ValueError("active weights and stable IDs must align")
        active_weights = active_weights / active_weights.sum().clamp_min(1e-12)
        chunk_size = max(
            int(getattr(self.config, "dlwm_prediction_chunk_size", 8)),
            1,
        )
        aggregated_chunks: List[torch.Tensor] = []
        final_world_logits: Optional[torch.Tensor] = None
        final_log_probs: Optional[torch.Tensor] = None
        world_delta = self.dlwm_output_projection(
            active_worlds - joint_repr.unsqueeze(0)
        )
        for start in range(0, token_states.size(0), chunk_size):
            token_chunk = token_states[start : start + chunk_size]
            contextual_worlds = token_chunk[:, None, :] + world_delta[None, :, :]
            contextual_worlds = self.dlwm_prediction_norm(contextual_worlds)
            chunk_logits = self.lm_head(contextual_worlds)
            aggregated, log_probs = self._aggregate_world_logits(
                chunk_logits,
                active_weights,
            )
            aggregated_chunks.append(aggregated)
            final_world_logits = chunk_logits[-1]
            final_log_probs = log_probs[-1]
        if final_world_logits is None or final_log_probs is None:
            raise RuntimeError("at least one prediction position is required")
        return (
            torch.cat(aggregated_chunks, dim=0),
            final_world_logits,
            final_log_probs,
            active_world_ids,
        )

    def _apply_dlwm(
        self,
        outputs,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        use_rl: bool,
        max_steps: int,
        sample_actions: Optional[bool],
        action_sequence: Optional[torch.Tensor],
    ):
        if outputs.hidden_states is None:
            raise RuntimeError("DLWM requires output_hidden_states=True")
        last_hidden = outputs.hidden_states[-1]
        batch_size = last_hidden.size(0)
        controller_reused = False
        if self._collect_dlwm_rollouts and self._dlwm_generation_cache is not None:
            cache = self._dlwm_generation_cache
            joint_repr = cache["joint_repr"]
            dlwm_results = dict(cache["results"])
            row_to_world = self._dlwm_generation_row_to_world
            if row_to_world is None or row_to_world.numel() != batch_size:
                raise RuntimeError(
                    "beam batch changed without reordering the persistent DLWM cache"
                )
            controller_reused = True
        else:
            all_prompt_positions = self._select_prompt_positions(
                last_hidden, attention_mask, labels
            )
            if self._collect_dlwm_rollouts:
                num_beams = self._dlwm_generation_num_beams
                if batch_size % num_beams:
                    raise RuntimeError(
                        "expanded generation batch is not divisible by num_beams"
                    )
                controller_rows = torch.arange(
                    0,
                    batch_size,
                    num_beams,
                    device=last_hidden.device,
                )
                row_to_world = torch.arange(
                    controller_rows.numel(),
                    device=last_hidden.device,
                ).repeat_interleave(num_beams)
                self._dlwm_generation_row_to_world = row_to_world
            else:
                controller_rows = torch.arange(batch_size, device=last_hidden.device)
                row_to_world = controller_rows
            backbone_joint_repr = last_hidden[
                controller_rows,
                all_prompt_positions.index_select(0, controller_rows),
            ]
            joint_repr = self.dlwm_input_projection(backbone_joint_repr)
            dlwm_results = self.dlwm(
                joint_repr,
                use_rl=use_rl,
                max_steps=max_steps,
                sample_actions=sample_actions,
                action_sequence=action_sequence,
            )
            if self._collect_dlwm_rollouts:
                self._dlwm_generation_controller_calls += 1
                self._dlwm_generation_cache = {
                    "joint_repr": joint_repr,
                    "results": dict(dlwm_results),
                }

        prediction_positions = self._prediction_positions(
            last_hidden, attention_mask, labels
        )
        logits = outputs.logits.clone()
        final_world_logits: List[torch.Tensor] = []
        final_world_log_probs: List[torch.Tensor] = []
        final_world_indices: List[torch.Tensor] = []
        final_world_weights: List[torch.Tensor] = []
        final_aggregated: List[torch.Tensor] = []
        final_hidden_positions: List[int] = []
        final_logit_positions: List[int] = []

        for row, positions in enumerate(prediction_positions):
            if positions.numel() == 0:
                continue
            controller_row = int(row_to_world[row].item())
            start = int(dlwm_results["world_offsets"][controller_row].item())
            end = int(dlwm_results["world_offsets"][controller_row + 1].item())
            active_weights = dlwm_results["packed_world_weights"][start:end]
            token_states = last_hidden[row].index_select(0, positions)
            aggregated, world_logits, world_log_probs, active_indices = (
                self._predict_sample_positions(
                    token_states,
                    joint_repr[controller_row],
                    dlwm_results["packed_world_reprs"][start:end],
                    dlwm_results["packed_world_ids"][start:end],
                    active_weights,
                )
            )
            if logits.size(1) == last_hidden.size(1):
                mapped_positions = positions
            elif positions.numel() == 1:
                mapped_positions = torch.full_like(positions, logits.size(1) - 1)
            else:
                raise RuntimeError(
                    "full-answer DLWM training requires logits for every hidden position"
                )
            logits[row, mapped_positions] = aggregated.to(logits.dtype)
            final_world_logits.append(world_logits)
            final_world_log_probs.append(world_log_probs)
            final_world_indices.append(active_indices)
            final_world_weights.append(active_weights)
            final_aggregated.append(aggregated[-1])
            final_hidden_positions.append(int(positions[-1].item()))
            final_logit_positions.append(int(mapped_positions[-1].item()))

        if len(final_world_logits) != batch_size:
            raise RuntimeError("every DLWM sample must contain a prediction position")
        max_active = max(item.size(0) for item in final_world_logits)
        vocab_size = final_world_logits[0].size(-1)
        per_world_logits = final_world_logits[0].new_full(
            (batch_size, max_active, vocab_size),
            torch.finfo(final_world_logits[0].dtype).min,
        )
        per_world_log_probs = final_world_log_probs[0].new_full(
            (batch_size, max_active, vocab_size),
            torch.finfo(final_world_log_probs[0].dtype).min,
        )
        per_world_indices = torch.full(
            (batch_size, max_active),
            -1,
            dtype=torch.long,
            device=last_hidden.device,
        )
        per_world_weights = last_hidden.new_zeros((batch_size, max_active))
        for row, (world_logits, world_log_probs, indices, weights) in enumerate(
            zip(
                final_world_logits,
                final_world_log_probs,
                final_world_indices,
                final_world_weights,
            )
        ):
            count = world_logits.size(0)
            per_world_logits[row, :count] = world_logits
            per_world_log_probs[row, :count] = world_log_probs
            per_world_indices[row, :count] = indices
            per_world_weights[row, :count] = weights

        outputs.logits = logits
        positions = torch.tensor(
            final_hidden_positions, dtype=torch.long, device=last_hidden.device
        )
        logit_positions = torch.tensor(
            final_logit_positions, dtype=torch.long, device=last_hidden.device
        )
        aggregated_logits = torch.stack(final_aggregated, dim=0)
        target_ids = torch.full(
            (batch_size,), -100, dtype=torch.long, device=last_hidden.device
        )
        if labels is not None:
            target_positions = positions + 1
            target_ids = (
                labels.to(last_hidden.device)
                .gather(1, target_positions[:, None])
                .squeeze(1)
            )

        dlwm_results.update(
            {
                "joint_repr": joint_repr,
                "per_world_logits": per_world_logits,
                "per_world_log_probs": per_world_log_probs,
                "per_world_indices": per_world_indices,
                "per_world_weights": per_world_weights,
                "aggregated_logits": aggregated_logits,
                "prediction_positions": positions,
                "all_prediction_positions": prediction_positions,
                "logit_positions": logit_positions,
                "target_ids": target_ids,
                "controller_reused": controller_reused,
                "row_to_world": row_to_world,
                "controller_batch_size": joint_repr.size(0),
            }
        )
        outputs["dlwm_results"] = dlwm_results
        outputs["dlwm_final_hidden"] = dlwm_results["final_hidden"].index_select(
            0, row_to_world
        )
        outputs["per_world_logits"] = per_world_logits
        outputs["per_world_indices"] = per_world_indices
        outputs["world_weights"] = per_world_weights
        outputs["aggregated_logits"] = aggregated_logits
        outputs["orth_loss"] = dlwm_results["orth_loss"]
        if not self.training:
            self._cache_rollout_statistics(dlwm_results)
        return outputs

    def compute_loss(
        self,
        model_outputs,
        labels: torch.Tensor,
        ignore_index: int = -100,
        stage: str = "stage1",
        external_rewards: Optional[torch.Tensor] = None,
        update_baseline: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Combine language modeling, diversity, and REINFORCE objectives."""
        logits = model_outputs.logits
        if logits.size(1) != labels.size(1):
            raise ValueError(
                "training logits and labels must have equal sequence length"
            )
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=ignore_index,
        )
        zero = task_loss.new_zeros(())
        results = model_outputs.get("dlwm_results")
        if results is None:
            return {
                "loss": task_loss,
                "task_loss": task_loss,
                "orth_loss": zero,
                "policy_loss": zero,
                "value_loss": zero,
                "entropy": zero,
                "reward": zero,
            }

        orth_loss = results["orth_loss"].float()
        stage2 = stage.lower() in {"stage2", "rl", "stage2_rl"}
        total_loss = (
            task_loss + self.dlwm.lambda_orth * orth_loss
            if not stage2
            else task_loss * 0.0
        )
        policy_loss = zero
        value_loss = zero
        entropy = zero
        reward_mean = zero

        if stage2:
            target_ids = results["target_ids"]
            valid_targets = target_ids.ne(ignore_index)
            if external_rewards is None:
                reward_mode = getattr(
                    self.config, "dlwm_reward_mode", "generated_exact_match"
                )
                if reward_mode == "generated_exact_match":
                    raise ValueError(
                        "generated_exact_match requires on-policy external rewards "
                        "from DLWMTrainer"
                    )
                if reward_mode == "sequence_exact_match":
                    shifted_targets = labels[..., 1:]
                    supervised = shifted_targets.ne(ignore_index)
                    shifted_predictions = logits[..., :-1, :].argmax(dim=-1)
                    accuracy_reward = (
                        (shifted_predictions.eq(shifted_targets) | ~supervised).all(
                            dim=-1
                        )
                        & supervised.any(dim=-1)
                    ).to(task_loss.dtype)
                elif reward_mode == "final_token_accuracy":
                    predicted = results["aggregated_logits"].argmax(dim=-1)
                    accuracy_reward = (predicted.eq(target_ids) & valid_targets).to(
                        task_loss.dtype
                    )
                else:
                    raise ValueError(
                        "dlwm_reward_mode must be generated_exact_match, "
                        "sequence_exact_match, or final_token_accuracy"
                    )
            else:
                accuracy_reward = external_rewards.to(
                    device=task_loss.device, dtype=task_loss.dtype
                ).reshape(-1)
            rewards = self.dlwm.compute_rl_reward(
                accuracy_reward,
                results["num_steps"].to(task_loss.dtype),
                results["avg_active_worlds"].to(task_loss.dtype),
            )
            reward_mean = rewards.mean()
            trajectory_mask = results["trajectory_mask"].to(task_loss.dtype)
            log_probs = results["trajectory_log_probs"].float()
            values = results["trajectory_values"].float()
            entropies = results["trajectory_entropies"].float()
            count = trajectory_mask.sum().clamp_min(1.0)
            if log_probs.numel():
                gamma = float(getattr(self.config, "dlwm_rl_gamma", 1.0))
                horizon = trajectory_mask.sum(dim=1, keepdim=True)
                time = torch.arange(
                    log_probs.size(1), device=log_probs.device
                ).unsqueeze(0)
                remaining = (horizon - time - 1).clamp_min(0)
                discounts = log_probs.new_tensor(gamma).pow(remaining)
                baseline_mode = getattr(
                    self.config, "dlwm_rl_baseline", "moving_average"
                )
                if baseline_mode == "moving_average":
                    baseline = self.dlwm_reward_baseline
                    advantages = (rewards.detach() - baseline.detach()).unsqueeze(
                        1
                    ) * discounts
                    policy_loss = (
                        -(log_probs * advantages * trajectory_mask).sum() / count
                    )
                    value_loss = zero
                    should_update_baseline = (
                        self.training
                        if update_baseline is None
                        else bool(update_baseline)
                    )
                    if should_update_baseline:
                        synchronized_mean = rewards.detach().mean().float()
                        if dist.is_available() and dist.is_initialized():
                            synchronized_mean = synchronized_mean.clone()
                            dist.all_reduce(synchronized_mean, op=dist.ReduceOp.SUM)
                            synchronized_mean /= dist.get_world_size()
                        momentum = float(
                            getattr(
                                self.config,
                                "dlwm_rl_baseline_momentum",
                                0.9,
                            )
                        )
                        with torch.no_grad():
                            self.dlwm_reward_baseline.mul_(momentum).add_(
                                synchronized_mean,
                                alpha=1.0 - momentum,
                            )
                            self.dlwm_reward_baseline_initialized.fill_(True)
                elif baseline_mode == "value":
                    returns = rewards.detach().unsqueeze(1) * discounts
                    advantages = returns - values.detach()
                    policy_loss = (
                        -(log_probs * advantages * trajectory_mask).sum() / count
                    )
                    value_loss = (
                        (values - returns).square() * trajectory_mask
                    ).sum() / count
                else:
                    raise ValueError("dlwm_rl_baseline must be moving_average or value")
                entropy = (entropies * trajectory_mask).sum() / count
                rl_lambda = float(getattr(self.config, "dlwm_rl_lambda", 0.01))
                value_coef = float(getattr(self.config, "dlwm_rl_value_coef", 0.5))
                entropy_coef = float(getattr(self.config, "dlwm_rl_entropy_coef", 0.0))
                total_loss = rl_lambda * (
                    policy_loss + value_coef * value_loss - entropy_coef * entropy
                )

        return {
            "loss": total_loss,
            "task_loss": task_loss,
            "orth_loss": orth_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "reward": reward_mean,
        }

    def generate(self, *args, **kwargs):
        """Use the upstream Qwen generation loop for version compatibility.

        QwenWithLVR carries a fork of an older private Transformers generation
        loop. DLWM only needs its custom ``forward`` and therefore delegates to
        the maintained upstream implementation.
        """
        return Qwen2_5_VLForConditionalGeneration.generate(self, *args, **kwargs)

    def _reorder_cache(
        self,
        past_key_values,
        beam_idx: torch.LongTensor,
    ):
        """Reorder the language cache and the beam-to-world indirection."""
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
            reordered = past_key_values
        else:
            reordered = tuple(
                tuple(
                    state.index_select(0, beam_idx.to(state.device)) for state in layer
                )
                for layer in past_key_values
            )
        if (
            self._collect_dlwm_rollouts
            and self._dlwm_generation_row_to_world is not None
        ):
            self._dlwm_generation_row_to_world = (
                self._dlwm_generation_row_to_world.index_select(
                    0,
                    beam_idx.to(self._dlwm_generation_row_to_world.device),
                )
            )
            self._dlwm_generation_beam_reorders += 1
        return reordered

    def generate_with_dlwm(self, inputs=None, generation_config=None, **kwargs):
        """Generate with one persistent controller rollout for the whole answer."""
        configured_beams = None
        if generation_config is not None:
            configured_beams = (
                generation_config.get("num_beams")
                if isinstance(generation_config, dict)
                else getattr(generation_config, "num_beams", None)
            )
        num_beams = int(kwargs.get("num_beams", configured_beams or 1))
        if num_beams < 1:
            raise ValueError("num_beams must be positive")
        kwargs.setdefault("enable_dlwm", True)
        kwargs.setdefault("dlwm_use_rl", True)
        kwargs.setdefault("dlwm_sample_actions", False)
        kwargs.setdefault("dlwm_stage", "inference")
        self._dlwm_rollout_history = []
        self._dlwm_generation_cache = None
        self._dlwm_generation_controller_calls = 0
        self._dlwm_generation_num_beams = num_beams
        self._dlwm_generation_row_to_world = None
        self._dlwm_generation_beam_reorders = 0
        self._collect_dlwm_rollouts = True
        try:
            return self.generate(
                inputs=inputs,
                generation_config=generation_config,
                **kwargs,
            )
        finally:
            self._collect_dlwm_rollouts = False
            self._dlwm_generation_cache = None
            self._dlwm_generation_row_to_world = None

    def _cache_rollout_statistics(self, results: Dict[str, Any]) -> None:
        def cpu_list(value):
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            return value

        row_to_world = results.get("row_to_world")
        controller_batch_size = int(
            results.get(
                "controller_batch_size",
                results["num_steps"].size(0),
            )
        )

        def expand_controller_tensor(value):
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.size(0) == controller_batch_size
                and row_to_world is not None
            ):
                return value.index_select(0, row_to_world.to(value.device))
            return value

        def expand_controller_list(value):
            if (
                isinstance(value, list)
                and len(value) == controller_batch_size
                and row_to_world is not None
            ):
                return [value[int(index)] for index in row_to_world.detach().cpu()]
            return value

        per_world_top_ids = results["per_world_logits"].detach().argmax(dim=-1)
        statistics = {
            "num_steps": cpu_list(expand_controller_tensor(results["num_steps"])),
            "num_active_worlds": cpu_list(
                expand_controller_tensor(results["num_active_worlds"])
            ),
            "avg_active_worlds": cpu_list(
                expand_controller_tensor(results["avg_active_worlds"])
            ),
            "remaining_budget": cpu_list(
                expand_controller_tensor(results["remaining_budget"])
            ),
            "world_weights": cpu_list(results["per_world_weights"]),
            "per_world_indices": cpu_list(results["per_world_indices"]),
            "merge_betas": cpu_list(expand_controller_tensor(results["merge_betas"])),
            "merge_parent_ids": cpu_list(
                expand_controller_tensor(results["merge_parent_ids"])
            ),
            "merge_beta_strategy": results["merge_beta_strategy"],
            "merge_beta_definition": results["merge_beta_definition"],
            "merge_beta_temperature": results["merge_beta_temperature"],
            "actions": expand_controller_list(results["decoded_actions"]),
            "action_indices": cpu_list(
                expand_controller_tensor(results["action_indices"])
            ),
            "trajectory_mask": cpu_list(
                expand_controller_tensor(results["trajectory_mask"])
            ),
            "active_world_count_history": cpu_list(
                expand_controller_tensor(results["active_world_count_history"])
            ),
            "packed_storage_history": cpu_list(
                expand_controller_tensor(results["packed_storage_history"])
            ),
            "controller_reused": bool(results.get("controller_reused", False)),
            "controller_batch_size": controller_batch_size,
            "beam_row_to_world": cpu_list(row_to_world),
            "per_world_top_token_ids": cpu_list(per_world_top_ids),
            "aggregated_top_token_ids": cpu_list(
                results["aggregated_logits"].detach().argmax(dim=-1)
            ),
        }
        self._last_dlwm_statistics = statistics
        if self._collect_dlwm_rollouts:
            self._dlwm_rollout_history.append(statistics)

    def get_last_dlwm_statistics(self) -> Dict[str, Any]:
        if not self._dlwm_rollout_history:
            return dict(self._last_dlwm_statistics)
        last = dict(self._dlwm_rollout_history[-1])
        controller_rollout = next(
            (
                rollout
                for rollout in self._dlwm_rollout_history
                if not rollout.get("controller_reused")
            ),
            self._dlwm_rollout_history[0],
        )
        last["num_steps"] = controller_rollout["num_steps"]
        last["avg_active_worlds"] = controller_rollout["avg_active_worlds"]
        last["num_active_worlds"] = controller_rollout["num_active_worlds"]
        last["actions"] = controller_rollout["actions"]
        last["action_indices"] = controller_rollout["action_indices"]
        last["trajectory_mask"] = controller_rollout["trajectory_mask"]
        last["generation_rollouts"] = list(self._dlwm_rollout_history)
        last["generation_forward_calls"] = len(self._dlwm_rollout_history)
        last["generation_controller_calls"] = self._dlwm_generation_controller_calls
        last["generation_num_beams"] = self._dlwm_generation_num_beams
        last["generation_beam_reorders"] = self._dlwm_generation_beam_reorders
        return last

    def get_dlwm_statistics(self) -> Dict[str, Any]:
        if not hasattr(self, "dlwm"):
            return {}
        stats: Dict[str, Any] = {
            "num_worlds": self.dlwm.num_worlds,
            "hidden_size": self.dlwm.hidden_size,
            "compute_budget": self.dlwm.compute_budget,
            "lambda_orth": self.dlwm.lambda_orth,
            "lambda_steps": self.dlwm.lambda_steps,
            "lambda_worlds": self.dlwm.lambda_worlds,
            "merge_mode": self.dlwm.world_merger.mode,
            "merge_beta_definition": self.dlwm.world_merger.beta_definition,
            "merge_fixed_beta": self.dlwm.world_merger.fixed_beta,
            "merge_temperature": self.dlwm.world_merger.temperature,
            "merge_topk_pairs": self.dlwm.merge_topk_pairs,
            "packed_active_worlds": True,
            "beam_safe_generation": True,
            "reasoner_layers": self.dlwm.reasoning_module.num_layers,
            "reasoner_heads": self.dlwm.reasoning_module.num_heads,
            "diversity_regularizer": self.dlwm.diversity_regularizer,
            "aggregation": getattr(
                self.config, "dlwm_aggregation", "probability_mixture"
            ),
        }
        if self._last_dlwm_statistics:
            stats["last_rollout"] = self.get_last_dlwm_statistics()
        return stats
