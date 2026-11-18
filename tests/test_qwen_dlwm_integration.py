import unittest
import tempfile
from types import SimpleNamespace

import torch
from transformers import Qwen2_5_VLConfig

from src.model.qwen_dlwm_model import QwenWithDLWM
from src.params import TrainingArguments
from src.train.train_dlwm import DLWMTrainer, configure_trainable_parameters


class TinyTokenizer:
    pad_token_id = 0

    def batch_decode(self, rows, **kwargs):
        del kwargs
        return [" ".join(str(int(token)) for token in row) for row in rows]


class TinyProcessor:
    tokenizer = TinyTokenizer()


def tiny_config():
    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "fullatt_block_indexes": [0],
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 8,
        },
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
        pad_token_id=0,
        eos_token_id=31,
        bos_token_id=1,
    )
    for name, value in {
        "lvr_head": False,
        "latent_end_token": False,
        "enable_dlwm": True,
        "dlwm_num_worlds": 3,
        "dlwm_compute_budget": 3,
        "dlwm_max_steps": 2,
        "dlwm_use_rl": False,
        "dlwm_dropout": 0.0,
        "dlwm_reasoner_layers": 1,
        "dlwm_merge_topk_pairs": 0,
        "dlwm_reward_mode": "sequence_exact_match",
        "dlwm_rl_entropy_coef": 0.0,
    }.items():
        setattr(config, name, value)
    return config


class QwenDLWMIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = QwenWithDLWM(tiny_config())
        self.input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 6, 7, 8, 9]])
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, -100, 7, 8, 9]])

    def test_per_world_predictions_reach_final_logits(self):
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            dlwm_stage="stage1",
        )
        self.assertEqual(tuple(output.per_world_logits.shape), (2, 3, 32))
        torch.testing.assert_close(output.world_weights.sum(-1), torch.ones(2))
        torch.testing.assert_close(
            output.aggregated_logits.float().softmax(-1).sum(-1),
            torch.ones(2),
        )
        rows = torch.arange(2)
        positions = output.dlwm_results["logit_positions"]
        torch.testing.assert_close(
            output.logits[rows, positions], output.aggregated_logits
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(
            [
                positions.tolist()
                for positions in output.dlwm_results["all_prediction_positions"]
            ],
            [[1, 2, 3], [1, 2, 3]],
        )
        self.assertEqual(output.dlwm_results["joint_repr"].shape, (2, 16))
        self.assertEqual(
            self.model.config.dlwm_merge_beta_definition,
            "softmax([s_psi(h_i), s_psi(h_j)] / temperature)[0]",
        )
        self.assertTrue(self.model.get_dlwm_statistics()["packed_active_worlds"])

    def test_stage2_policy_gradient_and_upstream_generation(self):
        self.model.train()
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            dlwm_stage="stage2",
            dlwm_use_rl=True,
            dlwm_sample_actions=True,
        )
        output.loss.backward()
        policy_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.dlwm.policy.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(policy_gradient, 0.0)

        self.model.eval()
        generated = self.model.generate_with_dlwm(
            input_ids=self.input_ids[:1, :3],
            attention_mask=self.attention_mask[:1, :3],
            max_new_tokens=2,
        )
        self.assertEqual(generated.size(1), 5)
        statistics = self.model.get_last_dlwm_statistics()
        self.assertEqual(statistics["generation_forward_calls"], 2)
        self.assertEqual(statistics["generation_controller_calls"], 1)
        self.assertEqual(len(statistics["generation_rollouts"]), 2)
        self.assertFalse(statistics["generation_rollouts"][0]["controller_reused"])
        self.assertTrue(statistics["generation_rollouts"][1]["controller_reused"])

    def test_prompt_world_state_does_not_read_gold_answer(self):
        self.model.eval()
        alternative = self.input_ids.clone()
        alternative[:, 2:] = torch.tensor([[10, 11, 12], [13, 14, 15]])
        first = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
            dlwm_max_steps=0,
        )
        second = self.model(
            input_ids=alternative,
            attention_mask=self.attention_mask,
            labels=self.labels,
            dlwm_max_steps=0,
        )
        torch.testing.assert_close(
            first.dlwm_results["joint_repr"],
            second.dlwm_results["joint_repr"],
        )

    def test_persistent_world_cache_supports_beam_reordering(self):
        self.model.eval()
        generated = self.model.generate_with_dlwm(
            input_ids=self.input_ids[:, :3],
            attention_mask=self.attention_mask[:, :3],
            max_new_tokens=2,
            num_beams=3,
            num_return_sequences=2,
        )
        self.assertEqual(generated.size(0), 4)
        statistics = self.model.get_last_dlwm_statistics()
        self.assertEqual(statistics["generation_num_beams"], 3)
        self.assertEqual(statistics["generation_controller_calls"], 1)
        self.assertGreaterEqual(statistics["generation_beam_reorders"], 1)
        self.assertEqual(statistics["controller_batch_size"], 2)
        self.assertEqual(len(statistics["beam_row_to_world"]), 6)
        self.assertEqual(statistics["beam_row_to_world"].count(0), 3)
        self.assertEqual(statistics["beam_row_to_world"].count(1), 3)

    def test_stage2_trains_only_controller(self):
        args = SimpleNamespace(
            dlwm_stage="stage2",
            dlwm_stage2_controller_only=True,
            freeze_llm=True,
            freeze_vision_tower=True,
            freeze_merger=True,
        )
        configure_trainable_parameters(self.model, args)
        trainable = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("dlwm.policy.") for name in trainable))

    def test_generated_answer_reward_replays_sampled_trajectory(self):
        args = TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            per_device_train_batch_size=1,
            report_to=[],
            dlwm_stage="stage2",
            dlwm_reward_mode="generated_exact_match",
            dlwm_rollout_max_new_tokens=1,
            dlwm_sample_actions=True,
            remove_unused_columns=False,
        )
        configure_trainable_parameters(self.model, args)
        with torch.no_grad():
            self.model.dlwm.policy.answer_head.bias.fill_(-10.0)
        trainer = DLWMTrainer(
            model=self.model,
            args=args,
            processing_class=TinyProcessor(),
        )
        device = next(self.model.parameters()).device
        inputs = {
            "input_ids": self.input_ids[:1].to(device),
            "attention_mask": self.attention_mask[:1].to(device),
            "labels": self.labels[:1].to(device),
        }
        self.model.train()
        loss, outputs = trainer.compute_loss(
            self.model,
            inputs,
            return_outputs=True,
        )
        loss.backward()
        policy_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.dlwm.policy.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(policy_gradient, 0.0)
        self.assertTrue(outputs.dlwm_results["decoded_actions"][0])
        self.assertTrue(bool(self.model.dlwm_reward_baseline_initialized))


if __name__ == "__main__":
    unittest.main()
