import unittest

import torch
import torch.nn as nn

from src.model.dlwm_modules import (
    DLWMReasoningModule,
    ResourceAwarePolicy,
    WorldMerger,
    WorldScoringNetwork,
)


class AddOne(nn.Module):
    def forward(self, value):
        return value + torch.ones_like(value)


class DLWMModulesTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_action_space_round_trip(self):
        policy = ResourceAwarePolicy(hidden_size=8, num_worlds=4)
        expected = [
            ("expand", 0, -1),
            ("stop", 3, -1),
            ("merge", 0, 3),
            ("answer", -1, -1),
        ]
        for action in expected:
            index = policy.action_index(*action)
            self.assertEqual(policy.decode_action(index), action)
        self.assertEqual(policy.action_space_size, 4 + 4 + 6 + 1)

    def test_action_mask_rejects_invalid_and_last_stop(self):
        policy = ResourceAwarePolicy(hidden_size=8, num_worlds=3)
        active = torch.tensor([[True, False, False], [True, True, False]])
        budget = torch.tensor([0.0, 0.6])
        mask = policy.valid_action_mask(
            active,
            budget,
            expand_cost=1.0,
            stop_cost=0.1,
            merge_cost=0.5,
        )
        self.assertFalse(mask[0, policy.action_index("expand", 0)])
        self.assertFalse(mask[0, policy.action_index("stop", 0)])
        self.assertTrue(mask[0, policy.action_index("answer")])
        self.assertTrue(mask[1, policy.action_index("stop", 0)])
        self.assertTrue(mask[1, policy.action_index("merge", 0, 1)])

    def test_execute_expand_stop_merge_answer(self):
        module = DLWMReasoningModule(
            hidden_size=4,
            num_worlds=3,
            compute_budget=3,
            merge_mode="fixed",
            merge_fixed_beta=0.25,
            expand_cost=1.0,
            stop_cost=0.1,
            merge_cost=0.5,
            dropout=0.0,
        )
        module.reasoning_module = AddOne()
        worlds = torch.arange(48, dtype=torch.float32).view(4, 3, 4)
        active = torch.ones(4, 3, dtype=torch.bool)
        candidates = active.clone()
        budget = torch.full((4,), 3.0)
        actions = torch.tensor(
            [
                module.policy.action_index("expand", 1),
                module.policy.action_index("stop", 1),
                module.policy.action_index("merge", 0, 2),
                module.policy.action_index("answer"),
            ]
        )
        result = module.execute_actions(worlds, active, candidates, budget, actions)
        updated, active, candidates, budget, done, betas = result
        torch.testing.assert_close(updated[0, 1], worlds[0, 1] + 1)
        self.assertFalse(active[1, 1])
        self.assertFalse(candidates[1, 1])
        self.assertFalse(active[2, 2])
        self.assertFalse(candidates[2, 2])
        torch.testing.assert_close(
            updated[2, 0], 0.25 * worlds[2, 0] + 0.75 * worlds[2, 2]
        )
        self.assertAlmostEqual(float(betas[2]), 0.25)
        self.assertTrue(done[3])
        torch.testing.assert_close(budget, torch.tensor([2.0, 2.9, 2.5, 3.0]))

    def test_forced_rollout_aggregates_only_remaining_worlds(self):
        module = DLWMReasoningModule(
            hidden_size=8,
            num_worlds=4,
            compute_budget=4,
            merge_topk_pairs=0,
            dropout=0.0,
        )
        policy_sequence_lengths = []

        def record_policy_length(_, inputs):
            policy_sequence_lengths.append(inputs[0].size(1))

        hook = module.policy.policy_network.register_forward_pre_hook(
            record_policy_length
        )
        actions = torch.tensor(
            [
                [
                    module.policy.action_index("stop", 3),
                    module.policy.action_index("merge", 0, 1),
                    module.policy.action_index("expand", 2),
                    module.policy.action_index("answer"),
                ]
            ]
        )
        output = module(
            torch.randn(1, 8),
            use_rl=True,
            max_steps=4,
            action_sequence=actions,
        )
        hook.remove()
        torch.testing.assert_close(
            output["packed_world_ids"],
            torch.tensor([0, 2]),
        )
        torch.testing.assert_close(output["world_offsets"], torch.tensor([0, 2]))
        self.assertEqual(tuple(output["packed_world_reprs"].shape), (2, 8))
        self.assertEqual(policy_sequence_lengths, [5, 4, 3, 3])
        self.assertEqual(output["num_steps"].tolist(), [3])
        self.assertEqual(output["num_active_worlds"].tolist(), [2])
        self.assertEqual(
            [action[0] for action in output["decoded_actions"][0]],
            ["stop", "merge", "expand", "answer"],
        )
        torch.testing.assert_close(output["world_weights"].sum(), torch.ones(()))
        self.assertEqual(
            output["active_world_count_history"].tolist(),
            [[4, 3, 2, 2]],
        )
        self.assertEqual(output["packed_storage_history"].tolist(), [[4, 3, 2, 2]])

    def test_learned_beta_is_convex_and_differentiable(self):
        merger = WorldMerger(hidden_size=8, mode="score_softmax")
        left = torch.randn(2, 8, requires_grad=True)
        right = torch.randn(2, 8, requires_grad=True)
        left_utility = left.mean(-1)
        right_utility = right.mean(-1)
        merged, beta = merger(left, right, left_utility, right_utility)
        expected_beta = torch.softmax(
            torch.stack((left_utility, right_utility), dim=-1), dim=-1
        )[:, 0]
        torch.testing.assert_close(beta, expected_beta)
        self.assertTrue(bool(((beta >= 0) & (beta <= 1)).all()))
        torch.testing.assert_close(
            merged, beta[:, None] * left + (1 - beta[:, None]) * right
        )
        merged.sum().backward()
        self.assertIsNotNone(left.grad)
        self.assertIsNotNone(right.grad)

    def test_reference_softmax_beta_uses_similarity_to_z(self):
        merger = WorldMerger(hidden_size=2, mode="reference_softmax")
        left = torch.tensor([1.0, 0.0])
        right = torch.tensor([0.0, 1.0])
        reference = torch.tensor([1.0, 0.0])
        _, beta = merger(left, right, reference=reference)
        expected = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[0]
        torch.testing.assert_close(beta, expected)
        self.assertEqual(
            merger.beta_definition,
            "softmax([cos(h_i,z), cos(h_j,z)] / temperature)[0]",
        )

    def test_packed_policy_skips_completed_samples(self):
        module = DLWMReasoningModule(
            hidden_size=8,
            num_worlds=4,
            compute_budget=4,
            merge_topk_pairs=0,
            dropout=0.0,
        )
        policy_sequence_lengths = []

        def record_policy_length(_, inputs):
            policy_sequence_lengths.append(inputs[0].size(1))

        hook = module.policy.policy_network.register_forward_pre_hook(
            record_policy_length
        )
        answer = module.policy.action_index("answer")
        actions = torch.tensor(
            [
                [
                    module.policy.action_index("stop", 3),
                    module.policy.action_index("merge", 0, 1),
                    answer,
                ],
                [answer, answer, answer],
            ]
        )
        output = module(
            torch.randn(2, 8),
            use_rl=True,
            max_steps=3,
            action_sequence=actions,
        )
        hook.remove()
        self.assertEqual(policy_sequence_lengths, [5, 5, 4, 3])
        self.assertEqual(output["world_offsets"].tolist(), [0, 2, 6])
        self.assertEqual(output["num_active_worlds"].tolist(), [2, 4])
        self.assertEqual(
            output["packed_storage_history"].tolist(),
            [[4, 3, 2], [4, 0, 0]],
        )

    def test_reference_beta_is_wired_into_packed_merge(self):
        module = DLWMReasoningModule(
            hidden_size=8,
            num_worlds=3,
            merge_mode="reference_softmax",
            merge_topk_pairs=0,
            dropout=0.0,
        )
        actions = torch.tensor(
            [
                [
                    module.policy.action_index("merge", 0, 1),
                    module.policy.action_index("answer"),
                ]
            ]
        )
        output = module(
            torch.randn(1, 8),
            use_rl=True,
            max_steps=2,
            action_sequence=actions,
        )
        self.assertTrue(torch.isfinite(output["merge_betas"][0, 0]))
        self.assertEqual(output["merge_parent_ids"][0, 0].tolist(), [0, 1])
        self.assertEqual(output["merge_beta_strategy"], "reference_softmax")

    def test_world_scorer_masks_consumed_slots(self):
        scorer = WorldScoringNetwork(hidden_size=8, dropout=0.0)
        weights = scorer(
            torch.randn(2, 3, 8),
            torch.tensor([[True, False, True], [False, True, False]]),
        )
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        self.assertEqual(float(weights[0, 1].detach()), 0.0)
        self.assertEqual(float(weights[1, 0].detach()), 0.0)
        self.assertEqual(float(weights[1, 2].detach()), 0.0)

    def test_semantic_merge_mask_keeps_most_similar_pair(self):
        policy = ResourceAwarePolicy(hidden_size=4, num_worlds=3, dropout=0.0)
        worlds = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
        )
        mask = policy.valid_action_mask(
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([3.0]),
            expand_cost=1.0,
            stop_cost=0.1,
            merge_cost=0.5,
            world_states=worlds,
            merge_topk_pairs=1,
        )
        self.assertTrue(mask[0, policy.action_index("merge", 0, 1)])
        self.assertFalse(mask[0, policy.action_index("merge", 0, 2)])
        self.assertFalse(mask[0, policy.action_index("merge", 1, 2)])

    def test_orthogonal_loss_matches_equation_sum(self):
        module = DLWMReasoningModule(
            hidden_size=2,
            num_worlds=2,
            reasoner_layers=1,
            dropout=0.0,
        )
        worlds = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        # Equation (4) sums i != j, so the two ordered off-diagonal terms sum to 2.
        torch.testing.assert_close(
            module._compute_orthogonal_loss(worlds),
            torch.tensor(2.0),
        )

    def test_sampled_rollout_has_policy_gradient(self):
        module = DLWMReasoningModule(
            hidden_size=8, num_worlds=3, compute_budget=3, dropout=0.0
        )
        output = module(
            torch.randn(3, 8), use_rl=True, max_steps=3, sample_actions=True
        )
        mask = output["trajectory_mask"].float()
        loss = -(output["trajectory_log_probs"] * mask).sum()
        loss.backward()
        policy_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.policy.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(policy_gradient, 0.0)

    def test_reward_matches_paper_definition(self):
        module = DLWMReasoningModule(
            hidden_size=4,
            num_worlds=2,
            lambda_steps=0.05,
            lambda_worlds=0.1,
        )
        reward = module.compute_rl_reward(
            torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([2.0])
        )
        torch.testing.assert_close(reward, torch.tensor([0.65]))


if __name__ == "__main__":
    unittest.main()
