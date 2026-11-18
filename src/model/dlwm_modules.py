"""Core modules for Diverse Latent World Models (DLWM).

World identifiers remain stable for action replay, but active states are stored
as one physically packed tensor with per-sample offsets. Stop and merge therefore
remove tensor rows instead of leaving masked ``K``-wide slots behind.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class LatentWorldGenerator(nn.Module):
    """Generate ``num_worlds`` diverse hypotheses from one joint state."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_worlds: int = 4,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if num_worlds < 1:
            raise ValueError("num_worlds must be positive")
        if num_worlds > hidden_size:
            raise ValueError("num_worlds cannot exceed hidden_size for orthogonal init")
        self.hidden_size = hidden_size
        self.num_worlds = num_worlds
        self.residual = bool(residual)
        intermediate_size = intermediate_size or 4 * hidden_size

        self.world_embeddings = nn.Parameter(torch.empty(num_worlds, hidden_size))
        self.world_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
        )
        self._init_world_embeddings()

    def _init_world_embeddings(self) -> None:
        # QR is both faster and numerically safer than an in-place Gram-Schmidt
        # loop. For the usual setting hidden_size >> num_worlds, rows of Q^T are
        # orthonormal.
        with torch.no_grad():
            basis = torch.randn(self.hidden_size, self.num_worlds)
            q, _ = torch.linalg.qr(basis, mode="reduced")
            self.world_embeddings.copy_(q.transpose(0, 1) * math.sqrt(self.hidden_size))

    def forward(
        self,
        joint_repr: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if joint_repr.ndim != 2 or joint_repr.size(-1) != self.hidden_size:
            raise ValueError(
                f"joint_repr must have shape [batch, {self.hidden_size}], "
                f"got {tuple(joint_repr.shape)}"
            )
        actual_batch = joint_repr.size(0)
        if batch_size is not None and batch_size != actual_batch:
            raise ValueError("batch_size does not match joint_repr")

        joint = joint_repr.unsqueeze(1).expand(-1, self.num_worlds, -1)
        world_ids = self.world_embeddings.to(dtype=joint_repr.dtype)
        world_ids = world_ids.unsqueeze(0).expand(actual_batch, -1, -1)
        generated = self.world_generator(torch.cat((joint, world_ids), dim=-1))
        # Equation (2) in the manuscript defines w_i = G(z, e_i), without an
        # additional residual. A residual remains available as an explicit
        # compatibility ablation rather than being silently enabled.
        return generated + joint if self.residual else generated


class SharedLatentReasoner(nn.Module):
    """Pre-LN Transformer stack shared by every world and rollout step.

    A world state is a single latent token, so worlds are folded into the batch
    dimension. This preserves independent trajectories: attention never mixes
    information between hypotheses. The complete stack is reused at every
    ``expand`` step, which implements the tied-across-time requirement.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("reasoner num_layers must be positive")
        valid_heads = [
            heads
            for heads in (num_heads, 8, 4, 2, 1)
            if heads > 0 and hidden_size % heads == 0
        ]
        if not valid_heads:
            raise ValueError(
                "hidden_size must be divisible by at least one attention head"
            )
        num_heads = valid_heads[0]
        ffn_size = max(int(hidden_size * ffn_multiplier), hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False,
        )
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_multiplier = float(ffn_multiplier)

    def forward(self, world_states: torch.Tensor) -> torch.Tensor:
        original_shape = world_states.shape
        if world_states.ndim == 1:
            world_states = world_states.unsqueeze(0)
        if world_states.ndim < 2:
            raise ValueError("world_states must end in a hidden dimension")
        hidden_size = world_states.size(-1)
        flattened = world_states.reshape(-1, 1, hidden_size)
        reasoned = self.layers(flattened).reshape(world_states.shape)
        return reasoned.reshape(original_shape)


class WorldScoringNetwork(nn.Module):
    """Return normalized aggregation weights over valid answer candidates."""

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        intermediate_size = intermediate_size or max(hidden_size // 2, 1)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, 1),
        )

    def raw_scores(self, world_reprs: torch.Tensor) -> torch.Tensor:
        return self.scorer(world_reprs).squeeze(-1)

    def forward(
        self,
        world_reprs: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scores = self.raw_scores(world_reprs)
        if candidate_mask is None:
            return F.softmax(scores, dim=-1)
        if candidate_mask.shape != scores.shape:
            raise ValueError("candidate_mask must have shape [batch, num_worlds]")
        if (~candidate_mask.any(dim=-1)).any():
            raise ValueError("every sample must retain at least one answer candidate")
        scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
        return F.softmax(scores, dim=-1)

    def score_packed(
        self,
        packed_world_reprs: torch.Tensor,
        world_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize scores independently within every packed sample."""
        scores = self.raw_scores(packed_world_reprs)
        weights = []
        for row in range(world_offsets.numel() - 1):
            start = int(world_offsets[row].item())
            end = int(world_offsets[row + 1].item())
            if end <= start:
                raise ValueError("every sample must retain at least one packed world")
            weights.append(F.softmax(scores[start:end], dim=0))
        return torch.cat(weights, dim=0)


class WorldMerger(nn.Module):
    """Convexly merge two worlds with an explicit beta definition.

    The manuscript leaves beta undefined. ``score_softmax`` ties it to the
    trained Eq. (6) scorer:

        beta_ij = exp(u_i / tau) / (exp(u_i / tau) + exp(u_j / tau))
        merge(i, j) = beta_ij * w_i + (1 - beta_ij) * w_j

    ``reference_softmax`` uses cosine agreement with the multimodal encoding
    ``z`` in the same pair-softmax. ``fixed`` is the deterministic ablation.
    The historical name ``learned`` is accepted as an alias of
    ``score_softmax``.
    """

    def __init__(
        self,
        hidden_size: int,
        mode: str = "score_softmax",
        fixed_beta: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        mode = {"learned": "score_softmax"}.get(mode, mode)
        if mode not in {"score_softmax", "reference_softmax", "fixed"}:
            raise ValueError(
                "merge mode must be score_softmax, reference_softmax, or fixed"
            )
        if not 0.0 <= fixed_beta <= 1.0:
            raise ValueError("fixed_beta must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.mode = mode
        self.fixed_beta = float(fixed_beta)
        self.temperature = float(temperature)
        self.hidden_size = int(hidden_size)

    @property
    def beta_definition(self) -> str:
        if self.mode == "score_softmax":
            return "softmax([s_psi(h_i), s_psi(h_j)] / temperature)[0]"
        if self.mode == "reference_softmax":
            return "softmax([cos(h_i,z), cos(h_j,z)] / temperature)[0]"
        return "fixed_beta"

    def beta(
        self,
        world_i: torch.Tensor,
        world_j: torch.Tensor,
        utility_i: Optional[torch.Tensor] = None,
        utility_j: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if world_i.shape != world_j.shape:
            raise ValueError("worlds passed to merge must have identical shapes")
        if self.mode == "fixed":
            return world_i.new_full(world_i.shape[:-1], self.fixed_beta)
        if self.mode == "reference_softmax":
            if reference is None or reference.shape != world_i.shape:
                raise ValueError(
                    "reference_softmax requires z with the same shape as each world"
                )
            utility_i = F.cosine_similarity(world_i.float(), reference.float(), dim=-1)
            utility_j = F.cosine_similarity(world_j.float(), reference.float(), dim=-1)
        elif (utility_i is None) != (utility_j is None):
            raise ValueError("utility_i and utility_j must be provided together")
        elif utility_i is None:
            raise ValueError(
                "score_softmax requires utilities from WorldScoringNetwork"
            )
        utilities = torch.stack((utility_i, utility_j), dim=-1)
        return F.softmax(utilities.float() / self.temperature, dim=-1)[..., 0].to(
            world_i.dtype
        )

    def forward(
        self,
        world_i: torch.Tensor,
        world_j: torch.Tensor,
        utility_i: Optional[torch.Tensor] = None,
        utility_j: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta(world_i, world_j, utility_i, utility_j, reference)
        merged = beta.unsqueeze(-1) * world_i + (1.0 - beta).unsqueeze(-1) * world_j
        return merged, beta


class ResourceAwarePolicy(nn.Module):
    """Resource-aware controller over expand, stop, merge, and answer actions."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_worlds: int = 4,
        compute_budget: float = 10,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_worlds = num_worlds
        self.compute_budget = float(compute_budget)
        self.merge_pairs: Tuple[Tuple[int, int], ...] = tuple(
            (i, j) for i in range(num_worlds) for j in range(i + 1, num_worlds)
        )
        intermediate_size = intermediate_size or max(hidden_size // 4, 1)
        num_heads = next(
            heads for heads in (8, 4, 2, 1) if intermediate_size % heads == 0
        )
        self.world_projection = nn.Linear(hidden_size, intermediate_size)
        self.budget_projection = nn.Linear(1, intermediate_size)
        self.world_slot_embeddings = nn.Parameter(
            torch.randn(num_worlds, intermediate_size) / math.sqrt(intermediate_size)
        )
        policy_layer = nn.TransformerEncoderLayer(
            d_model=intermediate_size,
            nhead=num_heads,
            dim_feedforward=4 * intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # The appendix specifies a two-layer Transformer controller.
        self.policy_network = nn.TransformerEncoder(
            policy_layer, num_layers=2, enable_nested_tensor=False
        )
        self.expand_head = nn.Linear(intermediate_size, 1)
        self.stop_head = nn.Linear(intermediate_size, 1)
        self.merge_head = nn.Sequential(
            nn.Linear(4 * intermediate_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, 1),
        )
        self.answer_head = nn.Linear(2 * intermediate_size, 1)
        self.value_head = nn.Linear(intermediate_size, 1)

    @property
    def action_space_size(self) -> int:
        return 2 * self.num_worlds + len(self.merge_pairs) + 1

    @property
    def answer_action_index(self) -> int:
        return self.action_space_size - 1

    def _get_action_space_size(self) -> int:
        """Backward-compatible alias used by earlier code."""
        return self.action_space_size

    def action_index(
        self, action_type: str, world_idx: int = -1, world_idx2: int = -1
    ) -> int:
        if action_type == "expand" and 0 <= world_idx < self.num_worlds:
            return world_idx
        if action_type == "stop" and 0 <= world_idx < self.num_worlds:
            return self.num_worlds + world_idx
        if action_type == "merge":
            pair = tuple(sorted((world_idx, world_idx2)))
            if pair in self.merge_pairs:
                return 2 * self.num_worlds + self.merge_pairs.index(pair)
        if action_type == "answer":
            return self.answer_action_index
        raise ValueError(
            f"invalid controller action: {(action_type, world_idx, world_idx2)}"
        )

    def decode_action(self, action_idx: int) -> Tuple[str, int, int]:
        if not 0 <= action_idx < self.action_space_size:
            raise ValueError(f"action index {action_idx} is out of range")
        if action_idx < self.num_worlds:
            return "expand", action_idx, -1
        if action_idx < 2 * self.num_worlds:
            return "stop", action_idx - self.num_worlds, -1
        merge_offset = action_idx - 2 * self.num_worlds
        if merge_offset < len(self.merge_pairs):
            i, j = self.merge_pairs[merge_offset]
            return "merge", i, j
        return "answer", -1, -1

    def get_action(self, action_logits: torch.Tensor) -> Tuple[str, int, int]:
        """Decode the greedy action (legacy public API)."""
        if action_logits.ndim != 1:
            raise ValueError("get_action expects logits for one sample")
        return self.decode_action(int(torch.argmax(action_logits).item()))

    def forward(
        self,
        world_states: torch.Tensor,
        remaining_budget: torch.Tensor,
        return_value: bool = False,
        active_mask: Optional[torch.Tensor] = None,
    ):
        if world_states.ndim == 2:
            world_states = world_states.unsqueeze(0)
        if world_states.shape[1:] != (self.num_worlds, self.hidden_size):
            raise ValueError(
                "world_states must have shape "
                f"[batch, {self.num_worlds}, {self.hidden_size}]"
            )
        batch_size = world_states.size(0)
        remaining_budget = torch.as_tensor(
            remaining_budget, dtype=world_states.dtype, device=world_states.device
        )
        if remaining_budget.ndim == 0:
            remaining_budget = remaining_budget.expand(batch_size)
        if remaining_budget.shape != (batch_size,):
            raise ValueError("remaining_budget must be scalar or have shape [batch]")

        if active_mask is None:
            active_mask = torch.ones(
                batch_size,
                self.num_worlds,
                dtype=torch.bool,
                device=world_states.device,
            )
        elif active_mask.shape != world_states.shape[:2]:
            raise ValueError("active_mask must have shape [batch, num_worlds]")

        budget_scale = max(self.compute_budget, 1.0)
        normalized_budget = (remaining_budget / budget_scale).unsqueeze(-1)
        world_tokens = self.world_projection(world_states)
        world_tokens = world_tokens + self.world_slot_embeddings.to(
            dtype=world_tokens.dtype
        ).unsqueeze(0)
        budget_token = self.budget_projection(normalized_budget).unsqueeze(1)
        tokens = torch.cat((world_tokens, budget_token), dim=1)
        padding_mask = torch.cat(
            (
                ~active_mask,
                torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=world_states.device
                ),
            ),
            dim=1,
        )
        encoded = self.policy_network(tokens, src_key_padding_mask=padding_mask)
        encoded_worlds = encoded[:, : self.num_worlds]
        encoded_budget = encoded[:, self.num_worlds]
        active_weights = active_mask.to(encoded_worlds.dtype).unsqueeze(-1)
        pooled = (encoded_worlds * active_weights).sum(dim=1) / active_weights.sum(
            dim=1
        ).clamp_min(1.0)

        expand_logits = self.expand_head(encoded_worlds).squeeze(-1)
        stop_logits = self.stop_head(encoded_worlds).squeeze(-1)
        merge_logits = []
        for i, j in self.merge_pairs:
            left = encoded_worlds[:, i]
            right = encoded_worlds[:, j]
            pair_features = torch.cat(
                (left, right, (left - right).abs(), left * right), dim=-1
            )
            merge_logits.append(self.merge_head(pair_features).squeeze(-1))
        if merge_logits:
            merge_logits_tensor = torch.stack(merge_logits, dim=1)
        else:
            merge_logits_tensor = encoded_worlds.new_empty((batch_size, 0))
        answer_logits = self.answer_head(torch.cat((pooled, encoded_budget), dim=-1))
        logits = torch.cat(
            (expand_logits, stop_logits, merge_logits_tensor, answer_logits), dim=1
        )
        if return_value:
            return logits, self.value_head(pooled).squeeze(-1)
        return logits

    def _validate_packed(
        self,
        packed_world_states: torch.Tensor,
        world_offsets: torch.Tensor,
        world_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
    ) -> int:
        if (
            packed_world_states.ndim != 2
            or packed_world_states.size(-1) != self.hidden_size
        ):
            raise ValueError(
                f"packed_world_states must have shape [active, {self.hidden_size}]"
            )
        if world_ids.shape != (packed_world_states.size(0),):
            raise ValueError("world_ids must align with packed_world_states")
        if world_offsets.ndim != 1 or world_offsets.numel() < 2:
            raise ValueError("world_offsets must have shape [batch + 1]")
        if int(world_offsets[0].item()) != 0 or int(world_offsets[-1].item()) != int(
            packed_world_states.size(0)
        ):
            raise ValueError("world_offsets must span all packed worlds")
        if bool((world_offsets[1:] < world_offsets[:-1]).any()):
            raise ValueError("world_offsets must be monotonic")
        if bool(((world_ids < 0) | (world_ids >= self.num_worlds)).any()):
            raise ValueError("world_ids contain an out-of-range stable identifier")
        batch_size = world_offsets.numel() - 1
        if remaining_budget.shape != (batch_size,):
            raise ValueError("remaining_budget must have shape [batch]")
        return batch_size

    def forward_packed(
        self,
        packed_world_states: torch.Tensor,
        world_offsets: torch.Tensor,
        world_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        return_value: bool = False,
    ):
        """Run the policy on genuinely variable-length world sequences.

        Each sample is encoded at its current active length rather than padded
        back to the initial ``K``. Logits are scattered into the stable action
        index space only after the variable-length Transformer computation.
        """
        batch_size = self._validate_packed(
            packed_world_states,
            world_offsets,
            world_ids,
            remaining_budget,
        )
        budget_scale = max(self.compute_budget, 1.0)
        normalized_budget = remaining_budget / budget_scale
        row_logits = []
        row_values = []
        for row in range(batch_size):
            start = int(world_offsets[row].item())
            end = int(world_offsets[row + 1].item())
            if end <= start:
                raise ValueError("policy cannot encode an empty active world set")
            states = packed_world_states[start:end]
            stable_ids = world_ids[start:end]
            world_tokens = self.world_projection(states)
            world_tokens = world_tokens + self.world_slot_embeddings.index_select(
                0, stable_ids
            ).to(dtype=world_tokens.dtype)
            budget_token = self.budget_projection(normalized_budget[row].reshape(1, 1))
            encoded = self.policy_network(
                torch.cat((world_tokens, budget_token), dim=0).unsqueeze(0)
            ).squeeze(0)
            encoded_worlds = encoded[:-1]
            encoded_budget = encoded[-1]
            pooled = encoded_worlds.mean(dim=0)

            logits = states.new_zeros(self.action_space_size)
            logits = logits.index_copy(
                0,
                stable_ids,
                self.expand_head(encoded_worlds).squeeze(-1),
            )
            logits = logits.index_copy(
                0,
                stable_ids + self.num_worlds,
                self.stop_head(encoded_worlds).squeeze(-1),
            )
            pair_logits = []
            pair_action_indices = []
            for left_index in range(stable_ids.numel()):
                for right_index in range(left_index + 1, stable_ids.numel()):
                    left = encoded_worlds[left_index]
                    right = encoded_worlds[right_index]
                    features = torch.cat(
                        (left, right, (left - right).abs(), left * right), dim=-1
                    )
                    pair_logits.append(self.merge_head(features).squeeze(-1))
                    pair_action_indices.append(
                        self.action_index(
                            "merge",
                            int(stable_ids[left_index].item()),
                            int(stable_ids[right_index].item()),
                        )
                    )
            if pair_logits:
                logits = logits.index_copy(
                    0,
                    torch.tensor(
                        pair_action_indices,
                        dtype=torch.long,
                        device=states.device,
                    ),
                    torch.stack(pair_logits),
                )
            logits = logits.index_copy(
                0,
                torch.tensor(
                    [self.answer_action_index],
                    dtype=torch.long,
                    device=states.device,
                ),
                self.answer_head(torch.cat((pooled, encoded_budget))).reshape(1),
            )
            row_logits.append(logits)
            row_values.append(self.value_head(pooled).squeeze(-1))
        stacked_logits = torch.stack(row_logits)
        if return_value:
            return stacked_logits, torch.stack(row_values)
        return stacked_logits

    def valid_action_mask_packed(
        self,
        packed_world_states: torch.Tensor,
        world_offsets: torch.Tensor,
        world_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        expand_cost: float,
        stop_cost: float,
        merge_cost: float,
        merge_topk_pairs: int = 0,
        enable_expand: bool = True,
        enable_stop: bool = True,
        enable_merge: bool = True,
    ) -> torch.Tensor:
        """Build action masks directly from the packed active set."""
        batch_size = self._validate_packed(
            packed_world_states,
            world_offsets,
            world_ids,
            remaining_budget,
        )
        mask = torch.zeros(
            batch_size,
            self.action_space_size,
            dtype=torch.bool,
            device=packed_world_states.device,
        )
        for row in range(batch_size):
            start = int(world_offsets[row].item())
            end = int(world_offsets[row + 1].item())
            states = packed_world_states[start:end]
            stable_ids = world_ids[start:end]
            budget = float(remaining_budget[row].item())
            if enable_expand and budget >= expand_cost:
                mask[row, stable_ids] = True
            if enable_stop and stable_ids.numel() > 1 and budget >= stop_cost:
                mask[row, stable_ids + self.num_worlds] = True
            if enable_merge and stable_ids.numel() > 1 and budget >= merge_cost:
                pair_entries = []
                normalized = F.normalize(states.float(), dim=-1)
                for left_index in range(stable_ids.numel()):
                    for right_index in range(left_index + 1, stable_ids.numel()):
                        action_index = self.action_index(
                            "merge",
                            int(stable_ids[left_index].item()),
                            int(stable_ids[right_index].item()),
                        )
                        similarity = (
                            normalized[left_index] * normalized[right_index]
                        ).sum()
                        pair_entries.append((action_index, similarity))
                if merge_topk_pairs > 0 and pair_entries:
                    similarities = torch.stack([entry[1] for entry in pair_entries])
                    keep = similarities.topk(
                        min(int(merge_topk_pairs), len(pair_entries))
                    ).indices.tolist()
                    pair_entries = [pair_entries[index] for index in keep]
                for action_index, _ in pair_entries:
                    mask[row, action_index] = True
            mask[row, self.answer_action_index] = True
        return mask

    def valid_action_mask(
        self,
        active_mask: torch.Tensor,
        remaining_budget: torch.Tensor,
        expand_cost: float,
        stop_cost: float,
        merge_cost: float,
        world_states: Optional[torch.Tensor] = None,
        merge_topk_pairs: int = 0,
        enable_expand: bool = True,
        enable_stop: bool = True,
        enable_merge: bool = True,
    ) -> torch.Tensor:
        """Build the legal action mask, optionally restricting merge by similarity.

        ``merge_topk_pairs`` keeps only the most cosine-similar active pairs per
        sample. Zero disables the restriction and is useful for action-space
        ablations; paper configs use one.
        """
        if active_mask.ndim != 2 or active_mask.size(1) != self.num_worlds:
            raise ValueError("active_mask must have shape [batch, num_worlds]")
        batch_size = active_mask.size(0)
        remaining_budget = remaining_budget.reshape(batch_size)
        mask = torch.zeros(
            batch_size,
            self.action_space_size,
            dtype=torch.bool,
            device=active_mask.device,
        )
        if enable_expand:
            mask[:, : self.num_worlds] = active_mask & (
                remaining_budget[:, None] >= expand_cost
            )
        # Stopping the last path would leave no distribution to aggregate.
        can_stop = (active_mask.sum(dim=-1, keepdim=True) > 1) & (
            remaining_budget[:, None] >= stop_cost
        )
        if enable_stop:
            mask[:, self.num_worlds : 2 * self.num_worlds] = active_mask & can_stop
        merge_valid = torch.zeros(
            batch_size,
            len(self.merge_pairs),
            dtype=torch.bool,
            device=active_mask.device,
        )
        for offset, (i, j) in enumerate(self.merge_pairs):
            merge_valid[:, offset] = (
                active_mask[:, i] & active_mask[:, j] & (remaining_budget >= merge_cost)
            )
        if enable_merge and merge_topk_pairs > 0 and self.merge_pairs:
            if world_states is None:
                raise ValueError(
                    "world_states are required when merge_topk_pairs is enabled"
                )
            normalized = F.normalize(world_states.float(), dim=-1)
            similarities = torch.stack(
                [
                    (normalized[:, i] * normalized[:, j]).sum(dim=-1)
                    for i, j in self.merge_pairs
                ],
                dim=1,
            )
            similarities = similarities.masked_fill(~merge_valid, float("-inf"))
            keep_count = min(int(merge_topk_pairs), len(self.merge_pairs))
            top_indices = similarities.topk(keep_count, dim=1).indices
            top_mask = torch.zeros_like(merge_valid)
            top_mask.scatter_(1, top_indices, True)
            finite_rows = merge_valid.any(dim=1, keepdim=True)
            merge_valid &= top_mask & finite_rows
        if enable_merge:
            start = 2 * self.num_worlds
            mask[:, start : start + len(self.merge_pairs)] = merge_valid
        # Answer is always legal, including at zero budget.
        mask[:, self.answer_action_index] = True
        return mask


class DLWMReasoningModule(nn.Module):
    """Generate worlds, execute controller actions, and aggregate candidates."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_worlds: int = 4,
        compute_budget: float = 10,
        lambda_orth: float = 0.1,
        lambda_steps: float = 0.05,
        lambda_worlds: float = 0.1,
        merge_mode: str = "score_softmax",
        merge_fixed_beta: float = 0.5,
        merge_temperature: float = 1.0,
        merge_topk_pairs: int = 1,
        expand_cost: float = 1.0,
        stop_cost: float = 0.1,
        merge_cost: float = 0.5,
        answer_cost: float = 0.0,
        dropout: float = 0.1,
        reasoner_layers: int = 6,
        reasoner_heads: int = 8,
        reasoner_ffn_multiplier: float = 4.0,
        world_generator_residual: bool = False,
        diversity_regularizer: str = "orthogonal",
        diversity_margin: float = 0.2,
        enable_expand: bool = True,
        enable_stop: bool = True,
        enable_merge: bool = True,
    ) -> None:
        super().__init__()
        if compute_budget < 0:
            raise ValueError("compute_budget must be non-negative")
        if min(expand_cost, stop_cost, merge_cost, answer_cost) < 0:
            raise ValueError("controller action costs must be non-negative")
        self.hidden_size = hidden_size
        self.num_worlds = num_worlds
        self.compute_budget = float(compute_budget)
        self.lambda_orth = lambda_orth
        self.lambda_steps = lambda_steps
        self.lambda_worlds = lambda_worlds
        self.expand_cost = float(expand_cost)
        self.stop_cost = float(stop_cost)
        self.merge_cost = float(merge_cost)
        self.answer_cost = float(answer_cost)
        self.merge_topk_pairs = int(merge_topk_pairs)
        self.diversity_regularizer = diversity_regularizer
        self.diversity_margin = float(diversity_margin)
        self.enable_expand = bool(enable_expand)
        self.enable_stop = bool(enable_stop)
        self.enable_merge = bool(enable_merge)
        if diversity_regularizer not in {
            "orthogonal",
            "cosine",
            "contrastive",
            "none",
        }:
            raise ValueError(
                "diversity_regularizer must be orthogonal, cosine, contrastive, or none"
            )
        if self.merge_topk_pairs < 0:
            raise ValueError("merge_topk_pairs cannot be negative")

        self.world_generator = LatentWorldGenerator(
            hidden_size,
            num_worlds,
            dropout=dropout,
            residual=world_generator_residual,
        )
        self.world_scorer = WorldScoringNetwork(hidden_size, dropout=dropout)
        self.world_merger = WorldMerger(
            hidden_size,
            mode=merge_mode,
            fixed_beta=merge_fixed_beta,
            temperature=merge_temperature,
        )
        self.policy = ResourceAwarePolicy(
            hidden_size, num_worlds, compute_budget, dropout=dropout
        )
        self.reasoning_module = SharedLatentReasoner(
            hidden_size=hidden_size,
            num_layers=reasoner_layers,
            num_heads=reasoner_heads,
            ffn_multiplier=reasoner_ffn_multiplier,
            dropout=dropout,
        )

    def forward(
        self,
        joint_repr: torch.Tensor,
        use_rl: bool = False,
        max_steps: int = 8,
        sample_actions: Optional[bool] = None,
        action_sequence: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        world_reprs = self.world_generator(joint_repr)
        initial_world_reprs = world_reprs
        orth_loss = self._compute_diversity_loss(initial_world_reprs)
        if use_rl:
            results = self._reason_with_rl(
                world_reprs,
                max_steps=max_steps,
                reference_reprs=joint_repr,
                sample_actions=(
                    self.training if sample_actions is None else sample_actions
                ),
                action_sequence=action_sequence,
            )
        else:
            results = self._reason_fixed_length(world_reprs, max_steps)

        packed_worlds = results["packed_world_reprs"]
        world_offsets = results["world_offsets"]
        packed_weights = self.world_scorer.score_packed(
            packed_worlds,
            world_offsets,
        )
        counts = world_offsets[1:] - world_offsets[:-1]
        batch_indices = torch.repeat_interleave(
            torch.arange(joint_repr.size(0), device=joint_repr.device),
            counts,
        )
        final_hidden = packed_worlds.new_zeros(
            (joint_repr.size(0), self.hidden_size)
        ).index_add(
            0,
            batch_indices,
            packed_worlds * packed_weights.unsqueeze(-1),
        )
        results.update(
            {
                "final_hidden": final_hidden,
                "packed_world_weights": packed_weights,
                # Backward-compatible alias; this tensor is packed, not [B, K].
                "world_weights": packed_weights,
                "orth_loss": orth_loss,
                "merge_beta_strategy": self.world_merger.mode,
                "merge_beta_definition": self.world_merger.beta_definition,
                "merge_beta_temperature": self.world_merger.temperature,
            }
        )
        return results

    def _compute_diversity_loss(self, world_reprs: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(world_reprs.float(), dim=-1)
        gram = normalized @ normalized.transpose(-1, -2)
        off_diagonal = ~torch.eye(
            self.num_worlds, dtype=torch.bool, device=world_reprs.device
        )
        pairwise = gram[:, off_diagonal]
        if pairwise.numel() == 0:
            return (gram.sum() * 0.0).to(world_reprs.dtype)
        if self.diversity_regularizer == "orthogonal":
            # Equation (4) is a sum over i != j; average only over the batch.
            loss = pairwise.square().sum(dim=-1).mean()
        elif self.diversity_regularizer == "cosine":
            loss = pairwise.abs().mean()
        elif self.diversity_regularizer == "contrastive":
            loss = F.relu(pairwise - self.diversity_margin).square().mean()
        else:
            loss = pairwise.sum() * 0.0
        return loss.to(world_reprs.dtype)

    # Kept for callers that used the previous private name.
    def _compute_orthogonal_loss(self, world_reprs: torch.Tensor) -> torch.Tensor:
        return self._compute_diversity_loss(world_reprs)

    def _reason_fixed_length(
        self,
        world_reprs: torch.Tensor,
        max_steps: int,
        batch_size: Optional[int] = None,
    ) -> Dict[str, object]:
        del batch_size  # kept for compatibility with the original signature
        batch = world_reprs.size(0)
        packed_worlds = world_reprs.reshape(batch * self.num_worlds, self.hidden_size)
        for _ in range(max_steps):
            packed_worlds = self.reasoning_module(packed_worlds)
        world_ids = torch.arange(
            self.num_worlds,
            dtype=torch.long,
            device=packed_worlds.device,
        ).repeat(batch)
        world_offsets = torch.arange(
            0,
            (batch + 1) * self.num_worlds,
            self.num_worlds,
            dtype=torch.long,
            device=packed_worlds.device,
        )
        empty_float = packed_worlds.new_zeros((batch, 0))
        empty_long = torch.empty(
            batch, 0, dtype=torch.long, device=packed_worlds.device
        )
        return {
            "packed_world_reprs": packed_worlds,
            "world_reprs": packed_worlds,
            "packed_world_ids": world_ids,
            "world_offsets": world_offsets,
            "remaining_budget": packed_worlds.new_full((batch,), self.compute_budget),
            "num_steps": torch.full(
                (batch,), max_steps, dtype=torch.long, device=packed_worlds.device
            ),
            "num_active_worlds": torch.full(
                (batch,),
                self.num_worlds,
                dtype=torch.long,
                device=packed_worlds.device,
            ),
            "avg_active_worlds": packed_worlds.new_full((batch,), self.num_worlds),
            "action_indices": empty_long,
            "trajectory_mask": empty_long.bool(),
            "trajectory_log_probs": empty_float,
            "trajectory_entropies": empty_float,
            "trajectory_values": empty_float,
            "merge_betas": empty_float,
            "merge_parent_ids": torch.empty(
                batch,
                0,
                2,
                dtype=torch.long,
                device=packed_worlds.device,
            ),
            "active_world_count_history": empty_long,
            "packed_storage_history": empty_long,
            "decoded_actions": [[] for _ in range(batch)],
        }

    def execute_actions(
        self,
        world_reprs: torch.Tensor,
        active_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        remaining_budget: torch.Tensor,
        actions: torch.Tensor,
        reference_reprs: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Execute one controller action per sample.

        Returns updated states/masks/budget, a done mask, and beta (NaN for
        non-merge actions). This public method makes action semantics directly
        testable and usable by alternative RL trainers.
        """
        batch_size = world_reprs.size(0)
        if actions.shape != (batch_size,):
            raise ValueError("actions must have shape [batch]")
        new_worlds = world_reprs.clone()
        new_active = active_mask.clone()
        new_candidates = candidate_mask.clone()
        new_budget = remaining_budget.clone()
        done = torch.zeros(batch_size, dtype=torch.bool, device=world_reprs.device)
        merge_betas = world_reprs.new_full((batch_size,), float("nan"))

        for batch_idx, action_idx in enumerate(actions.tolist()):
            action_type, i, j = self.policy.decode_action(action_idx)
            if action_type == "expand":
                expanded = self.reasoning_module(world_reprs[batch_idx, i])
                new_worlds[batch_idx, i] = expanded
                new_budget[batch_idx] -= self.expand_cost
            elif action_type == "stop":
                new_active[batch_idx, i] = False
                new_candidates[batch_idx, i] = False
                new_budget[batch_idx] -= self.stop_cost
            elif action_type == "merge":
                pair_scores = (
                    self.world_scorer.raw_scores(
                        torch.stack(
                            (world_reprs[batch_idx, i], world_reprs[batch_idx, j]),
                            dim=0,
                        )
                    )
                    if self.world_merger.mode == "score_softmax"
                    else (None, None)
                )
                reference = (
                    reference_reprs[batch_idx] if reference_reprs is not None else None
                )
                merged, beta = self.world_merger(
                    world_reprs[batch_idx, i],
                    world_reprs[batch_idx, j],
                    pair_scores[0],
                    pair_scores[1],
                    reference,
                )
                new_worlds[batch_idx, i] = merged
                new_active[batch_idx, j] = False
                new_candidates[batch_idx, j] = False
                merge_betas[batch_idx] = beta
                new_budget[batch_idx] -= self.merge_cost
            else:
                done[batch_idx] = True
                new_budget[batch_idx] -= self.answer_cost

        new_budget.clamp_(min=0.0)
        # The policy mask prevents stopping the final path. This guard also
        # terminates direct callers that supply an invalid final-stop action.
        done |= ~new_active.any(dim=-1)
        return (
            new_worlds,
            new_active,
            new_candidates,
            new_budget,
            done,
            merge_betas,
        )

    def execute_packed_actions(
        self,
        packed_world_reprs: torch.Tensor,
        world_offsets: torch.Tensor,
        packed_world_ids: torch.Tensor,
        remaining_budget: torch.Tensor,
        actions: torch.Tensor,
        reference_reprs: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Execute actions while physically removing stopped/merged tensor rows."""
        batch_size = self.policy._validate_packed(
            packed_world_reprs,
            world_offsets,
            packed_world_ids,
            remaining_budget,
        )
        if actions.shape != (batch_size,):
            raise ValueError("actions must have shape [batch]")
        if reference_reprs is not None and reference_reprs.shape != (
            batch_size,
            self.hidden_size,
        ):
            raise ValueError("reference_reprs must have shape [batch, hidden]")

        new_segments = []
        new_id_segments = []
        new_budget = remaining_budget.clone()
        done = torch.zeros(
            batch_size, dtype=torch.bool, device=packed_world_reprs.device
        )
        merge_betas = packed_world_reprs.new_full((batch_size,), float("nan"))
        merge_parent_ids = torch.full(
            (batch_size, 2),
            -1,
            dtype=torch.long,
            device=packed_world_reprs.device,
        )

        for row, action_idx in enumerate(actions.tolist()):
            start = int(world_offsets[row].item())
            end = int(world_offsets[row + 1].item())
            states = packed_world_reprs[start:end]
            stable_ids = packed_world_ids[start:end]
            action_type, world_i, world_j = self.policy.decode_action(action_idx)
            if action_type == "answer":
                done[row] = True
                new_budget[row] -= self.answer_cost
            else:
                match_i = (stable_ids == world_i).nonzero(as_tuple=False).flatten()
                if match_i.numel() != 1:
                    raise ValueError(
                        f"action references inactive world {world_i} in row {row}"
                    )
                index_i = int(match_i.item())
                if action_type == "expand":
                    expanded = self.reasoning_module(states[index_i])
                    states = states.index_copy(
                        0,
                        match_i,
                        expanded.reshape(1, -1),
                    )
                    new_budget[row] -= self.expand_cost
                elif action_type == "stop":
                    keep = torch.ones(
                        stable_ids.numel(),
                        dtype=torch.bool,
                        device=stable_ids.device,
                    )
                    keep[index_i] = False
                    states = states[keep]
                    stable_ids = stable_ids[keep]
                    new_budget[row] -= self.stop_cost
                else:
                    match_j = (stable_ids == world_j).nonzero(as_tuple=False).flatten()
                    if match_j.numel() != 1:
                        raise ValueError(
                            f"action references inactive world {world_j} in row {row}"
                        )
                    index_j = int(match_j.item())
                    if self.world_merger.mode == "score_softmax":
                        pair_scores = self.world_scorer.raw_scores(
                            torch.stack((states[index_i], states[index_j]))
                        )
                        utility_i, utility_j = pair_scores[0], pair_scores[1]
                    else:
                        utility_i = utility_j = None
                    reference = (
                        reference_reprs[row] if reference_reprs is not None else None
                    )
                    merged, beta = self.world_merger(
                        states[index_i],
                        states[index_j],
                        utility_i,
                        utility_j,
                        reference,
                    )
                    states = states.index_copy(
                        0,
                        match_i,
                        merged.reshape(1, -1),
                    )
                    keep = torch.ones(
                        stable_ids.numel(),
                        dtype=torch.bool,
                        device=stable_ids.device,
                    )
                    keep[index_j] = False
                    states = states[keep]
                    stable_ids = stable_ids[keep]
                    merge_betas[row] = beta
                    merge_parent_ids[row] = torch.tensor(
                        [world_i, world_j],
                        dtype=torch.long,
                        device=stable_ids.device,
                    )
                    new_budget[row] -= self.merge_cost

            if states.size(0) == 0:
                done[row] = True
            new_segments.append(states)
            new_id_segments.append(stable_ids)

        counts = torch.tensor(
            [segment.size(0) for segment in new_segments],
            dtype=torch.long,
            device=packed_world_reprs.device,
        )
        new_offsets = torch.cat(
            (
                counts.new_zeros(1),
                counts.cumsum(dim=0),
            )
        )
        new_budget.clamp_(min=0.0)
        return (
            torch.cat(new_segments, dim=0),
            new_offsets,
            torch.cat(new_id_segments, dim=0),
            new_budget,
            done,
            merge_betas,
            merge_parent_ids,
        )

    def _reason_with_rl(
        self,
        world_reprs: torch.Tensor,
        max_steps: int,
        batch_size: Optional[int] = None,
        reference_reprs: Optional[torch.Tensor] = None,
        sample_actions: bool = True,
        action_sequence: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        del batch_size  # compatibility with the old private method
        batch = world_reprs.size(0)
        device = world_reprs.device
        packed_world_reprs = world_reprs.reshape(
            batch * self.num_worlds, self.hidden_size
        )
        packed_world_ids = torch.arange(
            self.num_worlds,
            dtype=torch.long,
            device=device,
        ).repeat(batch)
        world_offsets = torch.arange(
            0,
            (batch + 1) * self.num_worlds,
            self.num_worlds,
            dtype=torch.long,
            device=device,
        )
        remaining_budget = world_reprs.new_full((batch,), self.compute_budget)
        done = torch.zeros(batch, dtype=torch.bool, device=device)
        num_steps = torch.zeros(batch, dtype=torch.long, device=device)
        active_sum = world_reprs.new_zeros(batch)
        decision_count = world_reprs.new_zeros(batch)

        action_history: List[torch.Tensor] = []
        mask_history: List[torch.Tensor] = []
        log_prob_history: List[torch.Tensor] = []
        entropy_history: List[torch.Tensor] = []
        value_history: List[torch.Tensor] = []
        beta_history: List[torch.Tensor] = []
        merge_parent_history: List[torch.Tensor] = []
        active_count_history: List[torch.Tensor] = []
        packed_storage_history: List[torch.Tensor] = []

        if action_sequence is not None:
            if action_sequence.ndim != 2 or action_sequence.size(0) != batch:
                raise ValueError("action_sequence must have shape [batch, steps]")
            if action_sequence.size(1) < max_steps:
                raise ValueError("action_sequence is shorter than max_steps")

        for step in range(max_steps):
            alive = ~done
            if not alive.any():
                break
            alive_rows = alive.nonzero(as_tuple=False).flatten()
            alive_state_segments = []
            alive_id_segments = []
            alive_counts = []
            for row in alive_rows.tolist():
                start = int(world_offsets[row].item())
                end = int(world_offsets[row + 1].item())
                alive_state_segments.append(packed_world_reprs[start:end])
                alive_id_segments.append(packed_world_ids[start:end])
                alive_counts.append(end - start)
            alive_counts_tensor = torch.tensor(
                alive_counts, dtype=torch.long, device=device
            )
            alive_offsets = torch.cat(
                (
                    alive_counts_tensor.new_zeros(1),
                    alive_counts_tensor.cumsum(dim=0),
                )
            )
            alive_worlds = torch.cat(alive_state_segments, dim=0)
            alive_world_ids = torch.cat(alive_id_segments, dim=0)
            alive_logits, alive_values = self.policy.forward_packed(
                alive_worlds,
                alive_offsets,
                alive_world_ids,
                remaining_budget.index_select(0, alive_rows),
                return_value=True,
            )
            alive_valid_actions = self.policy.valid_action_mask_packed(
                alive_worlds,
                alive_offsets,
                alive_world_ids,
                remaining_budget.index_select(0, alive_rows),
                expand_cost=self.expand_cost,
                stop_cost=self.stop_cost,
                merge_cost=self.merge_cost,
                merge_topk_pairs=self.merge_topk_pairs,
                enable_expand=self.enable_expand,
                enable_stop=self.enable_stop,
                enable_merge=self.enable_merge,
            )
            logits = packed_world_reprs.new_zeros(
                (batch, self.policy.action_space_size)
            ).index_copy(0, alive_rows, alive_logits)
            values = packed_world_reprs.new_zeros(batch).index_copy(
                0, alive_rows, alive_values
            )
            valid_actions = torch.zeros(
                batch,
                self.policy.action_space_size,
                dtype=torch.bool,
                device=device,
            )
            valid_actions[alive_rows] = alive_valid_actions
            # Finished rows are not part of the loss; selecting answer for them
            # keeps the distribution well-formed.
            valid_actions[done] = False
            valid_actions[done, self.policy.answer_action_index] = True
            masked_logits = logits.masked_fill(
                ~valid_actions, torch.finfo(logits.dtype).min
            )
            distribution = Categorical(logits=masked_logits)
            if action_sequence is not None:
                actions = action_sequence[:, step].to(device=device, dtype=torch.long)
                invalid = alive & ~valid_actions.gather(1, actions[:, None]).squeeze(1)
                if invalid.any():
                    raise ValueError(
                        f"forced invalid action at controller step {step}: "
                        f"rows {invalid.nonzero(as_tuple=False).flatten().tolist()}"
                    )
            elif sample_actions:
                actions = distribution.sample()
            else:
                actions = masked_logits.argmax(dim=-1)
            actions = torch.where(
                alive,
                actions,
                torch.full_like(actions, self.policy.answer_action_index),
            )

            log_probs = distribution.log_prob(actions)
            entropies = distribution.entropy()
            action_history.append(actions)
            mask_history.append(alive)
            log_prob_history.append(
                torch.where(alive, log_probs, torch.zeros_like(log_probs))
            )
            entropy_history.append(
                torch.where(alive, entropies, torch.zeros_like(entropies))
            )
            value_history.append(torch.where(alive, values, torch.zeros_like(values)))

            active_counts = world_offsets[1:] - world_offsets[:-1]
            active_count_history.append(torch.where(alive, active_counts, 0))
            packed_storage_history.append(torch.where(alive, active_counts, 0))
            active_sum += alive * active_counts.to(world_reprs.dtype)
            decision_count += alive.to(world_reprs.dtype)
            (
                packed_world_reprs,
                world_offsets,
                packed_world_ids,
                remaining_budget,
                action_done,
                betas,
                merge_parent_ids,
            ) = self.execute_packed_actions(
                packed_world_reprs,
                world_offsets,
                packed_world_ids,
                remaining_budget,
                actions,
                reference_reprs=reference_reprs,
            )
            beta_history.append(betas)
            merge_parent_history.append(merge_parent_ids)
            action_is_answer = actions.eq(self.policy.answer_action_index)
            num_steps += (alive & ~action_is_answer).long()
            done |= alive & action_done

        def stack_or_empty(
            values: Sequence[torch.Tensor], *, dtype: Optional[torch.dtype] = None
        ) -> torch.Tensor:
            if values:
                return torch.stack(tuple(values), dim=1)
            return torch.empty(
                batch,
                0,
                dtype=dtype or packed_world_reprs.dtype,
                device=device,
            )

        action_indices = stack_or_empty(action_history, dtype=torch.long)
        trajectory_mask = stack_or_empty(mask_history, dtype=torch.bool)
        decoded_actions: List[List[Tuple[str, int, int]]] = []
        for row in range(batch):
            decoded_actions.append(
                [
                    self.policy.decode_action(int(action_indices[row, col].item()))
                    for col in range(action_indices.size(1))
                    if bool(trajectory_mask[row, col])
                ]
            )

        avg_active = active_sum / decision_count.clamp_min(1.0)
        final_counts = world_offsets[1:] - world_offsets[:-1]
        return {
            "packed_world_reprs": packed_world_reprs,
            "world_reprs": packed_world_reprs,
            "packed_world_ids": packed_world_ids,
            "world_offsets": world_offsets,
            "remaining_budget": remaining_budget,
            "num_steps": num_steps,
            "num_active_worlds": final_counts,
            "avg_active_worlds": avg_active,
            "action_indices": action_indices,
            "trajectory_mask": trajectory_mask,
            "trajectory_log_probs": stack_or_empty(log_prob_history),
            "trajectory_entropies": stack_or_empty(entropy_history),
            "trajectory_values": stack_or_empty(value_history),
            "merge_betas": stack_or_empty(beta_history),
            "merge_parent_ids": (
                torch.stack(merge_parent_history, dim=1)
                if merge_parent_history
                else torch.empty(batch, 0, 2, dtype=torch.long, device=device)
            ),
            "active_world_count_history": stack_or_empty(
                active_count_history, dtype=torch.long
            ),
            "packed_storage_history": stack_or_empty(
                packed_storage_history, dtype=torch.long
            ),
            "decoded_actions": decoded_actions,
        }

    def compute_rl_reward(
        self,
        accuracy_reward,
        num_steps,
        num_active_worlds,
    ):
        """Compute ``R_acc - lambda_steps*T - lambda_worlds*K_avg``."""
        return (
            accuracy_reward
            - self.lambda_steps * num_steps
            - self.lambda_worlds * num_active_worlds
        )
