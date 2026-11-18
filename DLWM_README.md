# DLWM (Diverse Latent World Models)

Implementation notes for [DLWM: Diverse Latent World Models for Efficient Multimodal Reasoning](https://arxiv.org/abs/2606.15160).

This document describes the supplementary development snapshot. Its example
configurations are not a record of the final paper runs; see README.md and
REPRODUCIBILITY.md for the scope and unavailable result artifacts.

## Overview

DLWM extends latent visual reasoning (LVR) by introducing:
1. **Multi-hypothesis modeling**: Generate diverse latent world hypotheses from visual inputs
2. **Orthogonal regularization**: Encourage the latent world vectors to point in different directions
3. **Resource-aware RL**: Adaptive compute allocation for efficient reasoning
4. **Multi-world aggregation**: Combine predictions from multiple hypotheses

## Key Components

### 1. LatentWorldGenerator
Generates K diverse latent world representations from multimodal inputs. Each world represents a potential interpretation of the visual input.

**Features:**
- World embeddings for differentiation
- Orthogonal initialization
- MLP-based world generation

**Usage:**
```python
from src.model.dlwm_modules import LatentWorldGenerator

generator = LatentWorldGenerator(
    hidden_size=4096,
    num_worlds=4
)

world_reprs = generator(joint_repr)  # [batch_size, num_worlds, hidden_size]
```

### 2. WorldScoringNetwork
Scores different latent worlds for weighted aggregation.

**Features:**
- Learnable scoring function
- Softmax-normalized weights

**Usage:**
```python
from src.model.dlwm_modules import WorldScoringNetwork

scorer = WorldScoringNetwork(hidden_size=4096)
weights = scorer(world_reprs)  # [batch_size, num_worlds]
```

### 3. ResourceAwarePolicy
RL policy for adaptive compute allocation across worlds.

**Actions:**
- `expand(i)`: Expand reasoning for world i
- `stop(i)`: Stop reasoning for world i
- `merge(i,j)`: Merge worlds i and j
- `answer`: Generate final answer

For `merge(i,j)`, the implementation defines
`beta_ij = softmax([s_psi(h_i),s_psi(h_j)]/temperature)[0]` and returns
`beta_ij*h_i + (1-beta_ij)*h_j`. `dlwm_merge_mode=fixed` provides a fixed-beta mode.
`reference_softmax` is an additional documented similarity-to-`z` ablation.
The paper reports fixed beta 0.5; this development snapshot defaults to
`score_softmax`. The selected definition is persisted in the model config and
rollout logs.

**Usage:**
```python
from src.model.dlwm_modules import ResourceAwarePolicy

policy = ResourceAwarePolicy(
    hidden_size=4096,
    num_worlds=4,
    compute_budget=10
)

action_logits = policy(world_states, remaining_budget)
action_type, idx1, idx2 = policy.get_action(action_logits)
```

### 4. DLWMReasoningModule
Main module combining all components.

**Features:**
- Multi-world generation
- Orthogonal regularization (L_orth = Σ(w_i^T w_j)^2)
- Six-block Pre-LN Transformer reasoning shared across worlds and time
- Fixed-length or RL-based reasoning
- Physically packed variable-length active world states
- Full-answer token-level weighted aggregation
- Persistent world states across autoregressive generation
- Beam-safe cache sharing and reordering

## Training

### Stage 1: Supervised Pre-training

Train multi-world reasoning with orthogonal regularization.

**Script:**
```bash
bash scripts/finetune_dlwm_stage1_7b.sh
```

**Key Arguments:**
- `--enable_dlwm true`: Enable DLWM
- `--dlwm_num_worlds 4`: Number of latent worlds
- `--dlwm_compute_budget 10`: Total compute budget
- `--dlwm_lambda_orth 0.1`: Orthogonal regularization weight
- `--dlwm_use_rl false`: Use fixed-length reasoning (Stage 1)

**Loss Function:**
```
L_total = L_task + λ_orth * L_orth
```

### Stage 2: RL Fine-tuning

Fine-tune with resource-aware RL for adaptive compute allocation.

**Script:**
```bash
bash scripts/finetune_dlwm_stage2_7b.sh
```

**Key Arguments:**
- `--dlwm_use_rl true`: Enable RL policy
- `--dlwm_max_steps 8`: Maximum reasoning steps
- `--dlwm_stage stage2`: Enable the REINFORCE objective

**Reward Function:**
```
R = R_acc - λ_steps * T - λ_worlds * K_avg
```

Where:
- R_acc: Exact-match reward from a freely generated final answer
- T: Number of reasoning steps
- K_avg: Average number of active worlds over controller decisions

## Inference

### Basic Usage

```python
from transformers import AutoTokenizer, AutoProcessor
from src.model.qwen_dlwm_model import QwenWithDLWM

# Load model with DLWM
model = QwenWithDLWM.from_pretrained(
    "path/to/checkpoint",
    enable_dlwm=True,
    dlwm_num_worlds=4,
    dlwm_use_rl=True
)

# Generate with DLWM
outputs = model.generate_with_dlwm(
    inputs,
    generation_config=generation_config
)
```

### Generation Parameters

- `enable_dlwm`: Enable DLWM reasoning
- `dlwm_use_rl`: Use RL policy for compute allocation
- `dlwm_max_steps`: Maximum reasoning steps

## Configuration

### ModelArguments

```python
@dataclass
class ModelArguments:
    # DLWM configuration
    enable_dlwm: bool = False
    dlwm_num_worlds: int = 4
    dlwm_compute_budget: int = 10
    dlwm_lambda_orth: float = 0.1
    dlwm_lambda_steps: float = 0.05
    dlwm_lambda_worlds: float = 0.1
    dlwm_use_rl: bool = False
    dlwm_max_steps: int = 8
```

## Hyperparameters

### Recommended Settings

| Parameter | Value | Description |
|-----------|---------|-------------|
| `dlwm_num_worlds` | 4 | Number of latent worlds (K=4 provides good balance) |
| `dlwm_hidden_size` | 4096 | Paper latent hidden size |
| `dlwm_compute_budget` | 10 | Total compute budget |
| `dlwm_lambda_orth` | 0.1 | Orthogonal regularization weight |
| `dlwm_lambda_steps` | 0.05 | Steps penalty weight |
| `dlwm_lambda_worlds` | 0.1 | Active worlds penalty weight |
| `dlwm_max_steps` | 8 | Maximum reasoning steps |
| `dlwm_reasoner_layers` | 6 | Shared Pre-LN Transformer stack depth |
| `dlwm_merge_topk_pairs` | 1 | Most cosine-similar active merge pair |

### Ablation Studies

#### Number of Worlds (K)
- K=1: Single world (baseline)
- K=4: Recommended (good balance)
- K=6-8: Diminishing returns, higher cost

#### Compute Budget (B)
- B=5: Aggressive optimization, may hurt accuracy
- B=10: Recommended (balanced)
- B=15+: More computation, minimal accuracy gain

## Architecture

```
Input (Image + Text)
    ↓
Vision-Language Encoder
    ↓
Joint Representation z
    ↓
LatentWorldGenerator
    ↓
┌─────────┬─────────┬─────────┬─────────┐
│  w_1   │  w_2   │  w_3   │  w_4   │  (Diverse Worlds)
└─────────┴─────────┴─────────┴─────────┘
    ↓           ↓           ↓           ↓
  Reasoning  Reasoning  Reasoning  Reasoning
    ↓           ↓           ↓           ↓
  h^(1)_T   h^(2)_T   h^(3)_T   h^(4)_T
    └───────────┴───────────┴───────────┘
              ↓
        WorldScoringNetwork
              ↓
        Weighted Aggregation
              ↓
        Final Hidden State
              ↓
        Output Generation
```

## Algorithm

### Algorithm 1: DLWM Training Procedure

```
Require: training set D, encoder Enc, world generator G_φ,
         reasoning module f_θ, scoring head s_ψ, policy π_ω,
         number of worlds K, max rollout step T_max,
         compute budget B

Ensure: trained parameters {φ, θ, ψ, ω}

1: Initialize parameters φ, θ, ψ, ω
2: for each training iteration do
3:     Sample minibatch {(I, q, y)} from D
4:     for each sample (I, q, y) do
5:         z ← Enc(I, q)
6:         W ← {w_i = G_φ(z, e_i)}_i=1^K
7:         h_0^(i) ← w_i,  A ← {1, ..., K}
8:         t ← 0, B_t ← B,  Τ ← ∅
9:         while t < T_max and B_t > 0 and A ≠ ∅ do
10:            s_t ← {h_t^(i)}_i∈A, B_t
11:            a_t ~ π_ω(·|s_t)
12:            if a_t = expand(i) then
13:                h_{t+1}^(i) ← f_θ(h_t^(i))
14:                B_t ← B_t - c_expand
15:            else if a_t = stop(i) then
16:                A ← A \ {i}
17:                B_t ← B_t - c_stop
18:            else if a_t = merge(i, j) then
19:                w_merge ← α w_i + (1-α) w_j
20:                Replace w_i, w_j in A with w_merge
21:                B_t ← B_t - c_merge
22:            else if a_t = answer then
23:                break
24:            end if
25:            Store transition (s_t, a_t) in Τ
26:            t ← t + 1
27:        end while
28:        p_i(y) ← p(y|h_T^(i)), for i ∈ A
29:        α_i ← softmax(s_ψ(h_T^(i)))
30:        p(y|I, q) ← Σ_{i∈A} α_i p_i(y)
31:        L_task ← -log p(y|I, q)
32:        L_orth ← Σ_{i≠j} (w_i^⊤ w_j)^2
33:        R ← R_acc - λ_1 T - λ_2 K_avg
34:        L_RL ← -E_{τ_ω}[R]
35:        L ← L_task + λ_orth L_orth + λ_RL L_RL
36:        Update φ, θ, ψ by minimizing L_task + λ_orth L_orth
37:        Update ω by policy gradient on L_RL
38:     end for
39: end for
40: return {φ, θ, ψ, ω}
```

## Evaluation

### Metrics

1. **Task Accuracy**: Standard multimodal reasoning accuracy
2. **Orthogonal Loss**: L_orth = Σ(w_i^T w_j)^2
3. **Reasoning Steps**: Average number of steps T
4. **Active Worlds**: Average number of active worlds K
5. **GPU Memory**: Peak memory usage
6. **Throughput**: Samples per second

### Benchmarks

DLWM is evaluated on:
- **MMVP**: Visual perception and spatial reasoning
- **GQA**: Compositional and multi-hop reasoning
- **VQAv2**: Open-domain visual QA
- **ScienceQA**: Complex reasoning with cross-modal knowledge

## Results and ablations

The evaluation adapter writes task accuracy, controller steps, average active
worlds, peak GPU memory, throughput, action traces, beta values, and per-world
predictions. The submitted result checkpoint and logs are not present in this
checkout; see `REPRODUCIBILITY.md` before quoting or comparing result numbers.

## Troubleshooting

### Issue: High orthogonal loss

**Solution:**
- Increase `dlwm_lambda_orth` to enforce stronger diversity
- Check world initialization is orthogonal

### Issue: Too many reasoning steps

**Solution:**
- Increase `dlwm_lambda_steps` penalty
- Lower `dlwm_compute_budget`
- Ensure RL policy is learning

### Issue: Low diversity between worlds

**Solution:**
- Check world embeddings are properly initialized
- Increase `dlwm_lambda_orth`
- Verify orthogonal regularization is being applied

## License

Apache License 2.0

## Acknowledgments

This implementation extends the LVR framework:
- [LVR: Latent Visual Reasoning](https://arxiv.org/abs/2509.24251)
- Qwen2.5-VL model architecture
- DeepSpeed for distributed training
