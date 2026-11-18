from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments

try:
    from trl import GRPOConfig as GRPOConfigTRL
except ImportError:
    # TRL is only required by train_grpo.py. Keeping the common argument module
    # importable lets SFT/DLWM evaluation run in lightweight environments.
    GRPOConfigTRL = HFTrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")
    # set continuous reasoning mode
    coconut: bool = field(default=True)
    lvr_head: bool = field(default=False)
    lvr_head_type: str = field(default="simple")
    latent_end_token: bool = field(default=False)
    max_lvr_tokens: int = field(default=None)
    # DLWM (Diverse Latent World Models) configuration
    enable_dlwm: bool = field(
        default=False, metadata={"help": "Enable DLWM for multi-world latent reasoning"}
    )
    dlwm_num_worlds: int = field(
        default=4, metadata={"help": "Number of latent worlds to generate"}
    )
    dlwm_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Latent-world hidden size; defaults to backbone hidden size"},
    )
    dlwm_compute_budget: int = field(
        default=10, metadata={"help": "Total compute budget for RL"}
    )
    dlwm_lambda_orth: float = field(
        default=0.1, metadata={"help": "Weight for orthogonal regularization"}
    )
    dlwm_lambda_steps: float = field(
        default=0.05, metadata={"help": "Penalty weight for reasoning steps"}
    )
    dlwm_lambda_worlds: float = field(
        default=0.1, metadata={"help": "Penalty weight for active worlds"}
    )
    dlwm_use_rl: bool = field(
        default=False,
        metadata={"help": "Use RL policy for adaptive compute allocation"},
    )
    dlwm_max_steps: int = field(default=8, metadata={"help": "Maximum reasoning steps"})
    dlwm_merge_mode: str = field(
        default="score_softmax",
        metadata={"help": "Beta strategy: score_softmax, reference_softmax, or fixed"},
    )
    dlwm_merge_fixed_beta: float = field(
        default=0.5, metadata={"help": "Beta used by fixed merge mode"}
    )
    dlwm_merge_temperature: float = field(
        default=1.0, metadata={"help": "Temperature in pair-softmax beta"}
    )
    dlwm_merge_topk_pairs: int = field(
        default=1,
        metadata={
            "help": "Allow only the top-k most similar active merge pairs; 0 allows all"
        },
    )
    dlwm_expand_cost: float = field(default=1.0)
    dlwm_stop_cost: float = field(default=0.1)
    dlwm_merge_cost: float = field(default=0.5)
    dlwm_answer_cost: float = field(default=0.0)
    dlwm_dropout: float = field(default=0.1)
    dlwm_reasoner_layers: int = field(default=6)
    dlwm_reasoner_heads: int = field(default=8)
    dlwm_reasoner_ffn_multiplier: float = field(default=4.0)
    dlwm_world_generator_residual: bool = field(default=False)
    dlwm_diversity_regularizer: str = field(
        default="orthogonal",
        metadata={"help": "orthogonal, cosine, contrastive, or none"},
    )
    dlwm_diversity_margin: float = field(default=0.2)
    dlwm_enable_expand: bool = field(default=True)
    dlwm_enable_stop: bool = field(default=True)
    dlwm_enable_merge: bool = field(default=True)
    dlwm_prediction_chunk_size: int = field(default=8)
    dlwm_aggregation: str = field(
        default="probability_mixture",
        metadata={"help": "Final aggregation: probability_mixture or logit_mixture"},
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    loss_lvr_fct: str = field(default="mse")
    loss_lvr_lambda: float = field(default=1e-1)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768,  # This is the default value of the qwen2-vl model
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lvr_head_lr: Optional[float] = None
    lora_namespan_exclude: str = field(
        default=None, metadata={"help": "List of namespan to exclude for LoRA"}
    )
    num_lora_modules: int = -1
    # use_liger: bool = True
    run_name: Optional[str] = field(
        default="vscode debugger",
        metadata={"help": "Name of the run for logging purposes."},
    )
    # True if serving the checkpoints and data on oci
    online_checkpoint: Optional[bool] = False
    checkpoint_name: Optional[str] = None
    # data packing-related params
    enable_data_packing: bool = False
    max_packed_tokens: Optional[int] = None
    # long_seq_cut:Optional[int] = field(default=25600, metadata={"help": "Max Len of long single data instnace allowed"})
    long_seq_threshold: Optional[int] = field(
        default=4096, metadata={"help": "Threshold to be a long single instance"}
    )
    max_instance_per_batch: Optional[int] = 4
    max_steps: Optional[int] = 2500

    mode_switch_loss: Optional[bool] = False
    loss_mode_switch_fct: Optional[str] = field(default="mse")
    loss_mode_switch_lambda: Optional[float] = field(default=1e-1)

    # DLWM two-stage optimization. Stage 1 learns worlds with supervised LM
    # loss; stage 2 samples controller actions and adds REINFORCE loss.
    dlwm_stage: str = field(default="stage1")
    dlwm_rl_lambda: float = field(default=0.01)
    dlwm_rl_value_coef: float = field(default=0.5)
    dlwm_rl_entropy_coef: float = field(default=0.0)
    dlwm_rl_gamma: float = field(default=1.0)
    dlwm_rl_baseline: str = field(default="moving_average")
    dlwm_rl_baseline_momentum: float = field(default=0.9)
    dlwm_reward_mode: str = field(default="generated_exact_match")
    dlwm_stage2_controller_only: bool = field(default=True)
    dlwm_sample_actions: bool = field(default=True)
    dlwm_rollout_max_new_tokens: int = field(default=32)


@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(
        default=None, metadata={"help": "List of namespan to exclude for LoRA"}
    )
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512

    online_checkpoint: Optional[bool] = False
    checkpoint_name: Optional[str] = None
    decoding_strategy: str = "steps"
    lvr_steps: int = 16


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Optional evaluation data in the same format."}
    )
    dataset_format: str = field(
        default="sft", metadata={"help": "Dataset adapter: sft or lvr_bbox"}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
    random_seed: Optional[int] = field(default=None)
