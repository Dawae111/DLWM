# DLWM reproducibility protocol

This partial development snapshot implements the following workflow. Its
included configurations are not the final configurations for the paper results,
and the original result checkpoints and logs are not included.

1. A frozen multimodal backbone encodes the image and question. During
   teacher-forced training, `z` is taken from the token immediately before the
   first supervised answer token, so it never consumes the gold answer.
2. A two-layer MLP and learned world identifiers create `K` hypotheses.
3. A shared Pre-LN Transformer stack advances each world independently. The
   included example configs use six blocks; `dlwm_reasoner_layers` controls
   the depth.
4. The controller executes `expand(i)`, `stop(i)`, `merge(i,j)`, or `answer`
   under the configured action costs. Merge candidates are restricted to the
   most cosine-similar active pair by default.
   Active states are held in a packed tensor with per-sample offsets. Stop and
   merge physically remove rows, and the policy Transformer runs separately on
   each current variable-length active set; stable IDs are retained only for
   action encoding and exact REINFORCE replay.
5. Every surviving world predicts every autoregressive answer token. The final
   distribution is `p(y)=sum_i alpha_i p_i(y)`. Expensive LM-head work is
   performed only for compact surviving worlds.
6. During generation, the controller runs once for the full answer. World
   states, weights, and budget decisions persist across output tokens. Beam
   search shares the prompt-level world cache across sibling beams and reorders
   a beam-to-world indirection together with the language-model KV cache.
7. Stage 2 generates a complete answer from a sampled controller trajectory,
   computes normalized exact-match correctness, and replays the exact action
   sequence with gradients for REINFORCE. Only controller parameters are
   trainable in Stage 2.

## Merge beta

The manuscript writes

`h_merge = beta_ij * h_i + (1 - beta_ij) * h_j`

and reports fixed beta 0.5. The included development configurations instead
default to `score_softmax`. The implementation records the selected strategy,
formula, temperature, parent IDs, and realized beta values. In `score_softmax`
mode, the formula is:

`beta_ij = softmax([s_psi(h_i), s_psi(h_j)] / temperature)[0]`

The utilities come from the Stage-1-trained world scoring network, which also
defines the final `alpha_i` weights. This is equivalent to the previous sigmoid
score-difference expression, keeps beta convex, and avoids training an
undocumented merge network during Stage 2. Available strategies are:

- `score_softmax`: pair-normalized Stage-1 scorer utilities (default);
- `reference_softmax`: pair-normalized cosine similarity to the multimodal
  encoding `z`;
- `fixed`: `dlwm_merge_fixed_beta`, for deterministic ablations.

## Official data and split safeguards

Edit `configs/data/dlwm_official_train.yaml`, then build the training manifest:

```bash
python scripts/prepare_dlwm_data.py \
  --config configs/data/dlwm_official_train.yaml
```

The builder accepts only GQA balanced train, VQAv2 train2014, and ScienceQA
train for training. It rejects MMVP and evaluation splits. MMVP therefore
remains zero-shot. VQAv2 uses COCO train2014/val2014 images.

## Training

```bash
bash scripts/finetune_dlwm_stage1_7b.sh
bash scripts/finetune_dlwm_stage2_7b.sh
```

Stage 1 optimizes task loss plus the exact pair-sum orthogonality objective.
Stage 2 starts from Stage 1 and optimizes only controller `omega` using the
generated-answer reward:

`R = R_acc - lambda_T * T - lambda_K * K_avg`.

The included example configs use `K=4`, `B=10`, `Tmax=8`,
`c_expand=1.0`, `c_stop=0.1`, `c_merge=0.5`, AdamW at `2e-5`, global
batch size 64 on eight workers, and no undocumented entropy bonus.

## Evaluation

```bash
python evaluation/evaluation.py --config configs/eval/dlwm_7b.yaml
```

Evaluation uses deterministic greedy decoding and three warm-up batches before
timing. It writes benchmark-native prediction files and reports:

- GQA standard exact accuracy;
- the official VQAv2 leave-one-annotator-out consensus formula;
- ScienceQA option exact match;
- MMVP question accuracy and the official pair accuracy, where both questions
  in a pair must be correct.

If an official benchmark checkout is available, set
`official_evaluator_command` on that benchmark. `{predictions}` and
`{benchmark_dir}` are expanded without invoking a shell, and stdout/stderr are
saved beside the metrics.

Per-example logs include controller actions, action indices, beta, surviving
world IDs, beta strategy/definition and parent IDs, packed world-count history,
per-world top tokens, alpha, steps, active-world count, latency, and peak
memory. Greedy decoding remains the paper default; set
`generation.num_beams` above one to evaluate the beam-safe persistent cache.
Render qualitative trajectories with:

```bash
python evaluation/visualize_rollouts.py \
  --predictions evaluation/results/dlwm_7b/MMVP/predictions.jsonl \
  --output-dir evaluation/results/dlwm_7b/MMVP/trajectories
```

## Multi-seed runs and ablations

```bash
python scripts/run_dlwm_multiseed.py \
  --stage1-config configs/train/dlwm_stage1_7b.yaml \
  --stage2-config configs/train/dlwm_stage2_rl_7b.yaml \
  --eval-config configs/eval/dlwm_7b.yaml \
  --seeds 41 42 43

python scripts/generate_dlwm_ablations.py \
  --matrix configs/ablations/dlwm_7b.yaml
```

The multi-seed runner produces per-seed configs and a mean/std summary. The
ablation matrix covers K, budget, each controller action, beta, aggregation,
and orthogonal/cosine/contrastive/no diversity regularization. Direct and
explicit-CoT baselines use the same evaluator by setting `method.type` to
`direct` or `explicit_cot`; external baseline result summaries can be combined
with `evaluation/compare_methods.py`. Predictions produced by the public
Coconut, CoLaR, LVR, LaRe, Heima, or Soft Thinking implementations can be
rescored under the identical adapters with `evaluation/score_predictions.py`;
this avoids copying or silently modifying third-party method code.

## Submission artifacts

The original paper-result checkpoint, trainer state, W&B export, and prediction
files are not included in this partial code release.

When original run artifacts are available, record them in
`artifacts/submission_manifest.yaml` and verify them with:

```bash
python scripts/verify_submission_artifacts.py
```

The verifier requires the checkpoint, both configs, trainer state, log export,
automatically written run manifest, git commit, evaluation summary, prediction
files, and an optional SHA-256 match. Until those original artifacts are
supplied, the repository does not claim reproduction of the manuscript's
numerical tables.
