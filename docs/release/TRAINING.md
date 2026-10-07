# μ₀ — Training




This guide installs the environment and trains the **μ₀trace-prediction**
model used in our work. Given `(image, language, history of N keypoints)`, the
model predicts the future trace of those keypoints with a flow-matching head.

> **TL;DR:** create the conda env → (optionally) precompute normalization stats →
> run the single-GPU training command in §4 on one or two episodes.

> **Training data.** The instructions below train on **TraceExtract episodes** —
> the 3D keypoint-trace data produced by our extraction pipeline,
> [**TraceExtract**](https://github.com/Yoonkyo/TraceExtract). Use it to convert
> your own videos and robot datasets into episodes, or start right away with our
> [sample training set](#sample-training-set) (§2).

---

## 1. Environment

```bash
# 1. Python 3.12 conda env
conda create -n mu0 python=3.12 -y
conda activate mu0

# 2. PyTorch with the CUDA build matching your system (CUDA 12.8 example).
#    Pin to the range the repo requires (torch>=2.7,<2.11) so the editable
#    install in step 3 leaves this CUDA build in place instead of pulling the
#    default PyPI wheel over it.
pip install 'torch>=2.7,<2.11' torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. This repo, editable, with dataset + training + μ₀ backbone extras.
pip install -e '.[dataset,training,smolvla]'

# 4. (Optional) Authenticate. The SmolVLM2 and DINOv2 backbones are public and
#    download on first run without a token; `hf auth login` only raises HF
#    download rate limits (you may see an "unauthenticated requests" warning
#    otherwise — it is harmless).
hf auth login        # optional — higher HF download rate limits
wandb login          # optional, for training-time curves + overlays
```

---

## 2. Data layout

Training reads **TraceExtract episode directories**. The files the data loader
actually consumes are:

```
<episode>/
  images.npy                 # (T, H, W, 3) uint8 RGB frames        [required]
  depth.npy                  # (T, H, W)   float16 metric depth (m) [required when --use_depth=true]
  curated_training_texts.json # language annotations                [optional — drops captions if absent]
  samples/
    frame_indices.npy        # per-slot frame index                 [required]
    offsets.npy              # per-slot keypoint offsets             [required]
    is_moving.npy            # per-keypoint moving flag             [required]
    traj.npy                 # current+future keypoint trajectory   [required]
    traj_history.npy         # current+past keypoint trajectory     [required]
    valid_steps.npy          # future validity mask                 [required]
    valid_steps_history.npy  # history validity mask               [required]
    cluster_ids.npy          # per-keypoint DINO cluster id        [required only when rigidity loss is on]
```

TraceExtract episodes also ship extra arrays (`raw_traj*.npy`, `raw_valid_steps*.npy`,
`keypoints.npy`, `visibs.npy`, `cameras.npz`, `acceleration.npz`,
`movement_statistics.npz`, `description.txt`); the trainer ignores these, so they
are safe to keep or drop.

Pass one or more episode dirs (or globs) via `--video_dirs` pointing at your
[TraceExtract](https://github.com/Yoonkyo/TraceExtract) output.

### Sample training set

To try training without running TraceExtract yourself, download our sample
training set: 20 [DROID](https://droid-dataset.github.io) episodes already
processed by TraceExtract, with language annotations (~750 MB; not part of the
evaluation set).

```bash
hf download furonghuang-lab/mu0 sample_train_set.tar normalizer_stats.json --local-dir mu0_release
cd mu0_release && tar -xf sample_train_set.tar    # extracts to sample_train_set/droid/<episode>/
```

Then use the episodes as the `--video_dirs` of §3 / §4, e.g. in §4:

```bash
  --video_dirs='[/path/to/mu0_release/sample_train_set/droid/*]' \
  --delta_stats_path=/path/to/mu0_release/normalizer_stats.json \
```

`normalizer_stats.json` holds the normalization stats the released checkpoint was
trained with; alternatively, omit `--delta_stats_path` to compute stats from the
sample episodes (§3).

---

## 3. (Optional) Precompute normalization stats

The model predicts **anchor-relative deltas** normalized by per-axis scales, and
renders depth over a fixed log range. Both live in one JSON. Precompute once and
reuse with `--delta_stats_path`, or omit it to auto-compute into the run's output
dir on first launch.

```bash
python -m lerobot.datasets.trace_delta_stats \
    --video_dirs '/path/to/data_part1/*' '/path/to/data_part2/*' \
    --history_len 8 --future_len 32 --video_fraction 0.1 \
    --output_path /path/to/delta_stats.json
```

The stats JSON records `target_kind="anchor"` — the anchor-relative delta target
the golden recipe requires (the trainer validates this and errors on a mismatch).

---

## 4. Train — single GPU

This is **run2**, the main pretraining recipe, sized for one 48 GB GPU
(`batch_size=2` + gradient checkpointing). On a single GPU you launch the script
directly with `python` — no `accelerate launch` needed.

```bash
CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_train_trace_mu0.py \
  --video_dirs='[/path/to/episode_a, /path/to/episode_b]' \
  --delta_stats_path=/path/to/delta_stats.json \
  --batch_size=2 --num_workers=4 \
  --n_min=1 --n_max=256 \
  --image_size=512 \
  --steps=200000 --log_freq=50 --save_freq=3000 \
  --eval_every_n_steps=3000 --eval_episode_num_per_dir=1 --vis_num_samples=5 \
  --device=cuda --dtype=bfloat16 \
  --gradient_checkpointing=true \
  --num_inference_steps=4 \
  --load_vlm_weights=true \
  --vlm_model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --num_vlm_layers=20 --num_expert_layers=20 --expert_width_multiplier=0.5 \
  --freeze_vision_encoder=true --train_expert_only=true \
  --lr=1e-4 \
  --wandb_enable=false --wandb_project=mu0 \
  --history_len=8 --future_len=32 \
  --trace_group_horizon=8 \
  --use_depth=true --depth_lora_rank=8 --depth_lora_alpha=16 \
  --use_dino=true --dino_model_name=facebook/dinov2-base --dino_input_size=518 \
  --trace_history_full_mask_prob=0.5 --depth_dropout_prob=0.7 \
  --trace_history_kp_dropout_prob=0.3 \
  --rgb_jitter_strength=0.3 --depth_noise_std=0.01 \
  --trace_bspline_n_ctrl=10 --trace_bspline_ctrl_per_token=10 \
  --trace_done_head=true --done_loss_weight=0.5 \
  --rigidity_regularization_weight=5 \
  --trace_bspline_reg_lambda=0.2 --trace_bspline_reg_order=1 --trace_bspline_ctrl_clip=1.5 \
  --seed=0 \
  --job_name=trace_main
```

To just confirm the pipeline runs end-to-end on one or two episodes, add
`--steps=4 --save_checkpoint=false --eval_every_n_steps=0` — it does a couple of
optimizer steps and exits in a few minutes.

If you hit OOM, drop `--batch_size` to 1; if you have headroom, raise it (batch
size is **per GPU** and does not change the recipe).

---

## 5. Train — multi-GPU (full scale)

The full pretraining run uses 4 GPUs via `accelerate`. It is identical to §4
except for the launcher and the data/throughput knobs (`batch_size`,
`num_workers`, `steps` — none of which change the model recipe):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 --multi_gpu --mixed_precision=no \
  src/lerobot/scripts/lerobot_train_trace_mu0.py \
  --video_dirs='[/path/to/data/*]' \
  --batch_size=4 --num_workers=8 \
  ... (all the same recipe flags as §4) ...
  --job_name=trace_main
```

---

## 6. Outputs, checkpoints, resume

- Runs write to `outputs/train/<date>/<time>_<job_name>/` (override with
  `--output_dir`). Checkpoints land in `checkpoints/step_XXXXXXXX/`, each holding
  `model.safetensors`, `config.json`, `meta.json`, and `accel_state/`.
- The resolved `delta_stats.json` is copied into the run dir so it is
  self-contained for later inference.
- Resume a run with `--resume_from=outputs/train/.../checkpoints/step_XXXXXXXX`
  (restores model + optimizer + RNG + step). Add `--resume_new_run=true` to
  branch into a fresh dir/W&B run instead of continuing in place.
