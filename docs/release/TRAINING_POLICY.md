# μ₀ as Policy — RoboCasa Training

This guide trains a **Action Expert** on top of the frozen **μ₀ trace
world model** using **RoboCasa** demonstrations. The action head learns
flow-matching action chunks conditioned on μ₀'s 3D-trace features.

> **TL;DR:** activate the `mu0` env from [TRAINING.md §1](TRAINING.md#1-environment) →
> install robosuite + robocasa from `third_party/` → download kitchen assets →
> point at your RoboCasa demos → run §4 to train, then §6 to roll out the
> trained policy in RoboCasa.

---

## 1. Environment

This guide builds on the `mu0` conda env created in
[TRAINING.md §1](TRAINING.md#1-environment). Reuse it; do **not** create a new
env.

```bash
conda activate mu0

# robosuite (master pin) and robocasa, both editable from the third_party/
# submodules. robosuite must come first — robocasa imports it.
pip install -e third_party/robosuite
pip install -e third_party/robocasa
pip install -e '.[dataset,training,smolvla]'
```

> NOTE: if numba/numpy collide after install, follow robocasa's upstream
> fallback `conda install -c numba numba=0.56.4 -y`.

---

## 2. RoboCasa kitchen assets

Simulation needs environment macros plus the kitchen asset bundle (~10 GB).

```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

---

## 3. Demonstration data

RoboCasa ships its demos **directly in the LeRobotDataset format**. The download lands a v2.1-codebase dataset on disk; one extra migration step upgrades it to the v3.0 layout this repo's
loader expects.

```bash
# 1) Download the pretraining human demos into datasets/v1.0/...
python -m robocasa.scripts.download_datasets --split pretrain --source human

# 2) Migrate each task dataset in place from LeRobot codebase v2.1 → v3.0
#    (creates episodes_stats.jsonl, removes the deprecated stats.json, and
#    bumps codebase_version in info.json). Run once per task root.
python src/lerobot/scripts/convert_dataset_v21_to_v30.py \
  --root=<path-to-task-dataset> \
  --push-to-hub=false
```

---

## 4. Train

The frozen μ₀ trace checkpoint (`final_ckpt/`) and normalization stats
(`normalizer_stats.json`) ship in the release bundle from the [main
README](../../README.md#release-artifacts) (`mu0_rollout.tar`). The command
below assumes they have been extracted into the repo root.

Three parallel lists select the demonstration data prepared in
[§3](#3-demonstration-data) — `--dataset_repo_ids`, `--dataset_roots`, and
`--task_names` must all have the **same length**. `--dataset_roots` points at
the migrated v3.0 dataset folders on disk; `--dataset_repo_ids` is the
string identifier used as a per-dataset label (no network access when roots
are local); `--task_names` lists the RoboCasa task IDs to include.

Training launches via `accelerate` across 4 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 --multi_gpu --mixed_precision=no \
  src/lerobot/scripts/lerobot_train_policy_mu0.py \
  --trace_checkpoint=final_ckpt \
  --delta_stats_path=normalizer_stats.json \
  --dataset_repo_ids='[<TBD>]' \
  --dataset_roots='[<TBD>]' \
  --task_names='[<TBD>]' \
  --output_dir=outputs/robocasa/mu0_policy \
  --camera_name=observation.images.robot0_agentview_left \
  --gripper_camera_name=observation.images.robot0_eye_in_hand \
  --action_dim=12 --chunk_size=16 --state_dim=16 \
  --gripper_spatial_pool_size=8 --use_trace=true \
  --training_steps=60000 --batch_size=4 --lr=1e-4 --weight_decay=0.01 \
  --grad_clip_norm=1.0 --warmup_steps=1000 \
  --log_freq=10 --save_freq=5000 --val_l1_freq=1500 \
  --num_workers=8 \
  --wandb_enable=false --wandb_project=robocasa \
  --seed=42 \
  --job_name=mu0_policy_main
```

---

## 5. Outputs, checkpoints, resume

- Runs write to `outputs/robocasa/mu0_policy/` (override with `--output_dir`).
- The final checkpoint is at `<output_dir>/final/checkpoint.pt`. Periodic
  checkpoints land in `<output_dir>/checkpoints/step_XXXXXXXX.pt` (every
  `--save_freq` steps).
- Resume with `--resume_from=<path-to-checkpoint.pt>` — it restores model,
  optimizer, scheduler, and step.

---

## 6. Evaluation

`lerobot_eval_policy_mu0.py` loads the same frozen μ₀ trace checkpoint plus
the trained action expert checkpoint produced in §4, runs N rollouts in the
RoboCasa env, and writes `eval_<task>.json` (plus optional MP4 rollouts).

```bash
CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_eval_policy_mu0.py \
  --trace_checkpoint=final_ckpt \
  --delta_stats_path=normalizer_stats.json \
  --policy_checkpoint=outputs/robocasa/mu0_policy/final \
  --task_name=PickPlaceCounterToCabinet \
  --num_rollouts=50 \
  --num_videos_to_save=5 \
  --n_action_steps=8
```

`--policy_checkpoint` accepts either a `checkpoint.pt` file or the directory
containing it — the training output's `final/` (or any `step_*/`) directory
can be passed directly. Results land in `<output_dir>/eval_<task>.json`,
defaulting to the policy checkpoint's parent directory.
