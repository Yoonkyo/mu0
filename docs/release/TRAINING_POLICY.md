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

Two training presets are wrapped as bash scripts. Both read the release-bundle
paths (`final_ckpt/`, `normalizer_stats.json`) by default and launch on 4 GPUs
via `accelerate`. The task list and hyper-parameters (steps, save/warmup
frequencies, dataset roots) are baked into each script; edit the arrays inside
to swap tasks.

```bash
# 8-task subset for 60k steps
bash src/lerobot/scripts/train_policy_mu0_atomic8.sh

# All 65 atomic tasks for 150k steps
bash src/lerobot/scripts/train_policy_mu0_atomic65.sh
```

Override paths via env vars if the release bundle lives elsewhere or you have
a different GPU layout:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TRACE_CKPT=path/to/final_ckpt \
DELTA_STATS=path/to/normalizer_stats.json \
  bash src/lerobot/scripts/train_policy_mu0_atomic8.sh
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

`eval_policy_mu0_atomic8.sh` rolls out the atomic-8 policy on each of the 8
tasks (50 rollouts × 8) and writes MP4s + `eval_<task>.json` per task under
the checkpoint's parent directory.

```bash
bash src/lerobot/scripts/eval_policy_mu0_atomic8.sh
```

The default `--policy_checkpoint` is
`outputs/robocasa/mu0_policy/atomic8_60k_s42/final`, which matches the local
training output from §4. Override via env vars to point at a different
checkpoint or release bundle:

```bash
POLICY_CKPT=path/to/final \
TRACE_CKPT=path/to/final_ckpt \
DELTA_STATS=path/to/normalizer_stats.json \
  bash src/lerobot/scripts/eval_policy_mu0_atomic8.sh
```

### Pre-trained policy

To skip §4 and evaluate directly, download the pre-trained release and extract
into `outputs/robocasa/` so the tree matches the training output layout:

```bash
wget https://huggingface.co/furonghuang-lab/mu0-policy/resolve/main/mu0_policy.tar
tar -xf mu0_policy.tar -C outputs/robocasa/
```

Resulting tree:

```
outputs/robocasa/mu0_policy/
├── atomic8_60k_s42/final/checkpoint.pt
└── atomic65_all_150k_s42/final/checkpoint.pt
```

With that in place, `eval_policy_mu0_atomic8.sh` works out of the box
(defaults to the atomic-8 checkpoint). To evaluate the atomic-65 checkpoint on
the same 8 tasks, override `POLICY_CKPT`:

```bash
POLICY_CKPT=outputs/robocasa/mu0_policy/atomic65_all_150k_s42/final \
  bash src/lerobot/scripts/eval_policy_mu0_atomic8.sh
```
