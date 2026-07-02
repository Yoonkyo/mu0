#!/usr/bin/env bash
# Train the Mu0 policy on 8 RoboCasa atomic tasks.
#
# Overridable via env:
#   CUDA_VISIBLE_DEVICES  (default 0,1,2,3)
#   TRACE_CKPT            (default final_ckpt)
#   DELTA_STATS           (default normalizer_stats.json)
#
# Run from the repo root: bash src/lerobot/scripts/train_policy_mu0_atomic8.sh

set -euo pipefail

TASK_NAMES=(
  CloseFridge
  OpenFridge
  CoffeeServeMug
  PickPlaceFridgeShelfToDrawer
  TurnOnMicrowave
  SlideToasterOvenRack
  PickPlaceCounterToCabinet
  TurnOnToasterOven
)

DATASET_ROOTS=(
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenFridge/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CoffeeServeMug/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceFridgeShelfToDrawer/20250821/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnMicrowave/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/SlideToasterOvenRack/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToCabinet/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnToasterOven/20250820/lerobot
)

# Build parallel lists for draccus (comma-separated inside [...]).
n=${#TASK_NAMES[@]}
DATASET_REPO_IDS_STR=$(printf 'robocasa365,%.0s' $(seq 1 "$n"))
DATASET_REPO_IDS_STR=${DATASET_REPO_IDS_STR%,}
TASK_NAMES_STR=$(IFS=,; echo "${TASK_NAMES[*]}")
DATASET_ROOTS_STR=$(IFS=,; echo "${DATASET_ROOTS[*]}")

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} accelerate launch \
  --num_processes=4 --multi_gpu --mixed_precision=no \
  src/lerobot/scripts/lerobot_train_policy_mu0.py \
  --trace_checkpoint="${TRACE_CKPT:-final_ckpt}" \
  --delta_stats_path="${DELTA_STATS:-normalizer_stats.json}" \
  --dataset_repo_ids="[${DATASET_REPO_IDS_STR}]" \
  --dataset_roots="[${DATASET_ROOTS_STR}]" \
  --task_names="[${TASK_NAMES_STR}]" \
  --output_dir=outputs/robocasa/mu0_policy/atomic8_60k_s42 \
  --camera_name=observation.images.robot0_agentview_left \
  --gripper_camera_name=observation.images.robot0_eye_in_hand \
  --action_dim=12 --chunk_size=16 --state_dim=16 \
  --gripper_spatial_pool_size=8 --use_trace=true \
  --num_episodes=100 \
  --training_steps=60000 --batch_size=8 --lr=1e-4 --weight_decay=0.01 \
  --grad_clip_norm=1.0 --warmup_steps=1000 \
  --log_freq=10 --save_freq=10000 --val_l1_freq=1500 \
  --num_workers=8 \
  --wandb_enable=true --wandb_project=robocasa --wandb_entity=tracegenv2 \
  --seed=42 \
  --job_name=mu0_policy_atomic8_60k_s42
