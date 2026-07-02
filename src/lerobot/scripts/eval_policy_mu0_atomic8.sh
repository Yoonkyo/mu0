#!/usr/bin/env bash
# Roll out the atomic8-trained Mu0 policy on each of the 8 tasks.
#
# Overridable via env:
#   CUDA_VISIBLE_DEVICES  (default 0)
#   TRACE_CKPT            (default final_ckpt)
#   DELTA_STATS           (default normalizer_stats.json)
#   POLICY_CKPT           (default outputs/robocasa/mu0_policy/atomic8_60k_s42/final)
#
# Run from the repo root: bash src/lerobot/scripts/eval_policy_mu0_atomic8.sh

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

for TASK in "${TASK_NAMES[@]}"; do
  echo ">>> Evaluating ${TASK}"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/lerobot/scripts/lerobot_eval_policy_mu0.py \
    --trace_checkpoint="${TRACE_CKPT:-final_ckpt}" \
    --delta_stats_path="${DELTA_STATS:-normalizer_stats.json}" \
    --policy_checkpoint="${POLICY_CKPT:-outputs/robocasa/mu0_policy/atomic8_60k_s42/final}" \
    --task_name="${TASK}" \
    --num_rollouts=50 \
    --num_videos_to_save=5 \
    --n_action_steps=8
done
