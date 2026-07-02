#!/usr/bin/env bash
# Train the Mu0 policy on all 65 RoboCasa atomic tasks
#
# Overridable via env:
#   CUDA_VISIBLE_DEVICES  (default 0,1,2,3)
#   TRACE_CKPT            (default final_ckpt)
#   DELTA_STATS           (default normalizer_stats.json)
#
# Run from the repo root: bash src/lerobot/scripts/train_policy_mu0_atomic65.sh

set -euo pipefail

TASK_NAMES=(
  AdjustToasterOvenTemperature
  AdjustWaterTemperature
  CheesyBread
  CloseBlenderLid
  CloseCabinet
  CloseDishwasher
  CloseDrawer
  CloseElectricKettleLid
  CloseFridge
  CloseFridgeDrawer
  CloseMicrowave
  CloseOven
  CloseStandMixerHead
  CloseToasterOvenDoor
  CoffeeServeMug
  CoffeeSetupMug
  LowerHeat
  MakeIcedCoffee
  NavigateKitchen
  OpenBlenderLid
  OpenCabinet
  OpenDishwasher
  OpenDrawer
  OpenElectricKettleLid
  OpenFridge
  OpenFridgeDrawer
  OpenMicrowave
  OpenOven
  OpenStandMixerHead
  OpenToasterOvenDoor
  PackDessert
  PickPlaceCabinetToCounter
  PickPlaceCounterToBlender
  PickPlaceCounterToCabinet
  PickPlaceCounterToDrawer
  PickPlaceCounterToMicrowave
  PickPlaceCounterToOven
  PickPlaceCounterToSink
  PickPlaceCounterToStandMixer
  PickPlaceCounterToStove
  PickPlaceCounterToToasterOven
  PickPlaceDrawerToCounter
  PickPlaceFridgeDrawerToShelf
  PickPlaceFridgeShelfToDrawer
  PickPlaceMicrowaveToCounter
  PickPlaceSinkToCounter
  PickPlaceStoveToCounter
  PickPlaceToasterOvenToCounter
  PickPlaceToasterToCounter
  PreheatOven
  SlideDishwasherRack
  SlideOvenRack
  SlideToasterOvenRack
  StartCoffeeMachine
  TurnOffMicrowave
  TurnOffSinkFaucet
  TurnOffStove
  TurnOnBlender
  TurnOnElectricKettle
  TurnOnMicrowave
  TurnOnSinkFaucet
  TurnOnStove
  TurnOnToaster
  TurnOnToasterOven
  TurnSinkSpout
)

DATASET_ROOTS=(
  third_party/robocasa/datasets/v1.0/pretrain/atomic/AdjustToasterOvenTemperature/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/AdjustWaterTemperature/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CheesyBread/20250714/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseBlenderLid/20250822/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseCabinet/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseDishwasher/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseDrawer/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseElectricKettleLid/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseFridge/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseFridgeDrawer/20250821/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseMicrowave/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseOven/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseStandMixerHead/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CloseToasterOvenDoor/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CoffeeServeMug/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/CoffeeSetupMug/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/LowerHeat/20250805/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/MakeIcedCoffee/20250801/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenBlenderLid/20250822/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenDishwasher/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenElectricKettleLid/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenFridge/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenFridgeDrawer/20250821/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenMicrowave/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenOven/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenStandMixerHead/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/OpenToasterOvenDoor/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PackDessert/20250806/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCabinetToCounter/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToBlender/20250822/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToCabinet/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToDrawer/20250821/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToMicrowave/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToOven/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToStandMixer/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToStove/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToToasterOven/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceDrawerToCounter/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceFridgeDrawerToShelf/20250821/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceFridgeShelfToDrawer/20250821/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceMicrowaveToCounter/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceSinkToCounter/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceStoveToCounter/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceToasterOvenToCounter/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceToasterToCounter/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/PreheatOven/20250903/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/SlideDishwasherRack/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/SlideOvenRack/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/SlideToasterOvenRack/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/StartCoffeeMachine/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOffMicrowave/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOffSinkFaucet/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOffStove/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnBlender/20250822/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnElectricKettle/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnMicrowave/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnSinkFaucet/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnStove/20250819/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnToaster/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnOnToasterOven/20250820/lerobot
  third_party/robocasa/datasets/v1.0/pretrain/atomic/TurnSinkSpout/20250820/lerobot
)

# Build parallel lists for draccus (comma-separated inside [...]).
n=${#TASK_NAMES[@]}
if [[ "$n" != "${#DATASET_ROOTS[@]}" ]]; then
  echo "TASK_NAMES ($n) and DATASET_ROOTS (${#DATASET_ROOTS[@]}) length mismatch" >&2
  exit 1
fi
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
  --output_dir=outputs/robocasa/mu0_policy/atomic65_all_150k_s42 \
  --camera_name=observation.images.robot0_agentview_left \
  --gripper_camera_name=observation.images.robot0_eye_in_hand \
  --action_dim=12 --chunk_size=16 --state_dim=16 \
  --gripper_spatial_pool_size=8 --use_trace=true \
  --num_episodes=-1 \
  --training_steps=150000 --batch_size=8 --lr=1e-4 --weight_decay=0.01 \
  --grad_clip_norm=1.0 --warmup_steps=2500 \
  --log_freq=10 --save_freq=5000 --val_l1_freq=1500 \
  --num_workers=8 \
  --wandb_enable=true --wandb_project=robocasa --wandb_entity=tracegenv2 \
  --seed=42 \
  --job_name=mu0_policy_atomic65_all_150k_s42
