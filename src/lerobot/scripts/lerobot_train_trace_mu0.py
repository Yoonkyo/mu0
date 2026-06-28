#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Trace-prediction training entry point for SmolVLA (docs/trace_prediction_smolvla_plan.md §7).

Builds a `SmolVLAPolicy` with `trace_mode=True`. Tokenizer, image size, and LR
defaults match SmolVLA.
"""

import datetime as dt
import glob
import json
import logging
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import broadcast_object_list, set_seed as _accelerate_set_seed
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Avoid `/dev/shm`-backed tensor IPC. The default ("file_descriptor") routes every
# DataLoader worker→main tensor through a posix shared-memory segment under
# `/dev/shm`; on shared cluster nodes (e.g. cml30) per-host or per-cgroup shm
# limits trigger the canonical "ERROR: Unexpected bus error encountered in worker.
# This might be caused by insufficient shared memory (shm)." crash. Switching to
# "file_system" stages tensors as file-backed mmaps under /tmp instead, trading a
# few % data-loading throughput for reliability. Must be set before any
# DataLoader workers are spawned (i.e. before `accelerator.prepare`).
torch.multiprocessing.set_sharing_strategy("file_system")

from lerobot.datasets.trace_dataset import (
    CURRENT_KP_KEY,
    DEPTH_KEY,
    IMAGE_KEY,
    KEY_PAD_MASK_KEY,
    TASK_KEY,
    TRAJ_FUTURE_KEY,
    TRAJ_HISTORY_KEY,
    VALID_FUTURE_KEY,
    VALID_HISTORY_KEY,
    TraceDataset,
    trace_collate_fn,
)
from lerobot.datasets.trace_delta_stats import TraceStats, compute_trace_stats, load_trace_stats
from lerobot.policies.mu0.visualize_trace import (
    render_trace_motion_video,
    render_trace_overlay,
)
from lerobot.policies.mu0.configuration_smolvla import SmolVLAConfig
from lerobot.policies.mu0.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.logging_utils import AverageMeter
from lerobot.utils.utils import cycle, format_big_number, init_logging


@dataclass
class TraceTrainSmolVLAConfig:
    """Minimal config for SmolVLA trace-mode training."""

    video_dirs: list[str] = field(default_factory=list)

    # Trace-specific
    future_len: int = 32
    history_len: int = 16
    n_min: int = 1
    n_max: int = 32
    min_future_motion: float = 0.0  # drop keypoints whose future xy motion is below this in normalized coords.
    image_size: int = 512  # SmolVLM2 native; matches SmolVLAConfig.resize_imgs_with_padding.

    # Anchor-relative future targets: predict (future - current_kp) / delta_scale instead
    # of the absolute trajectory. delta_scale_{x,y,z} is a per-axis rescaling computed
    # from the dataset (default: 95th percentile of |delta|). xy deltas are measured in
    # normalized [-1, 1] coords; z deltas in raw meters.
    use_delta_target: bool = True
    delta_stats_path: Path | None = None  # if None, auto-compute to output_dir/delta_stats.json
    delta_stats_percentile: float = 95.0
    delta_stats_max_samples_per_video: int = 200_000

    # Rigidity regularization on predicted ctrl pts. For keypoints sharing a
    # DINO cluster id within a sample, penalize the variance of pairwise
    # ctrl-pt distances across the D ctrl-pt axis. 0 = off (no extra ops).
    # Auto-loads cluster_ids from TraceExtract `samples/cluster_ids.npy` when > 0.
    rigidity_regularization_weight: float = 0.0

    # Per-group "done" head (see docs/trace_done_head_plan.md). Default off.
    trace_done_head: bool = False
    done_loss_weight: float = 0.5
    done_bias_init: float = 3.0
    done_threshold: float = 0.5

    # Optimization
    batch_size: int = 8
    num_workers: int = 4
    steps: int = 200_000
    lr: float = 1e-4  # SmolVLAConfig.optimizer_lr default.
    vlm_lr_multiplier: float = 1.0
    weight_decay: float = 1e-10
    grad_clip_norm: float = 10.0
    seed: int = 0

    # Device / dtype
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Freeze / finetune (plan §3.5 defaults: VLM frozen, expert + trace projections train).
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    load_vlm_weights: bool = True  # start from pretrained SmolVLM2 by default.

    # P2 tokenization: one token per (keypoint, group of `trace_group_horizon`
    # adjacent timesteps). See docs/trace_prediction_p2_plan.md.
    trace_group_horizon: int = 1

    # B-spline ctrl-pt FM target (see docs/trace_bspline_ctrl_pts_plan.md).
    # The dataset emits a (D, 3) lstsq fit of the future trajectory and the
    # policy runs FM in ctrl-pt space; spline is rendered to (H, 3) only at
    # inference.
    trace_bspline_n_ctrl: int = 10  # one of {4, 7, 10}
    trace_bspline_ctrl_per_token: int = 10  # must divide trace_bspline_n_ctrl
    # Tikhonov / ridge regularization on the per-kp lstsq fit in TraceDataset.
    # `lambda > 0` augments the basis with `lambda * Gamma` (k-th order finite
    # difference of consecutive ctrl pts) so adjacent ctrl pts are pulled
    # together (compresses the FM target distribution). 0 = no-op (current
    # behavior). Recommended starting point: 0.1, order=1.
    # `bspline_ctrl_clip` element-wise hard-clips P_gt to [-c, c] post-fit;
    # use to bound outliers in scaled-delta space (e.g. 1.5). None = off.
    trace_bspline_reg_lambda: float = 0.0
    trace_bspline_reg_order: int = 1
    trace_bspline_ctrl_clip: float | None = None

    # History dropout (suffix-side; see docs/trace_history_dropout_plan.md).
    # Both default 0.0 (no-op). Recommended 0.1 / 0.3 with tokenization=flat.
    trace_history_full_mask_prob: float = 0.0
    trace_history_kp_dropout_prob: float = 0.0

    # Data augmentation (training-only; eval/predict are never jittered/noised).
    # rgb_jitter_strength: passed to torchvision v2 ColorJitter as
    #   brightness=contrast=saturation=s, hue=min(s/2, 0.5). 0 = off.
    # depth_noise_std: additive Gaussian on raw meters (valid pixels only)
    #   BEFORE the turbo colormap, preserving meter→color mapping. 0 = off.
    rgb_jitter_strength: float = 0.0
    depth_noise_std: float = 0.0

    # Depth input (see docs/trace_depth_smolvla_plan.md). When `use_depth=True`, the
    # dataset also produces a turbo-rendered depth RGB as a second vision input; the
    # policy routes it through depth-only LoRA adapters + a depth-only cloned patch
    # embedding. Requires a stats JSON with `depth_log_min/max` populated (auto-computed
    # or precomputed via `python -m lerobot.datasets.trace_delta_stats`).
    use_depth: bool = False
    depth_lora_rank: int = 8
    depth_lora_alpha: int = 16
    depth_clone_stem: bool = True
    # Per-sample Bernoulli dropout on the depth image (train-only). When fired,
    # the depth `img_mask` row is zeroed so that sample's depth tokens become
    # non-attending in the prefix; the RGB branch is untouched. 0.0 = no-op.
    depth_dropout_prob: float = 0.0

    # DINO per-keypoint visual feature (see docs/trace_dino_feature_plan.md).
    # When `use_dino=True`, a frozen DINOv2 backbone runs once per forward on the
    # RGB frame; features are bilinearly sampled at each keypoint's last (u, v)
    # and concat-fused into every suffix token via a 2-layer MLP.
    use_dino: bool = False
    dino_model_name: str = "facebook/dinov2-base"
    dino_input_size: int = 518
    dino_num_prefix_tokens: int = 1

    # Policy-side
    pretrained_path: str | None = None  # optional lerobot/smolvla_base checkpoint path.
    # Resume support. Point at a `checkpoints/step_XXXXXXXX/` dir produced by a prior
    # run; resume restores model + optimizer + RNG state via `accelerator.load_state`
    # from `<resume_from>/accel_state/`, reads the step counter from
    # `<resume_from>/meta.json`, and (when wandb is enabled) continues the original
    # wandb run via the run id stored in `<output_dir>/wandb_run.json`. Output dir
    # defaults to `<resume_from>.parent.parent` so the resumed run reuses the same
    # delta_stats.json + checkpoints/ tree. Mutually exclusive with `pretrained_path`
    # (resume always restores from accel_state, never from a separately-loaded ckpt).
    resume_from: str | None = None
    # When True, a resume restores model/optimizer/RNG/step from `resume_from` but
    # writes into a *fresh* dated output dir and starts a *new* wandb run, leaving
    # the original run's dir, checkpoints, and wandb curve untouched (no overwrite
    # or pruning of the source run). Best-checkpoint tracking restarts from scratch.
    # Ignored when `output_dir` is set explicitly. Default False keeps the legacy
    # behavior of continuing the original run in-place.
    resume_new_run: bool = False
    # SmolVLM2 backbone (also used for the tokenizer). Common choices:
    #   HuggingFaceTB/SmolVLM2-256M-Video-Instruct
    #   HuggingFaceTB/SmolVLM2-500M-Video-Instruct  (default)
    #   HuggingFaceTB/SmolVLM2-2.2B-Instruct
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    tokenizer_max_length: int = 48  # SmolVLA default.
    num_inference_steps: int = 10
    gradient_checkpointing: bool = False

    # VLM/expert depth. num_vlm_layers slices the VLM text decoder to its first N
    # layers (-1 = use all). num_expert_layers must divide num_vlm_layers; <=0 means
    # the expert matches num_vlm_layers. Lower expert with more VLM = expert reads
    # KV from every (num_vlm_layers/num_expert_layers)-th VLM layer.
    num_vlm_layers: int = 16
    num_expert_layers: int = -1
    expert_width_multiplier: float = 0.75

    # Logging / checkpointing
    output_dir: Path | None = None
    job_name: str = "smolvla_trace"
    log_freq: int = 100
    save_freq: int = 5_000
    save_checkpoint: bool = True
    # When True, only persist a checkpoint when eval/loss improves on the running
    # best (the just-saved dir becomes the sole survivor; all others are pruned).
    # Non-improving save steps write nothing to disk. Requires eval to run at the
    # save step (eval_every_n_steps aligned with save_freq) to have a fresh
    # eval/loss to compare. Default False keeps the latest + best pair.
    save_best_only: bool = False

    # Eval + visualization (run together). Split is per-source: each video_dirs
    # glob is expanded and split independently so eval mirrors the train source mix.
    eval_fraction: float = 0.05  # fraction of dirs per source reserved for eval.
    # Fixed-N alternative to `eval_fraction`: when > 0, take exactly this many
    # episode dirs per video_dirs source for eval (clamped so train stays
    # non-empty). Use this when sources have very different sizes. Set to 0 to fall back to fractional splitting.
    eval_episode_num_per_dir: int = 0
    eval_every_n_steps: int = 1_000  # 0 disables eval + vis.
    eval_max_batches: int = 20  # cap on eval-loop batches to keep walltime bounded. 0 = full eval set.
    vis_num_samples: int = 4  # overlays logged per eval call.
    # Best-of-N evaluation (docs/minADE_minFDE_plan.md). Number of independent
    # noise draws per condition during eval; produces eval/minADE_{N}_* and
    # eval/minFDE_{N}_* alongside the legacy single-sample ADE/FDE/DTW
    # (computed on sample-0 to stay comparable with prior runs). 1 disables
    # the multi-sample branch (minADE_1 == ADE).
    eval_num_samples: int = 10

    # External test set (held-out, not split from `video_dirs`). One sample per
    # episode is used — the first usable slot of each video — so the test
    # dataset size == number of test episodes. Metrics + motion/vis_color
    # are logged under `test/` and `test_vis/` at the same cadence as eval.
    # Empty disables the test branch entirely (no extra walltime).
    test_video_dirs: list[str] = field(default_factory=list)

    # Wandb
    wandb_enable: bool = False
    wandb_project: str = "lerobot"
    wandb_entity: str | None = None
    wandb_mode: str | None = None

    # NaN diagnostics (default OFF — re-enable for debugging only).
    # When True, register per-leaf-module forward hooks that localize the first
    # non-finite output and dump the offending batch + state to
    # `output_dir/nan_dumps/step_<step>/` so the failing step can be replayed.
    # Adds ~5–15% step overhead from per-module checks. Pass `--detect_nan=true`
    # if loss starts NaN'ing again and you want to find which module first trips.
    #
    # The DDP-coordinated NaN-skip guard on loss + grad_norm runs regardless of
    # this flag — it costs ~50 μs/step and prevents a single bad batch from
    # poisoning the optimizer state.
    detect_nan: bool = False

    def __post_init__(self) -> None:
        if self.resume_from is not None and self.pretrained_path is not None:
            raise ValueError(
                "resume_from and pretrained_path are mutually exclusive: resume restores "
                "full model state from <resume_from>/accel_state/, ignoring pretrained_path."
            )
        if self.output_dir is None:
            if self.resume_from is not None and not self.resume_new_run:
                # Resume keeps writing into the original run dir so wandb_run.json,
                # delta_stats.json, and prior checkpoints stay coherent.
                self.output_dir = Path(self.resume_from).parent.parent
            else:
                # Fresh run, or a resume with resume_new_run=True: a new dated dir
                # (and, lacking a wandb_run.json, a brand-new wandb run).
                now = dt.datetime.now()
                self.output_dir = Path("outputs/train") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
        if not self.video_dirs:
            raise ValueError("video_dirs is empty — pass at least one TraceExtract video dir or glob.")


def _split_video_dirs(
    video_dirs: list[str],
    eval_fraction: float,
    seed: int,
    eval_episode_num_per_dir: int = 0,
) -> tuple[list[str], list[str]]:
    """Per-source train/eval split over episode directories.

    Each entry in `video_dirs` is treated as an independent source: the glob is
    expanded and shuffled with a seeded RNG, then the first slice of the shuffled
    matches go to eval. A source that resolves to a single dir is kept in train
    only (splitting would leave train empty for that source).

    Two split policies:
      * `eval_episode_num_per_dir > 0`: take exactly that many dirs from each
        source for eval (clamped to `len(matches) - 1` so train stays non-empty).
        Use this to keep the eval set balanced when sources have wildly
        different sizes — e.g. preventing an agibot source with 10k episodes
        from drowning out a droid source with 100.
      * Otherwise (default): take `round(eval_fraction * len(matches))`
        dirs per source, with a floor of 1 when the source has 2+ dirs.
    """
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in [0, 1): {eval_fraction}")
    if eval_episode_num_per_dir < 0:
        raise ValueError(
            f"eval_episode_num_per_dir must be >= 0: {eval_episode_num_per_dir}"
        )
    use_fixed_n = eval_episode_num_per_dir > 0
    train_dirs: list[str] = []
    eval_dirs: list[str] = []
    rng = random.Random(seed)
    for pattern in video_dirs:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_dir():
            matches = [pattern]
        if not matches:
            logging.warning(f"_split_video_dirs: pattern matched no dirs: {pattern}")
            continue
        shuffled = matches.copy()
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train_part, eval_part = shuffled, []
        elif use_fixed_n:
            n_eval = min(eval_episode_num_per_dir, len(shuffled) - 1)
            eval_part = shuffled[:n_eval]
            train_part = shuffled[n_eval:]
        elif eval_fraction == 0.0:
            train_part, eval_part = shuffled, []
        else:
            n_eval = max(1, int(round(len(shuffled) * eval_fraction)))
            n_eval = min(n_eval, len(shuffled) - 1)  # never empty train
            eval_part = shuffled[:n_eval]
            train_part = shuffled[n_eval:]
        train_dirs.extend(train_part)
        eval_dirs.extend(eval_part)
        logging.info(
            f"  source {pattern}: {len(matches)} dirs → "
            f"train={len(train_part)} eval={len(eval_part)}"
        )
    if not train_dirs:
        raise ValueError("split produced zero train dirs — check video_dirs patterns")
    return train_dirs, eval_dirs


def _build_policy(
    cfg: TraceTrainSmolVLAConfig,
    delta_scale: tuple[float, float, float] | None = None,
) -> SmolVLAPolicy:
    sv_cfg = SmolVLAConfig(
        trace_mode=True,
        vlm_model_name=cfg.vlm_model_name,
        history_len=cfg.history_len,
        future_len=cfg.future_len,
        n_min=cfg.n_min,
        n_max=cfg.n_max,
        num_steps=cfg.num_inference_steps,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        train_expert_only=cfg.train_expert_only,
        load_vlm_weights=cfg.load_vlm_weights,
        vlm_lr_multiplier=cfg.vlm_lr_multiplier,
        num_vlm_layers=cfg.num_vlm_layers,
        num_expert_layers=cfg.num_expert_layers,
        expert_width_multiplier=cfg.expert_width_multiplier,
        resize_imgs_with_padding=(cfg.image_size, cfg.image_size),
        tokenizer_max_length=cfg.tokenizer_max_length,
        trace_group_horizon=cfg.trace_group_horizon,
        trace_history_full_mask_prob=cfg.trace_history_full_mask_prob,
        trace_history_kp_dropout_prob=cfg.trace_history_kp_dropout_prob,
        optimizer_lr=cfg.lr,
        device=cfg.device,
        use_depth=cfg.use_depth,
        depth_lora_rank=cfg.depth_lora_rank,
        depth_lora_alpha=cfg.depth_lora_alpha,
        depth_clone_stem=cfg.depth_clone_stem,
        depth_dropout_prob=cfg.depth_dropout_prob,
        use_dino=cfg.use_dino,
        dino_model_name=cfg.dino_model_name,
        dino_input_size=cfg.dino_input_size,
        dino_num_prefix_tokens=cfg.dino_num_prefix_tokens,
        rigidity_loss_weight=cfg.rigidity_regularization_weight,
        trace_done_head=cfg.trace_done_head,
        done_loss_weight=cfg.done_loss_weight,
        done_bias_init=cfg.done_bias_init,
        done_threshold=cfg.done_threshold,
        trace_bspline_n_ctrl=cfg.trace_bspline_n_ctrl,
        trace_bspline_ctrl_per_token=cfg.trace_bspline_ctrl_per_token,
        delta_scale=tuple(delta_scale) if delta_scale is not None else None,
    )
    sv_cfg.validate_features()
    if cfg.pretrained_path is None:
        return SmolVLAPolicy(sv_cfg)
    # strict=False: trace branch has trace_* params; vanilla smolvla ckpt has action_* + state_proj.
    return SmolVLAPolicy.from_pretrained(cfg.pretrained_path, config=sv_cfg, strict=False)


def _make_tokenizer(cfg: TraceTrainSmolVLAConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    tok.padding_side = "right"  # matches embed_prefix's cumsum-based position ids.
    return tok


def _tokenize_and_device(
    batch: dict[str, Any],
    tokenizer,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    """Tokenize batch[task] + move tensors to device."""
    # breakpoint()
    prompts = [f"Task: {(t.strip() or 'predict future trajectory')}\n" for t in batch[TASK_KEY]]
    enc = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    out: dict[str, Any] = {k: v for k, v in batch.items() if k != TASK_KEY}
    out[OBS_LANGUAGE_TOKENS] = enc["input_ids"]
    out[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].bool()
    for k, v in out.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
    return out


class _FirstNaNDetector:
    """Forward-hook NaN/Inf detector across leaf submodules.

    Each forward queues an async `~isfinite(out).all()` 0-d tensor per leaf module.
    `find_first_nan()` stacks them and does ONE host sync, returning the name of
    the first module whose output was non-finite (or None if all finite). Designed
    to add minimal overhead during normal forward passes — the only sync happens
    when the caller explicitly asks "did any module NaN?".

    Reset between steps. Hooks survive accelerator.prepare (DDP wraps without
    replacing module instances).
    """

    def __init__(self, model: torch.nn.Module):
        self._checks: list[tuple[str, torch.Tensor]] = []
        self._handles: list[Any] = []
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self._record(name, output)
        return hook

    def _record(self, name: str, output: Any) -> None:
        if isinstance(output, torch.Tensor):
            if torch.is_floating_point(output):
                self._checks.append((name, ~torch.isfinite(output).all()))
        elif isinstance(output, (tuple, list)):
            for i, o in enumerate(output):
                self._record(f"{name}[{i}]", o)

    def find_first_nan(self) -> str | None:
        if not self._checks:
            return None
        flags = torch.stack([c[1] for c in self._checks]).cpu().tolist()
        names = [c[0] for c in self._checks]
        self._checks = []
        for i, flag in enumerate(flags):
            if flag:
                return names[i]
        return None

    def reset(self) -> None:
        self._checks = []

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def num_hooked_modules(self) -> int:
        return len(self._handles)


def _dump_nan_batch(
    dump_dir: Path,
    step: int,
    loss_value: float,
    first_nan_module: str | None,
    batch: dict[str, Any],
) -> None:
    """Persist a NaN-step batch + summary for offline replay.

    Writes:
      - `summary.json`: step, loss, first non-finite module, per-tensor stats,
        per-sample mask sums.
      - `batch.pt`: full batch tensors (CPU) so the failing step can be reloaded.
    """
    import json

    dump_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        if torch.is_floating_point(v):
            stats[k] = {
                "shape": tuple(v.shape),
                "dtype": str(v.dtype),
                "min": float(v.min().item()),
                "max": float(v.max().item()),
                "mean": float(v.mean().item()),
                "isnan": bool(torch.isnan(v).any().item()),
                "isinf": bool(torch.isinf(v).any().item()),
            }
        else:
            stats[k] = {
                "shape": tuple(v.shape),
                "dtype": str(v.dtype),
                "true_count": int(v.sum().item()) if v.dtype == torch.bool else None,
            }
    summary: dict[str, Any] = {
        "step": step,
        "loss": float(loss_value),
        "first_non_finite_module": first_nan_module,
        "batch_stats": stats,
    }
    if KEY_PAD_MASK_KEY in batch:
        summary["per_sample_n_kp"] = batch[KEY_PAD_MASK_KEY].sum(dim=1).cpu().tolist()
    if VALID_FUTURE_KEY in batch:
        summary["per_sample_valid_future"] = (
            batch[VALID_FUTURE_KEY].flatten(1).sum(dim=1).cpu().tolist()
        )
    if VALID_HISTORY_KEY in batch:
        summary["per_sample_valid_history"] = (
            batch[VALID_HISTORY_KEY].flatten(1).sum(dim=1).cpu().tolist()
        )
    (dump_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_dict = {k: v.detach().cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    torch.save(save_dict, dump_dir / "batch.pt")


def _resolve_trace_stats(
    cfg: "TraceTrainSmolVLAConfig", accelerator: Accelerator
) -> TraceStats:
    """Resolve the trace stats (delta scale + depth log range), auto-computing on main rank if needed.

    Side effect: writes/copies the stats JSON to `output_dir/delta_stats.json` so
    the run is self-contained for later inference-time inversion. Always reads through
    `load_trace_stats`, which hard-errors on pre-depth JSONs (no auto-migration).
    """
    is_main = accelerator.is_main_process
    auto_path = cfg.output_dir / "delta_stats.json"

    if cfg.delta_stats_path is not None:
        src = Path(cfg.delta_stats_path)
        if not src.exists():
            raise FileNotFoundError(f"delta_stats_path does not exist: {src}")
        if is_main and src.resolve() != auto_path.resolve():
            import shutil
            auto_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, auto_path)
    else:
        if is_main and not auto_path.exists():
            logging.info(
                f"Auto-computing trace stats (percentile={cfg.delta_stats_percentile}) "
                f"for {len(cfg.video_dirs)} video_dir pattern(s)..."
            )
            stats = compute_trace_stats(
                video_dirs=cfg.video_dirs,
                history_len=cfg.history_len,
                future_len=cfg.future_len,
                percentile=cfg.delta_stats_percentile,
                max_samples_per_video=cfg.delta_stats_max_samples_per_video,
                seed=cfg.seed,
            )
            import json
            auto_path.parent.mkdir(parents=True, exist_ok=True)
            with open(auto_path, "w") as f:
                json.dump(stats, f, indent=2)
            logging.info(
                f"trace_stats saved to {auto_path}: "
                f"sx={stats['delta_scale_x']:.4f} sy={stats['delta_scale_y']:.4f} "
                f"sz={stats['delta_scale_z']:.4f} over {stats['num_samples']} samples; "
                f"depth log_min={stats['depth_log_min']} log_max={stats['depth_log_max']} "
                f"({stats['depth_num_samples']} samples across {stats['depth_num_dirs']} dirs)"
            )

    accelerator.wait_for_everyone()
    ts = load_trace_stats(auto_path)
    if is_main:
        logging.info(
            f"delta_scale (sx, sy, sz) = {ts.delta_scale}; "
            f"target_kind = {ts.target_kind}; "
            f"depth_log_range = {ts.depth_log_range}"
        )
    if cfg.use_depth and ts.depth_log_range is None:
        raise ValueError(
            f"use_depth=True requires depth stats, but {auto_path} has null depth_log_* "
            f"(no dir had depth.npy). Regenerate on a dataset with depth.npy present."
        )
    return ts


# Per-axis dim selections for ADE/FDE/DTW. uvz mixes normalized [-1,1] xy with
# raw-meter z, matching the existing `l1_abs` aggregation.
_TRAJ_METRIC_DIMS: dict[str, tuple[int, ...]] = {
    "uvz": (0, 1, 2),
    "uv": (0, 1),
    "depth": (2,),
}


@torch.no_grad()
def _ade_fde_sums(
    pred_abs: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
    dim_idx: tuple[int, ...],
) -> tuple[float, float, int, int]:
    """Per-trajectory ADE (mean L2 over valid t) and FDE (L2 at last valid t).

    Returns (ade_sum, fde_sum, n_traj_ade, n_traj_fde) so the caller can mean
    across batches with the right denominator.
    """
    diff = (pred_abs.float() - gt_abs.float())[..., list(dim_idx)]
    dist = diff.pow(2).sum(-1).sqrt()  # (B, N, F)

    valid_t = valid_future & kpm.unsqueeze(-1)  # (B, N, F)
    valid_t_f = valid_t.to(dist.dtype)

    per_count = valid_t_f.sum(-1)  # (B, N)
    has_valid_ade = per_count > 0
    per_ade = (dist * valid_t_f).sum(-1) / per_count.clamp_min(1.0)
    ade_sum = float((per_ade * has_valid_ade.to(per_ade.dtype)).sum().item())
    n_ade = int(has_valid_ade.sum().item())

    f_len = valid_future.shape[-1]
    t_idx = torch.arange(f_len, device=valid_t.device).expand_as(valid_t)
    masked_t = torch.where(valid_t, t_idx, torch.full_like(t_idx, -1))
    last_t = masked_t.max(-1).values  # (B, N), -1 if no valid
    has_valid_fde = last_t >= 0
    last_t_safe = last_t.clamp_min(0)
    last_dist = dist.gather(-1, last_t_safe.unsqueeze(-1)).squeeze(-1)  # (B, N)
    fde_sum = float((last_dist * has_valid_fde.to(last_dist.dtype)).sum().item())
    n_fde = int(has_valid_fde.sum().item())

    return ade_sum, fde_sum, n_ade, n_fde


@torch.no_grad()
def _dtw_sums(
    pred_abs: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
    dim_idx: tuple[int, ...],
) -> tuple[float, int]:
    """Per-trajectory DTW with L2 cost on selected dims, normalized by valid length.

    Compresses non-contiguous valid timesteps into temporal order, groups
    trajectories by valid length, and runs a vectorized numpy DP per length-group
    so Python overhead is O(num_unique_lengths * T^2) rather than O(B*N*T^2).
    Returns (dtw_sum, n_traj).
    """
    valid_t = (valid_future & kpm.unsqueeze(-1)).cpu().numpy()  # (B, N, F)
    pred_np = pred_abs[..., list(dim_idx)].detach().to(torch.float32).cpu().numpy()
    gt_np = gt_abs[..., list(dim_idx)].detach().to(torch.float32).cpu().numpy()
    b_dim, n_dim, _, _ = pred_np.shape

    by_length: dict[int, tuple[list, list]] = {}
    for b in range(b_dim):
        for n in range(n_dim):
            m = valid_t[b, n]
            t_valid = int(m.sum())
            if t_valid == 0:
                continue
            preds_l, gts_l = by_length.setdefault(t_valid, ([], []))
            preds_l.append(pred_np[b, n][m])
            gts_l.append(gt_np[b, n][m])

    if not by_length:
        return 0.0, 0

    total = 0.0
    count = 0
    for t_valid, (preds_l, gts_l) in by_length.items():
        m_count = len(preds_l)
        pred_arr = np.stack(preds_l, axis=0)  # (M, T, K)
        gt_arr = np.stack(gts_l, axis=0)
        if t_valid == 1:
            d = np.linalg.norm(pred_arr[:, 0, :] - gt_arr[:, 0, :], axis=-1)
            total += float(d.sum())
            count += m_count
            continue
        diff = pred_arr[:, :, None, :] - gt_arr[:, None, :, :]  # (M, T, T, K)
        pdist = np.linalg.norm(diff, axis=-1).astype(np.float64)  # (M, T, T)
        cost = np.full((m_count, t_valid + 1, t_valid + 1), np.inf, dtype=np.float64)
        cost[:, 0, 0] = 0.0
        for i in range(1, t_valid + 1):
            for j in range(1, t_valid + 1):
                d = pdist[:, i - 1, j - 1]
                best = np.minimum(
                    np.minimum(cost[:, i - 1, j], cost[:, i, j - 1]),
                    cost[:, i - 1, j - 1],
                )
                cost[:, i, j] = d + best
        normalized = cost[:, t_valid, t_valid] / float(t_valid)
        total += float(normalized.sum())
        count += m_count

    return total, count


def _accumulate_traj_metrics(
    acc: dict,
    pred_abs: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
    include_dtw: bool = True,
) -> None:
    """Update `acc` in place with per-batch sums and counts for ADE/FDE/(DTW)."""
    for name, dim_idx in _TRAJ_METRIC_DIMS.items():
        a, f, na, nf = _ade_fde_sums(pred_abs, gt_abs, valid_future, kpm, dim_idx)
        acc[f"ade_{name}_sum"] = acc.get(f"ade_{name}_sum", 0.0) + a
        acc[f"ade_{name}_n"] = acc.get(f"ade_{name}_n", 0) + na
        acc[f"fde_{name}_sum"] = acc.get(f"fde_{name}_sum", 0.0) + f
        acc[f"fde_{name}_n"] = acc.get(f"fde_{name}_n", 0) + nf
    if include_dtw:
        for name, dim_idx in _TRAJ_METRIC_DIMS.items():
            d, nd = _dtw_sums(pred_abs, gt_abs, valid_future, kpm, dim_idx)
            acc[f"dtw_{name}_sum"] = acc.get(f"dtw_{name}_sum", 0.0) + d
            acc[f"dtw_{name}_n"] = acc.get(f"dtw_{name}_n", 0) + nd


def _finalize_traj_metrics(acc: dict, prefix: str, include_dtw: bool = True) -> dict[str, float]:
    """Convert accumulated sums/counts into mean values keyed by prefix."""
    out: dict[str, float] = {}
    metrics = ["ade", "fde"] + (["dtw"] if include_dtw else [])
    for metric in metrics:
        for name in _TRAJ_METRIC_DIMS:
            s = acc.get(f"{metric}_{name}_sum", 0.0)
            n = acc.get(f"{metric}_{name}_n", 0)
            out[f"{prefix}{metric}_{name}"] = s / max(n, 1)
    return out


@torch.no_grad()
def _min_ade_fde_sums(
    pred_abs_S: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
    dim_idx: tuple[int, ...],
) -> tuple[float, float, int, int]:
    """Best-of-S minADE/minFDE per (b, kp). docs/minADE_minFDE_plan.md.

    `pred_abs_S` is `(B, S, N_kp, F, 3)` (S = num eval samples). Computes per-sample
    ADE/FDE then takes min across the S axis, mirroring `_ade_fde_sums`'s
    sum/count contract so the caller can mean across batches.
    """
    diff = (pred_abs_S.float() - gt_abs.float().unsqueeze(1))[..., list(dim_idx)]
    dist = diff.pow(2).sum(-1).sqrt()  # (B, S, N_kp, F)

    valid_t = valid_future & kpm.unsqueeze(-1)  # (B, N_kp, F)
    valid_t_S = valid_t.unsqueeze(1).to(dist.dtype)  # (B, 1, N_kp, F)
    per_count = valid_t.to(dist.dtype).sum(-1)  # (B, N_kp)
    has_valid_ade = per_count > 0  # (B, N_kp)
    per_ade = (dist * valid_t_S).sum(-1) / per_count.unsqueeze(1).clamp_min(1.0)  # (B, S, N_kp)
    min_ade = per_ade.min(dim=1).values  # (B, N_kp)
    ade_sum = float((min_ade * has_valid_ade.to(min_ade.dtype)).sum().item())
    n_ade = int(has_valid_ade.sum().item())

    f_len = valid_future.shape[-1]
    t_idx = torch.arange(f_len, device=valid_t.device).expand_as(valid_t)
    masked_t = torch.where(valid_t, t_idx, torch.full_like(t_idx, -1))
    last_t = masked_t.max(-1).values  # (B, N_kp), -1 if no valid
    has_valid_fde = last_t >= 0
    last_t_safe = last_t.clamp_min(0)
    last_idx = last_t_safe.unsqueeze(1).unsqueeze(-1).expand(-1, dist.shape[1], -1, 1)
    last_dist = dist.gather(-1, last_idx).squeeze(-1)  # (B, S, N_kp)
    min_fde = last_dist.min(dim=1).values  # (B, N_kp)
    fde_sum = float((min_fde * has_valid_fde.to(min_fde.dtype)).sum().item())
    n_fde = int(has_valid_fde.sum().item())

    return ade_sum, fde_sum, n_ade, n_fde


@torch.no_grad()
def _smoothness_sums(
    abs_traj: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
    dim_idx: tuple[int, ...],
) -> tuple[float, int]:
    """Per-trajectory mean L2 norm of the second time-difference (jerk-like).

    Accepts `(B, N_kp, F, 3)` (single sample) or `(B, S, N_kp, F, 3)` (multi-sample)
    — a missing S axis is unsqueezed so single-sample callers (GT) and multi-sample
    callers (predictions) share one path. A second-diff at interior index `i` is
    valid only when timesteps `i-1, i, i+1` are all valid for that kp; trajectories
    with no valid interior point are skipped. Returns `(smooth_sum, n_traj)`.
    """
    if abs_traj.dim() == 4:
        abs_traj = abs_traj.unsqueeze(1)
    p = abs_traj.float()[..., list(dim_idx)]  # (B, S, N_kp, F, K)
    d2 = p[..., 2:, :] - 2.0 * p[..., 1:-1, :] + p[..., :-2, :]  # (B, S, N_kp, F-2, K)
    norm = d2.pow(2).sum(-1).sqrt()  # (B, S, N_kp, F-2)

    vf = valid_future & kpm.unsqueeze(-1)  # (B, N_kp, F)
    mid_valid = vf[..., :-2] & vf[..., 1:-1] & vf[..., 2:]  # (B, N_kp, F-2)
    mid_valid_S = mid_valid.unsqueeze(1).to(norm.dtype)  # (B, 1, N_kp, F-2)
    per_count = mid_valid_S.sum(-1)  # (B, 1, N_kp)
    has_valid = per_count > 0  # (B, 1, N_kp)
    per_smooth = (norm * mid_valid_S).sum(-1) / per_count.clamp_min(1.0)  # (B, S, N_kp)
    has_valid_full = has_valid.expand_as(per_smooth)
    smooth_sum = float((per_smooth * has_valid_full.to(per_smooth.dtype)).sum().item())
    n = int(has_valid_full.sum().item())
    return smooth_sum, n


def _accumulate_min_metrics(
    acc: dict,
    pred_abs_S: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
) -> None:
    """Update `acc` with Best-of-S minADE/minFDE + per-sample prediction smoothness.

    GT smoothness is accumulated once per batch (independent of S) so it can be
    reported as a single reference number alongside the prediction-side smoothness.
    """
    for name, dim_idx in _TRAJ_METRIC_DIMS.items():
        a, f, na, nf = _min_ade_fde_sums(pred_abs_S, gt_abs, valid_future, kpm, dim_idx)
        acc[f"min_ade_{name}_sum"] = acc.get(f"min_ade_{name}_sum", 0.0) + a
        acc[f"min_ade_{name}_n"] = acc.get(f"min_ade_{name}_n", 0) + na
        acc[f"min_fde_{name}_sum"] = acc.get(f"min_fde_{name}_sum", 0.0) + f
        acc[f"min_fde_{name}_n"] = acc.get(f"min_fde_{name}_n", 0) + nf
        sp, npred = _smoothness_sums(pred_abs_S, valid_future, kpm, dim_idx)
        acc[f"smooth_pred_{name}_sum"] = acc.get(f"smooth_pred_{name}_sum", 0.0) + sp
        acc[f"smooth_pred_{name}_n"] = acc.get(f"smooth_pred_{name}_n", 0) + npred
        sg, ng = _smoothness_sums(gt_abs, valid_future, kpm, dim_idx)
        acc[f"smooth_gt_{name}_sum"] = acc.get(f"smooth_gt_{name}_sum", 0.0) + sg
        acc[f"smooth_gt_{name}_n"] = acc.get(f"smooth_gt_{name}_n", 0) + ng


def _finalize_min_metrics(
    acc: dict, prefix: str, num_samples: int
) -> dict[str, float]:
    """Convert min/smooth accumulators into wandb-ready means.

    Metric names embed `num_samples` for minADE/minFDE so different N runs are
    plotted as distinct series; smoothness omits N because GT smoothness is N-
    independent and the prediction-side average is reported across all N samples.
    """
    out: dict[str, float] = {}
    for name in _TRAJ_METRIC_DIMS:
        for kind, label in (("min_ade", "minADE"), ("min_fde", "minFDE")):
            s = acc.get(f"{kind}_{name}_sum", 0.0)
            n = acc.get(f"{kind}_{name}_n", 0)
            out[f"{prefix}{label}_{num_samples}_{name}"] = s / max(n, 1)
        for kind in ("smooth_pred", "smooth_gt"):
            s = acc.get(f"{kind}_{name}_sum", 0.0)
            n = acc.get(f"{kind}_{name}_n", 0)
            out[f"{prefix}{kind}_{name}"] = s / max(n, 1)
    return out


@torch.no_grad()
def _min_dtw_sums(
    pred_abs_S: torch.Tensor,    # (B, S, N_kp, F, 3)
    gt_abs: torch.Tensor,         # (B, N_kp, F, 3)
    valid_future: torch.Tensor,   # (B, N_kp, F)
    kpm: torch.Tensor,            # (B, N_kp)
    dim_idx: tuple[int, ...],
) -> tuple[float, int]:
    """Best-of-S minDTW per (b, kp). Returns (sum, n_traj).

    Per (b, kp): compress to its T valid future steps, run the standard DTW
    DP on each of the S predicted trajectories against the single GT, then
    take the min across S. The DP is vectorized over S (shape `(S, T+1, T+1)`)
    so per-trajectory Python overhead stays bounded. Length-normalized by `T`
    (consistent with `_dtw_sums`) so different trajectories are comparable.
    Mirrors the sum/count contract of `_dtw_sums` so callers can mean across
    batches with the right denominator.

    NOTE: Not invoked by `_accumulate_min_metrics`. minDTW costs S× a DTW
    pass and would slow the training-time eval; predict scripts that want
    it should call `_accumulate_min_dtw_metrics` explicitly.
    """
    valid_t = (valid_future & kpm.unsqueeze(-1)).cpu().numpy()                 # (B, N, F)
    pred_np = pred_abs_S[..., list(dim_idx)].detach().to(torch.float32).cpu().numpy()  # (B, S, N, F, K)
    gt_np = gt_abs[..., list(dim_idx)].detach().to(torch.float32).cpu().numpy()        # (B, N, F, K)
    B, S, N_kp, _F_, _K = pred_np.shape

    total = 0.0
    n_traj = 0
    for b in range(B):
        for n in range(N_kp):
            m = valid_t[b, n]
            t_valid = int(m.sum())
            if t_valid == 0:
                continue
            gt_traj = gt_np[b, n][m]                 # (T, K)
            pred_S = pred_np[b, :, n][:, m]          # (S, T, K)
            if t_valid == 1:
                d_per_s = np.linalg.norm(pred_S[:, 0] - gt_traj[0], axis=-1)  # (S,)
                total += float(d_per_s.min())
                n_traj += 1
                continue
            diff = pred_S[:, :, None, :] - gt_traj[None, None, :, :]    # (S, T, T, K)
            pdist = np.linalg.norm(diff, axis=-1).astype(np.float64)     # (S, T, T)
            cost = np.full((S, t_valid + 1, t_valid + 1), np.inf, dtype=np.float64)
            cost[:, 0, 0] = 0.0
            for i in range(1, t_valid + 1):
                for j in range(1, t_valid + 1):
                    d_ij = pdist[:, i - 1, j - 1]
                    best = np.minimum(
                        np.minimum(cost[:, i - 1, j], cost[:, i, j - 1]),
                        cost[:, i - 1, j - 1],
                    )
                    cost[:, i, j] = d_ij + best
            per_s_dtw = cost[:, t_valid, t_valid] / float(t_valid)       # (S,)
            total += float(per_s_dtw.min())
            n_traj += 1
    return total, n_traj


def _accumulate_min_dtw_metrics(
    acc: dict,
    pred_abs_S: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
) -> None:
    """Accumulate Best-of-S minDTW for uvz / uv / depth (same dim set as ADE/FDE)."""
    for name, dim_idx in _TRAJ_METRIC_DIMS.items():
        s, n = _min_dtw_sums(pred_abs_S, gt_abs, valid_future, kpm, dim_idx)
        acc[f"min_dtw_{name}_sum"] = acc.get(f"min_dtw_{name}_sum", 0.0) + s
        acc[f"min_dtw_{name}_n"] = acc.get(f"min_dtw_{name}_n", 0) + n


def _finalize_min_dtw_metrics(
    acc: dict, prefix: str, num_samples: int,
) -> dict[str, float]:
    """Convert minDTW accumulators into `{prefix}minDTW_{N}_{name}` means."""
    out: dict[str, float] = {}
    for name in _TRAJ_METRIC_DIMS:
        s = acc.get(f"min_dtw_{name}_sum", 0.0)
        n = acc.get(f"min_dtw_{name}_n", 0)
        out[f"{prefix}minDTW_{num_samples}_{name}"] = s / max(n, 1)
    return out


@torch.no_grad()
def _accumulate_done_metrics(
    acc: dict,
    done_prob: torch.Tensor,         # (B, N, H)
    valid_future: torch.Tensor,      # (B, N, H)
    key_pad_mask: torch.Tensor,      # (B, N)
    threshold: float,
) -> None:
    """Accumulate done-head metrics: BCE-style accuracy + end-step MAE.

    See docs/trace_done_head_plan.md §1.7. The done head is H-wide
    (docs/trace_bspline_ctrl_pts_plan.md §3), so target/step bookkeeping uses the
    raw `valid_future` (T = H, step_per_unit = 1). Non-monotone samples (a True
    after a False) skip the MAE accumulator but still contribute to accuracy.
    """
    B, N, H = valid_future.shape
    target_group = valid_future
    T = H
    step_per_unit = 1

    pred_pos = (done_prob >= threshold)  # True = "valid/ongoing", same convention as target.
    # Accuracy: matches over (kp, slot) pairs within key_pad_mask.
    kpm_b = key_pad_mask[:, :, None].expand(-1, -1, T)
    matches = (pred_pos == target_group) & kpm_b
    acc["done_match_sum"] = acc.get("done_match_sum", 0.0) + float(matches.sum().item())
    acc["done_match_n"] = acc.get("done_match_n", 0) + int(kpm_b.sum().item())

    # End-step MAE. "Stop unit" = first slot where pred/target says False; T if none.
    # Predicted slot: first slot where `pred_pos == False`.
    # Target slot:    first slot where `target_group == False`.
    pad = torch.full_like(target_group, T, dtype=torch.long)
    arange_T = torch.arange(T, device=target_group.device)
    pred_first_off = torch.where(~pred_pos, arange_T, pad).min(dim=-1).values    # (B, N)
    targ_first_off = torch.where(~target_group, arange_T, pad).min(dim=-1).values  # (B, N)
    # Monotone if no True after the first False — i.e., target_group[t:] is all False
    # past the first-False index. Equivalent: number of False slots == T - targ_first_off.
    n_false = (~target_group).sum(dim=-1)
    monotone = (n_false == (T - targ_first_off)) & key_pad_mask
    abs_err_steps = (pred_first_off - targ_first_off).abs() * step_per_unit
    acc["done_mae_sum"] = acc.get("done_mae_sum", 0.0) + float((abs_err_steps * monotone).sum().item())
    acc["done_mae_n"] = acc.get("done_mae_n", 0) + int(monotone.sum().item())
    # Track non-monotone fraction so the metric is interpretable.
    non_mono = (~monotone) & key_pad_mask
    acc["done_nonmono_n"] = acc.get("done_nonmono_n", 0) + int(non_mono.sum().item())
    acc["done_total_n"] = acc.get("done_total_n", 0) + int(key_pad_mask.sum().item())


def _finalize_done_metrics(acc: dict, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if acc.get("done_match_n", 0) > 0:
        out[f"{prefix}done_acc"] = acc["done_match_sum"] / acc["done_match_n"]
    if acc.get("done_mae_n", 0) > 0:
        out[f"{prefix}done_end_step_mae"] = acc["done_mae_sum"] / acc["done_mae_n"]
    if acc.get("done_total_n", 0) > 0:
        out[f"{prefix}done_nonmono_frac"] = (
            acc.get("done_nonmono_n", 0) / acc["done_total_n"]
        )
    return out


@torch.no_grad()
def _run_eval(
    policy: SmolVLAPolicy,
    eval_dl: DataLoader,
    tokenizer,
    tokenizer_max_length: int,
    device: torch.device,
    max_batches: int,
    delta_scale: tuple[float, float, float] | None,
    num_samples: int = 1,
    prefix: str = "eval/",
    drop_depth: bool = False,
) -> dict[str, float] | None:
    """Run held-out eval on up to `max_batches` batches (0 = no cap). Returns eval/* metrics or None if empty.

    `num_samples >= 2` enables Best-of-N sampling (docs/minADE_minFDE_plan.md):
    `predict_trace` is called `num_samples` times per batch with independent noise
    draws, and minADE/minFDE/smoothness metrics are computed across the stacked
    samples. Legacy single-sample ADE/FDE/DTW + done-head metrics are computed on
    sample-0 to stay comparable to prior runs (and `num_samples=1` collapses
    back to the original single-pass behavior).

    `drop_depth=True` strips the depth key (and its padding mask) from each
    batch before the policy forward pass, so `prepare_images` sees depth as a
    missing camera and the prefix is built RGB-only. Used to produce a
    `test_wo_depth/` view of the same data, for comparing with-vs-without-depth
    performance on wandb.
    """
    was_training = policy.training
    policy.eval()
    loss_sum = 0.0
    loss_done_sum = 0.0
    loss_done_count = 0
    loss_rigidity_sum = 0.0
    loss_rigidity_count = 0
    l1_abs_sum = 0.0
    l1_uv_sum = 0.0
    l1_depth_sum = 0.0
    l1_delta_sum = 0.0
    traj_acc: dict = {}
    min_acc: dict = {}
    done_acc_state: dict = {}
    n = 0
    n_samples = max(1, num_samples)
    try:
        for i, raw in enumerate(eval_dl):
            if max_batches > 0 and i >= max_batches:
                break
            batch = _tokenize_and_device(raw, tokenizer, tokenizer_max_length, device)
            if drop_depth:
                # Strip depth so `prepare_images` treats it as a missing camera
                # (with `empty_cameras=0` the depth block is simply not added to
                # the prefix). RGB / lang / suffix all see RGB-only positions.
                batch = {
                    k: v for k, v in batch.items()
                    if k != DEPTH_KEY and k != f"{DEPTH_KEY}_padding_mask"
                }
            loss, loss_dict = policy(batch)
            if "loss_done" in loss_dict:
                loss_done_sum += float(loss_dict["loss_done"])
                loss_done_count += 1
            if "loss_rigidity" in loss_dict:
                loss_rigidity_sum += float(loss_dict["loss_rigidity"])
                loss_rigidity_count += 1

            # Best-of-N: independent noise draws per call. Looping (vs. expanding the
            # condition tensors by N) keeps peak memory bounded at the cost of
            # recomputing the prefix N times — acceptable since eval is main-rank
            # only and small (eval_fraction=0.01 default).
            traces_list: list[torch.Tensor] = []
            done_prob_first: torch.Tensor | None = None
            for s in range(n_samples):
                pred = policy.predict_trace(batch, num_steps=policy.config.num_steps)
                if isinstance(pred, dict):
                    traces_s = pred["traces"]
                    if s == 0:
                        done_prob_first = pred["done_prob"]
                else:
                    traces_s = pred
                traces_list.append(traces_s)
            traces_S = torch.stack(traces_list, dim=1)  # (B, S, N_kp, F, 3)
            traces = traces_S[:, 0]  # (B, N_kp, F, 3) — sample-0 for legacy metrics.
            done_prob = done_prob_first

            valid_future = batch[VALID_FUTURE_KEY]
            kpm = batch[KEY_PAD_MASK_KEY]
            gt = batch[TRAJ_FUTURE_KEY]
            current_kp = batch[CURRENT_KP_KEY]
            mask = (valid_future.unsqueeze(-1) & kpm[:, :, None, None]).to(traces.dtype)
            denom = mask.sum().clamp_min(1.0)
            if done_prob is not None:
                _accumulate_done_metrics(
                    done_acc_state,
                    done_prob.float(),
                    valid_future,
                    kpm,
                    threshold=policy.config.done_threshold,
                )
            # `l1_delta` is the raw training-loss-space metric (anchor-deltas),
            # kept apples-to-apples with the loss curve.
            l1_delta = ((traces - gt).abs() * mask).sum() / denom

            gt_for_metric = gt

            if delta_scale is not None:
                scale_t = torch.as_tensor(delta_scale, dtype=traces.dtype, device=traces.device)
                anchor = current_kp[:, :, None, :3]
                traces_abs = traces * scale_t + anchor
                traces_S_abs = traces_S * scale_t + anchor.unsqueeze(1)  # (B, S, N_kp, F, 3)
                gt_abs = gt_for_metric * scale_t + anchor
            else:
                traces_abs = traces
                traces_S_abs = traces_S
                gt_abs = gt_for_metric
            abs_err = (traces_abs - gt_abs).abs() * mask
            l1_abs = abs_err.sum() / denom
            l1_uv = abs_err[..., :2].sum() / denom
            l1_depth = abs_err[..., 2:3].sum() / denom

            _accumulate_traj_metrics(traj_acc, traces_abs, gt_abs, valid_future, kpm, include_dtw=True)
            _accumulate_min_metrics(min_acc, traces_S_abs, gt_abs, valid_future, kpm)

            loss_sum += float(loss.item())
            l1_abs_sum += float(l1_abs.item())
            l1_uv_sum += float(l1_uv.item())
            l1_depth_sum += float(l1_depth.item())
            l1_delta_sum += float(l1_delta.item())
            n += 1
    finally:
        if was_training:
            policy.train()
    if n == 0:
        return None
    out = {
        f"{prefix}loss": loss_sum / n,
        f"{prefix}l1_abs": l1_abs_sum / n,
        f"{prefix}l1_uv": l1_uv_sum / n,
        f"{prefix}l1_depth": l1_depth_sum / n,
        f"{prefix}l1_delta": l1_delta_sum / n,
        f"{prefix}num_batches": float(n),
        f"{prefix}num_samples": float(n_samples),
    }
    if loss_done_count > 0:
        out[f"{prefix}loss_done"] = loss_done_sum / loss_done_count
    if loss_rigidity_count > 0:
        out[f"{prefix}loss_rigidity"] = loss_rigidity_sum / loss_rigidity_count
    out.update(_finalize_traj_metrics(traj_acc, prefix=prefix, include_dtw=True))
    out.update(_finalize_min_metrics(min_acc, prefix=prefix, num_samples=n_samples))
    out.update(_finalize_done_metrics(done_acc_state, prefix=prefix))
    return out


@torch.no_grad()
def _log_trace_vis(
    policy: SmolVLAPolicy,
    fixed_batch: dict[str, Any],
    step: int,
    wandb_run,
    vis_num_samples: int,
    delta_scale: tuple[float, float, float] | None = None,
    prefix: str = "vis/",
) -> None:
    was_training = policy.training
    policy.eval()
    try:
        pred = policy.predict_trace(fixed_batch, num_steps=policy.config.num_steps)
    finally:
        if was_training:
            policy.train()

    if isinstance(pred, dict):
        traces = pred["traces"]
        done_prob = pred["done_prob"]  # (B, N, G_fut) flat | (B, N, H) bspline
    else:
        traces = pred
        done_prob = None

    valid_future = fixed_batch[VALID_FUTURE_KEY]
    kpm = fixed_batch[KEY_PAD_MASK_KEY]
    gt = fixed_batch[TRAJ_FUTURE_KEY]
    traj_hist = fixed_batch[TRAJ_HISTORY_KEY]
    current_kp = fixed_batch[CURRENT_KP_KEY]
    mask = (valid_future.unsqueeze(-1) & kpm[:, :, None, None]).to(traces.dtype)
    denom = mask.sum().clamp_min(1.0)
    # `l1_delta` is in raw training-loss space (anchor-deltas), kept
    # apples-to-apples with the loss curve.
    l1_delta = ((traces - gt).abs() * mask).sum() / denom

    # Done-head truncate-and-freeze (docs/trace_done_head_plan.md §1.6).
    # Compute the step-resolution predicted-valid mask; the renderer uses it to
    # suppress points past the predicted stop boundary.
    pred_valid_step: torch.Tensor | None = None
    if done_prob is not None:
        # Done head is H-wide (bspline); see docs/trace_bspline_ctrl_pts_plan.md §3.
        pred_valid_step = (done_prob >= policy.config.done_threshold)  # (B, N, H)

    gt_for_metric = gt

    # If targets are delta-space, invert to abs (normalized xy + raw-meter z) for
    # interpretable metrics and for the overlay (render_trace_overlay expects absolute
    # normalized xy). `current_kp` is always in abs space regardless of delta_scale.
    if delta_scale is not None:
        scale_t = torch.as_tensor(delta_scale, dtype=traces.dtype, device=traces.device)
        anchor = current_kp[:, :, None, :3]  # (B, N, 1, 3)
        traces_abs = traces * scale_t + anchor
        gt_abs = gt_for_metric * scale_t + anchor
    else:
        traces_abs = traces
        gt_abs = gt_for_metric
    abs_err = (traces_abs - gt_abs).abs() * mask
    l1_abs = abs_err.sum() / denom
    l1_uv = abs_err[..., :2].sum() / denom
    l1_depth = abs_err[..., 2:3].sum() / denom

    traj_acc: dict = {}
    _accumulate_traj_metrics(traj_acc, traces_abs, gt_abs, valid_future, kpm, include_dtw=True)
    traj_metrics = _finalize_traj_metrics(traj_acc, prefix=prefix, include_dtw=True)

    if wandb_run is None:
        logging.info(
            f"[{prefix.rstrip('/')} @ step {step}] trace_l1_abs={l1_abs.item():.4f} "
            f"trace_l1_uv={l1_uv.item():.4f} trace_l1_depth={l1_depth.item():.4f} "
            f"trace_l1_delta={l1_delta.item():.4f} "
            f"ade_uvz={traj_metrics[f'{prefix}ade_uvz']:.4f} "
            f"fde_uvz={traj_metrics[f'{prefix}fde_uvz']:.4f} "
            f"dtw_uvz={traj_metrics[f'{prefix}dtw_uvz']:.4f} (wandb disabled)"
        )
        return

    import wandb

    images = fixed_batch[IMAGE_KEY]
    valid_hist = fixed_batch[VALID_HISTORY_KEY]
    # Reconstruct absolute history for the overlay: the dataset stored centered
    # values (near 0), so add `current_kp` back before rendering — the renderer
    # expects normalized xy in [-1, 1].
    traj_hist_abs = traj_hist.clone()
    traj_hist_abs[..., :3] = traj_hist_abs[..., :3] + current_kp[:, :, None, :3]
    bsz = min(vis_num_samples, images.shape[0])
    # Rainbow-by-current-uv: each kp's predicted polyline is colored from the
    # 2D color wheel applied to its current (u, v), matching the palette used
    # by lerobot_trace_gui / lerobot_visualize_trace_3d so 2D overlays line up
    # with the 3D viser scene. Drawn with a thinner stroke than history/GT.
    g_color = policy.config.trace_group_horizon  # only consumed by per_group mode; kept for the call below.
    color_mode = "per_kp_uv"
    overlays = []
    motion_videos_color = []
    motion_videos_pixel = []
    for b in range(bsz):
        img = (images[b].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        # Done-head truncation: pass the predicted per-step valid mask so the
        # renderer suppresses points past the predicted stop boundary.
        pvf_b = (
            pred_valid_step[b].cpu().numpy()
            if pred_valid_step is not None
            else None
        )
        traj_hist_np = traj_hist_abs[b].cpu().numpy()
        traces_np = traces_abs[b].cpu().numpy()
        valid_hist_np = valid_hist[b].cpu().numpy()
        valid_future_np = valid_future[b].cpu().numpy()
        kpm_np = kpm[b].cpu().numpy()
        current_kp_np = current_kp[b].cpu().numpy()
        overlay = render_trace_overlay(
            image=img,
            traj_history=traj_hist_np,
            traj_future_gt=gt_abs[b].cpu().numpy(),
            traj_future_pred=traces_np,
            valid_history=valid_hist_np,
            valid_future=valid_future_np,
            predicted_valid_future=pvf_b,
            key_pad_mask=kpm_np,
            current_kp=current_kp_np,
            # Post current_kp split (docs/trace_current_kp_split_plan.md):
            # prepend_current splices the current uv onto the front of the future
            # polyline; draw_last_uv_marker draws a red circle at current_kp;
            # connect_history_to_current extends the cyan past-only history into
            # the current uv so the polyline visibly meets the red circle.
            prepend_current=True,
            draw_last_uv_marker=True,
            connect_history_to_current=True,
            pred_color_mode=color_mode,
            pred_group_size=g_color,
        )
        overlays.append(wandb.Image(overlay, caption=f"sample {b}"))

        # Motion video: same uv-rainbow palette as the static overlay, dots
        # move along history -> current -> predicted future on a faded copy of
        # the (single) current frame, so the kp motion is visible against the
        # static background. wandb.Video expects (T, C, H, W) uint8.
        # Two motion videos per sample: (1) uv-rainbow disks (matches the static
        # overlay's colors); (2) circular patches sampled from the current frame
        # at each kp's uv, dragged along the trace so the underlying texture
        # appears to slide with the trajectory.
        mv_kwargs = dict(
            image=img,
            traj_history=traj_hist_np,
            traj_future_pred=traces_np,
            valid_history=valid_hist_np,
            valid_future=valid_future_np,
            current_kp=current_kp_np,
            key_pad_mask=kpm_np,
            predicted_valid_future=pvf_b,
            bg_alpha=0.35,
            future_only=True,
        )
        frames_color = render_trace_motion_video(**mv_kwargs, color_mode="uv_rainbow")
        frames_pixel = render_trace_motion_video(**mv_kwargs, color_mode="current_pixels")
        motion_videos_color.append(
            wandb.Video(
                np.ascontiguousarray(frames_color.transpose(0, 3, 1, 2)),
                fps=8, format="mp4", caption=f"sample {b}",
            )
        )
        motion_videos_pixel.append(
            wandb.Video(
                np.ascontiguousarray(frames_pixel.transpose(0, 3, 1, 2)),
                fps=8, format="mp4", caption=f"sample {b}",
            )
        )
    log_payload: dict[str, Any] = {
        f"{prefix}trace_l1_abs": l1_abs.item(),
        f"{prefix}trace_l1_uv": l1_uv.item(), # uv is in normalized [-1,1] coords
        f"{prefix}trace_l1_depth": l1_depth.item(), # depth in raw meters
        f"{prefix}trace_l1_delta": l1_delta.item(),
        f"{prefix}overlays": overlays,
        f"{prefix}motion_vis_color": motion_videos_color,
        f"{prefix}motion_vis": motion_videos_pixel,
    }
    log_payload.update(traj_metrics)
    if DEPTH_KEY in fixed_batch:
        depths = fixed_batch[DEPTH_KEY]
        depth_panels = []
        for b in range(bsz):
            d_img = (depths[b].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            depth_panels.append(wandb.Image(d_img, caption=f"sample {b}"))
        log_payload[f"{prefix}depth"] = depth_panels
    wandb_run.log(log_payload, step=step)


@draccus.wrap()
def train(cfg: TraceTrainSmolVLAConfig) -> None:
    # find_unused_parameters=True: trace mode prunes state_proj/action_* so DDP would otherwise
    # complain about unused params. See plan §9.2.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Stats auto-compute on rank 0 can take >10 min (the default NCCL PG timeout),
    # during which other ranks block in wait_for_everyone() and get killed.
    pg_kwargs = InitProcessGroupKwargs(timeout=dt.timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, pg_kwargs])
    init_logging()
    is_main = accelerator.is_main_process
    # `cfg.__post_init__` builds `output_dir` from `dt.datetime.now()` on each rank
    # independently, so ranks that cross a second-boundary disagree by 1s and end
    # up writing/reading from different `outputs/.../HH-MM-SS_*` directories. That
    # made `_resolve_trace_stats` fail on non-main ranks (rank-0 copied the JSON
    # into its dir, others tried to read from theirs). Broadcast rank-0's path so
    # all ranks share the same output_dir for stats / checkpoints / wandb_run.json.
    out_dir_holder = [str(cfg.output_dir)]
    broadcast_object_list(out_dir_holder, from_process=0)
    cfg.output_dir = Path(out_dir_holder[0])
    if is_main:
        logging.info(f"Output dir: {cfg.output_dir}  (world_size={accelerator.num_processes})")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _accelerate_set_seed(cfg.seed)

    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = True

    trace_stats = _resolve_trace_stats(cfg, accelerator)
    delta_scale = trace_stats.delta_scale if cfg.use_delta_target else None
    depth_log_range = trace_stats.depth_log_range if cfg.use_depth else None

    # Per-source train/eval split at the video-dir level (0.95/0.05 default).
    # Deterministic across ranks via cfg.seed so all ranks agree on the partition.
    if is_main:
        if cfg.eval_episode_num_per_dir > 0:
            split_desc = (
                f"eval_episode_num_per_dir={cfg.eval_episode_num_per_dir}"
            )
        else:
            split_desc = f"eval_fraction={cfg.eval_fraction}"
        logging.info(
            f"Splitting video_dirs per-source ({split_desc}, seed={cfg.seed})..."
        )
    train_dirs, eval_dirs = _split_video_dirs(
        cfg.video_dirs,
        cfg.eval_fraction,
        cfg.seed,
        eval_episode_num_per_dir=cfg.eval_episode_num_per_dir,
    )
    if is_main:
        logging.info(f"split totals: train={len(train_dirs)} eval={len(eval_dirs)}")

    # Cache the dataset scan next to delta_stats. Filename is derived from the
    # filter config; per-dir contents are mtime-validated. See TraceDataset's
    # _try_load_cache. None disables caching (legacy default).
    trace_cache_dir = (
        Path(cfg.delta_stats_path).parent if cfg.delta_stats_path is not None else None
    )

    dataset = TraceDataset(
        video_dirs=train_dirs,
        future_len=cfg.future_len,
        history_len=cfg.history_len,
        n_min=cfg.n_min,
        n_max=cfg.n_max,
        image_size=cfg.image_size,
        seed=cfg.seed,
        min_future_motion=cfg.min_future_motion,
        delta_scale=delta_scale,
        load_depth=cfg.use_depth,
        depth_log_range=depth_log_range,
        rgb_jitter_strength=cfg.rgb_jitter_strength,
        depth_noise_std=cfg.depth_noise_std,
        bspline_n_ctrl=cfg.trace_bspline_n_ctrl,
        bspline_reg_lambda=cfg.trace_bspline_reg_lambda,
        bspline_reg_order=cfg.trace_bspline_reg_order,
        bspline_ctrl_clip=cfg.trace_bspline_ctrl_clip,
        cache_dir=trace_cache_dir,
        show_progress=is_main,
        load_cluster_ids=cfg.rigidity_regularization_weight > 0.0,
    )
    if is_main:
        logging.info(f"TraceDataset[train]: {len(dataset)} samples from {len(train_dirs)} dirs")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=trace_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Eval dataset + dataloader. Built on main rank only; we do not pass it through
    # `accelerator.prepare` so it is not sharded — main runs the full eval while
    # other ranks idle at a barrier (see the training loop below). This keeps eval
    # deterministic and simple, at the cost of ~O(eval_max_batches * batch) walltime
    # per eval call (acceptable given eval_max_batches is capped).
    #
    # We pre-sample a fixed random subset of eval indices (seeded) so the *same*
    # frames are evaluated every call — making eval/loss curves comparable across
    # steps — while still spanning episodes uniformly. Without this, shuffle=False
    # over a per-video index layout would evaluate only the first few episodes.
    eval_dataset = None
    eval_dl: DataLoader | None = None
    if is_main and eval_dirs and cfg.eval_every_n_steps > 0:
        eval_dataset = TraceDataset(
            video_dirs=eval_dirs,
            future_len=cfg.future_len,
            history_len=cfg.history_len,
            n_min=cfg.n_min,
            n_max=cfg.n_max,
            image_size=cfg.image_size,
            seed=cfg.seed,
            min_future_motion=cfg.min_future_motion,
            delta_scale=delta_scale,
            load_depth=cfg.use_depth,
            depth_log_range=depth_log_range,
            bspline_n_ctrl=cfg.trace_bspline_n_ctrl,
            bspline_reg_lambda=cfg.trace_bspline_reg_lambda,
            bspline_reg_order=cfg.trace_bspline_reg_order,
            bspline_ctrl_clip=cfg.trace_bspline_ctrl_clip,
            cache_dir=trace_cache_dir,
            show_progress=True,
            load_cluster_ids=cfg.rigidity_regularization_weight > 0.0,
        )
        logging.info(
            f"TraceDataset[eval]: {len(eval_dataset)} samples from {len(eval_dirs)} dirs"
        )
        if cfg.eval_max_batches > 0:
            n_eval_target = cfg.eval_max_batches * cfg.batch_size
            n_eval_take = min(n_eval_target, len(eval_dataset))
            eval_rng = random.Random(cfg.seed + 1)
            eval_indices = eval_rng.sample(range(len(eval_dataset)), n_eval_take)
            eval_source: TraceDataset | Subset = Subset(eval_dataset, eval_indices)
            n_videos_in_eval = len({eval_dataset._index[i][0] for i in eval_indices})
            logging.info(
                f"Eval Subset: {len(eval_source)} samples drawn from "
                f"{n_videos_in_eval}/{len(eval_dataset._videos)} eval videos (seeded)"
            )
        else:
            eval_source = eval_dataset
            logging.info(
                f"Eval (full): {len(eval_source)} samples across "
                f"{len(eval_dataset._videos)} eval videos"
            )
        eval_dl = DataLoader(
            eval_source,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=trace_collate_fn,
            pin_memory=device.type == "cuda",
            drop_last=True,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
    elif is_main:
        logging.info("Eval disabled (no eval dirs or eval_every_n_steps=0).")

    # External test set (held-out, not split). One sample per episode (first
    # usable slot per video) so the test loop is "first image only" per episode.
    # Main-rank only, runs at the same cadence as eval. Shares `trace_cache_dir`
    # with train/eval — the cache filename is hashed from the dir list so test
    # gets its own .pkl alongside train's without invalidating it.
    test_dataset = None
    test_dl: DataLoader | None = None
    test_first_indices: list[int] = []
    if is_main and cfg.test_video_dirs and cfg.eval_every_n_steps > 0:
        test_dataset = TraceDataset(
            video_dirs=cfg.test_video_dirs,
            future_len=cfg.future_len,
            history_len=cfg.history_len,
            n_min=cfg.n_min,
            n_max=cfg.n_max,
            image_size=cfg.image_size,
            seed=cfg.seed,
            min_future_motion=cfg.min_future_motion,
            delta_scale=delta_scale,
            load_depth=cfg.use_depth,
            depth_log_range=depth_log_range,
            bspline_n_ctrl=cfg.trace_bspline_n_ctrl,
            bspline_reg_lambda=cfg.trace_bspline_reg_lambda,
            bspline_reg_order=cfg.trace_bspline_reg_order,
            bspline_ctrl_clip=cfg.trace_bspline_ctrl_clip,
            cache_dir=trace_cache_dir,
            show_progress=True,
            load_cluster_ids=cfg.rigidity_regularization_weight > 0.0,
        )
        # First-image-only: for each video, keep the slot with the lowest
        # frame index (== earliest usable frame in the episode). usable_slots
        # is built in slot order so the first global index per video already
        # points at the lowest slot; we pick the slot with min frame_indices
        # explicitly to be robust against future reordering.
        seen_videos: set[int] = set()
        by_video: dict[int, list[int]] = {}
        for global_idx, (v_idx, _slot) in enumerate(test_dataset._index):
            by_video.setdefault(v_idx, []).append(global_idx)
        for v_idx in sorted(by_video.keys()):
            best_idx = min(
                by_video[v_idx],
                key=lambda i: int(
                    test_dataset._videos[v_idx].frame_indices[test_dataset._index[i][1]]
                ),
            )
            test_first_indices.append(best_idx)
            seen_videos.add(v_idx)
        logging.info(
            f"TraceDataset[test]: {len(test_dataset)} samples → "
            f"{len(test_first_indices)} first-image-only samples across "
            f"{len(seen_videos)} episodes"
        )
        test_source = Subset(test_dataset, test_first_indices)
        test_dl = DataLoader(
            test_source,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=trace_collate_fn,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )
    elif is_main and cfg.test_video_dirs:
        logging.info("Test set provided but eval_every_n_steps=0 — test disabled.")

    tokenizer = _make_tokenizer(cfg)
    policy = _build_policy(cfg, delta_scale=delta_scale)
    policy.train()

    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if is_main:
        logging.info(f"Policy built: num_trainable={num_trainable} ({format_big_number(num_trainable)})")
        if cfg.use_depth:
            # Audit that under freeze_vision_encoder=True, the only trainable params inside
            # vision_model are lora_* adapters, and that the cloned depth stem is trainable.
            vwe = policy.model.vlm_with_expert
            vm = vwe.get_vlm_model().vision_model
            lora_n = stem_n = other_vm_n = 0
            for name, p in vm.named_parameters():
                if not p.requires_grad:
                    continue
                if "lora_" in name:
                    lora_n += p.numel()
                else:
                    other_vm_n += p.numel()
            if vwe._depth_patch_embedding is not None:
                stem_n = sum(p.numel() for p in vwe._depth_patch_embedding.parameters() if p.requires_grad)
            logging.info(
                f"Depth trainables — vision_lora={format_big_number(lora_n)} "
                f"depth_stem={format_big_number(stem_n)} other_vision_model={format_big_number(other_vm_n)}"
            )
            if cfg.freeze_vision_encoder and other_vm_n != 0:
                raise AssertionError(
                    f"freeze_vision_encoder=True but vision_model has {other_vm_n} non-LoRA trainable params"
                )
        if cfg.trace_history_full_mask_prob > 0 or cfg.trace_history_kp_dropout_prob > 0:
            logging.info(
                f"History dropout — full_mask_prob={cfg.trace_history_full_mask_prob:.2f} "
                f"kp_dropout_prob={cfg.trace_history_kp_dropout_prob:.2f} "
                f"(tokenization=flat required)"
            )
        if cfg.depth_dropout_prob > 0:
            logging.info(
                f"Depth dropout (train only) — depth_dropout_prob={cfg.depth_dropout_prob:.2f}"
            )
        if cfg.rgb_jitter_strength > 0 or cfg.depth_noise_std > 0:
            logging.info(
                f"Data aug (train only) — rgb_jitter_strength={cfg.rgb_jitter_strength:.2f} "
                f"depth_noise_std={cfg.depth_noise_std:.4f}m"
            )
        if cfg.use_dino:
            vla = policy.model
            dino_trainable = sum(
                p.numel() for p in vla.dino.parameters() if p.requires_grad
            )
            fuse_trainable = sum(
                p.numel()
                for name, p in vla.named_parameters()
                if p.requires_grad and name.startswith(("trace_dino_fuse_in", "trace_dino_fuse_out"))
            )
            logging.info(
                f"DINO trainables — dino_backbone={format_big_number(dino_trainable)} "
                f"dino_fuse_mlp={format_big_number(fuse_trainable)}"
            )
            if dino_trainable != 0:
                raise AssertionError(
                    f"use_dino=True but DINO backbone has {dino_trainable} trainable params"
                )
            if fuse_trainable == 0:
                raise AssertionError(
                    "use_dino=True but trace_dino_fuse_{in,out} have zero trainable params"
                )
        if cfg.trace_done_head:
            vla = policy.model
            done_trainable = sum(
                p.numel()
                for name, p in vla.named_parameters()
                if p.requires_grad and name.startswith(("trace_done_proj_flat", "trace_done_proj"))
            )
            logging.info(
                f"Done-head trainables — trace_done_proj={format_big_number(done_trainable)} "
                f"loss_weight={cfg.done_loss_weight} bias_init={cfg.done_bias_init} "
                f"threshold={cfg.done_threshold}"
            )
            if done_trainable == 0:
                raise AssertionError(
                    "trace_done_head=True but trace_done_proj{,_flat} has zero trainable params"
                )

    optim_params = (
        policy.get_optim_params()
        if hasattr(policy, "get_optim_params")
        else [p for p in policy.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    dl_iter = cycle(dataloader)

    # Resume from a prior `step_XXXXXXXX/` checkpoint dir. `load_state` restores
    # model weights, optimizer state, and per-rank RNG (Python/NumPy/torch CPU+CUDA)
    # from the snapshot Accelerate wrote at save time. The dataloader iterator is
    # NOT restored — workers are respawned on the next `next(dl_iter)` and reseeded
    # from the freshly-restored torch RNG, so resume sees a different shuffle than
    # a fresh run (close enough; we don't track dataloader position). meta.json
    # carries the step counter; the loop continues from start_step+1.
    start_step = 0
    if cfg.resume_from is not None:
        resume_dir = Path(cfg.resume_from)
        accel_state_dir = resume_dir / "accel_state"
        meta_path = resume_dir / "meta.json"
        if not accel_state_dir.is_dir():
            raise FileNotFoundError(
                f"resume_from={resume_dir} has no accel_state/ subdir. Only checkpoints "
                f"saved with the new resume-aware code can be resumed."
            )
        if not meta_path.is_file():
            raise FileNotFoundError(f"resume_from={resume_dir} is missing meta.json")
        accelerator.load_state(str(accel_state_dir))
        meta = json.loads(meta_path.read_text())
        start_step = int(meta["step"])
        if is_main:
            logging.info(f"Resumed from {resume_dir} at step {start_step}")
        if start_step >= cfg.steps:
            if is_main:
                logging.warning(
                    f"start_step ({start_step}) >= cfg.steps ({cfg.steps}); nothing to do. "
                    f"Bump --steps to extend training."
                )

    # Best-checkpoint tracking. On disk we keep two checkpoints: the most recent and
    # the one whose `eval/loss` was lowest. `best_meta.json` lives at the top of the
    # checkpoints/ tree so resume can restore the running best instead of resetting
    # it (which would let the next checkpoint clobber a real best from a prior run).
    best_eval_loss: float = float("inf")
    best_ckpt_step: int | None = None
    best_meta_path = cfg.output_dir / "checkpoints" / "best_meta.json"
    if best_meta_path.is_file():
        best_meta = json.loads(best_meta_path.read_text())
        best_eval_loss = float(best_meta["eval_loss"])
        best_ckpt_step = int(best_meta["step"])
        if is_main:
            logging.info(
                f"Loaded best-checkpoint state: step={best_ckpt_step} eval/loss={best_eval_loss:.4f}"
            )

    # Per-leaf forward-hook detector. Localizes the first non-finite module so we
    # can pinpoint *where* numerics break, not just observe a NaN at the loss.
    nan_detector = _FirstNaNDetector(policy) if cfg.detect_nan else None
    if is_main and nan_detector is not None:
        logging.info(
            f"NaN detector enabled — hooked {nan_detector.num_hooked_modules} leaf modules"
        )

    # vis_batch is pinned once so overlays track the same samples across training. Prefer
    # a held-out eval batch so we watch generalization; fall back to the train iter if
    # eval is disabled / empty.
    #
    # When eval is available, we hand-pick `vis_num_samples` indices spread across
    # distinct eval videos (round-robin, seeded), instead of taking the first eval
    # batch — which would otherwise come from a single video given the per-video
    # index layout. Decoupled from the eval Subset on purpose: vis tracks per-episode
    # qualitative behavior; eval tracks aggregate metrics.
    vis_batch = None
    if is_main:
        if eval_dataset is not None:
            by_video: dict[int, list[int]] = {}
            for global_idx, (vidx, _slot) in enumerate(eval_dataset._index):
                by_video.setdefault(vidx, []).append(global_idx)
            vis_rng = random.Random(cfg.seed + 2)
            for v_indices in by_video.values():
                vis_rng.shuffle(v_indices)
            video_keys = sorted(by_video.keys())
            vis_rng.shuffle(video_keys)
            cursors = {v: 0 for v in video_keys}
            vis_indices: list[int] = []
            while len(vis_indices) < cfg.vis_num_samples:
                progressed = False
                for v in video_keys:
                    if cursors[v] < len(by_video[v]):
                        vis_indices.append(by_video[v][cursors[v]])
                        cursors[v] += 1
                        progressed = True
                        if len(vis_indices) >= cfg.vis_num_samples:
                            break
                if not progressed:
                    break
            n_vis_videos = len({eval_dataset._index[i][0] for i in vis_indices})
            logging.info(
                f"Vis batch: pinned {len(vis_indices)} samples across "
                f"{n_vis_videos} distinct eval videos"
            )
            vis_raw = trace_collate_fn([eval_dataset[i] for i in vis_indices])
        else:
            vis_raw = next(dl_iter)
        vis_batch = _tokenize_and_device(vis_raw, tokenizer, cfg.tokenizer_max_length, device)

    # Test vis batch: pinned once over the first-image-only test indices so the
    # same N test episodes are rendered as motion/vis_color across training.
    # Capped at `vis_num_samples`; if fewer test episodes exist, use all of them.
    test_vis_batch = None
    if is_main and test_dataset is not None and test_first_indices:
        n_test_vis = min(cfg.vis_num_samples, len(test_first_indices))
        test_vis_indices = test_first_indices[:n_test_vis]
        test_vis_raw = trace_collate_fn([test_dataset[i] for i in test_vis_indices])
        test_vis_batch = _tokenize_and_device(
            test_vis_raw, tokenizer, cfg.tokenizer_max_length, device
        )
        logging.info(
            f"Test vis batch: pinned {len(test_vis_indices)} samples "
            f"(first image of each of the first {n_test_vis} test episodes)"
        )

    wandb_run = None
    if cfg.wandb_enable and is_main:
        import wandb

        # Persist run id once at init time so subsequent resumes attach to the same
        # wandb run (single curve, monotonic step axis). Delete wandb_run.json if you
        # want a resume to start a fresh wandb run instead.
        wandb_run_path = cfg.output_dir / "wandb_run.json"
        init_kwargs: dict[str, Any] = dict(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.job_name,
            mode=cfg.wandb_mode,
            config=draccus.encode(cfg),
        )
        if wandb_run_path.is_file():
            run_id = json.loads(wandb_run_path.read_text())["run_id"]
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "must"
            logging.info(f"Resuming wandb run id={run_id}")
        wandb_run = wandb.init(**init_kwargs)
        if not wandb_run_path.is_file():
            wandb_run_path.write_text(json.dumps({"run_id": wandb_run.id}))

    loss_meter = AverageMeter("loss", ":.4f")
    update_meter = AverageMeter("updt_s", ":.3f")
    data_meter = AverageMeter("data_s", ":.3f")
    grad_norm_meter = AverageMeter("grad_norm", ":.4f")
    # Populated only when smooth_loss_weight > 0 and the policy returns the
    # split components in `loss_dict`. See docs/improve_training_pipe_plan.md §4.5.
    loss_fm_meter = AverageMeter("loss_fm", ":.4f")
    # Populated only when trace_done_head=True. See docs/trace_done_head_plan.md.
    loss_done_meter = AverageMeter("loss_done", ":.4f")
    # Populated only when rigidity_regularization_weight > 0.
    loss_rigidity_meter = AverageMeter("loss_rigidity", ":.4f")
    nan_skip_count = 0

    progbar = tqdm(
        total=cfg.steps,
        initial=start_step,
        desc="Training",
        unit="step",
        disable=not is_main,
    )
    for step in range(start_step + 1, cfg.steps + 1):
        data_t0 = time.perf_counter()
        batch = _tokenize_and_device(next(dl_iter), tokenizer, cfg.tokenizer_max_length, device)
        data_meter.update(time.perf_counter() - data_t0)

        # Debug: on the first step, dump 3 history+GT overlays + stats so the user
        # can verify what TraceDataset is feeding the model. GT is passed as both
        # traj_future_gt and traj_future_pred so the "pred" markers sit exactly on
        # the GT path (we just want to see the sampled dataset, not model output).

        # if step == 1 and is_main:
        #     from PIL import Image as _PILImage

        #     dbg_dir = cfg.output_dir / "debug_samples"
        #     dbg_dir.mkdir(parents=True, exist_ok=True)
        #     imgs = batch[IMAGE_KEY]
        #     traj_hist = batch[TRAJ_HISTORY_KEY]
        #     traj_fut = batch[TRAJ_FUTURE_KEY]
        #     valid_hist_b = batch[VALID_HISTORY_KEY]
        #     valid_fut_b = batch[VALID_FUTURE_KEY]
        #     kpm_b = batch[KEY_PAD_MASK_KEY]
        #     n_vis = min(8, imgs.shape[0])
        #     logging.info(
        #         f"[debug sample] shapes: image={tuple(imgs.shape)} "
        #         f"traj_history={tuple(traj_hist.shape)} traj_future={tuple(traj_fut.shape)} "
        #         f"key_pad_mask={tuple(kpm_b.shape)}"
        #     )
        #     for i in range(n_vis):
        #         img_np = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        #         overlay = render_trace_overlay(
        #             image=img_np,
        #             traj_history=traj_hist[i].cpu().numpy(),
        #             traj_future_gt=traj_fut[i].cpu().numpy(),
        #             traj_future_pred=traj_fut[i].cpu().numpy(),
        #             valid_history=valid_hist_b[i].cpu().numpy(),
        #             valid_future=valid_fut_b[i].cpu().numpy(),
        #             key_pad_mask=kpm_b[i].cpu().numpy(),
        #         )
        #         n_kp = int(kpm_b[i].sum().item())
        #         n_vh = int(valid_hist_b[i].sum().item())
        #         n_vf = int(valid_fut_b[i].sum().item())
        #         hist_xy = traj_hist[i, :, :, :2]
        #         logging.info(
        #             f"  sample {i}: N_valid_kp={n_kp} valid_hist={n_vh} valid_fut={n_vf} "
        #             f"xy_hist=[{hist_xy.min().item():.2f}, {hist_xy.max().item():.2f}] "
        #             f"z_hist=[{traj_hist[i, :, :, 2].min().item():.2f}, "
        #             f"{traj_hist[i, :, :, 2].max().item():.2f}]"
        #         )
        #         _PILImage.fromarray(overlay).save(dbg_dir / f"sample_{i}.png")
        #     logging.info(f"[debug sample] saved {n_vis} overlays to {dbg_dir}")

        upd_t0 = time.perf_counter()
        loss, _loss_dict = policy(batch)

        # NaN-skip guard (forward). DDP-coordinated: any rank's non-finite loss
        # forces every rank to skip the optimizer step, so we never desync the
        # all-reduce in `accelerator.backward`. On trip, dump batch + first
        # non-finite module on main rank for offline inspection.
        loss_nonfinite = (~torch.isfinite(loss)).to(torch.int32)
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(loss_nonfinite, op=torch.distributed.ReduceOp.MAX)
        if loss_nonfinite.item() > 0:
            first_nan = nan_detector.find_first_nan() if nan_detector is not None else None
            nan_skip_count += 1
            if is_main:
                dump_dir = cfg.output_dir / "nan_dumps" / f"step_{step:08d}"
                try:
                    _dump_nan_batch(
                        dump_dir, step, float(loss.detach()), first_nan, batch,
                    )
                except Exception as e:
                    logging.warning(f"[nan-skip] dump failed: {e}")
                logging.error(
                    f"[nan-skip] step={step} loss={loss.item():.6f} "
                    f"first_non_finite_module={first_nan} (dumped to {dump_dir})"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {"train/nan_skip": float(nan_skip_count), "train/nan_step": float(step)},
                        step=step,
                    )
            optimizer.zero_grad(set_to_none=True)
            if nan_detector is not None:
                nan_detector.reset()
            update_meter.update(time.perf_counter() - upd_t0)
            progbar.update(1)
            continue

        accelerator.backward(loss)
        grad_norm = None
        if cfg.grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)

        # NaN-skip guard (gradients). bf16 + full-FT can produce non-finite grad
        # norm even when forward loss was finite (grad-only overflow). Skip the
        # optimizer step so non-finite grads don't poison every parameter.
        grad_nonfinite_local = torch.zeros((), dtype=torch.int32, device=device)
        if grad_norm is not None:
            grad_nonfinite_local = (~torch.isfinite(grad_norm.detach())).to(torch.int32)
        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(grad_nonfinite_local, op=torch.distributed.ReduceOp.MAX)
        if grad_nonfinite_local.item() > 0:
            # Hooks may have fired during recompute (gradient checkpointing) —
            # if so, the first non-finite module is informative for grad-only
            # overflows too. Returns None when forward was clean and the
            # overflow happened in backward proper.
            first_nan = nan_detector.find_first_nan() if nan_detector is not None else None
            nan_skip_count += 1
            if is_main:
                gn_str = (
                    f"{grad_norm.item():.4e}" if grad_norm is not None else "None"
                )
                logging.error(
                    f"[nan-skip] step={step} non-finite grad_norm={gn_str} "
                    f"first_non_finite_module={first_nan}, skipping optimizer step"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/nan_grad_skip": float(nan_skip_count),
                            "train/nan_step": float(step),
                        },
                        step=step,
                    )
            optimizer.zero_grad(set_to_none=True)
            if nan_detector is not None:
                nan_detector.reset()
            update_meter.update(time.perf_counter() - upd_t0)
            progbar.update(1)
            continue

        optimizer.step()
        optimizer.zero_grad()
        if nan_detector is not None:
            nan_detector.reset()
        update_meter.update(time.perf_counter() - upd_t0)
        loss_meter.update(loss.item())
        if "loss_fm" in _loss_dict:
            loss_fm_meter.update(_loss_dict["loss_fm"])
        if "loss_done" in _loss_dict:
            loss_done_meter.update(_loss_dict["loss_done"])
        if "loss_rigidity" in _loss_dict:
            loss_rigidity_meter.update(_loss_dict["loss_rigidity"])
        if grad_norm is not None:
            grad_norm_meter.update(float(grad_norm.detach()))
        progbar.update(1)

        if step % cfg.log_freq == 0 and is_main:
            msg = (
                f"step={step} loss={loss_meter.avg:.4f} "
                f"grad_norm={grad_norm_meter.avg:.4f} "
                f"data_s={data_meter.avg:.3f} updt_s={update_meter.avg:.3f} "
                f"nan_skips={nan_skip_count}"
            )
            logging.info(msg)
            if wandb_run is not None:
                payload = {
                    "train/loss": loss_meter.avg,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm_meter.avg,
                    "train/nan_skip_total": float(nan_skip_count),
                }
                if loss_fm_meter.count > 0:
                    payload["train/loss_fm"] = loss_fm_meter.avg
                if loss_done_meter.count > 0:
                    payload["train/loss_done"] = loss_done_meter.avg
                if loss_rigidity_meter.count > 0:
                    payload["train/loss_rigidity"] = loss_rigidity_meter.avg
                wandb_run.log(payload, step=step)
            loss_meter.reset()
            update_meter.reset()
            data_meter.reset()
            grad_norm_meter.reset()
            loss_fm_meter.reset()
            loss_done_meter.reset()
            loss_rigidity_meter.reset()

        # Held-out eval + overlays: both run on main only at `eval_every_n_steps`.
        # Non-main ranks block at the barriers so DDP allreduces stay paired.
        # `current_eval_loss` is captured here and read by the checkpoint block below
        # to decide whether the freshly-saved checkpoint is the new best.
        current_eval_loss: float | None = None
        if cfg.eval_every_n_steps > 0 and step % cfg.eval_every_n_steps == 0:
            accelerator.wait_for_everyone()
            if is_main:
                unwrapped = accelerator.unwrap_model(policy)
                if eval_dl is not None:
                    t0 = time.perf_counter()
                    metrics = _run_eval(
                        unwrapped,
                        eval_dl,
                        tokenizer,
                        cfg.tokenizer_max_length,
                        device,
                        cfg.eval_max_batches,
                        delta_scale=delta_scale,
                        num_samples=cfg.eval_num_samples,
                    )
                    eval_s = time.perf_counter() - t0
                    if metrics is not None:
                        current_eval_loss = float(metrics["eval/loss"])
                        N = cfg.eval_num_samples
                        logging.info(
                            f"[eval @ step {step}] "
                            f"loss={metrics['eval/loss']:.4f} "
                            f"l1_abs={metrics['eval/l1_abs']:.4f} "
                            f"l1_uv={metrics['eval/l1_uv']:.4f} "
                            f"l1_depth={metrics['eval/l1_depth']:.4f} "
                            f"l1_delta={metrics['eval/l1_delta']:.4f} "
                            f"ade_uvz={metrics['eval/ade_uvz']:.4f} "
                            f"fde_uvz={metrics['eval/fde_uvz']:.4f} "
                            f"dtw_uvz={metrics['eval/dtw_uvz']:.4f} "
                            f"minADE_{N}={metrics[f'eval/minADE_{N}_uvz']:.4f} "
                            f"minFDE_{N}={metrics[f'eval/minFDE_{N}_uvz']:.4f} "
                            f"smooth_pred={metrics['eval/smooth_pred_uvz']:.4f} "
                            f"smooth_gt={metrics['eval/smooth_gt_uvz']:.4f} "
                            f"n={int(metrics['eval/num_batches'])} ({eval_s:.1f}s)"
                        )
                        if wandb_run is not None:
                            wandb_run.log(metrics, step=step)
                _log_trace_vis(
                    unwrapped,
                    vis_batch,
                    step,
                    wandb_run,
                    cfg.vis_num_samples,
                    delta_scale=delta_scale,
                )
                # External test set: same metric set + motion/vis as eval,
                # logged under `test/` and `test_vis/`. One sample per episode
                # (first image only), so we run the full test loader uncapped.
                if test_dl is not None:
                    t0 = time.perf_counter()
                    test_metrics = _run_eval(
                        unwrapped,
                        test_dl,
                        tokenizer,
                        cfg.tokenizer_max_length,
                        device,
                        max_batches=0,
                        delta_scale=delta_scale,
                        num_samples=cfg.eval_num_samples,
                        prefix="test/",
                    )
                    test_s = time.perf_counter() - t0
                    if test_metrics is not None:
                        N = cfg.eval_num_samples
                        logging.info(
                            f"[test @ step {step}] "
                            f"loss={test_metrics['test/loss']:.4f} "
                            f"l1_abs={test_metrics['test/l1_abs']:.4f} "
                            f"ade_uvz={test_metrics['test/ade_uvz']:.4f} "
                            f"fde_uvz={test_metrics['test/fde_uvz']:.4f} "
                            f"dtw_uvz={test_metrics['test/dtw_uvz']:.4f} "
                            f"minADE_{N}={test_metrics[f'test/minADE_{N}_uvz']:.4f} "
                            f"minFDE_{N}={test_metrics[f'test/minFDE_{N}_uvz']:.4f} "
                            f"n={int(test_metrics['test/num_batches'])} ({test_s:.1f}s)"
                        )
                        if wandb_run is not None:
                            wandb_run.log(test_metrics, step=step)

                    # Companion eval with depth stripped from the batch, logged
                    # under `test_wo_depth/`. Lets us compare with-vs-without
                    # depth on the SAME test samples on wandb. Only meaningful
                    # when the policy was built with depth at all.
                    if cfg.use_depth:
                        t0 = time.perf_counter()
                        test_wo_depth_metrics = _run_eval(
                            unwrapped,
                            test_dl,
                            tokenizer,
                            cfg.tokenizer_max_length,
                            device,
                            max_batches=0,
                            delta_scale=delta_scale,
                            num_samples=cfg.eval_num_samples,
                            prefix="test_wo_depth/",
                            drop_depth=True,
                        )
                        test_wo_depth_s = time.perf_counter() - t0
                        if test_wo_depth_metrics is not None:
                            N = cfg.eval_num_samples
                            logging.info(
                                f"[test_wo_depth @ step {step}] "
                                f"loss={test_wo_depth_metrics['test_wo_depth/loss']:.4f} "
                                f"l1_abs={test_wo_depth_metrics['test_wo_depth/l1_abs']:.4f} "
                                f"ade_uvz={test_wo_depth_metrics['test_wo_depth/ade_uvz']:.4f} "
                                f"fde_uvz={test_wo_depth_metrics['test_wo_depth/fde_uvz']:.4f} "
                                f"dtw_uvz={test_wo_depth_metrics['test_wo_depth/dtw_uvz']:.4f} "
                                f"minADE_{N}={test_wo_depth_metrics[f'test_wo_depth/minADE_{N}_uvz']:.4f} "
                                f"minFDE_{N}={test_wo_depth_metrics[f'test_wo_depth/minFDE_{N}_uvz']:.4f} "
                                f"n={int(test_wo_depth_metrics['test_wo_depth/num_batches'])} ({test_wo_depth_s:.1f}s)"
                            )
                            if wandb_run is not None:
                                wandb_run.log(test_wo_depth_metrics, step=step)
                if test_vis_batch is not None:
                    _log_trace_vis(
                        unwrapped,
                        test_vis_batch,
                        step,
                        wandb_run,
                        cfg.vis_num_samples,
                        delta_scale=delta_scale,
                        prefix="test_vis/",
                    )
            accelerator.wait_for_everyone()

        if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
            # Decide on main whether this is a new best (lowest eval/loss seen),
            # then broadcast the decision so every rank agrees: save_state is
            # collective and must be entered (or skipped) by all ranks together.
            # `current_eval_loss` is only set on main, so non-main ranks rely on
            # the broadcast value rather than their own (None -> False).
            is_new_best = bool(
                current_eval_loss is not None and current_eval_loss < best_eval_loss
            )
            flag_holder = [is_new_best]
            broadcast_object_list(flag_holder, from_process=0)
            is_new_best = flag_holder[0]

            if cfg.save_best_only and not is_new_best:
                # Only best checkpoints are persisted: write nothing this step.
                if is_main:
                    if current_eval_loss is None:
                        logging.info(
                            f"[step {step}] save_best_only: no eval/loss at this step "
                            f"(align eval_every_n_steps with save_freq); skipping checkpoint."
                        )
                    else:
                        logging.info(
                            f"[step {step}] save_best_only: eval/loss "
                            f"{current_eval_loss:.4f} did not beat best "
                            f"{best_eval_loss:.4f}; skipping checkpoint."
                        )
            else:
                accelerator.wait_for_everyone()
                ckpt_dir = cfg.output_dir / "checkpoints" / f"step_{step:08d}"
                # save_state must be called from every rank (it shards RNG state per rank
                # and barriers internally). save_pretrained + meta.json are main-only.
                if is_main:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                accelerator.wait_for_everyone()
                accelerator.save_state(str(ckpt_dir / "accel_state"))
                if is_main:
                    accelerator.unwrap_model(policy).save_pretrained(ckpt_dir)
                    (ckpt_dir / "meta.json").write_text(json.dumps({"step": step}))
                    logging.info(f"saved checkpoint to {ckpt_dir}")

                    # Keep only the most recent checkpoint and the best (lowest eval/loss)
                    # one (under save_best_only the just-saved best is the only survivor).
                    # On a new best, persist `best_meta.json` so a later resume picks it up.
                    if is_new_best:
                        best_eval_loss = current_eval_loss
                        best_ckpt_step = step
                        best_meta_path.write_text(
                            json.dumps({"step": step, "eval_loss": best_eval_loss})
                        )
                        logging.info(
                            f"new best checkpoint: step={step} eval/loss={best_eval_loss:.4f}"
                        )

                    keep_steps: set[int] = set()
                    if not cfg.save_best_only:
                        keep_steps.add(step)
                    if best_ckpt_step is not None:
                        keep_steps.add(best_ckpt_step)
                    ckpt_root = cfg.output_dir / "checkpoints"
                    for old_ckpt in ckpt_root.glob("step_*"):
                        if not old_ckpt.is_dir():
                            continue
                        try:
                            old_step = int(old_ckpt.name.split("_")[1])
                        except (IndexError, ValueError):
                            continue
                        if old_step in keep_steps:
                            continue
                        shutil.rmtree(old_ckpt)
                        logging.info(f"pruned old checkpoint {old_ckpt}")

    progbar.close()
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
