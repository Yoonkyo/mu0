#!/usr/bin/env python
"""Image-only trace prediction for the SmolVLA trace policy.

Drops the dataset-side dependencies on `depth.npy` /
`samples/traj_history.npy` / `samples/frame_indices.npy` at inference time
by synthesizing them from `(RGB image, language task)`:

  - depth → Depth-Anything-V2 metric, aligned to training log-depth, then
    rendered with the same `render_depth_rgb` used at training time.
  - current_kp → GroundedSAM hand mask + 16 random pixels in [-1, 1] UV.
  - traj_history / traj_future → all-zeros + all-False in the GSAM path
    (history is zeroed by `_replace_kp_with_gsam` because we only have a
    single frame). Set `--use_kp_from_dataset=true` to feed real history
    from the dataset; the policy-side `mask_all_history` switch is exposed
    via `--no_provide_history` (default True, mirroring the training
    full-mask branch).

Two ablation flags swap a synthesized input back for its dataset original:

  --use_kp_from_dataset=true   skip GSAM, use TraceDataset's current_kp / future
  --use_depth_from={data,pred,none}
                                where to get the depth panel for the policy:
                                  data — TraceDataset's depth.npy
                                  pred — Depth-Anything-V2 (default)
                                  none — no depth panel; the depth img_mask is
                                         zeroed and the policy effectively sees
                                         RGB+text only (matches the train-time
                                         --depth_dropout_prob drop branch).
                                         Requires a checkpoint trained with
                                         depth_dropout_prob>0.

L1 metrics are gated on `use_kp_from_dataset=true` (needs GT trajectories
anchored to the same keypoints). With `--use_kp_from_dataset=true
--use_depth_from=data --no_provide_history=true` the script reduces to the
existing predict script with --no_provide_history=true (sanity check). To
run with real history, pass `--use_kp_from_dataset=true
--no_provide_history=false`.

See `docs/predict_trace_image_only_plan.md` for the full design.

Example (headline image-only mode, no GT, no L1):

    python src/lerobot/scripts/lerobot_predict_trace_mu0_image_only.py \
        --checkpoint=/path/to/checkpoints/step_NNNNNNNN \
        --test_dirs='[/path/to/test_dataset_vimabench/*]' \
        --first_frame_only=true --seed=0 \
        --delta_stats_path=/path/to/delta_stats.json \
        --output_dir=outputs/predict/eval_image_only \
        --max_samples=8 --save_depth_panels=true

Example (sanity check — both inputs from dataset, should match the existing
predict script's L1 with --no_provide_history=true):

    ... \
        --use_kp_from_dataset=true --use_depth_from=data \
        --output_dir=outputs/predict/agibot_apr24_000_May01_test2_sanity

Example (no-depth ablation; only valid for checkpoints trained with
--depth_dropout_prob>0):

    ... \
        --use_kp_from_dataset=true --use_depth_from=none \
        --output_dir=outputs/predict/agibot_apr24_000_May01_test2_no_depth
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
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
from lerobot.datasets.trace_delta_stats import load_trace_stats
from lerobot.datasets.trace_depth import render_depth_rgb
from lerobot.policies.mu0.visualize_trace import render_trace_motion_still
from lerobot.policies.mu0.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.utils import init_logging


_DEFAULT_DA_V2_CKPT = "infer_helpers/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
_DEFAULT_GDINO_CONFIG = (
    "infer_helpers/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
_DEFAULT_GDINO_CKPT = "infer_helpers/checkpoints/groundingdino_swint_ogc.pth"
_DEFAULT_SAM_CKPT = "infer_helpers/checkpoints/sam2.1_hiera_large.pt"
_DEFAULT_SAM_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass
class TracePredictImageOnlyConfig:
    # Required.
    checkpoint: Path | None = None
    test_dirs: list[str] = field(default_factory=list)
    delta_stats_path: Path | None = None

    # Output
    output_dir: Path | None = None

    # Inference
    batch_size: int = 8
    num_workers: int = 0  # DA-V2 / GSAM live on the inference GPU; keep single-process.
    num_inference_steps: int | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Sampling overrides (None → use ckpt config). n_min/n_max are consumed
    # only when --use_kp_from_dataset=true (the GSAM path forces N=num_keypoints).
    n_min: int | None = None
    n_max: int | None = None
    history_len: int | None = None
    future_len: int | None = None
    image_size: int | None = None
    min_future_motion: float = 0.0
    seed: int = 0

    # Limits / saving behavior
    max_samples: int = 0
    save_depth_panels: bool = False
    shuffle: bool = False
    # When True, restrict the test loader to the earliest usable frame of
    # each video (one sample per video). Renamed from `diverse_sampling`
    # to reflect the deterministic first-frame selection — see
    # `_build_first_frame_indices`.
    first_frame_only: bool = False
    # When `first_frame_only=True`, skip the first N usable slots of each
    # video and pick slot N (0 = earliest, the legacy behavior). Useful when
    # the model needs real history: the earliest usable slot has
    # history_len True bits in `valid_history` only if the source video
    # itself starts that many frames before; otherwise some/all past slots
    # are zero-padded. Set offset>0 to ensure history is non-trivial. Videos
    # with fewer than `offset+1` usable slots are skipped with a warning.
    # Accepts a single int (legacy) or a list of ints; with a list the chosen
    # slot for each offset is unioned (deduped) into one bigger sample set and
    # metrics are computed once over the union — i.e.
    # `--first_frame_offset=[4,8,12]` gives up to 3 samples per video and the
    # reported metrics are the sample-weighted average across all three offsets.
    first_frame_offset: int | list[int] = 0

    # Targeted query.
    query_video: Path | None = None
    query_frames: list[int] = field(default_factory=list)

    # 3D
    save_3d: bool = False
    use_full_video: bool = False
    full_video_frame_stride: int = 16
    full_video_pixel_downsample: int = 8
    full_video_max_frames: int = 64
    pixel_downsample: int = 4

    # ---- Ablation toggles (plan §1, §4.2) ----
    use_kp_from_dataset: bool = False
    # data → TraceDataset's depth.npy; pred → Depth-Anything-V2 (default);
    # none → no depth panel (zeroed mask, mirrors --depth_dropout_prob drop).
    use_depth_from: str = "pred"
    num_keypoints: int = 16  # consumed only when use_kp_from_dataset=false

    # When True (default), mask all history tokens at inference — mirrors the
    # training full-mask branch (--trace_history_full_mask_prob>0). Set False
    # together with --use_kp_from_dataset=true to feed real history through
    # the policy. --no_provide_history=false combined with
    # --use_kp_from_dataset=false is a hard error (GSAM has no history for
    # arbitrary pixels; the zeros+pad_mask=True state is silently OOD vs. training).
    no_provide_history: bool = True

    # Done-head truncation threshold (docs/trace_done_head_plan.md). Only
    # consulted when the loaded checkpoint has `trace_done_head=True`. Default
    # `None` means "use the value baked into the checkpoint config".
    done_threshold: float | None = None

    # Best-of-N sampling for metrics. Matches training-eval's
    # `--eval_num_samples` (docs/minADE_minFDE_plan.md): `predict_trace` is
    # called N times per batch with independent noise, sample-0 feeds
    # legacy ADE/FDE/DTW + L1 + visualization, and minADE/minFDE are
    # computed across the stacked samples. Only consulted when
    # `--use_kp_from_dataset=true` (no GT → no min metrics). Set to 1 to
    # collapse back to single-sample behavior.
    num_samples_for_metrics: int = 5

    # Extra horizons at which to compute short-term metrics (in addition to the
    # full --future_len, which always lands in metrics.json). For each H listed,
    # we recompute l1_* / ade_* / fde_* / dtw_* / frechet_* / minADE_N_* /
    # minFDE_N_* / minDTW_N_* / minFrechet_N_* on `gt[:, :, :H]` vs
    # `pred[:, :, :H]` and write the result to `metrics_{H}.json`. Horizons >
    # future_len are dropped with a warning.
    # Only consulted when --use_kp_from_dataset=true (no GT → no metrics).
    metric_horizons: list[int] = field(default_factory=lambda: [8, 16, 32])

    # ---- Depth-Anything-V2 ----
    depth_align: str = "log_percentile"  # log_percentile | median_scale | none
    depth_anything_ckpt: Path = Path(_DEFAULT_DA_V2_CKPT)
    depth_anything_encoder: str = "vitl"
    depth_anything_input_size: int = 518
    diagnose_alignment: bool = False

    # ---- GroundedSAM (Grounded-SAM-2: GroundingDINO + SAM 2.1) ----
    gdino_config: Path = Path(_DEFAULT_GDINO_CONFIG)
    gdino_ckpt: Path = Path(_DEFAULT_GDINO_CKPT)
    sam_ckpt: Path = Path(_DEFAULT_SAM_CKPT)
    sam_model_cfg: str = _DEFAULT_SAM_MODEL_CFG  # Hydra config name resolved by sam2 package
    gdino_box_threshold: float = 0.30
    gdino_text_threshold: float = 0.25
    gdino_box_threshold_pass2: float = 0.15
    gdino_text_threshold_pass2: float = 0.15
    fallback_random_on_no_mask: bool = True

    def __post_init__(self) -> None:
        if self.checkpoint is None:
            raise ValueError("--checkpoint is required.")
        if self.delta_stats_path is None:
            raise ValueError("--delta_stats_path is required.")
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint}")
        if not Path(self.delta_stats_path).exists():
            raise FileNotFoundError(f"delta_stats_path not found: {self.delta_stats_path}")
        if self.depth_align not in ("log_percentile", "median_scale", "none"):
            raise ValueError(
                f"--depth_align must be log_percentile/median_scale/none; got {self.depth_align!r}"
            )
        if self.use_depth_from not in ("data", "pred", "none"):
            raise ValueError(
                f"--use_depth_from must be data/pred/none; got {self.use_depth_from!r}"
            )
        if self.num_keypoints < 1:
            raise ValueError(f"--num_keypoints must be >= 1: {self.num_keypoints}")
        if not self.no_provide_history and not self.use_kp_from_dataset:
            raise ValueError(
                "--no_provide_history=false requires --use_kp_from_dataset=true. "
                "GSAM has no historical trace for arbitrary pixels, so the "
                "history would be all-zeros with pad_mask=True — silently OOD "
                "vs. training. Pass --use_kp_from_dataset=true to feed real "
                "history, or keep the default --no_provide_history=true."
            )
        if self.num_samples_for_metrics < 1:
            raise ValueError(
                f"--num_samples_for_metrics must be >= 1: {self.num_samples_for_metrics}"
            )
        if any(h < 1 for h in self.metric_horizons):
            raise ValueError(
                f"--metric_horizons entries must be >= 1: {self.metric_horizons}"
            )
        offsets_norm = (
            [self.first_frame_offset]
            if isinstance(self.first_frame_offset, int)
            else list(self.first_frame_offset)
        )
        if not offsets_norm or any(o < 0 for o in offsets_norm):
            raise ValueError(
                f"--first_frame_offset entries must be >= 0: {self.first_frame_offset}"
            )
        if self.query_video is not None:
            if not self.query_frames:
                raise ValueError("--query_video set but --query_frames is empty.")
            if not Path(self.query_video).exists():
                raise FileNotFoundError(f"--query_video not found: {self.query_video}")
            if not self.test_dirs:
                self.test_dirs = [str(self.query_video)]
        elif not self.test_dirs:
            raise ValueError("test_dirs is empty — pass at least one TraceExtract video dir or glob.")


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _make_tokenizer(vlm_model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(vlm_model_name)
    tok.padding_side = "right"
    return tok


def _tokenize_and_device(
    batch: dict[str, Any],
    tokenizer,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
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


_META_DIR_KEY = "__video_dir"
_META_FRAME_KEY = "__frame_idx"


class _MetaTraceDataset(Dataset):
    """Wraps `TraceDataset` to attach `(video_dir, frame_idx)` to each sample."""

    def __init__(self, base: TraceDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base._get_one(idx)
        video_idx, slot = self.base._index[idx]
        v = self.base._videos[video_idx]
        sample[_META_DIR_KEY] = str(v.video_dir)
        sample[_META_FRAME_KEY] = int(v.frame_indices[slot])
        return sample


def _meta_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    dirs = [s.get(_META_DIR_KEY, "") for s in samples]
    frames = [int(s.get(_META_FRAME_KEY, -1)) for s in samples]
    out = trace_collate_fn(samples)
    out[_META_DIR_KEY] = dirs
    out[_META_FRAME_KEY] = frames
    return out


def _build_first_frame_indices(
    base: TraceDataset, rng: random.Random, offset: int = 0,
) -> list[int]:
    """One sample per video, picking the (offset+1)-th-earliest usable slot.

    Slots are sorted by their actual frame index inside the source video so the
    order is robust to future changes in `usable_slots` ordering. When
    `offset=0` (default) we pick the earliest slot — legacy behavior. When
    `offset>0` we pick the (offset+1)-th — useful for getting valid history
    (the earliest usable slot may still have zero-padded `valid_history` if
    the source video itself starts at frame 0). Videos with fewer than
    `offset+1` usable slots are skipped with a warning. `rng` shuffles the
    final list so the DataLoader sees a mixed sequence rather than dir-glob order.
    """
    by_video: dict[int, list[int]] = {}
    for ds_idx, (video_idx, _slot) in enumerate(base._index):
        by_video.setdefault(video_idx, []).append(ds_idx)
    chosen: list[int] = []
    skipped = 0
    for video_idx, idxs in by_video.items():
        v = base._videos[video_idx]
        sorted_idxs = sorted(idxs, key=lambda di: int(v.frame_indices[base._index[di][1]]))
        if offset >= len(sorted_idxs):
            skipped += 1
            continue
        chosen.append(sorted_idxs[offset])
    if skipped > 0:
        logging.warning(
            f"--first_frame_offset={offset}: skipped {skipped} video(s) with "
            f"fewer than {offset + 1} usable slots"
        )
    rng.shuffle(chosen)
    return chosen


def _resolve_query_indices(
    base: TraceDataset, video_dir: str, frames: list[int]
) -> list[int]:
    target = str(Path(video_dir).resolve())
    target_video_idx: int | None = None
    for vi, v in enumerate(base._videos):
        if str(Path(v.video_dir).resolve()) == target:
            target_video_idx = vi
            break
    if target_video_idx is None:
        avail = [str(v.video_dir) for v in base._videos[:5]]
        raise ValueError(
            f"--query_video {video_dir} not found in dataset. "
            f"Make sure --test_dirs covers it. First few available: {avail}"
        )
    v = base._videos[target_video_idx]
    frame_to_slot = {int(v.frame_indices[s]): s for s in v.usable_slots}
    slot_to_dsidx = {
        slot: ds_idx
        for ds_idx, (vi, slot) in enumerate(base._index)
        if vi == target_video_idx
    }
    chosen: list[int] = []
    missing: list[int] = []
    for f in frames:
        if f in frame_to_slot:
            chosen.append(slot_to_dsidx[frame_to_slot[f]])
        else:
            missing.append(int(f))
    if missing:
        usable = sorted(frame_to_slot.keys())
        preview = usable[:20]
        more = "" if len(usable) <= 20 else f" ...({len(usable)} total)"
        logging.warning(
            f"--query_video {video_dir}: skipping frames {missing} (no usable slot). "
            f"Usable frame_idx (first 20): {preview}{more}"
        )
    if not chosen:
        raise ValueError(
            f"--query_video {video_dir}: none of {frames} are usable. "
            f"Try one of the indices listed above."
        )
    return chosen


def _load_policy(ckpt: Path, device: str) -> SmolVLAPolicy:
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.load_vlm_weights = False
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(str(ckpt), config=cfg, strict=False)
    return policy


def _image_chw_to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    """`(3, H, W)` float ∈ [0, 1] → `(H, W, 3)` uint8 RGB."""
    arr = img_chw.permute(1, 2, 0).float().cpu().numpy()
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _render_depth_to_chw(depth_meters: np.ndarray, log_min: float, log_max: float) -> torch.Tensor:
    """`(H, W)` meters → `(3, H, W)` float32 in [0, 1] (turbo RGB)."""
    rgb = render_depth_rgb(depth_meters, log_min, log_max)  # (H, W, 3) float32 in [0, 1]
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def _maybe_resize_chw(t: torch.Tensor, size: int) -> torch.Tensor:
    if t.shape[-2] == size and t.shape[-1] == size:
        return t
    return torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


def _synthesize_depth_batch(
    raw_images_uint8: list[np.ndarray],
    da_v2,
    align_method: str,
    log_min: float,
    log_max: float,
    image_size: int,
) -> tuple[torch.Tensor, list[np.ndarray], list[dict]]:
    """Per-image DA-V2 + align + render. Returns:
      - depth_turbo: (B, 3, image_size, image_size) float in [0, 1]
      - aligned_meters: list of (H, W) float32 (original resolution) for §3.6 z-sampling
      - diags: list of per-frame alignment percentile dicts.
    """
    panels: list[torch.Tensor] = []
    aligned_list: list[np.ndarray] = []
    diags: list[dict] = []
    for rgb in raw_images_uint8:
        d_aligned, diag = da_v2.infer_aligned(rgb, align_method, log_min, log_max)
        # `render_depth_rgb` treats NaN as invalid → far/dim endpoint, which is what we want.
        depth_chw = _render_depth_to_chw(d_aligned, log_min, log_max)
        depth_chw = _maybe_resize_chw(depth_chw, image_size)
        panels.append(depth_chw)
        aligned_list.append(d_aligned)
        diags.append(diag)
    return torch.stack(panels, dim=0), aligned_list, diags


def _replace_kp_with_gsam(
    batch: dict[str, Any],
    raw_images_uint8: list[np.ndarray],
    aligned_depth_meters: list[np.ndarray] | None,
    gsam,
    num_keypoints: int,
    history_len: int,
    future_len: int,
    rng: np.random.Generator,
) -> list[str]:
    """Overwrite (current_kp / traj_history / traj_future / valid_* / key_pad_mask)
    with GSAM-derived keypoints.

    For each sample i:
      - `current_kp[i, :K, :2]` ← GSAM-sampled UV in [-1, 1].
      - `current_kp[i, :K, 2]`  ← `aligned_depth_meters[i][row, col]` if available, else 0.
      - history / future / valid_* / key_pad_mask are zeroed and key_pad_mask[:K]=True.

    Returns `statuses`: one of {"pass1", "pass2", "random_fallback"} per sample.
    """
    B = len(raw_images_uint8)
    device = batch[IMAGE_KEY].device
    K = num_keypoints

    current_kp = torch.zeros(B, K, 3, dtype=torch.float32, device=device)
    traj_history = torch.zeros(B, K, history_len, 3, dtype=torch.float32, device=device)
    traj_future = torch.zeros(B, K, future_len, 3, dtype=torch.float32, device=device)
    valid_history = torch.zeros(B, K, history_len, dtype=torch.bool, device=device)
    valid_future = torch.zeros(B, K, future_len, dtype=torch.bool, device=device)
    key_pad_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    key_pad_mask[:] = True

    statuses: list[str] = []
    from lerobot.inference_helpers.grounded_sam import sample_keypoints_from_mask

    for i, rgb in enumerate(raw_images_uint8):
        seg = gsam.segment_hand(rgb)
        H, W = rgb.shape[:2]
        uv, rowcol, status = sample_keypoints_from_mask(
            seg, image_hw=(H, W), num_keypoints=K, rng=rng
        )
        statuses.append(status)
        current_kp[i, :, :2] = torch.from_numpy(uv).to(device)
        if aligned_depth_meters is not None:
            d = aligned_depth_meters[i]
            zs = d[rowcol[:, 0], rowcol[:, 1]]
            zs = np.where(np.isfinite(zs) & (zs > 1e-3), zs, 0.0).astype(np.float32)
            current_kp[i, :, 2] = torch.from_numpy(zs).to(device)

    batch[CURRENT_KP_KEY] = current_kp
    batch[TRAJ_HISTORY_KEY] = traj_history
    batch[TRAJ_FUTURE_KEY] = traj_future
    batch[VALID_HISTORY_KEY] = valid_history
    batch[VALID_FUTURE_KEY] = valid_future
    batch[KEY_PAD_MASK_KEY] = key_pad_mask
    return statuses


# Per-axis dim selections for Fréchet / minFréchet — mirror the train-side
# `_TRAJ_METRIC_DIMS` so the dict keys produced here (`frechet_uvz`, etc.)
# line up with the existing ADE/FDE/DTW series. uvz mixes normalized [-1,1]
# xy with raw-meter z, matching the `l1_abs` aggregation.
_FRECHET_METRIC_DIMS: dict[str, tuple[int, ...]] = {
    "uvz": (0, 1, 2),
    "uv": (0, 1),
    "depth": (2,),
}


@torch.no_grad()
def _frechet_sums(
    pred_abs: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
    dim_idx: tuple[int, ...],
) -> tuple[float, int]:
    """Per-trajectory Discrete Fréchet Distance with L2 cost on selected dims.

    Discrete Fréchet (Eiter & Mannila 1994) finds the min over monotonic
    alignments of the MAX pairwise distance along that alignment — the
    "tightest leash" bottleneck. Recurrence:

        c(i, j) = max( d(P_i, Q_j),
                       min(c(i-1, j), c(i, j-1), c(i-1, j-1)) )

    with `c(0, 0) = d(P_0, Q_0)` and border rows/cols `c(i, 0) =
    max(c(i-1, 0), d(P_i, Q_0))` (analogously for `c(0, j)`). Unlike DTW
    (sum, length-normalized), Fréchet is a max so no length normalization.

    Compresses non-contiguous valid timesteps into temporal order, groups
    trajectories by valid length, and runs a vectorized numpy DP per
    length-group so Python overhead is O(num_unique_lengths * T^2) rather
    than O(B*N*T^2). Returns (frechet_sum, n_traj).
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
        cost = np.empty((m_count, t_valid, t_valid), dtype=np.float64)
        cost[:, 0, 0] = pdist[:, 0, 0]
        for i in range(1, t_valid):
            cost[:, i, 0] = np.maximum(cost[:, i - 1, 0], pdist[:, i, 0])
        for j in range(1, t_valid):
            cost[:, 0, j] = np.maximum(cost[:, 0, j - 1], pdist[:, 0, j])
        for i in range(1, t_valid):
            for j in range(1, t_valid):
                best_prev = np.minimum(
                    np.minimum(cost[:, i - 1, j], cost[:, i, j - 1]),
                    cost[:, i - 1, j - 1],
                )
                cost[:, i, j] = np.maximum(pdist[:, i, j], best_prev)
        per_frechet = cost[:, t_valid - 1, t_valid - 1]  # (M,)
        total += float(per_frechet.sum())
        count += m_count

    return total, count


def _accumulate_frechet_metrics(
    acc: dict,
    pred_abs: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
) -> None:
    """Per-batch Fréchet sums/counts for uvz / uv / depth. Adds `frechet_*`
    keys alongside the `ade_*` / `fde_*` / `dtw_*` keys produced by the
    train-script `_accumulate_traj_metrics`."""
    for name, dim_idx in _FRECHET_METRIC_DIMS.items():
        d, nd = _frechet_sums(pred_abs, gt_abs, valid_future, kpm, dim_idx)
        acc[f"frechet_{name}_sum"] = acc.get(f"frechet_{name}_sum", 0.0) + d
        acc[f"frechet_{name}_n"] = acc.get(f"frechet_{name}_n", 0) + nd


def _finalize_frechet_metrics(acc: dict, prefix: str) -> dict[str, float]:
    """Convert Fréchet accumulators into `{prefix}frechet_{name}` means."""
    out: dict[str, float] = {}
    for name in _FRECHET_METRIC_DIMS:
        s = acc.get(f"frechet_{name}_sum", 0.0)
        n = acc.get(f"frechet_{name}_n", 0)
        out[f"{prefix}frechet_{name}"] = s / max(n, 1)
    return out


@torch.no_grad()
def _min_frechet_sums(
    pred_abs_S: torch.Tensor,    # (B, S, N_kp, F, 3)
    gt_abs: torch.Tensor,         # (B, N_kp, F, 3)
    valid_future: torch.Tensor,   # (B, N_kp, F)
    kpm: torch.Tensor,            # (B, N_kp)
    dim_idx: tuple[int, ...],
) -> tuple[float, int]:
    """Best-of-S minFréchet per (b, kp). Returns (sum, n_traj).

    Per (b, kp): compress to its T valid future steps, run the discrete
    Fréchet DP on each of the S predicted trajectories against the single
    GT, then take the min across S. DP is vectorized over S (shape
    `(S, T, T)`). Mean-over-(b, kp) reduction matches ADE/FDE/DTW callers.
    """
    valid_t = (valid_future & kpm.unsqueeze(-1)).cpu().numpy()
    pred_np = pred_abs_S[..., list(dim_idx)].detach().to(torch.float32).cpu().numpy()
    gt_np = gt_abs[..., list(dim_idx)].detach().to(torch.float32).cpu().numpy()
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
                d_per_s = np.linalg.norm(pred_S[:, 0] - gt_traj[0], axis=-1)
                total += float(d_per_s.min())
                n_traj += 1
                continue
            diff = pred_S[:, :, None, :] - gt_traj[None, None, :, :]
            pdist = np.linalg.norm(diff, axis=-1).astype(np.float64)
            cost = np.empty((S, t_valid, t_valid), dtype=np.float64)
            cost[:, 0, 0] = pdist[:, 0, 0]
            for i in range(1, t_valid):
                cost[:, i, 0] = np.maximum(cost[:, i - 1, 0], pdist[:, i, 0])
            for j in range(1, t_valid):
                cost[:, 0, j] = np.maximum(cost[:, 0, j - 1], pdist[:, 0, j])
            for i in range(1, t_valid):
                for j in range(1, t_valid):
                    best_prev = np.minimum(
                        np.minimum(cost[:, i - 1, j], cost[:, i, j - 1]),
                        cost[:, i - 1, j - 1],
                    )
                    cost[:, i, j] = np.maximum(pdist[:, i, j], best_prev)
            per_s_frechet = cost[:, t_valid - 1, t_valid - 1]
            total += float(per_s_frechet.min())
            n_traj += 1
    return total, n_traj


def _accumulate_min_frechet_metrics(
    acc: dict,
    pred_abs_S: torch.Tensor,
    gt_abs: torch.Tensor,
    valid_future: torch.Tensor,
    kpm: torch.Tensor,
) -> None:
    """Accumulate Best-of-S minFréchet for uvz / uv / depth."""
    for name, dim_idx in _FRECHET_METRIC_DIMS.items():
        s, n = _min_frechet_sums(pred_abs_S, gt_abs, valid_future, kpm, dim_idx)
        acc[f"min_frechet_{name}_sum"] = acc.get(f"min_frechet_{name}_sum", 0.0) + s
        acc[f"min_frechet_{name}_n"] = acc.get(f"min_frechet_{name}_n", 0) + n


def _finalize_min_frechet_metrics(
    acc: dict, prefix: str, num_samples: int,
) -> dict[str, float]:
    """Convert minFréchet accumulators into `{prefix}minFrechet_{N}_{name}` means."""
    out: dict[str, float] = {}
    for name in _FRECHET_METRIC_DIMS:
        s = acc.get(f"min_frechet_{name}_sum", 0.0)
        n = acc.get(f"min_frechet_{name}_n", 0)
        out[f"{prefix}minFrechet_{num_samples}_{name}"] = s / max(n, 1)
    return out


@draccus.wrap()
def predict(cfg: TracePredictImageOnlyConfig) -> None:
    """Image-only trace prediction for the SmolVLA trace policy.

    Overlays/videos colorize each trajectory segment by its position within
    that keypoint's own valid horizon (the renderer's ``trace_color_by="age"``):
    oldest step (nearest the current frame) violet/purple → last valid step
    red, so every trace spans the full purple→red ramp and ends red regardless
    of done-head / GT truncation length. See `render_trace_motion_still`.
    """
    init_logging()

    device = torch.device(cfg.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
    use_autocast = dtype != torch.float32 and device.type == "cuda"

    logging.info(f"Loading policy from {cfg.checkpoint}")
    logging.info(
        f"  use_kp_from_dataset={cfg.use_kp_from_dataset} "
        f"use_depth_from={cfg.use_depth_from}"
    )
    use_depth_from_data = cfg.use_depth_from == "data"
    use_depth_from_pred = cfg.use_depth_from == "pred"
    use_depth_from_none = cfg.use_depth_from == "none"
    logging.info(f"  no_provide_history={cfg.no_provide_history} (mask_all_history)")
    policy = _load_policy(Path(cfg.checkpoint), cfg.device)
    policy.to(device=device)
    policy.eval()
    pcfg = policy.config

    history_len = cfg.history_len if cfg.history_len is not None else pcfg.history_len
    future_len = cfg.future_len if cfg.future_len is not None else pcfg.future_len
    n_min = cfg.n_min if cfg.n_min is not None else pcfg.n_min
    n_max = cfg.n_max if cfg.n_max is not None else pcfg.n_max
    image_size = cfg.image_size if cfg.image_size is not None else pcfg.resize_imgs_with_padding[0]
    num_inference_steps = cfg.num_inference_steps if cfg.num_inference_steps is not None else pcfg.num_steps
    use_depth = bool(getattr(pcfg, "use_depth", False))

    if not use_depth:
        raise ValueError(
            "Checkpoint has use_depth=False; the image-only path assumes a depth-conditioned model."
        )
    if use_depth_from_none:
        ckpt_drop = float(getattr(pcfg, "depth_dropout_prob", 0.0))
        if ckpt_drop <= 0.0:
            logging.warning(
                "--use_depth_from=none but checkpoint was trained with "
                f"depth_dropout_prob={ckpt_drop}. The model has never seen a "
                "fully-masked depth slot at train time; expect OOD behavior."
            )

    logging.info(f"Loading delta stats from {cfg.delta_stats_path}")
    trace_stats = load_trace_stats(cfg.delta_stats_path)
    delta_scale = trace_stats.delta_scale
    depth_log_range = trace_stats.depth_log_range
    if depth_log_range is None:
        raise ValueError(
            f"Checkpoint has use_depth=True but {cfg.delta_stats_path} has no depth_log_* range."
        )
    log_min_train, log_max_train = depth_log_range
    logging.info(
        f"  delta_scale={delta_scale}; depth_log_range={depth_log_range}; "
        f"history_len={history_len} future_len={future_len} "
        f"n_min={n_min} n_max={n_max} image_size={image_size}"
    )

    # TraceDataset always emits IMAGE_KEY/TASK_KEY/current_kp/traj_*; load_depth
    # is gated on the toggle so the synthesized-depth path doesn't pay the
    # (large) depth.npy mmap cost.
    #
    # We intentionally do NOT pass `bspline_n_ctrl` here even when the
    # checkpoint runs in bspline mode. That filter is a training-side concern
    # (the lstsq target needs >= D valid future steps), and `predict_trace` /
    # `sample_traces` never consume `ctrl_pts_future` / `valid_kp`. Forwarding
    # it would drop tail-end slots from the test set; the predict path is
    # happy to score those frames over however few valid future steps they
    # have. Mask-based metrics (ADE/FDE/DTW/minADE/minFDE) handle this
    # naturally because they accumulate only valid (b, kp) pairs.
    dataset = TraceDataset(
        video_dirs=cfg.test_dirs,
        future_len=future_len,
        history_len=history_len,
        n_min=n_min,
        n_max=n_max,
        image_size=image_size,
        seed=cfg.seed,
        min_future_motion=cfg.min_future_motion,
        delta_scale=delta_scale,
        load_depth=use_depth_from_data,
        depth_log_range=depth_log_range if use_depth_from_data else None,
    )
    logging.info(f"TraceDataset[test]: {len(dataset)} samples across {len(dataset._videos)} dirs")

    meta_dataset: Dataset = _MetaTraceDataset(dataset)
    loader_shuffle = cfg.shuffle
    if cfg.query_video is not None:
        chosen = _resolve_query_indices(dataset, str(cfg.query_video), cfg.query_frames)
        meta_dataset = Subset(meta_dataset, chosen)
        loader_shuffle = False
        logging.info(
            f"--query_video {cfg.query_video}: querying {len(chosen)} of "
            f"{len(cfg.query_frames)} requested frames"
        )
    elif cfg.first_frame_only:
        offsets = (
            [cfg.first_frame_offset]
            if isinstance(cfg.first_frame_offset, int)
            else list(cfg.first_frame_offset)
        )
        rng = random.Random(cfg.seed)
        if len(offsets) == 1:
            chosen = _build_first_frame_indices(dataset, rng, offset=offsets[0])
        else:
            # Per-offset RNG re-seeded from cfg.seed so the legacy single-offset
            # behavior is recoverable as `offsets=[O]` (matches cached responses).
            seen: set[int] = set()
            chosen = []
            for off in offsets:
                for ds_idx in _build_first_frame_indices(
                    dataset, random.Random(cfg.seed), offset=off,
                ):
                    if ds_idx not in seen:
                        chosen.append(ds_idx)
                        seen.add(ds_idx)
            rng.shuffle(chosen)
        meta_dataset = Subset(meta_dataset, chosen)
        loader_shuffle = False
        logging.info(
            f"--first_frame_only (offsets={offsets}): {len(chosen)} samples "
            f"from {len(dataset)} slots"
        )

    dataloader = DataLoader(
        meta_dataset,
        batch_size=cfg.batch_size,
        shuffle=loader_shuffle,
        num_workers=cfg.num_workers,
        collate_fn=_meta_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    tokenizer = _make_tokenizer(pcfg.vlm_model_name)

    # Helper models loaded once and kept resident on the inference GPU (plan §4.4).
    da_v2 = None
    gsam = None
    if use_depth_from_pred:
        from lerobot.inference_helpers.depth_anything import DepthAnythingV2Wrapper

        logging.info(f"Loading Depth-Anything-V2 from {cfg.depth_anything_ckpt}")
        da_v2 = DepthAnythingV2Wrapper(
            checkpoint_path=cfg.depth_anything_ckpt,
            encoder=cfg.depth_anything_encoder,
            input_size=cfg.depth_anything_input_size,
            device=cfg.device,
        )
    if not cfg.use_kp_from_dataset:
        from lerobot.inference_helpers.grounded_sam import GroundedHandSegmenter

        logging.info(
            f"Loading GroundedSAM (gdino={cfg.gdino_ckpt}, sam={cfg.sam_ckpt})"
        )
        gsam = GroundedHandSegmenter(
            gdino_config=cfg.gdino_config,
            gdino_checkpoint=cfg.gdino_ckpt,
            sam_checkpoint=cfg.sam_ckpt,
            sam_model_cfg=cfg.sam_model_cfg,
            device=cfg.device,
            box_threshold=cfg.gdino_box_threshold,
            text_threshold=cfg.gdino_text_threshold,
            box_threshold_pass2=cfg.gdino_box_threshold_pass2,
            text_threshold_pass2=cfg.gdino_text_threshold_pass2,
            fallback_random_on_no_mask=cfg.fallback_random_on_no_mask,
        )
    np_rng = np.random.default_rng(cfg.seed)

    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(cfg.checkpoint) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_depth_panels and not use_depth_from_none:
        (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    (output_dir / "rollout_img").mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving overlays to {output_dir}")

    saved = 0
    scale_t = torch.as_tensor(delta_scale, dtype=torch.float32, device=device)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if use_autocast
        else _NullCtx()
    )

    # Best-of-N: each batch runs `predict_trace` `n_samples` times. Sample-0
    # feeds visualization + legacy L1/ADE/FDE/DTW; minADE/minFDE come from the
    # stacked samples. Without GT (use_kp_from_dataset=false) Best-of-N has
    # nothing to score, so collapse to N=1.
    n_samples = cfg.num_samples_for_metrics if cfg.use_kp_from_dataset else 1
    logging.info(f"  num_samples_for_metrics={n_samples}")

    # Lazy import: pulls in the train script (and `accelerate`) only when
    # metrics are actually requested. These are pure helpers that take
    # (acc dict, tensors), so they're safe to share across scripts.
    # Fréchet / minFréchet are defined locally in this file (not imported
    # from train) — keeps the train-time eval surface untouched.
    if cfg.use_kp_from_dataset:
        from lerobot.scripts.lerobot_train_trace_mu0 import (
            _accumulate_min_dtw_metrics,
            _accumulate_min_metrics,
            _accumulate_traj_metrics,
            _finalize_min_dtw_metrics,
            _finalize_min_metrics,
            _finalize_traj_metrics,
        )

    # Build the list of horizons to evaluate. Always includes the full
    # future_len (lands in metrics.json); other horizons land in
    # metrics_{H}.json. Horizons > future_len are dropped with a warning.
    requested_horizons = sorted(set(cfg.metric_horizons))
    dropped_horizons = [h for h in requested_horizons if h > future_len]
    if dropped_horizons:
        logging.warning(
            f"--metric_horizons: dropping {dropped_horizons} (> future_len={future_len})"
        )
    horizons = sorted(set(
        [h for h in requested_horizons if h <= future_len] + [future_len]
    ))
    if cfg.use_kp_from_dataset:
        logging.info(
            f"Computing metrics at horizons: {horizons} (full = {future_len})"
        )
    # Per-horizon accumulator state. Each horizon has its own set of sums so
    # the same per-batch tensors can be sliced and counted multiple ways.
    state_per_h: dict[int, dict] = {
        h: {
            "l1_abs_sum": 0.0,
            "l1_uv_sum": 0.0,
            "l1_depth_sum": 0.0,
            "l1_delta_sum": 0.0,
            "n": 0,
            "traj_acc": {},
            "min_acc": {},
            "min_dtw_acc": {},
            "min_frechet_acc": {},
        }
        for h in horizons
    }

    alignment_log: list[dict] = []
    hand_status_log: list[dict] = []

    with torch.no_grad():
        for batch_idx, raw in enumerate(tqdm(dataloader, desc="predict", unit="batch")):
            if cfg.max_samples > 0 and saved >= cfg.max_samples:
                break

            # Move to device first so subsequent overrides land on the right device.
            batch = _tokenize_and_device(raw, tokenizer, pcfg.tokenizer_max_length, device)

            # The DataLoader yields RGB images already resized to image_size; for
            # DA-V2 / GSAM we want the resized RGB as well (consistent with the
            # rendered depth panel below). Decode once to (B, H, W, 3) uint8.
            raw_images_uint8 = [_image_chw_to_uint8_hwc(batch[IMAGE_KEY][b]) for b in range(batch[IMAGE_KEY].shape[0])]

            # 1) Depth synthesis (overrides DEPTH_KEY when not from dataset).
            aligned_depth_meters: list[np.ndarray] | None = None
            B_now = batch[IMAGE_KEY].shape[0]
            if use_depth_from_pred:
                depth_turbo, aligned_depth_meters, diags = _synthesize_depth_batch(
                    raw_images_uint8,
                    da_v2,
                    cfg.depth_align,
                    log_min_train,
                    log_max_train,
                    image_size,
                )
                batch[DEPTH_KEY] = depth_turbo.to(device)
                if cfg.diagnose_alignment:
                    meta_dirs = batch.get(_META_DIR_KEY, [""] * len(diags))
                    meta_frames = batch.get(_META_FRAME_KEY, [-1] * len(diags))
                    for i, d in enumerate(diags):
                        alignment_log.append(
                            {
                                "video_dir": str(meta_dirs[i]) if i < len(meta_dirs) else "",
                                "frame_idx": int(meta_frames[i]) if i < len(meta_frames) else -1,
                                **d,
                            }
                        )
            elif use_depth_from_none:
                # Mirror the train-time depth_dropout drop branch: inject a zero
                # depth panel and set its padding_mask all-False so the depth
                # tokens become non-attending in the prefix. Shape matches the
                # turbo-rendered panel from `_synthesize_depth_batch`.
                batch[DEPTH_KEY] = torch.zeros(
                    B_now, 3, image_size, image_size, dtype=torch.float32, device=device
                )
                batch[f"{DEPTH_KEY}_padding_mask"] = torch.zeros(
                    B_now, dtype=torch.bool, device=device
                )

            # 2) Keypoint synthesis (overrides current_kp / traj_* / valid_* / key_pad_mask).
            if not cfg.use_kp_from_dataset:
                statuses = _replace_kp_with_gsam(
                    batch=batch,
                    raw_images_uint8=raw_images_uint8,
                    aligned_depth_meters=aligned_depth_meters,
                    gsam=gsam,
                    num_keypoints=cfg.num_keypoints,
                    history_len=history_len,
                    future_len=future_len,
                    rng=np_rng,
                )
                meta_dirs = batch.get(_META_DIR_KEY, [""] * len(statuses))
                meta_frames = batch.get(_META_FRAME_KEY, [-1] * len(statuses))
                for i, s in enumerate(statuses):
                    hand_status_log.append(
                        {
                            "video_dir": str(meta_dirs[i]) if i < len(meta_dirs) else "",
                            "frame_idx": int(meta_frames[i]) if i < len(meta_frames) else -1,
                            "status": s,
                        }
                    )

            # Best-of-N: loop predict_trace with independent noise draws.
            # Mirrors lerobot_train_trace_mu0.py:1046-1063. Looping (vs.
            # batch-tiling) keeps peak memory bounded at the cost of N prefix
            # recomputes. done_prob is read from sample-0 to stay comparable
            # with prior single-sample runs.
            traces_list: list[torch.Tensor] = []
            done_prob_first: torch.Tensor | None = None
            with autocast_ctx:
                for s in range(n_samples):
                    pred = policy.predict_trace(
                        batch,
                        num_steps=num_inference_steps,
                        mask_all_history=cfg.no_provide_history,
                    )
                    if isinstance(pred, dict):
                        traces_list.append(pred["traces"])
                        if s == 0:
                            done_prob_first = pred["done_prob"]
                    else:
                        traces_list.append(pred)
            traces_S = torch.stack(traces_list, dim=1).float()  # (B, S, N, F, 3)
            traces = traces_S[:, 0]  # (B, N, F, 3) — sample-0 for legacy paths.
            done_prob = done_prob_first.float() if done_prob_first is not None else None

            valid_future = batch[VALID_FUTURE_KEY]
            kpm = batch[KEY_PAD_MASK_KEY]
            gt = batch[TRAJ_FUTURE_KEY]
            traj_hist = batch[TRAJ_HISTORY_KEY]
            valid_hist = batch[VALID_HISTORY_KEY]
            current_kp = batch[CURRENT_KP_KEY]

            # Done-head truncate-and-freeze (docs/trace_done_head_plan.md §1.6).
            # Step-resolution predicted-valid mask used to mask the renderer.
            pred_valid_step: torch.Tensor | None = None
            if done_prob is not None:
                threshold = (
                    cfg.done_threshold
                    if cfg.done_threshold is not None
                    else pcfg.done_threshold
                )
                # bspline: done head emits (B, N, H) directly
                # — see docs/trace_bspline_ctrl_pts_plan.md §3.
                pred_valid_step = (done_prob >= threshold)  # (B, N, H)

            traces_anchor = traces
            traces_S_anchor = traces_S
            gt_anchor = gt
            anchor = current_kp[:, :, None, :3]
            traces_abs = traces_anchor * scale_t + anchor
            traces_S_abs = traces_S_anchor * scale_t + anchor.unsqueeze(1)
            gt_abs = gt_anchor * scale_t + anchor
            traj_hist_abs = traj_hist.clone()
            traj_hist_abs[..., :3] = traj_hist_abs[..., :3] + current_kp[:, :, None, :3]

            # Per-horizon accumulation. Slice the time axis of every (B, ..., F, ...)
            # tensor at each H and recompute L1/ADE/FDE/DTW (sample-0) and
            # minADE/minFDE/minDTW (over S). All horizons share the same
            # predictions — only the slice changes — so this is essentially
            # free relative to the model forward passes. `l1_delta` is in loss
            # space (anchor-deltas) — computed on the `traces`/`gt` tensors.
            if cfg.use_kp_from_dataset:
                for h in horizons:
                    state = state_per_h[h]
                    traces_h = traces[:, :, :h, :]
                    traces_S_h = traces_S[:, :, :, :h, :]
                    gt_delta_h = gt[:, :, :h, :]
                    traces_abs_h = traces_abs[:, :, :h, :]
                    traces_S_abs_h = traces_S_abs[:, :, :, :h, :]
                    gt_abs_h = gt_abs[:, :, :h, :]
                    valid_h = valid_future[:, :, :h]
                    mask_h = (
                        valid_h.unsqueeze(-1) & kpm[:, :, None, None]
                    ).to(traces_abs_h.dtype)
                    denom_h = mask_h.sum().clamp_min(1.0)

                    abs_err_h = (traces_abs_h - gt_abs_h).abs() * mask_h
                    state["l1_abs_sum"] += float(abs_err_h.sum() / denom_h)
                    state["l1_uv_sum"] += float(abs_err_h[..., :2].sum() / denom_h)
                    state["l1_depth_sum"] += float(abs_err_h[..., 2:3].sum() / denom_h)
                    state["l1_delta_sum"] += float(
                        ((traces_h - gt_delta_h).abs() * mask_h).sum() / denom_h
                    )
                    state["n"] += 1

                    _accumulate_traj_metrics(
                        state["traj_acc"], traces_abs_h, gt_abs_h, valid_h, kpm,
                        include_dtw=True,
                    )
                    _accumulate_frechet_metrics(
                        state["traj_acc"], traces_abs_h, gt_abs_h, valid_h, kpm,
                    )
                    _accumulate_min_metrics(
                        state["min_acc"], traces_S_abs_h, gt_abs_h, valid_h, kpm,
                    )
                    _accumulate_min_dtw_metrics(
                        state["min_dtw_acc"], traces_S_abs_h, gt_abs_h, valid_h, kpm,
                    )
                    _accumulate_min_frechet_metrics(
                        state["min_frechet_acc"], traces_S_abs_h, gt_abs_h, valid_h, kpm,
                    )

            images = batch[IMAGE_KEY]
            depths = batch.get(DEPTH_KEY)
            meta_dirs = batch.get(_META_DIR_KEY, [""] * images.shape[0])
            meta_frames = batch.get(_META_FRAME_KEY, [-1] * images.shape[0])
            tasks = raw.get(TASK_KEY, [""] * images.shape[0])
            bsz = images.shape[0]
            for b in range(bsz):
                if cfg.max_samples > 0 and saved >= cfg.max_samples:
                    break
                img = _image_chw_to_uint8_hwc(images[b])
                traj_hist_abs_b = traj_hist_abs[b].float().cpu().numpy()
                traces_abs_b = traces_abs[b].float().cpu().numpy()
                valid_hist_b = valid_hist[b].cpu().numpy()
                valid_future_b = valid_future[b].cpu().numpy()
                kpm_b = kpm[b].cpu().numpy()
                current_kp_b = current_kp[b].float().cpu().numpy()
                # In image-only mode `valid_future` is all-False (no GT exists),
                # which would also hide the predicted polyline — override with
                # all-True so the prediction stays visible.
                valid_future_for_overlay = (
                    valid_future_b if cfg.use_kp_from_dataset else np.ones_like(valid_future_b)
                )
                pred_valid_step_b = (
                    pred_valid_step[b].cpu().numpy()
                    if pred_valid_step is not None
                    else None
                )
                vdir_b = str(meta_dirs[b]) if b < len(meta_dirs) else ""
                fidx_b = int(meta_frames[b]) if b < len(meta_frames) else -1
                vname = Path(vdir_b).name if vdir_b else f"b{batch_idx:05d}_i{b:02d}"
                fname = f"sample_{saved:06d}_{vname}_f{fidx_b:06d}.png"
                still_kwargs = dict(
                    image=img,
                    traj_history=traj_hist_abs_b,
                    traj_future_pred=traces_abs_b,
                    valid_history=valid_hist_b,
                    valid_future=valid_future_for_overlay,
                    current_kp=current_kp_b,
                    key_pad_mask=kpm_b,
                    predicted_valid_future=pred_valid_step_b,
                    future_only=True,
                    trail_alpha_min=0.7,
                    trace_color_by="age",
                )
                Image.fromarray(
                    render_trace_motion_still(
                        **still_kwargs, bg_alpha=0.5, bg_blend_color=(235, 235, 235),
                    )
                ).save(output_dir / "rollout_img" / fname)
                if cfg.save_depth_panels and depths is not None and not use_depth_from_none:
                    d_img = (depths[b].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(d_img).save(output_dir / "depth" / fname)
                saved += 1

    if cfg.use_kp_from_dataset and any(state_per_h[h]["n"] > 0 for h in horizons):
        # Build metrics_{H}.json for every horizon; the full horizon also lands
        # in metrics.json for backwards compatibility with existing consumers.
        metrics_per_h: dict[int, dict] = {}
        for h in horizons:
            state = state_per_h[h]
            if state["n"] == 0:
                continue
            metrics_h: dict[str, float] = {
                "horizon": float(h),
                "l1_abs": state["l1_abs_sum"] / state["n"],
                "l1_uv": state["l1_uv_sum"] / state["n"],
                "l1_depth": state["l1_depth_sum"] / state["n"],
                "l1_delta": state["l1_delta_sum"] / state["n"],
                "n_batches": float(state["n"]),
                "n_samples": float(n_samples),
            }
            # ADE/FDE/DTW/Fréchet (sample-0), minADE_{N}/minFDE_{N}/smoothness
            # (across N), minDTW_{N} (across N), and minFrechet_{N} (across N).
            # Keys mirror training's `eval/` series sans the prefix so they
            # collide cleanly when diffed against W&B.
            metrics_h.update(_finalize_traj_metrics(
                state["traj_acc"], prefix="", include_dtw=True,
            ))
            metrics_h.update(_finalize_frechet_metrics(
                state["traj_acc"], prefix="",
            ))
            metrics_h.update(_finalize_min_metrics(
                state["min_acc"], prefix="", num_samples=n_samples,
            ))
            metrics_h.update(_finalize_min_dtw_metrics(
                state["min_dtw_acc"], prefix="", num_samples=n_samples,
            ))
            metrics_h.update(_finalize_min_frechet_metrics(
                state["min_frechet_acc"], prefix="", num_samples=n_samples,
            ))
            metrics_per_h[h] = metrics_h
            fname = "metrics.json" if h == future_len else f"metrics_{h}.json"
            with open(output_dir / fname, "w") as f:
                json.dump(metrics_h, f, indent=2)

        # Headline log line: one row per horizon.
        lines = [f"[done] saved {saved} overlays to {output_dir}"]
        for h in horizons:
            if h not in metrics_per_h:
                continue
            m = metrics_per_h[h]
            tag = "full" if h == future_len else f"H={h}"
            lines.append(
                f"  [{tag:>5}] l1_abs={m['l1_abs']:.4f} l1_uv={m['l1_uv']:.4f} "
                f"l1_depth={m['l1_depth']:.4f} l1_delta={m['l1_delta']:.4f}"
            )
            lines.append(
                f"         ade_uvz={m.get('ade_uvz', 0.0):.4f} "
                f"fde_uvz={m.get('fde_uvz', 0.0):.4f} "
                f"dtw_uvz={m.get('dtw_uvz', 0.0):.4f} "
                f"frechet_uvz={m.get('frechet_uvz', 0.0):.4f}"
            )
            lines.append(
                f"         minADE_{n_samples}_uvz={m.get(f'minADE_{n_samples}_uvz', 0.0):.4f} "
                f"minFDE_{n_samples}_uvz={m.get(f'minFDE_{n_samples}_uvz', 0.0):.4f} "
                f"minDTW_{n_samples}_uvz={m.get(f'minDTW_{n_samples}_uvz', 0.0):.4f} "
                f"minFrechet_{n_samples}_uvz={m.get(f'minFrechet_{n_samples}_uvz', 0.0):.4f}"
            )
        short_files = [
            f"metrics_{h}.json" for h in horizons
            if h != future_len and h in metrics_per_h
        ]
        lines.append(
            f"  (n_samples={n_samples}; wrote metrics.json"
            + (f" + {', '.join(short_files)}" if short_files else " (no short-horizon files)")
            + ")"
        )
        logging.info("\n".join(lines))
    else:
        logging.info(
            f"[done] saved {saved} overlays to {output_dir} "
            f"(L1 / ADE / minADE / minDTW / minFréchet metrics gated on --use_kp_from_dataset=true)"
        )

    if cfg.diagnose_alignment and alignment_log:
        with open(output_dir / "alignment.json", "w") as f:
            json.dump(alignment_log, f, indent=2)
    if not cfg.use_kp_from_dataset and hand_status_log:
        with open(output_dir / "hand_mask_status.json", "w") as f:
            json.dump(hand_status_log, f, indent=2)


def main() -> None:
    predict()


if __name__ == "__main__":
    main()
