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
"""Precompute trace stats: per-axis delta scales + dataset-wide log-depth range.

Trace targets are deltas from the current frame `t` (TraceExtract `traj_history[:, 0]`,
exposed post-split as `current_kp` in TraceDataset — see
`docs/trace_current_kp_split_plan.md`), rescaled so the typical delta lands in
roughly [-1, 1]; this module computes the rescaling constants from the dataset
(95th percentile of |delta| per axis by default). In the same pass it also samples
depth values across the dataset and reports percentile-based log-meter endpoints
used by `render_depth_rgb` for a fixed turbo colormap range. The current_kp split
is a layout refactor of how the current frame surfaces in TraceDataset — the
TraceExtract slot used as anchor is unchanged, so JSONs computed pre-split remain
valid.

The delta target is `|p_t - p_anchor|` for `t=1..F`, matching TraceDataset's
anchor-relative future target. The output JSON records `target_kind="anchor"`
so it is self-describing; the training pipeline reads this and errors on a JSON
computed in any other mode. Dirs that lack `depth.npy` are still counted toward
the delta stats; they simply contribute no depth samples.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


@dataclass
class TraceStats:
    """In-memory view of a stats JSON.

    `depth_log_range` is `None` iff *no* scanned dir contained a usable `depth.npy`.
    Passing `None` to `TraceDataset(load_depth=True, ...)` is an error.

    `target_kind` is always `"anchor"`; JSONs without the field load as
    `"anchor"` for backward compatibility.
    """

    delta_scale: tuple[float, float, float]
    depth_log_range: tuple[float, float] | None
    target_kind: str = "anchor"


def _collect_stats(
    video_dir: Path,
    history_len: int,
    future_len: int,
    max_samples_per_video: int,
    depth_max_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Per-dir collector: returns (dx, dy, dz, depth_samples_or_None) or None.

    Deltas are `|p_t - p_anchor|` for every valid `(n, t)` pair — matches the
    dataset's anchor-relative future target.

    Depth sampling: mmap `depth.npy`, uniformly pick up to `depth_max_frames` time
    indices, 4× spatial downsample, filter to finite + d > 1e-3, return a flat
    meter array. `depth.npy` missing → depth_samples_or_None = None (dir still
    contributes delta stats).
    """
    samples = video_dir / "samples"
    required = [
        "traj.npy",
        "traj_history.npy",
        "valid_steps.npy",
        "valid_steps_history.npy",
    ]
    if not all((samples / f).exists() for f in required) or not (video_dir / "images.npy").exists():
        return None

    traj = np.load(samples / "traj.npy", mmap_mode="r")
    traj_history = np.load(samples / "traj_history.npy", mmap_mode="r")
    valid_steps = np.load(samples / "valid_steps.npy", mmap_mode="r")
    valid_steps_history = np.load(samples / "valid_steps_history.npy", mmap_mode="r")
    images = np.load(video_dir / "images.npy", mmap_mode="r")
    img_h, img_w = int(images.shape[1]), int(images.shape[2])

    # Need future_len + 1 columns: TraceExtract `traj[:, 0]` is the current frame
    # `t` (duplicated with traj_history), and we slice [1 : F+1] for the future.
    if traj.shape[1] < future_len + 1 or traj_history.shape[1] < history_len:
        return None

    # Anchor at TraceExtract index 0 (== current frame `t`, == `current_kp` in
    # TraceDataset's post-split output). Future slice [1 : F+1] excludes the
    # duplicated current frame so deltas are (t+k) - t for k=1..F, matching
    # the TraceDataset target.
    anchor = np.asarray(traj_history[:, 0, :3], dtype=np.float32)  # (N, 3)
    fut = np.asarray(traj[:, 1 : future_len + 1, :3], dtype=np.float32)  # (N, F, 3)
    v_anchor = np.asarray(valid_steps_history[:, 0], dtype=np.bool_)  # (N,)
    v_fut = np.asarray(valid_steps[:, 1 : future_len + 1], dtype=np.bool_)  # (N, F)

    finite_anchor = np.isfinite(anchor).all(axis=-1)  # (N,)
    finite_fut = np.isfinite(fut).all(axis=-1)  # (N, F)

    scale_xy = np.asarray([img_w, img_h], dtype=np.float32)
    # Only evaluate the delta where both anchor and future are finite; TraceExtract pads
    # invalid slots with -inf, which otherwise triggers RuntimeWarning on subtract.
    with np.errstate(invalid="ignore"):
        anchor_xy_norm = anchor[:, :2] / scale_xy * 2.0 - 1.0  # (N, 2)
    # Match TraceDataset: current-frame uv must lie in [-1, 1]^2 after normalization.
    anchor_in_bounds = np.zeros(anchor.shape[0], dtype=np.bool_)
    if finite_anchor.any():
        anchor_in_bounds[finite_anchor] = (np.abs(anchor_xy_norm[finite_anchor]) <= 1.0).all(axis=-1)

    keep_anchor = v_anchor & finite_anchor & anchor_in_bounds  # (N,)
    keep_fut = v_fut & finite_fut  # (N, F)

    # Normalise the future xy to [-1, 1] once.
    with np.errstate(invalid="ignore"):
        fut_xy_norm = fut[..., :2] / scale_xy * 2.0 - 1.0  # (N, F, 2)

    # `prev` is always the anchor: the delta target is `|p_t - p_anchor|`.
    prev_xy = np.broadcast_to(anchor_xy_norm[:, None, :], fut_xy_norm.shape)
    prev_z = np.broadcast_to(anchor[:, None, 2], fut[..., 2].shape)
    # The "prev" is always the anchor — already filtered by `keep_anchor`, no
    # extra per-step prev validity to AND in.
    v_prev = np.broadcast_to(keep_anchor[:, None], (anchor.shape[0], future_len))

    mask = keep_anchor[:, None] & v_prev & keep_fut  # (N, F)
    if not mask.any():
        return None

    n_idx, f_idx = np.nonzero(mask)
    delta_xy_valid = fut_xy_norm[n_idx, f_idx, :] - prev_xy[n_idx, f_idx, :]  # (K, 2)
    delta_z_valid = fut[n_idx, f_idx, 2] - prev_z[n_idx, f_idx]  # (K,)

    dx = np.abs(delta_xy_valid[:, 0])
    dy = np.abs(delta_xy_valid[:, 1])
    dz = np.abs(delta_z_valid)

    if max_samples_per_video > 0 and dx.size > max_samples_per_video:
        sel = rng.choice(dx.size, max_samples_per_video, replace=False)
        dx, dy, dz = dx[sel], dy[sel], dz[sel]

    depth_samples: np.ndarray | None = None
    depth_path = video_dir / "depth.npy"
    if depth_path.exists():
        try:
            depth = np.load(depth_path, mmap_mode="r")
        except (ValueError, OSError) as e:
            logging.warning(f"depth stats: skipping {depth_path} ({type(e).__name__}: {e})")
            depth = None
        if depth is not None and depth.ndim == 3 and depth.shape[0] > 0:
            T = depth.shape[0]
            k = min(depth_max_frames, T)
            if k > 0:
                t_sel = rng.choice(T, size=k, replace=False) if k < T else np.arange(T)
                # 4x spatial downsample is plenty for percentile estimation and keeps RAM low.
                frames = np.asarray(depth[t_sel], dtype=np.float32)[:, ::4, ::4]
                mask_d = np.isfinite(frames) & (frames > 1e-3)
                if mask_d.any():
                    depth_samples = frames[mask_d].astype(np.float32)

    return dx, dy, dz, depth_samples


def compute_trace_stats(
    video_dirs: list[str],
    history_len: int,
    future_len: int,
    percentile: float = 95.0,
    max_samples_per_video: int = 200_000,
    depth_max_frames: int = 32,
    depth_percentile_low: float = 1.0,
    depth_percentile_high: float = 99.0,
    seed: int = 0,
    show_progress: bool = True,
    video_fraction: float = 1.0,
) -> dict[str, float | int | list[str] | None]:
    """Compute per-axis delta scales + dataset-wide log-depth range across `video_dirs`.

    xy is measured in normalized [-1, 1] coords (matches TraceDataset); z in raw meters.
    Uses the current frame `t` as anchor — TraceExtract `traj_history[:, 0]`, which is
    `traj_history[:, h-1]` post-reversal in TraceDataset (see
    `docs/trace_time_axis_fix_plan.md`); masks follow the same filters TraceDataset
    applies. Depth samples are pooled across all dirs with `depth.npy`, then
    `log_min = log(p_low)`, `log_max = log(p_high)` on that pool.

    `video_fraction < 1.0` sub-samples each pattern's matched dirs (deterministic
    via `seed`) before the heavy per-dir read loop — useful for fast iteration
    on large corpora where the 95th-percentile estimate converges well before
    you need every dir.
    """
    if not 0.0 < video_fraction <= 1.0:
        raise ValueError(
            f"video_fraction must be in (0, 1], got {video_fraction}"
        )
    rng = np.random.default_rng(seed)
    resolved: list[str] = []
    for pattern in video_dirs:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_dir():
            matches = [pattern]
        if video_fraction < 1.0 and len(matches) > 1:
            n_take = max(1, int(round(len(matches) * video_fraction)))
            sel = rng.choice(len(matches), size=n_take, replace=False)
            matches = [matches[i] for i in sorted(sel.tolist())]
        resolved.extend(matches)
    if not resolved:
        raise ValueError(f"no video dirs matched: {video_dirs}")

    all_dx, all_dy, all_dz = [], [], []
    all_depth: list[np.ndarray] = []
    used_dirs: list[str] = []
    depth_dirs: list[str] = []
    total_kept = 0
    pbar = tqdm(
        resolved,
        desc="trace stats",
        unit="vid",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for vdir in pbar:
        pbar.set_postfix_str(Path(vdir).name, refresh=False)
        try:
            out = _collect_stats(
                Path(vdir),
                history_len,
                future_len,
                max_samples_per_video,
                depth_max_frames,
                rng,
            )
        except (FileNotFoundError, ValueError, OSError) as e:
            logging.warning(f"trace stats: skipping {vdir} ({type(e).__name__}: {e})")
            continue
        if out is None:
            continue
        dx, dy, dz, depth_samples = out
        all_dx.append(dx)
        all_dy.append(dy)
        all_dz.append(dz)
        used_dirs.append(str(vdir))
        total_kept += dx.size
        if depth_samples is not None and depth_samples.size > 0:
            all_depth.append(depth_samples)
            depth_dirs.append(str(vdir))
        pbar.set_postfix(dirs=len(used_dirs), samples=total_kept, depth_dirs=len(depth_dirs), refresh=False)

    if not all_dx:
        raise ValueError(f"no valid (anchor, future) pairs found across {len(resolved)} video(s)")

    cat_dx = np.concatenate(all_dx)
    cat_dy = np.concatenate(all_dy)
    cat_dz = np.concatenate(all_dz)

    out_dict: dict[str, float | int | str | list[str] | None] = {
        "delta_scale_x": float(np.percentile(cat_dx, percentile)),
        "delta_scale_y": float(np.percentile(cat_dy, percentile)),
        "delta_scale_z": float(np.percentile(cat_dz, percentile)),
        "percentile": float(percentile),
        "num_samples": int(cat_dx.size),
        "history_len": int(history_len),
        "future_len": int(future_len),
        "target_kind": "anchor",
        "video_fraction": float(video_fraction),
        # Depth fields — may be null if no dir had a usable depth.npy.
        "depth_log_min": None,
        "depth_log_max": None,
        "depth_p1_m": None,
        "depth_p99_m": None,
        "depth_num_samples": 0,
        "depth_num_dirs": len(depth_dirs),
        "depth_percentile_low": float(depth_percentile_low),
        "depth_percentile_high": float(depth_percentile_high),
    }

    if all_depth:
        cat_depth = np.concatenate(all_depth)
        p_low = float(np.percentile(cat_depth, depth_percentile_low))
        p_high = float(np.percentile(cat_depth, depth_percentile_high))
        # Guard the log: if the sampled set somehow has p_low<=0, fall back to tiny eps.
        p_low = max(p_low, 1e-3)
        p_high = max(p_high, p_low * 1.01)
        out_dict["depth_log_min"] = float(np.log(p_low))
        out_dict["depth_log_max"] = float(np.log(p_high))
        out_dict["depth_p1_m"] = p_low
        out_dict["depth_p99_m"] = p_high
        out_dict["depth_num_samples"] = int(cat_depth.size)

    return out_dict


def load_trace_stats(path: str | Path) -> TraceStats:
    """Read a stats JSON written by `compute_trace_stats` and return a TraceStats.

    Raises ValueError on JSONs that predate depth support (no `depth_*` keys at all) —
    regenerate with `python -m lerobot.datasets.trace_delta_stats`. JSONs without
    a `target_kind` field default to `"anchor"`.
    """
    p = Path(path)
    with open(p) as f:
        data = json.load(f)
    if not any(k.startswith("depth_") for k in data):
        raise ValueError(
            f"stats JSON at {p} is missing depth_* keys — regenerate with "
            f"`python -m lerobot.datasets.trace_delta_stats`"
        )
    delta_scale = (
        float(data["delta_scale_x"]),
        float(data["delta_scale_y"]),
        float(data["delta_scale_z"]),
    )
    log_min = data.get("depth_log_min")
    log_max = data.get("depth_log_max")
    depth_log_range: tuple[float, float] | None
    depth_log_range = (float(log_min), float(log_max)) if log_min is not None and log_max is not None else None
    target_kind = str(data.get("target_kind", "anchor"))
    if target_kind != "anchor":
        raise ValueError(
            f"stats JSON at {p} has unsupported target_kind={target_kind!r}; expected 'anchor'"
        )
    return TraceStats(
        delta_scale=delta_scale,
        depth_log_range=depth_log_range,
        target_kind=target_kind,
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_dirs", nargs="+", required=True, help="globs or dirs (TraceExtract outputs)")
    p.add_argument("--history_len", type=int, required=True)
    p.add_argument("--future_len", type=int, required=True)
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--max_samples_per_video", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--video_fraction",
        type=float,
        default=1.0,
        help=(
            "Per-pattern sub-sampling ratio in (0, 1]. With 0.1 each glob's "
            "matched dirs are randomly sub-sampled to ~10%% before the heavy "
            "per-dir read loop. Default 1.0 reads every match. The sub-sample "
            "is deterministic given --seed and is recorded in the output JSON."
        ),
    )
    p.add_argument("--output_path", type=Path, required=True, help="where to write stats JSON")
    args = p.parse_args()

    stats = compute_trace_stats(
        video_dirs=args.video_dirs,
        history_len=args.history_len,
        future_len=args.future_len,
        percentile=args.percentile,
        max_samples_per_video=args.max_samples_per_video,
        seed=args.seed,
        video_fraction=args.video_fraction,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
