#!/usr/bin/env python
"""Eyeball the B-spline LS fit quality on real data BEFORE training.

Renders side-by-side overlays of the GT future trajectory vs the
``N @ lstsq(N, target)`` reconstruction onto the frame-`t` RGB image, for a
handful of randomly-sampled examples. Use it to decide whether
``--trace_bspline_n_ctrl`` (D) is large enough for your data: this fit is
the supervision ceiling of the bspline-mode trainer.

Example:

    python -m lerobot.scripts.lerobot_visualize_bspline_fit \
        --video_dirs '[/path/to/agibot_May3/*]' \
        --delta_stats_path /path/to/delta_stats.json \
        --n_ctrl 10 \
        --history_len 8 --future_len 32 \
        --image_size 512 \
        --num_samples 12 \
        --output_dir /tmp/bspline_fit_d10
"""

import glob
import logging
import random as _random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import numpy as np
import torch
from PIL import Image

from lerobot.datasets.bspline_basis import build_bspline_basis
from lerobot.datasets.trace_dataset import (
    CTRL_PTS_FUTURE_KEY,
    CURRENT_KP_KEY,
    IMAGE_KEY,
    TRAJ_FUTURE_KEY,
    TRAJ_HISTORY_KEY,
    VALID_FUTURE_KEY,
    VALID_HISTORY_KEY,
    VALID_KP_KEY,
    TraceDataset,
)
from lerobot.datasets.trace_delta_stats import load_trace_stats
from lerobot.policies.mu0.visualize_trace import (
    _kp_color_palette,
    render_trace_overlay,
)


def _safe_label(pattern: str) -> str:
    """Make a filesystem-safe short label from a glob pattern.

    e.g. ``/path/to/agibot_May3/*`` -> ``agibot_May3``.
    """
    parts = [p for p in pattern.replace("*", "").rstrip("/").split("/") if p]
    return parts[-1] if parts else "pattern"


@dataclass
class VisualizeBsplineFitConfig:
    video_dirs: list[str] = field(default_factory=list)
    delta_stats_path: "Path | None" = None
    output_dir: Path = Path("/tmp/bspline_fit_vis")
    n_ctrl: int = 10  # one of {4, 7, 10}
    history_len: int = 8
    future_len: int = 32
    n_min: int = 16
    n_max: int = 64  # smaller than train default to keep overlays readable
    image_size: int = 512
    samples_per_pattern: int = 10  # render this many random samples per video_dirs entry
    max_dirs_per_pattern: int = 30  # cap how many resolved dirs each pattern actually scans
    seed: int = 0
    min_future_motion: float = 0.0
    use_depth: bool = False  # ignored for the overlay (RGB-only); kept for stats compat
    # Tikhonov regularization on the per-kp lstsq (forwarded to TraceDataset).
    # `reg_lambda > 0` pulls adjacent ctrl pts together (order=1 → even
    # spacing, order=2 → smooth polygon). `ctrl_clip` element-wise clamps
    # P_gt to [-c, c] post-fit. All three default to no-op.
    bspline_reg_lambda: float = 0.0
    bspline_reg_order: int = 1
    bspline_ctrl_clip: float | None = None


@draccus.wrap()
def visualize(cfg: VisualizeBsplineFitConfig) -> None:
    # Lerobot's import side-effects pre-configure the root logger; force INFO.
    logging.getLogger().setLevel(logging.INFO)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.n_ctrl not in (4, 7, 10):
        raise ValueError(f"n_ctrl must be in {{4, 7, 10}}; got {cfg.n_ctrl}")

    # Load delta_scale from stats so traj_future / P_gt live in the same scaled
    # space as the trainer; we un-scale to absolute xy + raw z just before
    # passing to the overlay renderer.
    if cfg.delta_stats_path is None:
        raise ValueError("--delta_stats_path is required (sets delta_scale)")
    stats = load_trace_stats(Path(cfg.delta_stats_path))
    delta_scale: tuple[float, float, float] = stats.delta_scale
    depth_log_range = stats.depth_log_range if cfg.use_depth else None

    # (H, D) basis for rendering at the H non-anchor query points (same one the
    # model uses at inference, see modeling_smolvla.py:_bspline_render_basis).
    basis_render = build_bspline_basis(cfg.n_ctrl, cfg.future_len, include_anchor=False)
    delta_t = torch.as_tensor(delta_scale, dtype=torch.float32)
    rng = np.random.RandomState(cfg.seed + 1)
    residual_max_per_sample: list[float] = []
    residual_mean_per_sample: list[float] = []

    # Per-pattern processing: scan + sample only one pattern at a time so total
    # work is bounded by `samples_per_pattern` rather than the full corpus.
    for pat_i, pattern in enumerate(cfg.video_dirs):
        pattern_label = _safe_label(pattern)
        out_subdir = cfg.output_dir / f"{pat_i:02d}_{pattern_label}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        logging.info(f"\n=== pattern {pat_i + 1}/{len(cfg.video_dirs)}: {pattern} ===")

        # Resolve glob and pick a small random subset BEFORE TraceDataset's
        # per-dir parse runs — capping work to ~max_dirs_per_pattern dirs
        # rather than the full pattern (which can be thousands).
        matched = sorted(glob.glob(pattern))
        if not matched and Path(pattern).is_dir():
            matched = [pattern]
        if not matched:
            logging.warning(f"pattern matched no dirs; skipping")
            continue
        if cfg.max_dirs_per_pattern > 0 and len(matched) > cfg.max_dirs_per_pattern:
            picker = _random.Random(cfg.seed + pat_i)
            picker.shuffle(matched)
            matched = matched[: cfg.max_dirs_per_pattern]
        logging.info(f"  scanning {len(matched)} resolved dir(s)")

        dataset = TraceDataset(
            video_dirs=matched,
            future_len=cfg.future_len,
            history_len=cfg.history_len,
            n_min=cfg.n_min,
            n_max=cfg.n_max,
            image_size=cfg.image_size,
            seed=cfg.seed + pat_i,
            min_future_motion=cfg.min_future_motion,
            delta_scale=delta_scale,
            load_depth=cfg.use_depth,
            depth_log_range=depth_log_range,
            bspline_n_ctrl=cfg.n_ctrl,
            bspline_reg_lambda=cfg.bspline_reg_lambda,
            bspline_reg_order=cfg.bspline_reg_order,
            bspline_ctrl_clip=cfg.bspline_ctrl_clip,
            cache_dir=Path(cfg.delta_stats_path).parent,
        )
        if len(dataset) == 0:
            logging.warning(f"pattern matched no usable samples; skipping")
            continue
        logging.info(f"TraceDataset[{pattern_label}]: {len(dataset)} samples")

        n_pick = min(cfg.samples_per_pattern, len(dataset))
        indices = rng.choice(len(dataset), size=n_pick, replace=False)

        for vis_i, idx in enumerate(indices):
            sample = dataset[int(idx)]
            image = sample[IMAGE_KEY]
            traj_history = sample[TRAJ_HISTORY_KEY].clone()
            traj_future = sample[TRAJ_FUTURE_KEY].clone()
            valid_history = sample[VALID_HISTORY_KEY]
            valid_future = sample[VALID_FUTURE_KEY]
            current_kp = sample[CURRENT_KP_KEY].clone()
            P_gt = sample[CTRL_PTS_FUTURE_KEY]
            valid_kp = sample[VALID_KP_KEY]

            X_fit_scaled = torch.einsum("hd,ndc->nhc", basis_render, P_gt)

            # Residual on per-step valid mask only — gives a meaningful
            # number even for kps that lose validity partway through the
            # H=32 future (common in agibot-style short clips).
            vf = valid_future.bool()
            if vf.any():
                diff = (X_fit_scaled - traj_future)             # (n, H, 3)
                res_norm = diff.norm(dim=-1)                    # (n, H)
                # Mean over valid (kp, step) pairs only.
                res_norm_valid = res_norm[vf]
                residual_max_per_sample.append(float(res_norm_valid.max().item()))
                residual_mean_per_sample.append(float(res_norm_valid.mean().item()))
            else:
                residual_max_per_sample.append(float("nan"))
                residual_mean_per_sample.append(float("nan"))

            # Diagnostic: how many kps does the model actually train on?
            # `valid_kp` is True iff the kp has >= D valid future steps
            # (the post-fix threshold for a non-singular weighted lstsq).
            # `any_valid` is the upper bound — kps with at least one valid step.
            n_total_kp = vf.shape[0]
            n_model_eligible = int(valid_kp.bool().sum().item())
            n_any_valid = int(vf.any(dim=-1).sum().item())
            kp_coverage = (
                f"kps: {n_model_eligible}/{n_any_valid}/{n_total_kp} "
                f"(>=D-valid / any-valid / total)"
            )

            anchor = current_kp[:, None, :3]
            gt_abs = traj_future * delta_t + anchor
            fit_abs = X_fit_scaled * delta_t + anchor
            history_abs = traj_history.clone()
            history_abs[..., :3] = history_abs[..., :3] + current_kp[:, None, :3]

            img_u8 = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # GT (green) vs N @ P_gt (per-kp hues) — divergence on sharp turns is
            # immediately visible. History stays cyan.
            # Render every real kp; per-step validity is handled by `valid_future`.
            # A kp with valid_future=[T,T,T,F,F,..] gets its first 3 steps drawn
            # and the rest skipped — visually showing where supervision drops out.
            kpm = np.ones(n_total_kp, dtype=bool)

            # Render the GT-vs-fit overlay once, save a copy as the "no_cp"
            # version for clean trace comparison, then draw control polygons
            # on top and save as the "with_cp" version. Two files per sample
            # so users can flip between them — ctrl-pt clutter helps debug
            # the polygon but hides the underlying fit quality.
            overlay = render_trace_overlay(
                image=img_u8,
                traj_history=history_abs.numpy(),
                traj_future_gt=gt_abs.numpy(),
                traj_future_pred=fit_abs.numpy(),
                valid_history=valid_history.numpy(),
                valid_future=valid_future.numpy(),
                key_pad_mask=kpm,
                current_kp=current_kp.numpy(),
                prepend_current=True,
                draw_last_uv_marker=True,
                connect_history_to_current=True,
                pred_color_mode="per_kp",
            )

            no_cp_path = out_subdir / f"sample_{vis_i:03d}_idx{int(idx):06d}_no_cp.png"
            Image.fromarray(overlay).save(no_cp_path)

            # B-spline control points in absolute uv space — same affine
            # `delta_t * P + current_kp` we apply to gt_abs/fit_abs above.
            # Drawn in the same per-kp palette `render_trace_overlay` uses for
            # `pred_color_mode="per_kp"`, so each kp's control polygon is
            # visually tied to its predicted polyline.
            overlay_cp = overlay.copy()
            ctrl_abs = (P_gt * delta_t + current_kp[:, None, :3]).numpy()           # (n, D, 3)
            H_img, W_img = overlay_cp.shape[:2]
            ctrl_xy = (ctrl_abs[..., :2] + 1.0) / 2.0 * np.asarray(
                [W_img, H_img], dtype=np.float32
            )                                                                       # (n, D, 2)
            ctrl_radius = max(3, min(W_img, H_img) // 100)
            palette = _kp_color_palette(n_total_kp)
            vk = valid_kp.bool().numpy()
            for ki in range(n_total_kp):
                # Skip kps the lstsq dropped (P_gt zeroed → all ctrl pts pile
                # up at current_kp, uninformative).
                if not vk[ki]:
                    continue
                color = palette[ki]
                pts = ctrl_xy[ki]                                                   # (D, 2)
                # Control polygon: thin solid line connecting consecutive ctrl pts.
                for j in range(pts.shape[0] - 1):
                    p0 = tuple(int(round(float(v))) for v in pts[j])
                    p1 = tuple(int(round(float(v))) for v in pts[j + 1])
                    cv2.line(overlay_cp, p0, p1, color, 1, lineType=cv2.LINE_AA)
                # Filled circle + thin black outline at each ctrl pt for contrast.
                for j in range(pts.shape[0]):
                    cx = int(round(float(pts[j, 0])))
                    cy = int(round(float(pts[j, 1])))
                    cv2.circle(overlay_cp, (cx, cy), ctrl_radius, color,
                               thickness=-1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay_cp, (cx, cy), ctrl_radius, (0, 0, 0),
                               thickness=1, lineType=cv2.LINE_AA)

            cp_path = out_subdir / f"sample_{vis_i:03d}_idx{int(idx):06d}_with_cp.png"
            Image.fromarray(overlay_cp).save(cp_path)
            logging.info(
                f"  [{vis_i + 1}/{n_pick}] {cp_path.name} (+ {no_cp_path.name}) "
                f"residual mean={residual_mean_per_sample[-1]:.4f} "
                f"max={residual_max_per_sample[-1]:.4f} | {kp_coverage}"
            )

    if any(not np.isnan(x) for x in residual_max_per_sample):
        residuals_max = np.array([x for x in residual_max_per_sample if not np.isnan(x)])
        residuals_mean = np.array([x for x in residual_mean_per_sample if not np.isnan(x)])
        logging.info(
            f"\n=== Residual summary over {len(residuals_max)} kps-with-valid-futures-aggregated samples "
            f"(D={cfg.n_ctrl}, scaled-delta units, ~roughly normalized to [-1,1]) ===\n"
            f"  mean residual:  median={np.median(residuals_mean):.4f}  "
            f"p90={np.percentile(residuals_mean, 90):.4f}\n"
            f"  max residual:   median={np.median(residuals_max):.4f}  "
            f"p90={np.percentile(residuals_max, 90):.4f}\n"
            f"  (smaller is better; for context: 1.0 in scaled-delta ≈ the percentile "
            f"used to compute delta_scale in the stats file)"
        )
    logging.info(f"Done. Overlays in {cfg.output_dir.resolve()}")


def main() -> None:
    visualize()


if __name__ == "__main__":
    main()
