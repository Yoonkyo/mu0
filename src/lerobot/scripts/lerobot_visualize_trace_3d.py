#!/usr/bin/env python
"""Viser-based 3D viewer for `.npz` files saved by
`lerobot_predict_trace_mu0_image_only.py --save_3d`.

Usage (single sample):
    python src/lerobot/scripts/lerobot_visualize_trace_3d.py \
        --npz_path outputs/predict/.../scene3d/sample_000000_<vid>_f000123.npz \
        --port 8080

Usage (whole directory of samples — adds a "Sample" picker):
    python src/lerobot/scripts/lerobot_visualize_trace_3d.py \
        --scene_dir outputs/predict/.../scene3d \
        --port 8080

Usage (MP4 with a slight orbital camera, captured from viser):
    # 1) ssh -L 8080:localhost:12345 user@host
    # 2) on the host, run:
    python src/lerobot/scripts/lerobot_visualize_trace_3d.py \
        --scene_dir outputs/predict/.../scene3d \
        --save_video outputs/predict/.../orbit.mp4 \
        --port 12345
    # 3) open http://localhost:8080 in your local browser; the script will start
    #    rendering as soon as the client connects, then exit when done.
    # The orbit is fixed to DEFAULT_ORBIT_FIT (see below); use
    # --video_orbit_scale to widen/narrow the sweep.

Coloring: traces are colored by timestep with the dynamic rainbow ramp from
`visualize_trace.py` `trace_color_by="age"` (purple at the current kp → red at
each track's own last valid step) — matching the 2D rollout_img stills. Pass
`--trace_color_by uv` for the legacy per-track spatial-hue look.

MP4 progressive reveal: by default the orbit video starts showing only the
first future step of every trace and grows the visible horizon linearly until
the full trace is on screen at the last frame, so the propagation of the
predicted motion plays out over the video. `--no-video_reveal` restores the
static full-trace orbit.

MP4 output layout: `--save_video out.mp4` concatenates every sample's orbit
into one file; `--save_video_dir out_dir/` writes one `<npz_stem>.mp4` per
sample instead (stems match the rollout_img overlays and `--save_frames_dir`).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def _lazy_viser():
    try:
        import viser
        import viser.transforms as tf
    except ImportError as e:
        raise ImportError(
            "viser is required for 3D visualization. Install with `pip install viser`."
        ) from e
    return viser, tf


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    data.close()
    return out


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorized HSV→RGB. Both in `[0, 1]^3`. Shape preserved."""
    h = hsv[..., 0] * 6.0
    s = hsv[..., 1]
    v = hsv[..., 2]
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    rgb = np.zeros_like(hsv, dtype=np.float32)
    components = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    for k, (r_, g_, b_) in enumerate(components):
        m = i == k
        rgb[..., 0] = np.where(m, r_, rgb[..., 0])
        rgb[..., 1] = np.where(m, g_, rgb[..., 1])
        rgb[..., 2] = np.where(m, b_, rgb[..., 2])
    return rgb


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB→HSV. Both in `[0, 1]^3`. Shape preserved."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    diff = cmax - cmin
    eps = 1e-9
    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (diff > eps)
    mask_g = (cmax == g) & (diff > eps)
    mask_b = (cmax == b) & (diff > eps)
    h = np.where(mask_r, ((g - b) / (diff + eps)) % 6, h)
    h = np.where(mask_g, ((b - r) / (diff + eps)) + 2, h)
    h = np.where(mask_b, ((r - g) / (diff + eps)) + 4, h)
    h = (h / 6.0) % 1.0
    s = np.where(cmax > eps, diff / (cmax + eps), 0.0)
    return np.stack([h, s, cmax], axis=-1).astype(np.float32)


def _colors_from_uv(uv_norm: np.ndarray) -> np.ndarray:
    """`(N, 2)` in `[-1, 1]` (image-plane normalized uv) → `(N, 3)` RGB in `[0, 1]`.

    2D color wheel: hue from `atan2(v, u)` so spatially-near tracks get nearby
    hues. The phase offset places red near the top-left so the layout matches
    the user's reference image (top-left red, top-right green, bottom-right
    cyan, bottom-left blue/violet). Saturation is fixed; the time gradient
    handled by `_fade_colors` modulates it per segment.
    """
    if uv_norm.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    u = uv_norm[:, 0].astype(np.float32)
    v = uv_norm[:, 1].astype(np.float32)
    theta = np.arctan2(v, u)            # [-π, π]
    offset = 3.0 * np.pi / 4.0          # rotate so (-1, -1) → red
    hue = ((theta + offset) / (2.0 * np.pi)) % 1.0
    sat = np.full_like(u, 0.85)
    val = np.full_like(u, 0.95)
    hsv = np.stack([hue, sat, val], axis=-1)
    return _hsv_to_rgb(hsv)


def _colors_from_age(frac: np.ndarray) -> np.ndarray:
    """Temporal position `frac` in `[0, 1]` → `(..., 3)` RGB rainbow ramp.

    Vectorized port of `visualize_trace._color_from_age`: `hue = 0.75·(1 −
    frac)` sweeps violet/purple (`frac=0`, the ramp anchor at the current kp)
    → blue → green → yellow → red (`frac=1`, the newest step), with HSV
    s=0.9, v=0.95. Out-of-range fracs clamp, so anything *before* the anchor
    (e.g. history) clamps to purple.
    """
    f = np.clip(np.asarray(frac, dtype=np.float32), 0.0, 1.0)
    hue = 0.75 * (1.0 - f)
    hsv = np.stack([hue, np.full_like(f, 0.9), np.full_like(f, 0.95)], axis=-1)
    return _hsv_to_rgb(hsv)


def _segments_with_meta(
    points: np.ndarray,           # (L, 3)
    valid: np.ndarray,            # (L,) bool
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """`(L, 3)` polyline → kept `(k, 2, 3)` segments plus timeline metadata.

    A segment connects `points[j] → points[j+1]` and is kept when both ends
    are finite and valid. Returns `(segments, end_idx, first_valid,
    last_valid)` where `end_idx[s]` is the endpoint index `j+1` of kept
    segment `s` (its position along the timeline — the age colorizer and the
    video reveal mask key off this) and `first_valid`/`last_valid` are the
    first/last finite+valid point indices (the per-track "own valid horizon"
    the dynamic age ramp normalizes by).
    """
    if points.shape[0] < 2:
        return np.zeros((0, 2, 3), np.float32), np.zeros((0,), np.int64), 0, 0
    finite = np.isfinite(points).all(axis=-1) & np.asarray(valid, dtype=bool)
    keep = finite[:-1] & finite[1:]
    if not keep.any():
        return np.zeros((0, 2, 3), np.float32), np.zeros((0,), np.int64), 0, 0
    segs = np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)
    end_idx = np.nonzero(keep)[0].astype(np.int64) + 1
    valid_idx = np.nonzero(finite)[0]
    return segs[keep], end_idx, int(valid_idx[0]), int(valid_idx[-1])


def _fade_colors(
    end_idx: np.ndarray,          # (k,) endpoint indices of kept segments
    n_seg: int,                   # total segments in the full polyline
    base_rgb: np.ndarray,         # (3,) float in [0, 1]
    weight_start: float,          # blend weight at t_norm=0 (oldest end of polyline)
    weight_end: float,            # blend weight at t_norm=1 (newest end of polyline)
    neutral_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Per-segment colors: blend `base_rgb` toward `neutral_rgb` along time.

    The base hue stays fixed (preserving per-track color from
    `_colors_from_uv`); each segment's color is a linear blend toward
    `neutral_rgb` (white by default — gray is a useful alternative against
    light backgrounds):

        seg_rgb = w * base_rgb + (1 - w) * neutral_rgb

    where `w` slides from `weight_start` (oldest) to `weight_end` (newest)
    along the polyline. We do this in RGB space rather than via HSV
    saturation because viser's line_segments doesn't honor opacity, so the
    only way to get a fade-to-faint look is to mix the actual RGB values.
    """
    t_norm = (end_idx.astype(np.float32) - 0.5) / float(max(n_seg, 1))
    w = (weight_start + (weight_end - weight_start) * t_norm)[:, None]
    base = np.asarray(base_rgb, dtype=np.float32).reshape(1, 3)
    neutral = np.asarray(neutral_rgb, dtype=np.float32).reshape(1, 3)
    return np.clip(w * base + (1.0 - w) * neutral, 0.0, 1.0).astype(np.float32)


def _age_ramp_colors(end_idx: np.ndarray, first_valid: int, last_valid: int) -> np.ndarray:
    """Dynamic age ramp: each segment colored by its position within the
    track's OWN valid horizon (`first_valid → last_valid`), so every trace
    spans the full purple→red ramp and ends red at its last valid step —
    mirroring `visualize_trace` with `trace_color_by="age"`.
    """
    span = float(max(1, last_valid - first_valid))
    return _colors_from_age((end_idx.astype(np.float32) - float(first_valid)) / span)


def _track_colors(n: int, cmap_name: str = "turbo") -> np.ndarray:
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    ts = np.array([0.5]) if n == 1 else np.linspace(0.0, 1.0, n)
    try:
        from matplotlib import colormaps
        return np.array([colormaps[cmap_name](t)[:3] for t in ts], dtype=np.float32)
    except ImportError:
        # Minimal HSV→RGB fallback so this script doesn't require matplotlib
        # at video-render time (which runs in the visualization env, but the
        # checkpoint env may lack matplotlib). Cycles hue around the wheel
        # for a turbo-ish look.
        h = ts.astype(np.float32)
        s = np.full_like(h, 0.85)
        v = np.full_like(h, 0.95)
        i = np.floor(h * 6).astype(np.int32) % 6
        f = h * 6 - np.floor(h * 6)
        p = v * (1 - s)
        q = v * (1 - f * s)
        t_ = v * (1 - (1 - f) * s)
        rgb = np.zeros((n, 3), dtype=np.float32)
        for k in range(n):
            ik = i[k]
            if   ik == 0: rgb[k] = (v[k], t_[k], p[k])
            elif ik == 1: rgb[k] = (q[k], v[k], p[k])
            elif ik == 2: rgb[k] = (p[k], v[k], t_[k])
            elif ik == 3: rgb[k] = (p[k], q[k], v[k])
            elif ik == 4: rgb[k] = (t_[k], p[k], v[k])
            else:         rgb[k] = (v[k], p[k], q[k])
        return rgb


def _polyline_segments(points: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """`(L, 3)` -> `(L-1, 2, 3)` consecutive-pair segments. Drops invalid steps.

    `valid` shape `(L,)` bool. If a step is invalid the segments touching it
    are dropped (so the polyline breaks instead of jumping through padding).
    """
    if points.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=points.dtype)
    if valid is None:
        valid = np.ones(points.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    finite = np.isfinite(points).all(axis=-1) & valid
    keep_seg = finite[:-1] & finite[1:]
    if not keep_seg.any():
        return np.zeros((0, 2, 3), dtype=points.dtype)
    starts = points[:-1][keep_seg]
    ends = points[1:][keep_seg]
    return np.stack([starts, ends], axis=1)


def _add_axes_handle(server, name: str = "world_axes", scale: float = 0.2):
    return server.scene.add_line_segments(
        name=name,
        points=np.array(
            [
                [[0.0, 0.0, 0.0], [scale, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, scale, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, scale]],
            ],
            dtype=np.float32,
        ),
        colors=np.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        line_width=3.0,
    )


def _build_scene(server, tf_mod, scene: dict[str, Any], state: dict[str, Any]) -> None:
    """Create / replace all scene handles for a single sample.

    Existing handles are removed first so this can be called repeatedly when
    the user picks a different sample from the GUI.
    """
    for handle in state.get("handles", []):
        try:
            handle.remove()
        except Exception:
            pass
    state["handles"] = []
    handles: list = state["handles"]

    image = scene["image"]
    K = scene["K"].astype(np.float32)
    c2w = scene["c2w"].astype(np.float32)
    H = int(scene["H"])
    W = int(scene["W"])

    # Pick the point cloud: full-video aggregate if present and enabled,
    # otherwise the single-frame unprojection.
    use_full = state["gui_use_full"].value and ("pcd_xyz_full" in scene)
    if use_full:
        pcd_xyz = scene["pcd_xyz_full"]
        pcd_rgb = scene["pcd_rgb_full"]
    else:
        pcd_xyz = scene["pcd_xyz"]
        pcd_rgb = scene["pcd_rgb"]

    pc_handle = server.scene.add_point_cloud(
        name="point_cloud",
        points=pcd_xyz.astype(np.float32),
        colors=pcd_rgb.astype(np.uint8),
        point_size=state["gui_point_size"].value,
        point_shape="rounded",
    )
    pc_handle.visible = state["gui_show_pcd"].value
    state["pc_handle"] = pc_handle
    handles.append(pc_handle)

    # World-axis triad — small helper so the scene origin is locatable.
    axes = _add_axes_handle(server)
    axes.visible = state["gui_show_axes"].value
    state["axes_handle"] = axes
    handles.append(axes)

    # Per-track polylines. We render history (t-h..t-1), gt (t..t+F), and
    # pred (t..t+F) as separate line-segment groups so toggles are independent.
    n_kp = scene["traj_pred_world"].shape[0]
    pred_world = scene["traj_pred_world"].astype(np.float32)  # (N, F, 3)
    gt_world = scene["traj_gt_world"].astype(np.float32)
    hist_world = scene["traj_history_world"].astype(np.float32)
    cur_world = scene["current_kp_world"].astype(np.float32)  # (N, 3)
    valid_pred = scene["valid_pred"].astype(bool)
    valid_history = scene["valid_history"].astype(bool)
    # Per-track base color: hue from current_kp uv (2D color wheel) so spatial
    # neighbors get nearby hues. Falls back to the legacy turbo-by-index for
    # NPZs saved before the current_kp_uv field existed.
    if "current_kp_uv" in scene and scene["current_kp_uv"].size > 0:
        colors = _colors_from_uv(scene["current_kp_uv"][:n_kp])
    else:
        colors = _track_colors(n_kp, cmap_name=state["gui_cmap"].value)

    trace_color_by = str(state["gui_trace_color"].value)
    purple = _colors_from_age(np.zeros(1))[0]  # age-mode anchor color (frac=0)
    mid_gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    pred_segs: list[np.ndarray] = []
    pred_seg_colors: list[np.ndarray] = []
    pred_seg_steps: list[np.ndarray] = []
    gt_segs: list[np.ndarray] = []
    gt_seg_colors: list[np.ndarray] = []
    gt_seg_steps: list[np.ndarray] = []
    hist_segs: list[np.ndarray] = []
    hist_seg_colors: list[np.ndarray] = []
    for i in range(n_kp):
        col = colors[i]
        # Prepend current_kp so the prediction polyline starts at the actual
        # observation (not the first prediction step). For pred we keep all
        # F valid; the GT validity drives whether we draw a segment. On these
        # timelines the segment endpoint index j == future step j (index 0 is
        # the current kp) — that's what the reveal masks against.
        pred_pts = np.concatenate([cur_world[i : i + 1], pred_world[i]], axis=0)
        gt_pts = np.concatenate([cur_world[i : i + 1], gt_world[i]], axis=0)
        pred_valid = np.concatenate([[True], np.ones_like(valid_pred[i])], axis=0)
        gt_valid = np.concatenate([[True], valid_pred[i]], axis=0)
        # History: oldest -> ... -> t-1, then connect t-1 -> current.
        hist_pts = np.concatenate([hist_world[i], cur_world[i : i + 1]], axis=0)
        hist_valid = np.concatenate([valid_history[i], [True]], axis=0)

        pred_s, pred_e, pred_v0, pred_v1 = _segments_with_meta(pred_pts, pred_valid)
        gt_s, gt_e, gt_v0, gt_v1 = _segments_with_meta(gt_pts, gt_valid)
        hist_s, hist_e, _, _ = _segments_with_meta(hist_pts, hist_valid)

        if trace_color_by == "age":
            # Timestep coloring with the dynamic ramp (visualize_trace
            # trace_color_by="age"): purple at the current kp → red at the
            # track's own last valid step, exactly like the 2D rollout_img
            # stills. GT shares the ramp but is blended toward mid-gray —
            # hue now encodes time, so muting is what keeps overlapping
            # GT/pred layers distinguishable. History predates the ramp
            # anchor → constant purple, with the legacy oldest→newest white
            # fade so it still reads as a tail.
            pred_c = _age_ramp_colors(pred_e, pred_v0, pred_v1)
            gt_c = 0.5 * _age_ramp_colors(gt_e, gt_v0, gt_v1) + 0.5 * mid_gray
            hist_c = _fade_colors(
                hist_e, hist_pts.shape[0] - 1, purple,
                weight_start=0.15, weight_end=0.55,
            )
        else:
            # Legacy "uv" look. Time-blend gradient (fade by blending toward
            # white/gray, since viser line_segments don't honor opacity): mix
            # each segment toward a neutral color with a weight that slides
            # from oldest → newest. Hue stays per-track. Pred and GT share
            # the same per-track hue, but GT mixes toward MID-GRAY while pred
            # mixes toward WHITE — that gives GT a visibly muted appearance
            # even at the same blend weight, so the two layers don't overlap.
            pred_c = _fade_colors(
                pred_e, pred_pts.shape[0] - 1, col,
                weight_start=0.55, weight_end=1.00,
            )
            gt_c = _fade_colors(
                gt_e, gt_pts.shape[0] - 1, col,
                weight_start=0.55, weight_end=1.00,
                neutral_rgb=(0.5, 0.5, 0.5),
            )
            hist_c = _fade_colors(
                hist_e, hist_pts.shape[0] - 1, col,
                weight_start=0.15, weight_end=0.55,
            )

        pred_segs.append(pred_s); pred_seg_colors.append(pred_c); pred_seg_steps.append(pred_e)
        gt_segs.append(gt_s); gt_seg_colors.append(gt_c); gt_seg_steps.append(gt_e)
        hist_segs.append(hist_s); hist_seg_colors.append(hist_c)

    def _stack(seg_lists, color_lists, step_lists=None):
        keep = [k for k, s in enumerate(seg_lists) if s.shape[0] > 0]
        if not keep:
            return None, None, None
        segs = np.concatenate([seg_lists[k] for k in keep], axis=0)
        cols = np.concatenate([color_lists[k] for k in keep], axis=0)
        steps = (
            np.concatenate([step_lists[k] for k in keep], axis=0)
            if step_lists is not None
            else None
        )
        return segs, cols, steps

    pred_pts_arr, pred_cols_arr, pred_steps_arr = _stack(pred_segs, pred_seg_colors, pred_seg_steps)
    gt_pts_arr, gt_cols_arr, gt_steps_arr = _stack(gt_segs, gt_seg_colors, gt_seg_steps)
    hist_pts_arr, hist_cols_arr, _ = _stack(hist_segs, hist_seg_colors)

    # viser's add_line_segments wants per-vertex colors with shape (N, 2, 3),
    # not per-segment (N, 3). Repeat each segment color for both endpoints.
    # uint8 up front so the stashed arrays can be assigned straight back to
    # `handle.colors` during the progressive-reveal updates.
    def _per_vertex(c):
        u8 = (np.clip(c, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.repeat(u8[:, None, :], 2, axis=1)

    if pred_pts_arr is not None:
        pred_vert_cols = _per_vertex(pred_cols_arr)
        pred_handle = server.scene.add_line_segments(
            name="traj_pred",
            points=pred_pts_arr,
            colors=pred_vert_cols,
            line_width=state["gui_track_width"].value,
        )
        pred_handle.visible = state["gui_show_pred"].value
        state["pred_handle"] = pred_handle
        # Full (points, per-vertex colors, future-step index per segment) so
        # the headless reveal can re-mask the handle props frame by frame.
        state["pred_seg_data"] = (pred_pts_arr, pred_vert_cols, pred_steps_arr)
        handles.append(pred_handle)
    else:
        state["pred_handle"] = None
        state["pred_seg_data"] = None

    if gt_pts_arr is not None:
        gt_vert_cols = _per_vertex(gt_cols_arr)
        gt_handle = server.scene.add_line_segments(
            name="traj_gt",
            points=gt_pts_arr,
            colors=gt_vert_cols,
            line_width=state["gui_track_width"].value,
        )
        gt_handle.visible = state["gui_show_gt"].value
        state["gt_handle"] = gt_handle
        state["gt_seg_data"] = (gt_pts_arr, gt_vert_cols, gt_steps_arr)
        handles.append(gt_handle)
    else:
        state["gt_handle"] = None
        state["gt_seg_data"] = None

    if hist_pts_arr is not None:
        hist_handle = server.scene.add_line_segments(
            name="traj_history",
            points=hist_pts_arr,
            colors=_per_vertex(hist_cols_arr),
            line_width=state["gui_track_width"].value,
        )
        hist_handle.visible = state["gui_show_hist"].value
        state["hist_handle"] = hist_handle
        handles.append(hist_handle)
    else:
        state["hist_handle"] = None

    # Current keypoints — small spheres at each kp so it's easy to see
    # where the prediction starts. In age mode every dot takes the ramp's
    # frac=0 purple (the 2D still's start-disk color); in uv mode the
    # per-track spatial hue.
    if n_kp > 0:
        kp_rgb = np.tile(purple, (n_kp, 1)) if trace_color_by == "age" else colors
        kp_handle = server.scene.add_point_cloud(
            name="current_keypoints",
            points=cur_world.astype(np.float32),
            colors=(kp_rgb * 255).astype(np.uint8),
            point_size=state["gui_kp_size"].value,
            point_shape="circle",
        )
        kp_handle.visible = state["gui_show_kp"].value
        state["kp_handle"] = kp_handle
        handles.append(kp_handle)
    else:
        state["kp_handle"] = None

    # Camera frustum at frame t — uses the saved RGB for the image overlay.
    fov = 2.0 * np.arctan2(H / 2.0, K[0, 0])
    aspect = W / max(H, 1)
    frustum = server.scene.add_camera_frustum(
        name="camera_frustum",
        fov=fov,
        aspect=aspect,
        scale=state["gui_frustum_scale"].value,
        image=image.astype(np.float32) / 255.0,
        wxyz=tf_mod.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
    )
    frustum.visible = state["gui_show_frustum"].value
    state["frustum_handle"] = frustum
    handles.append(frustum)

    # When use_full_video is on we also have per-frame intrinsics/extrinsics —
    # render the whole camera path as small frustums (no per-frame image).
    extra_frustums: list = []
    if use_full and "frustum_K" in scene and "frustum_c2w" in scene:
        Ks = scene["frustum_K"].astype(np.float32)
        c2ws = scene["frustum_c2w"].astype(np.float32)
        for i in range(Ks.shape[0]):
            K_i = Ks[i]
            c2w_i = c2ws[i]
            fov_i = 2.0 * np.arctan2(H / 2.0, K_i[0, 0])
            f = server.scene.add_camera_frustum(
                name=f"path_frustum_{i:03d}",
                fov=fov_i,
                aspect=aspect,
                scale=state["gui_frustum_scale"].value * 0.4,
                color=(180, 180, 180),
                wxyz=tf_mod.SO3.from_matrix(c2w_i[:3, :3]).wxyz,
                position=c2w_i[:3, 3],
            )
            f.visible = state["gui_show_path"].value
            extra_frustums.append(f)
            handles.append(f)
    state["path_frustums"] = extra_frustums


def _wire_callbacks(server, state, refresh_fn) -> None:
    """Hook GUI callbacks. `refresh_fn` rebuilds the scene from current sample."""

    @state["gui_point_size"].on_update
    def _(_) -> None:
        if state.get("pc_handle") is not None:
            state["pc_handle"].point_size = state["gui_point_size"].value

    @state["gui_track_width"].on_update
    def _(_) -> None:
        for k in ("pred_handle", "gt_handle", "hist_handle"):
            h = state.get(k)
            if h is not None:
                h.line_width = state["gui_track_width"].value

    @state["gui_kp_size"].on_update
    def _(_) -> None:
        if state.get("kp_handle") is not None:
            state["kp_handle"].point_size = state["gui_kp_size"].value

    @state["gui_frustum_scale"].on_update
    def _(_) -> None:
        if state.get("frustum_handle") is not None:
            state["frustum_handle"].scale = state["gui_frustum_scale"].value
        for f in state.get("path_frustums", []):
            f.scale = state["gui_frustum_scale"].value * 0.4

    @state["gui_show_pcd"].on_update
    def _(_) -> None:
        if state.get("pc_handle") is not None:
            state["pc_handle"].visible = state["gui_show_pcd"].value

    @state["gui_show_pred"].on_update
    def _(_) -> None:
        if state.get("pred_handle") is not None:
            state["pred_handle"].visible = state["gui_show_pred"].value

    @state["gui_show_gt"].on_update
    def _(_) -> None:
        if state.get("gt_handle") is not None:
            state["gt_handle"].visible = state["gui_show_gt"].value

    @state["gui_show_hist"].on_update
    def _(_) -> None:
        if state.get("hist_handle") is not None:
            state["hist_handle"].visible = state["gui_show_hist"].value

    @state["gui_show_kp"].on_update
    def _(_) -> None:
        if state.get("kp_handle") is not None:
            state["kp_handle"].visible = state["gui_show_kp"].value

    @state["gui_show_frustum"].on_update
    def _(_) -> None:
        if state.get("frustum_handle") is not None:
            state["frustum_handle"].visible = state["gui_show_frustum"].value

    @state["gui_show_axes"].on_update
    def _(_) -> None:
        if state.get("axes_handle") is not None:
            state["axes_handle"].visible = state["gui_show_axes"].value

    @state["gui_show_path"].on_update
    def _(_) -> None:
        for f in state.get("path_frustums", []):
            f.visible = state["gui_show_path"].value

    # Toggling use_full_video / sample picker / cmap / trace coloring means we
    # have to rebuild the per-track and point-cloud handles (the underlying
    # data swaps).
    @state["gui_use_full"].on_update
    def _(_) -> None:
        refresh_fn()

    @state["gui_cmap"].on_update
    def _(_) -> None:
        refresh_fn()

    @state["gui_trace_color"].on_update
    def _(_) -> None:
        refresh_fn()

    if state.get("gui_sample") is not None:
        @state["gui_sample"].on_update
        def _(_) -> None:
            refresh_fn()


def visualize(
    npz_paths: list[Path], port: int, share: bool = False, trace_color_by: str = "age"
) -> None:
    if not npz_paths:
        raise ValueError("no .npz files supplied")

    viser, tf_mod = _lazy_viser()

    server = viser.ViserServer(port=port)
    if share:
        server.request_share_url()
    server.scene.set_up_direction("-y")

    sample_names = [p.name for p in npz_paths]
    state: dict[str, Any] = {"handles": [], "scene": None}

    with server.gui.add_folder("Sample"):
        if len(npz_paths) > 1:
            state["gui_sample"] = server.gui.add_dropdown(
                "Sample", options=sample_names, initial_value=sample_names[0]
            )
        else:
            state["gui_sample"] = None
        state["gui_use_full"] = server.gui.add_checkbox(
            "Use full-video pcd", initial_value=False
        )
        state["gui_cmap"] = server.gui.add_dropdown(
            "Track colormap", options=("turbo", "viridis", "plasma", "rainbow"),
            initial_value="turbo",
        )
        state["gui_trace_color"] = server.gui.add_dropdown(
            "Trace color by", options=("age", "uv"),
            initial_value=trace_color_by if trace_color_by in ("age", "uv") else "age",
        )

    with server.gui.add_folder("Trajectories"):
        state["gui_show_pred"] = server.gui.add_checkbox("Show pred", initial_value=True)
        state["gui_show_gt"] = server.gui.add_checkbox("Show GT", initial_value=True)
        state["gui_show_hist"] = server.gui.add_checkbox("Show history", initial_value=True)
        state["gui_show_kp"] = server.gui.add_checkbox("Show current kps", initial_value=True)
        state["gui_track_width"] = server.gui.add_slider(
            "Track width", min=0.5, max=8.0, step=0.5, initial_value=3.0
        )
        state["gui_kp_size"] = server.gui.add_slider(
            "KP size", min=0.005, max=0.05, step=0.005, initial_value=0.015
        )

    with server.gui.add_folder("Scene"):
        state["gui_show_pcd"] = server.gui.add_checkbox("Show point cloud", initial_value=True)
        state["gui_show_frustum"] = server.gui.add_checkbox("Show frustum", initial_value=True)
        state["gui_show_axes"] = server.gui.add_checkbox("Show world axes", initial_value=True)
        state["gui_show_path"] = server.gui.add_checkbox(
            "Show camera path (full-video only)", initial_value=False
        )
        state["gui_point_size"] = server.gui.add_slider(
            "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.006
        )
        state["gui_frustum_scale"] = server.gui.add_slider(
            "Frustum scale", min=0.01, max=0.5, step=0.01, initial_value=0.1
        )

    # ------- Recording panel: drive the camera yourself, save MP4 -----------
    # The polling thread captures (position, wxyz, fov) at `rec_fps` Hz while
    # `recording` is set. Save MP4 then replays each captured pose through
    # client.get_render so the rendered output exactly matches the camera
    # work the user just performed. Pose JSON load/save is provided so the
    # same path can be replayed across sessions or shared.
    camera_path: list[dict[str, Any]] = []
    recording = threading.Event()

    with server.gui.add_folder("Recording"):
        rec_status = server.gui.add_text(
            "Status", initial_value="idle (0 frames)", disabled=True
        )
        rec_output = server.gui.add_text(
            "MP4 path", initial_value=str(Path("recording.mp4").resolve())
        )
        rec_path_json = server.gui.add_text(
            "Camera path JSON", initial_value=str(Path("camera_path.json").resolve())
        )
        rec_width = server.gui.add_slider(
            "Width", min=320, max=1920, step=16, initial_value=1280
        )
        rec_height = server.gui.add_slider(
            "Height", min=240, max=1080, step=16, initial_value=720
        )
        rec_fps = server.gui.add_slider(
            "FPS / poll Hz", min=10, max=60, step=5, initial_value=30
        )
        rec_start = server.gui.add_button("Start recording")
        rec_stop = server.gui.add_button("Stop recording")
        rec_clear = server.gui.add_button("Clear recording")
        rec_save = server.gui.add_button("Save MP4 (uses current sample)")
        rec_save_json = server.gui.add_button("Save camera path JSON")
        rec_load_json = server.gui.add_button("Load camera path JSON")

    def _set_status(text: str) -> None:
        try:
            rec_status.value = text
        except Exception:
            pass
        logging.info(f"[recording] {text}")

    def _poll_loop() -> None:
        """Snapshot client.camera at rec_fps Hz until `recording` clears."""
        while recording.is_set():
            t0 = time.time()
            clients = server.get_clients()
            if not clients:
                _set_status("waiting for client...")
            else:
                client = next(iter(clients.values()))
                try:
                    cam = client.camera
                    camera_path.append({
                        "wxyz": np.asarray(cam.wxyz, dtype=np.float64).tolist(),
                        "position": np.asarray(cam.position, dtype=np.float64).tolist(),
                        "fov": float(cam.fov),
                    })
                    _set_status(f"recording... {len(camera_path)} frames")
                except Exception as e:
                    _set_status(f"poll failed: {type(e).__name__}: {e}")
            # Sleep the remainder of the tick so we hit `rec_fps` Hz on average.
            dt = 1.0 / float(max(rec_fps.value, 1))
            time.sleep(max(dt - (time.time() - t0), 0.0))

    @rec_start.on_click
    def _(_) -> None:
        if recording.is_set():
            _set_status("already recording")
            return
        recording.set()
        threading.Thread(target=_poll_loop, daemon=True).start()
        _set_status(f"recording at {int(rec_fps.value)} Hz...")

    @rec_stop.on_click
    def _(_) -> None:
        if not recording.is_set():
            return
        recording.clear()
        _set_status(f"stopped — {len(camera_path)} frames")

    @rec_clear.on_click
    def _(_) -> None:
        recording.clear()
        camera_path.clear()
        _set_status("cleared (0 frames)")

    @rec_save_json.on_click
    def _(_) -> None:
        if not camera_path:
            _set_status("no frames to save")
            return
        out = Path(rec_path_json.value)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"fps": int(rec_fps.value), "frames": camera_path}, f, indent=2)
        _set_status(f"wrote {len(camera_path)} poses → {out.name}")

    @rec_load_json.on_click
    def _(_) -> None:
        src = Path(rec_path_json.value)
        if not src.exists():
            _set_status(f"file not found: {src.name}")
            return
        with open(src) as f:
            d = json.load(f)
        camera_path.clear()
        camera_path.extend(d.get("frames", []))
        if "fps" in d:
            try:
                rec_fps.value = int(d["fps"])
            except Exception:
                pass
        _set_status(f"loaded {len(camera_path)} poses from {src.name}")

    def _do_save_mp4() -> None:
        """Render the captured path through client.get_render and write MP4."""
        import cv2
        if not camera_path:
            _set_status("no frames to save")
            return
        clients = server.get_clients()
        if not clients:
            _set_status("no client connected — open the browser tab first")
            return
        client = next(iter(clients.values()))
        out = Path(rec_output.value)
        out.parent.mkdir(parents=True, exist_ok=True)
        width = int(rec_width.value)
        height = int(rec_height.value)
        fps = int(rec_fps.value)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
        if not writer.isOpened():
            _set_status(f"VideoWriter failed for {out.name}")
            return
        n = len(camera_path)
        for i, pose in enumerate(camera_path):
            try:
                img = client.get_render(
                    height=height, width=width,
                    wxyz=tuple(pose["wxyz"]),
                    position=tuple(pose["position"]),
                    fov=float(pose["fov"]),
                    transport_format="jpeg",
                )
            except Exception as e:
                logging.warning(f"frame {i} get_render failed: {e}")
                continue
            if img.dtype != np.uint8:
                img = (img.clip(0, 1) * 255.0).astype(np.uint8)
            if img.shape[-1] == 4:
                img = img[..., :3]
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            if i % 10 == 0 or i == n - 1:
                _set_status(f"rendering... {i + 1}/{n}")
        writer.release()
        _set_status(f"saved → {out}")

    @rec_save.on_click
    def _(_) -> None:
        if recording.is_set():
            _set_status("stop recording first")
            return
        # Render off the GUI thread so the server stays responsive.
        threading.Thread(target=_do_save_mp4, daemon=True).start()

    name_to_path = {p.name: p for p in npz_paths}

    def refresh() -> None:
        if state["gui_sample"] is not None:
            chosen_name = state["gui_sample"].value
        else:
            chosen_name = sample_names[0]
        path = name_to_path[chosen_name]
        scene = _load_npz(path)
        state["scene"] = scene
        logging.info(f"Loaded {path}")
        _build_scene(server, tf_mod, scene, state)

    refresh()
    _wire_callbacks(server, state, refresh)

    logging.info(f"Viser server up at http://localhost:{port} (Ctrl+C to exit)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("Shutting down")


def _look_at_R(
    eye: np.ndarray, target: np.ndarray, up_world: np.ndarray
) -> np.ndarray:
    """3x3 rotation matrix in viser's camera convention (OpenCV: x right, y down, z forward).

    Viser's `client.get_render(wxyz, position, fov)` and
    `scene.add_camera_frustum(wxyz, position, ...)` both interpret wxyz as
    a c2w rotation where `R[:, 2]` is the camera's forward direction in
    world (verified against recorded poses where R[:, 2] equals
    (look_at − eye).normalized exactly).

    The camera looks at `target` from `eye`; `up_world` is the world's
    up direction (`(0, -1, 0)` for TraceExtract / `set_up_direction("-y")`).
    The image-down axis is the projection of `-up_world` onto the camera
    plane.
    """
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    # right = forward × up_world (right-handed, gives image-right in OpenCV)
    right = np.cross(forward, up_world)
    n = np.linalg.norm(right)
    if n < 1e-6:
        # Singular: forward parallel to up; fall back to a perpendicular axis.
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(right)
    right /= n + 1e-9
    down = np.cross(forward, right)  # image-down in OpenCV
    down /= np.linalg.norm(down) + 1e-9
    return np.stack([right, down, forward], axis=1).astype(np.float32)


# Hardcoded orbit fit baked from the user's vetted camera_path.json (post-trim
# 80%, fit with anchor=identity so the constants are stored in raw recording-
# world coords). At runtime, `_orbit_pose_from_fit` re-anchors them through
# each scene's c2w, so the orbit auto-rotates with the frustum:
#   look_at_world = scene_c2w @ look_at_local + scene_c2w_t
#   centroid_world = scene_c2w @ centroid_local + scene_c2w_t
#   axis_{u,v}_world = scene_c2w_R @ axis_{u,v}_local
DEFAULT_ORBIT_FIT: dict[str, Any] = {
    "look_at_local":  np.array([0.037598, -0.252789,  0.104124], dtype=np.float32),
    "centroid_local": np.array([0.006250, -0.452521, -0.321496], dtype=np.float32),
    "axis_u_local":   np.array([-0.964060, -0.185954,  0.189759], dtype=np.float32),
    "axis_v_local":   np.array([0.245591, -0.896193,  0.369491], dtype=np.float32),
    "semi_a":         0.177251,
    "semi_b":         0.119072,
    "sphere_radius":  0.484844,
    "direction":     -1.0,
    "fov":            1.308997,  # ~75°
    "n_recorded":     395,
    "n_after_skip":   316,
    "n_revs_recorded": 5.91,
}


def _orbit_pose_from_fit(
    fit: dict[str, Any], scene_c2w: np.ndarray, t_norm: float, n_revolutions: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize one orbit pose for a new scene from a `_fit_orbit_*` dict.

    Geometry: at angle θ, place the camera in the orbit plane at
        cam = centroid + a·cos(θ)·u + b·sin(θ)·v
    where (u, v) are the in-plane axes (rotated through this scene's c2w)
    and (a, b) are the recorded semi-axes. The camera always faces the
    look-at point (also re-rotated by this scene's c2w), so the POV
    direction matches what the user did in their recording.

    Returns (camera_position_world, look_at_world). Orientation is
    recomputed per-frame in the rendering loop via `_look_at_R_threejs`.
    """
    R = scene_c2w[:3, :3].astype(np.float32)
    t = scene_c2w[:3, 3].astype(np.float32)

    look_at_world = t + R @ fit["look_at_local"]
    centroid_world = t + R @ fit["centroid_local"]
    axis_u_world = R @ fit["axis_u_local"]
    axis_v_world = R @ fit["axis_v_local"]
    a = float(fit["semi_a"])
    b = float(fit["semi_b"])
    direction = float(fit["direction"])

    theta = direction * 2.0 * np.pi * float(n_revolutions) * float(t_norm)
    offset = a * np.cos(theta) * axis_u_world + b * np.sin(theta) * axis_v_world
    cam_pos = centroid_world + offset
    return cam_pos.astype(np.float32), look_at_world.astype(np.float32)


def _draw_caption(img: np.ndarray, text: str) -> None:
    """Stamp the task string in plain black, horizontally centered near the
    bottom of the frame (the viser background is light, so no backing box)."""
    import cv2

    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, _), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Shrink to fit if the task string is wider than the frame.
    max_w = img.shape[1] - 16
    if tw > max_w:
        scale *= max_w / float(tw)
        (tw, _), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max((img.shape[1] - tw) // 2, 4)
    y = img.shape[0] - 20 - baseline  # text baseline, ~20 px above the bottom edge
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


class _StaticGuiValue:
    """Stand-in for a viser GUI handle when there's no GUI to read from.

    `_build_scene` reads `state["gui_*"].value` for every visibility / size
    control. In headless video render we don't expose the controls, so each
    knob is replaced with this tiny shim carrying a fixed value.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


def _headless_render_state(
    *,
    use_full_video: bool,
    only_pred: bool,
    show_kp: bool,
    show_axes: bool,
    track_width: float,
    point_size: float,
    frustum_scale: float,
    trace_color_by: str = "age",
) -> dict[str, Any]:
    """Mock GUI knobs so `_build_scene` can run with no interactive controls.

    Shared by `render_video` (MP4) and `render_orbit_pngs` (subsampled PNGs) so
    both headless paths show the identical scene preset (pred-only by default,
    thicker tracks, small frustum).
    """
    return {
        "handles": [],
        "scene": None,
        "gui_sample": None,
        "gui_use_full": _StaticGuiValue(use_full_video),
        "gui_cmap": _StaticGuiValue("turbo"),  # unused if current_kp_uv is in NPZ
        "gui_trace_color": _StaticGuiValue(trace_color_by),
        "gui_show_pred": _StaticGuiValue(True),
        "gui_show_gt": _StaticGuiValue(not only_pred),
        "gui_show_hist": _StaticGuiValue(not only_pred),
        "gui_show_kp": _StaticGuiValue(show_kp),
        "gui_show_pcd": _StaticGuiValue(True),
        "gui_show_frustum": _StaticGuiValue(True),
        "gui_show_axes": _StaticGuiValue(show_axes),
        "gui_show_path": _StaticGuiValue(False),
        "gui_track_width": _StaticGuiValue(track_width),
        "gui_kp_size": _StaticGuiValue(0.015),
        "gui_point_size": _StaticGuiValue(point_size),
        "gui_frustum_scale": _StaticGuiValue(frustum_scale),
    }


def _wait_for_client(server, port: int):
    """Block until a browser connects, then return the (first) client handle."""
    print(f"Open http://localhost:{port}/ in a browser, then this script will start rendering.")
    print(f"  Over SSH: ssh -L 8080:localhost:{port} <host>; then visit http://localhost:8080")
    while not server.get_clients():
        time.sleep(0.5)
    client_id = next(iter(server.get_clients().keys()))
    client = server.get_clients()[client_id]
    print(f"Client {client_id} connected.")
    return client


def _scaled_orbit_fit(orbit_scale: float) -> dict[str, Any]:
    """Return DEFAULT_ORBIT_FIT, optionally with its in-plane semi-axes scaled.

    `orbit_scale<1` = subtler camera motion, `>1` = wider sweep. A copy is made
    so the module-level constant is never mutated.
    """
    fit = DEFAULT_ORBIT_FIT
    fov_rad = float(fit["fov"])
    print(
        f"  using DEFAULT_ORBIT_FIT (semi-axes a={fit['semi_a']:.3f}m, "
        f"b={fit['semi_b']:.3f}m, fov={np.degrees(fov_rad):.0f}°)"
    )
    if orbit_scale != 1.0:
        fit = dict(fit)
        fit["semi_a"] = float(fit["semi_a"]) * float(orbit_scale)
        fit["semi_b"] = float(fit["semi_b"]) * float(orbit_scale)
        print(
            f"  orbit scale={orbit_scale:.2f} → semi-axes "
            f"a={fit['semi_a']:.3f}m, b={fit['semi_b']:.3f}m"
        )
    return fit


def _render_scene_orbit_frames(
    client,
    server,
    tf_mod,
    scene: dict[str, Any],
    state: dict[str, Any],
    *,
    fit: dict[str, Any],
    fov_rad: float,
    n_frames_per_sample: int,
    width: int,
    height: int,
    up_world: np.ndarray,
    transport_format: str,
    path_name: str,
    reveal_future: bool = False,
) -> list[np.ndarray]:
    """Render one scene's full orbit (one revolution) and return BGR uint8 frames.

    Builds the 3D scene from `scene`, then sweeps the DEFAULT_ORBIT_FIT camera
    path through `n_frames_per_sample` unique poses, pulling each rasterized
    frame back via `client.get_render` (a browser round-trip). The task string
    is stamped bottom-center on every frame. Shared by the MP4 and PNG render
    paths so their output matches exactly.

    With `reveal_future=True` the visible future horizon grows over the orbit:
    the first frames draw only future step 1 of every trace and the cutoff
    sweeps linearly so the full horizon is on screen exactly at the last
    frame — the trace's propagation plays out across the video. Implemented by
    splitting the pred/GT traces into one line-segment handle per future step
    (from the per-segment step indices stashed by `_build_scene`) and toggling
    their `.visible` flags per frame — scalar prop updates the viser client
    applies reliably, unlike `points`/`colors` updates that change the buffer
    size. The point cloud is untouched, so updates are cheap.
    """
    import cv2

    task = str(scene.get("task", np.array(""))) if "task" in scene else ""
    frame_idx = int(scene.get("frame_idx", -1))
    video_dir = str(scene.get("video_dir", np.array(""))) if "video_dir" in scene else ""
    vid_name = Path(video_dir).name if video_dir else "?"
    print(f"  [{path_name}] video={vid_name} frame={frame_idx} task={task!r}")

    # Build the same 3D scene we'd show interactively (this populates
    # state["pred_handle"], etc., and adds them to the viser scene).
    _build_scene(server, tf_mod, scene, state)
    c2w_base = scene["c2w"].astype(np.float32)

    # Progressive-reveal setup. viser's fat-line buffers don't tolerate
    # per-frame SIZE changes of `points`/`colors` (the client only renders
    # again once the array size matches the original allocation), so instead
    # of re-masking one big handle we split each visible future-trace handle
    # into one handle PER FUTURE STEP and drive plain `.visible` toggles.
    reveal_groups: list[tuple[int, Any]] = []  # (future step, per-step handle)
    reveal_max_step = 0
    if reveal_future:
        for h_key, d_key, base_name in (
            ("pred_handle", "pred_seg_data", "traj_pred_reveal"),
            ("gt_handle", "gt_seg_data", "traj_gt_reveal"),
        ):
            h, d = state.get(h_key), state.get(d_key)
            if h is None or d is None or not h.visible or d[2].size == 0:
                continue
            seg_pts, seg_cols, seg_steps = d
            h.visible = False  # replaced by the per-step handles below
            for s in np.unique(seg_steps):
                m = seg_steps == s
                hs = server.scene.add_line_segments(
                    name=f"{base_name}/{int(s):03d}",
                    points=seg_pts[m],
                    colors=seg_cols[m],
                    line_width=state["gui_track_width"].value,
                )
                hs.visible = False
                state["handles"].append(hs)  # cleaned up on next _build_scene
                reveal_groups.append((int(s), hs))
            reveal_max_step = max(reveal_max_step, int(seg_steps.max()))
    prev_k = 0

    rendered: list[np.ndarray] = []
    for fi in tqdm(range(n_frames_per_sample), desc=f"  {path_name}", leave=False):
        if reveal_groups:
            # Unlike the orbit's loop-friendly t_norm = fi / N, the reveal
            # cutoff uses fi / (N - 1) so the FULL trace lands exactly on the
            # last frame: k sweeps 1 → max future step linearly. k only ever
            # grows, so switching on the (prev_k, k] band is enough.
            t_rev = fi / float(max(n_frames_per_sample - 1, 1))
            k = 1 + int(round(t_rev * (reveal_max_step - 1)))
            if k != prev_k:
                for s, hs in reveal_groups:
                    if prev_k < s <= k:
                        hs.visible = True
                prev_k = k
        # t_norm = fi / N (not i / (N-1)) so an MP4 loop wraps seamlessly.
        t_norm = fi / float(n_frames_per_sample)
        cam_pos, look_at = _orbit_pose_from_fit(fit, c2w_base, t_norm, n_revolutions=1.0)
        R = _look_at_R(cam_pos, look_at, up_world)
        wxyz = np.asarray(tf_mod.SO3.from_matrix(R).wxyz, dtype=np.float64)
        try:
            img = client.get_render(
                height=height,
                width=width,
                wxyz=tuple(wxyz.tolist()),
                position=tuple(cam_pos.tolist()),
                fov=fov_rad,
                transport_format=transport_format,  # "jpeg" → (H, W, 3) uint8
            )
        except Exception as e:
            print(f"    get_render failed at frame {fi}: {type(e).__name__}: {e}")
            continue
        if img.dtype != np.uint8:
            img = (img.clip(0, 1) * 255.0).astype(np.uint8)
        if img.shape[-1] == 4:
            img = img[..., :3]
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _draw_caption(bgr, task)
        rendered.append(bgr)
    return rendered


def render_video(
    npz_paths: list[Path],
    out_path: Path,
    n_frames_per_sample: int = 90,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    use_full_video: bool = False,
    only_pred: bool = True,
    show_kp: bool = True,
    n_loops: int = 1,
    port: int = 8080,
    track_width: float = 4.5,
    point_size: float = 0.01,
    frustum_scale: float = 0.07,
    show_axes: bool = False,
    orbit_scale: float = 0.75,
    transport_format: str = "jpeg",
    trace_color_by: str = "age",
    reveal_future: bool = True,
    per_sample: bool = False,
) -> None:
    """Drive a connected viser client through the DEFAULT_ORBIT_FIT camera
    path; capture each frame via `client.get_render` and stream them into
    an MP4.

    With `per_sample=False` (default) every sample's orbit is appended
    back-to-back into the single MP4 at `out_path`. With `per_sample=True`,
    `out_path` is a directory and each sample gets its own
    `<out_path>/<npz_stem>.mp4` — the stems match the rollout_img overlays and
    the PNG subdirs, so all three outputs line up 1:1.

    Output looks identical to interactive viser (point cloud, frustum,
    trajectories, light background) — viser does the actual rasterization
    on the client side. Requires a browser to connect to
    `http://localhost:<port>` first; works fine over an SSH tunnel.

    With `reveal_future=True` (default) each sample's orbit starts with only
    the first future step of every trace visible and grows the horizon
    linearly until the full trace is shown on the last frame — the predicted
    motion propagates across the video instead of sitting fully drawn.

    Per-sample text:
      - The task string (from the saved NPZ) is printed to stdout.
      - It's also stamped bottom-center on each rendered frame (just the raw
        task string — no video name / frame index / future-step counter).
    """
    import cv2

    viser_mod, tf_mod = _lazy_viser()

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    if per_sample:
        out_path.mkdir(parents=True, exist_ok=True)
        writer = None
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open VideoWriter for {out_path}")

    server = viser_mod.ViserServer(port=port)
    server.scene.set_up_direction("-y")

    # Mock GUI knobs so `_build_scene` runs headless. Defaults match the user's
    # preferred video preset: pred only, no GT/history/axes; thicker tracks;
    # small frustum so it reads as a "screen" floating in front of the camera.
    state = _headless_render_state(
        use_full_video=use_full_video,
        only_pred=only_pred,
        show_kp=show_kp,
        show_axes=show_axes,
        track_width=track_width,
        point_size=point_size,
        frustum_scale=frustum_scale,
        trace_color_by=trace_color_by,
    )

    client = _wait_for_client(server, port)
    print(
        f"Rendering {len(npz_paths)} sample(s) × {n_frames_per_sample} frames @ {fps} fps"
        f"{' with progressive future reveal' if reveal_future else ''}..."
    )

    up_world = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    # Baked-in fit (vetted from camera_path.json), re-anchored per scene inside
    # `_orbit_pose_from_fit` so the orbit auto-rotates with the frustum.
    fit = _scaled_orbit_fit(orbit_scale)
    fov_rad = float(fit["fov"])

    n_written = 0
    for path in npz_paths:
        scene = _load_npz(path)
        # Render each unique frame ONCE (get_render is a browser round-trip),
        # then write the cached buffer `n_loops` times so the MP4 plays the
        # orbit multiple revolutions back-to-back. With t_norm = i / N the wrap
        # from the end of one loop to the start of the next is seamless.
        rendered = _render_scene_orbit_frames(
            client, server, tf_mod, scene, state,
            fit=fit, fov_rad=fov_rad, n_frames_per_sample=n_frames_per_sample,
            width=width, height=height, up_world=up_world,
            transport_format=transport_format, path_name=path.name,
            reveal_future=reveal_future,
        )
        if not rendered:
            print(f"  [{path.name}] no frames rendered; skipping")
            continue
        if per_sample:
            sample_out = out_path / f"{path.stem}.mp4"
            writer = cv2.VideoWriter(str(sample_out), fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"  [{path.name}] VideoWriter failed for {sample_out}; skipping")
                continue
        loops = max(int(n_loops), 1)
        for _ in range(loops):
            for bgr in rendered:
                writer.write(bgr)
        if per_sample:
            writer.release()
            n_written += 1
            print(f"  [{path.name}] wrote {sample_out}")

    if per_sample:
        print(f"Wrote {n_written} mp4(s) → {out_path}")
    else:
        writer.release()
        print(f"Wrote {out_path}")


def render_orbit_pngs(
    npz_paths: list[Path],
    save_frames_dir: Path,
    *,
    frame_subsample: int = 10,
    n_frames_per_sample: int = 90,
    width: int = 1280,
    height: int = 720,
    use_full_video: bool = False,
    only_pred: bool = True,
    show_kp: bool = True,
    port: int = 8080,
    track_width: float = 4.5,
    point_size: float = 0.01,
    frustum_scale: float = 0.07,
    show_axes: bool = False,
    orbit_scale: float = 0.75,
    transport_format: str = "jpeg",
    trace_color_by: str = "age",
) -> None:
    """Like ``render_video`` but save still PNGs instead of an MP4.

    No progressive reveal here — every PNG shows the full trace, so the stills
    stay comparable across the orbit.

    For each sample, the same DEFAULT_ORBIT_FIT orbit is rendered
    (``n_frames_per_sample`` unique poses), then uniformly subsampled to every
    ``frame_subsample``-th frame (so the default 90 frames → 9 PNGs) and written
    to ``<save_frames_dir>/<npz_stem>/frame_NNN.png``. The per-sample subdir name
    is the NPZ stem, so the PNGs line up 1:1 with the rollout_img overlays saved by
    ``lerobot_predict_trace_mu0_image_only.py`` (same ``sample_*`` stem).

    Like ``render_video`` this drives a connected viser browser client via
    ``client.get_render``, so connect a browser to ``http://localhost:<port>``
    first (works over an SSH tunnel).
    """
    import cv2

    viser_mod, tf_mod = _lazy_viser()

    save_frames_dir.mkdir(parents=True, exist_ok=True)
    stride = max(int(frame_subsample), 1)

    server = viser_mod.ViserServer(port=port)
    server.scene.set_up_direction("-y")

    state = _headless_render_state(
        use_full_video=use_full_video,
        only_pred=only_pred,
        show_kp=show_kp,
        show_axes=show_axes,
        track_width=track_width,
        point_size=point_size,
        frustum_scale=frustum_scale,
        trace_color_by=trace_color_by,
    )

    client = _wait_for_client(server, port)
    print(
        f"Rendering {len(npz_paths)} sample(s) × {n_frames_per_sample} frames, "
        f"saving every {stride}th as PNG..."
    )

    up_world = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    fit = _scaled_orbit_fit(orbit_scale)
    fov_rad = float(fit["fov"])

    total_saved = 0
    for path in npz_paths:
        scene = _load_npz(path)
        rendered = _render_scene_orbit_frames(
            client, server, tf_mod, scene, state,
            fit=fit, fov_rad=fov_rad, n_frames_per_sample=n_frames_per_sample,
            width=width, height=height, up_world=up_world,
            transport_format=transport_format, path_name=path.name,
        )
        if not rendered:
            print(f"  [{path.name}] no frames rendered; skipping")
            continue
        # Uniform 1/stride subsample of the unique orbit frames.
        sel = list(range(0, len(rendered), stride))
        out_sub = save_frames_dir / path.stem
        out_sub.mkdir(parents=True, exist_ok=True)
        for j, fi in enumerate(sel):
            cv2.imwrite(str(out_sub / f"frame_{j:03d}.png"), rendered[fi])
        total_saved += len(sel)
        print(f"  [{path.name}] wrote {len(sel)} pngs → {out_sub}")

    print(f"Wrote {total_saved} pngs across {len(npz_paths)} sample(s) → {save_frames_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz_path", type=Path, default=None, help="Single .npz to view")
    p.add_argument(
        "--scene_dir", type=Path, default=None,
        help="Directory of .npz files (e.g. <out>/scene3d) — adds a sample picker",
    )
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--share", action="store_true", help="Request a public share URL")
    p.add_argument(
        "--trace_color_by", choices=("age", "uv"), default="age",
        help="Trace colorization: 'age' (default) = per-timestep rainbow with "
             "the dynamic ramp (purple at the current kp → red at each track's "
             "own last valid step; matches visualize_trace trace_color_by='age' "
             "used by the 2D rollout_img stills); 'uv' = legacy per-track hue "
             "from the current (u, v).",
    )
    # ---- Video mode (viser-driven) ---------------------------------------
    # Renders by sending camera params to a connected browser and pulling the
    # rasterized frame back via client.get_render(). Output looks identical to
    # interactive viser. Open the same URL you'd use for the GUI.
    p.add_argument(
        "--save_video", type=Path, default=None,
        help="If set, drive the connected viser client through the "
             "DEFAULT_ORBIT_FIT camera path and save the captured frames "
             "as an MP4. Connect a browser to http://localhost:<port> "
             "first; the script will start once a client connects.",
    )
    p.add_argument(
        "--save_video_dir", type=Path, default=None,
        help="Like --save_video but write ONE MP4 PER SAMPLE into this "
             "directory, named <npz_stem>.mp4 (so files line up 1:1 with the "
             "rollout_img overlays and --save_frames_dir subdirs). Same "
             "--video_* knobs apply.",
    )
    p.add_argument(
        "--save_frames_dir", type=Path, default=None,
        help="Like --save_video but save subsampled still PNGs (one subdir per "
             "sample, named by the NPZ stem) instead of an MP4. Uses the same "
             "--video_* knobs. Connect a browser to http://localhost:<port> first.",
    )
    p.add_argument(
        "--frame_subsample", type=int, default=10,
        help="With --save_frames_dir, keep every Nth orbit frame (default 10, "
             "so --video_n_frames=90 → 9 PNGs per sample).",
    )
    p.add_argument(
        "--video_reveal", action=argparse.BooleanOptionalAction, default=True,
        help="With --save_video: progressively reveal the future trace over "
             "the orbit — the first frames show only the first future step, "
             "the last frame the full horizon, so the trace's propagation "
             "plays out over the video. --no-video_reveal restores the "
             "static full-trace orbit. (PNG mode always renders the full "
             "trace.)",
    )
    p.add_argument("--video_n_frames", type=int, default=90,
                   help="Unique frames per orbit revolution.")
    p.add_argument("--video_loops", type=int, default=1,
                   help="Repeat the rendered orbit this many times in the MP4 "
                        "(default 1). With the loop-friendly t_norm=i/N "
                        "convention, each repeat wraps seamlessly.")
    p.add_argument("--video_fps", type=int, default=30)
    p.add_argument("--video_width", type=int, default=1280)
    p.add_argument("--video_height", type=int, default=720)
    p.add_argument("--video_full_pcd", action="store_true",
                   help="Aggregate the full-video point cloud "
                        "(default: just the single-frame pcd).")
    p.add_argument("--video_show_all_traces", action="store_true",
                   help="Show GT and history in the video too (default: pred only).")
    p.add_argument("--video_no_kp", action="store_true",
                   help="Hide the sampled current-keypoint dots in the video "
                        "(default: shown — only the n_kp keypoints actually used "
                        "for prediction, no padding).")
    p.add_argument("--video_track_width", type=float, default=4.5)
    p.add_argument("--video_point_size", type=float, default=0.01)
    p.add_argument("--video_frustum_scale", type=float, default=0.07)
    p.add_argument("--video_orbit_scale", type=float, default=0.75,
                   help="Multiplier on the orbit's in-plane semi-axes — "
                        "controls how much the camera physically moves during "
                        "the loop. <1 subtler, >1 wider sweep.")
    p.add_argument("--video_show_axes", action="store_true",
                   help="Render the world-axis triad in the video.")
    p.add_argument("--video_transport", choices=("jpeg", "png"), default="jpeg",
                   help="get_render transport format. PNG is lossless but heavier.")
    args = p.parse_args()

    if args.npz_path is None and args.scene_dir is None:
        p.error("provide --npz_path or --scene_dir")

    paths: list[Path]
    if args.scene_dir is not None:
        paths = sorted(args.scene_dir.glob("*.npz"))
        if not paths:
            p.error(f"no .npz files in {args.scene_dir}")
    else:
        paths = [args.npz_path]

    # Print task strings up front (also done per-sample inside the renderers).
    if args.save_video is None and args.save_frames_dir is None:
        for p_ in paths:
            try:
                d = np.load(p_, allow_pickle=False)
                if "task" in d.files:
                    print(f"[{p_.name}] task: {str(d['task'])!r}")
                d.close()
            except Exception:
                pass

    if args.save_frames_dir is not None:
        render_orbit_pngs(
            paths, args.save_frames_dir,
            frame_subsample=args.frame_subsample,
            n_frames_per_sample=args.video_n_frames,
            width=args.video_width,
            height=args.video_height,
            use_full_video=args.video_full_pcd,
            only_pred=not args.video_show_all_traces,
            show_kp=not args.video_no_kp,
            port=args.port,
            track_width=args.video_track_width,
            point_size=args.video_point_size,
            frustum_scale=args.video_frustum_scale,
            show_axes=args.video_show_axes,
            orbit_scale=args.video_orbit_scale,
            transport_format=args.video_transport,
            trace_color_by=args.trace_color_by,
        )
        return

    if args.save_video is not None or args.save_video_dir is not None:
        per_sample = args.save_video_dir is not None
        render_video(
            paths, args.save_video_dir if per_sample else args.save_video,
            n_frames_per_sample=args.video_n_frames,
            fps=args.video_fps,
            width=args.video_width,
            height=args.video_height,
            use_full_video=args.video_full_pcd,
            only_pred=not args.video_show_all_traces,
            show_kp=not args.video_no_kp,
            n_loops=args.video_loops,
            port=args.port,
            track_width=args.video_track_width,
            point_size=args.video_point_size,
            frustum_scale=args.video_frustum_scale,
            show_axes=args.video_show_axes,
            orbit_scale=args.video_orbit_scale,
            transport_format=args.video_transport,
            trace_color_by=args.trace_color_by,
            reveal_future=args.video_reveal,
            per_sample=per_sample,
        )
        return

    visualize(paths, port=args.port, share=args.share, trace_color_by=args.trace_color_by)


if __name__ == "__main__":
    main()
