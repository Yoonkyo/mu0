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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trace-prediction overlay renderer for wandb logging (see plan §4)."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np


def _kp_color_palette(num_kp: int) -> list[tuple[int, int, int]]:
    """Per-keypoint RGB colors via an evenly-spaced HSV hue ramp.

    Used when `pred_color_mode="per_kp"` so the user can visually correlate
    each predicted polyline with its keypoint (e.g. to compare done-head
    truncation lengths across kps). 85% saturation / 95% value reads well
    against both light and dark backgrounds.
    """
    if num_kp <= 0:
        return []
    return [
        tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(i / num_kp, 0.85, 0.95))
        for i in range(num_kp)
    ]


def _kp_colors_from_uv(uv_norm: np.ndarray) -> list[tuple[int, int, int]]:
    """`(N, 2)` uv in `[-1, 1]` → list of `(r, g, b)` ints in `[0, 255]`.

    2D color wheel: hue from `atan2(v, u)` so spatially-near keypoints get
    nearby hues. Mirrors `_colors_from_uv` in `lerobot_visualize_trace_3d.py`
    / `lerobot_trace_gui.py` so 2D overlays and the 3D viser scene agree on
    per-track color.
    """
    n = int(uv_norm.shape[0])
    if n == 0:
        return []
    u = uv_norm[:, 0].astype(np.float32)
    v = uv_norm[:, 1].astype(np.float32)
    theta = np.arctan2(v, u)
    offset = 3.0 * np.pi / 4.0
    hue = ((theta + offset) / (2.0 * np.pi)) % 1.0
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        rgb = colorsys.hsv_to_rgb(float(hue[i]), 0.85, 0.95)
        out.append(tuple(int(round(255 * c)) for c in rgb))
    return out


def _color_from_direction(du: float, dv: float) -> tuple[int, int, int]:
    """Motion direction ``(Δu, Δv)`` → ``(r, g, b)`` via the same color wheel
    as :func:`_kp_colors_from_uv`: hue from ``atan2(Δv, Δu)`` with a ``3π/4``
    offset so the orientation matches the uv-position palette. A zero-length
    delta maps to hue 0 (harmless — such a segment is a single point and is
    not drawn).
    """
    theta = float(np.arctan2(dv, du))
    offset = 3.0 * np.pi / 4.0
    hue = ((theta + offset) / (2.0 * np.pi)) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return tuple(int(round(255 * c)) for c in rgb)


def _color_from_age(frac: float) -> tuple[int, int, int]:
    """Temporal position ``frac`` ∈ ``[0, 1]`` → ``(r, g, b)`` rainbow ramp.

    ``frac=0`` (oldest step, nearest the current frame) is violet/purple and
    ``frac=1`` (newest step) is red; ``hue = 0.75·(1 − frac)`` sweeps purple →
    blue → green → yellow → red. The caller chooses what maps to ``frac`` — for
    ``trace_color_by="age"`` it is the step's position within that kp's own
    valid horizon, so ``frac=1`` lands on the kp's last valid step.
    """
    f = float(np.clip(frac, 0.0, 1.0))
    hue = 0.75 * (1.0 - f)
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return tuple(int(round(255 * c)) for c in rgb)


def _disk_direction_color(
    uv: np.ndarray,
    valid: np.ndarray,
    fi: int,
    fallback: tuple[int, int, int] = (200, 200, 200),
) -> tuple[int, int, int]:
    """Direction color for a dot at frame ``fi`` of a ``(T, 2)`` uv timeline.

    Prefers the *incoming* segment's heading (previous valid frame → ``fi``);
    falls back to the *outgoing* segment (``fi`` → next valid frame) when there
    is no earlier valid frame (e.g. the trace start), and to ``fallback`` light
    gray when the keypoint has no other valid frame at all. Used to color the
    filled start/leading dot in ``trace_color_by="direction"`` mode so it
    agrees with the trail segment touching it.
    """
    prev_f = next((f for f in range(fi - 1, -1, -1) if valid[f]), None)
    if prev_f is not None:
        return _color_from_direction(
            float(uv[fi, 0] - uv[prev_f, 0]), float(uv[fi, 1] - uv[prev_f, 1])
        )
    next_f = next((f for f in range(fi + 1, int(uv.shape[0])) if valid[f]), None)
    if next_f is not None:
        return _color_from_direction(
            float(uv[next_f, 0] - uv[fi, 0]), float(uv[next_f, 1] - uv[fi, 1])
        )
    return fallback


def _group_color_palette(num_groups: int) -> list[tuple[int, int, int]]:
    """Per-future-group RGB colors via an evenly-spaced HSV hue ramp.

    Used when `pred_color_mode="per_group"`: every keypoint's group-k segment
    is colored with `palette[k]`, so a kp that ran the full horizon shows all
    `num_groups` colors in sequence and a kp that truncated shows only the
    earliest colors. Truncation length is then visible at a glance.
    """
    if num_groups <= 0:
        return []
    return [
        tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(i / num_groups, 0.85, 0.95))
        for i in range(num_groups)
    ]


def _draw_polyline_per_group(
    canvas: np.ndarray,
    pts: np.ndarray,
    valid: np.ndarray,
    group_colors: list[tuple[int, int, int]],
    g: int,
    has_prepend: bool,
    thickness: int,
) -> None:
    """Draw a polyline coloring each segment by the future-group of its
    destination future-step. Used for `pred_color_mode="per_group"`.

    Layout:
      * `has_prepend=True`  → pts = [current, step_0, step_1, ..., step_{H-1}]
      * `has_prepend=False` → pts = [step_0, step_1, ..., step_{H-1}]
    Segment `i` connects pts[i] → pts[i+1]. Its color is taken from
    `group_colors[step_idx // g]` where `step_idx` is the destination's
    future-step index.
    """
    if pts.shape[0] < 2 or not group_colors:
        return
    last_group = len(group_colors) - 1
    for i in range(pts.shape[0] - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        # Destination's future-step index.
        step_idx = i if has_prepend else (i + 1)
        group_idx = min(max(step_idx // g, 0), last_group)
        color = group_colors[group_idx]
        p0 = tuple(int(round(v)) for v in pts[i])
        p1 = tuple(int(round(v)) for v in pts[i + 1])
        cv2.line(canvas, p0, p1, color, thickness, lineType=cv2.LINE_AA)


def _denorm_xy(xy_norm: np.ndarray, image_wh: tuple[int, int]) -> np.ndarray:
    """[-1, 1] normalized xy → pixel coords in an (W, H)-sized frame."""
    w, h = image_wh
    xy = (xy_norm + 1.0) / 2.0 * np.asarray([w, h], dtype=np.float32)
    return xy


def _draw_polyline(
    canvas: np.ndarray,
    pts: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    dashed: bool,
) -> None:
    """Draw a polyline skipping invalid slots. `dashed` toggles on/off each segment."""
    if pts.shape[0] < 2:
        return
    on = True
    for i in range(pts.shape[0] - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        if dashed and not on:
            on = True
            continue
        p0 = tuple(int(round(v)) for v in pts[i])
        p1 = tuple(int(round(v)) for v in pts[i + 1])
        cv2.line(canvas, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        if dashed:
            on = False


def render_trace_overlay(
    image: np.ndarray,
    traj_history: np.ndarray,
    traj_future_gt: np.ndarray | None,
    traj_future_pred: np.ndarray,
    valid_history: np.ndarray,
    valid_future: np.ndarray,
    key_pad_mask: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    *,
    current_kp: np.ndarray | None = None,
    prepend_current: bool = False,
    draw_last_uv_marker: bool = False,
    connect_history_to_current: bool = True,
    predicted_valid_future: np.ndarray | None = None,
    pred_color_mode: str = "per_kp",
    pred_group_size: int = 1,
) -> np.ndarray:
    """Overlay history + GT future + pred future onto `image`.

    Args:
        image: `(H, W, 3)` uint8 RGB frame at time `t`.
        traj_history: `(N, h, 3)` xy normalized to [-1, 1], z raw. Post current_kp
            split (`docs/trace_current_kp_split_plan.md`), `traj_history` is
            past-only (index 0 = oldest, index `h-1` = `t-1`); the current frame
            `t` arrives via the new `current_kp` argument. Pre-split callers (no
            `current_kp` passed) still get the legacy behavior — the renderer
            falls back to reading the current uv from `traj_history[:, h-1]`.
        traj_future_gt: `(N, F, 3)` or None.
        traj_future_pred: `(N, F, 3)`.
        valid_history: `(N, h)` bool.
        valid_future: `(N, F)` bool — applies to both GT and pred for drawing purposes.
        key_pad_mask: `(N,)` bool — skip padded keypoints.
        predicted_valid_future: optional `(N, F)` bool from the done head
            (docs/trace_done_head_plan.md). When provided, the *predicted*
            polyline is masked by this AND `valid_future`; the GT polyline is
            unaffected. Step-resolution (already upsampled from group level by
            the caller in flat mode).
        pred_color_mode: how to color the *predicted* polylines.
            - "per_kp" (default): each keypoint gets a distinct color from an
              HSV hue ramp; whole polyline uniform within a kp.
            - "per_kp_uv": each keypoint gets a color derived from its current
              `(u, v)` via a 2D color wheel (`atan2(v, u)`), so spatially-near
              tracks get nearby hues. Drawn with a thinner stroke. Matches the
              palette used by `lerobot_trace_gui` / `lerobot_visualize_trace_3d`
              so 2D overlays line up with the 3D viser scene.
            - "per_group": every kp's group-k segment shares the same color
              (group 0 = red across all kps, group 1 = orange, …, last group
              = magenta). Requires `pred_group_size`. Truncation length per kp
              is then immediately readable from how many colors each polyline
              shows. Recommended when `trace_group_horizon > 1` and the done
              head is enabled.
            - "uniform": single dim red `(255, 80, 80)` for every kp (legacy).
            History stays cyan and GT stays green regardless of this setting.
        pred_group_size: number of consecutive future steps that share a
            color in `pred_color_mode="per_group"` mode (== `trace_group_horizon`
            for the trace policy). Ignored in other modes. Must be >= 1.
        image_size: resize to (W_out, H_out) before drawing; default uses image's own size.
        current_kp: optional `(N, 3)` post-split current frame `t` (xy normalized to
            [-1, 1], z raw). When provided, the polyline-prepend, last-uv marker, and
            history-to-current connector all read from this; when None the renderer
            falls back to `traj_history[:, h-1]` for back-compat.
        prepend_current: if True, prepend the current-frame uv to the pred/GT future
            polylines (with a True slot prepended to `valid_future`) so the future line
            visibly starts at the keypoint's current position.
        draw_last_uv_marker: if True, draw an unfilled red circle at the current-frame
            `(u, v)` for every keypoint with `key_pad_mask[i]=True`.
        connect_history_to_current: only meaningful when `current_kp` is provided.
            When True (default), appends `current_kp[:, :2]` to the history polyline so
            the cyan past-only line walks oldest..t-1 and visibly connects into the
            red current-frame circle. Restores the visual continuity that came "for
            free" pre-split (when `traj_history[:, h-1]` *was* the current frame).

    Returns:
        `(H_out, W_out, 3)` uint8 RGB image.
    """
    if image.dtype != np.uint8:
        image = (image.clip(0, 1) * 255).astype(np.uint8) if image.dtype.kind == "f" else image.astype(np.uint8)
    if image_size is None:
        h, w = image.shape[:2]
    else:
        w, h = image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    canvas = image.copy()

    num_kp = traj_history.shape[0]
    h_len = traj_history.shape[1]
    if key_pad_mask is None:
        key_pad_mask = np.ones(num_kp, dtype=bool)

    hist_xy = _denorm_xy(traj_history[..., :2], (w, h))
    pred_xy = _denorm_xy(traj_future_pred[..., :2], (w, h))
    gt_xy = _denorm_xy(traj_future_gt[..., :2], (w, h)) if traj_future_gt is not None else None

    # Resolve where the "current uv" comes from: post-split callers pass `current_kp`
    # (the canonical source); legacy callers fall back to `traj_history[:, h-1]`.
    if current_kp is not None:
        cur_uv_xy = _denorm_xy(current_kp[..., :2], (w, h))            # (N, 2)
        cur_valid_flat = np.ones(num_kp, dtype=bool)                   # filter guarantees validity
    elif h_len > 0:
        cur_uv_xy = hist_xy[:, h_len - 1, :]                            # (N, 2)
        cur_valid_flat = valid_history[:, h_len - 1]                    # (N,)
    else:
        cur_uv_xy = None
        cur_valid_flat = None

    # Optionally extend the past-only history polyline with the current-frame uv so
    # the cyan line walks oldest -> t-1 -> t (red circle). Only meaningful when
    # current_kp is supplied — without it, traj_history[:, h-1] already IS `t`.
    if (
        connect_history_to_current
        and current_kp is not None
        and cur_uv_xy is not None
        and h_len > 0
    ):
        hist_xy = np.concatenate([hist_xy, cur_uv_xy[:, None, :]], axis=1)            # (N, h+1, 2)
        valid_history = np.concatenate(
            [valid_history, cur_valid_flat[:, None]], axis=1
        )

    # Predicted-valid mask for the pred polyline only. AND with `valid_future` so
    # both the GT-side filter (kp leaves frame in GT) and the done-head filter
    # (model says "stop here") apply. None → fall back to `valid_future` for
    # back-compat with all pre-done-head callers.
    pred_valid_future = (
        (predicted_valid_future & valid_future)
        if predicted_valid_future is not None
        else valid_future
    )

    if prepend_current and cur_uv_xy is not None:
        # Splice the current-frame uv onto the front of the future polylines so they
        # visibly start where the keypoint actually is.
        cur_xy = cur_uv_xy[:, None, :]                                                # (N, 1, 2)
        cur_valid = cur_valid_flat[:, None]                                           # (N, 1) bool
        pred_xy = np.concatenate([cur_xy, pred_xy], axis=1)
        if gt_xy is not None:
            gt_xy = np.concatenate([cur_xy, gt_xy], axis=1)
        valid_future = np.concatenate([cur_valid, valid_future], axis=1)
        pred_valid_future = np.concatenate([cur_valid, pred_valid_future], axis=1)

    if pred_color_mode == "per_kp":
        kp_colors = _kp_color_palette(num_kp)
        group_colors: list[tuple[int, int, int]] = []
    elif pred_color_mode == "per_kp_uv":
        # Pull the *normalized* current uv (not the pixel-space `cur_uv_xy`).
        # Source priority matches `cur_uv_xy` resolution above.
        if current_kp is not None:
            uv_for_color = np.asarray(current_kp[..., :2], dtype=np.float32)
        elif h_len > 0:
            uv_for_color = np.asarray(traj_history[:, h_len - 1, :2], dtype=np.float32)
        else:
            uv_for_color = np.zeros((num_kp, 2), dtype=np.float32)
        kp_colors = _kp_colors_from_uv(uv_for_color)
        group_colors = []
    elif pred_color_mode == "uniform":
        kp_colors = [(255, 80, 80)] * num_kp
        group_colors = []
    elif pred_color_mode == "per_group":
        if pred_group_size < 1:
            raise ValueError(f"pred_group_size must be >= 1; got {pred_group_size}")
        # Original H = (post-prepend length) - (1 if prepend ran else 0).
        H_pred = pred_xy.shape[1] - (1 if (prepend_current and cur_uv_xy is not None) else 0)
        num_groups = max(1, H_pred // pred_group_size)
        group_colors = _group_color_palette(num_groups)
        kp_colors = []  # not used in this mode
    else:
        raise ValueError(f"Unknown pred_color_mode={pred_color_mode!r}")

    pred_has_prepend = bool(prepend_current and cur_uv_xy is not None)
    pred_thickness = 1 if pred_color_mode == "per_kp_uv" else 2

    for i in range(num_kp):
        if not key_pad_mask[i]:
            continue
        _draw_polyline(canvas, hist_xy[i], valid_history[i], (80, 200, 255), 2, dashed=False)
        if gt_xy is not None:
            _draw_polyline(canvas, gt_xy[i], valid_future[i], (0, 255, 0), 2, dashed=False)
        if pred_color_mode == "per_group":
            _draw_polyline_per_group(
                canvas, pred_xy[i], pred_valid_future[i],
                group_colors, pred_group_size, pred_has_prepend, thickness=2,
            )
        else:
            _draw_polyline(
                canvas, pred_xy[i], pred_valid_future[i], kp_colors[i],
                pred_thickness, dashed=False,
            )

    if draw_last_uv_marker and cur_uv_xy is not None:
        # Drawn last so it sits on top of the polylines. (255, 0, 0) is RGB-red,
        # matching the rest of this module's convention (history is RGB sky-blue,
        # GT is green, pred is RGB-light-red). thickness=2 with positive radius =>
        # outlined circle.
        for i in range(num_kp):
            if not key_pad_mask[i]:
                continue
            uv = cur_uv_xy[i]
            cx, cy = int(round(float(uv[0]))), int(round(float(uv[1])))
            cv2.circle(canvas, (cx, cy), radius=6, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    return canvas


def _blend_line(
    canvas: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
    thickness: int,
) -> None:
    """Alpha-blend a single anti-aliased line of ``color`` at opacity ``alpha`` onto ``canvas``.

    Only the line's bounding box is touched, so cost scales with segment length
    rather than image area. The line is rasterized into a uint8 mask whose AA
    intensities act as a per-pixel coverage weight; the final per-pixel blend
    weight is ``alpha * mask/255``.
    """
    if alpha <= 0.0:
        return
    h_img, w_img = canvas.shape[:2]
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    margin = thickness + 2
    minx = max(0, min(x0, x1) - margin)
    maxx = min(w_img, max(x0, x1) + margin + 1)
    miny = max(0, min(y0, y1) - margin)
    maxy = min(h_img, max(y0, y1) + margin + 1)
    if maxx <= minx or maxy <= miny:
        return
    h_b, w_b = maxy - miny, maxx - minx
    mask = np.zeros((h_b, w_b), dtype=np.uint8)
    pp0 = (x0 - minx, y0 - miny)
    pp1 = (x1 - minx, y1 - miny)
    cv2.line(mask, pp0, pp1, 255, thickness, lineType=cv2.LINE_AA)
    if not mask.any():
        return
    m = (mask.astype(np.float32) * (alpha / 255.0))[..., None]
    c_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    roi = canvas[miny:maxy, minx:maxx].astype(np.float32)
    blended = roi * (1.0 - m) + c_arr * m
    np.clip(blended, 0, 255, out=blended)
    canvas[miny:maxy, minx:maxx] = blended.astype(np.uint8)


def render_trace_motion_still(
    image: np.ndarray,
    traj_history: np.ndarray,
    traj_future_pred: np.ndarray,
    valid_history: np.ndarray,
    valid_future: np.ndarray,
    current_kp: np.ndarray,
    *,
    key_pad_mask: np.ndarray | None = None,
    predicted_valid_future: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    bg_alpha: float = 0.3,
    bg_blend_color: tuple[int, int, int] = (0, 0, 0),
    dot_radius: int | None = None,
    future_only: bool = True,
    trail_thickness: int | None = None,
    trail_alpha_min: float = 0.7,
    trace_color_by: str = "uv",
) -> np.ndarray:
    """Single-image counterpart to ``render_trace_motion_video``'s last frame.

    Differences from the video's last frame:
      * the full uv-rainbow trail is drawn at *light* fade (``trail_alpha_min``
        defaults to 0.7 rather than the video's 0.15), so the whole trace
        stays bold while still hinting at direction.
      * the filled rainbow disk sits at the **start** of each kp's trace (its
        current uv) rather than at the end of the predicted trace.
      * the background is blended *toward* ``bg_blend_color`` at weight
        ``1 - bg_alpha`` instead of multiplied by ``bg_alpha`` against black
        — pass ``bg_blend_color=(255, 255, 255)`` for a washed-out near-white
        look. ``bg_blend_color=(0, 0, 0)`` reproduces the video's dim-to-black
        behavior.

    ``trace_color_by`` selects how each kp's trail (and start dot) is colored:
      * ``"uv"`` (default): one color per kp from its current ``(u, v)`` via the
        2D color wheel — the legacy behavior.
      * ``"direction"``: each segment colored by its motion direction
        ``atan2(Δv, Δu)`` (same wheel), so heading is read from hue.
      * ``"age"``: each segment colored by its position along the trace,
        normalized to that kp's *own* valid horizon — oldest (nearest the
        current frame) violet/purple, last valid step red. Every trace spans
        the full ramp, so the last color is always red regardless of how far
        the kp's (done-head / GT-masked) horizon reaches.
    """
    if trace_color_by not in ("uv", "direction", "age"):
        raise ValueError(
            f"trace_color_by must be 'uv'/'direction'/'age'; got {trace_color_by!r}"
        )
    if image.dtype != np.uint8:
        image = (image.clip(0, 1) * 255).astype(np.uint8) if image.dtype.kind == "f" else image.astype(np.uint8)
    if image_size is None:
        h_img, w_img = image.shape[:2]
    else:
        w_img, h_img = image_size
        image = cv2.resize(image, (w_img, h_img), interpolation=cv2.INTER_AREA)

    num_kp = traj_history.shape[0]
    h_len = traj_history.shape[1]
    f_len = traj_future_pred.shape[1]
    start_t = h_len if future_only else 0
    n_frames = (1 + f_len) if future_only else (h_len + 1 + f_len)

    if key_pad_mask is None:
        key_pad_mask = np.ones(num_kp, dtype=bool)
    pred_valid = (
        (predicted_valid_future & valid_future)
        if predicted_valid_future is not None
        else valid_future
    )

    rgb_colors = _kp_colors_from_uv(np.asarray(current_kp[..., :2], dtype=np.float32))

    if dot_radius is None:
        dot_radius = max(4, min(w_img, h_img) // 60)
    if trail_thickness is None:
        trail_thickness = max(1, dot_radius // 2)

    bg_alpha_clip = float(np.clip(bg_alpha, 0.0, 1.0))
    blend_arr = np.asarray(bg_blend_color, dtype=np.float32).reshape(1, 1, 3)
    bg = (
        image.astype(np.float32) * bg_alpha_clip
        + blend_arr * (1.0 - bg_alpha_clip)
    ).clip(0, 255).astype(np.uint8)

    hist_xy = _denorm_xy(traj_history[..., :2], (w_img, h_img))     # (N, h, 2)
    cur_xy = _denorm_xy(current_kp[..., :2], (w_img, h_img))         # (N, 2)
    pred_xy = _denorm_xy(traj_future_pred[..., :2], (w_img, h_img))  # (N, F, 2)

    full_xy = np.concatenate(
        [hist_xy, cur_xy[:, None, :], pred_xy], axis=1
    )  # (N, h + 1 + F, 2)
    kpm_col = key_pad_mask[:, None]
    full_valid = np.concatenate(
        [
            valid_history & kpm_col,
            kpm_col,
            pred_valid & kpm_col,
        ],
        axis=1,
    )  # (N, h + 1 + F)
    full_valid = full_valid & np.isfinite(full_xy).all(axis=-1)
    all_xy = full_xy[:, start_t : start_t + n_frames]      # (N, n_frames, 2)
    all_valid = full_valid[:, start_t : start_t + n_frames]  # (N, n_frames)
    all_xy_int = np.where(all_valid[..., None], np.round(all_xy), 0).astype(np.int32)
    # Normalized-uv timeline parallel to `all_xy`, kept un-rounded so the
    # direction colorizer measures `(Δu, Δv)` in the original [-1, 1] space
    # (correct regardless of image aspect ratio).
    full_uv = np.concatenate(
        [traj_history[..., :2], current_kp[..., :2][:, None, :], traj_future_pred[..., :2]],
        axis=1,
    )
    all_uv = full_uv[:, start_t : start_t + n_frames]  # (N, n_frames, 2)

    canvas = bg.copy()

    fi_last = n_frames - 1
    if fi_last > 0:
        max_age_ref = max(1, n_frames - 1)
        alpha_span = 1.0 - float(trail_alpha_min)
        # Dynamic "age" normalization: per-kp first/last valid step so each
        # trace's purple→red ramp spans its OWN valid horizon (after done-head /
        # GT masking) — the last segment of every trace is red, not just those
        # that run the full future_len. `argmax` finds the first True; the same
        # trick on the reversed row finds the last True.
        first_valid_fp = np.argmax(all_valid, axis=1)
        last_valid_fp = (n_frames - 1) - np.argmax(all_valid[:, ::-1], axis=1)
        age_span = np.maximum(1, last_valid_fp - first_valid_fp)
        segments: list[tuple[int, int, int, int, int, tuple[int, int, int]]] = []
        for i in range(num_kp):
            if not key_pad_mask[i]:
                continue
            prev_valid = False
            px = py = 0
            pu = pv = 0.0
            for fp in range(fi_last + 1):
                if not all_valid[i, fp]:
                    prev_valid = False
                    continue
                cx, cy = int(all_xy_int[i, fp, 0]), int(all_xy_int[i, fp, 1])
                cu, cv = float(all_uv[i, fp, 0]), float(all_uv[i, fp, 1])
                if prev_valid:
                    age = fi_last - fp
                    if trace_color_by == "age":
                        seg_color = _color_from_age(
                            (fp - first_valid_fp[i]) / age_span[i]
                        )
                    elif trace_color_by == "direction":
                        seg_color = _color_from_direction(cu - pu, cv - pv)
                    else:
                        seg_color = rgb_colors[i]
                    segments.append((age, px, py, cx, cy, seg_color))
                px, py = cx, cy
                pu, pv = cu, cv
                prev_valid = True
        segments.sort(key=lambda s: -s[0])
        for age, x0, y0, x1, y1, color in segments:
            a = float(trail_alpha_min) + alpha_span * (1.0 - age / max_age_ref)
            if a < float(trail_alpha_min):
                a = float(trail_alpha_min)
            elif a > 1.0:
                a = 1.0
            _blend_line(canvas, (x0, y0), (x1, y1), color, a, trail_thickness)

    # The filled disk marks each kp's trace start (its current uv). In
    # "direction"/"age" modes it takes the color of the segment it touches
    # (first heading / oldest-purple) so the dot reads consistently with the
    # trail.
    for i in range(num_kp):
        if not all_valid[i, 0]:
            continue
        cx, cy = int(all_xy_int[i, 0, 0]), int(all_xy_int[i, 0, 1])
        if trace_color_by == "age":
            disk_color = _color_from_age(0.0)
        elif trace_color_by == "direction":
            disk_color = _disk_direction_color(all_uv[i], all_valid[i], 0)
        else:
            disk_color = rgb_colors[i]
        cv2.circle(
            canvas, (cx, cy), dot_radius, disk_color,
            thickness=-1, lineType=cv2.LINE_AA,
        )

    return canvas


def render_trace_motion_video(
    image: np.ndarray,
    traj_history: np.ndarray,
    traj_future_pred: np.ndarray,
    valid_history: np.ndarray,
    valid_future: np.ndarray,
    current_kp: np.ndarray,
    *,
    key_pad_mask: np.ndarray | None = None,
    predicted_valid_future: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
    bg_alpha: float = 0.4,
    dot_radius: int | None = None,
    future_only: bool = False,
    color_mode: str = "uv_rainbow",
    draw_trail: bool = True,
    trail_thickness: int | None = None,
    trail_alpha_min: float = 0.15,
    trace_color_by: str = "uv",
) -> np.ndarray:
    """Animate per-keypoint motion as moving dots on a faded static background.

    Returns ``(T, H, W, 3)`` uint8 RGB frames where ``T = h_len + 1 + future_len``.
    Each frame shows the (single) input image dimmed to ``bg_alpha`` with a
    filled circle at every visible keypoint's ``(u, v)`` for that timestep.
    Per-keypoint color is the same uv-rainbow used by
    ``render_trace_overlay(pred_color_mode='per_kp_uv')``, so the same kp wears
    the same color across frames and across the static overlay.

    Args:
        image: ``(H, W, 3)`` uint8 RGB current frame (the only image we have;
            background is held constant across the whole video).
        traj_history: ``(N, h, 3)`` xy normalized to ``[-1, 1]``, z raw.
        traj_future_pred: ``(N, F, 3)``.
        valid_history: ``(N, h)`` bool.
        valid_future: ``(N, F)`` bool.
        current_kp: ``(N, 3)`` post-split current frame ``t`` (xy in ``[-1, 1]``).
        key_pad_mask: optional ``(N,)`` bool — skip padded keypoints.
        predicted_valid_future: optional ``(N, F)`` bool from the done head.
            When provided, future frames are masked by this AND ``valid_future``.
        image_size: resize to ``(W_out, H_out)`` before drawing.
        bg_alpha: 0.0 → black background, 1.0 → full image. Default 0.4 keeps the
            scene legible while making the dots pop.
        dot_radius: filled-circle radius in pixels; default scales with image.
        future_only: if True, the video starts at the current frame and only
            animates the predicted future (``T = 1 + future_len``); history
            frames are skipped.
        color_mode: ``"uv_rainbow"`` (default) draws each kp as a solid disk
            colored by its current ``(u, v)``. ``"current_pixels"`` instead
            samples a circular patch around each kp's current uv from the
            (resized) input image and stamps that patch along the trajectory,
            so the underlying texture appears to slide with the trace.
        draw_trail: if True (default), leave a polyline trace behind each kp
            connecting its valid positions from the first video frame up to
            the current frame. The trail uses the kp's uv-rainbow color (the
            same color as the disk in ``"uv_rainbow"`` mode) so each kp's
            motion is visible as a continuous trace instead of just an
            isolated moving dot. Applies in both color modes. Trail segments
            fade with age: the segment just behind the current dot is fully
            opaque, segments further in the past fade toward ``trail_alpha_min``.
        trail_thickness: line width in pixels for the trail polylines.
            Defaults to ``max(1, dot_radius // 2)``.
        trail_alpha_min: minimum opacity (in [0, 1]) for the oldest trail
            segment in the video. ``1.0`` disables the fade (all segments are
            fully opaque); ``0.0`` lets the oldest segment vanish entirely.
            Default ``0.15``.
        trace_color_by: hue scheme for the trail (and the ``uv_rainbow`` disk),
            orthogonal to ``color_mode``. ``"uv"`` (default) gives each kp one
            color from its current ``(u, v)``; ``"direction"`` colors each
            segment by its heading ``atan2(Δv, Δu)``; ``"age"`` colors each
            segment by its position within that kp's *own* valid horizon
            (oldest violet/purple → last-valid-step red), so every trace spans
            the full ramp and the moving disk sweeps purple→red over each kp's
            lifetime, reaching red exactly at its last valid step.
            The ``current_pixels`` texture patches are unaffected — only the
            trail recolors.
    """
    if color_mode not in ("uv_rainbow", "current_pixels"):
        raise ValueError(f"color_mode must be 'uv_rainbow' or 'current_pixels', got {color_mode!r}")
    if trace_color_by not in ("uv", "direction", "age"):
        raise ValueError(
            f"trace_color_by must be 'uv'/'direction'/'age'; got {trace_color_by!r}"
        )
    if image.dtype != np.uint8:
        image = (image.clip(0, 1) * 255).astype(np.uint8) if image.dtype.kind == "f" else image.astype(np.uint8)
    if image_size is None:
        h_img, w_img = image.shape[:2]
    else:
        w_img, h_img = image_size
        image = cv2.resize(image, (w_img, h_img), interpolation=cv2.INTER_AREA)

    num_kp = traj_history.shape[0]
    h_len = traj_history.shape[1]
    f_len = traj_future_pred.shape[1]
    start_t = h_len if future_only else 0
    n_frames = (1 + f_len) if future_only else (h_len + 1 + f_len)

    if key_pad_mask is None:
        key_pad_mask = np.ones(num_kp, dtype=bool)
    pred_valid = (
        (predicted_valid_future & valid_future)
        if predicted_valid_future is not None
        else valid_future
    )

    rgb_colors = _kp_colors_from_uv(np.asarray(current_kp[..., :2], dtype=np.float32))

    if dot_radius is None:
        dot_radius = max(4, min(w_img, h_img) // 60)
    if trail_thickness is None:
        trail_thickness = max(1, dot_radius // 2)

    bg_alpha_clip = float(np.clip(bg_alpha, 0.0, 1.0))
    bg = (image.astype(np.float32) * bg_alpha_clip).clip(0, 255).astype(np.uint8)

    hist_xy = _denorm_xy(traj_history[..., :2], (w_img, h_img))     # (N, h, 2)
    cur_xy = _denorm_xy(current_kp[..., :2], (w_img, h_img))         # (N, 2)
    pred_xy = _denorm_xy(traj_future_pred[..., :2], (w_img, h_img))  # (N, F, 2)

    # Per-frame, per-kp pixel position and validity across the full
    # (history → current → future) timeline. Sliced by `start_t` so the
    # arrays are aligned to the video frame index `fi`. Validity folds in
    # `key_pad_mask`, the original per-step valid flags, the done-head mask
    # (already merged into `pred_valid`), and a finite-xy check.
    full_xy = np.concatenate(
        [hist_xy, cur_xy[:, None, :], pred_xy], axis=1
    )  # (N, h + 1 + F, 2)
    kpm_col = key_pad_mask[:, None]
    full_valid = np.concatenate(
        [
            valid_history & kpm_col,
            kpm_col,
            pred_valid & kpm_col,
        ],
        axis=1,
    )  # (N, h + 1 + F)
    full_valid = full_valid & np.isfinite(full_xy).all(axis=-1)
    all_xy = full_xy[:, start_t : start_t + n_frames]      # (N, n_frames, 2)
    all_valid = full_valid[:, start_t : start_t + n_frames]  # (N, n_frames)
    all_xy_int = np.where(all_valid[..., None], np.round(all_xy), 0).astype(np.int32)
    # Normalized-uv timeline parallel to `all_xy` for the direction colorizer
    # (measures `(Δu, Δv)` in [-1, 1] space). `max_age_ref` is the constant
    # denominator for the "age" ramp (absolute temporal position, so a
    # segment's color is stable across frames).
    full_uv = np.concatenate(
        [traj_history[..., :2], current_kp[..., :2][:, None, :], traj_future_pred[..., :2]],
        axis=1,
    )
    all_uv = full_uv[:, start_t : start_t + n_frames]  # (N, n_frames, 2)
    max_age_ref = max(1, n_frames - 1)
    # Per-kp valid span for the dynamic "age" ramp — normalize each kp's color
    # to its own last valid step so every trace ends red (see
    # render_trace_motion_still). Computed once over the full timeline, so a
    # segment's color is stable across frames.
    first_valid_fp = np.argmax(all_valid, axis=1)
    last_valid_fp = (n_frames - 1) - np.argmax(all_valid[:, ::-1], axis=1)
    age_span = np.maximum(1, last_valid_fp - first_valid_fp)

    # Per-keypoint texture patches sampled once from the (resized) current frame
    # at each kp's current uv. Reused for every frame in the video so the same
    # appearance follows the kp along its trajectory. Patches whose source crop
    # falls fully outside the image are marked invalid and that kp is rendered
    # as a solid disk in `current_pixels` mode.
    if color_mode == "current_pixels":
        r = dot_radius
        d = 2 * r + 1
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk_mask = (xx * xx + yy * yy) <= (r * r)  # (d, d) bool
        patches = np.zeros((num_kp, d, d, 3), dtype=np.uint8)
        patch_valid = np.zeros(num_kp, dtype=bool)
        for i in range(num_kp):
            if not key_pad_mask[i]:
                continue
            cxi, cyi = cur_xy[i]
            if not (np.isfinite(cxi) and np.isfinite(cyi)):
                continue
            cx, cy = int(round(float(cxi))), int(round(float(cyi)))
            sy0, sy1 = cy - r, cy + r + 1
            sx0, sx1 = cx - r, cx + r + 1
            csy0, csy1 = max(0, sy0), min(h_img, sy1)
            csx0, csx1 = max(0, sx0), min(w_img, sx1)
            if csy1 <= csy0 or csx1 <= csx0:
                continue  # patch fully off-image; leave invalid
            py0, py1 = csy0 - sy0, csy1 - sy0
            px0, px1 = csx0 - sx0, csx1 - sx0
            patches[i, py0:py1, px0:px1] = image[csy0:csy1, csx0:csx1]
            patch_valid[i] = True
    else:
        patches = None
        patch_valid = None
        disk_mask = None

    frames = np.empty((n_frames, h_img, w_img, 3), dtype=np.uint8)
    for fi in range(n_frames):
        canvas = bg.copy()

        # Trail pass: redraw each kp's polyline through all valid positions
        # in [0, fi] before stamping the current dot/patch. The canvas is
        # rebuilt from `bg` every frame, so we redraw the full trail each
        # time rather than maintaining a persistent trail buffer.
        # Segments are blended oldest-first so newer segments (higher alpha)
        # composite on top — this gives a consistent z-order across kps when
        # different keypoints' trails cross. Per-segment alpha fades linearly
        # from 1.0 at the freshest segment to `trail_alpha_min` at the oldest
        # segment the video can possibly contain (age = n_frames - 1).
        if draw_trail and fi > 0:
            alpha_span = 1.0 - float(trail_alpha_min)
            segments: list[tuple[int, int, int, int, int, tuple[int, int, int]]] = []
            for i in range(num_kp):
                if not key_pad_mask[i]:
                    continue
                prev_valid = False
                px = py = 0
                pu = pv = 0.0
                for fp in range(fi + 1):
                    if not all_valid[i, fp]:
                        prev_valid = False
                        continue
                    cx, cy = int(all_xy_int[i, fp, 0]), int(all_xy_int[i, fp, 1])
                    cu, cv = float(all_uv[i, fp, 0]), float(all_uv[i, fp, 1])
                    if prev_valid:
                        age = fi - fp  # 0 for the freshest segment ending at fi
                        if trace_color_by == "age":
                            seg_color = _color_from_age(
                                (fp - first_valid_fp[i]) / age_span[i]
                            )
                        elif trace_color_by == "direction":
                            seg_color = _color_from_direction(cu - pu, cv - pv)
                        else:
                            seg_color = rgb_colors[i]
                        segments.append((age, px, py, cx, cy, seg_color))
                    px, py = cx, cy
                    pu, pv = cu, cv
                    prev_valid = True
            # Oldest first → highest age first.
            segments.sort(key=lambda s: -s[0])
            for age, x0, y0, x1, y1, color in segments:
                a = float(trail_alpha_min) + alpha_span * (1.0 - age / max_age_ref)
                if a < float(trail_alpha_min):
                    a = float(trail_alpha_min)
                elif a > 1.0:
                    a = 1.0
                _blend_line(canvas, (x0, y0), (x1, y1), color, a, trail_thickness)

        # Current-frame dot/patch pass.
        for i in range(num_kp):
            if not all_valid[i, fi]:
                continue
            cx, cy = int(all_xy_int[i, fi, 0]), int(all_xy_int[i, fi, 1])
            if color_mode == "current_pixels" and patch_valid[i]:
                r = dot_radius
                sy0, sy1 = cy - r, cy + r + 1
                sx0, sx1 = cx - r, cx + r + 1
                csy0, csy1 = max(0, sy0), min(h_img, sy1)
                csx0, csx1 = max(0, sx0), min(w_img, sx1)
                if csy1 > csy0 and csx1 > csx0:
                    py0, py1 = csy0 - sy0, csy1 - sy0
                    px0, px1 = csx0 - sx0, csx1 - sx0
                    m = disk_mask[py0:py1, px0:px1]
                    canvas[csy0:csy1, csx0:csx1][m] = patches[i, py0:py1, px0:px1][m]
            else:
                if trace_color_by == "age":
                    disk_color = _color_from_age(
                        (fi - first_valid_fp[i]) / age_span[i]
                    )
                elif trace_color_by == "direction":
                    disk_color = _disk_direction_color(all_uv[i], all_valid[i], fi)
                else:
                    disk_color = rgb_colors[i]
                cv2.circle(
                    canvas, (cx, cy), dot_radius, disk_color,
                    thickness=-1, lineType=cv2.LINE_AA,
                )
        frames[fi] = canvas

    return frames
