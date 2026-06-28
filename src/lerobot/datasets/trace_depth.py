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
"""Depth-image rendering utilities for the trace pipeline.

Converts a raw metric-depth image (meters, float) into a 3-channel turbo-colormap
RGB image suitable for consumption by the SmolVLA vision encoder. The log-domain
normalization range is dataset-wide (see `trace_delta_stats.compute_trace_stats`)
so the same meter value always maps to the same color across frames.
"""

from __future__ import annotations

import numpy as np


def _build_turbo_lut() -> np.ndarray:
    """256-entry turbo colormap LUT. Values in [0, 1], shape (256, 3) float32.

    Polynomial approximation published by Anton Mikhailov:
    https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
    Reproduces matplotlib's `turbo` to ~1e-3 without a matplotlib runtime dep.
    """
    red4 = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234], dtype=np.float64)
    green4 = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333], dtype=np.float64)
    blue4 = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771], dtype=np.float64)
    red2 = np.array([-152.94239396, 59.28637943], dtype=np.float64)
    green2 = np.array([4.27729857, 2.82956604], dtype=np.float64)
    blue2 = np.array([-89.90310912, 27.34824973], dtype=np.float64)

    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    v4 = np.stack([np.ones_like(x), x, x**2, x**3], axis=1)
    v2 = np.stack([x**4, x**5], axis=1)
    r = v4 @ red4 + v2 @ red2
    g = v4 @ green4 + v2 @ green2
    b = v4 @ blue4 + v2 @ blue2
    lut = np.stack([r, g, b], axis=1)
    return np.clip(lut, 0.0, 1.0).astype(np.float32)


TURBO_LUT: np.ndarray = _build_turbo_lut()
"""(256, 3) float32 turbo colormap LUT, indexed by uint8."""


def render_depth_rgb(depth_hw: np.ndarray, log_min: float, log_max: float) -> np.ndarray:
    """Render a metric-depth frame into a (H, W, 3) turbo-colored float32 image in [0, 1].

    Args:
        depth_hw: (H, W) depth in meters. NaNs or non-finite / non-positive values are
            treated as invalid and mapped to the "far/dim" endpoint.
        log_min, log_max: log-meter endpoints of the normalization range. Typically
            computed once over the dataset as (log(p1), log(p99)) by
            `trace_delta_stats.compute_trace_stats`.

    The same meter value always maps to the same color across frames, because the
    normalization is dataset-global rather than per-frame.
    """
    if depth_hw.ndim != 2:
        raise ValueError(f"expected (H, W) depth, got shape {depth_hw.shape}")
    d = np.asarray(depth_hw, dtype=np.float32)
    valid = np.isfinite(d) & (d > 1e-3)
    # Clip in meter space before taking the log so invalid entries never produce -inf.
    lo, hi = float(np.exp(log_min)), float(np.exp(log_max))
    clipped = np.where(valid, np.clip(d, lo, hi), lo)
    log_d = np.log(clipped)
    denom = max(log_max - log_min, 1e-6)
    norm = (log_d - log_min) / denom  # ∈ [0, 1]
    # Invert so near surfaces are bright (turbo high) and far surfaces dim.
    norm = 1.0 - norm
    # Invalid → far/dim endpoint (norm=0 after inversion). Matches "unknown = far".
    norm = np.where(valid, norm, 0.0)
    idx = np.clip((norm * 255.0).astype(np.int32), 0, 255)
    return TURBO_LUT[idx]
