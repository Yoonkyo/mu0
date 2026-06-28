"""Depth-Anything-V2 metric-depth wrapper with training-distribution alignment.

The policy was trained on TraceExtract VGGT4-derived metric depth rendered to
turbo RGB by `render_depth_rgb(D_meters, log_min_train, log_max_train)` from
`lerobot.datasets.trace_depth`. DA-V2 produces metric depth too, but its
calibration drifts from VGGT4's, so feeding raw DA-V2 meters into the same
turbo LUT produces panels that saturate at one end. We per-frame align DA-V2
into the training log range before rendering.

See `docs/predict_trace_image_only_plan.md` §2 for the design.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch


_DA_V2_REPO_RELPATH = "infer_helpers/Depth-Anything-V2/metric_depth"


def _find_repo_root() -> Path:
    """Walk up from this file to find the repo root (one that contains `infer_helpers/`)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _DA_V2_REPO_RELPATH).exists():
            return parent
    raise FileNotFoundError(
        f"Could not locate {_DA_V2_REPO_RELPATH} from {here}. "
        "Make sure the vendored Depth-Anything-V2 repo is present."
    )


_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


AlignMethod = Literal["log_percentile", "median_scale", "none"]


class DepthAnythingV2Wrapper:
    """Loads DA-V2 metric-depth once; per-frame infer + align to training range."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        encoder: str = "vitl",
        max_depth: float = 20.0,
        input_size: int = 518,
        device: str = "cuda",
    ):
        if encoder not in _MODEL_CONFIGS:
            raise ValueError(
                f"unknown encoder {encoder!r}; expected one of {list(_MODEL_CONFIGS)}"
            )
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"DA-V2 checkpoint not found: {ckpt}")

        repo_root = _find_repo_root()
        repo_path = str(repo_root / _DA_V2_REPO_RELPATH)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from depth_anything_v2.dpt import DepthAnythingV2

        cfg = _MODEL_CONFIGS[encoder]
        self.model = DepthAnythingV2(**{**cfg, "max_depth": max_depth})
        self.model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        self.model.to(device).eval()
        self.encoder = encoder
        self.input_size = int(input_size)
        self.device = device

    @torch.no_grad()
    def infer_meters(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        """`(H, W, 3)` uint8 RGB → `(H, W)` float32 raw DA-V2 metric depth (meters)."""
        if image_rgb_uint8.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 RGB, got dtype={image_rgb_uint8.dtype}"
            )
        if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[-1] != 3:
            raise ValueError(
                f"expected (H, W, 3) RGB, got shape={image_rgb_uint8.shape}"
            )
        # DA-V2's `infer_image` expects BGR uint8 (it does cv2.cvtColor BGR2RGB
        # internally), so flip channels here to keep our public API RGB.
        bgr = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)
        depth = self.model.infer_image(bgr, self.input_size)
        return np.asarray(depth, dtype=np.float32)

    def infer_aligned(
        self,
        image_rgb_uint8: np.ndarray,
        align_method: AlignMethod,
        log_min_train: float,
        log_max_train: float,
    ) -> tuple[np.ndarray, dict]:
        """Run DA-V2 + align into training log-depth range.

        Returns `(D_aligned, diag)` where `D_aligned` is `(H, W)` float32 meters
        and `diag` carries per-frame percentiles for sanity-check JSON dumping.
        """
        d_da = self.infer_meters(image_rgb_uint8)
        d_aligned, diag = align_depth(
            d_da, align_method, log_min_train, log_max_train
        )
        return d_aligned, diag


def _valid_mask(d: np.ndarray) -> np.ndarray:
    return np.isfinite(d) & (d > 1e-3)


def _percentiles_log_m(d: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not mask.any():
        return float("nan"), float("nan")
    log_d = np.log(np.clip(d[mask], 1e-3, None))
    p1, p99 = np.percentile(log_d, [1.0, 99.0])
    return float(p1), float(p99)


def align_depth(
    d_da: np.ndarray,
    method: AlignMethod,
    log_min_train: float,
    log_max_train: float,
) -> tuple[np.ndarray, dict]:
    """Per-frame align raw DA-V2 meters → meters in training log range.

    `method`:
      - `log_percentile`: 2-param affine remap in log-space, matching DA-V2's
        1st/99th log-percentiles to `(log_min_train, log_max_train)`.
      - `median_scale`: multiplicative scale matching `median(D_da)` to
        `sqrt(p1_train * p99_train)` (geometric mean in linear space).
      - `none`: passthrough.

    `diag` always carries the pre/post log-percentile pair for the sanity dump.
    """
    if d_da.ndim != 2:
        raise ValueError(f"expected (H, W) depth, got shape={d_da.shape}")
    valid = _valid_mask(d_da)
    p1_da_log, p99_da_log = _percentiles_log_m(d_da, valid)

    if method == "none" or not valid.any() or not np.isfinite(p1_da_log) or not np.isfinite(p99_da_log):
        d_aligned = d_da.astype(np.float32, copy=True)
    elif method == "log_percentile":
        log_d = np.log(np.clip(d_da, 1e-3, None))
        denom = max(p99_da_log - p1_da_log, 1e-6)
        slope = (log_max_train - log_min_train) / denom
        intercept = log_min_train - slope * p1_da_log
        log_aligned = slope * log_d + intercept
        d_aligned = np.exp(log_aligned).astype(np.float32, copy=False)
        # Keep originally-invalid pixels invalid so the renderer routes them
        # to the "far/dim" endpoint instead of inventing a meter value.
        d_aligned = np.where(valid, d_aligned, np.float32("nan"))
    elif method == "median_scale":
        m_da = float(np.median(d_da[valid]))
        m_train = float(np.exp(0.5 * (log_min_train + log_max_train)))
        # exp((log_min+log_max)/2) = sqrt(p1_train * p99_train), the geometric
        # mean of the training log-depth endpoints.
        scale = m_train / max(m_da, 1e-6)
        d_aligned = (d_da * scale).astype(np.float32, copy=False)
        d_aligned = np.where(valid, d_aligned, np.float32("nan"))
    else:
        raise ValueError(
            f"unknown align method {method!r}; expected log_percentile/median_scale/none"
        )

    valid_aligned = _valid_mask(d_aligned)
    p1_al_log, p99_al_log = _percentiles_log_m(d_aligned, valid_aligned)
    diag = {
        "method": method,
        "p1_da_m": float(np.exp(p1_da_log)) if np.isfinite(p1_da_log) else float("nan"),
        "p99_da_m": float(np.exp(p99_da_log)) if np.isfinite(p99_da_log) else float("nan"),
        "p1_aligned_m": float(np.exp(p1_al_log)) if np.isfinite(p1_al_log) else float("nan"),
        "p99_aligned_m": float(np.exp(p99_al_log)) if np.isfinite(p99_al_log) else float("nan"),
        "log_min_train": float(log_min_train),
        "log_max_train": float(log_max_train),
    }
    return d_aligned, diag
