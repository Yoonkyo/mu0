#!/usr/bin/env python
"""GPT VLM baseline for TraceGen.

Generates future keypoint trajectories using the GPT-5.5 multimodal API
instead of the trained SmolVLA-Trace policy. Input is `(RGB image, optional
depth panel, text prompt that lists current keypoints in [0, 1000] u/v and
optionally past trajectories)`; output is per-keypoint future u/v in [0, 1000]
which we map back to the dataset's [-1, 1] UV space for metric comparison.

CLI mirrors lerobot_predict_trace_mu0_image_only.py for the bits a VLM
baseline can actually express:

  --use_depth_from={none,pred,data}    none = RGB+text only,
                                       pred = Depth-Anything-V2 -> turbo panel,
                                       data = TraceDataset depth -> turbo panel.
  --no_provide_history={true,false}    when false, include each KP's past UV
                                       trajectory as text in the prompt.
  --n_min / --n_max                    KP sampling bounds passed to TraceDataset.
  --num_samples_for_metrics            Best-of-N: N independent calls at
                                       temperature>0 to compute minADE_N / minFDE_N.

We always use dataset-supplied keypoints (no GSAM) — without a policy to
compare against on synthesized KPs, the baseline has no anchor. Depth (z) is
not asked of GPT; we report UV-only metrics. Inside the metric helpers we
set `pred_z := gt_z` so `ade_uvz == ade_uv` and `ade_depth == 0`, which makes
the `_uv` series usable as the headline number.

Example:

    python src/lerobot/scripts/lerobot_predict_trace_gpt.py \\
        --test_dirs='[outputs/test_dataset_droid/*]' \\
        --output_dir=outputs/predict/gpt_droid_baseline \\
        --first_frame_only=true \\
        --n_min=16 --n_max=64 \\
        --history_len=8 --future_len=32 \\
        --use_depth_from=none --no_provide_history=true \\
        --num_samples_for_metrics=5 --seed=0
"""

import base64
import io
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

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
from lerobot.policies.mu0.visualize_trace import render_trace_motion_still
from lerobot.utils.utils import init_logging


_DEFAULT_DA_V2_CKPT = "infer_helpers/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"


@dataclass
class TracePredictGPTConfig:
    test_dirs: list[str] = field(default_factory=list)
    output_dir: Path | None = None

    # Required for any UV/metric setup
    history_len: int = 8
    future_len: int = 32
    n_min: int = 16
    n_max: int = 64
    image_size: int = 512
    min_future_motion: float = 0.0
    seed: int = 0

    # Dataloading
    batch_size: int = 1  # GPT calls are sample-serial; keep batch tiny.
    num_workers: int = 0
    max_samples: int = 0
    shuffle: bool = False
    first_frame_only: bool = False
    # Skip the first N usable slots of each video when --first_frame_only=true
    # so that `valid_history` has real (non-zero-padded) entries to fill the
    # text-history block. Videos with fewer than offset+1 usable slots are
    # dropped with a warning. Accepts a single int (legacy) or a list of ints;
    # with a list the chosen slot for each offset is unioned (deduped) into one
    # bigger sample set and metrics are computed once over the union — i.e.
    # `--first_frame_offset=[4,8,12]` gives up to 3 samples per video and the
    # reported metrics are the sample-weighted average across all three offsets.
    first_frame_offset: int | list[int] = 0

    # ---- Baseline knobs (mirror predict_trace_smolvla_image_only) ----
    # none — no depth panel sent to GPT.
    # pred — Depth-Anything-V2 turbo panel (requires --delta_stats_path).
    # data — TraceDataset depth (requires --delta_stats_path).
    use_depth_from: str = "none"
    no_provide_history: bool = True
    num_samples_for_metrics: int = 5
    # Extra horizons at which to compute short-term metrics (in addition to the
    # full --future_len, which always lands in metrics.json). For each H listed,
    # we recompute l1_uv / ade_uv / fde_uv / dtw_uv / frechet_uv /
    # minADE_N_uv / minFDE_N_uv / minDTW_N_uv / minFrechet_N_uv on
    # `gt[:, :, :H]` vs `pred[:, :, :H]` and write the result to
    # `metrics_{H}.json`. Horizons > future_len are dropped with a warning.
    metric_horizons: list[int] = field(default_factory=lambda: [8, 16, 32])

    # Targeted query
    query_video: Path | None = None
    query_frames: list[int] = field(default_factory=list)

    # ---- GPT ----
    gpt_model: str = "gpt-5.5"
    gpt_temperature: float = 1.0
    gpt_top_p: float = 0.95
    gpt_max_output_tokens: int = 65536
    gpt_request_timeout_s: float = 180.0
    gpt_max_retries: int = 3
    gpt_retry_backoff_s: float = 4.0
    # OpenAI's reasoning-effort knob for o-series / GPT-5.x models. Use "none"
    # to omit the param entirely (default for non-reasoning models). Accepted
    # otherwise: minimal/low/medium/high — same semantics as Gemini's
    # thinking_level mapping for cross-baseline comparability.
    gpt_reasoning_effort: str = "low"
    # Some OpenAI models reject `temperature`/`top_p` (e.g. o1/o3 reasoning
    # models). Set these flags to skip sending the corresponding param.
    gpt_send_temperature: bool = True
    gpt_send_top_p: bool = True
    # Override the OpenAI HTTP endpoint (e.g., Azure or proxy). Empty = SDK
    # default (https://api.openai.com/v1).
    gpt_base_url: str = ""
    env_file: Path = Path(".env")  # OPENAI_API_KEY=...
    save_raw_responses: bool = True
    # Replay-only mode: reuse cached `raw_responses/*.json` and NEVER call the
    # API. The client is not even created (no key needed), and any missing or
    # unparseable cached response raises instead of silently re-billing. Use to
    # re-visualize an existing run without re-paying for inference.
    offline: bool = False

    # ---- Optional depth requirements ----
    delta_stats_path: Path | None = None
    depth_align: str = "log_percentile"
    depth_anything_ckpt: Path = Path(_DEFAULT_DA_V2_CKPT)
    depth_anything_encoder: str = "vitl"
    depth_anything_input_size: int = 518

    # ---- Saving ----
    save_depth_panels: bool = False
    save_annotated_inputs: bool = True

    def __post_init__(self) -> None:
        if self.use_depth_from not in ("none", "pred", "data"):
            raise ValueError(
                f"--use_depth_from must be none/pred/data; got {self.use_depth_from!r}"
            )
        # --use_depth_from=data uses the dataset's precomputed depth, which
        # requires depth_log_range from delta_stats. --use_depth_from=pred can
        # fall back to per-image percentile rendering when delta_stats is
        # absent — the VLM was not trained on a specific log range, so as long
        # as the panel is visually informative, alignment is optional.
        if self.use_depth_from == "data" and self.delta_stats_path is None:
            raise ValueError(
                "--delta_stats_path is required when --use_depth_from=data "
                "(the TraceDataset needs depth_log_range to render depth.npy)."
            )
        if self.n_min < 1 or self.n_max < self.n_min:
            raise ValueError(f"need n_min>=1 and n_max>=n_min: {self.n_min}, {self.n_max}")
        if self.history_len < 1:
            raise ValueError(f"--history_len must be >= 1: {self.history_len}")
        if self.future_len < 1:
            raise ValueError(f"--future_len must be >= 1: {self.future_len}")
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
        if self.gpt_reasoning_effort.lower() not in (
            "none", "minimal", "low", "medium", "high",
        ):
            raise ValueError(
                f"--gpt_reasoning_effort must be none/minimal/low/medium/high; "
                f"got {self.gpt_reasoning_effort!r}"
            )
        if self.query_video is not None:
            if not self.query_frames:
                raise ValueError("--query_video set but --query_frames is empty.")
            if not Path(self.query_video).exists():
                raise FileNotFoundError(f"--query_video not found: {self.query_video}")
            if not self.test_dirs:
                self.test_dirs = [str(self.query_video)]
        elif not self.test_dirs:
            raise ValueError(
                "test_dirs is empty — pass at least one TraceForge video dir or glob."
            )


# ---------------------------------------------------------------------------
# Meta dataset wrapper (carry video_dir + frame_idx through the DataLoader)
# ---------------------------------------------------------------------------

_META_DIR_KEY = "__video_dir"
_META_FRAME_KEY = "__frame_idx"


class _MetaTraceDataset(Dataset):
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

    `offset=0` (default) keeps the earliest slot per video (legacy behavior).
    `offset>0` lets the chosen slot accumulate real history before the prompt
    is built. Videos with fewer than `offset+1` usable slots are skipped.
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


def _resolve_query_indices(base: TraceDataset, video_dir: str, frames: list[int]) -> list[int]:
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
            f"Make sure --test_dirs covers it. First few: {avail}"
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
        logging.warning(
            f"--query_video {video_dir}: skipping frames {missing} (no usable slot)."
        )
    if not chosen:
        raise ValueError(
            f"--query_video {video_dir}: none of {frames} are usable."
        )
    return chosen


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _image_chw_to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.permute(1, 2, 0).float().cpu().numpy()
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _uv_norm_to_thousand(uv_norm: np.ndarray) -> np.ndarray:
    """[-1, 1]^2 -> [0, 1000]^2 ints (u=col=x, v=row=y)."""
    arr = (uv_norm + 1.0) * 500.0
    return np.clip(np.round(arr), 0, 1000).astype(np.int32)


def _thousand_to_uv_norm(uv_thou: np.ndarray) -> np.ndarray:
    """[0, 1000]^2 -> [-1, 1]^2 floats."""
    return (uv_thou.astype(np.float32) / 500.0) - 1.0


def _annotate_image_with_kp(
    image_uint8: np.ndarray,
    current_kp_norm: np.ndarray,  # (N, 2/3) in [-1, 1]
    key_pad_mask: np.ndarray,     # (N,) bool
) -> np.ndarray:
    """Draw small numbered circles at each valid current keypoint.

    Numbers match the `id` field GPT is told to use in its JSON output, so
    the model can visually anchor each track. We use cv2 with a colored ring
    + black text for legibility against any background.
    """
    canvas = image_uint8.copy()
    h, w = canvas.shape[:2]
    n = current_kp_norm.shape[0]
    # Bright magenta ring (255, 0, 255) — distinct from history cyan, GT green, pred red
    ring_rgb = (255, 0, 255)
    text_rgb = (255, 255, 255)
    text_outline = (0, 0, 0)
    radius = max(4, int(round(min(h, w) / 96)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.32, min(h, w) / 1024.0)
    thick = max(1, int(round(scale * 2)))
    for i in range(n):
        if not bool(key_pad_mask[i]):
            continue
        u, v = current_kp_norm[i, 0], current_kp_norm[i, 1]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        cx = int(round((u + 1.0) * 0.5 * (w - 1)))
        cy = int(round((v + 1.0) * 0.5 * (h - 1)))
        cv2.circle(canvas, (cx, cy), radius, ring_rgb, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 1, ring_rgb, thickness=-1, lineType=cv2.LINE_AA)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        tx = cx + radius + 2
        ty = cy + th // 2
        if tx + tw > w - 2:
            tx = cx - radius - tw - 2
        ty = max(th + 1, min(ty, h - 2))
        cv2.putText(canvas, label, (tx, ty), font, scale, text_outline,
                    thickness=thick + 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, label, (tx, ty), font, scale, text_rgb,
                    thickness=thick, lineType=cv2.LINE_AA)
    return canvas


def _np_to_pil(rgb_uint8: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb_uint8)


def _pil_to_b64_data_url(img: Image.Image) -> str:
    """PNG-encode and base64 a PIL image for OpenAI's `image_url` content part."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA_TEMPLATE = (
    'Output strictly valid JSON in this format with no commentary:\n'
    '{\n'
    '  "trajectories": [\n'
    '    {"id": <int>, "future": [[u_1, v_1], [u_2, v_2], ...]},\n'
    '    ...\n'
    '  ]\n'
    '}\n'
    'where each u, v is an INTEGER in [0, 1000]; (0,0) = top-left pixel, '
    '(1000, 1000) = bottom-right; u is the horizontal (column) axis and v is '
    'the vertical (row) axis. Predict EXACTLY __FUTURE_LEN__ future timesteps for '
    'EVERY listed keypoint id, in chronological order (index 0 is the next '
    'frame after the current image). Do not skip ids; do not invent extra ids.'
)


def _output_schema_hint(future_len: int) -> str:
    return _OUTPUT_SCHEMA_TEMPLATE.replace("__FUTURE_LEN__", str(future_len))


def _format_history_block(
    traj_history_norm: np.ndarray,   # (N, H, 3) in [-1, 1] uv + meters z
    valid_history: np.ndarray,        # (N, H) bool
    key_pad_mask: np.ndarray,         # (N,) bool
    history_len: int,
) -> str:
    """Render per-KP history as text in [0, 1000] u/v.

    Chronological order: index 0 = oldest (t - H + 1), index H-1 = t - 1.
    Invalid steps are skipped from the listed series.
    """
    lines = ["Past trajectories (chronological; index 0 is oldest, the last is one frame before current):"]
    n = traj_history_norm.shape[0]
    for i in range(n):
        if not bool(key_pad_mask[i]):
            continue
        thou = _uv_norm_to_thousand(traj_history_norm[i, :, :2])  # (H, 2)
        vis: list[str] = []
        for h in range(history_len):
            if not bool(valid_history[i, h]):
                continue
            vis.append(f"[{int(thou[h, 0])},{int(thou[h, 1])}]")
        if not vis:
            lines.append(f"  id={i}: <no valid past frames>")
        else:
            lines.append(f"  id={i}: {','.join(vis)}")
    return "\n".join(lines)


def _format_current_kp_block(
    current_kp_norm: np.ndarray,      # (N, 3) in [-1, 1] uv + raw z
    key_pad_mask: np.ndarray,         # (N,) bool
) -> str:
    """Per-KP current-frame [u, v] in [0, 1000]."""
    lines = ["Current keypoints to track (id : [u, v] in [0,1000]):"]
    n = current_kp_norm.shape[0]
    thou = _uv_norm_to_thousand(current_kp_norm[:, :2])  # (N, 2)
    for i in range(n):
        if not bool(key_pad_mask[i]):
            continue
        lines.append(f"  id={i}: [{int(thou[i, 0])},{int(thou[i, 1])}]")
    return "\n".join(lines)


def _build_prompt(
    task: str,
    current_kp_norm: np.ndarray,
    key_pad_mask: np.ndarray,
    traj_history_norm: np.ndarray | None,
    valid_history: np.ndarray | None,
    history_len: int,
    future_len: int,
    has_depth: bool,
) -> str:
    parts: list[str] = []
    intro = (
        "You are predicting how a set of marked keypoints will move in the "
        "near future. The first image shows the current scene with each "
        "tracked keypoint annotated as a magenta ring and a numeric id."
    )
    if has_depth:
        intro += (
            " The second image is a depth visualization of the same scene "
            "(turbo colormap; closer = warmer colors)."
        )
    parts.append(intro)
    parts.append(f"Task being performed: {task.strip() or 'predict the future motion of the hand/manipulator'}.")
    parts.append(_format_current_kp_block(current_kp_norm, key_pad_mask))
    if traj_history_norm is not None and valid_history is not None:
        parts.append(_format_history_block(
            traj_history_norm, valid_history, key_pad_mask, history_len,
        ))
    parts.append(_output_schema_hint(future_len))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GPT client + JSON parsing
# ---------------------------------------------------------------------------

def _load_env_file(env_path: Path) -> None:
    """Read .env and inject KEY=value pairs into os.environ (only if absent)."""
    if not env_path.exists():
        return
    with open(env_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


class _GPTClient:
    def __init__(self, cfg: TracePredictGPTConfig) -> None:
        from openai import OpenAI
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
            UnprocessableEntityError,
        )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment. "
                f"Did you populate {cfg.env_file}?"
            )
        client_kwargs: dict[str, Any] = dict(
            api_key=api_key,
            timeout=cfg.gpt_request_timeout_s,
        )
        if cfg.gpt_base_url:
            client_kwargs["base_url"] = cfg.gpt_base_url
        self._client = OpenAI(**client_kwargs)
        self._cfg = cfg
        # Deterministic-failure classes: retrying with the same payload won't
        # help (bad request shape, wrong key, wrong endpoint, missing model,
        # rejected content) — surface immediately instead of burning the
        # retry budget.
        self._deterministic_classes: tuple[type, ...] = (
            BadRequestError,
            AuthenticationError,
            PermissionDeniedError,
            NotFoundError,
            UnprocessableEntityError,
        )
        self._rate_limit_cls = RateLimitError
        self._status_error_cls = APIStatusError
        self._timeout_error_cls = APITimeoutError
        self._connection_error_cls = APIConnectionError

    def generate_json(
        self,
        images_pil: list[Image.Image],
        prompt: str,
        seed: int,
    ) -> tuple[str, dict]:
        """One call to GPT; returns (raw_text, parsed_dict).

        Retries transient (5xx, network, timeout) failures with exponential
        backoff. Deterministic errors (400/401/403/404/422) raise immediately
        — retrying the same payload against the same endpoint with the same
        key wouldn't help and would just burn the retry budget.
        """
        cfg = self._cfg
        last_exc: Exception | None = None

        content_parts: list[dict[str, Any]] = []
        for img in images_pil:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": _pil_to_b64_data_url(img)},
            })
        content_parts.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_parts}]

        for attempt in range(cfg.gpt_max_retries):
            try:
                call_kwargs: dict[str, Any] = dict(
                    model=cfg.gpt_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    seed=seed,
                    max_completion_tokens=cfg.gpt_max_output_tokens,
                )
                if cfg.gpt_send_temperature:
                    call_kwargs["temperature"] = cfg.gpt_temperature
                if cfg.gpt_send_top_p:
                    call_kwargs["top_p"] = cfg.gpt_top_p
                if cfg.gpt_reasoning_effort.lower() != "none":
                    call_kwargs["reasoning_effort"] = cfg.gpt_reasoning_effort.lower()
                resp = self._client.chat.completions.create(**call_kwargs)
                choices = getattr(resp, "choices", None) or []
                text = ""
                if choices:
                    msg = getattr(choices[0], "message", None)
                    text = (getattr(msg, "content", None) or "") if msg is not None else ""
                parsed = _parse_vlm_json(text)
                return text, parsed
            except self._deterministic_classes:
                # 400/401/403/404/422 are deterministic — bad request shape,
                # wrong API key, wrong endpoint, missing model, or rejected
                # content. Surface immediately instead of burning the retry
                # budget on a payload that can't possibly succeed.
                raise
            except self._rate_limit_cls as exc:
                # 429 RateLimitError — respect server's suggested retry window
                # when it's longer than our retry budget can absorb (e.g.
                # daily-quota caps). Fail fast rather than burning attempts
                # before the same failure recurs.
                delay_s = _parse_retry_delay_seconds(str(exc))
                if delay_s is not None and delay_s > 60.0:
                    logging.error(
                        f"GPT quota exhausted with retryDelay={delay_s:.0f}s "
                        f"(~{delay_s / 60:.1f} min). Failing fast instead of "
                        f"retrying — wait for quota reset or lower "
                        f"--num_samples_for_metrics / --max_samples."
                    )
                    raise
                last_exc = exc
                sleep_s = cfg.gpt_retry_backoff_s * (2 ** attempt)
                logging.warning(
                    f"GPT call failed (attempt {attempt + 1}/{cfg.gpt_max_retries}): "
                    f"{type(exc).__name__}: {exc!s} — sleeping {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
            except self._status_error_cls as exc:
                # 5xx and other non-400 status errors — retry with backoff.
                status_code = int(getattr(exc, "status_code", 0) or 0)
                if status_code == 400:
                    raise
                last_exc = exc
                sleep_s = cfg.gpt_retry_backoff_s * (2 ** attempt)
                logging.warning(
                    f"GPT call failed (attempt {attempt + 1}/{cfg.gpt_max_retries}): "
                    f"{type(exc).__name__}: {exc!s} — sleeping {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
            except (self._timeout_error_cls, self._connection_error_cls) as exc:
                last_exc = exc
                sleep_s = cfg.gpt_retry_backoff_s * (2 ** attempt)
                logging.warning(
                    f"GPT call failed (attempt {attempt + 1}/{cfg.gpt_max_retries}): "
                    f"{type(exc).__name__}: {exc!s} — sleeping {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                sleep_s = cfg.gpt_retry_backoff_s * (2 ** attempt)
                logging.warning(
                    f"GPT call failed (attempt {attempt + 1}/{cfg.gpt_max_retries}): "
                    f"{type(exc).__name__}: {exc!s} — sleeping {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"'retryDelay'\s*:\s*'(\d+(?:\.\d+)?)s'")
_RETRY_IN_RE = re.compile(r"retry in\s+(\d+)h(\d+)m(\d+(?:\.\d+)?)s")
_RETRY_AFTER_RE = re.compile(r"retry[\s-]*after[:\s]*(\d+(?:\.\d+)?)", re.IGNORECASE)
_PLEASE_TRY_AGAIN_RE = re.compile(r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s|m)", re.IGNORECASE)


def _parse_retry_delay_seconds(err_text: str) -> float | None:
    """Extract the server's suggested retry delay from a 429 error message.

    Matches several forms (kept compatible with the Gemini variants for any
    proxy that reuses the format, plus OpenAI-style `Retry-After` and
    `try again in Xs` prose):
      - structured `'retryDelay': '10055s'`
      - prose `Please retry in 2h47m35.7s`
      - `Retry-After: 30`
      - `Please try again in 500ms`
    Returns the value in seconds, or None if no pattern matches.
    """
    m = _RETRY_DELAY_RE.search(err_text)
    if m:
        return float(m.group(1))
    m = _RETRY_IN_RE.search(err_text)
    if m:
        h, mn, s = m.groups()
        return float(h) * 3600 + float(mn) * 60 + float(s)
    m = _RETRY_AFTER_RE.search(err_text)
    if m:
        return float(m.group(1))
    m = _PLEASE_TRY_AGAIN_RE.search(err_text)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "ms":
            return val / 1000.0
        if unit == "m":
            return val * 60.0
        return val
    return None


def _parse_vlm_json(text: str) -> dict:
    """Tolerant JSON parse: strips markdown fences and trailing commas."""
    if not text or not text.strip():
        raise ValueError("Empty GPT response.")
    candidate = text.strip()
    # Strip markdown fence if present.
    m = _JSON_FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    # Find the first { and the last } — models occasionally prepend prose.
    lo = candidate.find("{")
    hi = candidate.rfind("}")
    if lo >= 0 and hi > lo:
        candidate = candidate[lo : hi + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        # Tolerate trailing commas before } or ].
        cleaned = re.sub(r",\s*(\}|\])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Final fallback: response was probably truncated mid-array.
            # Walk the original text and recover any complete `{...}` entries
            # inside the trajectories array. Returns `{"trajectories": [...]}`
            # with the entries we could parse; missing kps will be filled by
            # the stationary-fallback in `_parsed_to_pred_uv_thou`.
            salvaged = _salvage_truncated_trajectories(text)
            if salvaged is not None:
                logging.warning(
                    f"Salvaged {len(salvaged.get('trajectories', []))} complete "
                    "trajectories from a truncated GPT response."
                )
                return salvaged
            raise ValueError(
                f"Could not parse GPT JSON. First 400 chars:\n{text[:400]}"
            ) from exc


def _salvage_truncated_trajectories(text: str) -> dict | None:
    """Walk a (possibly-truncated) response and extract complete entries.

    Locates the `"trajectories": [` array opening, then iterates characters
    tracking string state and brace depth, attempting to parse every closed
    `{...}` at depth 1. Returns `{"trajectories": [entries...]}` (possibly
    empty) or `None` if the trajectories key isn't present at all. Trailing
    commas and quoted special chars inside step strings are handled.
    """
    m = re.search(r'"trajectories"\s*:\s*\[', text)
    if not m:
        return None
    entries: list[dict] = []
    depth = 0
    in_string = False
    escape = False
    obj_start: int | None = None
    for i in range(m.end(), len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                chunk = text[obj_start : i + 1]
                try:
                    entries.append(json.loads(chunk))
                except json.JSONDecodeError:
                    # Try with trailing-comma scrub.
                    try:
                        entries.append(
                            json.loads(re.sub(r",\s*(\}|\])", r"\1", chunk))
                        )
                    except json.JSONDecodeError:
                        pass
                obj_start = None
            elif depth < 0:
                # Stray closing brace — we've fallen off the trajectories array.
                break
    return {"trajectories": entries}


def _parsed_to_pred_uv_thou(
    parsed: dict,
    num_kp: int,
    future_len: int,
    key_pad_mask: np.ndarray,
    current_kp_norm: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Convert GPT JSON -> (num_kp, future_len, 2) in [0, 1000].

    Missing or malformed keypoint trajectories fall back to "stationary" (the
    current uv repeated `future_len` times). Returns the array and a list of
    per-KP statuses for logging.
    """
    out = np.zeros((num_kp, future_len, 2), dtype=np.float32)
    statuses = ["missing"] * num_kp

    # Default: stationary at current uv for valid kps; (0,0) for padded.
    cur_thou = _uv_norm_to_thousand(current_kp_norm[:, :2])
    for i in range(num_kp):
        if bool(key_pad_mask[i]):
            out[i, :, 0] = float(cur_thou[i, 0])
            out[i, :, 1] = float(cur_thou[i, 1])

    trajs = parsed.get("trajectories")
    if not isinstance(trajs, list):
        return out, statuses

    for entry in trajs:
        if not isinstance(entry, dict):
            continue
        try:
            kp_id = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if not (0 <= kp_id < num_kp) or not bool(key_pad_mask[kp_id]):
            continue
        fut = entry.get("future")
        if not isinstance(fut, list) or len(fut) == 0:
            statuses[kp_id] = "empty_future"
            continue
        # Coerce each step to (u, v) ints in [0, 1000].
        parsed_steps: list[tuple[float, float]] = []
        for step in fut:
            if not isinstance(step, (list, tuple)) or len(step) < 2:
                continue
            try:
                u = float(step[0])
                v = float(step[1])
            except (TypeError, ValueError):
                continue
            u = max(0.0, min(1000.0, u))
            v = max(0.0, min(1000.0, v))
            parsed_steps.append((u, v))
        if not parsed_steps:
            statuses[kp_id] = "bad_steps"
            continue
        # Pad/truncate to future_len.
        if len(parsed_steps) < future_len:
            last = parsed_steps[-1]
            parsed_steps = parsed_steps + [last] * (future_len - len(parsed_steps))
            statuses[kp_id] = f"padded_to_{future_len}"
        elif len(parsed_steps) > future_len:
            parsed_steps = parsed_steps[:future_len]
            statuses[kp_id] = f"truncated_to_{future_len}"
        else:
            statuses[kp_id] = "ok"
        for t, (u, v) in enumerate(parsed_steps):
            out[kp_id, t, 0] = u
            out[kp_id, t, 1] = v

    return out, statuses


# ---------------------------------------------------------------------------
# Depth helpers (only used when --use_depth_from != none)
# ---------------------------------------------------------------------------

def _render_depth_panel_chw(
    depth_meters: np.ndarray, log_min: float, log_max: float, image_size: int
) -> torch.Tensor:
    from lerobot.datasets.trace_depth import render_depth_rgb

    rgb = render_depth_rgb(depth_meters, log_min, log_max)
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    if t.shape[-2] != image_size or t.shape[-1] != image_size:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False,
        ).squeeze(0)
    return t


def _min_dtw_uv_sum(
    pred_abs_S: torch.Tensor,   # (B, S, N, F, 3) — only [..., :2] is read
    gt_abs: torch.Tensor,        # (B, N, F, 3)
    valid_future: torch.Tensor,  # (B, N, F)
    kpm: torch.Tensor,           # (B, N)
) -> tuple[float, int]:
    """Best-of-S minDTW on the (u, v) dims. Returns (sum, n_traj).

    Per (b, kp): compress to its T valid future steps, run the standard DTW
    DP on each of the S predicted trajectories against the single GT, then
    take the min across S. DP is vectorized across S (shape `(S, T+1, T+1)`)
    to keep Python overhead bounded. Length-normalized by `T` so different
    trajectories are comparable; consistent with `_dtw_sums` in the train
    script. UV-only — depth is set to gt_z elsewhere in this script, so a
    `_min_dtw_uvz` would degenerate to `_min_dtw_uv`.
    """
    B, S, N, _F, _ = pred_abs_S.shape
    valid_t = (valid_future & kpm.unsqueeze(-1)).cpu().numpy()  # (B, N, F)
    pred_np = pred_abs_S[..., :2].detach().to(torch.float32).cpu().numpy()  # (B, S, N, F, 2)
    gt_np = gt_abs[..., :2].detach().to(torch.float32).cpu().numpy()        # (B, N, F, 2)

    total = 0.0
    n_traj = 0
    for b in range(B):
        for n in range(N):
            m = valid_t[b, n]
            t_valid = int(m.sum())
            if t_valid == 0:
                continue
            gt_traj = gt_np[b, n][m]                 # (T, 2)
            pred_S = pred_np[b, :, n][:, m]          # (S, T, 2)
            if t_valid == 1:
                d_per_s = np.linalg.norm(pred_S[:, 0] - gt_traj[0], axis=-1)  # (S,)
                total += float(d_per_s.min())
                n_traj += 1
                continue
            diff = pred_S[:, :, None, :] - gt_traj[None, None, :, :]   # (S, T, T, 2)
            pdist = np.linalg.norm(diff, axis=-1).astype(np.float64)    # (S, T, T)
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
            per_s_dtw = cost[:, t_valid, t_valid] / float(t_valid)   # (S,)
            total += float(per_s_dtw.min())
            n_traj += 1
    return total, n_traj


def _frechet_uv_sum(
    pred_abs: torch.Tensor,      # (B, N, F, 3) — only [..., :2] is read
    gt_abs: torch.Tensor,         # (B, N, F, 3)
    valid_future: torch.Tensor,  # (B, N, F)
    kpm: torch.Tensor,           # (B, N)
) -> tuple[float, int]:
    """Single-sample Fréchet on (u, v) dims. Returns (sum, n_traj).

    Sample-0 counterpart to `_min_frechet_uv_sum` — same Eiter & Mannila
    DP, just one prediction per (b, kp) instead of S. Mirrors the
    sum/count contract of the train script's `_dtw_sums` so the caller
    can mean across batches with the right denominator. UV-only — depth
    is forced to `gt_z` elsewhere, so a `_frechet_uvz` would degenerate
    to `_frechet_uv`.
    """
    B, N, _F, _ = pred_abs.shape
    valid_t = (valid_future & kpm.unsqueeze(-1)).cpu().numpy()  # (B, N, F)
    pred_np = pred_abs[..., :2].detach().to(torch.float32).cpu().numpy()
    gt_np = gt_abs[..., :2].detach().to(torch.float32).cpu().numpy()

    total = 0.0
    n_traj = 0
    for b in range(B):
        for n in range(N):
            m = valid_t[b, n]
            t_valid = int(m.sum())
            if t_valid == 0:
                continue
            gt_traj = gt_np[b, n][m]      # (T, 2)
            pred_traj = pred_np[b, n][m]  # (T, 2)
            if t_valid == 1:
                total += float(np.linalg.norm(pred_traj[0] - gt_traj[0]))
                n_traj += 1
                continue
            diff = pred_traj[:, None, :] - gt_traj[None, :, :]       # (T, T, 2)
            pdist = np.linalg.norm(diff, axis=-1).astype(np.float64)  # (T, T)
            cost = np.empty((t_valid, t_valid), dtype=np.float64)
            cost[0, 0] = pdist[0, 0]
            for i in range(1, t_valid):
                cost[i, 0] = max(cost[i - 1, 0], pdist[i, 0])
            for j in range(1, t_valid):
                cost[0, j] = max(cost[0, j - 1], pdist[0, j])
            for i in range(1, t_valid):
                for j in range(1, t_valid):
                    best_prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
                    cost[i, j] = max(pdist[i, j], best_prev)
            total += float(cost[t_valid - 1, t_valid - 1])
            n_traj += 1
    return total, n_traj


def _min_frechet_uv_sum(
    pred_abs_S: torch.Tensor,   # (B, S, N, F, 3) — only [..., :2] is read
    gt_abs: torch.Tensor,        # (B, N, F, 3)
    valid_future: torch.Tensor,  # (B, N, F)
    kpm: torch.Tensor,           # (B, N)
) -> tuple[float, int]:
    """Best-of-S minFréchet on the (u, v) dims. Returns (sum, n_traj).

    Discrete Fréchet distance: the min over monotonic alignments of the MAX
    pairwise distance — the "tightest leash" bottleneck along the optimal
    coupling. DP recurrence:

        c(i, j) = max( d(P_i, Q_j),
                       min(c(i-1, j), c(i, j-1), c(i-1, j-1)) )

    Per (b, kp): compress to T valid future steps, run the Fréchet DP on
    each of the S predicted trajectories against the single GT, take min
    across S. DP vectorized across S (shape `(S, T, T)`). Fréchet is a max
    so no length normalization (unlike DTW). UV-only — depth is forced to
    `gt_z` elsewhere in this script, so a `_min_frechet_uvz` would
    degenerate to `_min_frechet_uv`.
    """
    B, S, N, _F, _ = pred_abs_S.shape
    valid_t = (valid_future & kpm.unsqueeze(-1)).cpu().numpy()  # (B, N, F)
    pred_np = pred_abs_S[..., :2].detach().to(torch.float32).cpu().numpy()
    gt_np = gt_abs[..., :2].detach().to(torch.float32).cpu().numpy()

    total = 0.0
    n_traj = 0
    for b in range(B):
        for n in range(N):
            m = valid_t[b, n]
            t_valid = int(m.sum())
            if t_valid == 0:
                continue
            gt_traj = gt_np[b, n][m]                 # (T, 2)
            pred_S = pred_np[b, :, n][:, m]          # (S, T, 2)
            if t_valid == 1:
                d_per_s = np.linalg.norm(pred_S[:, 0] - gt_traj[0], axis=-1)
                total += float(d_per_s.min())
                n_traj += 1
                continue
            diff = pred_S[:, :, None, :] - gt_traj[None, None, :, :]   # (S, T, T, 2)
            pdist = np.linalg.norm(diff, axis=-1).astype(np.float64)    # (S, T, T)
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
            per_s_frechet = cost[:, t_valid - 1, t_valid - 1]          # (S,)
            total += float(per_s_frechet.min())
            n_traj += 1
    return total, n_traj


def _per_image_log_bounds(
    depth_meters: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0,
) -> tuple[float, float]:
    """1st/99th log-percentile of valid (finite, positive) depth pixels.

    Falls back to (log(0.1), log(10.0)) when no valid pixels — yields a
    "near=warm, far=cool" panel rather than crashing the render.
    """
    valid = np.isfinite(depth_meters) & (depth_meters > 1e-3)
    if not valid.any():
        return float(np.log(0.1)), float(np.log(10.0))
    log_d = np.log(np.clip(depth_meters[valid], 1e-3, None))
    p_lo, p_hi = np.percentile(log_d, [lo_pct, hi_pct])
    # Guard against degenerate flat depth.
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi - p_lo < 1e-3:
        return float(np.log(0.1)), float(np.log(10.0))
    return float(p_lo), float(p_hi)


# ---------------------------------------------------------------------------
# Main predict entry
# ---------------------------------------------------------------------------

@draccus.wrap()
def predict(cfg: TracePredictGPTConfig) -> None:
    init_logging()

    _load_env_file(cfg.env_file)
    client = None if cfg.offline else _GPTClient(cfg)
    if cfg.offline:
        logging.info("--offline=true: replaying cached raw_responses; the API client is NOT created.")
    logging.info(f"Using GPT model: {cfg.gpt_model} (temperature={cfg.gpt_temperature})")
    logging.info(
        f"  use_depth_from={cfg.use_depth_from} no_provide_history={cfg.no_provide_history} "
        f"num_samples_for_metrics={cfg.num_samples_for_metrics}"
    )

    use_depth_from_pred = cfg.use_depth_from == "pred"
    use_depth_from_data = cfg.use_depth_from == "data"

    depth_log_range: tuple[float, float] | None = None
    if cfg.use_depth_from != "none" and cfg.delta_stats_path is not None:
        if not Path(cfg.delta_stats_path).exists():
            raise FileNotFoundError(f"delta_stats_path not found: {cfg.delta_stats_path}")
        trace_stats = load_trace_stats(cfg.delta_stats_path)
        depth_log_range = trace_stats.depth_log_range
        if depth_log_range is None and cfg.use_depth_from == "data":
            raise ValueError(
                f"{cfg.delta_stats_path} has no depth_log_* range; required for "
                f"--use_depth_from=data."
            )
        logging.info(f"  depth_log_range={depth_log_range}")
    elif cfg.use_depth_from == "pred":
        logging.info(
            "  no --delta_stats_path provided; will render DA-V2 depth with "
            "per-image log-percentile bounds (1st/99th)."
        )

    # Load dataset with delta_scale=None so traj_future / traj_history come out
    # in ABSOLUTE [-1, 1] UV (no anchor-relative delta, no step-by-step diff).
    dataset = TraceDataset(
        video_dirs=cfg.test_dirs,
        future_len=cfg.future_len,
        history_len=cfg.history_len,
        n_min=cfg.n_min,
        n_max=cfg.n_max,
        image_size=cfg.image_size,
        seed=cfg.seed,
        min_future_motion=cfg.min_future_motion,
        delta_scale=None,
        load_depth=use_depth_from_data,
        depth_log_range=depth_log_range if use_depth_from_data else None,
        center_history_on_current=False,
        step_by_step_delta=False,
    )
    logging.info(f"TraceDataset[test]: {len(dataset)} samples across {len(dataset._videos)} dirs")

    meta_dataset: Dataset = _MetaTraceDataset(dataset)
    loader_shuffle = cfg.shuffle
    if cfg.query_video is not None:
        chosen = _resolve_query_indices(dataset, str(cfg.query_video), cfg.query_frames)
        meta_dataset = Subset(meta_dataset, chosen)
        loader_shuffle = False
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
        pin_memory=False,
        drop_last=False,
    )

    da_v2 = None
    if use_depth_from_pred:
        from lerobot.inference_helpers.depth_anything import DepthAnythingV2Wrapper

        logging.info(f"Loading Depth-Anything-V2 from {cfg.depth_anything_ckpt}")
        da_v2 = DepthAnythingV2Wrapper(
            checkpoint_path=cfg.depth_anything_ckpt,
            encoder=cfg.depth_anything_encoder,
            input_size=cfg.depth_anything_input_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path("outputs/predict/gpt_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollout_img").mkdir(parents=True, exist_ok=True)
    if cfg.save_depth_panels and cfg.use_depth_from != "none":
        (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    if cfg.save_annotated_inputs:
        (output_dir / "annotated_inputs").mkdir(parents=True, exist_ok=True)
    if cfg.save_raw_responses:
        (output_dir / "raw_responses").mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving outputs to {output_dir}")

    # Lazy import of train-side metric helpers (also pull in accelerate).
    from lerobot.scripts.lerobot_train_trace_mu0 import (
        _accumulate_min_metrics,
        _accumulate_traj_metrics,
        _finalize_min_metrics,
        _finalize_traj_metrics,
    )

    saved = 0
    parse_log: list[dict] = []

    n_samples = cfg.num_samples_for_metrics
    logging.info(f"Best-of-N: n_samples={n_samples}")

    # Build the list of horizons to evaluate. Always includes the full
    # future_len (lands in metrics.json); other horizons land in
    # metrics_{H}.json. Horizons > future_len are dropped with a warning.
    requested_horizons = sorted(set(cfg.metric_horizons))
    dropped = [h for h in requested_horizons if h > cfg.future_len]
    if dropped:
        logging.warning(
            f"--metric_horizons: dropping {dropped} (> --future_len={cfg.future_len})"
        )
    horizons = sorted(set(
        [h for h in requested_horizons if h <= cfg.future_len] + [cfg.future_len]
    ))
    logging.info(f"Computing metrics at horizons: {horizons} (full = {cfg.future_len})")
    # Per-horizon accumulator state. Each horizon has its own set of sums so
    # the same per-batch tensors can be sliced and counted multiple ways
    # without re-running any API calls.
    state_per_h: dict[int, dict] = {
        h: {
            "l1_sum": 0.0,
            "l1_n": 0,
            "traj_acc": {},
            "min_acc": {},
            "min_dtw_sum": 0.0,
            "min_dtw_n": 0,
            "frechet_sum": 0.0,
            "frechet_n": 0,
            "min_frechet_sum": 0.0,
            "min_frechet_n": 0,
        }
        for h in horizons
    }

    total_samples = len(meta_dataset)  # type: ignore[arg-type]
    if cfg.max_samples > 0:
        total_samples = min(total_samples, cfg.max_samples)
    pbar = tqdm(
        total=total_samples,
        desc="gpt-predict",
        unit="sample",
        dynamic_ncols=True,
    )

    for batch_idx, raw in enumerate(dataloader):
        if cfg.max_samples > 0 and saved >= cfg.max_samples:
            break

        images = raw[IMAGE_KEY]  # (B, 3, H, W) float in [0, 1]
        current_kp = raw[CURRENT_KP_KEY]  # (B, N, 3) abs in [-1, 1] uv + raw z
        traj_history = raw[TRAJ_HISTORY_KEY]  # (B, N, H, 3)
        traj_future = raw[TRAJ_FUTURE_KEY]  # (B, N, F, 3) abs in [-1, 1] uv
        valid_history = raw[VALID_HISTORY_KEY]  # (B, N, H)
        valid_future = raw[VALID_FUTURE_KEY]  # (B, N, F)
        key_pad_mask = raw[KEY_PAD_MASK_KEY]  # (B, N)
        tasks = raw[TASK_KEY]  # list[str]
        depths = raw.get(DEPTH_KEY)  # (B, 3, H, W) or None

        meta_dirs = raw.get(_META_DIR_KEY, [""] * images.shape[0])
        meta_frames = raw.get(_META_FRAME_KEY, [-1] * images.shape[0])

        bsz = images.shape[0]
        num_kp = current_kp.shape[1]
        F = cfg.future_len

        # Optionally synthesize depth via DA-V2 per-image. When a training
        # log-range was supplied we align to it (matches what SmolVLA would
        # see); otherwise we render with per-image 1st/99th log-percentile
        # bounds so the panel is still informative to the VLM.
        depth_panels_chw: list[torch.Tensor | None] = [None] * bsz
        if use_depth_from_pred:
            assert da_v2 is not None
            for b in range(bsz):
                rgb = _image_chw_to_uint8_hwc(images[b])
                if depth_log_range is not None:
                    log_min_t, log_max_t = depth_log_range
                    d_aligned, _diag = da_v2.infer_aligned(
                        rgb, cfg.depth_align, log_min_t, log_max_t,
                    )
                    depth_panels_chw[b] = _render_depth_panel_chw(
                        d_aligned, log_min_t, log_max_t, cfg.image_size,
                    )
                else:
                    d_meters = da_v2.infer_meters(rgb)
                    log_min_b, log_max_b = _per_image_log_bounds(d_meters)
                    depth_panels_chw[b] = _render_depth_panel_chw(
                        d_meters, log_min_b, log_max_b, cfg.image_size,
                    )
        elif use_depth_from_data and depths is not None:
            for b in range(bsz):
                depth_panels_chw[b] = depths[b]

        # Loop over (sample, kp_query) — sequential, Best-of-N at temp>0.
        # preds_thou_S: (B, S, N, F, 2) in [0, 1000]
        preds_thou_S = np.zeros((bsz, n_samples, num_kp, F, 2), dtype=np.float32)

        for b in range(bsz):
            img_uint8 = _image_chw_to_uint8_hwc(images[b])
            kpm_b = key_pad_mask[b].cpu().numpy()
            current_kp_b = current_kp[b].cpu().numpy()
            annotated = _annotate_image_with_kp(img_uint8, current_kp_b, kpm_b)
            traj_history_b = traj_history[b].cpu().numpy() if not cfg.no_provide_history else None
            valid_history_b = valid_history[b].cpu().numpy() if not cfg.no_provide_history else None

            depth_pil = None
            if depth_panels_chw[b] is not None:
                d_arr = (
                    depth_panels_chw[b].permute(1, 2, 0).float().cpu().numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
                depth_pil = _np_to_pil(d_arr)

            prompt = _build_prompt(
                task=tasks[b],
                current_kp_norm=current_kp_b,
                key_pad_mask=kpm_b,
                traj_history_norm=traj_history_b,
                valid_history=valid_history_b,
                history_len=cfg.history_len,
                future_len=F,
                has_depth=depth_pil is not None,
            )

            images_pil = [_np_to_pil(annotated)]
            if depth_pil is not None:
                images_pil.append(depth_pil)

            vdir = str(meta_dirs[b]) if b < len(meta_dirs) else ""
            fidx = int(meta_frames[b]) if b < len(meta_frames) else -1

            if cfg.save_annotated_inputs:
                vname = Path(vdir).name if vdir else f"b{batch_idx:05d}_i{b:02d}"
                stem = f"sample_{saved + b:06d}_{vname}_f{fidx:06d}"
                _np_to_pil(annotated).save(output_dir / "annotated_inputs" / f"{stem}_rgb.png")
                if depth_pil is not None:
                    depth_pil.save(output_dir / "annotated_inputs" / f"{stem}_depth.png")
                with open(output_dir / "annotated_inputs" / f"{stem}_prompt.txt", "w") as f:
                    f.write(prompt)

            n_valid_kp = int(kpm_b.sum())
            pbar.set_postfix(
                video=Path(vdir).name[:24] if vdir else "?",
                frame=fidx,
                n_kp=n_valid_kp,
                call=f"0/{n_samples}",
                refresh=True,
            )
            for s in range(n_samples):
                seed_s = cfg.seed * 1000 + batch_idx * n_samples + s
                vname = Path(vdir).name if vdir else f"b{batch_idx:05d}_i{b:02d}"
                stem = f"sample_{saved + b:06d}_{vname}_f{fidx:06d}_s{s}"
                cached_path = output_dir / "raw_responses" / f"{stem}.json"
                # Resume: if a previous run already wrote this seed's response,
                # parse it from disk instead of re-billing an API call. Falls
                # back to a fresh call if the cached text fails to parse (e.g.,
                # a truncated leftover). Re-runs of the same command with the
                # same --seed produce identical (saved+b, vname, fidx, s)
                # filenames because `_build_first_frame_indices` is seed-driven.
                raw_text: str | None = None
                parsed: dict | None = None
                if cfg.save_raw_responses and cached_path.exists():
                    try:
                        with open(cached_path) as f:
                            raw_text = f.read()
                        parsed = _parse_vlm_json(raw_text)
                    except (OSError, ValueError) as exc:
                        logging.warning(
                            f"Cached response {cached_path.name} unusable "
                            f"({type(exc).__name__}); re-calling API."
                        )
                        raw_text, parsed = None, None
                if parsed is None:
                    if cfg.offline:
                        raise RuntimeError(
                            f"--offline=true but no usable cached response at "
                            f"{cached_path} (missing={not cached_path.exists()}). "
                            f"Refusing to call the API."
                        )
                    raw_text, parsed = client.generate_json(
                        images_pil, prompt, seed=seed_s,
                    )
                    if cfg.save_raw_responses:
                        with open(cached_path, "w") as f:
                            f.write(raw_text or "")
                pred_thou, statuses = _parsed_to_pred_uv_thou(
                    parsed, num_kp, F, kpm_b, current_kp_b,
                )
                preds_thou_S[b, s] = pred_thou
                parse_log.append({
                    "video_dir": vdir,
                    "frame_idx": fidx,
                    "sample": s,
                    "kp_statuses": statuses,
                })
                pbar.set_postfix(
                    video=Path(vdir).name[:24] if vdir else "?",
                    frame=fidx,
                    n_kp=n_valid_kp,
                    call=f"{s + 1}/{n_samples}",
                    refresh=True,
                )

        # Convert to tensors in [-1, 1] UV space; depth dim is forced to GT z so
        # ade_depth == 0 and ade_uvz == ade_uv (we only report _uv metrics).
        device = current_kp.device
        preds_uv_S = _thousand_to_uv_norm(preds_thou_S)  # (B, S, N, F, 2) in [-1, 1]
        preds_uv_S_t = torch.from_numpy(preds_uv_S).to(device).float()
        # Build (B, S, N, F, 3) with z = gt z (broadcasted across S).
        gt_abs = traj_future.float()  # (B, N, F, 3), uv in [-1, 1], z raw meters
        z_S = gt_abs[..., 2:3].unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
        traces_S_abs = torch.cat([preds_uv_S_t, z_S], dim=-1)  # (B, S, N, F, 3)
        traces_abs = traces_S_abs[:, 0]  # (B, N, F, 3) — sample-0 series.

        # Per-horizon accumulation. Slice (B, N, F, 3) / (B, S, N, F, 3) /
        # (B, N, F) tensors along the time axis at each horizon `h`, then
        # recompute l1_uv / ADE / FDE / DTW / Fréchet (sample-0) and
        # minADE / minFDE / minDTW / minFréchet (over S). All horizons share
        # the same predictions — only the slice changes — so this is
        # essentially free relative to the API calls.
        for h in horizons:
            gt_h = gt_abs[:, :, :h, :]
            traces_h = traces_abs[:, :, :h, :]
            traces_S_h = traces_S_abs[:, :, :, :h, :]
            valid_h = valid_future[:, :, :h]
            state = state_per_h[h]

            mask_h = (valid_h.unsqueeze(-1) & key_pad_mask[:, :, None, None]).to(traces_h.dtype)
            denom_h = mask_h.sum().clamp_min(1.0)
            l1_h = ((traces_h - gt_h).abs() * mask_h)[..., :2].sum() / denom_h
            state["l1_sum"] += float(l1_h)
            state["l1_n"] += 1

            _accumulate_traj_metrics(
                state["traj_acc"], traces_h, gt_h, valid_h, key_pad_mask,
                include_dtw=True,
            )
            _accumulate_min_metrics(
                state["min_acc"], traces_S_h, gt_h, valid_h, key_pad_mask,
            )
            # minDTW_uv — UV only (depth is forced to gt_z, so minDTW_uvz
            # degenerates to minDTW_uv; see _min_dtw_uv_sum docstring).
            dtw_s, dtw_n = _min_dtw_uv_sum(
                traces_S_h, gt_h, valid_h, key_pad_mask,
            )
            state["min_dtw_sum"] += dtw_s
            state["min_dtw_n"] += dtw_n
            # Sample-0 Fréchet_uv and Best-of-S minFréchet_uv — UV-only for
            # the same reason as DTW (depth is forced to gt_z).
            fre_s0, fre_n0 = _frechet_uv_sum(
                traces_h, gt_h, valid_h, key_pad_mask,
            )
            state["frechet_sum"] += fre_s0
            state["frechet_n"] += fre_n0
            fre_s, fre_n = _min_frechet_uv_sum(
                traces_S_h, gt_h, valid_h, key_pad_mask,
            )
            state["min_frechet_sum"] += fre_s
            state["min_frechet_n"] += fre_n

        # Visualization (single trace overlay per sample -> rollout_img/).
        for b in range(bsz):
            if cfg.max_samples > 0 and saved >= cfg.max_samples:
                break
            img = _image_chw_to_uint8_hwc(images[b])
            traj_hist_abs_b = traj_history[b].float().cpu().numpy()
            traces_abs_b = traces_abs[b].float().cpu().numpy()
            valid_hist_b = valid_history[b].cpu().numpy()
            valid_future_b = valid_future[b].cpu().numpy()
            kpm_b = key_pad_mask[b].cpu().numpy()
            current_kp_b = current_kp[b].float().cpu().numpy()

            vdir_b = str(meta_dirs[b]) if b < len(meta_dirs) else ""
            fidx_b = int(meta_frames[b]) if b < len(meta_frames) else -1
            vname = Path(vdir_b).name if vdir_b else f"b{batch_idx:05d}_i{b:02d}"
            fname = f"sample_{saved:06d}_{vname}_f{fidx_b:06d}.png"
            still_kwargs = dict(
                image=img,
                traj_history=traj_hist_abs_b,
                traj_future_pred=traces_abs_b,
                valid_history=valid_hist_b,
                valid_future=valid_future_b,
                current_kp=current_kp_b,
                key_pad_mask=kpm_b,
                predicted_valid_future=None,
                future_only=True,
                trail_alpha_min=0.7,
                trace_color_by="age",
            )
            Image.fromarray(
                render_trace_motion_still(
                    **still_kwargs, bg_alpha=0.5, bg_blend_color=(235, 235, 235),
                )
            ).save(output_dir / "rollout_img" / fname)

            if cfg.save_depth_panels and depth_panels_chw[b] is not None:
                d_img = (depth_panels_chw[b].permute(1, 2, 0).float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(d_img).save(output_dir / "depth" / fname)

            saved += 1
            pbar.update(1)

    pbar.close()

    # Build metrics_{h}.json for every horizon; the full horizon also lands in
    # metrics.json for backwards compatibility with existing report consumers.
    any_metrics_written = False
    metrics_per_h: dict[int, dict] = {}
    for h in horizons:
        state = state_per_h[h]
        if state["l1_n"] == 0:
            continue
        traj_metrics_h = _finalize_traj_metrics(
            state["traj_acc"], prefix="", include_dtw=True,
        )
        min_metrics_h = _finalize_min_metrics(
            state["min_acc"], prefix="", num_samples=n_samples,
        )
        min_dtw_uv_h = state["min_dtw_sum"] / max(state["min_dtw_n"], 1)
        frechet_uv_h = state["frechet_sum"] / max(state["frechet_n"], 1)
        min_frechet_uv_h = state["min_frechet_sum"] / max(state["min_frechet_n"], 1)
        metrics_h: dict[str, float] = {
            "horizon": float(h),
            "l1_uv": state["l1_sum"] / state["l1_n"],
            "ade_uv": traj_metrics_h.get("ade_uv", 0.0),
            "fde_uv": traj_metrics_h.get("fde_uv", 0.0),
            "dtw_uv": traj_metrics_h.get("dtw_uv", 0.0),
            "frechet_uv": frechet_uv_h,
            f"minADE_{n_samples}_uv": min_metrics_h.get(f"minADE_{n_samples}_uv", 0.0),
            f"minFDE_{n_samples}_uv": min_metrics_h.get(f"minFDE_{n_samples}_uv", 0.0),
            f"minDTW_{n_samples}_uv": min_dtw_uv_h,
            f"minFrechet_{n_samples}_uv": min_frechet_uv_h,
            "smooth_pred_uv": min_metrics_h.get("smooth_pred_uv", 0.0),
            "smooth_gt_uv": min_metrics_h.get("smooth_gt_uv", 0.0),
            "n_batches": float(state["l1_n"]),
            "n_samples": float(n_samples),
        }
        metrics_per_h[h] = metrics_h
        fname = "metrics.json" if h == cfg.future_len else f"metrics_{h}.json"
        with open(output_dir / fname, "w") as f:
            json.dump(metrics_h, f, indent=2)
        any_metrics_written = True

    if any_metrics_written:
        # Headline log line uses the full horizon (metrics.json).
        full_m = metrics_per_h.get(cfg.future_len, {})
        lines = [f"[done] saved {saved} overlays to {output_dir}"]
        for h in horizons:
            if h not in metrics_per_h:
                continue
            m = metrics_per_h[h]
            tag = "full" if h == cfg.future_len else f"H={h}"
            lines.append(
                f"  [{tag:>5}] l1_uv={m['l1_uv']:.4f} ade_uv={m['ade_uv']:.4f} "
                f"fde_uv={m['fde_uv']:.4f} dtw_uv={m['dtw_uv']:.4f} "
                f"frechet_uv={m['frechet_uv']:.4f} | "
                f"minADE_{n_samples}_uv={m[f'minADE_{n_samples}_uv']:.4f} "
                f"minFDE_{n_samples}_uv={m[f'minFDE_{n_samples}_uv']:.4f} "
                f"minDTW_{n_samples}_uv={m[f'minDTW_{n_samples}_uv']:.4f} "
                f"minFrechet_{n_samples}_uv={m[f'minFrechet_{n_samples}_uv']:.4f}"
            )
        lines.append(
            f"  (over {int(full_m.get('n_batches', 0))} batches, "
            f"{n_samples} samples; wrote metrics.json + "
            f"{', '.join(f'metrics_{h}.json' for h in horizons if h != cfg.future_len and h in metrics_per_h) or '(no short-horizon files)'})"
        )
        logging.info("\n".join(lines))
    else:
        logging.info(f"[done] saved {saved} overlays to {output_dir} (no metrics — no batches processed)")

    if parse_log:
        with open(output_dir / "parse_log.json", "w") as f:
            json.dump(parse_log, f, indent=2)


def main() -> None:
    predict()


if __name__ == "__main__":
    main()
