"""GroundingDINO + SAM 2 wrapper that turns an RGB frame into K hand-region keypoints.

Two-pass detection (loose-then-looser thresholds), then SAM 2 for the highest-
confidence box, then uniform-without-replacement sampling over the mask.

Switched from Grounded-Segment-Anything (v1) to Grounded-SAM-2 (v2):
- segment_anything.SamPredictor → sam2.SAM2ImagePredictor
- groundingdino.* import path → grounding_dino.groundingdino.* (the v2 fork's
  inference.py uses the latter for its own internal imports, so we add the
  Grounded-SAM-2/ repo root to sys.path to make the namespace package resolve)
- Vendored repo path: infer_helpers/Grounded-SAM-2/ (was Grounded-Segment-Anything/)

See `docs/predict_trace_image_only_plan.md` §3 for the design (the doc still
refers to v1 model names, but the pipeline is unchanged — only the underlying
implementations were swapped).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


_GSA_REPO_RELPATH = "infer_helpers/Grounded-SAM-2"
_DEFAULT_CLASSES = ("human hand", "robot hand")
_DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _GSA_REPO_RELPATH).exists():
            return parent
    raise FileNotFoundError(
        f"Could not locate {_GSA_REPO_RELPATH} from {here}. "
        "Make sure the vendored Grounded-SAM-2 repo is present."
    )


def _maybe_prefetch_nltk() -> None:
    """GroundingDINO downloads `punkt` / `averaged_perceptron_tagger` lazily;
    pull them once here so the first frame doesn't wait on a network call."""
    try:
        import nltk
    except ImportError:
        return
    for pkg in ("punkt", "averaged_perceptron_tagger", "punkt_tab", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:  # noqa: BLE001 — pre-fetch is best-effort.
            logging.warning(f"nltk pre-fetch failed for {pkg}: {e}")


def _patch_transformers_for_gdino() -> None:
    """Make GroundingDINO v2 work on transformers 5.x.

    Two breakages in `bertwarper.py`:
      1. `BertModel.get_head_mask` was removed in transformers 5.x. GDINO copies
         this into `BertModelWarper.__init__` (bertwarper.py:29).
      2. `ModuleUtilsMixin.get_extended_attention_mask` lost its `device` arg in
         transformers 5.x. GDINO calls it positionally with `(attention_mask,
         input_shape, device)` (bertwarper.py:109), so `device` lands in the
         `dtype` slot and the subsequent `.to(dtype=device)` raises TypeError.

    Both are restored as monkey-patches on `ModuleUtilsMixin`. Idempotent.
    """
    import torch
    from transformers import BertModel
    from transformers.modeling_utils import ModuleUtilsMixin

    # 1. get_head_mask
    if not hasattr(ModuleUtilsMixin, "get_head_mask"):
        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=self.dtype)
            return head_mask

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            head_mask = _convert_head_mask_to_5d(self, head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        ModuleUtilsMixin._convert_head_mask_to_5d = _convert_head_mask_to_5d
        ModuleUtilsMixin.get_head_mask = get_head_mask
        BertModel.get_head_mask = get_head_mask

    # 2. get_extended_attention_mask — restore the legacy `device` 3rd-positional arg.
    _orig_geam = ModuleUtilsMixin.get_extended_attention_mask
    if not getattr(_orig_geam, "_gdino_compat", False):
        def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
            # If called legacy-positional with device as 3rd arg, route correctly.
            if isinstance(device, torch.dtype) and dtype is None:
                dtype, device = device, None  # caller used new signature positionally
            return _orig_geam(self, attention_mask, input_shape, dtype=dtype)
        get_extended_attention_mask._gdino_compat = True
        ModuleUtilsMixin.get_extended_attention_mask = get_extended_attention_mask
        BertModel.get_extended_attention_mask = get_extended_attention_mask


@dataclass
class HandSegmentation:
    mask: np.ndarray | None         # (H, W) bool — None when no mask was found
    status: str                     # "pass1", "pass2", "random_fallback"
    confidence: float               # detection confidence; NaN for random_fallback


@dataclass
class Segmentation:
    """One detection from the user-prompted multi-class path (used by the
    interactive GUI). Unlike `HandSegmentation`, the mask is always populated
    (callers receive a list — empty if nothing was detected at all)."""

    mask: np.ndarray         # (H, W) bool
    label: str               # the class string from the user's prompt
    confidence: float        # GroundingDINO confidence
    box_xyxy: np.ndarray     # (4,) float32 — original-resolution box


class GroundedHandSegmenter:
    """Loads GroundingDINO + SAM 2 once; per-frame hand-mask + keypoint sampling."""

    def __init__(
        self,
        gdino_config: str | Path,
        gdino_checkpoint: str | Path,
        sam_checkpoint: str | Path,
        sam_model_cfg: str = _DEFAULT_SAM2_MODEL_CFG,
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        box_threshold_pass2: float = 0.15,
        text_threshold_pass2: float = 0.15,
        nms_threshold: float = 0.8,
        classes: tuple[str, ...] = _DEFAULT_CLASSES,
        fallback_random_on_no_mask: bool = True,
    ):
        for path, name in (
            (gdino_config, "gdino_config"),
            (gdino_checkpoint, "gdino_checkpoint"),
            (sam_checkpoint, "sam_checkpoint"),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found: {path}")

        repo_root = _find_repo_root()
        # Prepend Grounded-SAM-2/ to sys.path so the implicit namespace package
        # `grounding_dino` (no __init__.py) resolves. The v2 inference.py uses
        # `from grounding_dino.groundingdino.X import ...` for its own internal
        # imports, which only works when the repo root is on sys.path.
        gsa2_dir = str(repo_root / _GSA_REPO_RELPATH)
        if gsa2_dir not in sys.path:
            sys.path.insert(0, gsa2_dir)

        _maybe_prefetch_nltk()
        _patch_transformers_for_gdino()

        from grounding_dino.groundingdino.util.inference import Model as GDinoModel
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._gdino = GDinoModel(
            model_config_path=str(gdino_config),
            model_checkpoint_path=str(gdino_checkpoint),
            device=device,
        )
        sam2_model = build_sam2(sam_model_cfg, str(sam_checkpoint), device=device)
        self._sam = SAM2ImagePredictor(sam2_model)

        self.classes = tuple(classes)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.box_threshold_pass2 = float(box_threshold_pass2)
        self.text_threshold_pass2 = float(text_threshold_pass2)
        self.nms_threshold = float(nms_threshold)
        self.fallback_random_on_no_mask = bool(fallback_random_on_no_mask)
        self.device = device

    def _detect(
        self,
        image_bgr: np.ndarray,
        box_thresh: float,
        text_thresh: float,
        classes: Sequence[str] | None = None,
    ):
        cls = list(classes) if classes is not None else list(self.classes)
        det = self._gdino.predict_with_classes(
            image=image_bgr,
            classes=cls,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
        )
        if det.xyxy is None or len(det.xyxy) == 0:
            return None
        # NMS across both class IDs together — we only need the single best box.
        import torchvision

        keep = torchvision.ops.nms(
            torch.from_numpy(det.xyxy.astype(np.float32)),
            torch.from_numpy(det.confidence.astype(np.float32)),
            self.nms_threshold,
        ).numpy().tolist()
        det.xyxy = det.xyxy[keep]
        det.confidence = det.confidence[keep]
        det.class_id = det.class_id[keep]
        if len(det.xyxy) == 0:
            return None
        return det

    def _sam_mask(self, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        self._sam.set_image(image_rgb)
        # SAM 2 returns (N_box, M_mask, H, W) when multimask=True with batched
        # boxes; we pass the single best box with a leading batch dim.
        masks, scores, _ = self._sam.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy[None, :],
            multimask_output=True,
        )
        # masks: (1, M, H, W), scores: (1, M). Pick the highest-IoU mask.
        if masks.ndim == 4:
            best = int(np.argmax(scores[0]))
            return masks[0, best].astype(bool)
        # Single-box single-mask edge case: (M, H, W).
        best = int(np.argmax(scores))
        return masks[best].astype(bool)

    def segment_hand(self, image_rgb_uint8: np.ndarray) -> HandSegmentation:
        """`(H, W, 3)` uint8 RGB → best hand mask (bool) + status string."""
        if image_rgb_uint8.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB, got dtype={image_rgb_uint8.dtype}")
        if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[-1] != 3:
            raise ValueError(
                f"expected (H, W, 3) RGB, got shape={image_rgb_uint8.shape}"
            )
        bgr = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)

        for status, b_th, t_th in (
            ("pass1", self.box_threshold, self.text_threshold),
            ("pass2", self.box_threshold_pass2, self.text_threshold_pass2),
        ):
            det = self._detect(bgr, b_th, t_th)
            if det is None:
                continue
            best = int(np.argmax(det.confidence))
            mask = self._sam_mask(image_rgb_uint8, det.xyxy[best])
            return HandSegmentation(
                mask=mask, status=status, confidence=float(det.confidence[best])
            )

        return HandSegmentation(mask=None, status="random_fallback", confidence=float("nan"))

    def segment_with_classes(
        self,
        image_rgb_uint8: np.ndarray,
        classes: Sequence[str],
        top_k: int | None = None,
    ) -> list[Segmentation]:
        """Detect every box matching `classes`, run SAM 2 per box, return all
        results sorted by confidence desc.

        Tries pass-1 thresholds first; if no box passes, falls back to pass-2.
        Unlike `segment_hand`, no random fallback — callers handle the empty
        case (the interactive GUI shows a "no detections" status).
        """
        if image_rgb_uint8.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB, got dtype={image_rgb_uint8.dtype}")
        if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[-1] != 3:
            raise ValueError(
                f"expected (H, W, 3) RGB, got shape={image_rgb_uint8.shape}"
            )
        cls = list(classes)
        if not cls:
            return []
        bgr = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)

        for b_th, t_th in (
            (self.box_threshold, self.text_threshold),
            (self.box_threshold_pass2, self.text_threshold_pass2),
        ):
            det = self._detect(bgr, b_th, t_th, classes=cls)
            if det is None:
                continue
            order = np.argsort(-det.confidence)
            if top_k is not None:
                order = order[: int(top_k)]
            results: list[Segmentation] = []
            for idx in order:
                idx = int(idx)
                box = det.xyxy[idx].astype(np.float32)
                mask = self._sam_mask(image_rgb_uint8, box).astype(bool)
                cid = int(det.class_id[idx]) if det.class_id is not None else 0
                label = cls[cid] if 0 <= cid < len(cls) else ""
                results.append(
                    Segmentation(
                        mask=mask,
                        label=label,
                        confidence=float(det.confidence[idx]),
                        box_xyxy=box,
                    )
                )
            return results

        return []


def sample_keypoints_from_mask(
    seg: HandSegmentation,
    image_hw: tuple[int, int],
    num_keypoints: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sample `num_keypoints` pixels from `seg.mask` (or a centered fallback).

    Returns `(uv_norm, rowcol, status)`:
      - `uv_norm`: `(K, 2)` float32 in [-1, 1] using `(col/W*2-1, row/H*2-1)`.
      - `rowcol`: `(K, 2)` int64 — original-resolution `(row, col)` indices,
        reusable for `D_aligned[row, col]` to fill the `current_kp[:, 2]` z slot.
      - `status`: input `seg.status`, or upgraded to `"random_fallback"` when
        the mask was too small (< num_keypoints valid pixels).
    """
    H, W = image_hw
    status = seg.status
    if seg.mask is None or int(seg.mask.sum()) < num_keypoints:
        if seg.mask is not None and int(seg.mask.sum()) < num_keypoints:
            status = "random_fallback"
        # 60% central crop matches plan §3.3 pass-3 fallback.
        margin_h, margin_w = int(0.20 * H), int(0.20 * W)
        rows = rng.integers(margin_h, H - margin_h, size=num_keypoints)
        cols = rng.integers(margin_w, W - margin_w, size=num_keypoints)
    else:
        flat_idx = rng.choice(np.flatnonzero(seg.mask), size=num_keypoints, replace=False)
        rows, cols = np.divmod(flat_idx, W)

    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    u = cols.astype(np.float32) / float(W) * 2.0 - 1.0
    v = rows.astype(np.float32) / float(H) * 2.0 - 1.0
    uv = np.stack([u, v], axis=-1)
    rowcol = np.stack([rows, cols], axis=-1)
    return uv, rowcol, status


def sample_keypoints_from_union(
    masks: Sequence[np.ndarray],
    image_hw: tuple[int, int],
    num_keypoints: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sample `num_keypoints` pixels uniformly from the union of `masks`.

    Returns `(uv_norm, rowcol, status)` matching `sample_keypoints_from_mask`.
    `status` is one of:
      - `"mask_sampled"`: union had >= num_keypoints valid pixels.
      - `"mask_sampled_with_replacement"`: union non-empty but smaller than K.
      - `"random_fallback"`: empty / no masks supplied → 60% center-crop random.
    """
    H, W = image_hw
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        if m is None:
            continue
        if m.shape != (H, W):
            raise ValueError(f"mask shape {m.shape} does not match image {(H, W)}")
        union |= m.astype(bool, copy=False)
    n_valid = int(union.sum())

    if n_valid == 0:
        margin_h, margin_w = int(0.20 * H), int(0.20 * W)
        rows = rng.integers(margin_h, H - margin_h, size=num_keypoints)
        cols = rng.integers(margin_w, W - margin_w, size=num_keypoints)
        status = "random_fallback"
    elif n_valid < num_keypoints:
        flat_idx = rng.choice(np.flatnonzero(union), size=num_keypoints, replace=True)
        rows, cols = np.divmod(flat_idx, W)
        status = "mask_sampled_with_replacement"
    else:
        flat_idx = rng.choice(np.flatnonzero(union), size=num_keypoints, replace=False)
        rows, cols = np.divmod(flat_idx, W)
        status = "mask_sampled"

    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    u = cols.astype(np.float32) / float(W) * 2.0 - 1.0
    v = rows.astype(np.float32) / float(H) * 2.0 - 1.0
    uv = np.stack([u, v], axis=-1).astype(np.float32)
    rowcol = np.stack([rows, cols], axis=-1)
    return uv, rowcol, status
