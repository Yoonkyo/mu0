#!/usr/bin/env python
"""On-demand image+text → 3D trace GUI.

Loads SmolVLA + Depth-Anything-V2 + Grounded-SAM-2 once and runs a viser
server. In one browser tab you upload an image, type a task prompt and a
comma-separated list of objects to ground, click "Run segmentation" to see
the detected masks, tick which masks you want to track, then click
"Predict trace" — the predicted 3D trajectory shows up in the same scene.

No GT/history layers (single-shot inference, no dataset). Camera intrinsics
are synthesized from a user-controlled HFOV slider; extrinsics are identity
(world frame == input camera frame).

Example:

    CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_trace_gui.py \\
        --checkpoint=/path/to/checkpoints/step_NNNNNNNN \\
        --delta_stats_path=/path/to/delta_stats.json \\
        --port=8080

SSH tunnel from your laptop:

    ssh -L 8080:localhost:8080 user@host
    # then visit http://localhost:8080
"""

import io
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np
import torch
import viser
import viser.transforms as tf
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.trace_dataset import (
    CURRENT_KP_KEY,
    DEPTH_KEY,
    IMAGE_KEY,
    KEY_PAD_MASK_KEY,
    TRAJ_FUTURE_KEY,
    TRAJ_HISTORY_KEY,
    VALID_FUTURE_KEY,
    VALID_HISTORY_KEY,
)
from lerobot.datasets.trace_delta_stats import load_trace_stats
from lerobot.datasets.trace_depth import render_depth_rgb
from lerobot.inference_helpers.depth_anything import DepthAnythingV2Wrapper
from lerobot.inference_helpers.grounded_sam import (
    GroundedHandSegmenter,
    Segmentation,
    sample_keypoints_from_union,
)
from lerobot.policies.mu0.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_visualize_trace_3d import (
    _colors_from_uv,
    _fade_colors,
    _segments_with_meta,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.utils import init_logging


_DEFAULT_DA_V2_CKPT = "infer_helpers/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
_DEFAULT_GDINO_CONFIG = (
    "infer_helpers/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
_DEFAULT_GDINO_CKPT = "infer_helpers/checkpoints/groundingdino_swint_ogc.pth"
_DEFAULT_SAM_CKPT = "infer_helpers/checkpoints/sam2.1_hiera_large.pt"
_DEFAULT_SAM_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# "See sample" button payload — a frame shipped next to this script plus the
# task/classes that go with it, so a first-time user can run the full pipeline
# without uploading anything.
_SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent / "sample.png"
_SAMPLE_TASK = "pick up the doll"
_SAMPLE_CLASSES = "doll"


@dataclass
class TraceGuiConfig:
    # Required.
    checkpoint: Path | None = None
    delta_stats_path: Path | None = None

    # Server
    port: int = 8080
    share: bool = False

    # Inference
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_inference_steps: int | None = None
    image_size: int | None = None  # default = ckpt's resize_imgs_with_padding[0]
    history_len: int | None = None
    future_len: int | None = None
    seed: int = 0

    # Done-head truncation threshold (docs/trace_done_head_plan.md). Only
    # consulted when the loaded checkpoint has `trace_done_head=True`. Default
    # `None` means "use the value baked into the checkpoint config".
    done_threshold: float | None = None

    # GUI defaults. The task/classes are placeholder templates ("{your object}")
    # so the empty state signals what to type; "See sample" loads a concrete,
    # working example instead.
    default_task: str = "pick up {your object}"
    default_classes: str = "{your object}"
    default_num_keypoints: int = 16
    default_top_k_masks: int = 8
    default_initial_selected: int = 1   # how many masks pre-checked after segmentation
    default_fov_deg: float = 60.0       # horizontal FOV used to synthesize K
    pixel_downsample: int = 4           # depth-pcd stride

    # Depth-Anything-V2
    depth_align: str = "log_percentile"  # log_percentile | median_scale | none
    depth_anything_ckpt: Path = Path(_DEFAULT_DA_V2_CKPT)
    depth_anything_encoder: str = "vitl"
    depth_anything_input_size: int = 518

    # Grounded-SAM-2
    gdino_config: Path = Path(_DEFAULT_GDINO_CONFIG)
    gdino_ckpt: Path = Path(_DEFAULT_GDINO_CKPT)
    sam_ckpt: Path = Path(_DEFAULT_SAM_CKPT)
    sam_model_cfg: str = _DEFAULT_SAM_MODEL_CFG
    gdino_box_threshold: float = 0.30
    gdino_text_threshold: float = 0.25
    gdino_box_threshold_pass2: float = 0.15
    gdino_text_threshold_pass2: float = 0.15

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


# ---------- camera + geometry ----------

def _K_from_hfov(W: int, H: int, hfov_deg: float) -> np.ndarray:
    """Build a centered-pinhole intrinsics matrix from a horizontal FOV."""
    fov = float(np.deg2rad(hfov_deg))
    fx = (W * 0.5) / max(np.tan(fov * 0.5), 1e-6)
    fy = fx  # square pixels — no anisotropic FOV slider
    K = np.array(
        [[fx, 0.0, W * 0.5],
         [0.0, fy, H * 0.5],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K


def _unproject_to_world(
    uv_norm_z: np.ndarray, K: np.ndarray, c2w: np.ndarray, W: int, H: int
) -> np.ndarray:
    """`(..., 3)` of `(u_norm, v_norm, z_meters)` → world coords `(..., 3)`.

    Same convention as `lerobot_predict_trace_mu0_image_only.py`: `u_norm/v_norm` in
    [-1, 1] over the full image, `z` in meters in the camera frame at time t.
    """
    u_norm = uv_norm_z[..., 0]
    v_norm = uv_norm_z[..., 1]
    z = uv_norm_z[..., 2]
    px = (u_norm + 1.0) * 0.5 * W
    py = (v_norm + 1.0) * 0.5 * H
    x_cam = (px - K[0, 2]) / K[0, 0] * z
    y_cam = (py - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x_cam, y_cam, z], axis=-1)
    flat = cam.reshape(-1, 3).astype(np.float32, copy=False)
    flat_h = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    world = (c2w.astype(np.float32) @ flat_h.T).T[:, :3]
    return world.reshape(uv_norm_z.shape)


def _unproject_depth_frame(
    image: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-stride downsample + per-pixel unproject; returns (xyz, rgb) (M, 3)."""
    H, W = depth.shape
    ys = np.arange(0, H, downsample)
    xs = np.arange(0, W, downsample)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    z = depth[grid_y, grid_x].astype(np.float32)
    valid = (z > 1e-3) & np.isfinite(z)
    grid_x_v = grid_x[valid].astype(np.float32)
    grid_y_v = grid_y[valid].astype(np.float32)
    z_v = z[valid]
    x_cam = (grid_x_v - K[0, 2]) / K[0, 0] * z_v
    y_cam = (grid_y_v - K[1, 2]) / K[1, 1] * z_v
    cam = np.stack([x_cam, y_cam, z_v], axis=-1)
    cam_h = np.concatenate([cam, np.ones((cam.shape[0], 1), dtype=np.float32)], axis=1)
    world = (c2w.astype(np.float32) @ cam_h.T).T[:, :3]
    rgb_full = image[grid_y, grid_x].reshape(-1, 3)
    rgb = rgb_full[valid.reshape(-1)]
    return world.astype(np.float32), rgb.astype(np.uint8)


# ---------- image utilities ----------

# A small, stable color cycle for mask overlays. Indexed mod len.
_MASK_COLORS = np.array(
    [
        (255, 80, 80),    # red
        (80, 200, 120),   # green
        (90, 140, 255),   # blue
        (255, 200, 70),   # yellow
        (220, 120, 255),  # magenta
        (90, 220, 220),   # cyan
        (255, 150, 70),   # orange
        (180, 220, 90),   # chartreuse
    ],
    dtype=np.float32,
)


def _resize_to_square(img_uint8: np.ndarray, size: int) -> np.ndarray:
    """`(H, W, 3)` uint8 → `(size, size, 3)` uint8 via bilinear interp.

    Matches `TraceDataset._get_one`'s `F.interpolate(..., mode="bilinear")` so
    the policy sees the same pixels at training and inference time.
    """
    if img_uint8.shape[0] == size and img_uint8.shape[1] == size:
        return img_uint8.astype(np.uint8, copy=False)
    t = torch.from_numpy(img_uint8).permute(2, 0, 1).float()  # (3, H, W)
    t = torch.nn.functional.interpolate(
        t.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False,
    ).squeeze(0)
    out = t.permute(1, 2, 0).clamp(0, 255).byte().numpy()
    return out


def _make_mask_thumbnail(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int],
    alpha: float = 0.55, max_dim: int = 192,
) -> np.ndarray:
    """Alpha-blend `color` over `mask` and resize to fit `max_dim`."""
    overlay = image.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
    overlay = overlay.clip(0, 255).astype(np.uint8)
    H, W = overlay.shape[:2]
    longest = max(H, W)
    if longest > max_dim:
        scale = max_dim / longest
        new_h = max(int(round(H * scale)), 1)
        new_w = max(int(round(W * scale)), 1)
        overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return overlay


def _make_combined_overlay(
    image: np.ndarray, segs: list[Segmentation], alpha: float = 0.5
) -> np.ndarray:
    """Composite all visible masks onto the image with cycled colors."""
    overlay = image.astype(np.float32).copy()
    for i, seg in enumerate(segs):
        color = _MASK_COLORS[i % len(_MASK_COLORS)]
        overlay[seg.mask] = (1.0 - alpha) * overlay[seg.mask] + alpha * color
    return overlay.clip(0, 255).astype(np.uint8)


# ---------- model loading ----------

def _load_policy(ckpt: Path, device: str) -> SmolVLAPolicy:
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.load_vlm_weights = False
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(str(ckpt), config=cfg, strict=False)
    return policy


def _make_tokenizer(vlm_model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(vlm_model_name)
    tok.padding_side = "right"
    return tok


# ---------- prediction (B=1) ----------

def _build_batch(
    image_chw: torch.Tensor,         # (3, image_size, image_size) float [0, 1]
    depth_chw: torch.Tensor,         # (3, image_size, image_size) float [0, 1]
    current_kp: torch.Tensor,        # (K, 3) — uv in [-1, 1] + z meters
    history_len: int,
    future_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    K = current_kp.shape[0]
    batch = {
        IMAGE_KEY: image_chw[None].to(device),
        DEPTH_KEY: depth_chw[None].to(device),
        CURRENT_KP_KEY: current_kp[None].to(device),
        TRAJ_HISTORY_KEY: torch.zeros(1, K, history_len, 3, dtype=torch.float32, device=device),
        TRAJ_FUTURE_KEY: torch.zeros(1, K, future_len, 3, dtype=torch.float32, device=device),
        VALID_HISTORY_KEY: torch.zeros(1, K, history_len, dtype=torch.bool, device=device),
        VALID_FUTURE_KEY: torch.zeros(1, K, future_len, dtype=torch.bool, device=device),
        KEY_PAD_MASK_KEY: torch.ones(1, K, dtype=torch.bool, device=device),
    }
    return batch


def _tokenize_into_batch(
    batch: dict[str, torch.Tensor],
    task: str,
    tokenizer,
    max_length: int,
    device: torch.device,
) -> None:
    prompt = f"Task: {task.strip() or 'predict future trajectory'}\n"
    enc = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    batch[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].bool().to(device)


# ---------- 3D scene ----------

def _clear_handles(handles: list) -> None:
    """Best-effort remove() each handle (innermost-last)."""
    while handles:
        h = handles.pop()
        try:
            h.remove()
        except Exception:
            pass


def _build_scene(
    server: viser.ViserServer,
    state: dict[str, Any],
    K: np.ndarray,
    c2w: np.ndarray,
    image: np.ndarray,
    depth_meters: np.ndarray,
    current_kp_uv_z: np.ndarray,        # (N, 3)
    traces_abs: np.ndarray,             # (N, F, 3)
    pixel_downsample: int,
    pred_valid_step: np.ndarray | None = None,  # (N, F) bool, None → all valid
) -> None:
    """(Re)build the 3D scene from cached prediction outputs + current K."""
    _clear_handles(state["scene_handles"])
    handles: list = state["scene_handles"]

    H, W = image.shape[:2]
    n_kp = current_kp_uv_z.shape[0]

    # Point cloud from the (DA-V2) depth panel.
    pcd_xyz, pcd_rgb = _unproject_depth_frame(
        image, depth_meters, K, c2w, pixel_downsample
    )
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

    # World axes triad — small landmark at the origin.
    axes = server.scene.add_line_segments(
        name="world_axes",
        points=np.array(
            [
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
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
    axes.visible = state["gui_show_axes"].value
    state["axes_handle"] = axes
    handles.append(axes)

    # Predicted polylines (current_kp prepended so the line starts at the obs).
    cur_world = _unproject_to_world(current_kp_uv_z[:, None, :], K, c2w, W, H)[:, 0, :]
    pred_world = _unproject_to_world(traces_abs, K, c2w, W, H)
    colors = _colors_from_uv(current_kp_uv_z[:n_kp, :2])

    pred_segs: list[np.ndarray] = []
    pred_seg_colors: list[np.ndarray] = []
    for i in range(n_kp):
        pts = np.concatenate([cur_world[i:i + 1], pred_world[i]], axis=0)
        if pred_valid_step is not None:
            # Anchor (current_kp) is always valid; future steps follow the
            # done-head's stop_step mask so the polyline truncates instead
            # of running through the frozen-position tail.
            valid = np.concatenate(
                [np.ones(1, dtype=bool), pred_valid_step[i].astype(bool)], axis=0
            )
        else:
            valid = np.ones(pts.shape[0], dtype=bool)
        s, end_idx, _, _ = _segments_with_meta(pts, valid)
        c = _fade_colors(
            end_idx, pts.shape[0] - 1, colors[i],
            weight_start=0.55, weight_end=1.00,
            neutral_rgb=(1.0, 1.0, 1.0),
        )
        if s.shape[0] > 0:
            pred_segs.append(s)
            pred_seg_colors.append(c)

    if pred_segs:
        all_segs = np.concatenate(pred_segs, axis=0)
        all_cols = np.concatenate(pred_seg_colors, axis=0)
        per_vertex = np.repeat(all_cols[:, None, :], 2, axis=1).astype(np.float32)
        pred_handle = server.scene.add_line_segments(
            name="traj_pred",
            points=all_segs,
            colors=per_vertex,
            line_width=state["gui_track_width"].value,
        )
        pred_handle.visible = state["gui_show_pred"].value
        state["pred_handle"] = pred_handle
        handles.append(pred_handle)
    else:
        state["pred_handle"] = None

    # Current keypoints as small dots.
    if n_kp > 0:
        kp_handle = server.scene.add_point_cloud(
            name="current_keypoints",
            points=cur_world.astype(np.float32),
            colors=(colors * 255).astype(np.uint8),
            point_size=state["gui_kp_size"].value,
            point_shape="circle",
        )
        kp_handle.visible = state["gui_show_kp"].value
        state["kp_handle"] = kp_handle
        handles.append(kp_handle)
    else:
        state["kp_handle"] = None

    # Camera frustum at the input image.
    fov = 2.0 * np.arctan2(H / 2.0, K[0, 0])
    aspect = W / max(H, 1)
    frustum = server.scene.add_camera_frustum(
        name="camera_frustum",
        fov=fov,
        aspect=aspect,
        scale=state["gui_frustum_scale"].value,
        image=image.astype(np.float32) / 255.0,
        wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
    )
    frustum.visible = state["gui_show_frustum"].value
    state["frustum_handle"] = frustum
    handles.append(frustum)


# ---------- main entry ----------

@draccus.wrap()
def run(cfg: TraceGuiConfig) -> None:
    init_logging()

    device = torch.device(cfg.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
    use_autocast = dtype != torch.float32 and device.type == "cuda"

    logging.info(f"Loading SmolVLA from {cfg.checkpoint}")
    policy = _load_policy(Path(cfg.checkpoint), cfg.device)
    policy.to(device=device)
    policy.eval()
    pcfg = policy.config
    image_size = cfg.image_size if cfg.image_size is not None else pcfg.resize_imgs_with_padding[0]
    history_len = cfg.history_len if cfg.history_len is not None else pcfg.history_len
    future_len = cfg.future_len if cfg.future_len is not None else pcfg.future_len
    num_inference_steps = (
        cfg.num_inference_steps if cfg.num_inference_steps is not None else pcfg.num_steps
    )
    use_depth = bool(getattr(pcfg, "use_depth", False))
    if not use_depth:
        raise ValueError(
            "Checkpoint has use_depth=False; the GUI assumes a depth-conditioned model."
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
    delta_scale_t = torch.as_tensor(delta_scale, dtype=torch.float32, device=device)

    logging.info(f"Loading Depth-Anything-V2 from {cfg.depth_anything_ckpt}")
    da_v2 = DepthAnythingV2Wrapper(
        checkpoint_path=cfg.depth_anything_ckpt,
        encoder=cfg.depth_anything_encoder,
        input_size=cfg.depth_anything_input_size,
        device=cfg.device,
    )

    logging.info(f"Loading Grounded-SAM-2 (gdino={cfg.gdino_ckpt}, sam={cfg.sam_ckpt})")
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
    )

    tokenizer = _make_tokenizer(pcfg.vlm_model_name)
    np_rng = np.random.default_rng(cfg.seed)

    # Heavy ops are dispatched to a worker thread; the lock serializes them
    # so two button-clicks can't race on shared state / GPU.
    work_lock = threading.Lock()

    # ------- viser server + state -------

    server = viser.ViserServer(port=cfg.port)
    if cfg.share:
        server.request_share_url()
    server.scene.set_up_direction("-y")

    state: dict[str, Any] = {
        "image_resized": None,        # (image_size, image_size, 3) uint8
        "depth_aligned": None,        # (image_size, image_size) float32 meters
        "depth_turbo_chw": None,      # (3, image_size, image_size) float32 [0,1]
        "all_segmentations": [],      # list[Segmentation], full pool from last segment call
        "selection_checkboxes": [],   # [(idx, cb_handle)] for reading values on Predict
        "mask_handles": [],           # GUI handles to remove on rebuild
        "scene_handles": [],          # 3D scene handles
        "input_image_handle": None,   # GuiImageHandle for upload preview
        "last_pred": None,            # cached {"current_kp", "traces_abs", "image", "depth"}
    }

    # ----- GUI -----

    def set_status(msg: str) -> None:
        try:
            status_text.value = msg
        except Exception:
            pass
        logging.info(f"[gui] {msg}")

    with server.gui.add_folder("Input"):
        server.gui.add_markdown(
            "**Steps:** 1) Upload your image (or See sample image) → "
            "2) Run segmentation → 3) Predict trace"
        )
        upload_btn = server.gui.add_upload_button(
            "Upload your image", mime_type="image/*"
        )
        sample_btn = server.gui.add_button(
            "See sample image",
            hint="Load a bundled example image + matching task/classes.",
        )
        task_input = server.gui.add_text(
            "Task prompt", initial_value=cfg.default_task,
            hint="Type your task instruction (fed to the SmolVLA tokenizer).",
        )
        classes_input = server.gui.add_text(
            "Classes (comma)", initial_value=cfg.default_classes,
            hint="Type words for GroundingDINO detection + SAM segmentation "
            "(comma-separated).",
        )
        segment_btn = server.gui.add_button("Run segmentation")
        status_text = server.gui.add_text(
            "Status", initial_value="upload an image to begin", disabled=True
        )

    with server.gui.add_folder("Mask selection"):
        top_k_slider = server.gui.add_slider(
            "Top-K masks", min=1, max=16, step=1,
            initial_value=cfg.default_top_k_masks,
        )
        num_kp_slider = server.gui.add_slider(
            "# keypoints", min=4, max=64, step=1,
            initial_value=cfg.default_num_keypoints,
        )

    with server.gui.add_folder("Predict"):
        predict_btn = server.gui.add_button("Predict trace")
        fov_slider = server.gui.add_slider(
            "Camera HFOV (deg)", min=30.0, max=110.0, step=1.0,
            initial_value=cfg.default_fov_deg,
            hint="Synthesizes K=fx=fy from this FOV; identity extrinsics.",
        )

    with server.gui.add_folder("Display"):
        gui_show_pcd = server.gui.add_checkbox("Show point cloud", initial_value=True)
        gui_show_pred = server.gui.add_checkbox("Show predicted trace", initial_value=True)
        gui_show_kp = server.gui.add_checkbox("Show current keypoints", initial_value=True)
        gui_show_frustum = server.gui.add_checkbox("Show camera frustum", initial_value=True)
        gui_show_axes = server.gui.add_checkbox("Show world axes", initial_value=True)
        gui_track_width = server.gui.add_slider(
            "Track width", min=0.5, max=8.0, step=0.5, initial_value=4.0
        )
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.006
        )
        gui_kp_size = server.gui.add_slider(
            "KP size", min=0.005, max=0.05, step=0.005, initial_value=0.015
        )
        gui_frustum_scale = server.gui.add_slider(
            "Frustum scale", min=0.01, max=0.5, step=0.01, initial_value=0.10
        )

    state["gui_show_pcd"] = gui_show_pcd
    state["gui_show_pred"] = gui_show_pred
    state["gui_show_kp"] = gui_show_kp
    state["gui_show_frustum"] = gui_show_frustum
    state["gui_show_axes"] = gui_show_axes
    state["gui_track_width"] = gui_track_width
    state["gui_point_size"] = gui_point_size
    state["gui_kp_size"] = gui_kp_size
    state["gui_frustum_scale"] = gui_frustum_scale

    # ----- callback bodies -----

    def _ingest_pil(pil: Image.Image) -> None:
        raw = np.asarray(pil, dtype=np.uint8)
        resized = _resize_to_square(raw, image_size)
        # Run DA-V2 once when the image is ingested; it's the most expensive
        # helper, and caching here means the user can re-segment / re-predict
        # without paying the depth cost again.
        d_aligned, _diag = da_v2.infer_aligned(
            resized, cfg.depth_align, log_min_train, log_max_train
        )
        depth_turbo = render_depth_rgb(d_aligned, log_min_train, log_max_train)  # (H, W, 3)
        depth_chw = torch.from_numpy(depth_turbo).permute(2, 0, 1).contiguous().float()

        state["image_resized"] = resized
        state["depth_aligned"] = d_aligned.astype(np.float32, copy=False)
        state["depth_turbo_chw"] = depth_chw
        state["all_segmentations"] = []
        state["selection_checkboxes"] = []
        state["last_pred"] = None
        _clear_handles(state["mask_handles"])
        _clear_handles(state["scene_handles"])

        # Show input + depth in the GUI panel. We replace the image attribute
        # in place rather than removing-and-re-adding so the panel stays put.
        if state["input_image_handle"] is None:
            state["input_image_handle"] = server.gui.add_image(
                resized, label="Input (resized)", format="jpeg"
            )
        else:
            state["input_image_handle"].image = resized
        depth_uint8 = (depth_turbo * 255.0).clip(0, 255).astype(np.uint8)
        if state.get("depth_image_handle") is None:
            state["depth_image_handle"] = server.gui.add_image(
                depth_uint8, label="Depth (DA-V2)", format="jpeg"
            )
        else:
            state["depth_image_handle"].image = depth_uint8

        set_status(
            f"image loaded ({raw.shape[1]}×{raw.shape[0]} → {image_size}²); "
            f"DA-V2 depth ready. Edit prompts, then click Run segmentation."
        )

    def do_upload() -> None:
        f = upload_btn.value
        if f is None:
            return
        try:
            pil = Image.open(io.BytesIO(f.content)).convert("RGB")
        except Exception as e:
            set_status(f"upload failed: {type(e).__name__}: {e}")
            return
        _ingest_pil(pil)

    def do_sample() -> None:
        if not _SAMPLE_IMAGE_PATH.exists():
            set_status(f"sample image not found: {_SAMPLE_IMAGE_PATH}")
            return
        try:
            pil = Image.open(_SAMPLE_IMAGE_PATH).convert("RGB")
        except Exception as e:
            set_status(f"sample load failed: {type(e).__name__}: {e}")
            return
        # Fill the prompts with the example that goes with this image.
        task_input.value = _SAMPLE_TASK
        classes_input.value = _SAMPLE_CLASSES
        _ingest_pil(pil)
        set_status(
            f"sample loaded (task={_SAMPLE_TASK!r}, classes={_SAMPLE_CLASSES!r}). "
            "Next: click Run segmentation, then Predict trace."
        )

    def do_segment() -> None:
        if state["image_resized"] is None:
            set_status("upload an image first.")
            return
        classes_raw = classes_input.value or ""
        classes = [c.strip() for c in classes_raw.split(",") if c.strip()]
        if not classes:
            set_status("type at least one class to segment.")
            return
        set_status(f"segmenting with classes {classes!r}...")
        # Compute up to the slider's max so the slider becomes a live display
        # filter — moving it doesn't re-run GroundingDINO.
        max_pool = int(top_k_slider.max)
        try:
            segs = gsam.segment_with_classes(
                state["image_resized"], classes=classes, top_k=max_pool,
            )
        except Exception as e:
            set_status(f"segmentation failed: {type(e).__name__}: {e}")
            return
        state["all_segmentations"] = segs
        rebuild_mask_ui()
        set_status(
            f"detected {len(segs)} mask(s); pick which to track, then Predict trace."
        )

    def rebuild_mask_ui() -> None:
        _clear_handles(state["mask_handles"])
        state["selection_checkboxes"] = []
        all_segs = state["all_segmentations"]
        if not all_segs:
            return
        top_k = int(top_k_slider.value)
        visible = all_segs[:top_k]

        folder = server.gui.add_folder("Detected masks")
        state["mask_handles"].append(folder)
        with folder:
            # Combined overlay first, so users see all visible candidates at a glance.
            combined = _make_combined_overlay(state["image_resized"], visible)
            combined_handle = server.gui.add_image(
                combined, label="All visible masks (combined)", format="jpeg"
            )
            state["mask_handles"].append(combined_handle)

            for i, seg in enumerate(visible):
                color = _MASK_COLORS[i % len(_MASK_COLORS)]
                thumb = _make_mask_thumbnail(
                    state["image_resized"], seg.mask, tuple(int(x) for x in color)
                )
                area_pct = float(seg.mask.mean()) * 100.0
                img_handle = server.gui.add_image(
                    thumb,
                    label=f"#{i} {seg.label}  conf={seg.confidence:.2f}  area={area_pct:.1f}%",
                    format="jpeg",
                )
                cb_handle = server.gui.add_checkbox(
                    f"track #{i}",
                    initial_value=(i < cfg.default_initial_selected),
                )
                state["mask_handles"].extend([img_handle, cb_handle])
                state["selection_checkboxes"].append((i, cb_handle))

    def collect_selected_masks() -> list[np.ndarray]:
        sel: list[np.ndarray] = []
        all_segs = state["all_segmentations"]
        for i, cb in state["selection_checkboxes"]:
            try:
                if cb.value and i < len(all_segs):
                    sel.append(all_segs[i].mask)
            except Exception:
                pass
        return sel

    def do_predict() -> None:
        if state["image_resized"] is None or state["depth_aligned"] is None:
            set_status("upload an image first.")
            return
        masks = collect_selected_masks()
        num_kp = int(num_kp_slider.value)
        if not masks:
            set_status(
                "no masks selected — sampling from a 60% center crop "
                "(consider running segmentation first)."
            )

        H, W = state["image_resized"].shape[:2]
        uv, rowcol, kp_status = sample_keypoints_from_union(
            masks, image_hw=(H, W), num_keypoints=num_kp, rng=np_rng,
        )

        d_aligned = state["depth_aligned"]
        zs = d_aligned[rowcol[:, 0], rowcol[:, 1]]
        zs = np.where(np.isfinite(zs) & (zs > 1e-3), zs, 0.0).astype(np.float32)
        current_kp_np = np.zeros((num_kp, 3), dtype=np.float32)
        current_kp_np[:, :2] = uv
        current_kp_np[:, 2] = zs

        image_chw = (
            torch.from_numpy(state["image_resized"]).permute(2, 0, 1).contiguous().float() / 255.0
        )
        current_kp_t = torch.from_numpy(current_kp_np)
        batch = _build_batch(
            image_chw=image_chw,
            depth_chw=state["depth_turbo_chw"],
            current_kp=current_kp_t,
            history_len=history_len,
            future_len=future_len,
            device=device,
        )
        _tokenize_into_batch(
            batch, task_input.value or "", tokenizer, pcfg.tokenizer_max_length, device,
        )

        set_status(f"predicting trace ({kp_status}, K={num_kp}, F={future_len})...")
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=dtype)
            if use_autocast else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            pred = policy.predict_trace(
                batch, num_steps=num_inference_steps, mask_all_history=True,
            )
        # `predict_trace` returns a (B, N, H, 3) tensor by default and a dict
        # `{"traces", "done_prob"}` when the checkpoint has trace_done_head=True.
        if isinstance(pred, dict):
            traces = pred["traces"].float()
            done_prob = pred["done_prob"].float()
        else:
            traces = pred.float()
            done_prob = None

        # Done-head truncate-and-freeze (docs/trace_done_head_plan.md §1.6).
        # Step-resolution predicted-valid mask used to mask the renderer.
        pred_valid_step: torch.Tensor | None = None
        if done_prob is not None:
            threshold = (
                cfg.done_threshold
                if cfg.done_threshold is not None
                else pcfg.done_threshold
            )
            # bspline: done head emits (1, N, H) directly
            # — see docs/trace_bspline_ctrl_pts_plan.md §3.
            pred_valid_step = (done_prob >= threshold)  # (1, N, H)

        anchor = batch[CURRENT_KP_KEY][:, :, None, :3]
        traces_abs = (traces * delta_scale_t + anchor)[0].cpu().numpy()  # (K, F, 3)
        pred_valid_step_np = (
            pred_valid_step[0].cpu().numpy() if pred_valid_step is not None else None
        )

        state["last_pred"] = {
            "current_kp": current_kp_np,
            "traces_abs": traces_abs,
            "pred_valid_step": pred_valid_step_np,
            "image": state["image_resized"],
            "depth": d_aligned,
        }
        rebuild_3d_scene()
        set_status(
            f"predicted {num_kp} tracks × {future_len} steps ({kp_status})."
        )

    def rebuild_3d_scene() -> None:
        pred = state["last_pred"]
        if pred is None:
            return
        H, W = pred["image"].shape[:2]
        K = _K_from_hfov(W, H, float(fov_slider.value))
        c2w = np.eye(4, dtype=np.float32)
        _build_scene(
            server, state, K=K, c2w=c2w,
            image=pred["image"], depth_meters=pred["depth"],
            current_kp_uv_z=pred["current_kp"], traces_abs=pred["traces_abs"],
            pixel_downsample=cfg.pixel_downsample,
            pred_valid_step=pred.get("pred_valid_step"),
        )

    def in_worker(fn) -> None:
        """Run `fn` on a daemon thread, serialized by `work_lock`."""
        def _wrapped() -> None:
            with work_lock:
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    logging.exception("worker raised")
                    set_status(f"error: {type(e).__name__}: {e}")
        threading.Thread(target=_wrapped, daemon=True).start()

    # ----- callback wiring -----

    @upload_btn.on_upload
    def _(_) -> None:
        in_worker(do_upload)

    @sample_btn.on_click
    def _(_) -> None:
        in_worker(do_sample)

    @segment_btn.on_click
    def _(_) -> None:
        in_worker(do_segment)

    @predict_btn.on_click
    def _(_) -> None:
        in_worker(do_predict)

    @top_k_slider.on_update
    def _(_) -> None:
        # Live filter — crop the cached segmentation pool to the new top-K
        # without re-running GroundingDINO/SAM.
        if state["all_segmentations"]:
            rebuild_mask_ui()

    @fov_slider.on_update
    def _(_) -> None:
        if state["last_pred"] is not None:
            in_worker(rebuild_3d_scene)

    # In-place updates for visibility / sizing — no rebuild needed.
    @gui_point_size.on_update
    def _(_) -> None:
        if state.get("pc_handle") is not None:
            state["pc_handle"].point_size = gui_point_size.value

    @gui_track_width.on_update
    def _(_) -> None:
        if state.get("pred_handle") is not None:
            state["pred_handle"].line_width = gui_track_width.value

    @gui_kp_size.on_update
    def _(_) -> None:
        if state.get("kp_handle") is not None:
            state["kp_handle"].point_size = gui_kp_size.value

    @gui_frustum_scale.on_update
    def _(_) -> None:
        if state.get("frustum_handle") is not None:
            state["frustum_handle"].scale = gui_frustum_scale.value

    @gui_show_pcd.on_update
    def _(_) -> None:
        if state.get("pc_handle") is not None:
            state["pc_handle"].visible = gui_show_pcd.value

    @gui_show_pred.on_update
    def _(_) -> None:
        if state.get("pred_handle") is not None:
            state["pred_handle"].visible = gui_show_pred.value

    @gui_show_kp.on_update
    def _(_) -> None:
        if state.get("kp_handle") is not None:
            state["kp_handle"].visible = gui_show_kp.value

    @gui_show_frustum.on_update
    def _(_) -> None:
        if state.get("frustum_handle") is not None:
            state["frustum_handle"].visible = gui_show_frustum.value

    @gui_show_axes.on_update
    def _(_) -> None:
        if state.get("axes_handle") is not None:
            state["axes_handle"].visible = gui_show_axes.value

    logging.info(f"viser server up at http://localhost:{cfg.port} (Ctrl+C to exit)")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("Shutting down")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
