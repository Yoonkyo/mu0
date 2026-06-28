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

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from lerobot.datasets.bspline_basis import build_bspline_basis
from lerobot.datasets.trace_depth import render_depth_rgb
from lerobot.utils.constants import OBS_IMAGES

IMAGE_KEY = f"{OBS_IMAGES}.cam_front"
DEPTH_KEY = f"{OBS_IMAGES}.cam_front_depth"
TRAJ_HISTORY_KEY = "traj_history"
VALID_HISTORY_KEY = "valid_history"
TRAJ_FUTURE_KEY = "traj_future"
VALID_FUTURE_KEY = "valid_future"
CURRENT_KP_KEY = "current_kp"
KEY_PAD_MASK_KEY = "key_pad_mask"
TASK_KEY = "task"
# B-spline control-point fit (docs/trace_bspline_ctrl_pts_plan.md). Emitted
# only when `bspline_n_ctrl` is set on the dataset; absent otherwise so the
# default path stays bit-identical.
CTRL_PTS_FUTURE_KEY = "ctrl_pts_future"
VALID_KP_KEY = "valid_kp"
# Per-keypoint DINO cluster id (TraceExtract `samples/cluster_ids.npy`). Emitted
# only when `load_cluster_ids=True`; padded slots get -1 in the collated batch.
# Used by the rigidity regularization loss (variance of intra-cluster pairwise
# control-point distances).
CLUSTER_IDS_KEY = "cluster_ids"


# Per-process LRU cache of mmap handles. Without this, eagerly mmapping every
# large array at dataset-construction time made fd use scale with the number of
# video dirs (~8 fds per dir) and blew past `ulimit -n` at a few hundred dirs.
# Here mmaps are opened on first access and evicted LRU, so fd count stays
# ~O(_MMAP_CACHE_MAX) regardless of dataset size. DataLoader workers fork after
# dataset construction, so each worker maintains its own cache.
_MMAP_CACHE_MAX = int(os.environ.get("TRACE_DATASET_MMAP_CACHE", "512"))
_mmap_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _mmap(path: Path) -> np.ndarray:
    key = str(path)
    arr = _mmap_cache.get(key)
    if arr is not None:
        _mmap_cache.move_to_end(key)
        return arr
    arr = np.load(path, mmap_mode="r")
    _mmap_cache[key] = arr
    while len(_mmap_cache) > _MMAP_CACHE_MAX:
        _mmap_cache.popitem(last=False)
    return arr


def _peek_npy_shape(path: Path) -> tuple[int, ...]:
    """Read a .npy header to get its shape AND verify the file isn't truncated.

    A header peek alone is not enough: an interrupted untar / partial copy can
    leave a `.npy` with a valid header but fewer payload bytes than the header
    claims. `numpy.memmap` then raises `ValueError: mmap length is greater than
    file size` at first read inside a DataLoader worker, which kills the run
    after the dir already passed the construction check.

    We re-peek shape/dtype/fortran-order, compute the expected payload byte
    count, and raise `ValueError` if `stat().st_size` is short — same exception
    family the dataset constructor already converts into a "skip this dir"
    warning, so partially-extracted dirs are dropped automatically.
    """
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
        header_end = f.tell()
    # int() avoids numpy scalar overflow on huge images.npy payloads.
    n_elems = int(np.prod(shape, dtype=np.int64)) if shape else 1
    expected = header_end + n_elems * dtype.itemsize
    actual = path.stat().st_size
    if actual < expected:
        raise ValueError(
            f"{path}: truncated .npy (header expects {expected} bytes incl. "
            f"header, on-disk {actual}; short by {expected - actual} bytes). "
            f"shape={shape} dtype={dtype} fortran={fortran_order}"
        )
    return shape


class _VideoHandle:
    """Lazily opens one TraceExtract video directory and keeps mmap handles.

    TraceExtract `output_structure.md`: `samples/*.npy` are concatenated across
    frames, indexed by `(frame_indices, offsets)`. `curated_training_texts.json`
    (episode root) holds two overlapping sets of text annotations: `chunk_texts`
    (contiguous, mutually exclusive chunks) and `merged_texts` (merges spanning
    multiple chunks). Both carry `start_idx`, `end_idx`, and three paraphrases
    `instruction_1/2/3`. For a frame t we enumerate all entries covering it and
    sample: pick an entry uniformly, then pick one of its 3 instructions.
    """

    def __init__(
        self,
        video_dir: str,
        future_len: int,
        history_len: int,
        n_min: int,
        load_depth: bool = False,
        bspline_n_ctrl: int | None = None,
        load_cluster_ids: bool = False,
    ):
        self.video_dir = Path(video_dir)
        samples = self.video_dir / "samples"
        self._samples_dir = samples
        self._load_depth = bool(load_depth)
        self._load_cluster_ids = bool(load_cluster_ids)

        # Small per-slot arrays: eagerly loaded, fully in memory, no fds retained.
        self.frame_indices = np.load(samples / "frame_indices.npy")
        self.offsets = np.load(samples / "offsets.npy")
        self.is_moving = np.load(samples / "is_moving.npy")

        # Large arrays: validate shapes (and on-disk byte counts) via .npy header
        # peek, then serve them lazily through `_mmap()` in the properties below.
        # The peek doubles as a truncation check — partially-extracted shards
        # raise ValueError here and the constructor's caller skips the dir.
        # Includes valid_steps* even though their shapes aren't compared against
        # anything: catching their truncation at construction is still cheaper
        # than crashing a worker mid-training.
        img_shape = _peek_npy_shape(self.video_dir / "images.npy")
        traj_shape = _peek_npy_shape(samples / "traj.npy")
        traj_history_shape = _peek_npy_shape(samples / "traj_history.npy")
        _peek_npy_shape(samples / "valid_steps.npy")
        _peek_npy_shape(samples / "valid_steps_history.npy")

        # `traj` index 0 is the current frame `t` (duplicated with traj_history[:, 0]),
        # so we need future_len + 1 columns to slice traj[:, 1 : future_len + 1] downstream.
        if traj_shape[1] < future_len + 1:
            raise ValueError(
                f"{self.video_dir}: traj has only {traj_shape[1]} steps, "
                f"need {future_len + 1} (index 0 = current frame, indices 1..F = future)"
            )
        if traj_history_shape[1] < history_len + 1:
            raise ValueError(
                f"{self.video_dir}: traj_history has only {traj_history_shape[1]} steps, "
                f"need {history_len + 1} (index 0 = current frame, indices 1..h = past)"
            )

        self.img_h, self.img_w = img_shape[1:3]

        if self._load_depth:
            depth_path = self.video_dir / "depth.npy"
            if not depth_path.exists():
                raise FileNotFoundError(f"{depth_path} not found (load_depth=True)")
            depth_shape = _peek_npy_shape(depth_path)
            if depth_shape[0] != img_shape[0]:
                raise ValueError(
                    f"{depth_path}: depth T={depth_shape[0]} != images T={img_shape[0]}"
                )
            if tuple(depth_shape[1:3]) != tuple(img_shape[1:3]):
                raise ValueError(
                    f"{depth_path}: depth HW={tuple(depth_shape[1:3])} "
                    f"!= images HW={tuple(img_shape[1:3])}"
                )

        if self._load_cluster_ids:
            cluster_path = samples / "cluster_ids.npy"
            if not cluster_path.exists():
                raise FileNotFoundError(
                    f"{cluster_path} not found (load_cluster_ids=True)"
                )
            cluster_shape = _peek_npy_shape(cluster_path)
            if cluster_shape[0] != traj_shape[0]:
                raise ValueError(
                    f"{cluster_path}: cluster_ids N={cluster_shape[0]} != "
                    f"traj N={traj_shape[0]}"
                )

        # Prebuild the list of slots with at least n_min moving keypoints so we never
        # draw an unusable sample. is_moving is in memory, so this is cheap.
        # When bspline_n_ctrl is set we additionally require at least one kp in the
        # slot to have >= D valid future steps; otherwise the slot is guaranteed to
        # produce a "no eligible kp" sample at runtime, triggering the retry path
        # in __getitem__. The pre-filter shifts that cost from per-batch retries to
        # this one-time scan. valid_steps is mmap'd through the LRU cache; the
        # eligibility slice is small (~kps × F bytes per slot).
        valid_steps_for_filter = None
        if bspline_n_ctrl is not None:
            valid_steps_for_filter = _mmap(self._samples_dir / "valid_steps.npy")

        usable = []
        for slot in range(len(self.frame_indices)):
            lo, hi = int(self.offsets[slot]), int(self.offsets[slot + 1])
            if int(self.is_moving[lo:hi].sum()) < n_min:
                continue
            if valid_steps_for_filter is not None:
                vfut = valid_steps_for_filter[lo:hi, 1 : future_len + 1]
                if not (vfut.sum(axis=1) >= bspline_n_ctrl).any():
                    continue
            usable.append(slot)
        self.usable_slots = usable

        # Text annotation index: flat list of (start, end, [instruction_1/2/3])
        # over chunk_texts + merged_texts (inclusive ranges). Empty list if the
        # file is missing or has no entries. We eagerly materialize the
        # instruction lists so no per-__getitem__ JSON load is needed.
        texts_path = self.video_dir / "curated_training_texts.json"
        self.text_entries: list[tuple[int, int, list[str]]] = []
        if texts_path.exists():
            with open(texts_path) as f:
                td = json.load(f)
            for section in ("chunk_texts", "merged_texts"):
                for entry in td.get(section, []):
                    instructions = [
                        entry[k]
                        for k in ("instruction_1", "instruction_2", "instruction_3")
                        if k in entry and entry[k]
                    ]
                    if not instructions:
                        continue
                    self.text_entries.append(
                        (int(entry["start_idx"]), int(entry["end_idx"]), instructions)
                    )

    @property
    def images(self) -> np.ndarray:
        return _mmap(self.video_dir / "images.npy")

    @property
    def traj(self) -> np.ndarray:
        return _mmap(self._samples_dir / "traj.npy")

    @property
    def traj_history(self) -> np.ndarray:
        return _mmap(self._samples_dir / "traj_history.npy")

    @property
    def valid_steps(self) -> np.ndarray:
        return _mmap(self._samples_dir / "valid_steps.npy")

    @property
    def valid_steps_history(self) -> np.ndarray:
        return _mmap(self._samples_dir / "valid_steps_history.npy")

    @property
    def depth(self) -> np.ndarray | None:
        if not self._load_depth:
            return None
        return _mmap(self.video_dir / "depth.npy")

    @property
    def cluster_ids(self) -> np.ndarray | None:
        if not self._load_cluster_ids:
            return None
        return _mmap(self._samples_dir / "cluster_ids.npy")

    def caption_entries_for_frame(self, t: int) -> list[list[str]]:
        """Return all instruction-lists (one per chunk/merged entry) covering frame t."""
        return [inst for start, end, inst in self.text_entries if start <= t <= end]


class TraceDataset(Dataset):
    """Dataset for trace prediction on TraceExtract outputs.

    Each sample is a tuple (image at frame t, sampled keypoint history/future traces,
    current-frame keypoints, validity masks, language caption). Keypoint trajectories
    are retargeted (arc-length) and expressed in frame-`t` camera coordinates — see
    `TraceExtract/docs/output_structure.md`. xy is normalized to `[-1, 1]` by image
    `(W, H)`; z stays in raw meters.

    Time-axis convention (post-split, post-reversal at load):
      - `traj_history[:, 0]` is the OLDEST past step, `traj_history[:, h-1]` is `t-1`.
      - `current_kp` is the current frame `t` (separate `(N, 3)` tensor).
      - `traj_future[:, 0]` is `t+1` and `traj_future[:, F-1]` is `t+F`.
    `cat([history, current_kp[:, None], future])` along the time axis is strictly
    chronological. The model does not use that concatenation — `current_kp` is consumed
    only by the per-kp side branches (uv-pos Fourier embedding, DINO grid_sample, and
    the delta-target anchor reconstruction at eval/vis time). See
    `docs/trace_current_kp_split_plan.md` (and its predecessor
    `docs/trace_time_axis_fix_plan.md`).

    `history_len = h` now means "h past-only frames (`t-h..t-1`)"; that is one frame
    of past context more than the pre-split spelling, balanced by removing the
    current-frame slot from the suffix.

    When `delta_scale` is provided, `traj_future` is converted to anchor-relative deltas:
    `(future_xyz - current_kp) / delta_scale` (xy in normalized space, z in meters), so
    typical targets land in ~[-1, 1]. The anchor is mathematically identical to the
    pre-split anchor (TraceExtract index 0 in either spelling), so existing delta_stats
    JSONs are still valid.

    `traj_history[..., :3]` is centered on the current frame: replaced with
    `(history_abs - current_kp)` after normalization (no rescale; see plan §7 — the
    `delta_scale` used for the future is intentionally NOT applied here).

    """

    def __init__(
        self,
        video_dirs: list[str],
        future_len: int = 32,
        history_len: int = 16,
        n_min: int = 1,
        n_max: int = 32,
        image_size: int = 224,
        seed: int = 0,
        min_future_motion: float = 0.0,
        delta_scale: tuple[float, float, float] | None = None,
        load_depth: bool = False,
        depth_log_range: tuple[float, float] | None = None,
        rgb_jitter_strength: float = 0.0,
        depth_noise_std: float = 0.0,
        bspline_n_ctrl: int | None = None,
        bspline_reg_lambda: float = 0.0,
        bspline_reg_order: int = 1,
        bspline_ctrl_clip: float | None = None,
        cache_dir: str | Path | None = None,
        show_progress: bool = False,
        load_cluster_ids: bool = False,
    ):
        if n_min < 1 or n_max < n_min:
            raise ValueError(f"bad n_min/n_max: {n_min=} {n_max=}")
        if image_size < 1:
            raise ValueError(f"bad image_size: {image_size=}")
        if min_future_motion < 0.0:
            raise ValueError(f"min_future_motion must be >= 0: {min_future_motion=}")
        if delta_scale is not None:
            if len(delta_scale) != 3 or any(float(s) <= 0.0 for s in delta_scale):
                raise ValueError(f"delta_scale must be (sx, sy, sz) with all > 0: {delta_scale}")
        if load_depth:
            if depth_log_range is None:
                raise ValueError("load_depth=True requires depth_log_range=(log_min, log_max)")
            if len(depth_log_range) != 2 or depth_log_range[1] <= depth_log_range[0]:
                raise ValueError(f"depth_log_range must be (log_min, log_max) with max > min: {depth_log_range}")
        if rgb_jitter_strength < 0.0:
            raise ValueError(f"rgb_jitter_strength must be >= 0: {rgb_jitter_strength=}")
        if depth_noise_std < 0.0:
            raise ValueError(f"depth_noise_std must be >= 0: {depth_noise_std=}")
        self.future_len = future_len
        self.history_len = history_len
        self.n_min = n_min
        self.n_max = n_max
        self.image_size = image_size
        self.min_future_motion = float(min_future_motion)
        self.delta_scale = (
            np.asarray(delta_scale, dtype=np.float32) if delta_scale is not None else None
        )
        self.load_depth = bool(load_depth)
        self.depth_log_range = (
            (float(depth_log_range[0]), float(depth_log_range[1])) if depth_log_range is not None else None
        )
        self.rgb_jitter_strength = float(rgb_jitter_strength)
        self.depth_noise_std = float(depth_noise_std)
        # B-spline ctrl-pt fit (docs/trace_bspline_ctrl_pts_plan.md). When set,
        # `__getitem__` emits a (n, D, 3) least-squares fit of the future
        # trajectory against a fixed cubic B-spline basis with a prepended
        # anchor row at t=0. None disables the path.
        self.load_cluster_ids = bool(load_cluster_ids)
        self.bspline_n_ctrl = bspline_n_ctrl
        # Tikhonov / ridge regularization on the per-kp lstsq. Augments the
        # weighted basis with `lambda * Gamma` (Gamma = k-th order finite
        # difference operator on consecutive ctrl pts) and the target with
        # zeros, then solves the same lstsq. lambda > 0 pulls adjacent ctrl
        # pts together (order=1 → even spacing, order=2 → smooth polygon),
        # which compresses the FM target distribution and stabilizes flow
        # matching. lambda=0 is bit-identical to the original fit.
        # `bspline_ctrl_clip` is a post-fit element-wise hard clip on P_gt
        # (e.g. 1.5 keeps ctrl pts within 1.5 of the scaled-delta range).
        self.bspline_reg_lambda = float(bspline_reg_lambda)
        self.bspline_reg_order = int(bspline_reg_order)
        self.bspline_ctrl_clip = (
            float(bspline_ctrl_clip) if bspline_ctrl_clip is not None else None
        )
        if self.bspline_reg_lambda < 0.0:
            raise ValueError(
                f"bspline_reg_lambda must be >= 0; got {self.bspline_reg_lambda}"
            )
        if self.bspline_reg_order not in (1, 2):
            raise ValueError(
                f"bspline_reg_order must be 1 or 2; got {self.bspline_reg_order}"
            )
        if self.bspline_ctrl_clip is not None and self.bspline_ctrl_clip <= 0.0:
            raise ValueError(
                f"bspline_ctrl_clip must be > 0 when set; got {self.bspline_ctrl_clip}"
            )
        if self.bspline_n_ctrl is not None:
            if self.bspline_n_ctrl not in (4, 7, 10):
                raise ValueError(
                    f"bspline_n_ctrl must be in {{4, 7, 10}}; got {self.bspline_n_ctrl}"
                )
            if self.bspline_reg_order >= self.bspline_n_ctrl:
                raise ValueError(
                    f"bspline_reg_order ({self.bspline_reg_order}) must be < "
                    f"bspline_n_ctrl ({self.bspline_n_ctrl}) so the difference "
                    f"operator has at least one row."
                )
            # (H+1, D) basis cached once per worker. CPU float32 — lstsq runs
            # in the dataloader worker on CPU.
            self._bspline_basis: torch.Tensor | None = build_bspline_basis(
                self.bspline_n_ctrl, self.future_len, include_anchor=True,
                dtype=torch.float32, device="cpu",
            )
            # Pre-scaled (lambda * Gamma), shape (D - order, D). Built once.
            # None when reg is disabled so the fast path stays a single lstsq.
            if self.bspline_reg_lambda > 0.0:
                self._bspline_reg_matrix: torch.Tensor | None = (
                    self._build_reg_matrix(
                        self.bspline_n_ctrl, self.bspline_reg_order
                    ) * self.bspline_reg_lambda
                )
            else:
                self._bspline_reg_matrix = None
        else:
            self._bspline_basis = None
            self._bspline_reg_matrix = None
        # Photometric jitter on the RGB frame only. Depth's turbo-LUT encoding is
        # pinned to depth_log_range, so we must NOT photo-jitter it (would silently
        # remap meters → colors). Depth noise is applied in raw meters before the
        # colormap, preserving the meter-to-color mapping for valid pixels.
        if self.rgb_jitter_strength > 0.0:
            from torchvision.transforms import v2 as _v2_transforms
            s = self.rgb_jitter_strength
            self._rgb_jitter = _v2_transforms.ColorJitter(
                brightness=s, contrast=s, saturation=s, hue=min(s * 0.5, 0.5)
            )
        else:
            self._rgb_jitter = None

        resolved: list[str] = []
        for pattern in video_dirs:
            matches = sorted(glob.glob(pattern))
            if not matches and Path(pattern).is_dir():
                matches = [pattern]
            resolved.extend(matches)
        if not resolved:
            raise ValueError(f"no video dirs matched: {video_dirs}")

        # Resolve cache file path (one-time cache of the scan result, keyed by
        # filter config + per-dir mtime signature). Filter config goes into the
        # filename; mtime signature goes into the file body. Mismatch on either
        # → recompute. See TraceDataset docstring for invalidation rules.
        cache_path: Path | None = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_path = cache_dir / self._cache_filename(resolved)

        loaded = self._try_load_cache(cache_path, resolved) if cache_path else None
        if loaded is not None:
            self._videos, self._index = loaded
            # Cached video handles may have been pickled with a different
            # `_load_cluster_ids`/`_load_depth` setting; refresh both so the
            # cluster_ids / depth lazy-load path matches the current config
            # without invalidating the (expensive) usable-slot scan.
            for v in self._videos:
                v._load_cluster_ids = self.load_cluster_ids
                v._load_depth = self.load_depth
            self._rng = random.Random(seed)
            self._missing_logged: set[str] = set()
            logging.info(
                f"TraceDataset: loaded {len(self._index)} samples from cache "
                f"{cache_path} (skipped scan over {len(resolved)} dirs)"
            )
            return

        self._videos: list[_VideoHandle] = []
        self._index: list[tuple[int, int]] = []
        skipped: list[tuple[str, str]] = []
        iterator = tqdm(
            resolved,
            desc="TraceDataset: scanning video dirs",
            unit="dir",
            disable=not show_progress,
        )
        for vdir in iterator:
            # Dirs like visualize_caption/ get caught by the * glob but aren't real
            # TraceExtract outputs. Skip on any load-time error (missing images.npy,
            # missing sample arrays, wrong shapes) rather than killing the job.
            try:
                handle = _VideoHandle(
                    vdir, future_len, history_len, n_min,
                    load_depth=self.load_depth,
                    bspline_n_ctrl=self.bspline_n_ctrl,
                    load_cluster_ids=self.load_cluster_ids,
                )
            except (FileNotFoundError, ValueError, OSError, EOFError) as e:
                skipped.append((vdir, f"{type(e).__name__}: {e}"))
                continue
            if not handle.usable_slots:
                reason = "no usable slots (fewer than n_min moving keypoints"
                if self.bspline_n_ctrl is not None:
                    reason += f" or no kp with >= {self.bspline_n_ctrl} valid future steps"
                reason += ")"
                skipped.append((vdir, reason))
                continue
            video_idx = len(self._videos)
            self._videos.append(handle)
            self._index.extend((video_idx, slot) for slot in handle.usable_slots)

        if skipped:
            logging.warning(
                f"TraceDataset: skipped {len(skipped)}/{len(resolved)} video dir(s)"
            )
            for vdir, reason in skipped:
                logging.warning(f"  skip {vdir}: {reason}")

        if not self._index:
            raise ValueError(f"no usable slots found in {len(resolved)} video(s)")

        if cache_path is not None:
            self._save_cache(cache_path, resolved)

        self._rng = random.Random(seed)
        # Per-worker dedup so we log a missing/unreadable dir once, not per sample.
        # Forked workers each get their own copy; expect a handful of warnings (one
        # per worker) the first time a flaky dir is hit, then silence.
        self._missing_logged: set[str] = set()

    @staticmethod
    def _build_reg_matrix(n_ctrl: int, order: int) -> torch.Tensor:
        """Finite-difference operator on consecutive ctrl pts.

        order=1 → (n_ctrl-1, n_ctrl) with rows [-1, 1] sliding right.
        order=2 → (n_ctrl-2, n_ctrl) with rows [-1, 2, -1] sliding right.
        Used as Tikhonov term: minimize ||A P - B||^2 + ||lambda Gamma P||^2.
        """
        if order == 1:
            gamma = torch.zeros(n_ctrl - 1, n_ctrl, dtype=torch.float32)
            for i in range(n_ctrl - 1):
                gamma[i, i] = -1.0
                gamma[i, i + 1] = 1.0
        elif order == 2:
            gamma = torch.zeros(n_ctrl - 2, n_ctrl, dtype=torch.float32)
            for i in range(n_ctrl - 2):
                gamma[i, i] = -1.0
                gamma[i, i + 1] = 2.0
                gamma[i, i + 2] = -1.0
        else:
            raise ValueError(f"order must be 1 or 2; got {order}")
        return gamma

    def _cache_filename(self, resolved: list[str]) -> str:
        """Hash filter config + sorted dir set into a cache filename.

        Anything that affects `_videos` / `_index` content goes into the hash
        (filter config + the resolved dir set). Per-dir contents are validated
        separately via mtime — see ``_per_dir_signature``.
        """
        sig = {
            "future_len": self.future_len,
            "history_len": self.history_len,
            "n_min": self.n_min,
            "bspline_n_ctrl": self.bspline_n_ctrl,
            "load_depth": self.load_depth,
            "video_dirs": sorted(resolved),
        }
        h = hashlib.sha256(
            json.dumps(sig, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"trace_dataset_cache_{h}.pkl"

    @staticmethod
    def _per_dir_signature(resolved: list[str]) -> dict[str, float | None]:
        """Per-dir max-mtime over the .npy files we read during the scan.

        Returns ``None`` for dirs whose required files are missing (those
        get skipped during the rescan anyway). A mismatch on any single dir
        invalidates the entire cache.
        """
        sig: dict[str, float | None] = {}
        for vdir in resolved:
            samples = Path(vdir) / "samples"
            mtimes: list[float] = []
            for fname in (
                "frame_indices.npy",
                "offsets.npy",
                "is_moving.npy",
                "valid_steps.npy",
            ):
                p = samples / fname
                try:
                    mtimes.append(p.stat().st_mtime)
                except (FileNotFoundError, OSError):
                    pass
            sig[str(vdir)] = max(mtimes) if mtimes else None
        return sig

    def _try_load_cache(
        self, cache_path: Path, resolved: list[str],
    ) -> tuple[list[_VideoHandle], list[tuple[int, int]]] | None:
        """Return cached `(_videos, _index)` if the cache exists and the
        per-dir mtime signature matches; otherwise None."""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError, OSError, AttributeError) as e:
            logging.warning(
                f"TraceDataset: cache at {cache_path} unreadable "
                f"({type(e).__name__}: {e}); rescanning"
            )
            return None

        cached_per_dir = data.get("per_dir_signature")
        current_per_dir = self._per_dir_signature(resolved)
        if cached_per_dir != current_per_dir:
            logging.info(
                f"TraceDataset: cache at {cache_path} stale (per-dir contents "
                f"changed); rescanning"
            )
            return None

        videos = data.get("videos")
        index = data.get("index")
        if not isinstance(videos, list) or not isinstance(index, list):
            logging.warning(
                f"TraceDataset: cache at {cache_path} malformed; rescanning"
            )
            return None
        return videos, index

    def _save_cache(self, cache_path: Path, resolved: list[str]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(
                    {
                        "per_dir_signature": self._per_dir_signature(resolved),
                        "videos": self._videos,
                        "index": self._index,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            # Atomic-rename over any existing cache to handle concurrent DDP
            # ranks racing on the same path; last writer wins, no partial files.
            tmp.replace(cache_path)
            logging.info(f"TraceDataset: saved cache to {cache_path}")
        except OSError as e:
            logging.warning(
                f"TraceDataset: failed to save cache to {cache_path}: {e}"
            )
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Robustness: shape headers were validated at construction, but the underlying
        # .npy / image / depth files can disappear mid-run (NFS hiccups, external
        # cleanup) since they're mmapped lazily. Catch I/O failures and retry with a
        # different random sample so a single flaky dir doesn't kill an 8h run.
        # Exception family rationale (active-untar scenario):
        #   FileNotFoundError — file existed at construction but was unlinked / not yet
        #     present when worker reads.
        #   OSError           — generic filesystem / NFS read errors.
        #   ValueError        — `numpy.memmap` raising "mmap length is greater than
        #     file size" when a .npy is truncated (header valid, payload short).
        #   EOFError          — `np.load(mmap_mode="r")` reading a file whose magic
        #     header was overwritten / not yet flushed (untar mid-write produces a
        #     0-byte or sub-magic file). EOFError does NOT inherit from OSError on
        #     Python 3.12, so it has to be listed explicitly.
        #
        # Bspline mode also retries on "no eligible kp": if every kp has fewer
        # than D valid future steps, the sample contributes zero FM gradient
        # AND wastes compute on the noise-only ctrl-pt tokens. Treating it as
        # a soft skip mirrors the existing I/O retry path. See
        # docs/trace_bspline_ctrl_pts_plan.md.
        n = len(self._index)
        try:
            sample = self._get_one(idx)
            if not self._is_bspline_unusable(sample):
                return sample
            last_err: Exception | None = None
        except (FileNotFoundError, OSError, ValueError, EOFError) as e:
            video_idx, _slot = self._index[idx]
            bad_dir = str(self._videos[video_idx].video_dir)
            if bad_dir not in self._missing_logged:
                self._missing_logged.add(bad_dir)
                logging.warning(
                    f"TraceDataset: I/O error for {bad_dir}: {type(e).__name__}: {e} "
                    f"— retrying with a different sample (silencing further warnings for this dir)"
                )
            last_err = e
            sample = None  # type: ignore[assignment]

        for _ in range(16):
            new_idx = self._rng.randint(0, n - 1)
            if new_idx == idx:
                continue
            try:
                cand = self._get_one(new_idx)
            except (FileNotFoundError, OSError, ValueError, EOFError) as e:
                last_err = e
                continue
            if not self._is_bspline_unusable(cand):
                return cand
        if sample is not None:
            # Bspline-unusable last resort: return the original (zeroed P_gt,
            # all valid_kp=False). The trainer's loss mask makes it a no-op.
            return sample
        assert last_err is not None
        raise last_err

    def _is_bspline_unusable(self, sample: dict[str, Any]) -> bool:
        """In bspline mode, True iff the sample has zero kps with >= D valid
        future steps (so no kp contributes to the FM loss). Always False when
        bspline mode is off."""
        if self._bspline_basis is None:
            return False
        return not bool(sample[VALID_KP_KEY].any().item())

    def _get_one(self, idx: int) -> dict[str, Any]:
        video_idx, slot = self._index[idx]
        v = self._videos[video_idx]

        t = int(v.frame_indices[slot])
        lo, hi = int(v.offsets[slot]), int(v.offsets[slot + 1])

        is_moving = np.asarray(v.is_moving[lo:hi])
        moving_rows = np.flatnonzero(is_moving)

        # Read `history_len + 1` raw columns from TraceExtract traj_history. Index 0 is
        # the current frame `t` (becomes `current_kp`); indices 1..h are past steps,
        # reversed so post-flip index 0 = oldest (== t-h) and index h-1 = t-1. The
        # past chunk feeds the model's history token stream; the current frame is
        # consumed only by the per-kp side branches (uv-pos / DINO / delta anchor).
        # See docs/trace_current_kp_split_plan.md (and predecessor
        # docs/trace_time_axis_fix_plan.md).
        hist_raw = np.asarray(
            v.traj_history[lo:hi, : self.history_len + 1], dtype=np.float32
        )
        vsh_raw = np.asarray(
            v.valid_steps_history[lo:hi, : self.history_len + 1], dtype=np.bool_
        )
        current_kp_raw = hist_raw[:, 0]                                               # (M, 3)
        current_valid_raw = vsh_raw[:, 0]                                             # (M,)
        traj_history_all = np.ascontiguousarray(hist_raw[:, 1:][:, ::-1])             # (M, h, 3)
        valid_history_all = np.ascontiguousarray(vsh_raw[:, 1:][:, ::-1])             # (M, h)

        # Optional: drop keypoints whose future xy barely moves, measured in normalized
        # [-1, 1] coords so the threshold is resolution-independent. If the filter empties
        # the pool we fall back to the unfiltered moving_rows (always sampleable).
        if self.min_future_motion > 0.0 and moving_rows.size > 0:
            fut_xy = np.asarray(v.traj[lo:hi, 1 : self.future_len + 1, :2], dtype=np.float32)[moving_rows]
            vfut = np.asarray(v.valid_steps[lo:hi, 1 : self.future_len + 1], dtype=np.bool_)[moving_rows]
            vfut &= np.isfinite(fut_xy).all(axis=-1)
            scale = np.asarray([v.img_w, v.img_h], dtype=np.float32)
            xy_norm = fut_xy / scale * 2.0 - 1.0  # (N, F, 2) in [-1, 1]
            has_valid = vfut.any(axis=1)
            first_idx = np.argmax(vfut, axis=1)
            ref = xy_norm[np.arange(xy_norm.shape[0]), first_idx]  # (N, 2)
            dist = np.linalg.norm(xy_norm - ref[:, None, :], axis=-1)  # (N, F)
            dist = np.where(vfut, dist, 0.0)
            span = np.where(has_valid, dist.max(axis=1), 0.0)
            keep = span >= self.min_future_motion
            if keep.any():
                moving_rows = moving_rows[keep]

        # P1 §3.9 (post-split): keep only keypoints whose CURRENT-FRAME step
        # (== `current_kp`, == TraceExtract index 0) is valid, finite in all 3 axes,
        # and whose uv lies in [-1, 1]². The per-kp 2D positional embedding in
        # modeling_{smolvla,pi05}.py consumes `current_kp[:, :2]`; OOB or invalid
        # values would make that embedding ill-defined. When delta_scale is enabled
        # we also use `current_kp` as the anchor for the future delta, so z must be
        # finite too. Mathematics identical to the pre-split filter (it gated on the
        # same TraceExtract slot, just spelled `traj_history[:, h-1]` post-reversal).
        if moving_rows.size > 0:
            last_xyz = current_kp_raw[moving_rows]
            scale = np.asarray([v.img_w, v.img_h], dtype=np.float32)
            last_uv = last_xyz[:, :2] / scale * 2.0 - 1.0
            last_valid = current_valid_raw[moving_rows]
            last_finite = np.isfinite(last_xyz).all(axis=1)
            in_bounds = (np.abs(last_uv) <= 1.0).all(axis=1)
            keep = last_valid & last_finite & in_bounds
            if keep.any():
                moving_rows = moving_rows[keep]

        # B-spline mode: drop kps that don't have at least D valid future
        # steps. Without this, ineligible kps still occupy suffix slots
        # (carrying noise-only ctrl pts) and consume forward compute despite
        # contributing zero FM gradient. See
        # docs/trace_bspline_ctrl_pts_plan.md.
        if self._bspline_basis is not None and moving_rows.size > 0:
            vfut = np.asarray(
                v.valid_steps[lo:hi, 1 : self.future_len + 1], dtype=np.bool_,
            )[moving_rows]
            fut_xyz = np.asarray(
                v.traj[lo:hi, 1 : self.future_len + 1, :3], dtype=np.float32,
            )[moving_rows]
            vfut &= np.isfinite(fut_xyz).all(axis=-1)
            keep = vfut.sum(axis=1) >= self.bspline_n_ctrl
            if keep.any():
                moving_rows = moving_rows[keep]

        num_moving = moving_rows.size
        # When the motion filter drops below n_min we accept the smaller pool rather than
        # padding back to n_min (by design: static keypoints are useless supervision).
        n_lo = min(self.n_min, num_moving)
        n_hi = min(self.n_max, num_moving)
        n = self._rng.randint(n_lo, n_hi)
        picked = self._rng.sample(range(num_moving), n)
        rows = moving_rows[np.asarray(picked)]

        # TraceExtract `traj` index 0 == current frame (duplicated with traj_history[:, 0]
        # pre-reversal). We slice [1 : F+1] so traj_future contains only genuinely
        # future steps t+1..t+F. With the current_kp split, the chronological time
        # axis is `cat([traj_history, current_kp[:, None], traj_future])`: oldest
        # -> ... -> t-1 -> t -> +1 -> ... -> +F.
        traj_future = np.asarray(v.traj[lo:hi, 1 : self.future_len + 1], dtype=np.float32)[rows]
        valid_future = np.asarray(v.valid_steps[lo:hi, 1 : self.future_len + 1], dtype=np.bool_)[rows]
        traj_history = traj_history_all[rows].copy()
        valid_history = valid_history_all[rows].copy()
        current_kp = current_kp_raw[rows].copy()                                       # (n, 3)

        # Combine the stored mask with a finiteness check on (x, y, z): `-inf` padding
        # can leak in through rows that the mask left True. See output_structure.md §4.
        finite_future = np.isfinite(traj_future).all(axis=-1)
        finite_history = np.isfinite(traj_history).all(axis=-1)
        valid_future &= finite_future
        valid_history &= finite_history

        # xy: pixel coords in original image size → [-1, 1]; z: raw meters.
        scale = np.asarray([v.img_w, v.img_h], dtype=np.float32)
        traj_future[..., :2] = traj_future[..., :2] / scale * 2.0 - 1.0
        traj_history[..., :2] = traj_history[..., :2] / scale * 2.0 - 1.0
        current_kp[..., :2] = current_kp[..., :2] / scale * 2.0 - 1.0
        # Zero-out invalid slots post-normalization so they are a neutral (0, 0, 0) input;
        # loss/attention still mask them, this just keeps the numbers sane. The row filter
        # above guarantees `current_kp` is valid + finite, so no analogous mask there.
        traj_future = np.where(finite_future[..., None], traj_future, 0.0)
        traj_history = np.where(finite_history[..., None], traj_history, 0.0)

        # Anchor-relative future: predicting (future - current_xyz) / delta_scale is an
        # easier target than the absolute trajectory (smaller variance, matched to the
        # flow-matching noise range). The row filter above guarantees `current_kp` is
        # valid and finite in all 3 axes. Mathematically identical to the pre-split
        # anchor (TraceExtract index 0 == post-reversal traj_history[:, h-1]), so
        # existing delta_stats JSONs are still valid post-split.
        if self.delta_scale is not None:
            anchor = current_kp[:, None, :3]                                           # (n, 1, 3)
            traj_future[..., :3] = (traj_future[..., :3] - anchor) / self.delta_scale
            traj_future = np.where(finite_future[..., None], traj_future, 0.0)

        # Anchor-relative history (plan §1 decision 8). Subtraction only; the
        # future's `delta_scale` is intentionally NOT applied here (see plan §7
        # for the rescaling discussion — option 1 is "reuse delta_scale", deferred).
        traj_history[..., :3] = traj_history[..., :3] - current_kp[:, None, :3]
        traj_history = np.where(finite_history[..., None], traj_history, 0.0)

        image = np.array(v.images[t])  # (H_img, W_img, 3) uint8, copied off mmap
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        # Photometric jitter (brightness/contrast/saturation/hue) on RGB only.
        # ColorJitter produces values in [0, 1] given a [0, 1] input, so the
        # downstream resize is unchanged.
        if self._rgb_jitter is not None:
            image = self._rgb_jitter(image)
        # Resize to a common (image_size, image_size) so the collate can stack across
        # dirs with different native resolutions. xy normalization is already in
        # [-1, 1] over the original frame, which is invariant under uniform resize.
        if image.shape[-2] != self.image_size or image.shape[-1] != self.image_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Two-step sample: pick one chunk/merged entry covering frame t, then
        # pick one of its 3 paraphrases. Empty task => downstream substitutes a
        # default "predict future trajectory" prompt.
        entries = v.caption_entries_for_frame(t)
        if entries:
            entry = self._rng.choice(entries)
            task = self._rng.choice(entry)
        else:
            task = ""

        out = {
            IMAGE_KEY: image,
            TRAJ_HISTORY_KEY: torch.from_numpy(traj_history),
            VALID_HISTORY_KEY: torch.from_numpy(valid_history),
            TRAJ_FUTURE_KEY: torch.from_numpy(traj_future),
            VALID_FUTURE_KEY: torch.from_numpy(valid_future),
            CURRENT_KP_KEY: torch.from_numpy(current_kp),
            TASK_KEY: task,
        }

        if self.load_cluster_ids and v.cluster_ids is not None:
            cids = np.asarray(v.cluster_ids[lo:hi], dtype=np.int64)[rows]    # (n,)
            out[CLUSTER_IDS_KEY] = torch.from_numpy(cids)

        # B-spline ctrl-pt fit (docs/trace_bspline_ctrl_pts_plan.md). Per-kp
        # weighted least squares: rows where `valid_future[i, t] == False`
        # are zeroed on BOTH sides (basis row * 0, target row * 0) so they
        # contribute trivial 0 = 0 to the residual instead of pulling P
        # toward zero (which the original "lstsq on zero-padded target"
        # silently did, biasing P_gt for kps with short valid futures).
        # `valid_kp` becomes True for any kp with at least D valid future
        # steps — fewer rows than that risks rank-deficient solves; we
        # zero P_gt and let the trainer drop those kps from FM loss.
        if self._bspline_basis is not None:
            future_t = torch.from_numpy(traj_future)               # (n, H, 3)
            valid_future_t = torch.from_numpy(valid_future)        # (n, H)
            n_kp, H, _ = future_t.shape
            D = self.bspline_n_ctrl

            # Validity mask incl. anchor row (anchor is always valid).
            m_anchor = torch.ones(n_kp, 1, dtype=torch.float32)
            m_future = valid_future_t.to(torch.float32)
            M = torch.cat([m_anchor, m_future], dim=1)             # (n, H+1)

            # Anchor-prepended target.
            anchor_row = torch.zeros(n_kp, 1, 3, dtype=future_t.dtype)
            target = torch.cat([anchor_row, future_t], dim=1)      # (n, H+1, 3)

            # Row-weight basis and target so invalid rows drop out of the
            # least-squares objective. Batched per-kp.
            basis_per_kp = self._bspline_basis.unsqueeze(0).expand(n_kp, -1, -1)
            A_w = basis_per_kp * M.unsqueeze(-1)                   # (n, H+1, D)
            B_w = target * M.unsqueeze(-1)                         # (n, H+1, 3)

            # Tikhonov augmentation: stack `lambda * Gamma` below A_w and a
            # zero block below B_w. Solving the same lstsq on the augmented
            # system minimizes ||A P - B||^2 + ||lambda Gamma P||^2, which
            # pulls adjacent ctrl pts toward each other (order=1) or toward
            # collinearity (order=2). lambda=0 keeps the matrix unaugmented
            # and the solve bit-identical to the pre-reg path.
            if self._bspline_reg_matrix is not None:
                R = self._bspline_reg_matrix                              # (D-k, D)
                R_batch = R.unsqueeze(0).expand(n_kp, -1, -1)             # (n, D-k, D)
                zeros_pad = torch.zeros(
                    n_kp, R.shape[0], 3, dtype=B_w.dtype
                )
                A_w = torch.cat([A_w, R_batch], dim=1)                    # (n, H+1+D-k, D)
                B_w = torch.cat([B_w, zeros_pad], dim=1)                  # (n, H+1+D-k, 3)
            P_gt = torch.linalg.lstsq(A_w, B_w).solution                  # (n, D, 3)

            # Optional hard clip on each ctrl-pt component. Keeps the FM
            # target distribution in a predictable box (e.g. [-1.5, 1.5] in
            # scaled-delta space) when the unconstrained lstsq overshoots.
            if self.bspline_ctrl_clip is not None:
                P_gt = P_gt.clamp(
                    min=-self.bspline_ctrl_clip, max=self.bspline_ctrl_clip
                )

            # `valid_kp` is now permissive: kps with at least D valid future
            # steps participate in FM loss. The prior "all valid" rule was
            # too strict on short-clip data (agibot): saw all 32 future steps
            # required, dropped most kps.
            valid_kp = m_future.sum(dim=-1).long() >= D
            P_gt = torch.where(valid_kp[:, None, None], P_gt, torch.zeros_like(P_gt))
            out[CTRL_PTS_FUTURE_KEY] = P_gt
            out[VALID_KP_KEY] = valid_kp

        if self.load_depth and v.depth is not None:
            # Render on the mmap'd frame; turbo LUT + log-depth normalization happens per
            # frame but the range is dataset-wide so absolute meter-to-color mapping is
            # stable across frames (see render_depth_rgb docstring).
            log_min, log_max = self.depth_log_range  # type: ignore[misc]
            depth_meters = np.asarray(v.depth[t], dtype=np.float32)
            # Sensor-noise emulation: additive Gaussian (in meters) on valid pixels
            # only, applied BEFORE the colormap so the meter→color mapping is
            # preserved (noisy meters produce noisy-but-correctly-colored RGB).
            # Invalid pixels (NaN / non-positive) stay invalid.
            if self.depth_noise_std > 0.0:
                valid = np.isfinite(depth_meters) & (depth_meters > 1e-3)
                noise = np.random.normal(
                    0.0, self.depth_noise_std, size=depth_meters.shape
                ).astype(np.float32)
                depth_meters = np.where(valid, depth_meters + noise, depth_meters)
            depth_rgb = render_depth_rgb(depth_meters, log_min, log_max)  # (H, W, 3) float32
            depth_t = torch.from_numpy(depth_rgb).permute(2, 0, 1)  # (3, H, W)
            if depth_t.shape[-2] != self.image_size or depth_t.shape[-1] != self.image_size:
                depth_t = F.interpolate(
                    depth_t.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            out[DEPTH_KEY] = depth_t

        return out


def trace_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pads the N (keypoint) axis to the batch max and emits `key_pad_mask`.

    Images are stacked directly (fixed shape per dataset). Language strings
    are returned as a list and tokenized downstream.
    """
    n_max = max(s[TRAJ_HISTORY_KEY].shape[0] for s in samples)
    batch_size = len(samples)
    h = samples[0][TRAJ_HISTORY_KEY].shape[1]
    f = samples[0][TRAJ_FUTURE_KEY].shape[1]

    traj_history = torch.zeros(batch_size, n_max, h, 3, dtype=torch.float32)
    traj_future = torch.zeros(batch_size, n_max, f, 3, dtype=torch.float32)
    current_kp = torch.zeros(batch_size, n_max, 3, dtype=torch.float32)
    valid_history = torch.zeros(batch_size, n_max, h, dtype=torch.bool)
    valid_future = torch.zeros(batch_size, n_max, f, dtype=torch.bool)
    key_pad_mask = torch.zeros(batch_size, n_max, dtype=torch.bool)

    for i, s in enumerate(samples):
        n = s[TRAJ_HISTORY_KEY].shape[0]
        traj_history[i, :n] = s[TRAJ_HISTORY_KEY]
        traj_future[i, :n] = s[TRAJ_FUTURE_KEY]
        current_kp[i, :n] = s[CURRENT_KP_KEY]
        valid_history[i, :n] = s[VALID_HISTORY_KEY]
        valid_future[i, :n] = s[VALID_FUTURE_KEY]
        key_pad_mask[i, :n] = True

    out = {
        IMAGE_KEY: torch.stack([s[IMAGE_KEY] for s in samples]),
        TRAJ_HISTORY_KEY: traj_history,
        VALID_HISTORY_KEY: valid_history,
        TRAJ_FUTURE_KEY: traj_future,
        VALID_FUTURE_KEY: valid_future,
        CURRENT_KP_KEY: current_kp,
        KEY_PAD_MASK_KEY: key_pad_mask,
        TASK_KEY: [s[TASK_KEY] for s in samples],
    }
    # Gated on presence so load_depth=False path is byte-identical.
    if DEPTH_KEY in samples[0]:
        out[DEPTH_KEY] = torch.stack([s[DEPTH_KEY] for s in samples])
    # Same gating for B-spline ctrl-pt fit: present only when the dataset was
    # configured with `bspline_n_ctrl`.
    if CTRL_PTS_FUTURE_KEY in samples[0]:
        d = samples[0][CTRL_PTS_FUTURE_KEY].shape[1]
        ctrl_pts = torch.zeros(batch_size, n_max, d, 3, dtype=torch.float32)
        valid_kp = torch.zeros(batch_size, n_max, dtype=torch.bool)
        for i, s in enumerate(samples):
            n = s[CTRL_PTS_FUTURE_KEY].shape[0]
            ctrl_pts[i, :n] = s[CTRL_PTS_FUTURE_KEY]
            valid_kp[i, :n] = s[VALID_KP_KEY]
        out[CTRL_PTS_FUTURE_KEY] = ctrl_pts
        out[VALID_KP_KEY] = valid_kp
    # Per-keypoint DINO cluster id. Padded slots get `-1` so the rigidity loss
    # can drop them with a single `>= 0` mask.
    if CLUSTER_IDS_KEY in samples[0]:
        cluster_ids = torch.full((batch_size, n_max), -1, dtype=torch.long)
        for i, s in enumerate(samples):
            n = s[CLUSTER_IDS_KEY].shape[0]
            cluster_ids[i, :n] = s[CLUSTER_IDS_KEY]
        out[CLUSTER_IDS_KEY] = cluster_ids
    return out
