#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import copy
import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.datasets.bspline_basis import build_bspline_basis
from lerobot.datasets.trace_dataset import (
    CLUSTER_IDS_KEY,
    CTRL_PTS_FUTURE_KEY,
    CURRENT_KP_KEY,
    DEPTH_KEY,
    KEY_PAD_MASK_KEY,
    TRAJ_FUTURE_KEY,
    TRAJ_HISTORY_KEY,
    VALID_FUTURE_KEY,
    VALID_HISTORY_KEY,
    VALID_KP_KEY,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    populate_queues,
)
from .configuration_smolvla import SmolVLAConfig
from .smolvlm_with_expert import SmolVLMWithExpertModel


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


class TimestepEmbedder(nn.Module):
    """Shared DiT-style TimestepEmbedder for flow-matching τ ∈ [0, 1].

    Maps `t: (B,)` → sinusoidal features → 2-layer MLP → `(B, hidden_size)`.
    Combined with per-block `adaLN_modulation = SiLU → Linear(hidden, 6*hidden)`
    heads (built in the policy model), this produces adaLN-Zero modulation.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ) -> None:
        super().__init__()
        if frequency_embedding_size % 2 != 0:
            raise ValueError(
                f"frequency_embedding_size must be even; got {frequency_embedding_size}"
            )
        self.frequency_embedding_size = frequency_embedding_size
        self.min_period = min_period
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = create_sinusoidal_pos_embedding(
            t,
            self.frequency_embedding_size,
            self.min_period,
            self.max_period,
            device=t.device,
        )
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


def fourier_features_2d(uv: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Log-spaced Fourier features for a 2D input (P1 §3.8).

    uv: `(..., 2)` — last dim is (u, v) in [-1, 1]². Returns `(..., 4*num_freqs)`
    (sin/cos × {u, v} × num_freqs log-spaced frequencies `{pi, 2pi, ..., 2^(K-1)*pi}`).
    """
    # breakpoint()
    freqs = (2.0 ** torch.arange(num_freqs, device=uv.device, dtype=uv.dtype)) * math.pi
    proj = uv[..., None] * freqs  # (..., 2, K)
    return torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(-2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    # Not used in trace prediction mode.
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    # Not used in trace prediction mode.
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    # Not used in trace prediction mode.
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # Not used in trace prediction mode.
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Not used in trace prediction mode.
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Not used in trace prediction mode.
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Not used in trace prediction mode.
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self):
        """Parameter groups for the optimizer.

        In trace mode with vlm_lr_multiplier != 1.0, return two param groups so the VLM
        backbone trains at a different LR than the expert + trace projections (plan §3.9).
        """
        if not self.config.trace_mode or self.config.vlm_lr_multiplier == 1.0:
            return [p for p in self.parameters() if p.requires_grad]

        vlm_prefix = "model.vlm_with_expert.vlm."
        vlm_params, other_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(vlm_prefix):
                vlm_params.append(p)
            else:
                other_params.append(p)

        return [
            {"params": other_params, "lr": self.config.optimizer_lr},
            {"params": vlm_params, "lr": self.config.optimizer_lr * self.config.vlm_lr_multiplier},
        ]

    def prepare_trace(self, batch: dict[str, Tensor]):
        """Extract trace tensors from a batch (see `lerobot.datasets.trace_dataset`).

        The dataset emits ``ctrl_pts_future`` ``(B, N, D, 3)`` and ``valid_kp``
        ``(B, N)`` for the B-spline ctrl-pt FM target (see
        ``docs/trace_bspline_ctrl_pts_plan.md``).

        ``cluster_ids`` ``(B, N)`` is emitted only when the dataset was built
        with ``load_cluster_ids=True`` (driven by ``rigidity_loss_weight > 0``);
        ``-1`` flags padded slots.
        """
        traj_history = batch[TRAJ_HISTORY_KEY]
        traj_future = batch[TRAJ_FUTURE_KEY]
        current_kp = batch[CURRENT_KP_KEY]
        key_pad_mask = batch[KEY_PAD_MASK_KEY]
        valid_future = batch[VALID_FUTURE_KEY]
        valid_history = batch[VALID_HISTORY_KEY]
        ctrl_pts_future = batch.get(CTRL_PTS_FUTURE_KEY)
        valid_kp = batch.get(VALID_KP_KEY)
        cluster_ids = batch.get(CLUSTER_IDS_KEY)
        return (
            traj_history, traj_future, current_kp, key_pad_mask,
            valid_future, valid_history, ctrl_pts_future, valid_kp,
            cluster_ids,
        )

    @torch.no_grad()
    def predict_trace(
        self,
        batch: dict[str, Tensor],
        num_steps: int | None = None,
        mask_all_history: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Denoise a future trace given (image, language, clean history). Plan §3.7.

        Returns `traces: (B, N, H, 3)` when `trace_done_head=False` (default).
        When `trace_done_head=True`, returns a dict
        ``{"traces": (B, N, H, 3), "done_prob": (B, N, G_fut|H)}``. See
        docs/trace_done_head_plan.md.

        `mask_all_history=True` mirrors the training "full-mask" branch: every
        keypoint's history-group suffix tokens are masked out before the suffix
        attends. Only valid in flat tokenization. See docs/trace_history_dropout_plan.md.
        """
        self.eval()
        images, img_masks, is_depth_flags = self.prepare_images(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        traj_history = batch[TRAJ_HISTORY_KEY]
        current_kp = batch[CURRENT_KP_KEY]
        key_pad_mask = batch[KEY_PAD_MASK_KEY]
        return self.model.sample_traces(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            traj_history=traj_history,
            current_kp=current_kp,
            key_pad_mask=key_pad_mask,
            num_steps=num_steps,
            is_depth_flags=is_depth_flags,
            mask_all_history=mask_all_history,
        )

    def _forward_trace(
        self, batch: dict[str, Tensor], reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        """Trace-mode training step. Plan §3.6."""
        images, img_masks, is_depth_flags = self.prepare_images(batch)
        # Train-only depth dropout: per-sample Bernoulli on the depth image's
        # img_mask. embed_prefix ties the depth `<img_start>/<img_end>` envelope
        # mask to this per-sample img_mask, so zeroing rows here makes the
        # dropped samples' *entire* depth block (envelope + patches) non-attending
        # in the prefix; combined with the cumsum-based position scheme, the
        # block contributes zero RoPE positions and lang/suffix see positions
        # identical to a no-depth setup. RGB branch is untouched.
        if (
            self.training
            and self.config.depth_dropout_prob > 0.0
            and any(is_depth_flags)
        ):
            p = float(self.config.depth_dropout_prob)
            for i, is_depth in enumerate(is_depth_flags):
                if not is_depth:
                    continue
                mask = img_masks[i]
                drop = torch.rand(mask.shape[0], device=mask.device) < p
                img_masks[i] = mask & ~drop
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        (
            traj_history,
            traj_future,
            current_kp,
            key_pad_mask,
            valid_future,
            valid_history,
            ctrl_pts_future,
            valid_kp,
            cluster_ids,
        ) = self.prepare_trace(batch)

        if ctrl_pts_future is None or valid_kp is None:
            raise RuntimeError(
                "batch is missing ctrl_pts_future / valid_kp. Did you set "
                "bspline_n_ctrl on the dataset?"
            )
        if self.config.rigidity_loss_weight > 0.0 and cluster_ids is None:
            raise RuntimeError(
                "rigidity_loss_weight > 0 but batch is missing cluster_ids. "
                "Build the dataset with load_cluster_ids=True."
            )

        # `forward_trace` returns a dict {"fm_per_elem": (B, N, D, 3), ...}
        # with the FM target in B-spline ctrl-pt space.
        out = self.model.forward_trace(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            traj_history=traj_history,
            traj_future=traj_future,
            current_kp=current_kp,
            key_pad_mask=key_pad_mask,
            is_depth_flags=is_depth_flags,
            valid_future=valid_future,
            valid_history=valid_history,
            ctrl_pts_future=ctrl_pts_future,
            valid_kp=valid_kp,
            cluster_ids=cluster_ids,
        )
        losses = out["fm_per_elem"]  # (B, N, D, 3) -- ctrl-pt FM
        rigidity = out.get("rigidity")
        done_logits = out.get("done_logits")  # (B, N, H) | None

        # All-or-nothing per kp: every ctrl pt of an invalid kp is excluded.
        loss_mask = (
            valid_kp.unsqueeze(-1).unsqueeze(-1) & key_pad_mask[:, :, None, None]
        )
        loss_mask = loss_mask.to(losses.dtype)
        denom = loss_mask.sum().clamp_min(1.0)
        masked = losses * loss_mask

        loss_dict = {
            "loss_per_dim": (masked.sum(dim=(0, 1, 2)) / denom).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            per_sample_denom = loss_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            per_sample_loss = masked.sum(dim=(1, 2, 3)) / per_sample_denom
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        fm_loss = masked.sum() / denom
        total = fm_loss

        # Rigidity regularization on predicted clean ctrl pts: variance of
        # intra-cluster pairwise distances across the D ctrl-pt axis.
        if rigidity is not None:
            total = total + self.config.rigidity_loss_weight * rigidity
            loss_dict["loss_fm"] = float(fm_loss.detach())
            loss_dict["loss_rigidity"] = float(rigidity.detach())

        # Done head BCE (docs/trace_done_head_plan.md). The head is H-wide (see
        # docs/trace_bspline_ctrl_pts_plan.md §3), so the target is the raw
        # `valid_future` at full H granularity. Masked by key_pad_mask only.
        if done_logits is not None:
            B, N, H = valid_future.shape
            done_target = valid_future.to(done_logits.dtype)
            done_mask = key_pad_mask[:, :, None].expand(-1, -1, H).to(done_logits.dtype)
            done_per_elem = F.binary_cross_entropy_with_logits(
                done_logits, done_target, reduction="none",
            )
            denom_done = done_mask.sum().clamp_min(1.0)
            done_loss = (done_per_elem * done_mask).sum() / denom_done
            total = total + self.config.done_loss_weight * done_loss
            loss_dict["loss_fm"] = float(fm_loss.detach())
            loss_dict["loss_done"] = float(done_loss.detach())

        loss_dict["loss"] = float(total.detach())
        return total, loss_dict

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # Not used in trace prediction mode.
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks, _ = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Not used in trace prediction mode.
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # Not used in trace prediction mode.
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # Not used in trace prediction mode.
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        # Not used in trace prediction mode.
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        # Not used in trace prediction mode.
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.trace_mode:
            return self._forward_trace(batch, reduction)

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks, _ = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.

        Returns `(images, img_masks, is_depth_flags)`. The third list mirrors `images`
        with `True` where the key matches the depth camera (see DEPTH_KEY). Downstream,
        `embed_prefix` routes depth entries through `embed_image(apply_lora=True)`.
        """
        images = []
        img_masks = []
        is_depth_flags = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)
            is_depth_flags.append(key == DEPTH_KEY)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
            is_depth_flags.append(False)
        return images, img_masks, is_depth_flags

    def _pi_aloha_decode_state(self, state):
        # Not used in trace prediction mode.
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Not used in trace prediction mode.
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Not used in trace prediction mode.
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        # Not used in trace prediction mode.
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        # Not used in trace prediction mode.
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        # Not used in trace prediction mode.
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        # Not used in trace prediction mode.
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
            use_depth=self.config.use_depth,
            depth_lora_rank=self.config.depth_lora_rank,
            depth_lora_alpha=self.config.depth_lora_alpha,
            depth_lora_dropout=self.config.depth_lora_dropout,
            depth_lora_targets=self.config.depth_lora_targets,
            depth_clone_stem=self.config.depth_clone_stem,
        )
        expert_hidden = self.vlm_with_expert.expert_hidden_size

        if self.config.trace_mode:
            # Trace branch: no state path, no action projections. See
            # docs/trace_prediction_p1_plan.md §3 (per-keypoint suffix tokens) and
            # docs/trace_prediction_p2_plan.md §3 (flat tokenization).
            h_plus_H = self.config.history_len + self.config.future_len
            # --- Shared sub-modules ---
            self.trace_time_embedding = nn.Embedding(h_plus_H, expert_hidden)
            self.trace_seg_embedding = nn.Embedding(2, expert_hidden)
            self.trace_uv_num_freqs = self.config.trace_uv_num_freqs
            self.trace_uv_pos_mlp = nn.Sequential(
                nn.Linear(4 * self.trace_uv_num_freqs, expert_hidden),
                nn.SiLU(),
                nn.Linear(expert_hidden, expert_hidden),
            )
            nn.init.zeros_(self.trace_uv_pos_mlp[-1].weight)
            nn.init.zeros_(self.trace_uv_pos_mlp[-1].bias)

            # --- Projection heads (B-spline ctrl-pt FM, flat tokenization).
            # docs/trace_bspline_ctrl_pts_plan.md §3: split projections because
            # history (time-grouped 3*g) and future (ctrl-pt-grouped 3*g_cp)
            # carry different input dims. ---
            D_cp = self.config.trace_bspline_n_ctrl
            g_cp = self.config.trace_bspline_ctrl_per_token
            g_t = self.config.trace_group_horizon
            self.trace_in_proj_history = nn.Linear(3 * g_t, expert_hidden)
            self.trace_in_proj_future = nn.Linear(3 * g_cp, expert_hidden)
            self.trace_out_proj_future = nn.Linear(expert_hidden, 3 * g_cp)

            # --- Done head (docs/trace_done_head_plan.md). H-wide head reading
            # off a per-kp aggregator (mean over the kp's future ctrl-pt tokens);
            # see docs/trace_bspline_ctrl_pts_plan.md §3. Zero-init weights +
            # positive bias so an untrained head says "valid"
            # (sigmoid(done_bias_init) ≈ 0.95 at the default 3.0). ---
            if self.config.trace_done_head:
                self.trace_done_proj_bsp = nn.Linear(
                    expert_hidden, self.config.future_len
                )
                _done_head = self.trace_done_proj_bsp
                nn.init.zeros_(_done_head.weight)
                nn.init.constant_(_done_head.bias, self.config.done_bias_init)

            # B-spline render basis. Stored as a non-persistent buffer so it
            # follows .to(device) but isn't checkpointed (it's a deterministic
            # function of (n_ctrl, future_len)).
            # (H, D): no anchor row at inference — we render at the H future
            # timesteps directly. The training path doesn't use this; the LS
            # fit is done in the dataset on (H+1, D).
            self.register_buffer(
                "_bspline_render_basis",
                build_bspline_basis(
                    D_cp, self.config.future_len, include_anchor=False,
                    dtype=torch.float32, device="cpu",
                ),
                persistent=False,
            )

            # --- adaLN-Zero conditioning for τ (docs/trace_positional_encoding_plan.md §B) ---
            self.trace_timestep_embedder = TimestepEmbedder(
                hidden_size=expert_hidden,
                frequency_embedding_size=self.config.trace_adaln_freq_embed_size,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
            )
            num_expert_layers_actual = self.vlm_with_expert.num_expert_layers
            self.trace_adaln_modulation = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(expert_hidden, 6 * expert_hidden),
                    )
                    for _ in range(num_expert_layers_actual)
                ]
            )
            # Zero-init the final Linear of each per-block head so
            # shift=scale=gate=0 at step 0 → each expert block is exactly
            # the identity. The shared TimestepEmbedder is left at default
            # init so it still produces meaningful τ features once the
            # gates open during training.
            for head in self.trace_adaln_modulation:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        else:
            self.state_proj = nn.Linear(
                self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
            )
            self.action_in_proj = nn.Linear(self.config.max_action_dim, expert_hidden)
            self.action_out_proj = nn.Linear(expert_hidden, self.config.max_action_dim)

        # Flow-matching time fusion MLP — shared between trace and action paths.
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        # DINO per-keypoint visual feature (docs/trace_dino_feature_plan.md).
        # Frozen backbone; concat + 2-layer MLP fuses raw DINO features into each
        # suffix token. Attributes are set to None when disabled so
        # `if self.dino is None` guards work cleanly downstream.
        self.dino = None
        self.trace_dino_fuse_in = None
        self.trace_dino_fuse_out = None
        if self.config.trace_mode and self.config.use_dino:
            from transformers import AutoModel

            self.dino = AutoModel.from_pretrained(
                self.config.dino_model_name,
                torch_dtype=torch.bfloat16,
            )
            for p in self.dino.parameters():
                p.requires_grad_(False)
            self.dino.eval()
            d_dino = self.dino.config.hidden_size
            self.trace_dino_fuse_in = nn.Linear(expert_hidden + d_dino, expert_hidden)
            self.trace_dino_fuse_out = nn.Linear(expert_hidden, expert_hidden)
            # ImageNet stats as non-persistent buffers so they travel with the
            # module across devices but aren't written to checkpoints.
            self.register_buffer(
                "_dino_mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_dino_std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )

        self.set_requires_grad()
        if self.dino is not None:
            assert not any(p.requires_grad for p in self.dino.parameters()), (
                "DINO must be frozen but some params have requires_grad=True"
            )
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        # Not used in trace prediction mode.
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        if self.config.trace_mode:
            # No state_proj in trace mode — nothing to toggle here.
            return
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def train(self, mode: bool = True):
        # DINO is frozen and runs under `torch.no_grad()`; keep it in eval mode so
        # any stateful submodules (e.g. dropout) stay deterministic after the
        # outer `.train()` call during training setup.
        super().train(mode)
        if self.dino is not None:
            self.dino.eval()
        return self

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def _append_image_block(
        self,
        embs: list,
        pad_masks: list,
        att_masks: list,
        img: torch.Tensor,
        img_mask: torch.Tensor,
        is_depth: bool,
    ) -> None:
        """Append one image block (start envelope + patches + end envelope) to the
        running prefix lists.

        `img_mask` is the per-sample `(B,)` bool used for the patch rows; the
        same mask is broadcast onto the `<img_start>/<img_end>` envelope rows so
        that a fully-dropped sample (all-zero `img_mask`) contributes zero RoPE
        positions for its entire image block under the cumsum-based position
        scheme. For RGB blocks `img_mask` is all-True, so this is a no-op; for
        depth blocks dropped via `depth_dropout_prob`, the envelope collapses
        out alongside the patches.
        """
        bsize = img.shape[0]
        if self.add_image_special_tokens:
            image_start_token = (
                self.vlm_with_expert.embed_language_tokens(
                    self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                )
                .unsqueeze(0)
                .expand(bsize, -1, -1)
            )
            image_start_mask = img_mask[:, None].expand(bsize, image_start_token.shape[1]).contiguous()
            att_masks += [0] * (image_start_mask.shape[-1])
            embs.append(image_start_token)
            pad_masks.append(image_start_mask)

        img_emb = self.vlm_with_expert.embed_image(img, apply_lora=is_depth)

        # Normalize image embeddings
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

        _bsize, num_img_embs = img_emb.shape[:2]
        patch_mask = img_mask[:, None].expand(_bsize, num_img_embs).contiguous()

        embs.append(img_emb)
        pad_masks.append(patch_mask)
        att_masks += [0] * num_img_embs

        if self.add_image_special_tokens:
            image_end_token = (
                self.vlm_with_expert.embed_language_tokens(
                    self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                )
                .unsqueeze(0)
                .expand(bsize, -1, -1)
            )
            image_end_mask = img_mask[:, None].expand(bsize, image_end_token.shape[1]).contiguous()
            embs.append(image_end_token)
            pad_masks.append(image_end_mask)
            att_masks += [0] * image_end_mask.shape[-1]

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        is_depth_flags: list[bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.

        `is_depth_flags` (optional) must align with `images`; True entries are routed
        through the depth LoRA adapter + cloned stem via `embed_image(apply_lora=True)`.
        """
        embs = []
        pad_masks = []
        att_masks = []
        if is_depth_flags is None:
            is_depth_flags = [False] * len(images)

        for img, img_mask, is_depth in zip(images, img_masks, is_depth_flags, strict=False):
            self._append_image_block(embs, pad_masks, att_masks, img, img_mask, is_depth)

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        if state is not None:
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            device = state_emb.device

            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1] * (states_seq_len)
        else:
            bsize = lang_emb.shape[0]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def _suffix_groups_per_kp(self) -> int:
        """Number of suffix groups per keypoint in flat tokenization: ``h/g + D/g_cp``."""
        cfg = self.config
        return (
            cfg.history_len // cfg.trace_group_horizon
            + cfg.trace_bspline_n_ctrl // cfg.trace_bspline_ctrl_per_token
        )

    def _apply_causal_suffix_mask(self, suffix_sub_mask, N, G):
        """AND a causal-over-groups constraint into a `(B, N*G, N*G)` suffix sub-mask.

        Tokens are laid out kp-major then group-major: token `(i, g)` is at
        position `i*G + g` (see `_embed_suffix_trace_flat` line ~1329). For each
        query `(i_q, g_q)` we permit keys `(i_k, g_k)` iff `g_k <= g_q` — any
        keypoint, any earlier-or-equal group. Within a single time-group all
        keypoints remain bidirectional.

        The future-side groups are ctrl-pt-indexed (no time meaning), so causal
        ordering is restricted to the history sub-block: future ctrl-pt tokens
        are bidirectional w.r.t. each other
        AND w.r.t. all history tokens; history tokens remain causally ordered
        among themselves. (See docs/trace_bspline_ctrl_pts_plan.md §4.)

        The caller is expected to have already AND'd in pad-derived masks;
        this helper only adds the causal constraint on top. Both training
        (`forward_trace`) and inference (`denoise_step_trace`) use the same
        helper. See docs/improve_training_pipe_plan.md §1.
        """
        suffix_len = N * G
        device = suffix_sub_mask.device
        g_t = self.config.trace_group_horizon
        G_hist = self.config.history_len // g_t
        # Per-token group index. For history positions g_idx is the
        # history-group index (0..G_hist-1); for future positions it's set
        # to G_hist (a single sentinel "after all history") so all future
        # tokens see all history AND all other future tokens.
        local_idx = torch.arange(suffix_len, device=device) % G  # (N*G,)
        g_idx = torch.where(
            local_idx < G_hist, local_idx, torch.full_like(local_idx, G_hist),
        )
        causal = g_idx[None, :] <= g_idx[:, None]  # (N*G, N*G)
        return suffix_sub_mask & causal[None]

    def _suffix_out_to_velocity(self, suffix_out, num_kp):
        """Reshape the expert's suffix output into the FM velocity prediction.

        The FM target is in B-spline ctrl-pt space, so the output shape is
        `(B, N, D, 3)` (see docs/trace_bspline_ctrl_pts_plan.md §3).
        """
        bsize = suffix_out.shape[0]
        g_t = self.config.trace_group_horizon
        G_hist = self.config.history_len // g_t
        D_cp = self.config.trace_bspline_n_ctrl
        g_cp = self.config.trace_bspline_ctrl_per_token
        G_cp = D_cp // g_cp
        G = G_hist + G_cp
        suffix_out = suffix_out[:, -num_kp * G:].to(dtype=torch.float32)
        suffix_out = suffix_out.reshape(bsize, num_kp, G, suffix_out.shape[-1])
        future_out = suffix_out[:, :, -G_cp:]  # (B, N, G_cp, D_expert)
        v_flat = self.trace_out_proj_future(future_out)  # (B, N, G_cp, 3*g_cp)
        return v_flat.reshape(bsize, num_kp, G_cp * g_cp, 3).contiguous()

    def _suffix_out_to_done_logits(self, suffix_out, num_kp):
        """Done-head logits `(B, N, H)` — full-H head off a per-kp aggregator
        (mean over the kp's future ctrl-pt tokens). Returns None when
        `trace_done_head=False`. See docs/trace_bspline_ctrl_pts_plan.md §3 and
        docs/trace_done_head_plan.md.
        """
        if not self.config.trace_done_head:
            return None
        bsize = suffix_out.shape[0]
        g_t = self.config.trace_group_horizon
        G_hist = self.config.history_len // g_t
        D_cp = self.config.trace_bspline_n_ctrl
        g_cp = self.config.trace_bspline_ctrl_per_token
        G_cp = D_cp // g_cp
        G = G_hist + G_cp
        suffix_out = suffix_out[:, -num_kp * G:].to(dtype=torch.float32)
        suffix_out = suffix_out.reshape(bsize, num_kp, G, suffix_out.shape[-1])
        future_out = suffix_out[:, :, -G_cp:]  # (B, N, G_cp, D_expert)
        # Per-kp aggregator: mean over the kp's future ctrl-pt tokens.
        kp_token = future_out.mean(dim=2)  # (B, N, D_expert)
        return self.trace_done_proj_bsp(kp_token)  # (B, N, H)

    def _compute_dino_per_kp(
        self,
        images: list[torch.Tensor],
        is_depth_flags: list[bool] | None,
        current_uv: torch.Tensor,
        key_pad_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Frozen DINO forward + bilinear sample at each keypoint's current-frame (u, v).

        Returns `(B, N, D_dino)` or `None` when `use_dino=False`. The fusion MLP is
        applied downstream in `_embed_suffix_trace_flat`.

        Args:
            images: list of `(B, 3, H_img, W_img)` RGB tensors in `[-1, 1]` (SigLIP scale).
            is_depth_flags: optional list aligned with `images`; True entries are skipped.
            current_uv: `(B, N, 2)` in `[-1, 1]` (grid_sample's native frame),
                read from `current_kp[:, :, :2]` post-split.
            key_pad_mask: `(B, N)` bool — padded kps are zeroed out on return.
        """
        if self.dino is None:
            return None

        rgb = None
        if is_depth_flags is None:
            rgb = images[0]
        else:
            for img, is_depth in zip(images, is_depth_flags, strict=True):
                if not is_depth:
                    rgb = img
                    break
        if rgb is None:
            raise RuntimeError("use_dino=True but no non-depth image in batch")

        # [-1, 1] (SigLIP) -> [0, 1] -> ImageNet normalize.
        x = (rgb + 1.0) * 0.5
        x = (x - self._dino_mean) / self._dino_std
        S = self.config.dino_input_size
        x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = self.dino(pixel_values=x.to(dtype=self.dino.dtype))
        B = x.shape[0]
        D = self.dino.config.hidden_size
        P = S // self.dino.config.patch_size
        patch = out.last_hidden_state[:, self.config.dino_num_prefix_tokens:]  # drop CLS / registers
        fmap = patch.transpose(1, 2).reshape(B, D, P, P)
        grid = current_uv.view(B, 1, -1, 2).to(dtype=fmap.dtype)
        sampled = F.grid_sample(fmap, grid, mode="bilinear", align_corners=False)  # (B, D, 1, N)
        feat = sampled.squeeze(-2).transpose(1, 2)  # (B, N, D_dino)
        return feat * key_pad_mask.to(feat.dtype)[..., None]

    def embed_suffix_trace(
        self,
        traj_input,
        current_uv,
        timestep,
        key_pad_mask,
        dino_feat=None,
        mask_all_history: bool = False,
    ):
        """Build the flat-tokenization suffix (see docs/trace_prediction_p2_plan.md §4).

        Args:
            traj_input: `(B, N, h+H, 3)` — clean past history concat noisy future on
                time. Post-split (docs/trace_current_kp_split_plan.md), the time axis
                contains `[t-h..t-1, t+1..t+F]` — the current frame `t` is NOT in this
                concatenation; it enters via `current_uv` (and DINO).
            current_uv: `(B, N, 2)` in `[-1, 1]` — read from `current_kp[:, :, :2]`.
                Drives the per-kp 2D Fourier positional embedding.
            timestep: `(B,)` flow-matching τ in [0, 1].
            key_pad_mask: `(B, N)` bool — True for real keypoints.
            dino_feat: optional `(B, N, D_dino)` — raw DINO patch features sampled at
                each keypoint's current-frame `(u, v)`. When provided, fused into each
                suffix token via concat + 2-layer MLP.
            mask_all_history: when True, drop every keypoint's history-group suffix
                tokens from the attention (flat tokenization only). See
                docs/trace_history_dropout_plan.md.

        Returns:
            fused: `(B, N*G, D)`.
            pad_masks, att_masks: matching suffix length.
        """
        return self._embed_suffix_trace_flat(
            traj_input, current_uv, timestep, key_pad_mask,
            dino_feat=dino_feat, mask_all_history=mask_all_history,
        )

    def _build_expert_adaln_modulation(self, timestep: torch.Tensor) -> torch.Tensor | None:
        """Compute per-expert-layer adaLN modulation from τ.

        Returns `(B, num_expert_layers, 6, D_expert)` or None when disabled.
        """
        if not self.config.trace_mode:
            return None
        t_cond = self.trace_timestep_embedder(timestep)  # (B, D_expert)
        per_layer = torch.stack(
            [head(t_cond) for head in self.trace_adaln_modulation], dim=1
        )  # (B, L, 6*D_expert)
        return per_layer.reshape(per_layer.shape[0], per_layer.shape[1], 6, -1)

    def _embed_suffix_trace_flat(
        self,
        traj_input,
        current_uv,
        timestep,
        key_pad_mask,
        dino_feat=None,
        mask_all_history: bool = False,
    ):
        """Flat tokenization (P2 §4.1). Suffix is `(B, N*G, D)` with one token per
        `(keypoint, group of g adjacent timesteps)`.

        `dino_feat` (optional `(B, N, D_dino)`) is broadcast over the G axis and
        concat-fused into each `(kp, group)` token via a 2-layer MLP.

        Post current_kp split, the uv positional embedding reads `current_uv`
        (== `current_kp[:, :, :2]`) rather than indexing `traj_input` at h-1.

        History dropout (docs/trace_history_dropout_plan.md): when
        `mask_all_history=True`, every keypoint's history-group `pad_masks` slots
        are flipped to False. When `self.training` and
        `trace_history_{full_mask,kp_dropout}_prob > 0`, those slots are flipped
        stochastically per-batch element. Future-group tokens, `current_uv`, and
        `dino_feat` are never touched.
        """
        bsize, num_kp, time_len, _ = traj_input.shape
        h = self.config.history_len
        g = self.config.trace_group_horizon
        device = traj_input.device

        # docs/trace_bspline_ctrl_pts_plan.md §3: history is time-grouped at g,
        # future is ctrl-pt-grouped at g_cp. They have different input dims so
        # they go through separate projections.
        D_cp = self.config.trace_bspline_n_ctrl
        g_cp = self.config.trace_bspline_ctrl_per_token
        assert time_len == h + D_cp, (
            f"bspline-mode traj_input time axis ({time_len}) must equal h+D ({h + D_cp})"
        )
        assert h % g == 0, f"history_len={h} must be divisible by g={g}"
        assert D_cp % g_cp == 0, f"D={D_cp} must be divisible by g_cp={g_cp}"
        h_groups = h // g
        G_cp = D_cp // g_cp
        G = h_groups + G_cp

        x_hist = traj_input[:, :, :h].reshape(bsize, num_kp, h_groups, g * 3)
        x_hist = self.trace_in_proj_history(x_hist)             # (B, N, h_groups, D_expert)
        x_fut = traj_input[:, :, h:].reshape(bsize, num_kp, G_cp, g_cp * 3)
        x_fut = self.trace_in_proj_future(x_fut)                # (B, N, G_cp, D_expert)
        x = torch.cat([x_hist, x_fut], dim=2)                   # (B, N, G, D_expert)

        # --- Additive positional embeddings ---
        # docs/improve_training_pipe_plan.md §1.5: per-group indexing (one row
        # per token), replacing the legacy "indexed at group starts" stride-by-g
        # lookup. Spreads the learning signal across consecutive rows of the
        # embedding table instead of every g-th row. Hard change; flat-mode
        # checkpoints have their `trace_time_embedding` invalidated.
        group_ids = torch.arange(G, device=device)  # [0, 1, ..., G-1]
        x = x + self.trace_time_embedding(group_ids)  # (G, D) broadcast

        seg_ids = torch.cat(
            [
                torch.zeros(h_groups, dtype=torch.long, device=device),
                torch.ones(G - h_groups, dtype=torch.long, device=device),
            ]
        )
        x = x + self.trace_seg_embedding(seg_ids)  # (G, D) broadcast

        uv_feat = fourier_features_2d(current_uv, self.trace_uv_num_freqs)
        pos_emb = self.trace_uv_pos_mlp(uv_feat)  # (B, N, D)
        pos_emb = pos_emb * key_pad_mask.to(pos_emb.dtype)[..., None]
        x = x + pos_emb[:, :, None, :]  # broadcast over G

        # --- DINO per-keypoint concat fusion (docs/trace_dino_feature_plan.md §3.2d) ---
        if dino_feat is not None:
            dino_b = dino_feat[:, :, None, :].expand(-1, -1, G, -1).to(x.dtype)  # (B, N, G, D_dino)
            h_cat = torch.cat([x, dino_b], dim=-1)
            x = self.trace_dino_fuse_out(F.silu(self.trace_dino_fuse_in(h_cat)))

        # --- Flatten (B, N, G, D) -> (B, N*G, D), kp-major then group. ---
        e = x.reshape(bsize, num_kp * G, -1)
        suffix_len = num_kp * G

        fused = e  # τ injected per-expert-block via adaLN, not fused here.

        # --- Masks: N*G tokens, single bidirectional block, pad via kp mask ---
        pad_masks = key_pad_mask[:, :, None].expand(bsize, num_kp, G).reshape(bsize, suffix_len)
        att_masks = torch.zeros(suffix_len, dtype=fused.dtype, device=device)
        att_masks[0] = 1
        att_masks = att_masks[None, :].expand(bsize, suffix_len)

        # --- History dropout (docs/trace_history_dropout_plan.md §3) ---
        # Flip pad_masks=False for the (kp, history-group) slots we want to
        # drop. The future groups (G - h_groups per kp) are never touched.
        # current_uv / DINO are per-kp side branches, not in pad_masks; they
        # remain visible regardless. The leading att_masks=1 block-start signal
        # stays put: cumsum runs before pad_masks is AND-ed in make_att_2d_masks,
        # so the bidirectional block over surviving suffix tokens still forms.
        history_drop_NG: torch.Tensor | None = None
        if mask_all_history:
            history_drop_NG = torch.zeros(bsize, num_kp, G, dtype=torch.bool, device=device)
            history_drop_NG[:, :, :h_groups] = True
        elif self.training and (
            self.config.trace_history_full_mask_prob > 0.0
            or self.config.trace_history_kp_dropout_prob > 0.0
        ):
            p_full = float(self.config.trace_history_full_mask_prob)
            p_kp = float(self.config.trace_history_kp_dropout_prob)
            full_mask = torch.rand(bsize, device=device) < p_full  # (B,)
            rand_kp = torch.rand(bsize, num_kp, device=device)
            kp_drop = full_mask[:, None] | (rand_kp < p_kp)  # (B, N)
            history_drop_NG = torch.zeros(bsize, num_kp, G, dtype=torch.bool, device=device)
            history_drop_NG[:, :, :h_groups] = kp_drop[:, :, None]

        if history_drop_NG is not None:
            drop_flat = history_drop_NG.reshape(bsize, suffix_len)
            pad_masks = pad_masks & ~drop_flat

        return fused, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        # Not used in trace prediction mode.
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        # Not used in trace prediction mode (trace training goes through `forward_trace`).
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        # Not used in trace prediction mode (trace inference goes through `sample_traces`).
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        # Not used in trace prediction mode (trace inference uses `denoise_step_trace`).
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def _compute_rigidity_loss(
        self,
        x_t_future: Tensor,
        v_t: Tensor,
        t_scalar: Tensor,
        cluster_ids: Tensor,
        valid_kp: Tensor,
        key_pad_mask: Tensor,
    ) -> Tensor:
        """Variance-of-pairwise-distance rigidity regularizer.

        For keypoints sharing a DINO cluster id within a sample, the pairwise
        Euclidean distance between their predicted clean control points should
        be invariant across the D ctrl-pt axis (a rigid body's internal
        distances are preserved under translation+rotation). The loss is the
        variance of these pairwise distances over D, averaged over (cluster,
        sample). Only intra-(b, cluster) pairs with both kps valid contribute.
        Sign-corrected to recover the predicted clean ctrl pts via
        ``clean_hat = x_t - t * v_t``, matching the rest of this codebase's
        ``u_t = noise - clean`` convention.

        Operates in scaled-delta ctrl-pt space (the same space as the FM
        target), so all 3 axes are roughly normalized by ``delta_scale`` and
        contribute comparably to the pairwise distance.
        """
        # Predicted clean ctrl pts. Shape (B, N, D, 3) in bspline mode.
        P = x_t_future - t_scalar * v_t
        valid = valid_kp & key_pad_mask  # (B, N)

        Bsz = P.shape[0]
        losses: list[Tensor] = []
        for b in range(Bsz):
            m = valid[b]
            if int(m.sum().item()) < 2:
                continue
            cids_b = cluster_ids[b][m]              # (M,)
            P_b = P[b][m]                           # (M, D, 3)
            # Drop sentinel padded clusters defensively (shouldn't survive `m`,
            # but be explicit).
            keep = cids_b >= 0
            if not bool(keep.any()):
                continue
            cids_b = cids_b[keep]
            P_b = P_b[keep]
            uniq = torch.unique(cids_b)
            for c in uniq:
                sel = cids_b == c
                K = int(sel.sum().item())
                if K < 2:
                    continue
                P_c = P_b[sel]                       # (K, D, 3)
                iu, ju = torch.triu_indices(
                    K, K, offset=1, device=P_c.device,
                )
                diff = P_c[iu] - P_c[ju]             # (P, D, 3)
                dist = diff.norm(dim=-1)             # (P, D)
                # Population variance over D for each pair, mean over pairs.
                var_d = dist.var(dim=-1, unbiased=False)  # (P,)
                losses.append(var_d.mean())

        if not losses:
            return P.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_fm_loss(self, u_t: Tensor, v_t: Tensor) -> Tensor:
        """Per-element FM loss `(B, N, H, 3)` (MSE). Caller masks and reduces."""
        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_trace(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        traj_history,
        traj_future,
        current_kp,
        key_pad_mask,
        noise=None,
        time=None,
        is_depth_flags: list[bool] | None = None,
        valid_future: Tensor | None = None,
        valid_history: Tensor | None = None,
        ctrl_pts_future: Tensor | None = None,
        valid_kp: Tensor | None = None,
        cluster_ids: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Trace-mode training forward pass (plan §3.6).

        Returns a dict
        ``{"fm_per_elem": (B, N, D, 3), "smooth": scalar | None, ...}`` with the
        FM target in B-spline ctrl-pt space. The caller masks `fm_per_elem` and
        reduces.
        """
        # FM target lives in B-spline ctrl-pt space.
        assert ctrl_pts_future is not None, (
            "B-spline ctrl-pt FM requires ctrl_pts_future on forward_trace."
        )
        fm_target = ctrl_pts_future

        if noise is None:
            noise = self.sample_noise(fm_target.shape, fm_target.device)
        if time is None:
            time = self.sample_time(fm_target.shape[0], fm_target.device)

        t = time[:, None, None, None]
        x_t_future = t * noise + (1 - t) * fm_target
        u_t = noise - fm_target

        # In bspline mode the suffix carries (h history timesteps + D ctrl pts).
        traj_input = torch.cat([traj_history, x_t_future], dim=2)  # (B, N, h+F, 3)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=None,
            is_depth_flags=is_depth_flags,
        )
        current_uv = current_kp[:, :, :2]
        dino_feat = self._compute_dino_per_kp(images, is_depth_flags, current_uv, key_pad_mask)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_trace(
            traj_input, current_uv, time, key_pad_mask, dino_feat=dino_feat,
        )
        expert_adaln_modulation = self._build_expert_adaln_modulation(time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # Causal-over-groups suffix mask (golden); see _apply_causal_suffix_mask.
        G = self._suffix_groups_per_kp()
        num_kp = traj_history.shape[1]
        s = prefix_pad_masks.shape[1]
        suffix_len = num_kp * G
        sub = att_2d_masks[:, s:s + suffix_len, s:s + suffix_len]
        att_2d_masks = att_2d_masks.clone()
        att_2d_masks[:, s:s + suffix_len, s:s + suffix_len] = self._apply_causal_suffix_mask(
            sub, num_kp, G,
        )
        # P1 §3.8: pin all suffix position_ids to L_real_prefix so RoPE adds no
        # positional signal between suffix tokens (kp ordering on N is arbitrary).
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_real_len = prefix_pad_masks.sum(dim=1, keepdim=True)
        suffix_position_ids = prefix_real_len.expand(-1, suffix_pad_masks.shape[1])
        position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            expert_adaln_modulation=expert_adaln_modulation,
        )

        bsize, num_kp, h, _ = traj_history.shape
        v_t = self._suffix_out_to_velocity(suffix_out, num_kp)
        done_logits = self._suffix_out_to_done_logits(suffix_out, num_kp)
        fm_per_elem = self._compute_fm_loss(u_t, v_t)

        rigidity: Tensor | None = None
        if self.config.rigidity_loss_weight > 0.0:
            assert cluster_ids is not None and valid_kp is not None, (
                "rigidity_loss_weight > 0 requires cluster_ids + valid_kp"
            )
            rigidity = self._compute_rigidity_loss(
                x_t_future=x_t_future,
                v_t=v_t,
                t_scalar=t,
                cluster_ids=cluster_ids,
                valid_kp=valid_kp,
                key_pad_mask=key_pad_mask,
            )

        return {
            "fm_per_elem": fm_per_elem,
            "rigidity": rigidity,
            "done_logits": done_logits,
        }

    @torch.no_grad()
    def sample_traces(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        traj_history,
        current_kp,
        key_pad_mask,
        noise=None,
        num_steps=None,
        is_depth_flags: list[bool] | None = None,
        mask_all_history: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Inference-time denoising of the future trace (plan §3.7).

        When `trace_done_head=False` (default), returns `traces: (B, N, H, 3)` —
        same shape and behavior as before this PR. When `trace_done_head=True`,
        returns a dict ``{"traces": (B, N, H, 3), "done_prob": (B, N, G_fut)}``
        (or `(B, N, H)` in bspline mode) where `done_prob` is `sigmoid(done_logits)`
        from the final denoise step. See docs/trace_done_head_plan.md.

        The denoise loop runs in `(B, N, D, 3)` ctrl-pt space and the final ctrl
        pts are rendered to `(B, N, H, 3)` via the cached cubic B-spline basis
        exactly once at the end. The returned
        shape is unchanged so callers stay agnostic. See
        docs/trace_bspline_ctrl_pts_plan.md §3.

        `mask_all_history=True` mirrors the training "full-mask" branch — every
        keypoint's history-group suffix tokens are dropped before attention.
        See docs/trace_history_dropout_plan.md.
        """
        if num_steps is None:
            num_steps = self.config.num_steps

        bsize, num_kp, h, _ = traj_history.shape
        device = traj_history.device
        # Denoise-loop axis size: D ctrl pts.
        F_dim = self.config.trace_bspline_n_ctrl

        if noise is None:
            noise = self.sample_noise((bsize, num_kp, F_dim, self.config.trace_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=None,
            is_depth_flags=is_depth_flags,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        # DINO features are a function of the image only, so compute once and
        # re-use across every denoising iteration (the concat-MLP still runs per
        # step since its other input depends on the current suffix).
        current_uv = current_kp[:, :, :2]
        dino_feat = self._compute_dino_per_kp(images, is_depth_flags, current_uv, key_pad_mask)

        dt = -1.0 / num_steps
        # P1 §3.7: carry only the future slice across denoising iterations; the
        # integrator never touches the history, which is re-concatenated clean each step.
        x_future = noise  # (B, N, H, 3)
        last_done_logits: Tensor | None = None
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t, done_logits = self.denoise_step_trace(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                traj_history=traj_history,
                current_uv=current_uv,
                x_t_future=x_future,
                key_pad_mask=key_pad_mask,
                timestep=time_tensor,
                dino_feat=dino_feat,
                mask_all_history=mask_all_history,
            )
            x_future = x_future + dt * v_t  # (B, N, F_dim, 3)
            if step == num_steps - 1:
                last_done_logits = done_logits

        # x_future is the ctrl-pt prediction P_pred (B, N, D, 3); render to
        # (B, N, H, 3) via the cached basis.
        basis = self._bspline_render_basis.to(
            device=x_future.device, dtype=x_future.dtype
        )
        traces = torch.einsum("hd,bndc->bnhc", basis, x_future).contiguous()

        if self.config.trace_done_head:
            return {
                "traces": traces,
                "done_prob": torch.sigmoid(last_done_logits),
            }
        return traces

    def denoise_step_trace(
        self,
        prefix_pad_masks,
        past_key_values,
        traj_history,
        current_uv,
        x_t_future,
        key_pad_mask,
        timestep,
        dino_feat=None,
        mask_all_history: bool = False,
    ):
        """One denoising step on the future trace (P1 §3.7). Returns
        `(v̂, done_logits)` where `v̂` has shape `(B, N, H, 3)` and
        `done_logits` is `(B, N, G_fut)` (or `(B, N, H)` in bspline mode) when
        `trace_done_head=True`, else `None`. See docs/trace_done_head_plan.md.

        `traj_history` is the original clean (past-only) history tensor; it is
        re-concatenated fresh every step so the history slots never carry forward
        integrator noise. `current_uv` is the per-kp (u, v) for the uv positional
        embedding (== `current_kp[:, :, :2]`). `dino_feat`, when provided, is the
        cached `(B, N, D_dino)` tensor produced once per `sample_traces` call and
        re-used across denoise steps. `mask_all_history` propagates to the suffix
        embedder; see docs/trace_history_dropout_plan.md.
        """
        traj_input = torch.cat([traj_history, x_t_future], dim=2)  # (B, N, h+H, 3)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_trace(
            traj_input, current_uv, timestep, key_pad_mask,
            dino_feat=dino_feat, mask_all_history=mask_all_history,
        )
        expert_adaln_modulation = self._build_expert_adaln_modulation(timestep)

        bsize, num_kp, h, _ = traj_history.shape
        suffix_len = suffix_pad_masks.shape[1]  # == N*G (flat tokenization)
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        # Causal-over-groups suffix mask (golden); see _apply_causal_suffix_mask.
        G = self._suffix_groups_per_kp()
        suffix_att_2d_masks = self._apply_causal_suffix_mask(
            suffix_att_2d_masks, num_kp, G,
        )
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # P1 §3.8: pin all suffix positions to L_real_prefix.
        prefix_offsets = prefix_pad_masks.sum(dim=-1, keepdim=True)
        position_ids = prefix_offsets.expand(-1, suffix_len)

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            expert_adaln_modulation=expert_adaln_modulation,
        )
        suffix_out = outputs_embeds[1]
        v_t = self._suffix_out_to_velocity(suffix_out, num_kp)
        done_logits = self._suffix_out_to_done_logits(suffix_out, num_kp)
        return v_t, done_logits
