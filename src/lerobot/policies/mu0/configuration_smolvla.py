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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Trace prediction mode (see docs/trace_prediction_smolvla_plan.md). When True:
    #  - Input is (image, language, history trace of N keypoints); target is future trace.
    #  - chunk_size / max_state_dim / max_action_dim are ignored on the suffix path.
    #  - State path is removed (no state_proj in the prefix).
    trace_mode: bool = False
    history_len: int = 16
    future_len: int = 32
    trace_dim: int = 3
    n_min: int = 1
    n_max: int = 32

    # Trace-mode fine-tune knobs (plan §3.5/§3.9).
    # Defaults preserve freeze_vision_encoder=True, train_expert_only=True behaviour.
    # vlm_lr_multiplier<1.0 puts the VLM on a smaller LR than the expert when the VLM
    # is also trainable.
    vlm_lr_multiplier: float = 1.0

    # P1 trace-mode refinements. See docs/trace_prediction_p1_plan.md.
    # Number of log-spaced frequencies for the 2D uv positional embedding (§3.8).
    trace_uv_num_freqs: int = 8

    # P2. Suffix tokenization: one token per (keypoint, group of
    # `trace_group_horizon` adjacent timesteps). See docs/trace_prediction_p2_plan.md.
    # `history_len` and `future_len` must both be divisible by `trace_group_horizon`.
    trace_group_horizon: int = 1

    # History dropout for the suffix (see docs/trace_history_dropout_plan.md).
    # Both probabilities default to 0.0 (no-op) so existing checkpoints/runs are
    # unaffected. Applied only when `self.training=True`.
    #   trace_history_full_mask_prob: per-sample Bernoulli — when fired, drop
    #     every keypoint's history-group suffix tokens.
    #   trace_history_kp_dropout_prob: per-(sample, keypoint) Bernoulli — applied
    #     only on samples that did *not* hit the full-mask branch.
    trace_history_full_mask_prob: float = 0.0
    trace_history_kp_dropout_prob: float = 0.0

    # Depth input (see docs/trace_depth_smolvla_plan.md). When `use_depth=True`,
    # the trace dataset renders a turbo-colormapped depth RGB as a second vision
    # input; it goes through the same SigLIP encoder with a depth-only LoRA
    # adapter and a depth-only trainable clone of the patch-embedding stem.
    # RGB forward path is bit-equivalent to the pretrained encoder when
    # `use_depth=False` (all fields default to off).
    use_depth: bool = False
    depth_lora_rank: int = 8
    depth_lora_alpha: int = 16
    depth_lora_dropout: float = 0.0
    depth_lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )
    depth_clone_stem: bool = True
    # Per-sample Bernoulli dropout on the depth image at train time. When fired,
    # the dropped sample's depth-image `img_mask` is zeroed so all of its depth
    # tokens become non-attending in the prefix (the RGB branch is untouched).
    # Only applied when `self.training=True` and `use_depth=True`. 0.0 = no-op.
    depth_dropout_prob: float = 0.0

    # DINO per-keypoint visual feature (see docs/trace_dino_feature_plan.md).
    # When `use_dino=True`, a frozen DINOv2 backbone is run once per forward on
    # the frame-t RGB image; patch features are bilinearly sampled at each
    # keypoint's last-history (u, v) and concatenated into every suffix token
    # via a small MLP fusion head.
    use_dino: bool = False
    dino_model_name: str = "facebook/dinov2-base"
    dino_input_size: int = 518  # resized inside the model; must be multiple of DINO patch size (14 for v2).
    dino_num_prefix_tokens: int = 1  # 1 CLS, no registers for DINOv2-{small,base,large}.

    # adaLN-Zero conditioning for flow-matching time `t` (see
    # docs/trace_positional_encoding_plan.md). `t` is injected per-expert-block
    # via adaLN: shift/scale on pre-attn and pre-mlp LN outputs, gate on attn and
    # mlp outputs. Gates are zero-initialised so each block starts as identity.
    # Width of the sinusoidal-frequency input to the shared TimestepEmbedder MLP.
    trace_adaln_freq_embed_size: int = 256

    # Rigidity regularization on predicted control points. For keypoints with
    # the same DINO cluster id within a sample, the pairwise Euclidean distance
    # between their predicted clean control points should be invariant across
    # the D ctrl-pt axis (rigid body translation+rotation preserves internal
    # distances). Penalty is the variance of these pairwise distances over D,
    # averaged over (cluster, sample). 0 = off (no extra ops enter the graph).
    rigidity_loss_weight: float = 0.0

    # Per-keypoint "done" head (see docs/trace_done_head_plan.md). When True, an
    # H-wide head off a per-kp aggregator (mean over the kp's future ctrl-pt
    # tokens) predicts the `valid_future` label via BCE. At rollout the head's
    # sigmoid threshold is used to truncate-and-freeze the predicted trace so the
    # chaotic tail past the true horizon is suppressed. Backwards-compat default off.
    trace_done_head: bool = False
    done_loss_weight: float = 0.5
    done_bias_init: float = 3.0       # sigmoid(3) ≈ 0.95: untrained head says "valid".
    done_threshold: float = 0.5

    # B-spline ctrl-pt FM target (see docs/trace_bspline_ctrl_pts_plan.md).
    # The dataset emits a (D, 3) lstsq fit of the future trajectory and the model
    # runs Flow Matching in ctrl-pt space: x_t = t * noise + (1 - t) * P_gt, MSE
    # on (noise - P_gt). Spline is rendered to (H, 3) only at inference.
    # Number of control points D. Restricted to {4, 7, 10} because we reuse
    # the precomputed knot vectors from the Trace Anything codebase.
    trace_bspline_n_ctrl: int = 10
    # Future-side suffix grouping: how many ctrl pts pack into one suffix
    # token in flat tokenization. Must divide trace_bspline_n_ctrl. Independent
    # from trace_group_horizon (which keeps its history-side time-grouping role).
    trace_bspline_ctrl_per_token: int = 10
    # `delta_scale` is the per-axis (sx, sy, sz) factor used to scale future
    # deltas at the dataset; the model only reads it when smoothness loss is
    # active (to convert clean-hat scaled-deltas back to raw anchor-relative
    # units for the 2nd-diff penalty). Set by the trainer at policy build.
    delta_scale: tuple[float, float, float] | None = None

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    # Device (optional; if None, leaves placement to the caller).
    device: str | None = None

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.trace_mode and self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.trace_mode:
            if self.trace_dim != 3:
                raise ValueError(f"trace_dim must be 3 (x, y, z); got {self.trace_dim}")
            if self.history_len < 1 or self.future_len < 1:
                raise ValueError(
                    f"bad history/future lens: history_len={self.history_len} future_len={self.future_len}"
                )
            if self.n_min < 1 or self.n_max < self.n_min:
                raise ValueError(f"bad n_min/n_max: n_min={self.n_min} n_max={self.n_max}")
            if self.trace_uv_num_freqs < 1:
                raise ValueError(f"trace_uv_num_freqs must be >= 1; got {self.trace_uv_num_freqs}")
            g = self.trace_group_horizon
            if g < 1:
                raise ValueError(f"trace_group_horizon must be >= 1, got {g}")
            # In bspline mode the future side is no longer time-grouped, so
            # only the history-side divisibility constraint applies.
            if self.history_len % g != 0:
                raise ValueError(
                    f"trace_group_horizon={g} must divide history_len="
                    f"{self.history_len} (flat tokenization)."
                )
            for name, val in (
                ("trace_history_full_mask_prob", self.trace_history_full_mask_prob),
                ("trace_history_kp_dropout_prob", self.trace_history_kp_dropout_prob),
                ("depth_dropout_prob", self.depth_dropout_prob),
            ):
                if not 0.0 <= val <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]; got {val}")
            if self.depth_dropout_prob > 0.0 and not self.use_depth:
                raise ValueError(
                    "depth_dropout_prob > 0 requires use_depth=True "
                    f"(got depth_dropout_prob={self.depth_dropout_prob}, use_depth=False)"
                )
            if self.trace_adaln_freq_embed_size < 2 or self.trace_adaln_freq_embed_size % 2 != 0:
                raise ValueError(
                    f"trace_adaln_freq_embed_size must be a positive even integer; "
                    f"got {self.trace_adaln_freq_embed_size}"
                )
            if self.use_dino:
                if self.dino_input_size <= 0:
                    raise ValueError(
                        f"dino_input_size must be positive; got {self.dino_input_size}"
                    )
                # DINOv2 uses patch_size=14. Relax to a generic positive check for
                # DINOv3 / other variants (runtime will error loudly on mismatch).
                if "dinov2" in self.dino_model_name and self.dino_input_size % 14 != 0:
                    raise ValueError(
                        f"dino_input_size must be a multiple of 14 for DINOv2; "
                        f"got {self.dino_input_size} with model_name={self.dino_model_name!r}"
                    )
                if self.dino_num_prefix_tokens < 1:
                    raise ValueError(
                        f"dino_num_prefix_tokens must be >= 1 (CLS token); "
                        f"got {self.dino_num_prefix_tokens}"
                    )
            if self.trace_done_head:
                if self.done_loss_weight <= 0.0:
                    raise ValueError(
                        f"done_loss_weight must be > 0 when trace_done_head=True; "
                        f"got {self.done_loss_weight}"
                    )
                if not (0.0 < self.done_threshold < 1.0):
                    raise ValueError(
                        f"done_threshold must be in (0, 1); got {self.done_threshold}"
                    )
            if self.trace_bspline_n_ctrl not in (4, 7, 10):
                raise ValueError(
                    f"trace_bspline_n_ctrl must be in {{4, 7, 10}} "
                    f"(only the precomputed knot vectors are supported); "
                    f"got {self.trace_bspline_n_ctrl}"
                )
            if self.trace_bspline_ctrl_per_token < 1:
                raise ValueError(
                    f"trace_bspline_ctrl_per_token must be >= 1; "
                    f"got {self.trace_bspline_ctrl_per_token}"
                )
            if self.trace_bspline_n_ctrl % self.trace_bspline_ctrl_per_token != 0:
                raise ValueError(
                    f"trace_bspline_ctrl_per_token={self.trace_bspline_ctrl_per_token} "
                    f"must divide trace_bspline_n_ctrl={self.trace_bspline_n_ctrl}"
                )
            if self.rigidity_loss_weight < 0.0:
                raise ValueError(
                    f"rigidity_loss_weight must be >= 0; got {self.rigidity_loss_weight}"
                )

    def validate_features(self) -> None:
        if self.trace_mode:
            cam_key = f"{OBS_IMAGES}.cam_front"
            if cam_key not in self.input_features:
                self.input_features[cam_key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, *self.resize_imgs_with_padding),
                )
            if self.use_depth:
                depth_key = f"{OBS_IMAGES}.cam_front_depth"
                if depth_key not in self.input_features:
                    self.input_features[depth_key] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, *self.resize_imgs_with_padding),
                    )
            return

        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
