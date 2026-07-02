"""Mu0 policy wrapper for RoboCasa trace-to-action training.

This is the narrowed version of the RoboCasa ``action_decoder.py`` path:
the frozen Mu0 trace policy provides VLM/expert features, and a trainable
Gemma action expert predicts action chunks with flow matching.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE
from lerobot.utils.device_utils import get_safe_dtype


GATED_MIXER_NUM_HEADS = 16
GATED_MIXER_GATE_INIT = 0.0
GEMMA_MAX_ACTION_DIM = 32
GEMMA_PREFIX_SCALE = 1.0
GEMMA_ACTION_EXPERT_VARIANT = "gemma_300m"
GRID_SIZE = 20
NUM_INFERENCE_STEPS = 10
MOTION_TOTAL_SOLVER_STEPS = 4
MOTION_GUIDANCE_STEPS = 1
MASK_ALL_HISTORY = True


@dataclass
class Mu0PolicyConfig:
    """Small RoboCasa action-policy config.

    Non-configurable choices are intentionally fixed to the values used by the
    integration run: Gemma scratch backend, language prefix enabled, no depth
    predictor, uniform 20x20 keypoints, spatial gripper tokens, and partial trace
    guidance with 4 total solver steps / 1 executed step.
    """

    action_dim: int = 12
    chunk_size: int = 16
    state_dim: int = 16
    gripper_spatial_pool_size: int = 8
    use_trace: bool = True
    debug_overlay_dir: str | None = None
    debug_overlay_every_n: int = 1
    delta_scale: tuple[float, float, float] | None = None

    min_period: float = 4e-3
    max_period: float = 4.0
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001


class GemmaConfig:
    """Local copy of the Gemma action-expert variant config."""

    def __init__(
        self,
        width: int,
        depth: int,
        mlp_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:
    if variant != GEMMA_ACTION_EXPERT_VARIANT:
        raise ValueError(f"Only {GEMMA_ACTION_EXPERT_VARIANT!r} is supported, got {variant!r}.")
    return GemmaConfig(
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device | str = "cpu",
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size,)`.")

    device = torch.device(device)
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """pi/openpi prefix-LM mask construction."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class GatedMotionMixer(nn.Module):
    """Single gated cross-attention block that injects trace expert features."""

    def __init__(
        self,
        vlm_hidden: int,
        expert_hidden: int,
        num_heads: int = GATED_MIXER_NUM_HEADS,
        gate_init: float = GATED_MIXER_GATE_INIT,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(vlm_hidden)
        self.norm_kv = nn.LayerNorm(vlm_hidden)
        self.w_proj = nn.Linear(expert_hidden, vlm_hidden)
        self.ca = nn.MultiheadAttention(vlm_hidden, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        z: torch.Tensor,
        z_m: torch.Tensor,
        q_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_motion = self.norm_kv(self.w_proj(z_m))
        ca_out, _ = self.ca(self.norm_q(z), h_motion, h_motion, need_weights=False)
        if q_padding_mask is not None:
            keep = (~q_padding_mask.to(torch.bool)).unsqueeze(-1).to(ca_out.dtype)
            ca_out = ca_out * keep
        return z + torch.sigmoid(self.gate) * ca_out


class GemmaActionExpert(nn.Module):
    """Gemma-only action expert used by Mu0Policy."""

    def __init__(
        self,
        config: Mu0PolicyConfig,
        vlm_hidden_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = config.chunk_size
        self.max_action_dim = GEMMA_MAX_ACTION_DIM
        self.prefix_scale = GEMMA_PREFIX_SCALE

        if config.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim={config.action_dim} exceeds fixed gemma max action dim "
                f"{self.max_action_dim}"
            )

        try:
            from transformers.models.auto import CONFIG_MAPPING

            from lerobot.policies.pi_gemma import PiGemmaForCausalLM
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GemmaActionExpert requires transformers and pi_gemma dependencies."
            ) from exc

        action_expert_config = get_gemma_config(GEMMA_ACTION_EXPERT_VARIANT)
        width = action_expert_config.width
        self.hidden_size = width

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=True,
            adarms_cond_dim=width,
        )
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self.gemma_expert.lm_head = None

        self.z_norm = nn.LayerNorm(vlm_hidden_size)
        self.z_proj = nn.Linear(vlm_hidden_size, width)
        self.lang_norm = nn.LayerNorm(vlm_hidden_size)
        self.lang_proj = nn.Linear(vlm_hidden_size, width)
        self.visual_norm = nn.LayerNorm(vlm_hidden_size)
        self.visual_proj = nn.Linear(vlm_hidden_size, width)
        self.state_proj = nn.Linear(config.state_dim, width)

        self.action_in_proj = nn.Linear(self.max_action_dim, width)
        self.action_out_proj = nn.Linear(width, self.max_action_dim)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] == self.max_action_dim:
            return actions
        if actions.shape[-1] > self.max_action_dim:
            raise ValueError(
                f"Action dim {actions.shape[-1]} exceeds fixed gemma max action dim "
                f"{self.max_action_dim}"
            )
        return F.pad(actions, (0, self.max_action_dim - actions.shape[-1]))

    def _build_prefix(
        self,
        visual_tokens: torch.Tensor,
        state: torch.Tensor,
        z_guided: torch.Tensor,
        vlm_key_padding_mask: torch.Tensor | None,
        language_tokens: torch.Tensor,
        language_key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.z_proj(self.z_norm(z_guided)) * self.prefix_scale
        language = (
            self.lang_proj(self.lang_norm(language_tokens.to(device=z_guided.device)))
            * self.prefix_scale
        )
        visual = self.visual_proj(self.visual_norm(visual_tokens)) * self.prefix_scale
        state_token = self.state_proj(state.to(dtype=visual.dtype)).unsqueeze(1) * self.prefix_scale
        prefix = torch.cat([z, language, visual, state_token], dim=1)

        if vlm_key_padding_mask is not None:
            z_valid = ~vlm_key_padding_mask.to(device=prefix.device, dtype=torch.bool)
            if z_valid.shape != z.shape[:2]:
                raise ValueError(
                    f"vlm_key_padding_mask shape {tuple(z_valid.shape)} does not match "
                    f"z_guided shape {tuple(z.shape[:2])}"
                )
        else:
            z_valid = torch.ones(z.shape[:2], dtype=torch.bool, device=prefix.device)

        if language_key_padding_mask is not None:
            language_valid = ~language_key_padding_mask.to(device=prefix.device, dtype=torch.bool)
            if language_valid.shape != language.shape[:2]:
                raise ValueError(
                    f"language_key_padding_mask shape {tuple(language_valid.shape)} does not match "
                    f"language tokens shape {tuple(language.shape[:2])}"
                )
        else:
            language_valid = torch.ones(language.shape[:2], dtype=torch.bool, device=prefix.device)

        pad_masks = torch.cat(
            [
                z_valid,
                language_valid,
                torch.ones(visual.shape[:2], dtype=torch.bool, device=prefix.device),
                torch.ones(state_token.shape[:2], dtype=torch.bool, device=prefix.device),
            ],
            dim=1,
        )
        att_masks = torch.zeros_like(pad_masks, dtype=torch.long)
        return prefix, pad_masks, att_masks

    def _embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_emb = self.action_in_proj(self._pad_actions(noisy_actions))
        timestep = timestep.to(device=noisy_actions.device, dtype=torch.float32)
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).to(dtype=action_emb.dtype)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        adarms_cond = F.silu(time_emb)

        bsize = action_emb.shape[0]
        pad_masks = torch.ones(bsize, action_emb.shape[1], dtype=torch.bool, device=action_emb.device)
        att_masks = torch.tensor(
            [1] + ([0] * (self.chunk_size - 1)),
            dtype=torch.long,
            device=action_emb.device,
        )[None, :].expand(bsize, -1)
        return action_emb, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        visual_tokens: torch.Tensor,
        state: torch.Tensor,
        z_guided: torch.Tensor,
        vlm_key_padding_mask: torch.Tensor | None,
        language_tokens: torch.Tensor,
        language_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        prefix, prefix_pad_masks, prefix_att_masks = self._build_prefix(
            visual_tokens,
            state,
            z_guided,
            vlm_key_padding_mask,
            language_tokens,
            language_key_padding_mask,
        )
        suffix, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix(
            noisy_actions,
            timestep,
        )
        full_tokens = torch.cat([prefix, suffix], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = _make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        gemma_dtype = self.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        self.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs = self.gemma_expert.model.forward(
            inputs_embeds=full_tokens.to(dtype=gemma_dtype),
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            adarms_cond=adarms_cond.to(dtype=gemma_dtype),
        )
        action_out = outputs.last_hidden_state[:, -self.chunk_size :].to(dtype=torch.float32)
        return self.action_out_proj(action_out)[..., : self.config.action_dim]


class Mu0Policy(nn.Module):
    """Frozen Mu0 trace policy plus trainable Gemma action decoder."""

    def __init__(
        self,
        trace_checkpoint_path: str,
        config: Mu0PolicyConfig,
        trace_device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.mu0.modeling_smolvla import SmolVLAPolicy

        trace_config = PreTrainedConfig.from_pretrained(trace_checkpoint_path)
        if trace_device is not None:
            trace_config.device = str(trace_device)
        self.trace_policy = SmolVLAPolicy.from_pretrained(
            trace_checkpoint_path,
            config=trace_config,
            strict=False,
        )
        self.trace_model = self.trace_policy.model
        for param in self.trace_model.parameters():
            param.requires_grad = False

        text_config = self.trace_model.vlm_with_expert.config.text_config
        vlm_hidden = text_config.hidden_size
        expert_hidden = self.trace_model.vlm_with_expert.expert_hidden_size

        img_size = self.trace_policy.config.resize_imgs_with_padding
        vm = self.trace_model.vlm_with_expert.get_vlm_model().vision_model
        vm_device = next(vm.parameters()).device
        vm_dtype = next(vm.parameters()).dtype
        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, *img_size, device=vm_device, dtype=vm_dtype)
            self._num_img_patches = int(
                self.trace_model.vlm_with_expert.embed_image(dummy_img).shape[1]
            )

        self.dino = self.trace_model.dino
        d_dino = self.dino.config.hidden_size
        self.gripper_proj = nn.Linear(d_dino, vlm_hidden)
        self.gripper_spatial_pos_embed = nn.Parameter(
            torch.randn(config.gripper_spatial_pool_size**2, vlm_hidden) * 0.02
        )

        self.gated_mixer = GatedMotionMixer(vlm_hidden=vlm_hidden, expert_hidden=expert_hidden)
        if not config.use_trace:
            self.gated_mixer.requires_grad_(False)

        self.action_decoder = GemmaActionExpert(
            config=config,
            vlm_hidden_size=vlm_hidden,
        )

        n_dec = sum(param.numel() for param in self.action_decoder.parameters())
        n_fuse = sum(param.numel() for param in self.gated_mixer.parameters())
        n_fuse_trainable = sum(
            param.numel() for param in self.gated_mixer.parameters() if param.requires_grad
        )
        print(
            f"[params] action_decoder={n_dec / 1e6:.1f}M "
            f"gated_mixer={n_fuse / 1e6:.1f}M(trainable={n_fuse_trainable / 1e6:.1f}M) "
            f"use_trace={config.use_trace}"
        )

        self._debug_forward_counter = 0

    def _compute_gripper_dino_features(self, gripper_image: torch.Tensor) -> torch.Tensor:
        dino_mean = self.trace_model._dino_mean.to(gripper_image.device)
        dino_std = self.trace_model._dino_std.to(gripper_image.device)
        x = (gripper_image - dino_mean) / dino_std
        size = self.trace_policy.config.dino_input_size
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = self.dino(pixel_values=x.to(dtype=self.dino.dtype))
        num_prefix = self.trace_policy.config.dino_num_prefix_tokens
        return out.last_hidden_state[:, num_prefix:]

    def _compute_visual_tokens(self, gripper_image: torch.Tensor) -> torch.Tensor:
        if gripper_image is None:
            raise ValueError("gripper_image is required for Mu0Policy.")

        with torch.no_grad():
            gripper_dino_feats = self._compute_gripper_dino_features(gripper_image)

        bsize, num_patches, dim = gripper_dino_feats.shape
        patch_side = int(num_patches**0.5)
        if patch_side * patch_side != num_patches:
            raise ValueError(
                f"Expected square DINO patch grid; got {num_patches} tokens."
            )
        pool_side = self.config.gripper_spatial_pool_size
        feats_chw = gripper_dino_feats.transpose(1, 2).reshape(bsize, dim, patch_side, patch_side)
        feats_pooled = F.adaptive_avg_pool2d(feats_chw, (pool_side, pool_side))
        visual_tokens = feats_pooled.flatten(2).transpose(1, 2)
        visual_tokens = self.gripper_proj(visual_tokens)
        pos = self.gripper_spatial_pos_embed.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        return visual_tokens + pos.unsqueeze(0)

    def _reduce_expert(self, emb: torch.Tensor) -> torch.Tensor:
        n_kp = GRID_SIZE**2
        bsize, num_tokens, dim = emb.shape
        if num_tokens % n_kp == 0 and num_tokens // n_kp > 1:
            emb = emb.reshape(bsize, n_kp, num_tokens // n_kp, dim)[:, :, -1, :]
        if emb.shape[1] == GRID_SIZE * GRID_SIZE:
            emb = emb.reshape(bsize, GRID_SIZE, GRID_SIZE, dim).permute(0, 3, 1, 2)
            emb = F.avg_pool2d(emb, 2, 2)
            emb = emb.permute(0, 2, 3, 1).reshape(bsize, -1, dim)
        return emb

    def _build_vlm_key_padding_mask(
        self,
        img_masks: list[torch.Tensor],
        lang_masks: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        bsize = lang_masks.shape[0]
        device = lang_masks.device
        valid_parts: list[torch.Tensor] = []

        add_special = bool(getattr(self.trace_model, "add_image_special_tokens", False))
        image_start_len = int(self.trace_model.global_image_start_token.numel()) if add_special else 0
        image_end_len = int(self.trace_model.image_end_token.numel()) if add_special else 0

        for img_mask in img_masks:
            img_mask = img_mask.to(device=device, dtype=torch.bool).reshape(bsize)
            if image_start_len:
                valid_parts.append(img_mask[:, None].expand(bsize, image_start_len))
            valid_parts.append(img_mask[:, None].expand(bsize, self._num_img_patches))
            if image_end_len:
                valid_parts.append(img_mask[:, None].expand(bsize, image_end_len))

        valid_parts.append(lang_masks.to(device=device, dtype=torch.bool))
        valid_mask = torch.cat(valid_parts, dim=1)
        if valid_mask.shape[1] < target_len:
            pad = torch.zeros(bsize, target_len - valid_mask.shape[1], dtype=torch.bool, device=device)
            valid_mask = torch.cat([valid_mask, pad], dim=1)
        elif valid_mask.shape[1] > target_len:
            raise ValueError(
                f"Rebuilt VLM prefix mask length {valid_mask.shape[1]} exceeds "
                f"VLM prefix length {target_len}"
            )
        return ~valid_mask

    def _denoise_step_trace_with_expert_output(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        traj_history: torch.Tensor,
        current_uv: torch.Tensor,
        x_t_future: torch.Tensor,
        key_pad_mask: torch.Tensor,
        timestep: torch.Tensor,
        dino_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        from lerobot.policies.mu0.modeling_smolvla import make_att_2d_masks

        trace_model = self.trace_model
        traj_input = torch.cat([traj_history, x_t_future], dim=2)
        suffix_embs, suffix_pad_masks, suffix_att_masks = trace_model.embed_suffix_trace(
            traj_input,
            current_uv,
            timestep,
            key_pad_mask,
            dino_feat=dino_feat,
            mask_all_history=MASK_ALL_HISTORY,
        )
        expert_adaln_modulation = trace_model._build_expert_adaln_modulation(timestep)

        bsize, num_kp = traj_history.shape[:2]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        suffix_att_2d_masks = trace_model._apply_causal_suffix_mask(
            suffix_att_2d_masks,
            num_kp,
            trace_model._suffix_groups_per_kp(),
        )
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = prefix_pad_masks.sum(dim=-1, keepdim=True)
        position_ids = prefix_offsets.expand(-1, suffix_len)

        outputs_embeds, _, pre_final_norm_embeds = trace_model.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            inputs_embeds=[None, suffix_embs],
            use_cache=trace_model.config.use_cache,
            fill_kv_cache=False,
            expert_adaln_modulation=expert_adaln_modulation,
            return_pre_final_norm=True,
        )
        expert_for_mixer = pre_final_norm_embeds[1]
        if expert_for_mixer is None:
            raise RuntimeError("Trace denoise forward did not return expert final output.")
        suffix_out = outputs_embeds[1]
        if suffix_out is None:
            raise RuntimeError("Trace denoise forward did not return expert normalized output.")
        v_t = trace_model._suffix_out_to_velocity(suffix_out, num_kp)
        done_logits = trace_model._suffix_out_to_done_logits(suffix_out, num_kp)
        return v_t, done_logits, expert_for_mixer

    def _run_trace_model_partial_guidance(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        traj_history: torch.Tensor,
        current_kp: torch.Tensor,
        key_pad_mask: torch.Tensor,
        is_depth_flags: list[bool] | None,
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        from lerobot.policies.mu0.modeling_smolvla import make_att_2d_masks

        bsize, num_kp, _, _ = traj_history.shape
        device = traj_history.device
        cfg = self.trace_model.config
        trace_bspline_mode = bool(getattr(cfg, "trace_bspline_mode", True))
        f_dim = cfg.trace_bspline_n_ctrl if trace_bspline_mode else cfg.future_len
        noise = self.trace_model.sample_noise((bsize, num_kp, f_dim, cfg.trace_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.trace_model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=None,
            is_depth_flags=is_depth_flags,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values, pre_final_norm_embeds = self.trace_model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=cfg.use_cache,
            fill_kv_cache=True,
            return_pre_final_norm=True,
        )
        vlm_final = pre_final_norm_embeds[0]
        if vlm_final is None:
            raise RuntimeError("Trace prefix forward did not return VLM final output.")

        current_uv = current_kp[:, :, :2]
        dino_feat = self.trace_model._compute_dino_per_kp(
            images,
            is_depth_flags,
            current_uv,
            key_pad_mask,
        )

        dt = -1.0 / MOTION_TOTAL_SOLVER_STEPS
        x_future = noise
        last_done_logits: torch.Tensor | None = None
        expert_final: torch.Tensor | None = None
        for step in range(MOTION_GUIDANCE_STEPS):
            time_value = 1.0 + step * dt
            time_tensor = torch.tensor(time_value, dtype=torch.float32, device=device).expand(bsize)
            v_t, done_logits, expert_final = self._denoise_step_trace_with_expert_output(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                traj_history=traj_history,
                current_uv=current_uv,
                x_t_future=x_future,
                key_pad_mask=key_pad_mask,
                timestep=time_tensor,
                dino_feat=dino_feat,
            )
            x_future = x_future + dt * v_t
            last_done_logits = done_logits
        if expert_final is None:
            raise RuntimeError("Trace denoise did not run, so expert final output is unavailable.")

        if trace_bspline_mode:
            basis = self.trace_model._bspline_render_basis.to(device=x_future.device, dtype=x_future.dtype)
            traces = torch.einsum("hd,bndc->bnhc", basis, x_future).contiguous()
        else:
            traces = x_future

        if cfg.trace_done_head:
            if last_done_logits is None:
                raise RuntimeError("trace_done_head=True but partial denoise did not produce done logits.")
            return {"traces": traces, "done_prob": torch.sigmoid(last_done_logits)}, vlm_final, expert_final
        return traces, vlm_final, expert_final

    def _run_vlm_prefix_only(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from lerobot.datasets.trace_dataset import TRAJ_HISTORY_KEY
        from lerobot.policies.mu0.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, img_masks, is_depth_flags = self.trace_policy.prepare_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.trace_model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=None,
                is_depth_flags=is_depth_flags,
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, _, pre_final_norm_embeds = self.trace_model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                fill_kv_cache=True,
                return_pre_final_norm=True,
            )

        vlm_final = pre_final_norm_embeds[0]
        if vlm_final is None:
            raise RuntimeError("VLM prefix forward did not return final output.")
        vlm_key_padding_mask = self._build_vlm_key_padding_mask(
            img_masks,
            lang_masks,
            target_len=vlm_final.shape[1],
        )
        traj_history = batch[TRAJ_HISTORY_KEY]
        bsize, num_kp = traj_history.shape[:2]
        future_len = self.trace_model.config.future_len
        trace_tensor = torch.zeros(
            bsize,
            num_kp,
            future_len,
            self.trace_model.config.trace_dim,
            device=traj_history.device,
            dtype=traj_history.dtype,
        )
        return vlm_final, trace_tensor, vlm_key_padding_mask

    def _run_trace_model(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from lerobot.datasets.trace_dataset import (
            CURRENT_KP_KEY,
            KEY_PAD_MASK_KEY,
            TRAJ_HISTORY_KEY,
        )
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, img_masks, is_depth_flags = self.trace_policy.prepare_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        traj_history = batch[TRAJ_HISTORY_KEY]
        current_kp = batch[CURRENT_KP_KEY]
        key_pad_mask = batch[KEY_PAD_MASK_KEY]

        with torch.no_grad():
            traces, vlm_final, expert_final = self._run_trace_model_partial_guidance(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                traj_history=traj_history,
                current_kp=current_kp,
                key_pad_mask=key_pad_mask,
                is_depth_flags=is_depth_flags,
            )

        trace_tensor = traces["traces"] if isinstance(traces, dict) else traces
        vlm_key_padding_mask = self._build_vlm_key_padding_mask(
            img_masks,
            lang_masks,
            target_len=vlm_final.shape[1],
        )
        return vlm_final, expert_final, trace_tensor, vlm_key_padding_mask

    def _compute_action_language_tokens(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
            raise ValueError("Mu0Policy requires language tokens and attention mask in the batch.")
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device=lang_tokens.device, dtype=torch.bool)
        with torch.no_grad():
            language_tokens = self.trace_model.vlm_with_expert.embed_language_tokens(lang_tokens)
        return language_tokens.detach(), ~lang_masks

    def _flow_matching_loss(
        self,
        e_v: torch.Tensor,
        proprioception: torch.Tensor,
        z_guided: torch.Tensor,
        actions: torch.Tensor,
        vlm_key_padding_mask: torch.Tensor | None,
        language_tokens: torch.Tensor,
        language_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = actions.shape[0]
        device = actions.device
        alpha = torch.tensor(self.config.time_sampling_beta_alpha, dtype=torch.float32)
        beta = torch.tensor(self.config.time_sampling_beta_beta, dtype=torch.float32)
        sample = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(
            device=device,
            dtype=actions.dtype,
        )
        t = sample * self.config.time_sampling_scale + self.config.time_sampling_offset
        t_b = t[:, None, None]
        noise = torch.randn_like(actions)
        x_t = t_b * noise + (1.0 - t_b) * actions
        v_target = noise - actions
        v_pred = self.action_decoder(
            x_t,
            t.to(dtype=torch.float32),
            e_v,
            proprioception,
            z_guided,
            vlm_key_padding_mask=vlm_key_padding_mask,
            language_tokens=language_tokens,
            language_key_padding_mask=language_key_padding_mask,
        )
        return F.mse_loss(v_pred, v_target)

    def _sample_actions(
        self,
        e_v: torch.Tensor,
        proprioception: torch.Tensor,
        z_guided: torch.Tensor,
        vlm_key_padding_mask: torch.Tensor | None,
        language_tokens: torch.Tensor,
        language_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = e_v.shape[0]
        dt = -1.0 / NUM_INFERENCE_STEPS
        x = torch.randn(
            batch_size,
            self.config.chunk_size,
            self.config.action_dim,
            device=e_v.device,
            dtype=e_v.dtype,
        )
        for step in range(NUM_INFERENCE_STEPS):
            time = 1.0 + step * dt
            decoder_timestep = torch.full((batch_size,), time, device=e_v.device, dtype=torch.float32)
            v = self.action_decoder(
                x,
                decoder_timestep,
                e_v,
                proprioception,
                z_guided,
                vlm_key_padding_mask=vlm_key_padding_mask,
                language_tokens=language_tokens,
                language_key_padding_mask=language_key_padding_mask,
            )
            x = x + dt * v
        return x

    def _save_trace_overlay(self, batch: dict[str, torch.Tensor], trace_tensor: torch.Tensor) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        if self.config.delta_scale is None:
            return

        import numpy as np
        from PIL import Image

        from lerobot.datasets.trace_dataset import CURRENT_KP_KEY, IMAGE_KEY, KEY_PAD_MASK_KEY
        from lerobot.policies.mu0.visualize_trace import render_trace_overlay

        out_dir = Path(self.config.debug_overlay_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        current_kp = batch[CURRENT_KP_KEY]
        step_by_step = getattr(self.trace_policy.config, "step_by_step_delta", False)
        traces_anchor = torch.cumsum(trace_tensor, dim=2) if step_by_step else trace_tensor
        scale = torch.as_tensor(
            self.config.delta_scale,
            dtype=traces_anchor.dtype,
            device=traces_anchor.device,
        )
        traces_abs = traces_anchor * scale + current_kp[:, :, None, :3]

        img_chw = batch[IMAGE_KEY][0]
        img_hwc = (img_chw.permute(1, 2, 0).float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        num_kp, future_steps = traces_abs.shape[1], traces_abs.shape[2]
        key_pad_mask = batch.get(KEY_PAD_MASK_KEY, None)
        trace_group_horizon = getattr(self.trace_policy.config, "trace_group_horizon", 1)
        trace_bspline_mode = getattr(self.trace_policy.config, "trace_bspline_mode", True)
        pred_color_mode = "per_group" if (trace_group_horizon > 1 and not trace_bspline_mode) else "per_kp"

        overlay = render_trace_overlay(
            image=img_hwc,
            traj_history=np.zeros((num_kp, 1, 3), dtype=np.float32),
            traj_future_pred=traces_abs[0].detach().cpu().numpy(),
            traj_future_gt=None,
            valid_history=np.zeros((num_kp, 1), dtype=bool),
            valid_future=np.ones((num_kp, future_steps), dtype=bool),
            current_kp=current_kp[0, :, :3].detach().cpu().numpy(),
            key_pad_mask=(key_pad_mask[0].detach().cpu().numpy() if key_pad_mask is not None else None),
            prepend_current=True,
            draw_last_uv_marker=False,
            pred_color_mode=pred_color_mode,
            pred_group_size=trace_group_horizon,
        )
        filename = f"debug_step{self._debug_forward_counter:08d}.png"
        Image.fromarray(overlay).save(out_dir / filename)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        proprioception: torch.Tensor,
        gripper_image: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.config.use_trace:
            (
                vlm_last,
                expert_final,
                trace_tensor,
                vlm_key_padding_mask,
            ) = self._run_trace_model(batch)
            expert_last = self._reduce_expert(expert_final)
        else:
            vlm_last, trace_tensor, vlm_key_padding_mask = self._run_vlm_prefix_only(batch)
            expert_last = None

        if self.config.debug_overlay_dir:
            if self._debug_forward_counter % self.config.debug_overlay_every_n == 0:
                self._save_trace_overlay(batch, trace_tensor)
            self._debug_forward_counter += 1

        if self.config.use_trace:
            if expert_last is None:
                raise RuntimeError("use_trace=True requires trace expert features.")
            z_for_decoder = self.gated_mixer(
                vlm_last,
                expert_last,
                q_padding_mask=vlm_key_padding_mask,
            )
        else:
            z_for_decoder = vlm_last

        e_v = self._compute_visual_tokens(gripper_image)
        language_tokens, language_key_padding_mask = self._compute_action_language_tokens(batch)

        if actions is not None:
            loss = self._flow_matching_loss(
                e_v,
                proprioception,
                z_for_decoder,
                actions,
                vlm_key_padding_mask=vlm_key_padding_mask,
                language_tokens=language_tokens,
                language_key_padding_mask=language_key_padding_mask,
            )
            return {"loss": loss, "traces": trace_tensor}

        pred_actions = self._sample_actions(
            e_v,
            proprioception,
            z_for_decoder,
            vlm_key_padding_mask=vlm_key_padding_mask,
            language_tokens=language_tokens,
            language_key_padding_mask=language_key_padding_mask,
        )
        return {"actions": pred_actions, "traces": trace_tensor}
