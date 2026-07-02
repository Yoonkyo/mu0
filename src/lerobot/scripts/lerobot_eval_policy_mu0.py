"""Evaluate a trained Mu0 RoboCasa policy by running rollouts in RoboCasa.

Loads the frozen Mu0 trace policy + trained action-decoder checkpoint produced
by ``lerobot_train_policy_mu0.py``, runs rollouts in the RoboCasa env, and
reports success rate / episode statistics.

Usage:
    python src/lerobot/scripts/lerobot_eval_policy_mu0.py \
        --trace_checkpoint=final_ckpt \
        --delta_stats_path=normalizer_stats.json \
        --policy_checkpoint=outputs/robocasa/mu0_policy/final \
        --task_name=PickPlaceCounterToCabinet \
        --num_rollouts=50
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import draccus
import numpy as np
import torch
from tqdm import trange

from lerobot.datasets.trace_delta_stats import load_trace_stats
from lerobot.policies.mu0.mu0_policy import GRID_SIZE, Mu0Policy, Mu0PolicyConfig


@dataclass
class EvalPolicyMu0Config:
    # Required
    trace_checkpoint: str = ""
    delta_stats_path: str = ""
    policy_checkpoint: str = ""  # path to checkpoint.pt or the directory containing it
    task_name: str = "PickPlaceCounterToCabinet"

    # Env
    camera_name: str = "robot0_agentview_left"
    gripper_camera_name: str = "robot0_eye_in_hand"
    observation_height: int = 256
    observation_width: int = 256

    # Eval
    num_rollouts: int = 50
    max_steps: int | None = None
    output_dir: str = ""  # default = policy_checkpoint parent dir
    num_videos_to_save: int = 0
    n_action_steps: int = -1  # -1 = execute the full chunk; N = execute only the first N actions per prediction

    # Policy (must match training)
    action_dim: int = 12
    chunk_size: int = 16
    state_dim: int = 16
    gripper_spatial_pool_size: int = 8
    use_trace: bool = True

    # Misc
    seed: int = 42
    device: str = "cuda"

    def build_policy_config(
        self,
        delta_scale: tuple[float, float, float] | None = None,
    ) -> Mu0PolicyConfig:
        return Mu0PolicyConfig(
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            state_dim=self.state_dim,
            gripper_spatial_pool_size=self.gripper_spatial_pool_size,
            use_trace=self.use_trace,
            delta_scale=delta_scale,
        )


def _required_checkpoint_prefixes(use_trace: bool) -> tuple[str, ...]:
    """Mirror of lerobot_train_policy_mu0._required_checkpoint_prefixes."""
    prefixes = [
        "gripper_proj.",
        "gripper_spatial_pos_embed",
        "action_decoder.gemma_expert.model.layers.0.",
        "action_decoder.gemma_expert.model.norm.",
        "action_decoder.action_in_proj.",
        "action_decoder.action_out_proj.",
        "action_decoder.time_mlp_in.",
        "action_decoder.time_mlp_out.",
        "action_decoder.z_norm.",
        "action_decoder.z_proj.",
        "action_decoder.lang_norm.",
        "action_decoder.lang_proj.",
        "action_decoder.visual_norm.",
        "action_decoder.visual_proj.",
        "action_decoder.state_proj.",
    ]
    if use_trace:
        prefixes.append("gated_mixer.")
    return tuple(prefixes)


def _find_camera_key(obs_pixels: dict, camera_name: str) -> str:
    """Find the key in obs['pixels'] that matches camera_name (exact or substring)."""
    for key in obs_pixels:
        if key == camera_name or camera_name in key:
            return key
    raise ValueError(
        f"Camera {camera_name!r} not found in obs pixels keys: {list(obs_pixels.keys())}"
    )


def _obs_to_batch(
    obs: dict,
    task_description: str,
    tokenizer,
    tokenizer_max_length: int,
    history_len: int,
    device: torch.device,
    camera_name: str,
    gripper_camera_name: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Convert a single env observation dict to Mu0Policy batch format."""
    from lerobot.datasets.trace_dataset import (
        CURRENT_KP_KEY,
        IMAGE_KEY,
        KEY_PAD_MASK_KEY,
        TRAJ_HISTORY_KEY,
    )
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    trace_batch: dict[str, torch.Tensor] = {}

    # --- Third-view image ---
    img_key = _find_camera_key(obs["pixels"], camera_name)
    img_np = obs["pixels"][img_key]  # (H, W, 3) uint8
    img_t = torch.from_numpy(img_np.copy()).permute(2, 0, 1).float() / 255.0
    trace_batch[IMAGE_KEY] = img_t.unsqueeze(0).to(device)  # (1, 3, H, W)

    # --- Language ---
    task_str = f"Task: {task_description.strip()}\n" if task_description else "Task: manipulate object\n"
    tok_out = tokenizer(
        [task_str],
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    trace_batch[OBS_LANGUAGE_TOKENS] = tok_out["input_ids"].to(device)
    trace_batch[OBS_LANGUAGE_ATTENTION_MASK] = tok_out["attention_mask"].bool().to(device)

    # --- Keypoints (uniform GRID_SIZE x GRID_SIZE grid; matches training) ---
    u = torch.linspace(-1.0, 1.0, GRID_SIZE, device=device)
    v = torch.linspace(-1.0, 1.0, GRID_SIZE, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    grid_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    num_kp = grid_uv.shape[0]
    current_kp = torch.zeros(1, num_kp, 3, device=device)
    current_kp[0, :, :2] = grid_uv
    trace_batch[CURRENT_KP_KEY] = current_kp
    trace_batch[TRAJ_HISTORY_KEY] = torch.zeros(1, num_kp, history_len, 3, device=device)
    trace_batch[KEY_PAD_MASK_KEY] = torch.ones(1, num_kp, dtype=torch.bool, device=device)

    # --- Proprioception (16D) ---
    if "agent_pos" in obs:
        proprio = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32))
        proprio = proprio.unsqueeze(0).to(device)
    else:
        proprio = torch.zeros(1, 16, device=device)

    # --- Gripper image ---
    grip_key = _find_camera_key(obs["pixels"], gripper_camera_name)
    grip_np = obs["pixels"][grip_key]
    grip_t = torch.from_numpy(grip_np.copy()).permute(2, 0, 1).float() / 255.0
    gripper_image = grip_t.unsqueeze(0).to(device)

    return trace_batch, proprio, gripper_image


@draccus.wrap()
def evaluate(cfg: EvalPolicyMu0Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    # ------------------------------------------------------------------
    # 1. Load delta_stats (for delta_scale) and build the model
    # ------------------------------------------------------------------
    delta_scale = None
    if cfg.delta_stats_path:
        trace_stats = load_trace_stats(cfg.delta_stats_path)
        delta_scale = trace_stats.delta_scale
        logging.info("Loaded delta_stats: delta_scale=%s", delta_scale)

    logging.info("Loading trace checkpoint: %s", cfg.trace_checkpoint)
    model = Mu0Policy(
        trace_checkpoint_path=cfg.trace_checkpoint,
        config=cfg.build_policy_config(delta_scale=delta_scale),
        trace_device=device,
    )

    # ------------------------------------------------------------------
    # 2. Load trained policy checkpoint (action decoder + gripper proj + gated mixer)
    # ------------------------------------------------------------------
    policy_ckpt_path = Path(cfg.policy_checkpoint)
    if policy_ckpt_path.is_dir():
        policy_ckpt_path = policy_ckpt_path / "checkpoint.pt"
    if not policy_ckpt_path.is_file():
        raise FileNotFoundError(f"policy_checkpoint not found: {policy_ckpt_path}")

    logging.info("Loading policy checkpoint: %s", policy_ckpt_path)
    ckpt = torch.load(policy_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    loaded_keys = set(state_dict.keys())
    for prefix in _required_checkpoint_prefixes(cfg.use_trace):
        if not any(key.startswith(prefix) for key in loaded_keys):
            raise RuntimeError(
                f"Policy checkpoint is missing keys with prefix {prefix!r}. "
                "Either the checkpoint comes from a different policy variant, "
                "or it was saved with a buggy save filter."
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info(
        "Loaded policy checkpoint: %d keys, missing=%d, unexpected=%d",
        len(state_dict),
        len(missing),
        len(unexpected),
    )
    if unexpected:
        logging.warning("Unexpected checkpoint keys (first 8): %s", unexpected[:8])

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 3. Tokenizer (matches the trace model's VLM)
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    pcfg = model.trace_policy.config
    tokenizer = AutoTokenizer.from_pretrained(pcfg.vlm_model_name)
    tokenizer.padding_side = "right"
    tokenizer_max_length = pcfg.tokenizer_max_length
    history_len = pcfg.history_len

    # ------------------------------------------------------------------
    # 4. Build RoboCasa environment
    # ------------------------------------------------------------------
    from lerobot.envs.robocasa import RoboCasaEnv, remap_dataset_action_to_env

    env = RoboCasaEnv(
        task_name=cfg.task_name,
        camera_name=f"{cfg.camera_name},{cfg.gripper_camera_name}",
        obs_type="pixels_agent_pos",
        observation_width=cfg.observation_width,
        observation_height=cfg.observation_height,
        seed=cfg.seed,
    )

    max_steps = cfg.max_steps if cfg.max_steps is not None else env.max_episode_steps
    logging.info(
        "Task: %s | max_steps: %d | num_rollouts: %d",
        cfg.task_name,
        max_steps,
        cfg.num_rollouts,
    )

    # ------------------------------------------------------------------
    # 5. Output dir + video setup
    # ------------------------------------------------------------------
    out_dir = Path(cfg.output_dir) if cfg.output_dir else policy_ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir: Path | None = None
    if cfg.num_videos_to_save > 0:
        video_dir = out_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Saving the first %d rollout videos to %s", cfg.num_videos_to_save, video_dir)

    # ------------------------------------------------------------------
    # 6. Rollout loop
    # ------------------------------------------------------------------
    successes: list[bool] = []
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []

    for ep in trange(cfg.num_rollouts, desc="rollouts"):
        obs, info = env.reset(seed=cfg.seed + ep)
        task_description = env.task_description or cfg.task_name

        action_queue: deque[np.ndarray] = deque()
        ep_reward = 0.0
        save_video = ep < cfg.num_videos_to_save
        frames: list[np.ndarray] = []
        step = 0

        if save_video:
            third_view_key = _find_camera_key(obs["pixels"], cfg.camera_name)
            frames.append(obs["pixels"][third_view_key].copy())

        for step in range(max_steps):
            if len(action_queue) == 0:
                batch, proprio, gripper_image = _obs_to_batch(
                    obs,
                    task_description,
                    tokenizer,
                    tokenizer_max_length,
                    history_len,
                    device,
                    camera_name=cfg.camera_name,
                    gripper_camera_name=cfg.gripper_camera_name,
                )
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    output = model(batch, proprio, gripper_image=gripper_image)
                actions = output["actions"][0].float().cpu().numpy()  # (chunk_size, action_dim)
                n_use = len(actions) if cfg.n_action_steps < 0 else cfg.n_action_steps
                for action in actions[:n_use]:
                    action_queue.append(action)

            action = remap_dataset_action_to_env(action_queue.popleft())
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if save_video:
                third_view_key = _find_camera_key(obs["pixels"], cfg.camera_name)
                frames.append(obs["pixels"][third_view_key].copy())

            if terminated or truncated:
                break

        is_success = bool(info.get("is_success", False))
        successes.append(is_success)
        episode_lengths.append(step + 1)
        episode_rewards.append(ep_reward)

        if save_video and frames and video_dir is not None:
            import cv2

            success_str = "success" if is_success else "fail"
            video_path = video_dir / f"ep{ep:03d}_{success_str}.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                env.metadata.get("render_fps", 20),
                (w, h),
            )
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            logging.info("  Saved video: %s (%d frames)", video_path, len(frames))

        if (ep + 1) % 10 == 0 or ep == cfg.num_rollouts - 1:
            sr = sum(successes) / len(successes) * 100
            logging.info(
                "  [%d/%d] success_rate=%.1f%% avg_len=%.0f avg_reward=%.2f",
                ep + 1,
                cfg.num_rollouts,
                sr,
                float(np.mean(episode_lengths)),
                float(np.mean(episode_rewards)),
            )

    env.close()

    # ------------------------------------------------------------------
    # 7. Report and save
    # ------------------------------------------------------------------
    success_rate = sum(successes) / len(successes) * 100
    avg_length = float(np.mean(episode_lengths))
    avg_reward = float(np.mean(episode_rewards))

    results = {
        "task_name": cfg.task_name,
        "num_rollouts": cfg.num_rollouts,
        "success_rate": success_rate,
        "avg_episode_length": avg_length,
        "avg_reward": avg_reward,
        "successes": successes,
        "episode_lengths": episode_lengths,
    }

    logging.info("=" * 50)
    logging.info("Task: %s", cfg.task_name)
    logging.info("Success Rate: %.1f%% (%d/%d)", success_rate, sum(successes), len(successes))
    logging.info("Avg Episode Length: %.0f", avg_length)
    logging.info("Avg Reward: %.2f", avg_reward)
    logging.info("=" * 50)

    out_path = out_dir / f"eval_{cfg.task_name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logging.info("Results saved to %s", out_path)


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
