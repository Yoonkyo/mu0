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

import json
import os
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import robocasa  # noqa: F401 — registers envs with gymnasium
import robosuite
from gymnasium import spaces
from robosuite.controllers import load_part_controller_config

from lerobot.utils.constants import HF_LEROBOT_HOME


@dataclass
class EnvArgs:
    """Environment arguments for creating robosuite environments."""

    env_name: str = ""
    robots: str | list[str] = "PandaOmron"
    controller: str | None = "OSC_POSE"
    has_renderer: bool = False
    renderer: str = "mjviewer"
    control_freq: int = 20
    use_object_obs: bool = False
    use_camera_obs: bool = True
    camera_names: list = field(default_factory=lambda: ["robot0_agentview_left", "robot0_eye_in_hand"])
    camera_heights: int = 128
    camera_widths: int = 128
    camera_depths: bool = False
    seed: int = 0
    controller_configs: dict = field(default_factory=dict)
    layout_ids: list = field(default_factory=lambda: [-1])
    style_ids: list = field(default_factory=lambda: [-1])
    translucent_robot: bool = False
    reward_shaping: bool = False
    has_offscreen_renderer: bool = False
    ignore_done: bool = False
    render_collision_mesh: bool = False
    render_visual_mesh: bool = True
    render_gpu_device_id: int = -1

    def __post_init__(self):
        if list(self.controller_configs.keys()) == [] and robosuite.__version__ > "1.4.0":
            self.controller_configs = load_part_controller_config(
                default_controller=self.controller,
            )

    def env_dict(self):
        exclude_keys = ["controller"]
        return {k: v for k, v in self.__dict__.items() if k not in exclude_keys}


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list, tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _parse_env_args_from_hdf5(dataset_path: str) -> dict[str, Any]:
    """Extract environment arguments from dataset."""
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"]) if "data" in f else json.loads(f.attrs["env_args"])
        if isinstance(env_args, str):
            env_args = json.loads(env_args)
    return env_args


def _parse_env_meta_from_hdf5(dataset_path: str, episode_index: int = 0) -> dict[str, Any]:
    """Extract environment metadata from dataset."""
    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as f:
        data = f.get("data", f)
        keys = list(data.keys())
        env_meta = data[keys[episode_index]].attrs["ep_meta"]
        env_meta = json.loads(env_meta)
        assert isinstance(env_meta, dict), f"Expected dict type but got {type(env_meta)}"
    return env_meta


def _parse_env_meta_from_repo_id(repo_id: str, episode_index: int = 0) -> dict[str, Any]:
    """Extract environment metadata from dataset."""
    dataset_path = HF_LEROBOT_HOME / repo_id
    with open(dataset_path / "meta" / "episodes" / "ep_metas.json") as f:
        env_metas = json.load(f)
    return env_metas[episode_index]


def get_robocasa_zero_action(env):
    """Get zero/no-op action for PandaOmron robot."""
    active_robot = env.robots[0]
    if env.action_dim == 12:
        assert len(env.robots) == 1, "Only one robot is supported"
        assert env.robots[0].name == "PandaOmron", "Only PandaOmron is supported"
        arms = ["right"]
        zero_action_dict = {}
        for arm in arms:
            zero_action = np.zeros(7)
            if active_robot.part_controllers[arm].input_type == "absolute":
                raise NotImplementedError("Dummy actions assume relative actions")
            zero_action_dict[f"{arm}"] = zero_action[: zero_action.shape[0] - 1]
            zero_action_dict[f"{arm}_gripper"] = zero_action[zero_action.shape[0] - 1 :]
        zero_action_dict["base_mode"] = -1
        zero_action_dict["base"] = np.zeros(3)
        zero_action = active_robot.create_action_vector(zero_action_dict)
    else:
        raise NotImplementedError("Only 12D PandaOmron actions are supported")
    return zero_action


def remap_dataset_action_to_env(action: np.ndarray) -> np.ndarray:
    """Remap action from dataset order to env (robosuite) order.

    Dataset order (LeRobot/RoboCasa v1.0):
        [0:4]  = base_motion (x, y, yaw, ?)
        [4]    = control_mode (-1=arm, 1=base)
        [5:8]  = eef_pos delta (dx, dy, dz)
        [8:11] = eef_rot delta (drx, dry, drz)
        [11]   = gripper (-1=open, 1=close)

    Env order (robosuite create_action_vector):
        [0:6]  = arm OSC_POSE (dx, dy, dz, drx, dry, drz)
        [6]    = gripper
        [7:10] = base (x, y, yaw)
        [10]   = padding (torso, always 0)
        [11]   = base_mode
    """
    env_action = np.zeros(12, dtype=action.dtype)
    env_action[0:3] = action[5:8]    # eef_pos → arm[0:3]
    env_action[3:6] = action[8:11]   # eef_rot → arm[3:6]
    env_action[6] = action[11]       # gripper → [6]
    env_action[7:10] = action[0:3]   # base_motion[:3] → base[7:10]
    env_action[10] = 0.0             # padding
    env_action[11] = action[4]       # control_mode → base_mode[11]
    return env_action


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBS_STATE_DIM = 16
ACTION_DIM = 12
AGENT_POS_LOW = -1000.0
AGENT_POS_HIGH = 1000.0
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
DEFAULT_MAX_EPISODE_STEPS = 1000
DEFAULT_MAX_EPISODE_STEPS_BY_TASK = {
    # single_stage tasks
    "CloseDoubleDoor": 474,
    "CloseDrawer": 227,
    "CloseSingleDoor": 322,
    "CoffeePressButton": 156,
    "CoffeeServeMug": 433,
    "CoffeeSetupMug": 376,
    "NavigateKitchen": 322,
    "OpenDoubleDoor": 889,
    "OpenDrawer": 260,
    "OpenSingleDoor": 414,
    "PnPCabToCounter": 477,
    "PnPCounterToCab": 364,
    "PnPCounterToMicrowave": 509,
    "PnPCounterToSink": 680,
    "PnPCounterToStove": 404,
    "PnPMicrowaveToCounter": 430,
    "PnPSinkToCounter": 351,
    "PnPStoveToCounter": 417,
    "TurnOffMicrowave": 318,
    "TurnOffSinkFaucet": 336,
    "TurnOffStove": 338,
    "TurnOnMicrowave": 279,
    "TurnOnSinkFaucet": 342,
    "TurnOnStove": 349,
    "TurnSinkSpout": 187,
    # multi_stage tasks
    "ArrangeVegetables": 1132,
    "MicrowaveThawing": 906,
    "PreSoakPan": 1439,
    "PrepareCoffee": 980,
    "RestockPantry": 925,
}

# RoboCasa available cameras
ROBOCASA_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


class RoboCasaEnv(gym.Env):
    """RoboCasa environment wrapper for kitchen manipulation tasks.

    Observations returned by reset()/step():
      - pixels: RGB image(s) from selected camera(s), (H, W, 3) uint8
      - agent_pos: robot proprioception (16D) — base_pos(3) + base_rot(4) + eef_pos(3) + eef_rot(4) + gripper(2)

    Depth is handled at the policy level (DA-V2), not in the environment.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task_name: str,
        camera_name: str | Sequence[str] = "robot0_agentview_left,robot0_eye_in_hand",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        max_episode_steps: int | None = None,
        ep_meta: dict | None = None,
        seed: int = 0,
        return_raw_obs: bool = False,
        **env_kwargs,
    ):
        super().__init__()
        self.task_name = task_name
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.num_steps_wait = num_steps_wait
        self.max_episode_steps = max_episode_steps or DEFAULT_MAX_EPISODE_STEPS_BY_TASK.get(
            task_name, DEFAULT_MAX_EPISODE_STEPS
        )
        self._max_episode_steps = self.max_episode_steps
        self.return_raw_obs = return_raw_obs
        self._step_count = 0

        # Camera setup
        self.camera_name = _parse_camera_names(camera_name)

        if camera_name_mapping is None:
            camera_name_mapping = {
                "robot0_agentview_left_image": "robot0_agentview_left",
                "robot0_agentview_right_image": "robot0_agentview_right",
                "robot0_eye_in_hand_image": "robot0_eye_in_hand",
            }
        self.camera_name_mapping = camera_name_mapping

        # Create robosuite environment
        env_args = EnvArgs(
            env_name=task_name,
            robots="PandaOmron",
            controller="OSC_POSE",
            has_renderer=(render_mode == "human"),
            has_offscreen_renderer=(render_mode == "rgb_array"),
            use_camera_obs=(render_mode == "rgb_array"),
            camera_names=self.camera_name,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            camera_depths=False,
            seed=seed,
            style_ids=ep_meta.get("style_ids", [-1]) if ep_meta is not None else [-1],
            layout_ids=ep_meta.get("layout_ids", [-1]) if ep_meta is not None else [-1],
        )

        env_dict = env_args.env_dict()
        self._env = robosuite.make(**env_dict)
        if ep_meta is not None:
            self._env.set_ep_meta(ep_meta)
        self._env_args = env_args

        # ----- Observation space -----
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping.get(cam, cam)] = spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        obs_spaces: dict[str, spaces.Space] = {"pixels": spaces.Dict(images)}

        if self.obs_type == "pixels_agent_pos":
            obs_spaces["agent_pos"] = spaces.Box(
                low=AGENT_POS_LOW, high=AGENT_POS_HIGH,
                shape=(OBS_STATE_DIM,), dtype=np.float64,
            )
        elif self.obs_type != "pixels":
            raise ValueError(f"Unknown obs_type: {self.obs_type}")

        self.observation_space = spaces.Dict(obs_spaces)

        # ----- Action space -----
        # action_dim is None before reset() in robosuite/robocasa,
        # so use the known PandaOmron action dim (12).
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

        self.task = self.task_name
        self.task_description = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def render(self):
        raw_obs = self._env._get_observations()
        first_cam = self.camera_name[0]
        return self._format_raw_obs(raw_obs)["pixels"].get(
            self.camera_name_mapping.get(first_cam, first_cam)
        )

    def _format_raw_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        if self.return_raw_obs:
            return raw_obs

        images = {}
        for camera_name in self.camera_name:
            image_key = f"{camera_name}_image"
            if image_key in raw_obs:
                image = raw_obs[image_key]
                # Vertical flip to match standard image convention
                images[self.camera_name_mapping.get(camera_name, camera_name)] = image[::-1]
            else:
                raise ValueError(
                    f"Camera key '{image_key}' not found in raw observations: {list(raw_obs.keys())}"
                )

        obs: dict[str, Any] = {"pixels": images.copy()}

        # Robot proprioception (16D):
        #   base_pos(3) + base_rot(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper(2)
        if self.obs_type == "pixels_agent_pos":
            if "robot0_base_pos" in raw_obs:
                state = np.concatenate(
                    (
                        raw_obs["robot0_base_pos"],           # (3,)
                        raw_obs["robot0_base_quat"],          # (4,)
                        raw_obs["robot0_base_to_eef_pos"],    # (3,)
                        raw_obs["robot0_base_to_eef_quat"],   # (4,)
                        raw_obs["robot0_gripper_qpos"],       # (2,)
                    )
                )
            else:
                state = np.zeros(OBS_STATE_DIM)
            obs["agent_pos"] = state

        return obs

    def reset(self, seed: int | None = None, ep_meta: dict | None = None, **kwargs):
        super().reset(seed=seed)
        self._step_count = 0

        if ep_meta is not None:
            self._env.set_ep_meta(ep_meta)

        raw_obs = self._env.reset()
        self.task_description = self._env.get_ep_meta().get("lang", None)

        # Settle physics with no-op actions
        zero_action = get_robocasa_zero_action(self._env)
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(zero_action)

        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )

        self._step_count += 1
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env._check_success()
        terminated = done or is_success
        truncated = self._step_count >= self.max_episode_steps

        info.update({"task": self.task, "done": done, "is_success": is_success})
        observation = self._format_raw_obs(raw_obs)

        if terminated or truncated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": bool(is_success),
            }

        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Vectorized environment factory
# ---------------------------------------------------------------------------

def _make_env_fns(
    *,
    task_name: str,
    n_envs: int,
    camera_names: list[str],
    gym_kwargs: Mapping[str, Any],
    ep_metas: list[dict[str, Any]] | None = None,
) -> list[Callable[[], RoboCasaEnv]]:
    """Build n_envs factory callables for a task."""

    def _make_env(episode_index: int, **kwargs) -> RoboCasaEnv:
        local_kwargs = dict(kwargs)

        if ep_metas is not None:
            if len(ep_metas) <= episode_index:
                raise ValueError(
                    f"ep_metas list has {len(ep_metas)} elements, but episode_index {episode_index} "
                    f"requires at least {episode_index + 1} elements."
                )
            ep_meta = ep_metas[episode_index].copy()
        else:
            ep_meta = None

        seed = local_kwargs.pop("seed", episode_index)
        local_kwargs.pop("ep_meta", None)

        return RoboCasaEnv(
            task_name=task_name,
            camera_name=camera_names,
            seed=seed,
            ep_meta=ep_meta,
            **local_kwargs,
        )

    fns: list[Callable[[], RoboCasaEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


def create_robocasa_envs(
    task_name: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "",
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, Any]]:
    """Create vectorized RoboCasa environments.

    Returns:
        dict[suite_name][task_id] -> vec_env
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    gym_kwargs_camera_name = gym_kwargs.pop("camera_name", None)
    camera_name = camera_name if camera_name != "" else gym_kwargs_camera_name
    parsed_camera_names = _parse_camera_names(camera_name)

    ep_metas = gym_kwargs.pop("ep_metas", None)
    if ep_metas is not None:
        if not isinstance(ep_metas, (list, tuple)):
            raise TypeError(f"ep_metas must be a list or tuple, got {type(ep_metas).__name__}")
        if len(ep_metas) < n_envs:
            raise ValueError(
                f"ep_metas list has {len(ep_metas)} elements, but n_envs={n_envs} "
                f"requires at least {n_envs} elements."
            )

    suite_name = "robocasa"
    task_id = 0

    print(f"Creating RoboCasa envs | task={task_name} | n_envs(per task)={n_envs}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    fns = _make_env_fns(
        task_name=task_name,
        n_envs=n_envs,
        camera_names=parsed_camera_names,
        gym_kwargs=gym_kwargs,
        ep_metas=ep_metas,
    )
    out[suite_name][task_id] = env_cls(fns)
    print(f"Built vec env | suite={suite_name} | task_id={task_id} | n_envs={n_envs}")

    return {suite: dict(task_map) for suite, task_map in out.items()}