"""Gymnasium environment wrapping screen capture + input simulation."""

import time

import cv2
import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from sbg.actions import ActionSpace
from sbg.capture import ScreenCapture
from sbg.reward import RewardDetector


class SuperBattleGolfEnv(gym.Env):
    """Custom Gymnasium env for Super Battle Golf via screen capture.

    Observations: Stacked grayscale frames (frame_stack, H, W)
    Actions: Discrete game inputs (aim, power, shoot)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str = "configs/default.yaml"):
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        cap_cfg = self.config["capture"]
        prep_cfg = self.config["preprocessing"]
        reward_cfg = self.config["reward"]

        # Screen capture
        self.capture = ScreenCapture(
            monitor=cap_cfg["monitor"],
            region=cap_cfg.get("region"),
            fps=cap_cfg["fps"],
        )

        # Actions
        self.action_handler = ActionSpace(self.config["actions"])
        self.action_space = spaces.Discrete(self.action_handler.n)

        # Preprocessing
        self.img_size = tuple(prep_cfg["resize"])
        self.grayscale = prep_cfg["grayscale"]
        self.frame_stack_size = prep_cfg["frame_stack"]

        # Observation space: stacked frames
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_stack_size, *self.img_size),
            dtype=np.uint8,
        )

        # Reward
        self.reward_detector = RewardDetector(
            method=reward_cfg["method"],
            score_region=reward_cfg.get("score_region"),
        )

        # Episode tracking
        ep_cfg = self.config["episode"]
        self.max_steps = ep_cfg["max_steps"]
        self.step_delay = ep_cfg["step_delay"]
        self.current_step = 0
        self.frame_stack = None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert raw frame to model input: resize + grayscale."""
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.img_size)
        return frame

    def _get_obs(self) -> np.ndarray:
        """Capture frame, preprocess, and return stacked observation."""
        raw = self.capture.grab()
        processed = self._preprocess(raw)

        self.frame_stack = np.roll(self.frame_stack, shift=-1, axis=0)
        self.frame_stack[-1] = processed

        return self.frame_stack.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.frame_stack is None:
            self.capture.start()

        self.current_step = 0
        self.reward_detector.reset()

        # Initialize frame stack with current screen
        raw = self.capture.grab()
        processed = self._preprocess(raw)
        self.frame_stack = np.stack([processed] * self.frame_stack_size)

        return self.frame_stack.copy(), {}

    def step(self, action: int):
        self.action_handler.act(action)
        time.sleep(self.step_delay)

        obs = self._get_obs()
        raw_frame = self.capture.grab()

        reward = self.reward_detector.compute(raw_frame)
        terminated = self.reward_detector.detect_episode_end(raw_frame)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.capture.stop()
