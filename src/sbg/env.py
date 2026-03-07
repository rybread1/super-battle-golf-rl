"""Frame-level Gymnasium environment for Super Battle Golf.

The agent controls both navigation (walking to the ball) and shot
execution (aim, angle, power) through a unified action space.

Each env.step() is one atomic action — either a movement or a shot attempt.
"""

import time

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sbg.actions import (
    walk_forward, walk_forward_turn_left, walk_forward_turn_right,
    enter_stance, aim, set_angle, charge_and_shoot,
)
from sbg.capture import ScreenCapture
from sbg.detect import is_in_stance, is_loading_screen, get_player_progress
from sbg.navigate import navigate_to_match, wait_for_next_hole
from sbg.reward import (
    compute_step_reward,
    compute_failed_shot_reward,
    compute_shot_reward,
)
from sbg.window import setup_game_window

# Move types
MOVE_FORWARD = 0
MOVE_FORWARD_LEFT = 1
MOVE_FORWARD_RIGHT = 2
ATTEMPT_SHOT = 3


class SuperBattleGolfEnv(gym.Env):
    """Frame-level RL environment for Super Battle Golf.

    Observation: 84x84x3 RGB game frame.
    Action: MultiDiscrete([4, 16, 4, 10])
        - move_type: 0=forward, 1=forward+turn left, 2=forward+turn right, 3=attempt shot
        - aim_direction: 0-15 (8=straight) — only used when move_type=3
        - shot_angle: 0-3 — only used when move_type=3
        - power_level: 0-9 — only used when move_type=3

    One step = one atomic action. Navigation steps are fast (~0.5s).
    Shot steps are slow (~3-10s for animation).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        obs_size: tuple[int, int] = (84, 84),
        max_steps_per_hole: int = 500,
        max_holes: int = 4,
        auto_launch: bool = True,
    ):
        super().__init__()

        self.obs_size = obs_size
        self.max_steps_per_hole = max_steps_per_hole
        self.max_holes = max_holes
        self.auto_launch = auto_launch

        # Action space: move_type (4), aim (16), angle (4), power (10)
        self.action_space = spaces.MultiDiscrete([4, 16, 4, 10])

        # Observation: RGB frame resized
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(obs_size[0], obs_size[1], 3),
            dtype=np.uint8,
        )

        # State
        self.hwnd = None
        self.region = None
        self.capture = None
        self.prev_progress = None
        self.hole_steps = 0
        self.hole_strokes = 0
        self.holes_played = 0
        self._initialized = False

    def _get_frame(self) -> np.ndarray | None:
        return self.capture.grab()

    def _get_obs(self) -> np.ndarray:
        frame = self._get_frame()
        if frame is None:
            return np.zeros((*self.obs_size, 3), dtype=np.uint8)
        return cv2.resize(frame, self.obs_size)

    def _wait_for_ball_to_land(self, timeout: float = 10.0):
        """Wait for the ball to land after a shot."""
        time.sleep(2.0)
        start = time.time()
        stable_frames = 0
        while time.time() - start < timeout:
            frame = self._get_frame()
            if frame is not None and not is_loading_screen(frame):
                stable_frames += 1
                if stable_frames >= 3:
                    return
            else:
                stable_frames = 0
            time.sleep(0.3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._initialized:
            self.hwnd, self.region = setup_game_window(launch=self.auto_launch)
            self.capture = ScreenCapture(region=self.region, fps=30)
            self.capture.start()
            time.sleep(1)

            print("Navigating to match...")
            time.sleep(3)
            navigate_to_match(self.hwnd, self.region, self.capture)
            self._initialized = True
        else:
            if self.holes_played >= self.max_holes:
                print("Match complete — need to start new match")
                self.holes_played = 0

        time.sleep(1)
        frame = self._get_frame()
        self.prev_progress = get_player_progress(frame) if frame is not None else None
        self.hole_steps = 0
        self.hole_strokes = 0

        obs = self._get_obs()
        return obs, {"holes_played": self.holes_played}

    def step(self, action):
        move_type, aim_dir, angle_idx, power_level = action
        self.hole_steps += 1

        if move_type == MOVE_FORWARD:
            walk_forward()
            reward = compute_step_reward()
            obs = self._get_obs()
            truncated = self.hole_steps >= self.max_steps_per_hole
            return obs, reward, False, truncated, self._info()

        elif move_type == MOVE_FORWARD_LEFT:
            walk_forward_turn_left()
            reward = compute_step_reward()
            obs = self._get_obs()
            truncated = self.hole_steps >= self.max_steps_per_hole
            return obs, reward, False, truncated, self._info()

        elif move_type == MOVE_FORWARD_RIGHT:
            walk_forward_turn_right()
            reward = compute_step_reward()
            obs = self._get_obs()
            truncated = self.hole_steps >= self.max_steps_per_hole
            return obs, reward, False, truncated, self._info()

        else:  # ATTEMPT_SHOT
            return self._do_shot(int(aim_dir), int(angle_idx), int(power_level))

    def _do_shot(self, aim_dir: int, angle_idx: int, power_level: int):
        """Attempt to enter stance and take a shot."""
        # Try entering stance
        enter_stance()
        time.sleep(0.3)

        frame = self._get_frame()
        if frame is None or not is_in_stance(frame):
            # Failed — not close enough to ball
            reward = compute_failed_shot_reward()
            obs = self._get_obs()
            truncated = self.hole_steps >= self.max_steps_per_hole
            return obs, reward, False, truncated, self._info(shot_failed=True)

        # In stance — read progress bar NOW (after walking to ball).
        # This reflects where the previous shot landed: if the last shot
        # moved the ball closer to the hole, we walked less and progress
        # is higher than prev_progress.
        current_progress = get_player_progress(frame)

        # Execute the shot
        aim(aim_dir)
        set_angle(angle_idx)
        time.sleep(0.2)
        charge_and_shoot(power_level)
        self.hole_strokes += 1

        # Wait for ball to land
        self._wait_for_ball_to_land()

        # Check if hole is complete
        frame = self._get_frame()
        hole_complete = frame is not None and is_loading_screen(frame)

        # Reward based on progress delta (prev shot position → current position)
        reward = compute_shot_reward(self.prev_progress, current_progress, hole_complete)
        self.prev_progress = current_progress

        terminated = hole_complete
        truncated = self.hole_steps >= self.max_steps_per_hole

        if terminated:
            self.holes_played += 1
            if self.holes_played < self.max_holes:
                wait_for_next_hole(self.capture)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, self._info(
            hole_complete=hole_complete,
        )

    def _info(self, hole_complete=False, shot_failed=False) -> dict:
        return {
            "hole_steps": self.hole_steps,
            "hole_strokes": self.hole_strokes,
            "holes_played": self.holes_played,
            "hole_complete": hole_complete,
            "shot_failed": shot_failed,
            "progress": self.prev_progress,
        }

    def close(self):
        if self.capture:
            self.capture.stop()
