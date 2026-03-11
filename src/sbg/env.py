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

from sbg.game.actions import (
    navigate, release_all_keys, enter_stance, exit_stance, aim, set_angle,
    charge_and_shoot, restart_hole, reset_camera_pitch,
)
from sbg.game.capture import ScreenCapture
from sbg.game.navigate import navigate_to_match, wait_for_next_hole
from sbg.game.window import setup_game_window
from sbg.vision.detect import (
    detect_player_state, is_loading_screen, detect_scoreboard,
    get_player_progress, is_out_of_bounds, find_ball_icon,
)
from sbg.reward import compute_reward, STALE_PROGRESS_PENALTY, BAD_SHOT_PENALTY

class SuperBattleGolfEnv(gym.Env):
    """Frame-level RL environment for Super Battle Golf.

    Observation: 84x84x3 RGB game frame.
    Action: MultiDiscrete([2, 2, 3, 9, 4, 10])
        - action_type: 0=navigate, 1=attempt shot
        - move_dir: 0=W, 1=S (navigate only)
        - turn: 0=left, 1=none, 2=right (mouse turn, navigate only)
        - aim_x: 0-8 (4=center, <4=left, >4=right) (shot only)
        - shot_angle: 0-3 (shot only)
        - power_level: 0-9 (shot only)

    One step = one atomic action. Navigation steps are fast (~0.4s).
    Shot steps are slow (~3-10s for animation).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        obs_size: tuple[int, int] = (84, 84),
        max_steps_per_hole: int = 500,
        max_holes: int = 9,
        auto_launch: bool = True,
        skip_navigate: bool = False,
        steps_to_first_hit: int = 50,
        steps_between_hits: int = 250,
        max_stale_steps: int = 15,
        reset_countdown: bool = True,
    ):
        super().__init__()

        self.obs_size = obs_size
        self.max_steps_per_hole = max_steps_per_hole
        self.max_holes = max_holes
        self.auto_launch = auto_launch
        self.skip_navigate = skip_navigate
        self.steps_to_first_hit = steps_to_first_hit
        self.steps_between_hits = steps_between_hits
        self.max_stale_steps = max_stale_steps
        self.reset_countdown = reset_countdown

        # Action space: action_type(2), move_dir(2), turn(3), aim_x(9), angle(4), power(10)
        self.action_space = spaces.MultiDiscrete([2, 2, 3, 9, 4, 10])

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
        self.prev_state = "none"
        self.hole_steps = 0
        self.hole_strokes = 0
        self.holes_played = 0
        self.steps_since_last_hit = 0
        self._shot_taken = False
        self.last_reward_breakdown = {}
        self.last_ball_pos = None
        self.last_ball_status = None
        self._ball_x_frac = None
        self._ball_y_frac = None
        self._initialized = False
        self._first_reset = True
        self.steps_since_progress_change = 0

    def _get_frame(self) -> np.ndarray | None:
        return self.capture.grab()

    def _get_obs(self) -> np.ndarray:
        frame = self._get_frame()
        if frame is None:
            return np.zeros((*self.obs_size, 3), dtype=np.uint8)
        return cv2.resize(frame, self.obs_size)

    def _reset_hole_state(self):
        """Reset all hole tracking to fresh-start state."""
        release_all_keys()
        reset_camera_pitch()
        frame = self._get_frame()
        self.prev_progress = get_player_progress(frame) if frame is not None else None
        self.prev_state = detect_player_state(frame) if frame is not None else "none"
        self.hole_steps = 0
        self.hole_strokes = 0
        self.steps_since_last_hit = 0
        self._shot_taken = False
        self.steps_since_progress_change = 0

    def _wait_for_respawn(self, timeout: float = 15.0):
        """Wait for OOB respawn animation to finish (~13s)."""
        print("  [OOB] Waiting for respawn...")
        start = time.time()
        # First wait until banner disappears
        while time.time() - start < timeout:
            frame = self._get_frame()
            if frame is not None and not is_out_of_bounds(frame):
                # Banner gone — wait for respawn animation to finish
                time.sleep(8.0)
                return
            time.sleep(0.5)
        print("  [OOB] Respawn timeout")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._initialized:
            self.hwnd, self.region = setup_game_window(launch=self.auto_launch)
            self.capture = ScreenCapture(region=self.region, fps=30)
            self.capture.start()
            time.sleep(1)

            if not self.skip_navigate:
                print("Navigating to match...")
                time.sleep(3)
                navigate_to_match(self.hwnd, self.region, self.capture)
            else:
                print("Skipping menu navigation (already in game)")
            self._initialized = True
        elif self.holes_played >= self.max_holes:
            # All 9 holes done — no restart logic yet, just keep counter going
            print(f"Match complete ({self.holes_played} holes). "
                  "Game restart not implemented yet.")

        # Restart the hole to reset player position to the tee
        if self._initialized and not self._first_reset:
            release_all_keys()
            print("[Reset] Restarting hole (hold R)...")
            restart_hole()
        self._first_reset = False

        time.sleep(2.0)

        if self.reset_countdown:
            for i in range(3, 0, -1):
                print(f"  [{i}...]")
                time.sleep(1.0)

        self._reset_hole_state()

        print(f"[Reset] Hole {self.holes_played + 1}, progress={self.prev_progress}")

        obs = self._get_obs()
        return obs, {"holes_played": self.holes_played}

    def _compute_step_reward(self, frame, hole_complete=False, out_of_bounds=False,
                             nav_action=None):
        """Detect state, compute reward from transition, update tracking.

        nav_action: (move_dir, turn) tuple for the navigation action taken,
                    or None if this wasn't a navigation step.
                    move_dir: 0=W (forward), 1=S (backward)
                    turn: 0=left, 1=none, 2=right
        """
        new_state = detect_player_state(frame) if frame is not None else "none"
        new_progress = get_player_progress(frame) if frame is not None else None

        # Ball icon position check (only during navigation, every 3rd step)
        # Last known position carries forward until next detection or state change
        if new_state != "none":
            self.last_ball_pos = None
            self._ball_x_frac = None
            self._ball_y_frac = None
        elif frame is not None and self.hole_steps % 3 == 0:
            ball_pos = find_ball_icon(frame)
            if ball_pos is not None:
                h, w = frame.shape[:2]
                self._ball_x_frac = ball_pos[0] / w
                self._ball_y_frac = ball_pos[1] / h
                self.last_ball_pos = ball_pos
            # If detection fails, keep last known position

        # Compute ball nav score: how well does the action match the ball direction?
        # Score is -1 (wrong) to +1 (correct), 0 = no info
        ball_nav_score = 0.0
        if nav_action is not None and self._ball_x_frac is not None:
            move_dir, turn = nav_action
            x_frac = self._ball_x_frac  # 0=left, 0.5=center, 1=right
            y_frac = self._ball_y_frac  # 0=top, 1=bottom

            # Deadzone: X 0.35-0.65, Y 0.55-0.78 (centered on player character)
            # Score scales linearly from 0 at deadzone edge to 1.0 at screen edge

            # Movement score: W correct when ball is ahead (top), S when behind (bottom)
            move_score = 0.0
            if y_frac < 0.55:       # ball is ahead (above deadzone)
                magnitude = (0.55 - y_frac) / 0.55  # 0.0 at edge, 1.0 at top
                move_score = magnitude if move_dir == 0 else -magnitude
            elif y_frac > 0.78:     # ball is behind (below deadzone)
                magnitude = (y_frac - 0.78) / (1.0 - 0.78)  # 0.0 at edge, 1.0 at bottom
                move_score = magnitude if move_dir == 1 else -magnitude

            # Turn score: turn toward the ball's horizontal position
            turn_score = 0.0
            if x_frac < 0.35:       # ball is to the left
                magnitude = (0.35 - x_frac) / 0.35  # 0.0 at edge, 1.0 at left
                turn_score = magnitude if turn == 0 else (-magnitude if turn == 2 else 0.0)
            elif x_frac > 0.65:     # ball is to the right
                magnitude = (x_frac - 0.65) / (1.0 - 0.65)  # 0.0 at edge, 1.0 at right
                turn_score = magnitude if turn == 2 else (-magnitude if turn == 0 else 0.0)

            # Combine: average of both scores (each -1 to +1)
            ball_nav_score = (move_score + turn_score) / 2.0

            # Update overlay status
            if move_score > 0 and turn_score >= 0:
                self.last_ball_status = "ahead"
            elif move_score < 0:
                self.last_ball_status = "behind"
            elif turn_score != 0:
                self.last_ball_status = "offcenter"
            else:
                self.last_ball_status = "ahead"
        elif self._ball_x_frac is None:
            self.last_ball_status = None

        reward, breakdown = compute_reward(
            prev_progress=self.prev_progress,
            new_progress=new_progress,
            hole_complete=hole_complete,
            strokes=self.hole_strokes,
            shot_taken=self._shot_taken,
            out_of_bounds=out_of_bounds,
            ball_nav_score=ball_nav_score,
        )

        if self._shot_taken:
            self._shot_taken = False

        self.last_reward_breakdown = breakdown
        self.last_progress_debug = {
            "prev": self.prev_progress,
            "new": new_progress,
            "delta": (new_progress - self.prev_progress)
                     if (new_progress is not None and self.prev_progress is not None)
                     else None,
        }

        # Track stale progress (no change = stuck)
        if (self.prev_progress is not None and new_progress is not None
                and abs(new_progress - self.prev_progress) > 0.001):
            self.steps_since_progress_change = 0
        else:
            self.steps_since_progress_change += 1

        self.prev_state = new_state
        if new_progress is not None:
            self.prev_progress = new_progress

        return reward

    def _hit_deadline_exceeded(self) -> bool:
        """Check if we've gone too long without hitting the ball."""
        limit = (self.steps_to_first_hit if self.hole_strokes == 0
                 else self.steps_between_hits)
        return self.steps_since_last_hit >= limit

    def _progress_stale(self) -> bool:
        """Check if progress hasn't changed for too long."""
        return self.steps_since_progress_change >= self.max_stale_steps

    def step(self, action):
        action_type, move_dir, turn, aim_x, angle_idx, power_level = action
        self.hole_steps += 1
        self.steps_since_last_hit += 1
        self.last_action_state = self.prev_state  # state when action was chosen

        # Penalize shot attempts when not near the ball, then convert to walk forward
        bad_shot = (action_type == 1 and
                    self.prev_state not in ("near_ball", "stance_no_hit", "stance_can_hit"))
        if bad_shot:
            action_type = 0
            move_dir = 0  # walk forward (W)
            turn = 1      # no turn

        if action_type == 0:  # Navigate
            navigate(int(move_dir), int(turn))

            frame = self._get_frame()

            # Check for OOB (can happen mid-flight after a shot)
            oob = frame is not None and is_out_of_bounds(frame)
            if oob:
                release_all_keys()
                self._wait_for_respawn()
                self._reset_hole_state()
                frame = self._get_frame()

            # Check for hole complete (ball may have landed in the hole)
            hole_complete = not oob and frame is not None and (
                detect_scoreboard(frame) or is_loading_screen(frame)
            )

            reward = self._compute_step_reward(frame, hole_complete=hole_complete,
                                               out_of_bounds=oob,
                                               nav_action=(int(move_dir), int(turn)))
            if bad_shot:
                reward += BAD_SHOT_PENALTY
                self.last_reward_breakdown["bad_shot"] = BAD_SHOT_PENALTY
            obs = self._get_obs()

            terminated = hole_complete
            if terminated:
                release_all_keys()
                self.holes_played += 1
                if self.holes_played < self.max_holes:
                    wait_for_next_hole(self.capture)

            stale = self._progress_stale()
            truncated = not terminated and (self.hole_steps >= self.max_steps_per_hole or
                                            self._hit_deadline_exceeded() or
                                            stale)
            if stale and truncated:
                reward += STALE_PROGRESS_PENALTY
                self.last_reward_breakdown["stale"] = STALE_PROGRESS_PENALTY
            return obs, reward, terminated, truncated, self._info(
                hole_complete=hole_complete, out_of_bounds=oob)

        else:  # Attempt shot (only reached when near ball)
            return self._do_shot(int(aim_x), int(angle_idx), int(power_level))

    def _do_shot(self, aim_x: int, angle_idx: int, power_level: int):
        """Attempt to enter stance and take a shot.

        Only called when prev_state indicates we're near the ball.
        """
        release_all_keys()
        enter_stance()
        time.sleep(0.2)

        frame = self._get_frame()
        state = detect_player_state(frame) if frame is not None else "none"
        self.prev_state = state  # update so overlay can see it

        if state not in ("stance_no_hit", "stance_can_hit", "swinging"):
            # Detection disagreed — bail out, normal step cost
            exit_stance()
            reward = self._compute_step_reward(frame)
            obs = self._get_obs()
            truncated = (self.hole_steps >= self.max_steps_per_hole or
                         self._hit_deadline_exceeded() or self._progress_stale())
            return obs, reward, False, truncated, self._info()

        if state == "stance_no_hit":
            # In stance but too far — bail out, normal step cost
            exit_stance()
            reward = self._compute_step_reward(frame)
            obs = self._get_obs()
            truncated = (self.hole_steps >= self.max_steps_per_hole or
                         self._hit_deadline_exceeded() or self._progress_stale())
            return obs, reward, False, truncated, self._info()

        # In stance with can_hit — execute the shot
        self._shot_taken = True
        aim(aim_x)
        set_angle(angle_idx)
        time.sleep(0.1)
        charge_and_shoot(power_level)
        exit_stance()
        time.sleep(0.4)
        self.hole_strokes += 1
        self.steps_since_last_hit = 0

        frame = self._get_frame()
        reward = self._compute_step_reward(frame)
        obs = self._get_obs()
        truncated = (self.hole_steps >= self.max_steps_per_hole or
                     self._hit_deadline_exceeded())
        return obs, reward, False, truncated, self._info()

    def _info(self, hole_complete=False, out_of_bounds=False) -> dict:
        return {
            "hole_steps": self.hole_steps,
            "hole_strokes": self.hole_strokes,
            "holes_played": self.holes_played,
            "hole_complete": hole_complete,
            "out_of_bounds": out_of_bounds,
            "progress": self.prev_progress,
        }

    def close(self):
        release_all_keys()
        if self.capture:
            self.capture.stop()
