"""Training entrypoint for the frame-level RL agent.

Includes a live debug overlay window that shows game state, reward
signals, and training stats while training runs.
"""

import argparse
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from sbg.env import SuperBattleGolfEnv
from sbg.reward import (
    STEP_PENALTY, BALL_NAV_REWARD, BALL_NAV_PENALTY,
    SHOT_BONUS, BAD_SHOT_PENALTY, PROGRESS_SCALE, OOB_PENALTY, HOLE_COMPLETE_BONUS,
    STROKE_PENALTY,
)

CHECKPOINT_DIR = "checkpoints"

# Colors (BGR for OpenCV)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
DARK_BG = (30, 30, 30)

MOVE_LABELS = ["W", "S"]
TURN_LABELS = ["<", "-", ">"]


def draw_panel(img, x, y, width, lines):
    """Draw a panel with multiple lines of colored text. Returns new y."""
    panel_h = len(lines) * 18 + 10
    cv2.rectangle(img, (x - 4, y - 4), (x + width, y + panel_h), DARK_BG, -1)
    cv2.rectangle(img, (x - 4, y - 4), (x + width, y + panel_h), (80, 80, 80), 1)
    cy = y + 14
    for text, color in lines:
        cv2.putText(img, text, (x, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cy += 18
    return y + panel_h + 8


class OverlayThread:
    """Background thread that continuously displays the game with stats overlay."""

    def __init__(self, env: SuperBattleGolfEnv, target_fps: int = 15):
        self.env = env
        self.target_fps = target_fps
        self._stop_event = threading.Event()
        self._quit_pressed = False
        self._thread = None
        self._positioned = False

        # Shared stats (updated from callback on main thread)
        self.num_timesteps = 0
        self.episodes_completed = 0
        self.episode_reward = 0.0
        self.avg_reward = 0.0
        self.best_reward = float("-inf")
        self.action_history = []  # list of (action_str, reward, breakdown) tuples
        self.max_history = 30

    @property
    def quit_requested(self) -> bool:
        return self._quit_pressed

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()

    def _run(self):
        interval = 1.0 / self.target_fps
        while not self._stop_event.is_set():
            start = time.time()
            self._draw()
            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _draw(self):
        if not self.env.capture:
            return
        frame = self.env.capture.grab()
        if frame is None:
            return

        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = display.shape[:2]

        # --- Ball icon marker ---
        ball_pos = getattr(self.env, "last_ball_pos", None)
        ball_status = getattr(self.env, "last_ball_status", None)
        if ball_pos is not None:
            bx, by = ball_pos
            marker_color = GREEN if ball_status == "ahead" else RED if ball_status == "behind" else ORANGE
            cv2.drawMarker(display, (bx, by), marker_color, cv2.MARKER_CROSS, 30, 2)
            cv2.circle(display, (bx, by), 20, marker_color, 1)

        # --- Training stats panel (top-left) ---
        train_lines = [
            ("=== TRAINING ===", YELLOW),
            (f"Timestep: {self.num_timesteps}", WHITE),
            (f"Episodes: {self.episodes_completed}", WHITE),
            (f"Ep reward: {self.episode_reward:+.2f}", WHITE),
            (f"Avg reward (20): {self.avg_reward:+.2f}", CYAN),
            (f"Best reward: {self.best_reward:+.2f}", GREEN),
        ]
        draw_panel(display, 10, 10, 250, train_lines)

        # --- Game state panel (top-right) ---
        state_colors = {
            "none": WHITE,
            "near_ball": YELLOW,
            "stance_no_hit": ORANGE,
            "stance_can_hit": GREEN,
            "swinging": CYAN,
        }
        state_x = w - 280

        # Ball icon status
        if ball_pos is not None:
            if ball_status == "ahead":
                ball_label, ball_color = "AHEAD", GREEN
            elif ball_status == "behind":
                ball_label, ball_color = "BEHIND", RED
            else:
                ball_label, ball_color = "OFF-CENTER", ORANGE
        else:
            ball_label, ball_color = "not found", WHITE

        state_lines = [
            ("=== GAME STATE ===", YELLOW),
            (f"State: {self.env.prev_state}",
             state_colors.get(self.env.prev_state, WHITE)),
            (f"Progress: {self.env.prev_progress:.3f}" if self.env.prev_progress is not None
             else "Progress: N/A", CYAN),
            (f"Ball icon: {ball_label}", ball_color),
            (f"Hole: {self.env.holes_played + 1}/{self.env.max_holes}", WHITE),
            (f"Steps: {self.env.hole_steps}/{self.env.max_steps_per_hole}", WHITE),
            (f"Strokes: {self.env.hole_strokes}  |  Since hit: {self.env.steps_since_last_hit}", WHITE),
        ]

        # Progress detection diagnostics
        prog_dbg = getattr(self.env, "last_progress_debug", {})
        if prog_dbg:
            p_delta = prog_dbg.get("delta")
            delta_str = f"{p_delta:+.4f}" if p_delta is not None else "N/A"
            delta_color = GREEN if (p_delta and p_delta > 0) else RED if (p_delta and p_delta < 0) else WHITE
            state_lines.append((f"Progress delta: {delta_str}", delta_color))
        draw_panel(display, state_x, 10, 270, state_lines)

        # --- Reward constants panel (bottom-left, above action history) ---
        reward_lines = [
            ("=== REWARDS ===", YELLOW),
            (f"Step:         {STEP_PENALTY:+.3f}", WHITE),
            (f"Ball nav OK:  {BALL_NAV_REWARD:+.2f}", GREEN),
            (f"Ball nav bad: {BALL_NAV_PENALTY:+.2f}", RED),
            (f"Shot:         {SHOT_BONUS:+.1f}", GREEN),
            (f"Bad shot:     {BAD_SHOT_PENALTY:+.1f}", RED),
            (f"Progress:     {PROGRESS_SCALE:.0f}x delta", WHITE),
            (f"OOB:          {OOB_PENALTY:+.1f}", RED),
            (f"Hole:         {HOLE_COMPLETE_BONUS:+.1f}", WHITE),
            (f"Per stroke:   {STROKE_PENALTY:+.1f}", WHITE),
        ]
        draw_panel(display, w - 280, h - len(reward_lines) * 18 - 18, 270, reward_lines)

        # --- Action history panel (left side, below training panel) ---
        history = self.action_history
        if history:
            line_h = 15
            header_h = 20
            panel_w = 450
            # Calculate how many entries fit between training panel bottom and screen bottom
            panel_top = 140  # below training stats panel
            available_h = h - panel_top - 8
            max_visible = max(1, (available_h - header_h - 10) // line_h)
            visible = history[-max_visible:]

            panel_h = len(visible) * line_h + header_h + 10
            px, py = 10, panel_top
            cv2.rectangle(display, (px - 4, py - 4), (px + panel_w, py + panel_h), DARK_BG, -1)
            cv2.rectangle(display, (px - 4, py - 4), (px + panel_w, py + panel_h), (80, 80, 80), 1)

            cv2.putText(display, "=== ACTION HISTORY ===", (px, py + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1, cv2.LINE_AA)
            cy = py + header_h + 8

            # State abbreviations for compact display
            STATE_SHORT = {"none": "---", "near_ball": "NBL", "stance_no_hit": "SNH",
                           "stance_can_hit": "SCH", "swinging": "SWG"}

            for entry in visible:
                action_str, reward, breakdown = entry[0], entry[1], entry[2]
                prog_delta = entry[3] if len(entry) > 3 else None
                player_state = entry[4] if len(entry) > 4 else "?"
                state_short = STATE_SHORT.get(player_state, player_state[:3])
                reward_color = GREEN if reward > 0 else RED if reward < 0 else WHITE
                # Progress column: always shown so you can track it
                if prog_delta is not None:
                    prog_str = f"{prog_delta:+.4f}"
                else:
                    prog_str = "  N/A "
                # Compact breakdown (non-step, non-progress since progress has its own column)
                parts = [f"{k}={v:+.2f}" for k, v in breakdown.items()
                         if k not in ("step", "progress") and v != 0]
                parts_str = " ".join(parts)
                line = f"{state_short} {action_str:<16} {reward:>+6.2f} p:{prog_str}  {parts_str}"
                cv2.putText(display, line, (px, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, reward_color, 1, cv2.LINE_AA)
                cy += line_h

        cv2.imshow("SBG Training", display)
        if not self._positioned:
            cv2.moveWindow("SBG Training", 1300, 0)
            self._positioned = True
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._quit_pressed = True


class OverlayCallback(BaseCallback):
    """Training callback that feeds stats to the overlay thread."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episodes_completed = 0
        self.best_episode_reward = float("-inf")
        self.recent_episode_rewards = []
        self._start_time = None
        self._overlay: OverlayThread | None = None
        # Action tracking for current episode
        self._ep_nav_count = 0
        self._ep_shot_count = 0
        self._ep_shot_fail_count = 0
        self._ep_oob_count = 0
        # Cumulative stats
        self._total_nav = 0
        self._total_shot = 0
        self._total_shot_fail = 0
        self._total_oob = 0
        # Per-episode reward breakdown
        self._ep_reward_parts = {}

    def _get_env(self) -> SuperBattleGolfEnv:
        return self.training_env.envs[0]

    def _on_training_start(self):
        self._start_time = time.time()
        self._overlay = OverlayThread(self._get_env())
        self._overlay.start()

    def _on_step(self) -> bool:
        env = self._get_env()

        action = self.locals.get("actions", [None])[0]
        reward = self.locals.get("rewards", [0.0])[0]
        done = self.locals.get("dones", [False])[0]
        info = self.locals.get("infos", [{}])[0]

        self.episode_reward += float(reward)
        self.episode_steps += 1

        # Track action distribution
        if action is not None:
            if action[0] == 0:
                self._ep_nav_count += 1
                self._total_nav += 1
            else:
                self._ep_shot_count += 1
                self._total_shot += 1
            if info.get("out_of_bounds"):
                self._ep_oob_count += 1
                self._total_oob += 1

        # Track reward components
        breakdown = getattr(env, "last_reward_breakdown", {})
        for key, val in breakdown.items():
            self._ep_reward_parts[key] = self._ep_reward_parts.get(key, 0.0) + val

        if done:
            self.recent_episode_rewards.append(self.episode_reward)
            if len(self.recent_episode_rewards) > 20:
                self.recent_episode_rewards.pop(0)
            if self.episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.episode_reward
            self.episodes_completed += 1

            elapsed = time.time() - self._start_time if self._start_time else 0
            avg_reward = np.mean(self.recent_episode_rewards) if self.recent_episode_rewards else 0.0
            hole_complete = info.get("hole_complete", False)
            strokes = info.get("hole_strokes", env.hole_strokes)
            outcome = "HOLED" if hole_complete else "TRUNC"

            progress = info.get("progress")
            prog_str = f"{progress:.1%}" if progress is not None else "N/A"

            # Episode summary line
            print(f"\n{'=' * 72}")
            print(f"  Episode {self.episodes_completed:>3}  |  {outcome}  |  "
                  f"{elapsed // 60:.0f}m{elapsed % 60:02.0f}s elapsed  |  "
                  f"timestep {self.num_timesteps}")
            print(f"  {'─' * 68}")

            # Performance
            print(f"  Steps: {self.episode_steps:>4}  |  "
                  f"Strokes: {strokes}  |  "
                  f"Progress: {prog_str}  |  "
                  f"OOB: {self._ep_oob_count}")

            # Actions
            ball_approach_total = self._ep_reward_parts.get("ball_approach", 0.0)
            shot_total = self._ep_reward_parts.get("shot", 0.0)
            print(f"  Actions: {self._ep_nav_count} nav, "
                  f"{self._ep_shot_count} shots  |  "
                  f"Shot: {shot_total:+.1f}  Ball approach: {ball_approach_total:+.1f}")

            # Reward breakdown
            print(f"  Reward: {self.episode_reward:>+8.2f}  "
                  f"(avg20: {avg_reward:>+7.2f}  best: {self.best_episode_reward:>+7.2f})")
            parts = sorted(self._ep_reward_parts.items(), key=lambda x: -abs(x[1]))
            parts_str = "  "
            for key, val in parts:
                if val != 0:
                    parts_str += f"  {key}={val:+.2f}"
            if parts_str.strip():
                print(parts_str)
            print(f"{'=' * 72}")

            # Reset episode counters
            self.episode_reward = 0.0
            self.episode_steps = 0
            self._ep_nav_count = 0
            self._ep_shot_count = 0
            self._ep_shot_fail_count = 0
            self._ep_oob_count = 0
            self._ep_reward_parts = {}
            if self._overlay:
                self._overlay.action_history.clear()

        # Push stats to overlay thread
        if self._overlay:
            avg = np.mean(self.recent_episode_rewards) if self.recent_episode_rewards else 0.0
            self._overlay.num_timesteps = self.num_timesteps
            self._overlay.episodes_completed = self.episodes_completed
            self._overlay.episode_reward = self.episode_reward
            self._overlay.avg_reward = avg
            self._overlay.best_reward = self.best_episode_reward

            # Build action string and push to history
            breakdown = dict(getattr(env, "last_reward_breakdown", {}))
            prog_dbg = getattr(env, "last_progress_debug", {})
            prog_delta = prog_dbg.get("delta")
            player_state = getattr(env, "last_action_state", getattr(env, "prev_state", "?"))
            if action is not None:
                if action[0] == 0:
                    move = MOVE_LABELS[action[1]] if action[1] < len(MOVE_LABELS) else "?"
                    turn = TURN_LABELS[action[2]] if action[2] < len(TURN_LABELS) else "?"
                    action_str = f"NAV {move} {turn}"
                else:
                    action_str = f"SHOT a={action[3]} n={action[4]} p={action[5]}"
                self._overlay.action_history.append(
                    (action_str, float(reward), breakdown, prog_delta, player_state))
                if len(self._overlay.action_history) > self._overlay.max_history:
                    self._overlay.action_history.pop(0)

            if self._overlay.quit_requested:
                print("\n[Overlay] Training stopped by user (q pressed)")
                return False

        return True

    def _on_training_end(self):
        if self._overlay:
            self._overlay.stop()


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Super Battle Golf")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-launch", action="store_true", help="Don't auto-launch game")
    parser.add_argument("--skip-nav", action="store_true", help="Skip menu navigation (already in a match)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per hole episode")
    parser.add_argument("--first-hit", type=int, default=50, help="Steps allowed before first hit")
    parser.add_argument("--between-hits", type=int, default=250, help="Steps allowed between hits")
    parser.add_argument("--save-freq", type=int, default=1000, help="Checkpoint save frequency")
    parser.add_argument("--no-overlay", action="store_true", help="Disable debug overlay")
    parser.add_argument("--name", default=None, help="Custom run name (appended to timestamp)")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        run_id = f"{run_id}_{args.name}"
    run_dir = os.path.join(CHECKPOINT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    env = DummyVecEnv([lambda: SuperBattleGolfEnv(
        auto_launch=not args.no_launch,
        skip_navigate=args.skip_nav,
        max_steps_per_hole=args.max_steps,
        steps_to_first_hit=args.first_hit,
        steps_between_hits=args.between_hits,
    )])

    if args.resume:
        model = PPO.load(args.resume, env=env)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./runs",
        )

    callbacks = [
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=run_dir,
            name_prefix="ppo_sbg",
        ),
    ]
    if not args.no_overlay:
        callbacks.append(OverlayCallback())

    print(f"Run: {run_id}")
    print(f"Checkpoints: {run_dir}")
    print(f"Training for {args.timesteps} timesteps...")
    print("Press 'q' in the overlay window to stop early.")

    save_path = os.path.join(run_dir, "ppo_sbg_final")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving model before exit...")
    finally:
        model.save(save_path)
        print(f"Model saved to {save_path}")
        env.close()


if __name__ == "__main__":
    main()
