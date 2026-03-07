"""Reward computation for frame-level RL.

Rewards are given per-step for navigation efficiency,
per-shot for reaching the ball and making progress toward the pin.
"""

# Navigation
STEP_PENALTY = -0.01         # small cost per navigation step
FAILED_SHOT_PENALTY = -0.05  # tried to shoot but wasn't near ball

# Shot
SUCCESSFUL_SHOT_BONUS = 1.0  # reached the ball and took a swing
PROGRESS_SCALE = 5.0         # multiplier on progress bar delta

# Hole completion
HOLE_BONUS = 10.0


def compute_step_reward() -> float:
    """Reward for a single navigation step (forward/turn)."""
    return STEP_PENALTY


def compute_failed_shot_reward() -> float:
    """Reward when agent tries to shoot but can't enter stance."""
    return FAILED_SHOT_PENALTY


def compute_shot_reward(
    prev_progress: float | None,
    new_progress: float | None,
    hole_complete: bool = False,
) -> float:
    """Compute reward for a successful shot.

    prev_progress/new_progress come from the progress bar (character
    distance to pin). Progress is read right before each shot, so the
    delta reflects how much closer the previous shot moved the ball.

    Args:
        prev_progress: Progress bar reading before previous shot (0.0-1.0).
        new_progress: Progress bar reading before this shot.
        hole_complete: Whether the ball went in the hole.
    """
    reward = SUCCESSFUL_SHOT_BONUS

    if prev_progress is not None and new_progress is not None:
        delta = new_progress - prev_progress
        reward += delta * PROGRESS_SCALE

    if hole_complete:
        reward += HOLE_BONUS

    return reward
