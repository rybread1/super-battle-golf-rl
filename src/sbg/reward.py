"""Reward computation for frame-level RL.

Six clean signals:
1. Step penalty      — flat time cost, keeps agent moving
2. Ball nav          — reward/penalize based on whether action moves toward the ball icon
3. Shot bonus        — flat reward for successfully hitting the ball
4. Progress          — reward proportional to progress bar delta (walking toward hole)
5. Penalties         — OOB, stale, bad shot attempt
6. Hole completion   — big bonus minus per-stroke cost
"""

# --- Tunable constants ---
STEP_PENALTY = -0.01            # flat per-step cost
BALL_NAV_REWARD = 0.3           # correct action toward ball icon
BALL_NAV_PENALTY = -0.2         # wrong action away from ball icon
SHOT_BONUS = 10.0               # flat reward for a successful hit
BAD_SHOT_PENALTY = -0.5         # tried to shoot when not near ball
PROGRESS_SCALE = 20.0           # multiplier on progress bar delta
STALE_PROGRESS_PENALTY = -3.0   # no progress change for too long (stuck/spinning)
OOB_PENALTY = -5.0              # went out of bounds
HOLE_COMPLETE_BONUS = 25.0      # holed out
STROKE_PENALTY = -1.0           # per stroke taken


def compute_reward(
    prev_progress: float | None,
    new_progress: float | None,
    hole_complete: bool,
    strokes: int,
    shot_taken: bool = False,
    out_of_bounds: bool = False,
    ball_nav_score: float = 0.0,
) -> tuple[float, dict]:
    """Compute reward for a single env step.

    Args:
        prev_progress: Progress bar reading before this step (0-1).
        new_progress: Progress bar reading after this step (0-1).
        hole_complete: Whether the hole was completed this step.
        strokes: Total strokes taken this hole.
        shot_taken: Whether a successful shot was executed this step.
        out_of_bounds: Whether the player went out of bounds.
        ball_nav_score: How well the nav action matched the ball direction.
                        Positive = correct, negative = wrong, 0 = no info.

    Returns:
        (total_reward, breakdown_dict) where breakdown shows each component.
    """
    breakdown = {}

    # 1. Step penalty
    breakdown["step"] = STEP_PENALTY

    # 2. Ball navigation — reward correct action toward ball
    if ball_nav_score > 0:
        breakdown["ball_nav"] = ball_nav_score * BALL_NAV_REWARD
    elif ball_nav_score < 0:
        breakdown["ball_nav"] = -ball_nav_score * BALL_NAV_PENALTY  # penalty is negative

    # 3. Shot bonus
    if shot_taken:
        breakdown["shot"] = SHOT_BONUS

    # 4. Progress toward hole
    if prev_progress is not None and new_progress is not None:
        delta = new_progress - prev_progress
        if delta != 0:
            breakdown["progress"] = delta * PROGRESS_SCALE

    # 5. Penalties
    if out_of_bounds:
        breakdown["oob"] = OOB_PENALTY

    # 6. Hole completion
    if hole_complete:
        breakdown["hole"] = HOLE_COMPLETE_BONUS
        breakdown["strokes"] = strokes * STROKE_PENALTY

    total = sum(breakdown.values())
    return total, breakdown
