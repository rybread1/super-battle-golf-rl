"""Reward detection from screen state."""

import numpy as np


class RewardDetector:
    """Detects reward signals by comparing consecutive frames.

    This is a starting point — you'll need to calibrate regions and methods
    based on the actual game UI. Run scripts/test_capture.py to identify
    where score, health, and game-state info appears on screen.
    """

    def __init__(self, method: str = "pixel_diff", score_region: tuple | None = None):
        self.method = method
        self.score_region = score_region
        self._prev_frame = None

    def compute(self, frame: np.ndarray) -> float:
        """Compute reward from current frame."""
        if self.method == "pixel_diff":
            return self._pixel_diff_reward(frame)
        return 0.0

    def _pixel_diff_reward(self, frame: np.ndarray) -> float:
        """Basic reward: measure change in score region between frames.

        This is a placeholder. Replace with game-specific logic once you
        understand the screen layout (OCR for score, template matching
        for game events, etc.)
        """
        if self._prev_frame is None:
            self._prev_frame = frame
            return 0.0

        if self.score_region:
            l, t, r, b = self.score_region
            curr = frame[t:b, l:r]
            prev = self._prev_frame[t:b, l:r]
        else:
            curr = frame
            prev = self._prev_frame

        diff = np.mean(np.abs(curr.astype(float) - prev.astype(float)))
        self._prev_frame = frame

        # Normalize to small reward signal
        return diff / 255.0

    def detect_episode_end(self, frame: np.ndarray) -> bool:
        """Detect if the episode (hole/round) has ended.

        TODO: Implement based on game-specific signals like:
        - Score screen appearing
        - Specific pixel patterns at known locations
        - Template matching for "hole complete" UI elements
        """
        return False

    def reset(self):
        self._prev_frame = None
