# Super Battle Golf RL

## Goal
Train an RL agent to play Super Battle Golf (Steam, ID 4069520) end-to-end via screen capture and simulated keyboard input. The agent sees raw pixels, decides how to navigate to its ball and how to hit shots, and learns entirely from reward signals.

## How It Works
- **Screen capture** (dxcam) grabs 1280x720 frames from the game window
- **Computer vision** (OpenCV template matching, color analysis, OCR) extracts game state from pixels
- **Gymnasium environment** wraps the game loop: observation is an 84x84 RGB frame, actions are movement or shot attempts
- **Stable Baselines 3** trains the policy

The agent alternates between two phases each hole:
1. **Navigation** — walk toward the ball (forward, forward+left, forward+right)
2. **Shot execution** — enter stance near the ball, aim, set angle, choose power, swing

## Project Layout
- `src/sbg/env.py` — Gymnasium environment
- `src/sbg/reward.py` — Reward constants and computation
- `src/sbg/game/` — Game interaction layer
  - `actions.py` — Keyboard input helpers
  - `capture.py` — dxcam screen capture
  - `window.py` — Game window management
  - `navigate.py` — Menu navigation, hole transitions
- `src/sbg/vision/` — Computer vision detection
  - `detect.py` — CV detection (icons, progress bar, stance, loading screen, OCR)
  - `templates/` — BGRA template images for icon matching
- `src/sbg/models/` — Learned models
  - `icon_net.py` — CNN for ball/pin icon detection (320x180 input, ~456K params)
  - `dataset.py` — Dataset loader for annotated frames (JSON format)
  - `loss.py` — Loss function (BCE + smooth L1)
- `scripts/` — Scripts organized by purpose
  - `training/train.py` — RL training entrypoint (SB3 PPO)
  - `training/train_icons.py` — Training script for icon detection CNN
  - `tools/debug_overlay.py` — Live debug visualization
  - `tools/record_gameplay.py` — Record gameplay screenshots
  - `tools/screenshot.py` — Single screenshot capture
  - `tools/start_game.py` — Launch game and navigate to match
- `screenshots/` — Recorded gameplay frames for offline testing

## Gotchas and Known Issues
- **EasyOCR is slow to import** — the reader is lazily initialized on first use to avoid blocking startup
- **Template matching false positives** — autumn trees (green+orange) and the progress bar flag (top-left) can look like icons. Filtered by UI exclusion zones and orange pixel count bounds.
- **UI exclusion zones** — right 22% (club selector), top 12% + left 40% (progress bar area), bottom 8% (prompts). Icon detections in these areas are rejected.
- **Tesseract is not on PATH** — installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- **Ball icon at screen edges = bad** — when the ball icon appears near the top or bottom edge of the screen, it means the player has walked away from their ball. This is a critical negative signal the agent needs to learn to avoid.
- **Stance detection relies on power bar brightness** — looks for bright white dots (>240) in the left 40% of screen. Can false-positive if other bright UI elements appear there.
- **Loading screen = hole complete** — a uniform dark purplish-grey frame signals transition between holes. This is how we detect hole completion.
- **Progress bar is the most reliable distance signal** — the dark circle with "?" (player) and orange flag (hole) in the top-left bar. More reliable than OCR distance readings.
- **Game must be 1280x720 windowed** — all detection regions are calibrated to this resolution.
