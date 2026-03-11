# Super Battle Golf RL

## Goal
Train an RL agent to play Super Battle Golf (Steam, ID 4069520) end-to-end via screen capture and simulated keyboard input. The agent sees raw pixels, decides how to navigate to its ball and how to hit shots, and learns entirely from reward signals.

## How It Works
- **Screen capture** (dxcam) grabs 1280x720 frames from the game window
- **Computer vision** (OpenCV template matching, color analysis) extracts game state from pixels
- **CNN detection** (IconNet) detects ball/pin icons and actual ball/pin objects
- **Gymnasium environment** wraps the game loop: observation is an 84x84 RGB frame, actions are movement or shot attempts
- **Stable Baselines 3** trains the RL policy

The agent alternates between two phases each hole:
1. **Navigation** — walk toward the ball (forward, forward+left, forward+right)
2. **Shot execution** — enter stance near the ball, aim, set angle, choose power, swing

## Project Layout
- `src/sbg/env.py` — Gymnasium environment
- `src/sbg/reward.py` — Reward constants and computation
- `src/sbg/game/` — Game interaction layer
  - `actions.py` — Keyboard input helpers
  - `capture.py` — dxcam screen capture (validates 1280x720)
  - `window.py` — Game window management
  - `navigate.py` — Menu navigation, hole transitions
- `src/sbg/vision/` — Computer vision detection
  - `detect.py` — CV detection (icons, progress bar, stance, loading screen)
  - `templates/` — BGRA template images for icon matching
- `src/sbg/models/` — Learned models
  - `icon_net.py` — CNN for ball/pin icon + object detection (640x360 input, 5 conv blocks + skip connections, spatial heatmap heads, ~825K params)
  - `dataset.py` — Dataset loader with augmentation (flip, crop/zoom, color jitter, blur, noise)
  - `loss.py` — Loss function (BCE presence + 10x-weighted smooth L1 coordinates + 5x heatmap supervision)
- `scripts/training/` — Training scripts
  - `train.py` — RL training entrypoint (SB3 PPO)
  - `train_icons.py` — CNN training with TensorBoard, ReduceLROnPlateau, checkpoint resume
- `scripts/tools/` — Utilities
  - `annotate.py` — Click-to-annotate tool (4 targets: ball_icon, pin_icon, ball, pin)
  - `debug_overlay.py` — Live debug visualization
  - `live_cnn.py` — Live CNN overlay showing model predictions on game
  - `record_gameplay.py` — Record gameplay screenshots (validates frame dimensions)
  - `reorder_frames.py` — Fix duplicate frame numbering across recording sessions
  - `screenshot.py` — Single screenshot capture
  - `start_game.py` — Launch game and navigate to match
  - `view_augmentations.py` — Visualize training data augmentations
  - `visualize_architecture.py` — Generate visual diagram of IconNet architecture
  - `visualize_predictions.py` — Visualize CNN predictions on validation frames
- `screenshots/` — Recorded gameplay frames and annotations
- `checkpoints/` — Saved model weights
- `runs/` — TensorBoard logs

## CNN Detection (IconNet)
Four detection targets, each with presence + (x, y) coordinates:
- `ball_icon` — UI ball indicator overlay
- `pin_icon` — UI flag indicator overlay
- `ball` — actual golf ball in the scene (important when close and icon disappears)
- `pin` — actual flagstick on the green

Training: `uv run python scripts/training/train_icons.py --data screenshots/cnn_training/annotations.json --name v1`
Resume: `--resume checkpoints/icon_net/v1_best.pt`
TensorBoard: `uv run tensorboard --logdir runs/icon_net`
Annotation: `uv run python scripts/tools/annotate.py --dir cnn_training`

## Gotchas and Known Issues
- **Game must be 1280x720 windowed** — all detection regions and CNN input are calibrated to this resolution.
- **Template matching false positives** — autumn trees (green+orange) and the progress bar flag (top-left) can look like icons. Filtered by UI exclusion zones and orange pixel count bounds.
- **UI exclusion zones** — pin: top 12% & left 40% (progress bar flag). Ball: top 12% full-width and bottom 12% (distance marker icons). Icon detections in these areas are rejected.
- **Ball/pin object labels are sparse** — only ~31% and ~20% of training frames have these visible. Need more close-range recordings to improve CNN accuracy.
- **Ball icon at screen edges = bad** — when the ball icon appears near the top or bottom edge of the screen, it means the player has walked away from their ball.
- **Power bar can_hit detection** — uses HSV-based chunked comparison of strips flanking the power bar's left edge. Detects contrast difference between outside and inside strips to determine if power bar is active.
- **Loading screen = hole complete** — a uniform dark purplish-grey frame signals transition between holes.
- **Progress bar is the most reliable distance signal** — the dark circle with "?" (player) and orange flag (hole) in the top-left bar.
