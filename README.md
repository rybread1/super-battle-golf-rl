# Super Battle Golf RL

Training an RL agent to play [Super Battle Golf](https://store.steampowered.com/app/4069520) end-to-end via screen capture and simulated keyboard input.

The agent sees raw pixels, decides how to navigate to its ball and hit shots, and learns entirely from reward signals. Still very much a work in progress.

## How It Works

- **Screen capture** (dxcam) grabs 1280x720 frames from the game window
- **Computer vision** (OpenCV template matching) extracts game state from pixels
- **CNN detection** (IconNet) detects ball/pin icons and objects in the scene
- **Gymnasium environment** wraps the game loop with an 84x84 RGB observation space
- **Stable Baselines 3** trains the RL policy (PPO)

## Project Layout

```
src/sbg/
├── env.py              # Gymnasium environment
├── reward.py           # Reward signals
├── game/               # Game interaction (capture, input, window management)
├── vision/             # OpenCV-based detection (icons, progress bar, stance)
└── models/             # CNN icon/object detection (IconNet)

scripts/
├── training/           # RL and CNN training scripts
└── tools/              # Annotation, recording, and debug utilities
```

## Quick Start

Requires Python 3.12, Windows, and the game running at 1280x720 windowed.

```bash
# Install dependencies
uv sync

# Annotate training data
uv run python scripts/tools/annotate.py --dir cnn_training

# Train the CNN detector
uv run python scripts/training/train_icons.py --data screenshots/cnn_training/annotations.json --name v1

# Train the RL agent
uv run python scripts/training/train.py
```

## Status

Early development — core game loop and CNN detection are functional, RL training is ongoing.
