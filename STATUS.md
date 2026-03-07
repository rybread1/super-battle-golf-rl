# Super Battle Golf RL - Project Status

## What Is This?

An RL agent that learns to play **Super Battle Golf** (Steam) entirely from screen pixels and simulated keyboard/mouse input. No game API — just raw vision and reward signals.

**Pipeline**: dxcam captures 1280x720 frames -> OpenCV extracts game state -> Gymnasium environment wraps the game loop -> Stable Baselines 3 (PPO) trains the policy.

---

## Architecture

```
Screen (1280x720)
    |
    v
capture.py -----> env.py (Gymnasium) -----> train.py (PPO)
    |                 |
    v                 v
detect.py         actions.py
(CV detection)    (keyboard/mouse)
    |
    v
reward.py
```

The agent alternates between two phases per hole:
1. **Navigation** — walk toward the ball (forward / forward+left / forward+right)
2. **Shot execution** — enter stance, aim, set angle, choose power, swing

---

## Module Status

### Screen Capture (`capture.py`) — Working

| Feature | Status | Notes |
|---------|--------|-------|
| dxcam capture | Working | Primary method, fast DirectX capture |
| mss fallback | Working | Slower but reliable backup |
| Region capture | Working | Arbitrary rectangular region support |
| FPS target | Working | Default 30 FPS |

No known issues. This module is solid.

---

### Window Management (`window.py`) — Working

| Feature | Status | Notes |
|---------|--------|-------|
| Find game window | Working | Enumerates windows, matches title substring |
| Position window | Working | Moves/restores/foregrounds via Win32 API |
| Get client region | Working | Returns capture area excluding title bar |
| Launch game | Working | Via `steam://rungameid/4069520` |

No known issues. Assumes 1280x720 windowed mode.

---

### Detection (`detect.py`) — Working, Some Limitations

#### Icon Detection (Pin & Ball)
| Feature | Status | Notes |
|---------|--------|-------|
| Template matching | Working | TM_CCORR_NORMED, threshold 0.96, 8 scales (0.5-1.2x) |
| Pin detection | Working | 46/70 on test set, zero false positives |
| Ball detection | Working | 52/70 on test set, zero false positives |
| Color validation | Working | Orange count (pin), white center (ball), green ring (both) |
| UI exclusion zones | Working | Filters progress bar area, club selector, bottom strip |
| Combined find_icons() | Working | Single pass for both icons, shared BGR/HSV conversion |

**Known issues**:
- ~30% miss rate on pin, ~25% on ball (icons at unusual scales or partially occluded)
- Template matching at 8 scales is the main performance bottleneck (~1s per frame for both icons)
- Autumn trees (green+orange) can resemble pin icon — filtered by orange pixel bounds

#### Progress Bar
| Feature | Status | Notes |
|---------|--------|-------|
| Player marker detection | Working | Dark circle with white "?" inside |
| Flag (hole) detection | Working | Orange color in progress bar region |
| Progress ratio (0-1) | Working | Most reliable distance-to-hole signal |

**Known issues**: Returns None if bar is obscured or markers aren't visible.

#### Other Detections
| Feature | Status | Notes |
|---------|--------|-------|
| Stance detection | Working, fragile | Bright white pixels (>240) in left 40% = power bar visible. Can false-positive on other bright UI. |
| Loading screen | Working | Uniform dark purplish-grey check. Could miss if rendering stutters. |
| Bottom text detection | Basic | Only distinguishes "text_present" vs "none" — doesn't identify which prompt. |
| Strokes text | Basic | White pixel count in top-left region. |
| Distance OCR | Working, very slow | EasyOCR (~0.5-1s per call). Lazy import adds ~3s on first use. Throttled to every 30 frames in overlay. |

---

### Actions (`actions.py`) — Working

| Feature | Status | Notes |
|---------|--------|-------|
| Walk forward | Working | Hold W for 0.4s |
| Walk + turn left | Working | W + mouse move -80px |
| Walk + turn right | Working | W + mouse move +80px |
| Enter stance | Working | Right-click + 0.5s delay |
| Aim direction | Working | 16 directions, 50px per step from center |
| Set shot angle | Working | Keys 1-4 |
| Charge and shoot | Working | 10 power levels (0.15s to 2.0s hold) |
| Full shot sequence | Working | enter_stance -> aim -> set_angle -> charge_and_shoot |

**Known issues**:
- Mouse movement via SendInput can fail if game loses focus
- No feedback on whether actions actually registered (relies on vision)
- Power levels are coarse (10 steps)

---

### Menu Navigation (`navigate.py`) — Working, Fragile

| Feature | Status | Notes |
|---------|--------|-------|
| Navigate to match | Working | 7-step sequence from main menu to first hole |
| Loading screen wait | Working | Waits for loading to appear then disappear |
| Countdown detection | Fragile | White pixel count in HUD region (>400 = countdown active) |
| Early start | Working | Agent can start acting 1s before countdown ends |
| Next hole transition | Working | Loading screen + countdown wait |

**Known issues**:
- Menu button coordinates are hardcoded for 1280x720 — breaks if resolution changes
- Countdown detection is brittle (relies on white pixel count threshold)
- No error recovery if menu clicks miss or buttons move
- Duplicate `is_loading_screen()` — should import from detect.py instead

---

### Reward System (`reward.py`) — Working, Needs Tuning

| Signal | Value | Notes |
|--------|-------|-------|
| Step penalty | -0.01 | Per navigation step (discourages wandering) |
| Failed shot | -0.05 | Tried to shoot but wasn't near ball |
| Successful shot | +1.0 | Reached ball and swung |
| Progress delta | x5.0 | Multiplier on progress bar change toward hole |
| Hole complete | +10.0 | Loading screen after shot = hole finished |

**Known issues**:
- Very coarse — no intermediate rewards for getting closer to ball during navigation
- No penalty for shots that move ball backwards (just negative delta * 5)
- Hole completion only detected via loading screen (could miss putts without transition)
- First shot of each hole has no progress delta (prev_progress starts as None)

---

### Environment (`env.py`) — Working, Needs Polish

| Feature | Status | Notes |
|---------|--------|-------|
| Gymnasium interface | Working | MultiDiscrete action space, 84x84 RGB observations |
| Navigation phase | Working | 3 movement actions (forward, forward+left, forward+right) |
| Shot phase | Working | Full shot execution with aim/angle/power |
| Progress tracking | Working | Reads progress bar before/after shots |
| Hole transitions | Working | Detects loading screen, waits for next hole |
| Episode management | Working | Max 500 steps/hole, max 4 holes/session |

**Known issues**:
- Ball landing detection is polling-based (every 0.3s) — can miss fast transitions
- No pause/resume within a hole
- Agent doesn't know if it's walking toward or away from ball (only gets 84x84 pixels)

---

### Training (`train.py`) — Minimal

| Feature | Status | Notes |
|---------|--------|-------|
| PPO training | Working | CnnPolicy, standard hyperparameters |
| Checkpoint save | Working | Saves to `checkpoints/ppo_sbg` |
| Resume from checkpoint | Working | `--resume` flag |
| TensorBoard logging | Working | Logs to `./runs` |

**Known issues**:
- Hyperparameters are untuned defaults (lr=3e-4, n_steps=256, batch_size=64)
- Single environment only (slow data collection)
- No periodic evaluation callback
- No curriculum learning
- `checkpoints/` directory not auto-created (will error on first save)

---

### Debug & Test Scripts (`scripts/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `debug_overlay.py` | Real-time detection visualization with all game state | Working, throttled detections for FPS |
| `record_gameplay.py` | Record frames at intervals for offline testing | Working |
| `test_detect.py` | Batch detection on saved frames | Working |
| `test_env.py` | Random actions through env.step() | Working |
| `test_loop.py` | Manual walk-to-ball -> shoot sequence | Working |
| `test_pin_detect.py` | Batch icon detection with annotated output | Working |
| `test_progress.py` | Live progress bar diagnostic | Working |
| `test_stance.py` | Compare 4 stance detection methods | Working |

---

## Top Priorities for Improvement

1. **Icon detection speed** — Template matching at 8 scales is the #1 bottleneck. Consider reducing scales, downscaling the frame before matching, or switching to a faster method.
2. **Stance detection reliability** — Power bar brightness check has false-positive risk. Could ensemble with other signals explored in `test_stance.py` (dark buttons, orange triangle, mint-green bar).
3. **Reward shaping** — Current rewards are too sparse for efficient learning. Add navigation rewards (getting closer to ball icon center), penalty for ball at screen edges, shaped distance rewards.
4. **Menu navigation robustness** — Hardcoded coordinates and pixel-count thresholds are fragile. Consider template matching for buttons or more robust state detection.
5. **Training infrastructure** — Multi-environment setup, hyperparameter tuning, curriculum learning, evaluation callbacks.
6. **Code cleanup** — Consolidate duplicate `is_loading_screen()` in navigate.py, auto-create checkpoint directory.
