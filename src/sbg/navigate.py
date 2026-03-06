"""Menu navigation and game state detection."""

import time

import cv2
import numpy as np
import pydirectinput

pydirectinput.PAUSE = 0.02

# Menu coordinates (relative to client area at 1280x720)
START_GAME_BUTTON = (270, 315)
START_MATCH_BUTTON = (675, 650)


def click_at(hwnd: int, region: tuple[int, int, int, int], x: int, y: int):
    """Click at a position relative to the game's client area."""
    screen_x = region[0] + x
    screen_y = region[1] + y
    pydirectinput.click(screen_x, screen_y)


# --- State detection ---

def is_loading_screen(frame: np.ndarray) -> bool:
    """Check if a frame is the loading screen.

    The loading screen is a uniform dark purplish-grey with:
    - Mean RGB close to (50, 46, 50)
    - Very low standard deviation (< 15)
    """
    mean_rgb = np.mean(frame, axis=(0, 1))
    std = np.std(frame.astype(float))
    target = np.array([50, 46, 50])
    color_dist = np.linalg.norm(mean_rgb - target)
    return std < 15 and color_dist < 20


def _count_hud_white_pixels(frame: np.ndarray) -> int:
    """Count white pixels in the top-right HUD region (course info box).

    This box appears during the pre-hole countdown and disappears when
    gameplay starts. Calibrated for 1280x720.
    """
    hud = frame[15:80, 480:650]
    gray = cv2.cvtColor(hud, cv2.COLOR_RGB2GRAY)
    return int(np.sum(gray > 200))


def is_countdown_active(frame: np.ndarray) -> bool:
    """Check if the pre-hole countdown HUD is visible."""
    return _count_hud_white_pixels(frame) > 400


# --- Wait functions ---

def wait_for_loading(capture, timeout: float = 30.0) -> bool:
    """Wait for a loading screen to appear and then finish."""
    start = time.time()
    loading_detected = False

    while time.time() - start < timeout:
        frame = capture.grab()
        if frame is None:
            time.sleep(0.1)
            continue

        loading = is_loading_screen(frame)

        if not loading_detected and loading:
            loading_detected = True
            print("  Loading screen detected...")
        elif loading_detected and not loading:
            print("  Loading complete.")
            return True

        time.sleep(0.25)

    if not loading_detected:
        print("  No loading screen detected, continuing anyway...")
        return True

    print("  Warning: loading screen timed out")
    return False


def wait_for_hole_ready(capture, early_start: float = 1.0, timeout: float = 30.0) -> bool:
    """Wait for the pre-hole countdown to finish (reusable between holes).

    The countdown HUD (course info box) appears during 5-4-3-2-1-Golf!
    and disappears when gameplay starts.

    Args:
        capture: ScreenCapture instance
        early_start: Seconds before countdown ends to return (allows early action).
                     Set to 0 for exact countdown end.
        timeout: Max seconds to wait.

    Returns:
        True if gameplay is ready, False if timed out.
    """
    start = time.time()
    hud_appeared = False
    hud_appear_time = None

    while time.time() - start < timeout:
        frame = capture.grab()
        if frame is None:
            time.sleep(0.1)
            continue

        countdown_active = is_countdown_active(frame)

        if not hud_appeared and countdown_active:
            hud_appeared = True
            hud_appear_time = time.time()
            print("  Countdown started (HUD visible)...")

        elif hud_appeared and not countdown_active:
            print("  Countdown finished — gameplay started.")
            return True

        elif hud_appeared and countdown_active and early_start > 0:
            # The countdown lasts ~5 seconds from HUD appearing.
            # Allow early return so agent can start backstroke.
            elapsed_since_hud = time.time() - hud_appear_time
            countdown_duration = 5.0  # 5-4-3-2-1
            early_time = countdown_duration - early_start
            if elapsed_since_hud >= early_time:
                print(f"  Countdown ~{early_start:.0f}s remaining — ready for early action.")
                return True

        time.sleep(0.25)

    if not hud_appeared:
        print("  Warning: HUD never appeared, continuing anyway...")
        return True

    print("  Warning: countdown timed out")
    return False


# --- Navigation sequences ---

def navigate_to_match(hwnd: int, region: tuple[int, int, int, int], capture):
    """Navigate from main menu to first hole gameplay.

    1. Click "Start Game"
    2. Wait for loading
    3. Walk forward + open match setup
    4. Click "Start Match"
    5. Wait for loading
    6. Wait for countdown (with early start)
    """
    print("Step 1: Clicking 'Start Game'...")
    click_at(hwnd, region, *START_GAME_BUTTON)
    time.sleep(0.5)

    print("Step 2: Waiting for loading screen...")
    wait_for_loading(capture)
    time.sleep(1.0)

    print("Step 3: Walking forward...")
    pydirectinput.keyDown("w")
    time.sleep(0.75)
    pydirectinput.keyUp("w")
    time.sleep(0.5)

    print("Step 4: Opening match setup (E)...")
    pydirectinput.press("e")
    time.sleep(1.0)

    print("Step 5: Clicking 'Start Match'...")
    click_at(hwnd, region, *START_MATCH_BUTTON)
    time.sleep(0.5)

    print("Step 6: Waiting for match to load...")
    wait_for_loading(capture)

    print("Step 7: Waiting for countdown...")
    wait_for_hole_ready(capture, early_start=1.0)

    print("Navigation complete — ready to play.")


def wait_for_next_hole(capture):
    """Wait for the next hole to be ready (between holes during a match).

    Handles the loading screen + countdown that occurs between holes.
    """
    print("Waiting for next hole...")
    wait_for_loading(capture)
    wait_for_hole_ready(capture, early_start=1.0)
    print("Next hole ready — ready to play.")
