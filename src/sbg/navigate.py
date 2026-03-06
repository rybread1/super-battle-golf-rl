"""Menu navigation — automate the sequence from main menu to gameplay."""

import time

import numpy as np
import pydirectinput

pydirectinput.PAUSE = 0.02

# Menu coordinates (relative to client area at 1280x720)
START_GAME_BUTTON = (270, 315)
START_MATCH_BUTTON = (675, 650)


def click_at(hwnd: int, region: tuple[int, int, int, int], x: int, y: int):
    """Click at a position relative to the game's client area.

    Converts client-relative coords to absolute screen coords and clicks.
    """
    screen_x = region[0] + x
    screen_y = region[1] + y
    pydirectinput.click(screen_x, screen_y)


def wait_for_loading(capture, timeout: float = 30.0, grey_threshold: float = 15.0):
    """Wait for a loading screen (mostly grey/uniform) to finish.

    Detects loading by checking if the frame has low variance (uniform color).
    Waits for loading to start, then waits for it to end.
    """
    # First, wait for loading screen to appear
    start = time.time()
    loading_detected = False

    while time.time() - start < timeout:
        frame = capture.grab()
        if frame is None:
            time.sleep(0.1)
            continue

        variance = np.std(frame.astype(float))

        if not loading_detected and variance < grey_threshold:
            loading_detected = True
            print("  Loading screen detected...")
        elif loading_detected and variance > grey_threshold:
            print("  Loading complete.")
            return True

        time.sleep(0.25)

    if not loading_detected:
        print("  No loading screen detected, continuing anyway...")
        return True

    print("  Warning: loading screen timed out")
    return False


def navigate_to_match(hwnd: int, region: tuple[int, int, int, int], capture):
    """Full navigation sequence: main menu -> gameplay.

    1. Click "Start Game"
    2. Wait for loading
    3. Walk forward (W key)
    4. Press E to open match setup
    5. Click "Start Match"
    6. Wait for loading
    """
    print("Step 1: Clicking 'Start Game'...")
    click_at(hwnd, region, *START_GAME_BUTTON)
    time.sleep(0.5)

    print("Step 2: Waiting for loading screen...")
    wait_for_loading(capture)
    time.sleep(1.0)  # Extra buffer after loading

    print("Step 3: Walking forward...")
    pydirectinput.keyDown("w")
    time.sleep(0.75)
    pydirectinput.keyUp("w")
    time.sleep(0.5)

    print("Step 4: Opening match setup (E)...")
    pydirectinput.press("e")
    time.sleep(1.0)  # Wait for menu to open

    print("Step 5: Clicking 'Start Match'...")
    click_at(hwnd, region, *START_MATCH_BUTTON)
    time.sleep(0.5)

    print("Step 6: Waiting for match to load...")
    wait_for_loading(capture)
    time.sleep(1.0)  # Extra buffer

    print("Navigation complete — match should be starting.")
