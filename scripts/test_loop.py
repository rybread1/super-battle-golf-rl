"""Test the basic control loop: walk to ball, enter stance, take a shot.

Holds W to walk continuously and steers toward the ball icon.
Run with the game already on a hole (not in a menu).
Gives you 5 seconds to alt-tab to the game window.
"""

import time

import pydirectinput

from sbg.capture import ScreenCapture
from sbg.detect import (
    find_ball_icon,
    find_pin_icon,
    get_player_progress,
    is_in_stance,
    is_loading_screen,
)
from sbg.actions import (
    _move_mouse_raw, TURN_PIXELS,
    enter_stance, aim, set_angle, charge_and_shoot,
)
from sbg.window import find_game_window, position_window, get_client_region

SCREEN_CENTER_X = 640
FACING_THRESHOLD = 100

# How often (seconds) to check ball position and try stance
STEER_INTERVAL = 0.3
STANCE_INTERVAL = 4.0
MAX_WALK_TIME = 90.0


def main():
    print("Finding game window...")
    hwnd = find_game_window(timeout=5)
    position_window(hwnd)
    time.sleep(0.3)
    region = get_client_region(hwnd)
    print(f"Capture region: {region}")

    cap = ScreenCapture(region=region, fps=30)
    cap.start()
    time.sleep(0.5)

    frame = cap.grab()
    if frame is None:
        print("ERROR: Could not grab frame")
        cap.stop()
        return

    print(f"Frame shape: {frame.shape}")
    print(f"Loading screen: {is_loading_screen(frame)}")
    print(f"In stance: {is_in_stance(frame)}")
    print(f"Progress: {get_player_progress(frame)}")
    print(f"Ball icon: {find_ball_icon(frame)}")
    print(f"Pin icon: {find_pin_icon(frame)}")

    print("\n--- You have 5 seconds to focus the game window ---")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    # === Phase 1: Walk to ball ===
    # Hold W the entire time. Steer by sending mouse movement while walking.
    # Periodically stop to try entering stance.
    print("\n=== Phase 1: Navigate to ball ===")
    in_stance = False

    pydirectinput.keyDown("w")
    start = time.time()
    last_steer = 0.0
    last_stance = 0.0

    try:
        while time.time() - start < MAX_WALK_TIME:
            elapsed = time.time() - start

            # Try entering stance periodically
            if elapsed - last_stance >= STANCE_INTERVAL:
                last_stance = elapsed
                pydirectinput.keyUp("w")
                time.sleep(0.05)

                enter_stance()
                time.sleep(0.3)

                frame = cap.grab()
                if frame is not None and is_in_stance(frame):
                    print(f"  [{elapsed:.1f}s] Entered stance!")
                    in_stance = True
                    break

                # Resume walking
                pydirectinput.keyDown("w")
                continue

            # Steer toward ball periodically (while still holding W)
            if elapsed - last_steer >= STEER_INTERVAL:
                last_steer = elapsed
                frame = cap.grab()
                if frame is not None:
                    ball_pos = find_ball_icon(frame)
                    if ball_pos is not None:
                        bx, _ = ball_pos
                        offset = bx - SCREEN_CENTER_X
                        if abs(offset) > FACING_THRESHOLD:
                            direction = "right" if offset > 0 else "left"
                            print(f"  [{elapsed:.1f}s] ball x={bx}, turning {direction}")
                            _move_mouse_raw(
                                TURN_PIXELS if offset > 0 else -TURN_PIXELS,
                                0,
                            )

            time.sleep(0.02)
    finally:
        pydirectinput.keyUp("w")

    if not in_stance:
        print("FAILED: Could not enter stance")
        cap.stop()
        return

    # === Phase 2: Take a shot ===
    print("\n=== Phase 2: Taking shot (aim=8/straight, angle=2/mid, power=5/50%) ===")
    frame = cap.grab()
    if frame is not None:
        print(f"  Progress (pre-shot): {get_player_progress(frame)}")

    aim(8)
    set_angle(2)
    time.sleep(0.2)
    charge_and_shoot(5)

    # === Phase 3: Wait for ball to land ===
    print("\n=== Phase 3: Waiting for ball to land ===")
    time.sleep(2.0)
    stable = 0
    for _ in range(30):
        frame = cap.grab()
        if frame is not None and not is_loading_screen(frame):
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        time.sleep(0.3)

    # === Phase 4: Post-shot state ===
    print("\n=== Phase 4: Post-shot state ===")
    frame = cap.grab()
    if frame is not None:
        print(f"  Loading screen: {is_loading_screen(frame)}")
        print(f"  In stance: {is_in_stance(frame)}")
        print(f"  Progress (post-shot): {get_player_progress(frame)}")
        print(f"  Ball icon: {find_ball_icon(frame)}")
        print(f"  Pin icon: {find_pin_icon(frame)}")

    print("\nDone!")
    cap.stop()


if __name__ == "__main__":
    main()
