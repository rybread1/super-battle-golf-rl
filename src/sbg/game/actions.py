"""Low-level action execution for the RL agent.

Provides atomic actions the agent can take each step:
  - Navigation: move in 6 directions (W, S, W+A, W+D, S+A, S+D) with
    independent camera turning (mouse) at 2 intensities per direction
  - Shot: enter stance + aim + set angle + charge/release
"""

import ctypes
import time

import pydirectinput

pydirectinput.PAUSE = 0.02

# --- Raw mouse movement via SendInput ---
# pydirectinput.moveRel doesn't work in some Unity games.
# We use SendInput with MOUSEEVENTF_MOVE for raw relative movement.

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("u", _U),
    ]


def _move_mouse_raw(dx: int, dy: int):
    """Move mouse by (dx, dy) pixels using SendInput."""
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.mi.dx = dx
    inp.mi.dy = dy
    inp.mi.dwFlags = MOUSEEVENTF_MOVE
    inp.mi.mouseData = 0
    inp.mi.time = 0
    inp.mi.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


# --- Navigation constants ---

# Duration of a single movement step (seconds)
MOVE_STEP_DURATION = 0.1

# Mouse pixels for camera turning (per step)
TURN_PIXELS_SMALL = 120

# Turn amounts indexed by turn action (0-2)
TURN_AMOUNTS = [-TURN_PIXELS_SMALL, 0, TURN_PIXELS_SMALL]

# Movement direction keys indexed by move_dir (0-1)
# Turning is handled entirely by mouse movement
MOVE_KEYS = [
    ["w"],        # 0 = forward
    ["s"],        # 1 = backward
]

# --- Shot constants ---

AIM_X_STEPS = 9   # 0-8, center=4
AIM_PX_PER_STEP = 50  # pixels per aim step from center

POWER_DURATIONS = [0.15, 0.35, 0.55, 0.75, 0.95]

ANGLE_KEYS = ["1", "2", "3", "4"]


# --- Navigation actions ---

# Track currently held keys so we can keep them across steps
_held_keys: set[str] = set()


def navigate(move_dir: int, turn: int):
    """Execute a navigation step: move + turn simultaneously.

    Keys are held across steps for smooth movement — only keys that
    change between steps are pressed/released.

    Args:
        move_dir: 0=W, 1=S
        turn: 0=left, 1=none, 2=right
    """
    move_dir = max(0, min(move_dir, len(MOVE_KEYS) - 1))
    turn = max(0, min(turn, len(TURN_AMOUNTS) - 1))

    wanted = set(MOVE_KEYS[move_dir])
    turn_px = TURN_AMOUNTS[turn]

    # Release keys no longer needed
    for k in _held_keys - wanted:
        pydirectinput.keyUp(k)
    # Press keys not yet held
    for k in wanted - _held_keys:
        pydirectinput.keyDown(k)
    _held_keys.clear()
    _held_keys.update(wanted)

    # Apply camera turn
    if turn_px != 0:
        _move_mouse_raw(turn_px, 0)

    time.sleep(MOVE_STEP_DURATION)


def release_all_keys():
    """Release all held movement keys. Call before shots, resets, etc."""
    for k in _held_keys:
        pydirectinput.keyUp(k)
    _held_keys.clear()



# --- Hole management ---

def restart_hole():
    """Restart the current hole by holding R."""
    pydirectinput.keyDown("r")
    time.sleep(3.0)
    pydirectinput.keyUp("r")
    time.sleep(2.0)


# --- Shot actions ---

def enter_stance() -> None:
    """Enter swing stance by holding right mouse button."""
    pydirectinput.mouseDown(button="right")
    time.sleep(0.2)


def exit_stance() -> None:
    """Exit swing stance by releasing right mouse button."""
    pydirectinput.mouseUp(button="right")


# Camera pitch reset: slam down to hit the pitch limit, then pull back up.
# This guarantees a consistent overhead angle regardless of prior drift.
CAMERA_SLAM_DOWN = 800    # pixels to slam down (overshoots to hit limit)
CAMERA_PULL_UP = 535      # pixels to pull back up from the limit


def reset_camera_pitch():
    """Reset camera to a consistent overhead angle.

    Slams the mouse down to hit the game's max pitch-down limit,
    then pulls back up a fixed amount. This gives a repeatable
    camera angle every time.
    """
    _move_mouse_raw(0, CAMERA_SLAM_DOWN)
    time.sleep(0.05)
    _move_mouse_raw(0, -CAMERA_PULL_UP)
    time.sleep(0.05)


def aim(aim_x: int):
    """Aim by moving the mouse horizontally from center.

    aim_x: 0-8 (4=center, <4=left, >4=right).
    Maps to pixel offset: each step = AIM_PX_PER_STEP pixels.
    """
    dx = (aim_x - AIM_X_STEPS // 2) * AIM_PX_PER_STEP
    if dx == 0:
        return
    _move_mouse_raw(dx, 0)
    time.sleep(0.1)


def set_angle(angle_idx: int):
    """Set shot angle by pressing 1-4.

    0 = key 1 (putt/low), 3 = key 4 (high arc).
    """
    if 0 <= angle_idx < len(ANGLE_KEYS):
        pydirectinput.press(ANGLE_KEYS[angle_idx])
        time.sleep(0.1)


def charge_and_shoot(power_level: int):
    """Charge shot to the given power level and release.

    power_level: 0-9 (maps to ~10%-100% power).
    """
    power_level = max(0, min(power_level, len(POWER_DURATIONS) - 1))
    duration = POWER_DURATIONS[power_level]
    pydirectinput.mouseDown(button="left")
    time.sleep(duration)
    pydirectinput.mouseUp(button="left")
