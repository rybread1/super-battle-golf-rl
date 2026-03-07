"""Low-level action execution for the RL agent.

Provides atomic actions the agent can take each frame:
  - Navigation: walk forward, turn left, turn right
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

# Mouse pixels per turn action (controls turn speed)
TURN_PIXELS = 80

# Duration of a single forward walk step (seconds)
WALK_STEP_DURATION = 0.4

# --- Shot constants ---

AIM_STEP_PIXELS = 50

POWER_DURATIONS = [0.15, 0.35, 0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 2.0]

ANGLE_KEYS = ["1", "2", "3", "4"]


# --- Navigation actions ---

def walk_forward():
    """Walk forward for one step (hold W for WALK_STEP_DURATION)."""
    pydirectinput.keyDown("w")
    time.sleep(WALK_STEP_DURATION)
    pydirectinput.keyUp("w")


def walk_forward_turn_left():
    """Walk forward while turning left."""
    pydirectinput.keyDown("w")
    _move_mouse_raw(-TURN_PIXELS, 0)
    time.sleep(WALK_STEP_DURATION)
    pydirectinput.keyUp("w")


def walk_forward_turn_right():
    """Walk forward while turning right."""
    pydirectinput.keyDown("w")
    _move_mouse_raw(TURN_PIXELS, 0)
    time.sleep(WALK_STEP_DURATION)
    pydirectinput.keyUp("w")


# --- Shot actions ---

def enter_stance() -> None:
    """Enter swing stance by right-clicking."""
    pydirectinput.click(button="right")
    time.sleep(0.5)


def aim(direction: int):
    """Aim by moving the mouse horizontally.

    direction: 0-15 (8 = straight, <8 = left, >8 = right)
    """
    offset = direction - 8
    if offset == 0:
        return
    pixels = offset * AIM_STEP_PIXELS
    _move_mouse_raw(pixels, 0)
    time.sleep(0.1)


def set_angle(angle_idx: int):
    """Set shot angle by pressing 1-4.

    0 = key 1 (putt/low), 3 = key 4 (high arc).
    """
    if 0 <= angle_idx < len(ANGLE_KEYS):
        pydirectinput.press(ANGLE_KEYS[angle_idx])
        time.sleep(0.2)


def charge_and_shoot(power_level: int):
    """Charge shot to the given power level and release.

    power_level: 0-9 (maps to ~10%-100% power).
    """
    power_level = max(0, min(power_level, len(POWER_DURATIONS) - 1))
    duration = POWER_DURATIONS[power_level]
    pydirectinput.mouseDown(button="left")
    time.sleep(duration)
    pydirectinput.mouseUp(button="left")


def execute_shot(aim_direction: int, angle_idx: int, power_level: int):
    """Execute a complete shot: enter stance, aim, set angle, charge and release.

    Returns True if stance was entered successfully, False otherwise.
    """
    enter_stance()
    time.sleep(0.3)

    # Check if we actually entered stance (caller should verify via frame)
    # For now just execute — the env checks stance separately
    aim(aim_direction)
    set_angle(angle_idx)
    time.sleep(0.2)
    charge_and_shoot(power_level)
