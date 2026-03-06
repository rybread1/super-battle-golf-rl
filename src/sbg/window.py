"""Game window management — launch, find, and position the game window."""

import subprocess
import time

import win32con
import win32gui

# Steam app ID for Super Battle Golf
STEAM_APP_ID = "4069520"

# Default window config — consistent across sessions
DEFAULT_WINDOW = {
    "x": 0,
    "y": 0,
    "width": 1280,
    "height": 720,
}


def launch_game():
    """Launch Super Battle Golf via Steam."""
    subprocess.Popen(
        ["cmd", "/c", "start", f"steam://rungameid/{STEAM_APP_ID}"],
        shell=False,
    )


def find_game_window(title_substring: str = "Super Battle Golf", timeout: float = 30.0) -> int:
    """Find the game window by title. Returns the window handle (hwnd).

    Waits up to `timeout` seconds for the window to appear.
    """
    start = time.time()
    while time.time() - start < timeout:
        hwnd = _find_window_by_title(title_substring)
        if hwnd:
            return hwnd
        time.sleep(1.0)

    raise TimeoutError(
        f"Could not find window with title containing '{title_substring}' "
        f"within {timeout} seconds"
    )


def position_window(
    hwnd: int,
    x: int = DEFAULT_WINDOW["x"],
    y: int = DEFAULT_WINDOW["y"],
    width: int = DEFAULT_WINDOW["width"],
    height: int = DEFAULT_WINDOW["height"],
):
    """Move the game window to exact screen coordinates.

    The game should already be in windowed mode at 1280x720 (set via
    Config.json and Unity registry keys). This just ensures consistent
    positioning across sessions.
    """
    # Restore window if minimized or maximized
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(0.2)

    # Move to target position (keep current size since game controls that)
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_NOTOPMOST,
        x, y, 0, 0,
        win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
    )

    # Bring to foreground
    win32gui.SetForegroundWindow(hwnd)


def get_client_region(hwnd: int) -> tuple[int, int, int, int]:
    """Get the client area (excluding title bar/borders) as (left, top, width, height).

    This is the actual game render area, which is what screen capture should target.
    """
    rect = win32gui.GetClientRect(hwnd)
    client_width = rect[2]
    client_height = rect[3]

    # Convert client (0,0) to screen coordinates
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))

    return (left, top, client_width, client_height)


def setup_game_window(
    x: int = DEFAULT_WINDOW["x"],
    y: int = DEFAULT_WINDOW["y"],
    width: int = DEFAULT_WINDOW["width"],
    height: int = DEFAULT_WINDOW["height"],
    launch: bool = True,
) -> tuple[int, tuple[int, int, int, int]]:
    """Full setup: launch game, find window, position it, return capture region.

    Returns:
        (hwnd, (left, top, width, height)) — the window handle and client region
        for screen capture.
    """
    if launch:
        print("Launching Super Battle Golf...")
        launch_game()

    print("Waiting for game window...")
    hwnd = find_game_window()
    print(f"Found window (hwnd={hwnd})")

    print(f"Positioning window to ({x}, {y})...")
    position_window(hwnd, x, y, width, height)
    time.sleep(0.5)  # Let window settle

    region = get_client_region(hwnd)
    print(f"Client capture region: left={region[0]}, top={region[1]}, "
          f"width={region[2]}, height={region[3]}")

    return hwnd, region


def _find_window_by_title(substring: str) -> int | None:
    """Enumerate all windows and return the first matching hwnd."""
    result = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if substring.lower() in title.lower():
                result.append(hwnd)
        return True

    win32gui.EnumWindows(callback, None)
    return result[0] if result else None
