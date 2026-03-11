"""Game interaction layer — input, capture, navigation, window management."""

from sbg.game.actions import (
    navigate, release_all_keys, enter_stance, exit_stance, aim, set_angle,
    charge_and_shoot, restart_hole, reset_camera_pitch,
)
from sbg.game.capture import ScreenCapture
from sbg.game.navigate import navigate_to_match, wait_for_next_hole
from sbg.game.window import setup_game_window, find_game_window, get_client_region, position_window

__all__ = [
    "navigate", "release_all_keys", "enter_stance", "exit_stance", "aim",
    "set_angle", "charge_and_shoot", "restart_hole", "reset_camera_pitch",
    "ScreenCapture",
    "navigate_to_match", "wait_for_next_hole",
    "setup_game_window", "find_game_window", "get_client_region", "position_window",
]
