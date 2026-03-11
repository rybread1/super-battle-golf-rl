"""Computer vision — game state detection from screen frames."""

from sbg.vision.detect import (
    detect_player_state, is_in_stance, is_loading_screen, detect_scoreboard,
    get_player_progress, is_out_of_bounds, read_strokes_text,
    find_icons, find_pin_icon, find_ball_icon,
    detect_distances, detect_ball_distance, detect_pin_distance,
)

__all__ = [
    "detect_player_state", "is_in_stance", "is_loading_screen", "detect_scoreboard",
    "get_player_progress", "is_out_of_bounds", "read_strokes_text",
    "find_icons", "find_pin_icon", "find_ball_icon",
    "detect_distances", "detect_ball_distance", "detect_pin_distance",
]
