"""Test input simulation — verify that key presses reach the game.

Usage: uv run python scripts/test_actions.py

This will cycle through each action with a delay so you can verify
the game responds to each input.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml
from sbg.actions import ActionSpace


def main():
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    actions = ActionSpace(config["actions"])

    print("Testing actions in 5 seconds...")
    print("Switch to the game window now!")
    time.sleep(5)

    for i, action in enumerate(actions.actions):
        print(f"  Action {i}: {action['name']} (key: {action['key']})")
        actions.act(i, hold_duration=0.1)
        time.sleep(1.0)

    print("Done. Did the game respond to all inputs?")


if __name__ == "__main__":
    main()
