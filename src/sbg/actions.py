"""Input simulation for game control via pydirectinput."""

import time

import pydirectinput

# Disable the default pause between actions
pydirectinput.PAUSE = 0.02


class ActionSpace:
    """Maps discrete action indices to game inputs."""

    def __init__(self, action_config: list[dict]):
        self.actions = action_config
        self.n = len(self.actions)

    def act(self, action_idx: int, hold_duration: float = 0.05):
        """Execute a discrete action by index."""
        if action_idx < 0 or action_idx >= self.n:
            return

        action = self.actions[action_idx]
        key = action["key"]

        pydirectinput.keyDown(key)
        time.sleep(hold_duration)
        pydirectinput.keyUp(key)

    def no_op(self):
        """Do nothing for one step."""
        pass

    def __len__(self):
        return self.n
