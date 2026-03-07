"""Test the environment loop with random actions.

Usage: uv run python scripts/test_env.py --no-launch

Make sure the game is at the main menu before running.
Takes a few random shots and prints the results.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbg.env import SuperBattleGolfEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--shots", type=int, default=5, help="Number of shots to take")
    args = parser.parse_args()

    env = SuperBattleGolfEnv(auto_launch=not args.no_launch)

    print("Resetting environment (this will navigate to a match)...")
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}, info: {info}\n")

    for i in range(args.shots):
        action = env.action_space.sample()
        aim, angle, power = action
        print(f"Shot {i+1}: aim={aim}, angle={angle+1}, power={(power+1)*10}%")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        print(f"  info={info}\n")

        if terminated or truncated:
            print("Hole ended. Resetting...")
            obs, info = env.reset()
            print(f"Reset info: {info}\n")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
