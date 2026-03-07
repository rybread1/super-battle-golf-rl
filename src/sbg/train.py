"""Training entrypoint for the frame-level RL agent."""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sbg.env import SuperBattleGolfEnv


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Super Battle Golf")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-launch", action="store_true", help="Don't auto-launch game")
    args = parser.parse_args()

    env = DummyVecEnv([lambda: SuperBattleGolfEnv(auto_launch=not args.no_launch)])

    if args.resume:
        model = PPO.load(args.resume, env=env)
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./runs",
        )

    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    model.save("checkpoints/ppo_sbg")
    print("Model saved to checkpoints/ppo_sbg")

    env.close()


if __name__ == "__main__":
    main()
