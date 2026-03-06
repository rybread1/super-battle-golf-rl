"""Training entrypoint for the RL agent."""

import argparse

import yaml
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from sbg.env import SuperBattleGolfEnv

ALGORITHMS = {
    "PPO": PPO,
    "DQN": DQN,
}


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Super Battle Golf")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    algo_name = train_cfg["algorithm"]
    AlgoClass = ALGORITHMS[algo_name]

    env = DummyVecEnv([lambda: SuperBattleGolfEnv(config_path=args.config)])

    if args.resume:
        model = AlgoClass.load(args.resume, env=env)
        print(f"Resumed from {args.resume}")
    else:
        model = AlgoClass(
            "CnnPolicy",
            env,
            learning_rate=train_cfg["learning_rate"],
            batch_size=train_cfg["batch_size"],
            gamma=train_cfg["gamma"],
            verbose=1,
            tensorboard_log=train_cfg["tensorboard_log"],
        )

    print(f"Training with {algo_name} for {train_cfg['total_timesteps']} steps...")
    model.learn(total_timesteps=train_cfg["total_timesteps"])

    model.save(f"checkpoints/{algo_name.lower()}_sbg")
    print("Model saved to checkpoints/")

    env.close()


if __name__ == "__main__":
    main()
