"""Training script for fast PPO experiments on CartPole."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import make_full_cartpole_env, make_partial_cartpole_env

MODEL_FILENAME = "ppo_model.zip"
STATS_FILENAME = "vecnormalize.pkl"

EnvFactory = Callable[[Optional[float], Optional[str]], gym.Env]

ENV_BUILDERS: Dict[str, EnvFactory] = {
    "partial": make_partial_cartpole_env,
    "full": make_full_cartpole_env,
}


def set_global_seeds(seed: int) -> None:
    """Force deterministic behavior across libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env_builder(
    condition: str,
    seed: int,
    monitor_path: Optional[Path],
    gravity: Optional[float],
    render_mode: Optional[str] = None,
):
    """Return a thunk that builds the configured environment."""
    factory = ENV_BUILDERS[condition]

    def _init():
        env = factory(gravity=gravity, render_mode=render_mode)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        if monitor_path is not None:
            env = Monitor(env, str(monitor_path))
        return env

    return _init


def export_training_curve(monitor_file: Path, output_csv: Path) -> None:
    """Convert the SB3 monitor log into a 2-column CSV (timesteps, reward)."""
    if not monitor_file.exists():
        return

    rewards = []
    lengths = []
    with monitor_file.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("#") or first.lower() == "r":
                continue
            try:
                reward = float(row[0])
                length = float(row[1])
            except (ValueError, IndexError):
                continue
            rewards.append(reward)
            lengths.append(length)

    if not rewards:
        return

    timesteps = np.cumsum(lengths)
    series = np.column_stack((timesteps, rewards))
    header = "timestep,reward"
    np.savetxt(output_csv, series, delimiter=",", header=header, comments="")


def run_single_evaluation(
    run_dir: Path,
    condition: str,
    seed: int,
    episodes: int,
    gravity: Optional[float],
    normalize_obs: bool,
    device: str,
) -> Dict[str, float]:
    """Evaluate a saved model and return summary statistics."""
    from stable_baselines3.common.vec_env import VecNormalize as VecNormClass

    model_path = run_dir / MODEL_FILENAME
    stats_path = run_dir / STATS_FILENAME

    env_builder = make_env_builder(
        condition=condition,
        seed=seed + 10_000,
        monitor_path=None,
        gravity=gravity,
    )
    eval_env = DummyVecEnv([env_builder])
    if normalize_obs and stats_path.exists():
        eval_env = VecNormClass.load(str(stats_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device=device)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
        render=False,
    )
    eval_env.close()
    return {
        "episodes": float(episodes),
        "gravity": float(gravity if gravity is not None else 9.8),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic PPO on CartPole")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(ENV_BUILDERS.keys()),
        default=sorted(ENV_BUILDERS.keys()),
        help="Which agent configurations to train.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds for reproducibility.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=150_000,
        help="Training timesteps per run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="Rollout horizon per PPO update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size (must divide n-steps).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Directory to store logs and models.",
    )
    parser.add_argument(
        "--train-gravity",
        type=float,
        default=None,
        help="Override gravity during training (default: env default).",
    )
    parser.add_argument(
        "--eval-gravity",
        type=float,
        default=None,
        help="Optional gravity override during evaluation.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Episodes per evaluation run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for SB3 (cpu or cuda).",
    )
    parser.add_argument(
        "--do-eval",
        action="store_true",
        help="Run evaluations after training.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (requires tensorboard package).",
    )
    parser.add_argument(
        "--normalize-obs",
        dest="normalize_obs",
        action="store_true",
        help="Enable VecNormalize for observations.",
    )
    parser.add_argument(
        "--no-normalize-obs",
        dest="normalize_obs",
        action="store_false",
        help="Disable VecNormalize.",
    )
    parser.set_defaults(normalize_obs=True)
    return parser.parse_args()


def ensure_batch_size(n_steps: int, batch_size: int) -> int:
    if n_steps % batch_size != 0:
        raise ValueError("batch_size must divide n_steps for PPO.")
    return batch_size


def train_condition(
    condition: str,
    seeds: Iterable[int],
    args: argparse.Namespace,
) -> None:
    for seed in seeds:
        run_dir = args.log_dir / condition / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        monitor_file = run_dir / "monitor.csv"

        set_global_seeds(seed)
        env_builder = make_env_builder(
            condition=condition,
            seed=seed,
            monitor_path=monitor_file,
            gravity=args.train_gravity,
        )
        vec_env = DummyVecEnv([env_builder])
        if args.normalize_obs:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            )

        policy_kwargs = dict(net_arch=[32, 32], ortho_init=True)
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=ensure_batch_size(args.n_steps, args.batch_size),
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=1,
            tensorboard_log=str(run_dir / "tb") if args.tensorboard else None,
            policy_kwargs=policy_kwargs,
            device=args.device,
        )

        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        model.save(run_dir / MODEL_FILENAME)
        if args.normalize_obs:
            vec_env.save(run_dir / STATS_FILENAME)
        vec_env.close()

        export_training_curve(monitor_file, run_dir / "episode_rewards.csv")

        if args.do_eval:
            eval_results = {
                "nominal": run_single_evaluation(
                    run_dir=run_dir,
                    condition=condition,
                    seed=seed,
                    episodes=args.eval_episodes,
                    gravity=args.train_gravity,
                    normalize_obs=args.normalize_obs,
                    device=args.device,
                )
            }
            if args.eval_gravity is not None:
                eval_results["modified_gravity"] = run_single_evaluation(
                    run_dir=run_dir,
                    condition=condition,
                    seed=seed,
                    episodes=args.eval_episodes,
                    gravity=args.eval_gravity,
                    normalize_obs=args.normalize_obs,
                    device=args.device,
                )
            with open(run_dir / "evaluation.json", "w", encoding="utf-8") as fp:
                json.dump(eval_results, fp, indent=2)


def main() -> None:
    args = parse_args()
    args.log_dir = Path(args.log_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        if condition not in ENV_BUILDERS:
            raise ValueError(f"Unknown condition: {condition}")
        train_condition(condition, args.seeds, args)


if __name__ == "__main__":
    main()
