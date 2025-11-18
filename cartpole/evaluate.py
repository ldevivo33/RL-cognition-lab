"""Evaluate saved PPO agents on CartPole."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import make_full_cartpole_env, make_partial_cartpole_env

MODEL_FILENAME = "ppo_model.zip"
STATS_FILENAME = "vecnormalize.pkl"

ENV_BUILDERS = {
    "partial": make_partial_cartpole_env,
    "full": make_full_cartpole_env,
}


def make_env_builder(
    condition: str,
    seed: int,
    gravity: Optional[float],
    render_mode: Optional[str] = None,
):
    factory = ENV_BUILDERS[condition]

    def _init():
        env = factory(gravity=gravity, render_mode=render_mode)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def evaluate_run(
    run_dir: Path,
    condition: str,
    seed: int,
    episodes: int,
    gravity: Optional[float],
    deterministic: bool,
    normalize_obs: bool,
    device: str,
) -> Dict[str, float]:
    model_path = run_dir / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model at {model_path}")

    env_builder = make_env_builder(condition, seed + 1000, gravity)
    eval_env = DummyVecEnv([env_builder])

    stats_path = run_dir / STATS_FILENAME
    if normalize_obs and stats_path.exists():
        eval_env = VecNormalize.load(str(stats_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device=device)
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=deterministic,
        render=False,
    )
    eval_env.close()
    return {
        "condition": condition,
        "seed": seed,
        "episodes": float(episodes),
        "gravity": float(gravity if gravity is not None else 9.8),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO CartPole agents.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(ENV_BUILDERS.keys()),
        default=sorted(ENV_BUILDERS.keys()),
        help="Agent conditions to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds that identify each saved run.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory storing experiment outputs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Evaluation episodes per run.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=None,
        help="Optional gravity override for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used to run inference.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (default deterministic).",
    )
    parser.add_argument(
        "--normalize-obs",
        dest="normalize_obs",
        action="store_true",
        help="Load VecNormalize statistics if available.",
    )
    parser.add_argument(
        "--no-normalize-obs",
        dest="normalize_obs",
        action="store_false",
        help="Ignore VecNormalize statistics.",
    )
    parser.set_defaults(normalize_obs=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to store evaluation summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir = Path(args.log_dir)
    deterministic = not args.stochastic

    summaries = []
    for condition in args.conditions:
        for seed in args.seeds:
            run_dir = args.log_dir / condition / f"seed_{seed}"
            if not run_dir.exists():
                print(f"[WARN] Run directory missing: {run_dir}")
                continue
            summary = evaluate_run(
                run_dir=run_dir,
                condition=condition,
                seed=seed,
                episodes=args.episodes,
                gravity=args.gravity,
                deterministic=deterministic,
                normalize_obs=args.normalize_obs,
                device=args.device,
            )
            summaries.append(summary)
            print(
                f"{condition} seed={seed} gravity={summary['gravity']:.1f} -> "
                f"mean={summary['mean_reward']:.2f} +/- {summary['std_reward']:.2f}"
            )

    if args.output is not None and summaries:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(summaries, fp, indent=2)


if __name__ == "__main__":
    main()
