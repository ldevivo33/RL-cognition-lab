"""Plot reward curves for CartPole PPO experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
import csv
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

MODEL_COLORS = {
    "partial": "tab:blue",
    "full": "tab:orange",
}


def load_monitor_series(monitor_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return cumulative timesteps and rewards from a monitor.csv file."""
    if not monitor_file.exists():
        return np.array([]), np.array([])

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
        return np.array([]), np.array([])
    timesteps = np.cumsum(lengths)
    return timesteps, np.array(rewards)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def gather_runs(condition_dir: Path) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    runs = []
    for seed_dir in sorted(condition_dir.glob("seed_*")):
        seed_str = seed_dir.name.split("_")[-1]
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        steps, rewards = load_monitor_series(seed_dir / "monitor.csv")
        if steps.size == 0:
            continue
        runs.append((seed, steps, rewards))
    return runs


def compute_mean_curve(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    num_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    if not curves:
        return np.array([]), np.array([])
    min_max = min(curve[0][-1] for curve in curves)
    grid = np.linspace(0, min_max, num_points)
    interpolated = []
    for steps, rewards in curves:
        interpolated.append(np.interp(grid, steps, rewards))
    mean_rewards = np.mean(interpolated, axis=0)
    return grid, mean_rewards


def plot_results(
    log_dir: Path,
    conditions: List[str],
    smooth_window: int,
    output: Path,
) -> None:
    num_conditions = len(conditions)
    fig, axes = plt.subplots(
        num_conditions,
        1,
        sharex=True,
        sharey=True,
        figsize=(8, 4 * num_conditions),
    )
    if num_conditions == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        cond_dir = log_dir / condition
        if not cond_dir.exists():
            print(f"[WARN] Missing condition directory: {cond_dir}")
            ax.set_title(f"{condition} (missing data)")
            ax.axis("off")
            continue
        runs = gather_runs(cond_dir)
        if not runs:
            ax.set_title(f"{condition} (no runs)")
            ax.axis("off")
            continue
        color = MODEL_COLORS.get(condition, None)
        first_label_used = False
        curves_for_mean: List[Tuple[np.ndarray, np.ndarray]] = []
        for seed, steps, rewards in runs:
            smoothed_rewards = moving_average(rewards, smooth_window)
            if smooth_window > 1 and rewards.size >= smooth_window:
                smoothed_steps = steps[smooth_window - 1 :]
            else:
                smoothed_steps = steps
            label = f"{condition}-seed{seed}" if not first_label_used else None
            ax.plot(
                smoothed_steps,
                smoothed_rewards,
                color=color,
                alpha=0.35,
                label=label,
            )
            first_label_used = True
            curves_for_mean.append((steps, rewards))

        mean_steps, mean_rewards = compute_mean_curve(curves_for_mean)
        if mean_steps.size > 0:
            smoothed_mean = moving_average(mean_rewards, smooth_window)
            if smooth_window > 1 and smoothed_mean.size < mean_steps.size:
                mean_steps_plot = mean_steps[smooth_window - 1 :]
            else:
                mean_steps_plot = mean_steps
            ax.plot(
                mean_steps_plot,
                smoothed_mean,
                color=color,
                linewidth=2.5,
                label=f"{condition} mean",
            )
        ax.set_title(f"{condition.capitalize()} agent")
        ax.set_ylabel("Episode reward")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axes[-1].set_xlabel("Environment steps")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CartPole PPO results.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing condition/seed outputs.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["partial", "full"],
        help="Conditions to include in the plot.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving average window applied to rewards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/training_rewards.png"),
        help="Path for the saved plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_results(args.log_dir, args.conditions, args.smooth_window, args.output)


if __name__ == "__main__":
    main()
