"""Full-observation CartPole utilities."""
from __future__ import annotations

from typing import Optional

import gymnasium as gym


def make_full_cartpole_env(
    gravity: Optional[float] = None, render_mode: Optional[str] = None
) -> gym.Env:
    """Return the standard 4D CartPole environment."""
    env = gym.make("CartPole-v1", render_mode=render_mode)
    if gravity is not None:
        env.unwrapped.gravity = gravity
    return env


__all__ = ["make_full_cartpole_env"]
