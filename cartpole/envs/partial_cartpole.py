"""Partial observation wrapper for CartPole.

This module exposes utilities to create a version of CartPole where the agent
only observes the cart position and pole angle. The velocities are masked out
to simulate a uni-modal (limited sensory) agent.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PartialCartPoleWrapper(gym.ObservationWrapper):
    """Expose only cart position and pole angle to the agent."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("CartPole observation space must be a Box.")

        idx = np.array([0, 2], dtype=np.int64)  # cart position and pole angle
        original_space: spaces.Box = env.observation_space
        low = original_space.low[idx]
        high = original_space.high[idx]

        self._indices = idx
        self.observation_space = spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Select the masked dimensions and ensure float32 precision."""
        return observation[self._indices].astype(np.float32)


def make_partial_cartpole_env(
    gravity: Optional[float] = None, render_mode: Optional[str] = None
) -> gym.Env:
    """Return a CartPole environment with masked observations."""
    env = gym.make("CartPole-v1", render_mode=render_mode)
    if gravity is not None:
        env.unwrapped.gravity = gravity
    return PartialCartPoleWrapper(env)


__all__ = ["PartialCartPoleWrapper", "make_partial_cartpole_env"]
