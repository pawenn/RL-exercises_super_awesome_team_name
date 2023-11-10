from __future__ import annotations

from typing import DefaultDict

import gymnasium as gym
import numpy as np


class EpsilonGreedyPolicy(object):
    """A Policy doing Epsilon Greedy Exploration."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        seed: int = 0,
    ) -> None:
        """Init

        Parameters
        ----------
        env : gym.Env
            Environment
        epsilon: float
            Exploration rate
        seed : int, optional
            Seed, by default None
        """
        self.env = env
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, Q: DefaultDict, state: tuple, exploration_rate: float = 0.0, eval: bool = False) -> int:  # type: ignore # noqa: E501
        """Select action

        Parameters
        ----------
        state : tuple
            State
        exploration_rate : float, optional
            exploration rate (epsilon), by default 0.0
        eval: bool
            evaluation mode - if true, exploration should be turned off.

        Returns
        -------
        int
            action
        """
        # If not evaluation, randomly sample action based on epsilon
        if not eval and np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()

        q_values = Q[state]

        action = np.argmax(q_values).item()
        return action
