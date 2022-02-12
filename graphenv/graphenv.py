import logging
from typing import Dict, Tuple

import gym
import numpy as np
from gym.spaces import Box

from graphenv.node import N

logger = logging.getLogger(__name__)


class GraphEnv(gym.Env):
    def __init__(self, state: N) -> None:
        super().__init__()
        self.state = state
        self.max_num_actions = state.max_num_actions

        self.observation_space: gym.Space = gym.spaces.Dict(
            {
                "action_mask": Box(
                    False, True, shape=(self.max_num_actions,), dtype=bool
                ),
                "action_observations": gym.spaces.Tuple(
                    (self.state.observation_space,) * self.max_num_actions
                ),
                "state_observation": self.state.observation_space,
            }
        )

    def reset(self) -> Dict[str, np.ndarray]:
        self.state = self.state.get_root()
        return self.make_observation()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        assert action < len(
            self.state.next_actions
        ), f"Action {action} outside the action space of state {self.state}: "
        "{len(self.state.next_actions)} max actions"

        # Move the state to the next action
        self.state = self.state.next_actions[action]

        result = (
            self.make_observation(),
            self.state.reward(),
            self.state.terminal,
            self.state.info,
        )
        logger.debug(
            f"{type(self)}: {result[1]} {result[2]}, {result[3]},"
            f" {len(self.state.next_actions)}"
        )
        return result

    def make_observation(self) -> Dict[str, np.ndarray]:
        action_mask = [False] * self.max_num_actions
        action_observations = [self.state.null_observation] * self.max_num_actions

        for i, successor in enumerate(self.state.next_actions):
            if i >= self.max_num_actions:
                logger.debug("state {self.state} exceeds maximum number of actions")

            action_mask[i] = True
            action_observations[i] = successor.observation

        return {
            "action_mask": np.array(action_mask, dtype=bool),
            "action_observations": tuple(action_observations),
            "state_observation": self.state.observation,
        }
