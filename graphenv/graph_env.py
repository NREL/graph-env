import logging
from typing import Dict, Tuple

import gym
import numpy as np

from graphenv.vertex import N

logger = logging.getLogger(__name__)


class GraphEnv(gym.Env):
    """
    Defines an OpenAI Gym Env for traversing a graph using the current vertex
    as the state, and the successor verticies as actions.

    GraphEnv uses composition to supply the per-vertex model of type Vertex,
    which defines the graph via it's get_next_actions() method.
    """

    def __init__(self, state: N) -> None:
        super().__init__()
        logger.debug("GraphEnv init")
        self.state = state
        self.max_num_actions = state.max_num_actions
        num_actions = 1 + self.max_num_actions
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(
                    False, True, shape=(num_actions,), dtype=bool
                ),
                "action_observations": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            low=np.repeat(value.low, num_actions, axis=0),
                            high=np.repeat(value.high, num_actions, axis=0),
                            shape=(num_actions * value.shape[0], *value.shape[1:]),
                            dtype=value.dtype,
                        )
                        for key, value in self.state.observation_space.spaces.items()
                    }
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(self.max_num_actions)

    def reset(self) -> Dict[str, np.ndarray]:
        self.state = self.state.root
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
            self.state.reward,
            self.state.terminal,
            self.state.info,
        )
        logger.debug(
            f"{type(self)}: {result[1]} {result[2]}, {result[3]},"
            f" {len(self.state.next_actions)}"
        )
        return result

    def make_observation(self) -> Dict[str, any]:
        """
        Makes an observation for this state which includes observations of
        each possible action, and the current state.

        Expects the action observations to all be Dicts with the same keys.

        Returns a column-oriented representation, a Dict with keys matching
        the action observation keys, and values that are the current state
        and every action's values for that key concatenated into numpy arrays.

        The current state is the 0th entry in these arrays, and the actions
        are offset by one index to accomodate that.
        """

        num_actions = 1 + self.max_num_actions
        action_mask = np.zeros(num_actions, dtype=bool)
        action_observations = [self.state.observation] * num_actions
        for i, successor in enumerate(self.state.next_actions):
            action_observations[i + 1] = successor.observation
            action_mask[i + 1] = True

        flat_action_observations = {
            k: np.concatenate([o[k] for o in action_observations], axis=0)
            for k in action_observations[0].keys()
        }

        return {
            "action_mask": action_mask,
            "action_observations": flat_action_observations,
        }
