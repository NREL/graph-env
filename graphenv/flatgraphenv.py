from typing import Dict

import gym
import numpy as np

from graphenv.graphenv import GraphEnv


class FlatGraphEnv(GraphEnv):
    """Version of the GraphEnv environment that flattens observations to a single vector
    per dictionary key, rather than a tuple of dictionaries. This makes the model
    training and inference faster"""

    @property
    def observation_space(self) -> gym.spaces.Dict:
        obs_space = super().observation_space
        obs_space["action_observations"] = gym.spaces.Dict(
            {
                key: gym.spaces.Box(
                    low=np.repeat(value.low, self.max_num_actions, axis=0),
                    high=np.repeat(value.high, self.max_num_actions, axis=0),
                    shape=(self.max_num_actions * value.shape[0], *value.shape[1:]),
                    dtype=value.dtype,
                )
                for key, value in self.state.observation_space.spaces.items()
            }
        )
        return obs_space

    def make_observation(self) -> Dict[str, any]:
        obs = super().make_observation()

        flat_action_observations = {}
        for key in obs["action_observations"][0].keys():
            action_observations_sublist = [
                action_observation[key]
                for action_observation in obs["action_observations"]
            ]
            flat_action_observations[key] = np.concatenate(
                action_observations_sublist, axis=0
            )

        obs["action_observations"] = flat_action_observations
        return obs
