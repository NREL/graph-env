import random
from typing import Dict, Optional, Sequence

import gym
import numpy as np
from graphenv.vertex import Vertex
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
layers = tf.keras.layers


class HallwayState(Vertex):
    def __init__(
        self,
        size: int,
        max_steps: Optional[int] = None,
        position: int = 0,
        episode_steps: int = 0,
    ) -> None:
        super().__init__(max_num_actions=2)
        self.size = size
        self.max_steps = max_steps if max_steps is not None else np.inf
        self.position = position
        self.episode_steps = episode_steps

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=np.array([0]), high=np.array([self.size]), dtype=int
                ),
            }
        )

    @property
    def root(self) -> "HallwayState":
        return self.new(0, 0)

    @property
    def reward(self) -> float:
        return random.random() * 2 if self.position >= self.size else -0.1

    def new(self, position: int, episode_steps: int):
        """Convenience function for duplicating the existing node"""
        return HallwayState(self.size, self.max_steps, position, episode_steps)

    @property
    def info(self) -> Dict:
        info = super().info
        info["position"] = self.position
        return info

    def _get_next_actions(self) -> Sequence["HallwayState"]:
        if (self.position < self.size) & (self.episode_steps < self.max_steps):
            if self.position > 0:  # Stop the hallway from going negative
                yield self.new(self.position - 1, self.episode_steps + 1)
            yield self.new(self.position + 1, self.episode_steps + 1)

    def _make_observation(self) -> Dict[str, np.ndarray]:
        return {
            "position": np.array([self.position], dtype=int),
        }
