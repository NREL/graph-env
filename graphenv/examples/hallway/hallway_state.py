import random
from typing import Dict, Sequence

import gym
import numpy as np
from graphenv import tf
from graphenv.vertex import Vertex

layers = tf.keras.layers


class HallwayState(Vertex):
    def __init__(
        self,
        corridor_length: int,
        cur_pos: int = 0,
    ) -> None:
        super().__init__(max_num_actions=2)
        self.end_pos = corridor_length
        self.cur_pos = cur_pos

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "cur_pos": gym.spaces.Box(
                    low=np.array([0]), high=np.array([self.end_pos]), dtype=int
                ),
            }
        )

    @property
    def root(self) -> "HallwayState":
        return self.new(0)

    @property
    def reward(self) -> float:
        return random.random() * 2 if self.cur_pos >= self.end_pos else -0.1

    def new(self, cur_pos: int):
        """Convenience function for duplicating the existing node"""
        return HallwayState(self.end_pos, cur_pos)

    @property
    def info(self) -> Dict:
        info = super().info
        info["cur_pos"] = self.cur_pos
        return info

    def _get_next_actions(self) -> Sequence["HallwayState"]:
        if self.cur_pos < self.end_pos:
            if self.cur_pos > 0:  # Stop the hallway from going negative
                yield self.new(self.cur_pos - 1)
            yield self.new(self.cur_pos + 1)

    def _make_observation(self) -> Dict[str, np.ndarray]:
        return {
            "cur_pos": np.array([self.cur_pos], dtype=int),
        }
