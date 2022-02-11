from typing import Dict, Sequence

import gym
import numpy as np
from graphenv.node import Node


class Hallway(Node):
    def __init__(
        self, size: int, max_steps: int, position: int, episode_steps: int
    ) -> None:
        super().__init__(max_num_actions=2)
        self.size = size
        self.max_steps = max_steps
        self.position = position
        self.episode_steps = episode_steps

    def get_next_actions(self) -> Sequence["Hallway"]:
        if (self.position < self.size) & (self.episode_steps < self.max_steps):
            if self.position > 0:  # Stop the hallway from going negative
                yield self.new(self.position - 1, self.episode_steps + 1)
            yield self.new(self.position + 1, self.episode_steps + 1)

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=np.array([0]), high=np.array([self.size]), dtype=np.int
                ),
                "steps": gym.spaces.Box(
                    low=np.array([0]), high=np.array([self.max_steps]), dtype=np.int
                ),
            }
        )

    @property
    def null_observation(self) -> Dict[str, np.ndarray]:
        return {"position": np.array([-1]), "steps": np.array([-1])}

    def make_observation(self) -> Dict[str, np.ndarray]:
        return {
            "position": np.array([self.position]),
            "steps": np.array([self.episode_steps]),
        }

    def get_root(self) -> "Hallway":
        return self.new(0, 0)

    def reward(self) -> float:
        return float(self.position - self.size) if self.is_terminal() else -1.0

    def new(self, position: int, episode_steps: int):
        """Convenience function for duplicating the existing node"""
        return Hallway(self.size, self.max_steps, position, episode_steps)
