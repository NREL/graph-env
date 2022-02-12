from typing import Dict, Sequence

import gym
import numpy as np
import tensorflow as tf
from graphenv.models.graph_model import GraphGymModel
from graphenv.node import Node
from tensorflow.keras import layers


class Hallway(Node):
    def __init__(
        self, size: int, max_steps: int, position: int = 0, episode_steps: int = 0
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
                    low=np.array([0]), high=np.array([self.size]), dtype=int
                ),
                "steps": gym.spaces.Box(
                    low=np.array([0]), high=np.array([self.max_steps]), dtype=int
                ),
            }
        )

    @property
    def null_observation(self) -> Dict[str, np.ndarray]:
        return {"position": np.array([0]), "steps": np.array([0])}

    def make_observation(self) -> Dict[str, np.ndarray]:
        return {
            "position": np.array([self.position]),
            "steps": np.array([self.episode_steps]),
        }

    def get_root(self) -> "Hallway":
        return self.new(0, 0)

    def reward(self) -> float:
        return float(self.position - self.size) if self.terminal else -1.0

    def new(self, position: int, episode_steps: int):
        """Convenience function for duplicating the existing node"""
        return Hallway(self.size, self.max_steps, position, episode_steps)

    @property
    def info(self) -> Dict:
        info = super().info
        info["position"] = self.position
        info["episode_steps"] = self.episode_steps
        return info


class HallwayModel(GraphGymModel):
    def __init__(
        self,
        *args,
        embedding_dim: int = 16,
        size: int = 5,
        max_steps: int = 10,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        self.pos_embedding = layers.Embedding(size, embedding_dim)
        self.steps_embedding = layers.Embedding(max_steps, embedding_dim)
        self.output_dense = layers.Dense(1)

    def forward_per_action(self, input_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        out = self.pos_embedding(input_dict["position"])
        out += self.steps_embedding(input_dict["steps"])
        return self.output_dense(out)
