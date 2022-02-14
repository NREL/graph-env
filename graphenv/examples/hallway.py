import random
from typing import Dict, Optional, Sequence, Tuple

import gym
import numpy as np
from graphenv.node import Node
from ray.rllib.models.tf import TFModelV2
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
layers = tf.keras.layers


class Hallway(Node):
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
                # "steps": gym.spaces.Box(
                #     low=np.array([0]), high=np.array([self.max_steps]), dtype=int
                # ),
            }
        )

    @property
    def null_observation(self) -> Dict[str, np.ndarray]:
        return {
            "position": np.array([0]),
            # "steps": np.array([0])
        }

    def make_observation(self) -> Dict[str, np.ndarray]:
        return {
            "position": np.array([self.position]),
            # "steps": np.array([self.episode_steps]),
        }

    def get_root(self) -> "Hallway":
        return self.new(0, 0)

    def reward(self) -> float:
        return random.random() * 2 if self.position >= self.size else -0.1
        # return float(self.position - self.size) if self.terminal else -1.0

    def new(self, position: int, episode_steps: int):
        """Convenience function for duplicating the existing node"""
        return Hallway(self.size, self.max_steps, position, episode_steps)

    @property
    def info(self) -> Dict:
        info = super().info
        info["position"] = self.position
        # info["episode_steps"] = self.episode_steps
        return info


class HallwayModelMixin(TFModelV2):
    def __init__(
        self,
        *args,
        hidden_dim: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        position = layers.Input(shape=(1,), name="position", dtype=tf.float32)
        # steps = layers.Input(shape=(1,), name="steps", dtype=tf.float32)

        # pos_embedding_layer = layers.Embedding(
        #     size + 1, embedding_dim, name="position_embedding"
        # )
        # steps_embedding_layer = layers.Embedding(
        #     max_steps + 1, embedding_dim, name="size_embedding"
        # )

        hidden_layer = layers.Dense(hidden_dim, name="hidden_layer")
        action_value_output = layers.Dense(1, name="action_value_output")
        action_weight_output = layers.Dense(1, name="action_weight_output")

        # out = layers.Concatenate()([position, steps])
        out = hidden_layer(position)
        action_values = action_value_output(out)
        action_weights = action_weight_output(out)

        # pos_embedding = pos_embedding_layer(position)
        # steps_embedding = steps_embedding_layer(steps)
        # summed_embedding = layers.Add()([pos_embedding, steps_embedding])
        # action_values = action_value_output(summed_embedding)
        # action_weights = action_weight_output(summed_embedding)

        self.base_model = tf.keras.Model([position], [action_values, action_weights])

    def forward_per_action(
        self, input_dict: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.base_model(input_dict)
