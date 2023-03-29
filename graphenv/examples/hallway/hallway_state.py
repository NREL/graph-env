import random
from typing import Dict, Sequence

import gymnasium as gym
import numpy as np
from graphenv import tf
from graphenv.vertex import Vertex

layers = tf.keras.layers


class HallwayState(Vertex):
    """Example Vertex implementation of a simple hallway process graph.
    The hallway graph is a simple bidirectional chain of vertices. The root
    vertex is on one end of the chain and the terminal goal vertex is on the
    opposite end. The length is configurable.

    Args:
        corridor_length (int): length of the vertex chain
        cur_pos (int, optional): initial vertex index. Defaults to 0.
    """

    def __init__(
        self,
        corridor_length: int,
        cur_pos: int = 0,
    ) -> None:
        super().__init__()
        self.end_pos = corridor_length - 1
        self.cur_pos = cur_pos

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """HallwayStates are observed with a dictionary containing a single
        key, 'cur_pos', with an integer value between 0 and self.end_pos,
        indicating the index of the vertex.

        Returns:
            gym.spaces.Dict: The observation space for HallwayStates.
        """
        return gym.spaces.Dict(
            {
                "cur_pos": gym.spaces.Box(
                    low=np.array([0]), high=np.array([self.end_pos]), dtype=int
                ),
            }
        )

    @property
    def root(self) -> "HallwayState":
        """
        Returns:
            HallwayState: initial state (vertex at index 0)
        """
        return self.new(0)

    @property
    def reward(self) -> float:
        """The reward function for the HallwayState graph.

        Returns:
            float: random reward between 0 and 2 on the goal vertex, -0.1
                otherwise.
        """
        return random.random() * 2 if self.cur_pos >= self.end_pos else -0.1

    def new(self, cur_pos: int):
        """Convenience function for duplicating the existing node.

        Returns:
            HallwayState : a copy of this HallwayState.
        """
        return HallwayState(self.end_pos + 1, cur_pos)

    @property
    def info(self) -> Dict:
        """
        Debugging information compiled and returned by the environment step()
        method about vertices passed through or considered.

        Returns:
            Dict: Debugging information including the index of this vertex.
        """
        info = super().info
        info["cur_pos"] = self.cur_pos
        return info

    def _get_children(self) -> Sequence["HallwayState"]:
        """Gets child vertices of this vertex. Each vertex has both larger
        and smaller adjacent index vertices as children, except for the initial
        and goal vertices.

        Yields:
            HallwayState: Child vertices of this vertex.
        """
        if self.cur_pos < self.end_pos:
            if self.cur_pos > 0:  # Stop the hallway from going negative
                yield self.new(self.cur_pos - 1)
            yield self.new(self.cur_pos + 1)

    def _make_observation(self) -> Dict[str, np.ndarray]:
        """Makes an observation of this HallwayState vertex.

        Returns:
            Dict[str, np.ndarray]: dictionary containing the current position
            index under the key 'cur_pos'.
        """
        return {
            "cur_pos": np.array([self.cur_pos], dtype=int),
        }
