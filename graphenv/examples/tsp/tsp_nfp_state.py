from math import sqrt
from typing import Dict, List

import gym
import networkx as nx
import numpy as np
from graphenv.examples.tsp.tsp_preprocessor import TSPPreprocessor
from graphenv.examples.tsp.tsp_state import TSPState


class TSPNFPState(TSPState):
    def __init__(self, G: nx.Graph, tour: List[int] = [0]) -> None:
        super().__init__(G, tour)
        self.preprocessor = TSPPreprocessor()
        self.num_edges = self.num_nodes ** 2 - self.num_nodes

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "current_node": gym.spaces.Box(
                    low=0,
                    high=self.num_nodes,
                    shape=(),
                    dtype=int,
                ),
                "node_visited": gym.spaces.Box(
                    low=0,
                    high=2,
                    shape=(self.num_nodes,),
                    dtype=int,
                ),
                "edge_weights": gym.spaces.Box(
                    low=0,
                    high=sqrt(2),
                    shape=(self.num_edges,),
                    dtype=float,
                ),
                "connectivity": gym.spaces.Box(
                    low=0,
                    high=self.num_nodes,
                    shape=(self.num_edges, 2),
                    dtype=int,
                ),
            }
        )

    def _make_observation(self) -> Dict[str, np.ndarray]:
        """Return an observation.  The dict returned here needs to match
        both the self.observation_space in this class, as well as the input
        layer in tsp_model.TSPModel

        Returns:
            Observation dict.
        """
        return self.preprocessor(self.G, self.tour)
