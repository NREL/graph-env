from math import sqrt
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from graphenv.examples.tsp.tsp_preprocessor import TSPPreprocessor
from graphenv.examples.tsp.tsp_state import TSPState


class TSPNFPState(TSPState):
    def __init__(
        self,
        *args,
        graph_inputs: Optional[Dict] = None,
        max_num_neighbors: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if graph_inputs is None:
            graph_inputs = TSPPreprocessor(max_num_neighbors=max_num_neighbors)(self.G)
        self.graph_inputs = graph_inputs
        self.max_num_neighbors = max_num_neighbors
        self.num_edges = len(graph_inputs["edge_weights"])

    def new(self, *args, new_graph=False, **kwargs):
        graph_inputs = None if new_graph else self.graph_inputs
        return super().new(
            *args,
            graph_inputs=graph_inputs,
            max_num_neighbors=self.max_num_neighbors,
            new_graph=new_graph,
            **kwargs
        )

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
                "distance": gym.spaces.Box(
                    low=0,
                    high=sqrt(2),
                    shape=(),
                    dtype=float,
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
        outputs = dict(self.graph_inputs)  # not sure the shallow copy is necessary

        node_visited = np.ones(self.num_nodes, dtype=np.int64)
        node_visited[self.tour] += 1

        if len(self.tour) > 1:
            distance = self.G.get_edge_data(self.tour[-2], self.tour[-1])["weight"]
        else:
            # First node
            distance = 0.0

        outputs.update(
            {
                "current_node": self.tour[-1],
                "distance": distance,
                "node_visited": node_visited,
            }
        )
        return outputs
