from typing import Dict, Sequence, List, Tuple

import gym
import numpy as np
import networkx as nx

from graphenv import tf
from graphenv.vertex import Vertex

layers = tf.keras.layers


class TSPState(Vertex):
    def __init__(
        self,
        G: nx.Graph,
        tour: List[int] = [],
    ) -> None:
        """Create a TSP vertex that defines the graph search problem.

        Args:
            G: A fully connected networkx graph.
            tour: A list of nodes in visitation order that led to this 
                state. Defaults to [].
        """    

        super().__init__()
        
        self.G = G
        self.num_nodes = self.G.number_of_nodes()
        self.tour = tour


    @property
    def observation_space(self) -> gym.spaces.Dict:
        """Returns the graph env's observation space.

        Returns:
            Dict observation space.
        """        
        return gym.spaces.Dict(
            {
                "node_obs": gym.spaces.Box(low=0., high=self.num_nodes - 1, 
                    shape=(1,), dtype=np.float),
            }
        )


    @property
    def root(self) -> "TSPState":
        """Returns the root node of the graph env.

        Returns:
            Node with node 0 as the starting point of the tour.
        """        
        return self.new(self.G, [0])


    @property
    def reward(self) -> float:
        """Returns the graph env reward.

        Returns:
            Negative distance between last two nodes in the tour.
        """        
        
        if len(self.tour) < 2:
            # First node in the tour does not have a reward associatd with it.
            rew = 0.
        else:        
            # Otherwise, reward is negative distance between last two nodes.
            src, dst = self.tour[-2:]
            rew = -self.G[src][dst]["weight"]

        return rew


    def new(self, G: nx.Graph, tour: List[int]):
        """Convenience function for duplicating the existing node

        Args:
            G:  Networkx graph.
            tour: List of visited nodes.

        Returns:
            New TSP state.
        """        
        return TSPState(G, tour)


    @property
    def info(self) -> Dict:
        return {}


    def _get_next_actions(self) -> Sequence["TSPState"]:
        """Yields a sequence of TSPState instances associated with the next
        accessible nodes.

        Yields:
            New instance of the TSPState with the next node added to 
            the tour.
        """
        G = self.G
        cur_node = self.tour[-1]
        
        # Look at neighbors not already on the path.
        nbrs = [n for n in G.neighbors(cur_node) if n not in self.tour]

        # Go back to the first node if we've visited every other already.
        if len(nbrs) == 0 and len(self.tour) == self.num_nodes:
            nbrs = [self.tour[0]]

        # Conditions for completing the circuit.
        if len(nbrs) == 0 and len(self.tour) == self.num_nodes + 1:
            nbrs = []

        # Loop over the neighbors and update paths.
        for nbr in nbrs:

            # Update the node path with next node.
            tour = self.tour.copy()
            tour.append(nbr)

            yield self.new(self.G, tour)


    def _make_observation(self) -> Dict[str, np.ndarray]:
        """Return an observation.  The dict returned here needs to match 
        both the self.observation_space in this class, as well as the input
        layer in tsp_model.TSPModel

        Returns:
            Observation dict.  We define the node_obs to be the degree of the 
            current node.  This is a placeholder for a more meaningful feature!
        """        
        return {
            "node_obs": np.array([self.G.degree[self.tour[-1]]], dtype=np.float)
        }
