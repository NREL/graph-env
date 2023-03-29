from typing import Any, Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import networkx as nx
import numpy as np
from graphenv import tf
from graphenv.examples.tsp.graph_utils import plot_network
from graphenv.vertex import Vertex

layers = tf.keras.layers


class TSPState(Vertex):
    """Create a TSP vertex that defines the graph search problem.

    Args:
        generator: a function that creates a networkx graph
        G: A fully connected networkx graph.
        tour: A list of nodes in visitation order that led to this
            state. Defaults to [0] which begins the tour at node 0.
    """

    def __init__(
        self,
        graph_generator: Callable[[], nx.Graph],
        G: Optional[nx.Graph] = None,
        tour: List[int] = [0],
    ) -> None:

        super().__init__()

        self.G = G if G is not None else graph_generator()
        self.num_nodes = self.G.number_of_nodes()
        self.tour = tour
        self.graph_generator = graph_generator

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """Returns the graph env's observation space.

        Returns:
            Dict observation space.
        """
        return gym.spaces.Dict(
            {
                "node_obs": gym.spaces.Box(
                    low=np.zeros(2), high=np.ones(2), dtype=float
                ),
                "node_idx": gym.spaces.Box(
                    low=0, high=self.num_nodes, shape=(1,), dtype=int
                ),
                "parent_dist": gym.spaces.Box(
                    low=0.0, high=np.sqrt(2), shape=(1,), dtype=float
                ),
                "nbr_dist": gym.spaces.Box(
                    low=0.0, high=np.sqrt(2), shape=(1,), dtype=float
                ),
            }
        )

    @property
    def root(self) -> "TSPState":
        """Returns the root node of the graph env.

        Returns:
            Node with node 0 as the starting point of the tour, and generates a new
            graph using the given constructor
        """

        return self.new([0], new_graph=True)

    @property
    def reward(self) -> float:
        """Returns the graph env reward.

        Returns:
            Negative distance between last two nodes in the tour.
        """

        if len(self.tour) == 1:
            # This should never be called
            rew = 0

        elif len(self.tour) >= 2:
            # Otherwise, reward is negative distance between last two nodes.
            src, dst = self.tour[-2:]
            rew = -self.G[src][dst]["weight"]

        else:
            raise RuntimeError(f"Invalid tour: {self.tour}")

        return rew

    def new(self, tour: List[int] = [0], new_graph=False, **kwargs):
        """Convenience function for duplicating the existing node.

        Args:
            G:  Networkx graph.
            tour: List of visited nodes.

        Returns:
            New TSP state.
        """
        G = self.G if not new_graph else self.graph_generator()
        return self.__class__(self.graph_generator, G=G, tour=tour, **kwargs)

    def render(self) -> Any:
        return plot_network(self.G, self.tour, draw_all_edges=False)

    @property
    def info(self) -> Dict:
        return {}

    def _get_children(self) -> Sequence["TSPState"]:
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

            yield self.new(tour)

    def _make_observation(self) -> Dict[str, np.ndarray]:
        """Return an observation.  The dict returned here needs to match
        both the self.observation_space in this class, as well as the input
        layer in tsp_model.TSPModel

        Returns:
            Observation dict.  We define the node_obs to be the degree of the
            current node.  This is a placeholder for a more meaningful feature!
        """

        cur_node = self.tour[-1]
        cur_pos = np.array(self.G.nodes[cur_node]["pos"], dtype=float).squeeze()

        # Compute distance to parent node, or 0 if this is the root.
        if len(self.tour) == 1:
            parent_dist = 0.0
        else:
            parent_dist = self.G[cur_node][self.tour[-2]]["weight"]

        # Get list of all neighbors that are unvisited.  If none, then the only
        # remaining neighbor is the root so dist is 0.
        nbrs = [n for n in self.G.neighbors(cur_node) if n not in self.tour]
        nbr_dist = 0.0
        if len(nbrs) > 0:
            nbr_dist = np.min([self.G[cur_node][n]["weight"] for n in nbrs])

        return {
            "node_obs": cur_pos,
            "node_idx": np.array([cur_node]),
            "parent_dist": np.array([parent_dist]),
            "nbr_dist": np.array([nbr_dist]),
        }
