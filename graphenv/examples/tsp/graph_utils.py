import random
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.approximation.traveling_salesman import (
    greedy_tsp,
    traveling_salesman_problem,
)
from scipy.spatial import distance_matrix


def make_complete_planar_graph(N, seed: int = None) -> nx.Graph:
    """Returns a fully connected graph with xy positions for each
    node and edge weights equal to pairwise distances.

    Args:
        N: Number of nodes in graph.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Networkx complete graph with Euclidean distance weights.
    """

    np.random.seed(seed)

    # Complete graph on points in xy-plane with pairwise distances as edge weights
    G = nx.complete_graph(N)

    pos = np.random.rand(N, 2)
    d = distance_matrix(pos, pos)

    for ei, ej in G.edges:
        G[ei][ej]["weight"] = d[ei][ej]

    for node in G.nodes:
        G.nodes[node]["pos"] = pos[node, :]

    return G


def plot_network(G, path: list = None, draw_all_edges=True) -> Tuple[any, any]:
    """Plots the network and a path if specified.

    Args:
        G: networkx graph.
        path: List of node indexes in a path. Defaults to None.

    Returns:
        (fig, ax) from plt.subplots
    """

    fig, ax = plt.subplots()

    # Use pos attribute if it exists
    if "pos" not in G.nodes[0]:
        pos = nx.spring_layout(
            G, seed=7
        )  # positions for all nodes - seed for reproducibility
    else:
        pos = [G.nodes[n]["pos"] for n in G.nodes]

    if not path:
        _ = nx.draw_networkx_nodes(G, pos, node_size=200)

    else:
        _ = nx.draw_networkx_nodes(G, pos, node_size=200, node_color="#808080")
        _ = nx.draw_networkx_nodes(
            G.subgraph(list(set(path))),
            pos,
            node_size=200,
        )

    _ = nx.draw_networkx_labels(G, pos, font_size=12, font_color="white")

    if path is None and draw_all_edges:
        _ = nx.draw_networkx_edges(G, pos, edgelist=list(G.edges), width=1)
    else:
        if draw_all_edges:
            _ = nx.draw_networkx_edges(G, pos, edgelist=list(G.edges), width=0.1)

        edgelist = [(path[i + 1], path[i]) for i in range(len(path) - 1)]
        _ = nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=1)

    return fig, ax


def random_tsp(G, weight="weight", source=None, seed=None):
    """Return a baseline cost cycle starting at `source` and its cost.
    Randomly chooses nodes from each position.

    Parameters:
        G: The Graph should be a complete weighted undirected graph.
        weight: string, optional (default="weight").
        source: node, optional (default: first node in list(G))
        seed: Optional(int) a random seed

    Returns:
        cycle: list of nodes

    """
    if seed is not None:
        random.seed(seed)

    # Check that G is a complete graph
    N = len(G) - 1
    # This check ignores selfloops which is what we want here.
    if any(len(nbrdict) - (n in nbrdict) != N for n, nbrdict in G.adj.items()):
        raise nx.NetworkXError("G must be a complete graph.")

    if source is None:
        source = nx.utils.arbitrary_element(G)

    if G.number_of_nodes() == 2:
        neighbor = next(G.neighbors(source))
        return [source, neighbor, source]

    nodeset = set(G)
    nodeset.remove(source)
    cycle = [source]
    while nodeset:
        next_node = random.choice(list(nodeset))
        cycle.append(next_node)
        nodeset.remove(next_node)
    cycle.append(cycle[0])
    return cycle


def calc_greedy_dist(G: nx.Graph) -> float:
    """Calculate the distance for a greedy search tour over the given graph.
    Parameters:
        G: The Graph should be a complete weighted undirected graph.

    Returns:
        dist: a positive distance for the resulting search
    """
    path = traveling_salesman_problem(G, cycle=True, method=greedy_tsp)
    return sum(
        [G[path[i]][path[i + 1]]["weight"] for i in range(0, G.number_of_nodes())]
    )
