from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

import networkx as nx
    

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


def plot_network(G, path: list = None) -> Tuple[any, any]:
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
        pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
    else:
        pos = [G.nodes[n]["pos"] for n in G.nodes]
    
    _ = nx.draw_networkx_nodes(G, pos, node_size=200)
    _ = nx.draw_networkx_labels(G, pos, font_size=12, font_color="white")
    
    if path is None:
        _ = nx.draw_networkx_edges(G, pos, edgelist=list(G.edges), width=1)
    else:
        _ = nx.draw_networkx_edges(G, pos, edgelist=list(G.edges), width=.1)
        edgelist = [(path[i+1], path[i]) for i in range(len(path)-1)]
        _ = nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=1)
        
    return fig, ax
