from typing import Dict, List

import networkx as nx
import nfp
import numpy as np
from graphenv import tf


class TSPPreprocessor(nfp.Preprocessor):
    def create_nx_graph(self, G: nx.Graph, tour: List[int]) -> nx.DiGraph:
        dG = G.to_directed()
        dG.graph["current_node"] = tour[-1]
        for node in dG.nodes:
            dG.nodes[node]["visited"] = True if node in tour else False
        return dG

    def get_edge_features(
        self, edge_data: list, max_num_edges: int
    ) -> Dict[str, np.ndarray]:
        edge_weights = np.empty(max_num_edges)
        edge_weights[:] = np.nan
        for n, (start, end, data) in enumerate(edge_data):
            edge_weights[n] = data["weight"]

        return {"edge_weights": edge_weights}

    def get_node_features(
        self, node_data: list, max_num_nodes: int
    ) -> Dict[str, np.ndarray]:
        visited = np.zeros(max_num_nodes, dtype=int)
        for n, data in node_data:
            visited[n] = data["visited"] + 1  # 1 for not visted, 2 for visited
        return {"node_visited": visited}

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {"current_node": np.asarray(graph_data["current_node"])}

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        return {
            "current_node": tf.TensorSpec(shape=(), dtype=tf.int32),
            "edge_weights": tf.TensorSpec(shape=(None,), dtype=float),
            "node_visited": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "connectivity": tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
        }

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        """Defaults to zero for each output"""
        return {
            "current_node": tf.constant(0, dtype=tf.int32),
            "edge_weights": tf.constant(np.nan, dtype=float),
            "node_visited": tf.constant(0, dtype=tf.int32),
            "connectivity": tf.constant(0, dtype=tf.int32),
        }

    @property
    def tfrecord_features(self):
        pass
