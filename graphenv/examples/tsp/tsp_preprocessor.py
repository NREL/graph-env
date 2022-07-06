from typing import Dict, Optional

import networkx as nx
import nfp
import numpy as np


class TSPPreprocessor(nfp.Preprocessor):
    def __init__(self, max_num_neighbors: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_neighbors = max_num_neighbors

    def create_nx_graph(self, G: nx.Graph) -> nx.DiGraph:
        dG = G.to_directed()
        if self.max_num_neighbors:
            edges_to_keep = []
            for node in dG.nodes:
                for edge in sorted(
                    dG.out_edges(node, data=True), key=lambda x: x[2]["weight"]
                )[: self.max_num_neighbors]:
                    edges_to_keep += [(edge[0], edge[1])]

            dG = dG.edge_subgraph(edges_to_keep)

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
        return {}

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {}

    @property
    def output_signature(self):
        pass

    @property
    def padding_values(self):
        pass

    @property
    def tfrecord_features(self):
        pass
