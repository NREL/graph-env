from typing import Tuple

import nfp
from graphenv import tf
from graphenv.graph_model import GraphModel
from graphenv.graph_model_bellman_mixin import GraphModelBellmanMixin
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.typing import TensorStructType, TensorType

layers = tf.keras.layers


class BaseTSPGNNModel(GraphModel):
    def __init__(
        self,
        *args,
        num_messages: int = 3,
        embed_dim: int = 32,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.base_model = self._create_base_model(num_messages, embed_dim)

    @staticmethod
    def _create_base_model(
        num_messages: int = 3, embed_dim: int = 32
    ) -> tf.keras.Model:

        current_node = layers.Input(shape=[], dtype=tf.int32, name="current_node")
        node_visited = layers.Input(shape=[None], dtype=tf.int32, name="node_visited")
        edge_weights = layers.Input(shape=[None], dtype=tf.float32, name="edge_weights")
        distance = layers.Input(shape=[], dtype=tf.float32, name="distance")
        connectivity = layers.Input(
            shape=[None, 2], dtype=tf.int32, name="connectivity"
        )

        node_state = layers.Embedding(
            3, embed_dim, name="node_embedding", mask_zero=True
        )(node_visited)

        edge_state = nfp.layers.RBFExpansion(
            embed_dim, init_gap=1, init_max_distance=1
        )(edge_weights)

        for _ in range(num_messages):  # Do the message passing
            new_edge_state = nfp.EdgeUpdate()([node_state, edge_state, connectivity])
            edge_state = layers.Add()([edge_state, new_edge_state])

            new_node_state = nfp.NodeUpdate()([node_state, edge_state, connectivity])
            node_state = layers.Add()([node_state, new_node_state])

        current_node_embedding = nfp.layers.Gather()([node_state, current_node])
        current_node_embedding = layers.Flatten()(current_node_embedding)

        action_values = layers.Dense(
            1,
            name="action_value_output",
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1e-6, seed=None
            ),
        )(current_node_embedding)

        action_weights = layers.Dense(
            1,
            name="action_weight_output",
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=1e-6, seed=None
            ),
        )(current_node_embedding)

        reshaped_distance = layers.Reshape((1,))(distance)
        distance_values = layers.Dense(
            1,
            name="distance_values",
            kernel_initializer=tf.keras.initializers.Constant(-20),
        )(reshaped_distance)

        distance_weights = layers.Dense(
            1,
            name="distance__weights",
            kernel_initializer=tf.keras.initializers.Constant(-20),
        )(reshaped_distance)

        action_values = layers.Add()([distance_values, action_values])
        action_weights = layers.Add()([distance_weights, action_weights])

        return tf.keras.Model(
            [current_node, distance, node_visited, edge_weights, connectivity],
            [action_values, action_weights],
            name="policy_model",
        )

    def forward_vertex(
        self,
        input_dict: TensorStructType,
    ) -> Tuple[TensorType, TensorType]:
        return tuple(self.base_model(input_dict))


class TSPGNNModel(BaseTSPGNNModel, TFModelV2):
    pass


class TSPGNNQModel(BaseTSPGNNModel, DistributionalQTFModel):
    pass


class TSPGNNQModelBellman(
    GraphModelBellmanMixin, BaseTSPGNNModel, DistributionalQTFModel
):
    pass
