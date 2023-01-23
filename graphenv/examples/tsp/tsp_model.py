from typing import Tuple

from graphenv import tf
from graphenv.graph_model import GraphModel
from graphenv.graph_model_bellman_mixin import GraphModelBellmanMixin
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.typing import TensorStructType, TensorType

layers = tf.keras.layers


class BaseTSPModel(GraphModel):
    def __init__(
        self,
        *args,
        num_nodes: int,
        hidden_dim: int = 32,
        embed_dim: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_model = self._create_base_model(num_nodes, hidden_dim, embed_dim)

    @staticmethod
    def _create_base_model(
        num_nodes: int, hidden_dim: int = 32, embed_dim: int = 32
    ) -> tf.keras.Model:

        node_obs = layers.Input(shape=(2,), name="node_obs", dtype=tf.float32)
        node_idx = layers.Input(shape=(1,), name="node_idx", dtype=tf.int32)
        parent_dist = layers.Input(shape=(1,), name="parent_dist", dtype=tf.float32)
        nbr_dist = layers.Input(shape=(1,), name="nbr_dist", dtype=tf.float32)

        embed_layer = layers.Embedding(
            num_nodes, embed_dim, name="embed_layer", input_length=1
        )
        hidden_layer_1 = layers.Dense(
            hidden_dim, name="hidden_layer_1", activation="relu"
        )
        hidden_layer_2 = layers.Dense(
            hidden_dim, name="hidden_layer_2", activation="linear"
        )
        action_value_output = layers.Dense(
            1, name="action_value_output", bias_initializer="ones"
        )
        action_weight_output = layers.Dense(
            1, name="action_weight_output", bias_initializer="ones"
        )

        # Process the positional node data.  Here we need to expand the
        # middle axis to match the embedding output dimension.
        x = layers.Concatenate(axis=-1)([node_obs, parent_dist, nbr_dist])
        hidden = layers.Reshape((1, hidden_dim))(hidden_layer_1(x))

        # Process the embedding.
        embed = embed_layer(node_idx)

        # Concatenate and flatten for dense output layers.
        out = layers.Concatenate(axis=-1)([hidden, embed])
        out = layers.Flatten()(out)
        out = hidden_layer_2(out)

        # Action values and weights for RLLib algorithms
        action_values = action_value_output(out)
        action_weights = action_weight_output(out)

        return tf.keras.Model(
            [node_obs, node_idx, parent_dist, nbr_dist], [action_values, action_weights]
        )

    def forward_vertex(
        self,
        input_dict: TensorStructType,
    ) -> Tuple[TensorType, TensorType]:
        return tuple(self.base_model(input_dict))


class TSPModel(BaseTSPModel, TFModelV2):
    pass


class TSPQModel(BaseTSPModel, DistributionalQTFModel):
    pass


class TSPQModelBellman(GraphModelBellmanMixin, BaseTSPModel, DistributionalQTFModel):
    pass
