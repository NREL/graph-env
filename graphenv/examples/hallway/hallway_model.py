from typing import Tuple

from graphenv import tf
from graphenv.graph_model import GraphModel
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.typing import TensorStructType, TensorType

layers = tf.keras.layers


class BaseHallwayModel(GraphModel):
    """An example GraphModel implementation for the HallwayEnv and HallwayState
    Graph. Uses a dense fully connected Keras network.

    Args:
        hidden_dim (int, optional): The number of hidden layers to use. Defaults to 1.
    """

    # #: tf.keras.Model: The Keras model used to evaluate vertex observations
    # base_model: "tf.keras.Model"

    def __init__(
        self,
        *args,
        hidden_dim: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        cur_pos = layers.Input(shape=(1,), name="cur_pos", dtype=tf.float32)

        hidden_layer = layers.Dense(hidden_dim, name="hidden_layer")
        action_value_output = layers.Dense(
            1, name="action_value_output", bias_initializer="ones"
        )
        action_weight_output = layers.Dense(
            1, name="action_weight_output", bias_initializer="ones"
        )

        out = hidden_layer(cur_pos)
        action_values = action_value_output(out)
        action_weights = action_weight_output(out)

        self.base_model = tf.keras.Model([cur_pos], [action_values, action_weights])

    def forward_vertex(
        self,
        input_dict: TensorStructType,
    ) -> Tuple[TensorType, TensorType]:
        """Forward function computing the evaluation of vertex observations.

        Args:
            input_dict (TensorStructType): vertex observations

        Returns:
            Tuple[TensorType, TensorType]: Tensor of value and weights for each
                input observation.
        """
        return tuple(self.base_model(input_dict))


class HallwayQModel(BaseHallwayModel, DistributionalQTFModel):
    pass


class HallwayModel(BaseHallwayModel, TFModelV2):
    pass
