from typing import Tuple

from graphenv import tf
from graphenv.graph_model import GraphModel, GraphModelObservation

layers = tf.keras.layers


class HallwayModel(GraphModel):
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
        input_dict: GraphModelObservation,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return tuple(self.base_model(input_dict))
