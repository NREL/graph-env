from typing import Dict, Tuple

from graphenv.graph_model import GraphModel
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
layers = tf.keras.layers


class HallwayModel(GraphModel):
    def __init__(
        self,
        *args,
        hidden_dim: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        position = layers.Input(shape=(1,), name="position", dtype=tf.float32)

        hidden_layer = layers.Dense(hidden_dim, name="hidden_layer")
        action_value_output = layers.Dense(
            1, name="action_value_output", bias_initializer="ones"
        )
        action_weight_output = layers.Dense(
            1, name="action_weight_output", bias_initializer="ones"
        )

        out = hidden_layer(position)
        action_values = action_value_output(out)
        action_weights = action_weight_output(out)

        self.base_model = tf.keras.Model([position], [action_values, action_weights])

    def forward_vertex(
        self, input_dict: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return tuple(self.base_model(input_dict))
