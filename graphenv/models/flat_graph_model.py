from typing import Dict, List, Tuple

from graphenv.models.graph_model import GraphEnvModel
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class FlatGraphEnvModel(GraphEnvModel):
    def forward(
        self,
        input_dict: Dict[str, tf.Tensor],
        state: List[tf.Tensor],
        seq_lens: tf.Tensor,
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:

        # Extract the available actions tensor from the observation.
        observation = input_dict["obs"]
        action_mask = observation["action_mask"]

        # Do we expect this wouldn't be the case?
        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)

        action_mask_shape = tf.shape(action_mask)  # batch size, num_actions
        action_observations = observation["action_observations"]
        # state_observation = observation["state_observation"]  # currently unused

        # flatten action observations into a single dict with tensors stacked like:
        # [(batch 0, action 0), (b0,a1), ..., (b1,a0), ...])
        flat_batch_size = action_mask_shape[0] * action_mask_shape[1]
        flat_observations = {
            key: tf.reshape(
                value,
                (
                    flat_batch_size,
                    tf.shape(value)[1] // action_mask_shape[1],
                    *value.shape[2:],
                ),
            )
            for key, value in action_observations.items()
        }
        flat_action_values, flat_action_weights = tuple(
            self.forward_per_action(flat_observations)
        )

        action_values = tf.reshape(flat_action_values, action_mask_shape)
        action_values = tf.where(action_mask, action_values, action_values.dtype.min)
        self.total_value = tf.reduce_max(action_values, axis=1)

        action_weights = tf.reshape(flat_action_weights, action_mask_shape)
        action_weights = tf.where(action_mask, action_weights, action_weights.dtype.min)
        return action_weights, state

    def value_function(self):
        # TODO: ideally this behavior would be identical to the GraphEnvModel's value
        # function, we should decide on one behavior and stick to it (or refactor such
        # that it's easy to alter both envs)
        return self.total_value
