from typing import Dict, List, Tuple
import gym

from ray.rllib.models.tf import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from abc import abstractmethod

tf1, tf, tfv = try_import_tf()


class GraphModel(TFModelV2):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name)
        self.action_values = None

    @abstractmethod
    def forward_per_action(
        self, input_dict: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """This makes the MRO important, but also makes the requirement for this model
        more obvious"""
        raise NotImplementedError(
            "You must implement a value function for a single action in a derived class"
        )

    def forward(
        self,
        input_dict: Dict[str, tf.Tensor],
        state: List[tf.Tensor],
        seq_lens: tf.Tensor,
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:

        # Extract the available actions tensor from the observation.
        observation = input_dict["obs"]
        action_mask = observation["action_mask"]
        action_observations = observation["action_observations"]

        # Ray likes to make these bool arrays into floats for some unkown reason
        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)

        action_mask_shape = tf.shape(action_mask)  # batch size, num_actions

        # flatten action observations into a single dict with tensors like:
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
        action_values = tf.where(
            action_mask, action_values, action_values.dtype.min)

        # TODO: optional direct assesment instead of bellman backup
        self.total_value = tf.reduce_max(action_values, axis=1)

        action_weights = tf.reshape(flat_action_weights, action_mask_shape)
        action_weights = tf.where(
            action_mask, action_weights, action_weights.dtype.min)
        action_weights = action_weights[:, 1:]  # trim off current state entry
        return action_weights, state

    def value_function(self):
        return self.total_value
