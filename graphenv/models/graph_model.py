from abc import abstractmethod
from typing import Dict, List, Tuple

import gym
import tensorflow as tf
from ray.rllib.models.tf import TFModelV2


class GraphGymModel(TFModelV2):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.action_values = None

    @abstractmethod
    def forward_per_action(self, input_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
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

        # Do we expect this wouldn't be the case?
        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)

        action_observations = observation["action_observations"]
        state_observation = observation["state_observation"]

        action_mask_shape = tf.shape(action_mask)  # batch size, num_actions

        flat_observations = {}
        for key in state_observation.keys():
            action_observations_sublist = [state_observation[key]] + [
                action_observation[key] for action_observation in action_observations
            ]
            stacked_observations = tf.stack(action_observations_sublist, axis=1)
            stacked_shape = tf.shape(
                stacked_observations
            )  # batch size, feature sizes ...
            flat_shape = tf.concat(
                [
                    tf.reshape(stacked_shape[0] * stacked_shape[1], (1,)),
                    stacked_shape[2:],
                ],
                axis=0,
            )
            flat_observations[key] = tf.reshape(stacked_observations, flat_shape)

        # run flattened action observations through the per action model to evaluate each action
        # flat_values, flat_weights = tuple(self.per_action_model.forward(flat_observations))
        flat_values, flat_weights = tuple(self.forward_per_action(flat_observations))
        composite_shape = tf.stack(
            [action_mask_shape[0], action_mask_shape[1] + 1], axis=0
        )
        action_weights = tf.where(
            action_mask,
            tf.reshape(flat_weights, composite_shape)[:, 1:],
            flat_weights.dtype.min,
        )

        self.action_values = tf.reshape(flat_values, composite_shape)[:, 0]
        return action_weights, state

    def value_function(self):
        return self.action_values
