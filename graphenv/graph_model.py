from abc import abstractmethod
from typing import Dict, List, Tuple

import graphenv.space_util as space_util

import gym
from ray.rllib.models.tf import TFModelV2
from ray.rllib.utils.framework import try_import_tf


tf1, tf, tfv = try_import_tf()


class GraphModel(TFModelV2):
    """
    Defines a RLLib TFModelV2 compatible model for using RL algorithms on a
    GraphEnv.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
        *args,
        action_mask_key: str = 'action_mask',
        vertex_observation_key: str = 'vertex_observations',
        **kwargs,
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, *args, **kwargs
        )

        self._action_mask_key = action_mask_key
        self._vertex_observation_key = vertex_observation_key

        self.action_mask = None

        self.current_vertex_value = None
        self.action_values = None

        self.current_vertex_weight = None
        self.action_weights = None

    def forward(
        self,
        input_dict: Dict[str, tf.Tensor],
        state: List[tf.Tensor],
        seq_lens: tf.Tensor,
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:

        # Extract the available actions tensor from the observation.
        observation = input_dict['obs']

        vertex_observations = observation[self._vertex_observation_key]
        flattened_observations = \
            space_util.flatten_first_dim(vertex_observations)
        flat_action_values, flat_action_weights = \
            self.forward_vertex(flattened_observations)

        # Ray likes to make these bool arrays into floats for some unkown reason
        action_mask = observation[self._action_mask_key]
        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)

        # mask out invalid actions and get current vertex value
        self.current_vertex_value, self.action_values = \
            self._get_current_and_mask_action_values(
                action_mask, flat_action_values)
        
        self.current_vertex_weight, self.action_weights = \
            self._get_current_and_mask_action_values(
                action_mask, flat_action_weights)

        self.total_value = self._forward_total_value()

        return self.action_weights, state

    def value_function(self):
        return self.total_value

    @abstractmethod
    def forward_vertex(
        self, input_dict: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward function returning a value and weight tensor for the verticies
        observed via input_dict (a dict of tensors for each vertex property)
        """
        pass

    def _forward_total_value(self):
        """
        Forward method computing the value assesment of the current state,
        as returned by the value_function() method.

        The default implementation return the action value of the current state.

        Breaking this into a separate method allows subclasses to override the
        state value assesment, for example with a Bellman backup returning
        the max over all successor states's values.
        """
        return self.current_vertex_value

    def _get_current_and_mask_action_values(self, action_mask, values):
        """
        Returns the value for the current vertex (index 0 of values),
        and the masked values of the action verticies.
        """
        values = tf.reshape(values, tf.shape(action_mask))
        current_value = values[:, 0]
        masked_action_values = \
            tf.where(action_mask[:, 1:], values[:, 1:], values.dtype.min)
        return current_value, masked_action_values
