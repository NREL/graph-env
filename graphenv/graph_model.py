from typing import Dict, List, Tuple
import gym

from ray.rllib.models.tf import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from abc import abstractmethod

tf1, tf, tfv = try_import_tf()


class GraphModel(TFModelV2):
    '''
    Defines a RLLib TFModelV2 compatible model for using RL algorithms on a
    GraphEnv.
    '''

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
        *args,
        **kwargs,
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name,
            *args, **kwargs)


        self.state_value = None
        self.action_values = None
        self.action_weights = None
        self.action_mask = None

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

        action_values_shape = tf.shape(action_mask)  # batch, num actions
        action_mask = action_mask[:, 1:]  # trim off current state


        

        # flatten action observations into a single dict with tensors like:
        # [(batch 0, action 0), (b0,a1), ..., (b1,a0), ...])
        flat_batch_size = action_values_shape[0] * action_values_shape[1]
        flat_observations = {
            key: tf.reshape(
                value,
                (
                    flat_batch_size,
                    tf.shape(value)[1] // action_values_shape[1],
                    *value.shape[2:],
                ),
            )
            for key, value in action_observations.items()
        }
        flat_action_values, flat_action_weights = tuple(
            self.forward_vertex(flat_observations)
        )

        action_values = tf.reshape(flat_action_values, action_values_shape)
        self.state_value = action_values[:, 0]
        action_values = tf.where(
            action_mask, action_values[:, 1:], action_values.dtype.min)
        self.action_values = action_values

        action_weights = tf.reshape(flat_action_weights, action_values_shape)
        action_weights = tf.where(
            action_mask, action_weights[:, 1:], action_weights.dtype.min)
        self.action_weights = action_weights

        self.total_value = self._forward_total_value()
        # tf.reduce_max(action_values, axis=1)

        return action_weights, state

    def value_function(self):
        return self.total_value

    def _forward_total_value(self):
        '''
        Forward method computing the value assesment of the current state,
        as returned by the value_function() method.

        The default implementation return the action value of the current state.

        Breaking this into a separate method allows subclasses to override the 
        state value assesment, for example with a Bellman backup returning
        the max over all successor states's values.
        '''
        return self.state_value

    @abstractmethod
    def forward_vertex(
        self, input_dict: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        '''
        Forward function returning a value and weight tensor for the verticies 
        observed via input_dict (a dict of tensors for each vertex property)
        '''
        pass
