import logging
import math
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import gym
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.utils.typing import TensorStructType

from graphenv import tf

logger = logging.getLogger(__file__)


class GraphModel:
    """Defines a RLLib compatible model for using RL algorithms on a GraphEnv.

    Args:
        obs_space: The observation space to use.
        action_space: The action space to use.
        num_outputs: The number of scalar outputs per state to produce.
        model_config: Config forwarded to TFModelV2.__init()__.
        name: Config forwarded to TFModelV2.__init()__.
        action_mask_key: Key used to retrieve the action mask from the observation
            space dictionary. Defaults to "action_mask".
        vertex_observation_key: Key used to retrieve the per-action vertex
            observations from the observation space dictionary. Defaults to
            "vertex_observations".

    """

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
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            *args,
            **kwargs,
        )
        self.current_vertex_value = None
        self.action_values = None
        self.current_vertex_weight = None
        self.action_weights = None
        self.num_outputs = num_outputs
        logger.debug(f"num_outputs: {num_outputs}")

    def forward(
        self,
        input_dict: Dict[str, tf.Tensor],
        state: List[tf.Tensor],
        seq_lens: tf.Tensor,
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Tensorflow/Keras style forward method. Sets up the computation graph used by
        this model.

        Args:
            input_dict: Observation input to the model. Consists of a dictionary
                including key 'obs' which stores the raw observation from the process.
            state: Tensor of current states. Passes through this function untouched.
            seq_lens: Unused. Required by API.

        Returns:
            (action weights tensor, state)
        """

        flattened_observations = _stack_batch_dim(input_dict["obs"], tf)
        flat_values, flat_weights = self.forward_vertex(flattened_observations)

        # Create the action mask array from the `lengths` keyword
        mask = _create_action_mask(input_dict["obs"], tf)

        # mask out invalid children and get current vertex value
        self.current_vertex_value, _ = _mask_and_split_values(flat_values, mask, tf)
        _, action_weights = _mask_and_split_values(flat_weights, mask, tf)

        self.total_value = self._forward_total_value()
        return action_weights, state

    def value_function(self):
        """

        Returns:
            A tensor of current state values.
        """
        return self.total_value

    @abstractmethod
    def forward_vertex(
        self,
        input_dict,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward function returning a value and weight tensor for the verticies
        observed via input_dict (a dict of tensors for each vertex property)

        Args:
            input_dict: per-vertex observations

        Returns:
            (value tensor, weight tensor) for the given observations
        """
        pass

    def _forward_total_value(self):
        """Forward method computing the value assesment of the current state,
        as returned by the value_function() method.

        The default implementation return the action value of the current state.

        Breaking this into a separate method allows subclasses to override the
        state value assesment, for example with a Bellman backup returning
        the max over all successor states's values.

        Returns:
            current value tensor
        """
        # print("value: ", self.current_vertex_value)
        return self.current_vertex_value


def _stack_batch_dim(obs: TensorStructType, tensorlib: Any = tf):
    if isinstance(obs, dict):
        return {k: _stack_batch_dim(v, tensorlib) for k, v in obs.items()}

    elif isinstance(obs, tuple):
        return tuple(_stack_batch_dim(u, tensorlib) for u in obs)

    elif isinstance(obs, RepeatedValues):
        return _stack_batch_dim(obs.values, tensorlib)

    else:
        if tensorlib == tf:

            def get_value(v):
                if v is None:
                    return -1
                elif isinstance(v, int):
                    return v
                elif v.value is None:
                    return -1
                else:
                    return v.value

            batch_dims = [get_value(v) for v in obs.shape[:2]]
        else:
            batch_dims = list(obs.shape[:2])

        flat_batch_dim = math.prod(batch_dims)
        return tensorlib.reshape(obs, [flat_batch_dim] + list(obs.shape[2:]))


def _mask_and_split_values(values, action_mask, tensorlib: Any = tf):
    """Returns the value for the current vertex (index 0 of values),
    and the masked values of the action verticies.
    Args:
        values: Tensor to apply the action mask to.
    Returns:
        (a current state value tensor, a masked action values tensor)
    """

    if tensorlib == tf:
        values = tf.reshape(values, tf.shape(action_mask))
        current_value = values[:, 0]
        masked_action_values = tf.where(
            action_mask[:, 1:], values[:, 1:], values.dtype.min
        )
    else:
        raise NotImplementedError

    return current_value, masked_action_values


def _create_action_mask(obs, tensorlib: Any = tf):
    if tensorlib == tf:
        row_lengths = tf.cast(obs.lengths, tf.int32)
        num_elements = tf.reduce_sum(row_lengths)
        action_mask = tf.RaggedTensor.from_row_lengths(
            tf.ones(num_elements, dtype=tf.bool),
            row_lengths,
        ).to_tensor(shape=(None, obs.max_len))
    else:
        raise NotImplementedError

    return action_mask
