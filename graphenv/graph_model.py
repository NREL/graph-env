import logging
from abc import abstractmethod
from typing import Dict, List, Tuple

import gym

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
        # flat_values is structured like this: (vertex values, vertex weights)
        # flat_values = self.forward_vertex(input_dict["obs_flat"])
        # print()
        # print("The unpacked input tensors:", input_dict["obs"], flush=True)
        # print()
        # print("Unbatched repeat dim", input_dict["obs"].unbatch_repeat_dim())
        # print()
        # print("outer repetition", input_dict["obs"].lengths)
        # print()
        # print("The flattened input tensors:", input_dict["obs_flat"], flush=True)

        unbatched = input_dict["obs"].unbatch_repeat_dim()
        if isinstance(unbatched, dict):
            unbatched = [
                {key: unbatched[key][i] for key in unbatched.keys()}
                for i in range(input_dict["obs"].max_len)
            ]

        current_vertex_inputs, *action_inputs = unbatched

        action_weights = []
        for i, input in enumerate(action_inputs):
            _, action_weight = self.forward_vertex(input)
            # print(f"length test {i + 1}: ", input_dict["obs"].lengths > i + 1)
            action_weight = tf.where(
                input_dict["obs"].lengths > i + 1,
                action_weight,
                action_weight.dtype.min,
            )
            action_weights += [action_weight]

        action_weights = tf.concat(action_weights, -1)
        # print("action weights: ", action_weights)

        self.current_vertex_value, _ = self.forward_vertex(current_vertex_inputs)
        self.current_vertex_value = tf.squeeze(self.current_vertex_value, [-1])
        self.total_value = self._forward_total_value()

        # _, action_weights = zip(
        #     *

        # print("action weights: ", action_weights)

        return action_weights, state

        # return self.forward_vertex(input_dict)

        # # mask out invalid children and get current vertex value
        # def mask_values(values):
        #     """Returns the value for the current vertex (index 0 of values),
        #     and the masked values of the action verticies.

        #     Args:
        #         values: Tensor to apply the action mask to.

        #     Returns:
        #         (a current state value tensor, a masked action values tensor)
        #     """

        #     values = tf.reshape(values, tf.shape(action_mask))
        #     current_value = values[:, 0]
        #     masked_action_values = tf.where(
        #         action_mask[:, 1:], values[:, 1:], values.dtype.min
        #     )
        #     return current_value, masked_action_values

        # (self.current_vertex_value, self.action_values), (
        #     self.current_vertex_weight,
        #     self.action_weights,
        # ) = tuple((mask_values(v) for v in flat_values))

        # self.total_value = self._forward_total_value()
        # return self.action_weights, state

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
