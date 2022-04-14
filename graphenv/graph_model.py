import logging
from abc import abstractmethod
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import gym
# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel

import graphenv.space_util as space_util
from graphenv import tf

logger = logging.getLogger(__file__)


# Type defining the contents of vertex observations as passed to forward()
GraphModelObservation = Union[
    tf.Tensor,
    Iterable["GraphModelObservation"],
    Mapping[str, "GraphModelObservation"],
]


class GraphModel:
    """
    Defines a RLLib compatible model for using RL algorithms on a
    GraphEnv.

    Attributes:
        _action_mask_key observation : space key for the action mask
        _vertex_observation_key : observation space key for the vertex observations
        action_mask : bool tensor of valid next actions
        current_vertex_value : value of current vertex
        action_values : values of each action vertex
        current_vertex_weight : weight of current vertex
        action_weights : action weights of each action vertex
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
        *args,
        action_mask_key: str = "action_mask",
        vertex_observation_key: str = "vertex_observations",
        **kwargs,
    ):
        """Initializes a GraphModel instance.

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

        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            *args,
            **kwargs,
        )
        self._action_mask_key = action_mask_key
        self._vertex_observation_key = vertex_observation_key
        self.action_mask = None
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
        # Extract the available actions tensor from the observation.
        observation = input_dict["obs"]

        vertex_observations = observation[self._vertex_observation_key]
        flattened_observations = space_util.flatten_first_dim(vertex_observations)

        # flat_values is structured like this: (vertex values, vertex weights)
        flat_values = self.forward_vertex(flattened_observations)

        action_mask = observation[self._action_mask_key]

        # Ray likes to make bool arrays into floats, so we undo it here.
        if action_mask.dtype != tf.dtypes.bool:
            action_mask = tf.equal(action_mask, 1.0)

        # mask out invalid actions and get current vertex value
        def mask_values(values):
            """Returns the value for the current vertex (index 0 of values),
            and the masked values of the action verticies.

            Args:
                values: Tensor to apply the action mask to.

            Returns:
                (a current state value tensor, a masked action values tensor)
            """

            values = tf.reshape(values, tf.shape(action_mask))
            current_value = values[:, 0]
            masked_action_values = tf.where(
                action_mask[:, 1:], values[:, 1:], values.dtype.min
            )
            return current_value, masked_action_values

        (self.current_vertex_value, self.action_values), (
            self.current_vertex_weight,
            self.action_weights,
        ) = tuple((mask_values(v) for v in flat_values))

        self.total_value = self._forward_total_value()
        return self.action_weights, state

    def value_function(self):
        """

        Returns:
            A tensor of current state values.
        """
        return self.total_value

    @abstractmethod
    def forward_vertex(
        self,
        input_dict: GraphModelObservation,
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
        return self.current_vertex_value
