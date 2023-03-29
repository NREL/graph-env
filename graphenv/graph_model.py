import logging
from abc import abstractmethod
from typing import Dict, List, Tuple

import gymnasium as gym
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.utils.typing import TensorStructType, TensorType

from graphenv import tf, torch

logger = logging.getLogger(__file__)


class GraphModel:
    """Defines a RLLib compatible model for using RL algorithms on a GraphEnv.

    Args:
        obs_space: The observation space to use.
        action_space: The action space to use.
        num_outputs: The number of scalar outputs per state to produce.
        model_config: Config forwarded to TFModelV2.__init()__.
        name: Config forwarded to TFModelV2.__init()__.
    """

    _tensorlib = "tf"

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
        assert self._tensorlib is not None

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
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

        mask = _create_action_mask(input_dict["obs"], self._tensorlib)
        flattened_observations = _stack_batch_dim(
            input_dict["obs"], mask, self._tensorlib
        )
        flat_values, flat_weights = self.forward_vertex(flattened_observations)

        # mask out invalid children and get current vertex value
        self.current_vertex_value, _ = _mask_and_split_values(
            flat_values, input_dict["obs"], self._tensorlib
        )
        _, action_weights = _mask_and_split_values(
            flat_weights, input_dict["obs"], self._tensorlib
        )

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
    ) -> Tuple[TensorType, TensorType]:
        """Forward function returning a value and weight tensor for the vertices
        observed via input_dict (a dict of tensors for each vertex property)

        Args:
            input_dict: per-vertex observations

        Returns:
            (value tensor, weight tensor) for the given observations
        """
        pass

    def _forward_total_value(self):
        """Forward method computing the value assessment of the current state,
        as returned by the value_function() method.

        The default implementation return the action value of the current state.

        Breaking this into a separate method allows subclasses to override the
        state value assessment, for example with a Bellman backup returning
        the max over all successor states's values.

        Returns:
            current value tensor
        """
        return self.current_vertex_value


class TorchGraphModel(GraphModel):
    _tensorlib = "torch"


def _create_action_mask(obs: RepeatedValues, tensorlib: str = "tf") -> TensorType:
    """Create an action mask array of valid actions from a given RepeatedValues tensor.

    Args:
        obs (RepeatedValues): The input observations
        tensorlib (Any, optional): A reference to the current framework. Defaults to tf.

    Raises:
        NotImplementedError: if the given framework is not supported

    Returns:
        TensorType: The boolean mask for valid actions (includes the current state as
        the first index).
    """
    if tensorlib == "tf":
        # the "dummy batch" rllib provides to initialize the policy model is a matrix of
        # all zeros, which ends with a batch size of zero provided to the policy model.
        # We can assume that at least the input state is valid, and clip the row_lengths
        # vector to a minimum of 1 per (state, *actions) entry.
        row_lengths = tf.clip_by_value(tf.cast(obs.lengths, tf.int32), 1, tf.int32.max)
        num_elements = tf.reduce_sum(row_lengths)
        action_mask = tf.RaggedTensor.from_row_lengths(
            tf.ones(num_elements, dtype=tf.bool),
            row_lengths,
        ).to_tensor(shape=(None, obs.max_len))

    elif tensorlib == "torch":
        # Integer torch index tensors must be long type
        row_lengths = torch.clip(obs.lengths.long(), 1, torch.iinfo(torch.long).max)
        num_elements = row_lengths.sum().item()
        action_mask = torch.zeros(len(row_lengths), obs.max_len, dtype=bool)
        mask_index = torch.LongTensor(
            [(i, j) for i in range(len(row_lengths)) for j in range(row_lengths[i])]
        )
        action_mask.index_put_(
            tuple(mask_index.t()), torch.ones(num_elements, dtype=bool)
        )

    else:
        raise NotImplementedError(f"tensorlib {tensorlib} not implemented")

    return action_mask


def _apply_mask(
    values: TensorType, action_mask: TensorType, tensorlib: str = "tf"
) -> TensorType:

    if tensorlib == "tf":
        return tf.boolean_mask(values, action_mask)

    elif tensorlib == "torch":
        # masked_select returns a 1D tensor so needs reshaping. Pretty sure the last
        # dimension will always be the feature dim -- will action_mask always be 2d?
        # The .view(-1, feature_dim) call will fail if more than 2d.
        feature_dim = values.shape[-1]
        values = torch.masked_select(values, action_mask.view(*action_mask.shape, 1))
        return values.view(-1, feature_dim)

    else:
        raise NotImplementedError(f"tensorlib {tensorlib} not implemented")


def _stack_batch_dim(
    obs: TensorStructType, mask: TensorType, tensorlib: str = "tf"
) -> TensorType:
    if isinstance(obs, dict):
        return {k: _stack_batch_dim(v, mask, tensorlib) for k, v in obs.items()}

    elif isinstance(obs, tuple):
        return tuple(_stack_batch_dim(u, mask, tensorlib) for u in obs)

    elif isinstance(obs, RepeatedValues):
        return _stack_batch_dim(obs.values, mask, tensorlib)

    else:
        return _apply_mask(obs, mask, tensorlib)


def _mask_and_split_values(
    flat_values: TensorType, obs: RepeatedValues, tensorlib: str = "tf"
) -> Tuple[TensorType, TensorType]:
    """Returns the value for the current vertex (index 0 of values),
    and the masked values of the action vertices.
    Args:
        values: Tensor to apply the action mask to.
    Returns:
        (a current state value tensor, a masked action values tensor)
    """

    if tensorlib == "tf":
        row_lengths = tf.clip_by_value(tf.cast(obs.lengths, tf.int32), 1, tf.int32.max)
        flat_values = tf.squeeze(flat_values, axis=[-1])
        values = tf.RaggedTensor.from_row_lengths(flat_values, row_lengths)
        values = values.to_tensor(
            default_value=values.dtype.min,
            shape=(None, obs.max_len),
        )
        current_value = values[:, 0]
        masked_action_values = values[:, 1:]

    elif tensorlib == "torch":
        row_lengths = torch.clip(obs.lengths.long(), 1, torch.iinfo(torch.long).max)
        flat_values = flat_values.squeeze(dim=-1)
        value_index = torch.LongTensor(
            [(i, j) for i in range(len(row_lengths)) for j in range(row_lengths[i])]
        )
        _fmin = torch.finfo(flat_values.dtype).min
        values = _fmin * torch.ones(
            len(row_lengths), obs.max_len, dtype=flat_values.dtype
        )
        values.index_put_(tuple(value_index.t()), flat_values)
        current_value = values[:, 0]
        masked_action_values = values[:, 1:]
    else:
        raise NotImplementedError(f"tensorlib {tensorlib} not implemented")

    return current_value, masked_action_values
