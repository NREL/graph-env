from functools import singledispatch, singledispatchmethod
import logging
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Protocol, Tuple, Type, TypeVar, Union

import gym
import gym.spaces
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.utils.typing import TensorStructType, TensorType

from graphenv import tf, torch

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

logger = logging.getLogger(__file__)


ModelSuperclass = TypeVar('ModelSuperclass', TFModelV2, TorchModelV2)

'''
+ need different superclass
+ need to switch some method implementations

'''


class GraphModelInterface(Protocol):

    @property
    def action_values(self) -> Any:
        pass


class GraphModel(Generic[ModelSuperclass]):
    """Defines a RLLib compatible model for using RL algorithms on a GraphEnv.

    Args:
        obs_space: The observation space to use.
        action_space: The action space to use.
        num_outputs: The number of scalar outputs per state to produce.
        model_config: Config forwarded to TFModelV2.__init()__.
        name: Config forwarded to TFModelV2.__init()__.
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
            obs_space,  # type: ignore
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

        mask = self._create_action_mask(input_dict["obs"])
        flattened_observations = self._stack_batch_dim(
            input_dict["obs"], mask,
        )
        flat_values, flat_weights = self.forward_vertex(flattened_observations)

        # mask out invalid children and get current vertex value
        self.current_vertex_value, _ = self._mask_and_split_values(
            flat_values, input_dict["obs"],
        )
        _, action_weights = self._mask_and_split_values(
            flat_weights, input_dict["obs"],
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

    @singledispatchmethod
    def _stack_batch_dim(
        self,
        obs: TensorStructType,
        mask: TensorType,
    ) -> TensorType:
        return self._apply_mask(obs, mask)

    @_stack_batch_dim.register
    def _(
        self,
        obs: dict,
        mask: TensorType,
    ) -> TensorType:
        return {k: self._stack_batch_dim(v, mask) for k, v in obs.items()}

    @_stack_batch_dim.register
    def _(
        self,
        obs: tuple,
        mask: TensorType,
    ) -> TensorType:
        return tuple(self._stack_batch_dim(u, mask) for u in obs)

    @_stack_batch_dim.register
    def _(
        self,
        obs: RepeatedValues,
        mask: TensorType,
    ) -> TensorType:
        return self._stack_batch_dim(obs.values, mask)

    @abstractmethod
    def _create_action_mask(self, obs: RepeatedValues) -> TensorType:
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
        pass

    @abstractmethod
    def _apply_mask(
        self,
        values: TensorType,
        action_mask: TensorType,
    ) -> TensorType:
        pass

    @abstractmethod
    def _mask_and_split_values(
        self,
        flat_values: TensorType,
        obs: RepeatedValues,
    ) -> Tuple[TensorType, TensorType]:
        """Returns the value for the current vertex (index 0 of values),
        and the masked values of the action verticies.
        Args:
            values: Tensor to apply the action mask to.
        Returns:
            (a current state value tensor, a masked action values tensor)
        """
        pass


class GraphModelTF(GraphModel[TFModelV2]):
    """A GraphModel using Tensorflow and TFModelV2
    """

    @abstractmethod
    def _create_action_mask(self, obs: RepeatedValues) -> TensorType:
        # the "dummy batch" rllib provides to initialize the policy model is a matrix of
        # all zeros, which ends with a batch size of zero provided to the policy model.
        # We can assume that at least the input state is valid, and clip the row_lengths
        # vector to a minimum of 1 per (state, *actions) entry.
        assert tf is not None
        row_lengths = tf.clip_by_value(tf.cast(obs.lengths, tf.int32), 1, tf.int32.max)
        num_elements = tf.reduce_sum(row_lengths)
        action_mask = tf.RaggedTensor.from_row_lengths(
            tf.ones(num_elements, dtype=tf.bool),
            row_lengths,
        ).to_tensor(shape=(None, obs.max_len))
        return action_mask

    @abstractmethod
    def _apply_mask(
        self,
        values: TensorType,
        action_mask: TensorType,
    ) -> TensorType:
        assert tf is not None
        return tf.boolean_mask(values, action_mask)

    @abstractmethod
    def _mask_and_split_values(
        self,
        flat_values: TensorType,
        obs: RepeatedValues,
    ) -> Tuple[TensorType, TensorType]:
        assert tf is not None
        row_lengths = tf.clip_by_value(tf.cast(obs.lengths, tf.int32), 1, tf.int32.max)
        flat_values = tf.squeeze(flat_values, axis=[-1])
        values = tf.RaggedTensor.from_row_lengths(flat_values, row_lengths)
        values = values.to_tensor(
            default_value=values.dtype.min,
            shape=(None, obs.max_len),
        )
        current_value = values[:, 0]
        masked_action_values = values[:, 1:]
        return current_value, masked_action_values


class GraphModelTorch(GraphModel[TorchModelV2]):
    """A GraphModel using PyTorch and TorchModelV2
    """

    @abstractmethod
    def _create_action_mask(self, obs: RepeatedValues) -> TensorType:
        # Integer torch index tensors must be long type
        assert torch is not None
        row_lengths = torch.clip(obs.lengths.long(), 1, torch.iinfo(torch.long).max)
        num_elements = row_lengths.sum().item()
        action_mask = torch.zeros(len(row_lengths), obs.max_len, dtype=bool)
        mask_index = torch.LongTensor(
            [(i, j) for i in range(len(row_lengths)) for j in range(row_lengths[i])]
        )
        action_mask.index_put_(
            tuple(mask_index.t()), torch.ones(num_elements, dtype=bool)
        )
        return action_mask

    @abstractmethod
    def _apply_mask(
        self,
        values: TensorType,
        action_mask: TensorType,
    ) -> TensorType:
        # masked_select returns a 1D tensor so needs reshaping. Pretty sure the last
        # dimension will always be the feature dim -- will action_mask always be 2d?
        # The .view(-1, feature_dim) call will fail if more than 2d.
        assert torch is not None
        feature_dim = values.shape[-1]
        values = torch.masked_select(values, action_mask.view(*action_mask.shape, 1))
        return values.view(-1, feature_dim)

    @abstractmethod
    def _mask_and_split_values(
        self,
        flat_values: TensorType,
        obs: RepeatedValues,
    ) -> Tuple[TensorType, TensorType]:
        assert torch is not None
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
        return current_value, masked_action_values
