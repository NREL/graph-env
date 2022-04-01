import collections
from functools import singledispatch
from typing import OrderedDict, Tuple

import gym.spaces as spaces
import numpy as np

from graphenv import tf


@singledispatch
def broadcast_space(target: spaces.Space, prefix_shape: Tuple[int]):
    """Broadcasts this space into the given shape.

    Args:
        target (spaces.Space): space to broadcast
        prefix_shape (Tuple[int]): shape to broadcast to

    Raises:
        NotImplementedError: If the space is or contains an unsupported type.

    Returns:
        A space recursively expanded to match the given prefix_shape.
    """
    raise NotImplementedError(f"Unsupported Space, {target}.")


@broadcast_space.register(spaces.Box)
def _(target: spaces.Box, prefix_shape: Tuple[int]):
    shape = (*prefix_shape, *target.shape)
    return spaces.Box(
        low=np.broadcast_to(target.low, shape),
        high=np.broadcast_to(target.high, shape),
        shape=shape,
        dtype=target.dtype,
    )


@broadcast_space.register(spaces.MultiBinary)
def _(target: spaces.MultiBinary, prefix_shape: Tuple[int]):
    return spaces.MultiBinary([*prefix_shape, *target.n])


@broadcast_space.register(spaces.Discrete)
def _(target: spaces.Discrete, prefix_shape: Tuple[int]):
    return spaces.MultiDiscrete(
        np.broadcast_to(
            np.array([target.n], dtype=int),
            (*prefix_shape, 1),
        )
    )


@broadcast_space.register(spaces.MultiDiscrete)
def _(target: spaces.MultiDiscrete, prefix_shape: Tuple[int]):
    return spaces.MultiDiscrete(
        np.broadcast_to(target.nvec, (*prefix_shape, *target.shape))
    )


@broadcast_space.register(spaces.Tuple)
def _(target: spaces.Tuple, prefix_shape: Tuple[int]):
    return spaces.Tuple(
        tuple((broadcast_space(s, prefix_shape) for s in target.spaces))
    )


@broadcast_space.register(spaces.Dict)
def _(target: spaces.Dict, prefix_shape: Tuple[int]):
    return spaces.Dict(
        [(k, broadcast_space(s, prefix_shape)) for k, s in target.spaces.items()]
    )


@singledispatch
def stack_observations(space: spaces.Space, space_values):
    """Stacks multiple values together using the supplied space to define the shape of
    the result. For example, stacking values on a Box space simply returns
    np.stack(space_values, axis=0). Stacking values on a Dict space returns a Dict with
    keys corresponding to the keys of the Dict space, and values equal to the result of
    recursively stacking the Spaces of the Dict space values.

    Args:
        space (spaces.Space): Space to use to stack the values into.
        space_values (_type_): Values to stack.

    Raises:
        NotImplementedError: If the space (or one it contains) is unsupported.

    Returns:
        A recursively stacked observation matching the given space's shape, made by
        stacking the values in space_values.
    """
    raise NotImplementedError(f"Unsupported space, {space}.")


@stack_observations.register(spaces.Box)
@stack_observations.register(spaces.MultiBinary)
@stack_observations.register(spaces.MultiDiscrete)
def _(space, space_values):
    return np.stack(space_values, axis=0)


@stack_observations.register(spaces.Discrete)
def _(space: spaces.Discrete, space_values):
    return np.reshape(np.array(space_values), (len(space_values), 1))


@stack_observations.register(spaces.Tuple)
def _(space: spaces.Tuple, space_values):
    return tuple(
        (
            stack_observations(s, [o[i] for o in space_values])
            for i, s in enumerate(space.spaces)
        )
    )


@stack_observations.register(spaces.Dict)
def _(space: spaces.Dict, space_values):
    return {
        k: stack_observations(s, [o[k] for o in space_values])
        for k, s in space.spaces.items()
    }


@singledispatch
def flatten_first_dim(target: any):
    r"""
    Merges the first and second dimensions of the tensor(s) in the target
    data structure. Valid targets include tensors, Iterables (including list),
    and Mappings (including dict). A single tensor with shape (x, y, \*z) becomes
    (x+y, \*z). A list of tensors will result in a list of tensors with this
    operation applied to each one. A dict of tensors will result in an
    OrderdDict of such tensors, etc. Iterables result in lists, Mappings result
    in OrderedDict's in the order returned by the mappings items() iteration.

    Args:
        target (any): The object to recursively reshape

    Raises:
        NotImplementedError: If the target type (or one it contains) is not supported.

    Returns:
        A recursively reshaped value object.
    """
    raise NotImplementedError(f"Unsupported target, {target}.")


@flatten_first_dim.register(collections.abc.Iterable)
def _(target: collections.abc.Iterable):
    return [flatten_first_dim(e) for e in target]


@flatten_first_dim.register(collections.abc.Mapping)
def _(target: collections.abc.Mapping):
    return OrderedDict([(k, flatten_first_dim(e)) for k, e in target.items()])


@flatten_first_dim.register(tf.Tensor)
def _(target: tf.Tensor):
    shape = tf.shape(target)
    dest_shape = (shape[0] * shape[1], *shape[2:])
    return tf.reshape(target, dest_shape)


@flatten_first_dim.register(np.ndarray)
def _(target: np.ndarray):
    shape = np.shape(target)
    dest_shape = (shape[0] * shape[1], *shape[2:])
    return np.reshape(target, dest_shape)
