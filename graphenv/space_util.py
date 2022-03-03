import abc
import collections
from functools import singledispatch
from typing import OrderedDict, Tuple
import warnings
import numpy as np
import gym.spaces as spaces

from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


@singledispatch
def broadcast_space(target: spaces.Space, prefix_shape: Tuple[int]):
    raise NotImplementedError(f'Unsupported Space, {target}.')


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
        ))


@broadcast_space.register(spaces.MultiDiscrete)
def _(target: spaces.MultiDiscrete, prefix_shape: Tuple[int]):
    return spaces.MultiDiscrete(
        np.broadcast_to(target.nvec, (*prefix_shape, *target.shape)))


@broadcast_space.register(spaces.Tuple)
def _(target: spaces.Tuple, prefix_shape: Tuple[int]):
    return spaces.Tuple(tuple(
        (
            broadcast_space(s, prefix_shape)
            for s in target.spaces
        )))


@broadcast_space.register(spaces.Dict)
def _(target: spaces.Dict, prefix_shape: Tuple[int]):
    return spaces.Dict(
        [(k, broadcast_space(s, prefix_shape))
         for k, s in target.spaces.items()
         ])


@singledispatch
def stack_observations(space: spaces.Space, space_values):
    raise NotImplementedError(f'Unsupported space, {space}.')


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
    return (stack_observations(s, [o[i] for o in space_values])
            for i, s in enumerate(space.spaces))


@stack_observations.register(spaces.Dict)
def _(space: spaces.Dict, space_values):
    return {k: stack_observations(s, [o[k] for o in space_values])
            for k, s in space.spaces.items()}


@singledispatch
def flatten_first_dim(target: any):
    raise NotImplementedError(f'Unsupported target, {target}.')


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
