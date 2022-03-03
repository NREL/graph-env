from functools import singledispatch
from typing import Tuple
import numpy as np
import gym.spaces as spaces


@singledispatch
def broadcast_space(space: spaces.Space, prefix_shape: Tuple[int]):
    raise NotImplementedError(f'Unsupported Space, {space}.')


@broadcast_space.register(spaces.Box)
def _(space: spaces.Box, prefix_shape: Tuple[int]):
    shape = (*prefix_shape, *space.shape)
    return spaces.Box(
        low=np.broadcast_to(space.low, shape),
        high=np.broadcast_to(space.high, shape),
        shape=shape,
        dtype=space.dtype,
    )


@broadcast_space.register(spaces.MultiBinary)
def _(space: spaces.MultiBinary, prefix_shape: Tuple[int]):
    return spaces.MultiBinary([*prefix_shape, *space.n])


@broadcast_space.register(spaces.Discrete)
def _(space: spaces.Discrete, prefix_shape: Tuple[int]):
    return spaces.MultiDiscrete(
        np.broadcast_to(
            np.array([space.n]),
            (*prefix_shape, 1),
            subok=True,
        ))


@broadcast_space.register(spaces.MultiDiscrete)
def _(space: spaces.MultiDiscrete, prefix_shape: Tuple[int]):
    return spaces.MultiDiscrete(
        np.broadcast_to(space.nvec, (*prefix_shape, *space.shape)))


@broadcast_space.register(spaces.Tuple)
def _(space: spaces.Tuple, prefix_shape: Tuple[int]):
    return spaces.Tuple(tuple(
        (
            broadcast_space(s, prefix_shape)
            for s in space.spaces
        )))


@broadcast_space.register(spaces.Dict)
def _(space: spaces.Dict, prefix_shape: Tuple[int]):
    return spaces.Dict(
        [(k, broadcast_space(s, prefix_shape))
         for k, s in space.spaces.items()
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
