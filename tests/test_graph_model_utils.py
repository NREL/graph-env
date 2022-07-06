import numpy as np
import pytest
from graphenv import tf
from graphenv.graph_model import (
    _create_action_mask,
    _mask_and_split_values,
    _stack_batch_dim,
)
from ray.rllib.models.repeated_values import RepeatedValues


@pytest.fixture
def obs_tf():
    return RepeatedValues(
        {
            "key1": tf.constant(np.random.rand(5, 10, 2)),
            "key2": tf.constant(np.random.rand(5, 10, 4)),
        },
        lengths=tf.constant([1, 4, 8, 2, 4]),
        max_len=10,
    )


@pytest.fixture
def vals_tf():
    return tf.constant(np.random.rand(19, 1))


def test_create_action_mask_tf(obs_tf):
    mask = _create_action_mask(obs_tf, tf)
    assert np.alltrue(mask.numpy().sum(1) == np.array([1, 4, 8, 2, 4]))


def test_stack_batch_dim_tf(obs_tf):
    mask = _create_action_mask(obs_tf, tf)
    stacked_obs = _stack_batch_dim(obs_tf, mask, tf)
    assert stacked_obs["key1"].shape == (19, 2)
    assert stacked_obs["key2"].shape == (19, 4)


def test_mask_and_split_values_tf(vals_tf, obs_tf):
    state_vals, action_vals = _mask_and_split_values(vals_tf, obs_tf, tf)
    assert state_vals.numpy().shape == (5,)
    assert action_vals.numpy().shape == (5, 9)
    assert state_vals.numpy().min() >= 0  # make sure these don't get masked

    for length, val_row in zip([1, 4, 8, 2, 4], action_vals.numpy()):
        assert np.allclose(val_row[length - 1 :], action_vals.dtype.min)
