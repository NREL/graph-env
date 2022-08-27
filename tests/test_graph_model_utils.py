import numpy as np
import pytest
from graphenv import tf, torch
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
        lengths=tf.constant([1, 4, 8, 2, 4], dtype=tf.float32),
        max_len=10,
    )


@pytest.fixture
def obs_torch():
    return RepeatedValues(
        {
            "key1": torch.rand(5, 10, 2),
            "key2": torch.rand(5, 10, 4),
        },
        lengths=torch.LongTensor([1, 4, 8, 2, 4]).float(),
        max_len=10,
    )


@pytest.fixture
def vals_tf():
    return tf.constant(np.random.rand(19, 1))


@pytest.fixture
def vals_torch():
    return torch.rand(19, 1)


@pytest.mark.parametrize("obs,tensorlib", [("obs_tf", "tf"), ("obs_torch", "torch")])
def test_create_action_mask(obs, tensorlib, request):
    obs = request.getfixturevalue(obs)
    mask = _create_action_mask(obs, tensorlib)
    assert np.alltrue(mask.numpy().sum(1) == np.array([1, 4, 8, 2, 4]))


@pytest.mark.parametrize("obs,tensorlib", [("obs_tf", "tf"), ("obs_torch", "torch")])
def test_stack_batch_dim(obs, tensorlib, request):
    obs = request.getfixturevalue(obs)
    mask = _create_action_mask(obs, tensorlib)
    stacked_obs = _stack_batch_dim(obs, mask, tensorlib)
    assert stacked_obs["key1"].shape == (19, 2)
    assert stacked_obs["key2"].shape == (19, 4)


@pytest.mark.parametrize(
    "obs,vals,tensorlib",
    [("obs_tf", "vals_tf", "tf"), ("obs_torch", "vals_torch", "torch")],
)
def test_mask_and_split_values(obs, vals, tensorlib, request):
    obs = request.getfixturevalue(obs)
    vals = request.getfixturevalue(vals)
    state_vals, action_vals = _mask_and_split_values(vals, obs, tensorlib)
    assert state_vals.numpy().shape == (5,)
    assert action_vals.numpy().shape == (5, 9)
    assert state_vals.numpy().min() >= 0  # make sure these don't get masked

    if tensorlib == "tf":
        dtype_min = action_vals.dtype.min

    elif tensorlib == "torch":
        dtype_min = torch.finfo(action_vals.dtype).min

    for length, val_row in zip([1, 4, 8, 2, 4], action_vals.numpy()):
        assert np.allclose(val_row[length - 1:], dtype_min)
