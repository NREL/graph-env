import pytest
import ray
from ray.rllib.agents import dqn, ppo


@pytest.fixture(scope="module")
# @pytest.fixture
def ray_init():
    ray.init(num_cpus=1, local_mode=True)
    yield None
    ray.shutdown()


@pytest.fixture
def ppo_config():

    config = {
        "num_gpus": 0,
        "num_workers": 1,  # parallelism
        "framework": "tf2",
        "eager_tracing": False,
        "eager_max_retraces": 20,
        "rollout_fragment_length": 5,
        "train_batch_size": 20,
        "sgd_minibatch_size": 2,
        "shuffle_sequences": True,
        "num_sgd_iter": 1,
        "lr": 1e-3,
    }

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)

    return ppo_config


@pytest.fixture
def dqn_config():

    config = {
        "num_gpus": 0,
        "num_workers": 1,  # parallelism
        "framework": "tf2",
        "eager_tracing": False,
        "eager_max_retraces": 20,
        "rollout_fragment_length": 5,
        "train_batch_size": 20,
        "lr": 1e-3,
        "timesteps_per_iteration": 100,
        "target_network_update_freq": 50,
    }

    dqn_config = dqn.DEFAULT_CONFIG.copy()
    dqn_config.update(config)

    return dqn_config
