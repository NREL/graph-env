import pytest
import ray
from ray.rllib.agents import dqn, ppo


@pytest.fixture(scope="session")
# @pytest.fixture
def ray_init():
    ray.init(num_cpus=1, local_mode=True)
    yield None
    ray.shutdown()


@pytest.fixture
def global_config():
    return {
        "num_gpus": 0,
        "num_workers": 1,
        "framework": "tf2",
        "eager_tracing": False,
        "eager_max_retraces": 20,
        "rollout_fragment_length": 5,
        "train_batch_size": 20,
        "lr": 1e-3,
    }


@pytest.fixture
def ppo_config(global_config):

    specific_config = {
        "sgd_minibatch_size": 2,
        "shuffle_sequences": True,
        "num_sgd_iter": 1,
    }

    config = ppo.DEFAULT_CONFIG.copy()
    config.update(global_config)
    config.update(specific_config)

    return config


@pytest.fixture
def dqn_config(global_config):

    specific_config = {
        "timesteps_per_iteration": 100,
        "target_network_update_freq": 50,
    }

    config = dqn.DEFAULT_CONFIG.copy()
    config.update(global_config)
    config.update(specific_config)

    return config
