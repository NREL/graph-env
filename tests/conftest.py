import pytest
import ray
from ray.rllib.agents import a3c, dqn, marwil, ppo


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
        "hiddens": False,
        "dueling": False,
        "timesteps_per_iteration": 100,
        "target_network_update_freq": 50,
    }

    config = dqn.DEFAULT_CONFIG.copy()
    config.update(global_config)
    config.update(specific_config)

    return config


@pytest.fixture
def a3c_config(global_config):

    specific_config = {}

    config = a3c.DEFAULT_CONFIG.copy()
    config.update(global_config)
    config.update(specific_config)

    return config


@pytest.fixture
def marwil_config(global_config):

    specific_config = {}

    config = marwil.DEFAULT_CONFIG.copy()
    config.update(global_config)
    config.update(specific_config)

    return config


@pytest.fixture(scope="function", params=["a3c", "dqn", "marwil", "ppo"])
def agent(request, ppo_config, dqn_config, a3c_config, marwil_config):
    """Returns trainer, config, needs_q_model"""

    if request.param == "ppo":
        return ppo.PPOTrainer, ppo_config, False

    elif request.param == "dqn":
        return dqn.DQNTrainer, dqn_config, True

    elif request.param == "a3c":
        return a3c.A3CTrainer, a3c_config, False

    elif request.param == "marwil":
        return marwil.MARWILTrainer, marwil_config, False
