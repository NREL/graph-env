import pytest
import ray
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.algorithms.ppo import PPOConfig


@pytest.fixture(scope="session")
# @pytest.fixture
def ray_init():
    ray.init(num_cpus=1, local_mode=True)
    yield None
    ray.shutdown()


#@pytest.fixture
def set_global_config(config):
    config.training(lr=1e-3, train_batch_size=20)\
          .resources(num_gpus=0)\
          .framework("tf2")\
          .rollouts(num_rollout_workers=1, rollout_fragment_length=5)\
          .debugging(log_level="DEBUG")
    return config


@pytest.fixture
def ppo_config():
    
    config = PPOConfig()
    config = set_global_config(config)

    config = config.training(sgd_minibatch_size=2,
                             shuffle_sequences=True,
                             num_sgd_iter=1)

    return config


@pytest.fixture
def dqn_config():
    config = DQNConfig()
    config = set_global_config(config)
    
    config = config.training(hiddens=False,
                             dueling=False,
                             target_network_update_freq=50)\
                   .reporting(min_train_timesteps_per_iteration=100)
            
    return config


@pytest.fixture
def a3c_config():

    config = A3CConfig()
    config = set_global_config(config)

    return config


@pytest.fixture
def marwil_config():

    config = MARWILConfig()
    config = set_global_config(config)

    return config


@pytest.fixture(scope="function", params=["a3c", "dqn", "marwil", "ppo"])
def agent(request, ppo_config, dqn_config, a3c_config, marwil_config):
    """Returns config, needs_q_model"""

    if request.param == "ppo":
        return ppo_config, False

    elif request.param == "dqn":
        return dqn_config, True

    elif request.param == "a3c":
        return a3c_config, False

    elif request.param == "marwil":
        return marwil_config, False
