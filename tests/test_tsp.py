import logging
import secrets

import pytest
from graphenv.examples.tsp.graph_utils import make_complete_planar_graph
from graphenv.examples.tsp.tsp_env import TSPEnv
from graphenv.examples.tsp.tsp_model import TSPModel
from graphenv.graph_model_bellman_mixin import GraphModelBellmanMixin
from ray.rllib.agents import dqn, ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env


@pytest.fixture
def N():
    return 4


@pytest.fixture
def G(N):
    seed = 1
    return make_complete_planar_graph(N=N, seed=seed)


@pytest.mark.parametrize(
    "model_classes",
    [
        [TSPModel],
        (GraphModelBellmanMixin, TSPModel),
    ],
)
def test_ppo(ray_init, ppo_config, N, G, model_classes):

    runid = secrets.token_hex(nbytes=8)

    class ThisModel(*model_classes):
        pass

    ModelCatalog.register_custom_model("ThisModel" + runid, TSPModel)
    register_env("ThisEnv" + runid, lambda config: TSPEnv(config))

    config = {
        "env": "ThisEnv" + runid,
        "env_config": {"G": G},
        "model": {
            "custom_model": "ThisModel" + runid,
            "custom_model_config": {
                "num_nodes": N,
                "hidden_dim": 256,
                "embed_dim": 256,
            },
        },
    }
    ppo_config.update(config)
    trainer = ppo.PPOTrainer(config=ppo_config)
    trainer.train()


@pytest.mark.parametrize(
    "model_classes",
    [
        [TSPModel],
        (GraphModelBellmanMixin, TSPModel),
    ],
)
def test_dqn(ray_init, dqn_config, caplog, N, G, model_classes):

    runid = secrets.token_hex(nbytes=8)

    class ThisModel(*model_classes):
        pass

    caplog.set_level(logging.DEBUG)

    ModelCatalog.register_custom_model("ThisModel" + runid, ThisModel)
    register_env("ThisEnv" + runid, lambda config: TSPEnv(config))

    config = {
        "env": "ThisEnv" + runid,
        "env_config": {"G": G},
        "hiddens": False,
        "dueling": False,
        "model": {
            "custom_model": "ThisModel" + runid,
            "custom_model_config": {
                "num_nodes": N,
                "hidden_dim": 256,
                "embed_dim": 256,
            },
        },
    }
    dqn_config.update(config)
    trainer = dqn.DQNTrainer(config=dqn_config)
    trainer.train()
