import pytest
from graphenv.examples.tsp.graph_utils import make_complete_planar_graph
from graphenv.examples.tsp.tsp_model import TSPModel, TSPQModel
from graphenv.examples.tsp.tsp_nfp_model import TSPGNNModel, TSPGNNQModel
from graphenv.examples.tsp.tsp_nfp_state import TSPNFPState
from graphenv.examples.tsp.tsp_state import TSPState
from graphenv.graph_env import GraphEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env


@pytest.fixture
def N():
    return 4


@pytest.fixture
def G(N):
    return lambda: make_complete_planar_graph(N=N)


def test_graphenv():

    env = GraphEnv(
        {
            "state": TSPState(
                lambda: make_complete_planar_graph(10),
            ),
            "max_num_children": 10,
        }
    )

    obs = env.reset()
    assert env.observation_space.contains(obs)


def test_graphenv_nfp():

    env = GraphEnv(
        {
            "state": TSPNFPState(
                lambda: make_complete_planar_graph(10), max_num_neighbors=5
            ),
            "max_num_children": 10,
        }
    )

    obs = env.reset()
    assert env.observation_space.contains(obs)


def test_rllib_base(ray_init, agent, N, G):

    trainer_fn, config, needs_q_model = agent
    model = TSPQModel if needs_q_model else TSPModel

    ModelCatalog.register_custom_model("this_model", model)
    register_env("graphenv", lambda config: GraphEnv(config))

    config.update(
        {
            "env": "graphenv",
            "env_config": {
                "state": TSPState(G),
                "max_num_children": N,
            },
            "model": {
                "custom_model": "this_model",
                "custom_model_config": {
                    "num_nodes": N,
                    "hidden_dim": 256,
                    "embed_dim": 256,
                },
            },
        }
    )

    trainer = trainer_fn(config=config)
    trainer.train()


def test_rllib_nfp(ray_init, agent, N, G):

    trainer_fn, config, needs_q_model = agent
    model = TSPGNNQModel if needs_q_model else TSPGNNModel

    ModelCatalog.register_custom_model("this_model", model)
    register_env("graphenv", lambda config: GraphEnv(config))

    config.update(
        {
            "env": "graphenv",
            "env_config": {
                "state": TSPNFPState(G),
                "max_num_children": N,
            },
            "model": {
                "custom_model": "this_model",
                "custom_model_config": {
                    "num_messages": 1,
                    "embed_dim": 32,
                },
            },
        }
    )

    trainer = trainer_fn(config=config)
    trainer.train()
