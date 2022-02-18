import os
import pytest
from docs.examples.run_hallway import HallwayEnv
from graphenv.examples.hallway import Hallway, HallwayModel
from graphenv.graph_env import GraphEnv
from graphenv.graph_model import GraphModel
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog


@pytest.fixture
def hallway() -> Hallway:
    return Hallway(5, 10, 0, 0)


@pytest.fixture
def hallway_env(hallway) -> GraphEnv:
    return GraphEnv(hallway)


def test_observation_space(hallway: Hallway):
    assert hallway.observation_space


def test_next_actions(hallway: Hallway):
    actions_list = hallway.next_actions
    assert len(actions_list) == 1
    assert actions_list[0].episode_steps == 1

    actions_list = actions_list[0].next_actions
    assert len(actions_list) == 2
    assert actions_list[0].episode_steps == 2


def test_terminal(hallway: Hallway):
    assert hallway.new(0, 0).terminal is False
    assert hallway.new(5, 8).terminal is True
    assert hallway.new(3, 10).terminal is True


def test_reward(hallway: Hallway):
    assert hallway.new(5, 5).reward() > 0
    assert hallway.new(3, 5).reward() == -0.1


def test_graphenv_reset(hallway_env: GraphEnv, hallway: Hallway):
    obs = hallway_env.reset()
    assert len(obs["action_mask"]) == 3
    assert obs["action_mask"].sum() == 1


@pytest.mark.parametrize("envtype", [("hallway_env")])
def test_graphenv_step(request, envtype):
    hallway_env: GraphEnv = request.getfixturevalue(envtype)
    obs, reward, terminal, info = hallway_env.step(0)

    for _ in range(4):
        assert terminal is False
        assert reward == -0.1
        assert hallway_env.observation_space.contains(obs)
        assert hallway_env.action_space.contains(1)
        obs, reward, terminal, info = hallway_env.step(1)

    assert terminal is True
    assert reward > 0


# @pytest.mark.skip
@pytest.mark.parametrize("env,model", [(GraphEnv, GraphModel)])
def test_ppo(ray_init, ppo_config, env, model):
    
    register_env("HallwayEnv", lambda config: HallwayEnv(config))
    ModelCatalog.register_custom_model("HallwayModel", HallwayModel)
    config = {
        "env": "HallwayEnv",  # or "corridor" if registered above
        "env_config": {
            "size": 5,
            "max_steps": 100,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "HallwayModel",
            "custom_model_config": {"hidden_dim": 32},
        },
        "num_workers": 1,  # parallelism
        "framework": "tf2",
        "eager_tracing": False,
        "eager_max_retraces": 20,
    }
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)
    trainer = ppo.PPOTrainer(config=ppo_config)
    trainer.train()
