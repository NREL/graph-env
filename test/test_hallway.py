import pytest
from graphenv.examples.hallway.hallway_env import HallwayEnv
from graphenv.examples.hallway.hallway_model import HallwayModel
from graphenv.examples.hallway.hallway_state import HallwayState
from graphenv.graph_model_bellman_mixin import GraphModelBellmanMixin
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env


@pytest.fixture
def hallway_state() -> HallwayState:
    return HallwayState(5)


@pytest.fixture
def hallway_env() -> HallwayEnv:
    return HallwayEnv({"corridor_length": 5})


def test_observation_space(hallway_state: HallwayState):
    assert hallway_state.observation_space


def test_next_actions(hallway_state: HallwayState):
    actions_list = hallway_state.next_actions
    assert len(actions_list) == 1

    actions_list = actions_list[0].next_actions
    assert len(actions_list) == 2


def test_terminal(hallway_state: HallwayState):
    assert hallway_state.new(0).terminal is False
    assert hallway_state.new(5).terminal is True
    # assert hallway_state.new(3, 10).terminal is True


def test_reward(hallway_state: HallwayState):
    assert hallway_state.new(5).reward > 0
    assert hallway_state.new(3).reward == -0.1


def test_graphenv_reset(hallway_env: HallwayEnv):
    obs = hallway_env.reset()
    assert len(obs["action_mask"]) == 3
    assert obs["action_mask"].sum() == 1


def test_graphenv_step(hallway_env: HallwayEnv):
    obs, reward, terminal, info = hallway_env.step(0)

    for _ in range(4):
        assert terminal is False
        assert reward == -0.1
        assert hallway_env.observation_space.contains(obs)
        assert hallway_env.action_space.contains(1)
        obs, reward, terminal, info = hallway_env.step(1)

    assert terminal is True
    assert reward > 0


@pytest.mark.parametrize(
    "model_classes",
    [
        [HallwayModel],
        (GraphModelBellmanMixin, HallwayModel),
    ],
)
def test_ppo(ray_init, ppo_config, model_classes):
    class ThisModel(*model_classes):
        pass

    ModelCatalog.register_custom_model("ThisModel", ThisModel)

    register_env("HallwayEnv", lambda config: HallwayEnv(config))

    config = {
        "env": "HallwayEnv",
        "env_config": {"corridor_length": 5},
        "model": {
            "custom_model": "ThisModel",
            "custom_model_config": {"hidden_dim": 32},
        },
    }
    ppo_config.update(config)
    trainer = ppo.PPOTrainer(config=ppo_config)
    trainer.train()
