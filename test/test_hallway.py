import pytest
from graphenv.examples.hallway import Hallway, HallwayModelMixin
from graphenv.graphenv import GraphEnv
from graphenv.models.graph_model import GraphEnvModel
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext


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
    assert len(obs["action_mask"]) == 2
    assert obs["action_mask"].sum() == 1
    assert obs["action_observations"][1] == hallway.null_observation


def test_graphenv_step(hallway_env: GraphEnv):
    obs, reward, terminal, info = hallway_env.step(0)

    for _ in range(4):
        assert terminal is False
        assert reward == -0.1
        assert hallway_env.observation_space.contains(obs)
        assert hallway_env.action_space.contains(1)
        obs, reward, terminal, info = hallway_env.step(1)

    assert terminal is True
    assert reward > 0


@pytest.mark.parametrize("backend_model", [GraphEnvModel])
def test_ppo(ray_init, ppo_config, backend_model):
    class HallwayEnv(GraphEnv):
        def __init__(self, config: EnvContext):
            super().__init__(Hallway(config["size"], config["max_steps"]))

    class HallwayModel(HallwayModelMixin, backend_model):
        pass

    config = {
        "env": HallwayEnv,  # or "corridor" if registered above
        "env_config": {
            "size": 5,
            "max_steps": 100,
        },
        "model": {
            "custom_model": HallwayModel,
            "custom_model_config": {"hidden_dim": 8},
        },
    }
    ppo_config.update(config)
    trainer = ppo.PPOTrainer(config=ppo_config, env=HallwayEnv)
    trainer.train()
