import pytest
from graphenv.examples.hallway import Hallway
from graphenv.graphenv import GraphEnv


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
    assert hallway.new(0, 0).terminal == False
    assert hallway.new(5, 8).terminal == True
    assert hallway.new(3, 10).terminal == True


def test_reward(hallway: Hallway):
    assert hallway.new(5, 5).reward() == 0
    assert hallway.new(3, 5).reward() == -1


def test_graphenv_reset(hallway_env: GraphEnv, hallway: Hallway):
    obs = hallway_env.reset()
    assert len(obs["action_mask"]) == 2
    assert obs["action_mask"].sum() == 1
    assert obs["action_observations"][1] == hallway.null_observation


def test_graphenv_step(hallway_env: GraphEnv):
    obs, reward, terminal, info = hallway_env.step(0)
    for _ in range(4):
        assert terminal is False
        assert reward == -1.0
        obs, reward, terminal, info = hallway_env.step(1)

    assert terminal is True
    assert reward == 0.0
