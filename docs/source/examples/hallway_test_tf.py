import ray
from graphenv.examples.hallway.hallway_model import HallwayModel
from graphenv.examples.hallway.hallway_state import HallwayState
from graphenv.graph_env import GraphEnv
from ray import tune

config = {
    "env": GraphEnv,
    "env_config": {
        "state": HallwayState(5),
        "max_num_children": 2,
    },
    "model": {
        "custom_model": HallwayModel,
        "custom_model_config": {"hidden_dim": 32},
    },
    "framework": "tf2",
    "eager_tracing": True,
    "num_workers": 1,
}

stop = {
    "training_iteration": 5,
}

if __name__ == "__main__":

    ray.init()

    tune.run(
        "PPO",
        config=config,
        stop=stop,
    )
