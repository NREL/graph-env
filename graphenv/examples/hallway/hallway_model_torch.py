from typing import Tuple

from graphenv import torch, nn
from graphenv.graph_model import GraphModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorStructType, TensorType


class TorchHallwayModel(GraphModel, nn.Module, TorchModelV2):
    """An example GraphModel implementation for the HallwayEnv and HallwayState
    Graph. Uses a dense fully connected Torch network.

    Args:
        hidden_dim (int, optional): The number of hidden layers to use. Defaults to 1.
    """

    def __init__(
        self,
        *args,
        hidden_dim: int = 1,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.hidden_layer = nn.Linear(1, hidden_dim)
        self.action_value_output = nn.Linear(hidden_dim, 1)
        self.action_weight_output = nn.Linear(hidden_dim, 1)


    def forward(
        self, 
        input_dict: TensorStructType
    ) -> Tuple[TensorType, TensorType]:
        
        x = self.hidden(x)
        x = nn.ReLU(x)

        return (self.action_value_output(x), self.action_weight_output(x), )


    def forward_vertex(
        self,
        input_dict: TensorStructType,
    ) -> Tuple[TensorType, TensorType]:
        
        return self.forward(input_dict)


# There doesn't appear to be an equivalent DistributionalQTorchModel.


if __name__ == "__main__":

    import ray
    from ray.rllib.agents import ppo
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    from graphenv.examples.hallway.hallway_state import HallwayState
    from graphenv.graph_env import GraphEnv
    from graphenv.examples.hallway.hallway_model_torch import TorchHallwayModel

    ray.init()

    ModelCatalog.register_custom_model("this_model", TorchHallwayModel)
    register_env("graphenv", lambda config: GraphEnv(config))

    config = ppo.DEFAULT_CONFIG.copy()
    config.update(
        {
            "env": "graphenv",
            "env_config": {
                "state": HallwayState(5),
                "max_num_children": 2,
            },
            "framework": "torch",
            "log_level": "DEBUG",
            "model": {
                "custom_model": "this_model",
                "custom_model_config": {"hidden_dim": 32},
            },
        }
    )

    trainer = ppo.PPOTrainer(config=config)
    trainer.train()
