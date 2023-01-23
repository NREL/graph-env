from typing import Tuple

from graphenv import nn
from graphenv.graph_model import TorchGraphModel
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorStructType, TensorType


class TorchHallwayModel(
    TorchGraphModel,
    TorchModelV2,
    nn.Module,
):
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
        nn.Module.__init__(self)

        self.hidden_layer = nn.Linear(1, hidden_dim)
        self.action_value_output = nn.Linear(hidden_dim, 1)
        self.action_weight_output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward_vertex(
        self,
        input_dict: TensorStructType,
    ) -> Tuple[TensorType, TensorType]:

        x = self.hidden_layer(input_dict["cur_pos"].float())
        x = self.relu(x)

        return (
            self.action_value_output(x),
            self.action_weight_output(x),
        )


class TorchHallwayQModel(TorchHallwayModel, DQNTorchModel):
    pass


# class TorchHallwayModel(BaseTorchHallwayModel):
# pass
