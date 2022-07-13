from ray.rllib.utils.typing import TensorType

from graphenv import tf, torch
from graphenv.graph_model import GraphModel


class GraphModelBellmanMixin:
    """
    Mixin for use with GraphModel that evaluates the current state as the
    max of the successor state value assesments.
    """

    def _forward_total_value(self: GraphModel):
        """
        Overrides state evaluation, replacing it with a Bellman backup returning
        the max over all successor states's values.

        Returns:
            Tensor of state values.
        """
        return _reduce_max(self.action_values, self._tensorlib)


def _reduce_max(action_values: TensorType, tensorlib: str = "tf") -> TensorType:

    if tensorlib == "tf":
        assert tf is not None
        return tf.reduce_max(action_values, axis=1)

    elif tensorlib == "torch":
        assert torch is not None
        return torch.max(action_values, 1, keepdim=False)
