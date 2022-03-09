from graphenv import tf
from graphenv.graph_env import GraphEnv
from ray.rllib.env.env_context import EnvContext

from .hallway_state import HallwayState

layers = tf.keras.layers


class HallwayEnv(GraphEnv):
    """
    Convience class of a GraphEnv using a HallwayState as the vertex state.
    """

    def __init__(self, config: EnvContext, *args, **kwargs):
        super().__init__(
            HallwayState(config["corridor_length"]),
            *args,
            **kwargs,
        )
