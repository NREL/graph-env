
from typing import Dict, Optional, Sequence, Tuple

from graphenv.graph_env import GraphEnv
from ray.rllib.utils.framework import try_import_tf

from ray.rllib.env.env_context import EnvContext
from .hallway_state import HallwayState

tf1, tf, tfv = try_import_tf()
layers = tf.keras.layers


class HallwayEnv(GraphEnv):
    '''
    Convience class of a GraphEnv using a HallwayState as the vertex state.
    '''

    def __init__(self, config: EnvContext, *args, **kwargs):
        super().__init__(
            HallwayState(config["size"], config["max_steps"]),
            *args,
            **kwargs,
        )
