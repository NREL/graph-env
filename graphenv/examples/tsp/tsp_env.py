from graphenv import tf
from graphenv.graph_env import GraphEnv
from ray.rllib.env.env_context import EnvContext

from graphenv.examples.tsp.tsp_state import TSPState


class TSPEnv(GraphEnv):
    """
    Convience class of a GraphEnv using a TSPState as the vertex state.
    """

    def __init__(self, config: EnvContext, *args, **kwargs):
        G = config["G"]
        super().__init__(
            TSPState(G),
            *args,
            max_num_actions=G.number_of_nodes(),
            **kwargs,
        )
