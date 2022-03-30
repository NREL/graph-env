from graphenv import tf
from graphenv.graph_env import GraphEnv
from ray.rllib.env.env_context import EnvContext

from graphenv.examples.tsp.tsp_state import TSPState


class TSPEnv(GraphEnv):
    """
    Convenience class of a GraphEnv using a TSPState as the vertex state.
    """

    def __init__(self, config: EnvContext, *args, **kwargs):
        G = config["G"]
        super().__init__(
            TSPState(G),
            *args,
            max_num_actions=G.number_of_nodes(),
            **kwargs,
        )


if __name__ == "__main__":

    import argparse
    import random
    import networkx as nx

    from graphenv.examples.tsp.graph_utils import make_complete_planar_graph

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--N", 
        type=int, 
        default=5, 
        help="Number of nodes")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for creating graph")
    args = parser.parse_args()

    N = args.N
    G = make_complete_planar_graph(N=N, seed=args.seed)

    # Solve with networkx heurisitics
    tsp = nx.approximation.traveling_salesman_problem
    path = tsp(G, cycle=True)
    cost = sum([G[path[i]][path[i+1]]["weight"] for i in range(0, N-1)])
    print(f"Networkx heuristic cost: {cost:1.3f}")

    # Use the TSP env and generate a random circuit
    config = {"G": G}
    env = TSPEnv(config)

    def sampler(mask):
        choices = list(range(len([i for i in mask if i])))
        return random.choice(choices)

    obs = env.reset()
    done = False
    reward = 0.
    while not done:
        action = sampler(obs["action_mask"])
        obs, rew, done, _ = env.step(action)
        cost -= rew
    
    print(f"TSPEnv random action cost: {cost:1.3f}")
