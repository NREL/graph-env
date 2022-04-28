import argparse

import ray
from graphenv.examples.tsp.graph_utils import make_complete_planar_graph
from graphenv.graph_env import GraphEnv
from graphenv.examples.tsp.tsp_nfp_model import TSPGNNModel
from graphenv.examples.tsp.tsp_nfp_state import TSPNFPState
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", 
    type=str,
    default="PPO",
    choices=["PPO"], 
    help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--N",
    type=int,
    default=5,
    help="Number of nodes in TSP network"
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed used to generate networkx graph"
)
parser.add_argument(
    "--num-workers", type=int, default=1, help="Number of rllib workers"
)
parser.add_argument(
    "--num-gpus", type=int, default=0, help="Number of GPUs"
)
parser.add_argument(
    "--lr", type=float, default=1e-4, help="learning rate"
)
parser.add_argument(
    "--entropy-coeff", type=float, default=0., help="entropy coefficient"
)
parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.0, help="Reward at which we stop training."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


if __name__ == "__main__":

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    N = args.N
    G = make_complete_planar_graph(N=N, seed=args.seed)

    # Compute the reward baseline with heuristic
    import networkx as nx
    tsp_approx = nx.approximation.traveling_salesman_problem
    path = tsp_approx(G, cycle=True)
    reward_baseline = -sum([G[path[i]][path[i + 1]]["weight"] for i in range(0, N - 1)])
    print(f"Networkx heuristic reward: {reward_baseline:1.3f}")

    ModelCatalog.register_custom_model("TSPGNNModel", TSPGNNModel)

    env_name = f"graphenv_{N}_lr={args.lr:1.1e}_ec={args.entropy_coeff:1.1e}"
    register_env(env_name, lambda config: GraphEnv(config))

    config = {
        "env": env_name,
        "env_config": {
            "state": TSPNFPState(G),
            "max_num_children": G.number_of_nodes(),
        },
        "model": {
            "custom_model": "TSPGNNModel",
            "custom_model_config": {
                "num_messages": 1,
                "embed_dim": 32,
            },
        },
        "num_workers": args.num_workers,  # parallelism
        "num_gpus": args.num_gpus,
        "framework": "tf2",
        "eager_tracing": False,
        "rollout_fragment_length": N,  # a multiple of N (collect whole episodes)
        "train_batch_size": 10 * N * args.num_workers,  # a multiple of N * num workers
        "entropy_coeff": args.entropy_coeff,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 5,
        "lr": args.lr,
        "log_level": "DEBUG"
    }

    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config)

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    tune.run(
        args.run,
        config=ppo_config,
        stop=stop,
        local_dir="/scratch/dbiagion/ray_results"
    )

    ray.shutdown()
