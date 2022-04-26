import argparse

import ray
from graphenv.examples.tsp.graph_utils import make_complete_planar_graph
from graphenv.graph_env import GraphEnv
from graphenv.examples.tsp.tsp_model import TSPModel
from graphenv.examples.tsp.tsp_state import TSPState
from graphenv.examples.tsp.tsp_nfp_model import TSPGNNModel
from graphenv.examples.tsp.tsp_nfp_state import TSPNFPState
from ray import tune
from ray.rllib.agents import ppo, dqn
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", choices=["PPO"], 
    help="The RLlib-registered algorithm to use."
)
parser.add_argument("--N", type=int, default=5, help="Number of nodes in TSP network")
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed used to generate networkx graph"
)
parser.add_argument(
    "--use-nfp",
    action="store_true",
    help="Whether to use the GNN model and state.",
)
parser.add_argument(
    "--num-workers", type=int, default=1, help="Number of rllib workers"
)
parser.add_argument(
    "--num-gpus", type=int, default=0, help="Number of GPUs"
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
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
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

    # Customize based on whether using NFP or not.
    if args.use_nfp:
        state = TSPNFPState(G)
        custom_model_config = {
            "num_messages": 1,
            "embed_dim": 32
        }
        custom_model = "TSPGNNModel"
        ModelCatalog.register_custom_model(custom_model, TSPGNNModel)
    else:
        state = TSPState(G)
        custom_model_config = {
            "num_nodes": args.N,
            "hidden_dim": 256,
            "embed_dim": 256,
        }
        # Model and env registration.
        custom_model = "TSPModel"
        ModelCatalog.register_custom_model(custom_model, TSPModel)

    # Create a baseline solution using networkx heuristics
    import networkx as nx

    # Compute the reward baseline with heuristic
    tsp = nx.approximation.traveling_salesman_problem
    path = tsp(G, cycle=True)
    reward_baseline = -sum([G[path[i]][path[i + 1]]["weight"] for i in range(0, N - 1)])
    print(f"Networkx heuristic reward: {reward_baseline:1.3f}")

    # Put the size of the graph in the name
    env_name = f"tsp-{N}"
    register_env(env_name, lambda config: GraphEnv(config))

    config = ppo.DEFAULT_CONFIG.copy()
    
    config.update({
        "env": env_name,  # or "corridor" if registered above
        "env_config": {
            "state": state,
            "max_num_children": N
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": args.num_gpus,  # int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": custom_model,
            "custom_model_config": custom_model_config
        },
        "num_workers": args.num_workers,  # parallelism
        "framework": "tf2",
        "rollout_fragment_length": N,  # a multiple of tour length
        "train_batch_size": 10 * N * args.num_workers,  # a multiple of num workers
        "entropy_coeff": 0.,
        "lr": 1e-3
    })

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": min(reward_baseline, args.stop_reward),
    }

    results = tune.run(
        args.run,
        config=config,
        stop=stop,
        local_dir="/scratch/dbiagion/ray_results"
    )

    ray.shutdown()
