import argparse
import logging
import os
from pathlib import Path

import ray
from graphenv.examples.tsp.graph_utils import make_complete_planar_graph
from graphenv.examples.tsp.tsp_model import TSPModel, TSPQModel
from graphenv.examples.tsp.tsp_nfp_model import TSPGNNModel
from graphenv.examples.tsp.tsp_nfp_state import TSPNFPState
from graphenv.examples.tsp.tsp_state import TSPState
from graphenv.graph_env import GraphEnv
from networkx.algorithms.approximation.traveling_salesman import greedy_tsp
from ray import tune
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.marwil import MARWILConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    choices=["PPO", "DQN", "A3C", "MARWIL"],
    help="The RLlib-registered algorithm to use.",
)
parser.add_argument("--N", type=int, default=5, help="Number of nodes in TSP network")
parser.add_argument(
    "--use-gnn", action="store_true", help="use the nfp state and gnn model"
)
parser.add_argument(
    "--max-num-neighbors",
    type=int,
    default=5,
    help="Number of nearest neighbors for the gnn model",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed used to generate networkx graph"
)
parser.add_argument(
    "--num-workers", type=int, default=1, help="Number of rllib workers"
)
parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument(
    "--entropy-coeff", type=float, default=0.0, help="entropy coefficient"
)
parser.add_argument(
    "--rollouts-per-worker",
    type=int,
    default=1,
    help="Number of rollouts for each worker to collect",
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
parser.add_argument("--log-level", type=str, default="INFO")


if __name__ == "__main__":

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    logging.basicConfig(level=args.log_level.upper())

    ray.init(local_mode=args.local_mode)

    N = args.N
    G = make_complete_planar_graph(N=N, seed=args.seed)

    # Compute the reward baseline with heuristic
    import networkx as nx

    tsp_approx = nx.approximation.traveling_salesman_problem
    path = tsp_approx(G, cycle=True)
    reward_baseline = -sum([G[path[i]][path[i + 1]]["weight"] for i in range(0, N)])
    print(f"Networkx heuristic reward: {reward_baseline:1.3f}")

    path = tsp_approx(G, cycle=True, method=greedy_tsp)
    reward_baseline = -sum([G[path[i]][path[i + 1]]["weight"] for i in range(0, N)])
    print(f"Networkx greedy reward: {reward_baseline:1.3f}")

    # Algorithm-specific config, common ones are in the main config dict below
    if args.run == "PPO":
        run_config = PPOConfig()
        train_batch_size = args.rollouts_per_worker * N * args.num_workers
        sgd_minibatch_size = 16 if train_batch_size > 16 else 2
        run_config.training(entropy_coeff=args.entropy_coeff,
                            sgd_minibatch_size=sgd_minibatch_size,
                            num_sgd_iter=5,
        )
    elif args.run in ["DQN"]:
        run_config = DQNConfig()
        # Update here with custom config
        run_config.training(hiddens=False,
                        dueling=False
        )
        run_config.exploration(exploration_config={"epsilon_timesteps": 250000})
    elif args.run == "A3C":
        run_config = A3CConfig()
    elif args.run == "MARWIL":
        run_config = MARWILConfig()
    else:
        raise ValueError(f"Import agent {args.run} and try again")

    # Define custom_model, config, and state based on GNN yes/no
    if args.use_gnn:
        custom_model = "TSPGNNModel"
        custom_model_config = {"num_messages": 3, "embed_dim": 32}
        ModelCatalog.register_custom_model(custom_model, TSPGNNModel)
        _tag = "gnn"
        state = TSPNFPState(
            lambda: make_complete_planar_graph(N=N),
            max_num_neighbors=args.max_num_neighbors,
        )
    else:
        custom_model_config = {"hidden_dim": 256, "embed_dim": 256, "num_nodes": N}
        custom_model = "TSPModel"
        Model = TSPQModel if args.run in ["DQN", "R2D2"] else TSPModel
        ModelCatalog.register_custom_model(custom_model, Model)
        _tag = f"basic{args.run}"
        state = TSPState(lambda: make_complete_planar_graph(N=N))

    # Register env name with hyperparams that will help tracking experiments
    # via tensorboard
    env_name = f"graphenv_{N}_{_tag}_lr={args.lr}"
    register_env(env_name, lambda config: GraphEnv(config))

    run_config = (
        run_config
        .resources(num_gpus=args.num_gpus)
        .framework("tf2")
        .rollouts(num_rollout_workers=args.num_workers, 
                  # a multiple of N (collect whole episodes)
                  rollout_fragment_length=N)
        .environment(env=env_name,
                     env_config={"state": state, 
                                 "max_num_children": G.number_of_nodes()}
                  )
        .training(lr=args.lr,
                  train_batch_size=args.rollouts_per_worker * N * args.num_workers,
                  model={"custom_model": custom_model, 
                         "custom_model_config": custom_model_config}
                  )
        .evaluation(evaluation_config={"explore": False},
                    evaluation_interval=1, 
                    evaluation_duration=100,
                  )
        .debugging(log_level=args.log_level)
        .framework(eager_tracing=True)
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    tune.run(
        args.run,
        config=run_config.to_dict(),
        stop=stop,
        local_dir=Path("/scratch", os.environ["USER"], "ray_results"),
    )

    ray.shutdown()
