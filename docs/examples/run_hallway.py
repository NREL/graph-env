import argparse
import os

import ray
from graphenv.examples.hallway.hallway_env import HallwayEnv
from graphenv.examples.hallway.hallway_model import HallwayModel
from graphenv.graph_env import GraphEnv
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)

parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=50, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
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

    # Can also register the env creator function explicitly with:
    register_env("hallway", lambda config: HallwayEnv(config))
    ModelCatalog.register_custom_model("my_model", HallwayModel)

    config = {
        "env": "hallway",  # or "corridor" if registered above
        "env_config": {
            "size": 5,
            "max_steps": 100,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "custom_model_config": {"hidden_dim": 32},
        },
        "num_workers": 1,  # parallelism
        "framework": "tf2",
        "eager_tracing": False,
        "eager_max_retraces": 20,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        # use fixed learning rate instead of grid search (needs tune)
        ppo_config["lr"] = 1e-3
        trainer = ppo.PPOTrainer(config=ppo_config, env=HallwayEnv)
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        results = tune.run(args.run, config=config, stop=stop)

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
