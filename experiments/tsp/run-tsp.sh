#!/bin/bash --login
#SBATCH --account=rlmolecule
#SBATCH --job-name=tsp
#SBATCH --time=2-00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36

clear
source env.sh
python run_tsp.py \
    --N=40 \
    --run=PPO \
    --use-gnn \
    --lr=0.001 \
    --entropy-coeff=0.01 \
    --seed=0 \
    --rollouts-per-worker=10 \
    --num-workers=35 \
    --num-gpus=1 \
    --stop-iters=1000000 \
    --stop-timesteps=100000000 \

