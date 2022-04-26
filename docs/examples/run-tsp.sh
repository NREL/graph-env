#!/bin/bash --login
#SBATCH --account=rlmolecule
#SBATCH --job-name=tsp
#SBATCH --qos=high
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36

clear
source env.sh
python run_tsp.py \
    --N=10 \
    --use-nfp \
    --seed=0 \
    --num-workers 35 \
    --num-gpus 1 \
    --stop-iters 100000 \
    --stop-timesteps 10000000
