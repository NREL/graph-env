#!/bin/bash --login
#SBATCH --account=rlmolecule
#SBATCH --job-name=tsp
#SBATCH --qos=high
#SBATCH --time=2:00:00
#SBATCH --nodes=1

clear
source env.sh
python run_tsp.py \
    --N=10 \
    --seed=0 \
    --num-workers 35 \
    --stop-iters 10000 \
    --stop-timesteps 1000000
 