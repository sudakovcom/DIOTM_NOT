#!/bin/bash
#SBATCH --job-name=fid_male2femaleNOT_1gpu
#SBATCH --output=outputs/out.log
#SBATCH --error=outputs/out.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint="type_a|type_b|type_c"

python3 train.py