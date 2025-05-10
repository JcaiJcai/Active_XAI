#!/bin/bash
#SBATCH --job-name="transmil"
#SBATCH --output="/u/jcai1/code/usefulxai/paper_results/pure/scripts/transmil2.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bdem-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00

source activate clam_latest
cd /u/jcai1/code/usefulxai/code/my_method

python main.py --project=transmil2 \
--model_path=/u/jcai1/code/usefulxai/paper_results \
--k_fold=3 \
--title=transmil \
--model=pure \
--baseline=selfattn \
--seed=2021