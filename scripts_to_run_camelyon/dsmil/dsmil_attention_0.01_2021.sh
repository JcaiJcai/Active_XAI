#!/bin/bash
#SBATCH --job-name=dsmil_attention_0.01
#SBATCH --output=/u/jcai1/code/usefulxai/paper_results/pure/dsmil/dsmil_attention_0.01.out
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=scratch
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bcqc-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00

source activate clam_latest
cd /u/jcai1/code/usefulxai/code/my_method

python3 main.py --project=dsmil_attention_0.01 --model_path=/u/jcai1/code/usefulxai/paper_results/trained_models --k_fold=3 --teacher_init=/u/jcai1/code/usefulxai/paper_results/dsmil_0507_1818 --cl_alpha=0.1 --mm_sche --init_stu_type=fc --seed=2021 --use_attention_loss=True --attn_alpha=0.01 --baseline=dsmil
