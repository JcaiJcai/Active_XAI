#!/bin/bash
#SBATCH --job-name="dsmil_m0"
#SBATCH --output="/u/jcai1/code/usefulxai/code/my_method/results/scripts/dsmil_mask0"
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
python3 main.py --project=dsmil_mask0_0414 \
--h5_folder_name=h5_files_labels \
--model_path=/u/jcai1/code/usefulxai/code/my_method/results \
--k_fold=3 \
--teacher_init=/u/jcai1/code/usefulxai/code/my_method/results/dsmil_0405/fold_0_model_best_auc.pt \
--mask_ratio_h=0.03 \
--mask_ratio_hr=0.5 \
--mrh_sche \
--mask_ratio=0. \
--mask_ratio_l=0.8 \
--cl_alpha=0.1 \
--mm_sche \
--init_stu_type=fc \
--attn_layer=0 \
--seed=2021