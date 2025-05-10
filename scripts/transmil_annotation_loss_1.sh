#!/bin/bash
#SBATCH --job-name=transmil_annotation_loss_1
#SBATCH --output=/u/jcai1/code/usefulxai/paper_results/pure/transmil_annotation_loss_1.out
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

python3 main.py --project=transmil_annotation_loss_1 --h5_folder_name=h5_files_labels --model_path=/u/jcai1/code/usefulxai/paper_results/pure --k_fold=3 --teacher_init=/u/jcai1/code/usefulxai/paper_results/transmil_0507_1818 --mask_ratio_h=0.03 --mask_ratio_hr=0.5 --mrh_sche --mask_ratio=0. --mask_ratio_l=0.8 --cl_alpha=0.1 --mm_sche --init_stu_type=fc --attn_layer=0 --seed=2021 --use_annotation_loss=True --annotation_alpha=1 --anno_loss_type=energy

