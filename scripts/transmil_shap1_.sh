#!/bin/bash
#SBATCH --job-name=transmil_shap1_
#SBATCH --output=/u/jcai1/code/usefulxai/code/results/scripts/transmil_shap1_.out
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=scratch
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bdem-delta-gpu
#SBATCH --no-requeue
#SBATCH -t 24:00:00

source activate clam_latest
cd /u/jcai1/code/usefulxai/code/my_method
python3 main_explanation.py --project=transmil_shap1_ --explained_model=/u/jcai1/code/usefulxai/code/results/transmil_0405/fold_0_model_best_auc.pt --h5_folder_name=h5_files_labels --k_fold=3 --title=transmil --model=pure --baseline=selfattn --seed=2021 --explanation=shap1 --search_num=10000
