#!/bin/bash
for search_num in 2000 5000 10000; do
    date_str=$(date +%m%d)
    project_name="transmil_shap1_${search_num}"
    # 每个实验的脚本文件路径
    output_script="/u/jcai1/code/usefulxai/code/my_method/scripts/generate_explanation/${project_name}.sh"

    # 生成脚本文件
    cat > "$output_script" <<EOT
#!/bin/bash
#SBATCH --job-name=${project_name}
#SBATCH --output=/u/jcai1/code/usefulxai/code/results/out_files/generate_explanation/${project_name}.out
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

python3 main_explanation.py --project=${project_name} \
--explained_model=/u/jcai1/code/usefulxai/code/results/trained_models/transmil_0405/fold_0_model_best_auc.pt \
--h5_folder_name=h5_files_labels \
--k_fold=3 \
--title=transmil \
--model=pure \
--baseline=selfattn \
--seed=2021 \
--explanation=shap1 \
--search_num=${search_num}
EOT

    # 赋予执行权限并提交
    chmod +x "$output_script"
    sbatch "$output_script"
done