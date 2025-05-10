# 只使用attnloss，不用human_annotation mask
#  train_loss = args.cls_alpha * logit_loss + attn_loss*args.attn_alpha
#!/bin/bash
for alpha in 0.0001 0.001 0.01 0.1 1 10; do
    project_name="transmil_annotation_loss_${alpha}"
    # 每个实验的脚本文件路径
    output_script="/u/jcai1/code/usefulxai/code/my_method/scripts/${project_name}.sh"

    # 生成脚本文件
    cat > "$output_script" <<EOT
#!/bin/bash
#SBATCH --job-name=${project_name}
#SBATCH --output=/u/jcai1/code/usefulxai/paper_results/pure/${project_name}.out
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

python3 main.py --project=${project_name} \
--h5_folder_name=h5_files_labels \
--model_path=/u/jcai1/code/usefulxai/paper_results/pure \
--k_fold=3 \
--teacher_init=/u/jcai1/code/usefulxai/paper_results/transmil_0507_1818 \
--mask_ratio_h=0.03 \
--mask_ratio_hr=0.5 \
--mrh_sche \
--mask_ratio=0. \
--mask_ratio_l=0.8 \
--cl_alpha=0.1 \
--mm_sche \
--init_stu_type=fc \
--attn_layer=0 \
--seed=2021 \
--use_annotation_loss=True \
--annotation_alpha=${alpha} \
--anno_loss_type=energy

EOT

    # 赋予执行权限并提交
    chmod +x "$output_script"
    sbatch "$output_script"
done