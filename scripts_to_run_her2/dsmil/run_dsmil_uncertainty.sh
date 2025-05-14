# 只使用attnloss，不用human_annotation mask
#  train_loss = args.cls_alpha * logit_loss + attn_loss*args.attn_alpha
#!/bin/bash
for top_k_for_annotation in 20 30 40; do
    for seed in 2021; do
        annotation_alpha=0.01
        attn_alpha=0.01
        project_name="dsmil_uncertainty_top${top_k_for_annotation}_${attn_alpha}_${annotation_alpha}_${seed}"
        # 每个实验的脚本文件路径
        output_script="/u/jcai1/code/usefulxai/code/my_method/scripts_to_run_her2/dsmil/${project_name}.sh"

    # 生成脚本文件
    cat > "$output_script" <<EOT
#!/bin/bash
#SBATCH --job-name=${project_name}
#SBATCH --output=/u/jcai1/code/usefulxai/paper_results_her2/dsmil/${project_name}.out
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

python3 main.py --project=${project_name} \
--dataset=her2 \
--h5_folder_name=h5_files_labels \
--model_path=/u/jcai1/code/usefulxai/paper_results_her2/trained_models \
--k_fold=3 \
--teacher_init=/u/jcai1/code/usefulxai/paper_results_her2/pure_model/dsmil_0511_0652 \
--cl_alpha=0.1 \
--mm_sche \
--init_stu_type=fc \
--seed=${seed} \
--use_attention_loss=True \
--use_annotation_loss=True \
--attn_alpha=${attn_alpha} \
--annotation_alpha=${annotation_alpha} \
--top_k_for_annotation=${top_k_for_annotation} \
--uncertainty \
--start_using_annotation=40 \
--lr=1e-4 \
--baseline=dsmil


EOT

    # 赋予执行权限并提交
    chmod +x "$output_script"
    sbatch "$output_script"
    done
done