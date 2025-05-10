import argparse 
import datetime

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    
    # Dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
    parser.add_argument('--h5_folder_name', type=str, default='h5_files_labels', choices=['h5_files', 'h5_files_annotations', 'h5_files_labels','None']) # h5_files_annotations: 根据annotation过滤后的数据集h5_files_annotations，对于tumor数据，只保留注释区域，对于normal数据则不变；h5_files_labels: 根据annotation为每个坐标赋予标签（0，1，2）
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]') # 是否强制让所有 patch 保持相同的尺寸
    parser.add_argument('--k_fold', type=int, default=10, help='number of folds (default: 10)') # Jie # 交叉验证（cross-validation）要分几折（fold），默认是 3/10 折
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)') # Jie
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)') # Jie
    parser.add_argument('--split_dir', type=str, default=None, # Jie
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping']) # Jie
    parser.add_argument('--label_frac', type=float, default=1.0, # Jie
                    help='fraction of training labels (default: 1.0)')
    
    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='mhim', type=str, help='Model name') ###################
    # parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
    #                 help='type of model (default: clam_sb, clam w/ single attention branch)') ###################
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase') # train的时候每轮都要test
    parser.add_argument('--best_thr_val', action='store_true', help='Cal the best thr with val set in the test phase. Thanks Weiyi Wu!')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool') # Jie
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling') # Jie
    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--baseline', default='selfattn', type=str, help='Baselin model [attn,selfattn]') ################
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # MHIM
    # Mask ratio
    parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
    parser.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio')
    parser.add_argument('--mask_ratio_h', default=0., type=float, help='High attention mask ratio')
    parser.add_argument('--mask_ratio_hr', default=1., type=float, help='Randomly high attention mask ratio')
    parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM') # 是否动态调整 mask_ratio_h
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
    parser.add_argument('--attn_layer', default=0, type=int) # 根据第几层的attn做mask，默认为0
    
    # Siamese framework
    parser.add_argument('--cl_alpha', default=0., type=float, help='Auxiliary loss alpha')
    parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
    parser.add_argument('--teacher_init', default='none', type=str, help='Path to initial teacher model')
    parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization') # 默认值为 False，表示默认会初始化 teacher 模型。如果你加上 --no_tea_init，就会跳过 teacher 加载 checkpoint 的步骤
    parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]') # 学生模型的初始化方式。'none'	不进行任何初始化，student 从头训练。'fc'只初始化前面的 patch embedding / projection 层。'all'用 teacher checkpoint 初始化整个 student 模型
    parser.add_argument('--tea_type', default='none', type=str, help='[none,same]') # teacher 的来源方式，same是直接复制 student
    parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
    parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay') # 启用一个 EMA 衰减系数的 cosine 调度策略

    # Misc
    parser.add_argument('--title', default='default', type=str, help='Title of exp')
    parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    # parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader') # !!!!!!!!
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--model_path', default=None, type=str, help='Output path')
    
    # Jie
    parser.add_argument('--explanation', type=str, default="attention", choices=["attention", "shap1", "shap2"])
    parser.add_argument('--use_human_mask', type=bool, default=False)
    parser.add_argument('--use_attention_loss', type=bool, default=False, help='Enable attention-based auxiliary loss')
    parser.add_argument('--attn_alpha', default=1., type=float, help='Weight for attention loss')
    parser.add_argument('--use_annotation_loss', type=bool, default=False)
    parser.add_argument('--annotation_alpha', default=1., type=float, help='Weight for annotation_loss')
    parser.add_argument('--anno_loss_type', default="energy", type=str, choices=["L1", "L2", "energy", "entropy"])
    parser.add_argument('--energy_alphas', default=None, type=list) # 
    parser.add_argument('--explained_model', default='none', type=str, help='Path to explained model')
    parser.add_argument('--uncertainty', default=False, type=bool, help='Whether calculate uncertainty score or not')
    # parser.add_argument('--uncertainty', action='store_true', help='Enable uncertainty score calculation')
    parser.add_argument('--top_k_for_annotation', default=50, type=int, help='Number of images to be annotated')
    parser.add_argument('--start_using_annotation', default=40, type=int, help='From which epoch to use annotation')
    
    # Shap
    parser.add_argument('--search_num', default=None, type=int, help='Number of patches to calculate shap values')
  
    
    args = parser.parse_args()
    now = datetime.datetime.now().strftime('%m%d_%H%M')
    args.project = f"{args.project}_{now}"
    return args

if __name__ == "__main__":
    
    args = parse_arguments()