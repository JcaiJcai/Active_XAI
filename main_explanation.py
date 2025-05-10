import time
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from args_parser import parse_arguments

from explanation.perturbation import *
from explanation.shapley import *
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from loop_in_fold import one_fold
from modules import mhim
from utils_clam.utils import get_split_loader

import functools
print = functools.partial(print, flush=True)

def seed_torch(seed=2021):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
def get_data_list_camelyon16(args):
    csv_path = '/u/jcai1/code/usefulxai/code/CLAM/dataset_csv/tumor_vs_normal_dummy_my.csv'
    dataset = Generic_MIL_Dataset(csv_path = csv_path,
                                data_dir= '/u/jcai1/code/usefulxai/data/camelyon16-clam/tumor_vs_normal_resnet_features',
                                h5_folder_name = args.h5_folder_name,
                                use_h5 = args.use_h5,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                                patient_strat=False,
                                ignore=[],) #!
    # dataset.load_from_h5(True)
    print("use h5 files:",dataset.use_h5,"h5_folder_name:", dataset.h5_folder_name)
    
    df = pd.read_csv(csv_path)
    slide_id2_list = df['slide_id2'].tolist()
    
    return dataset, slide_id2_list

def load_trained_model(args, device):
    if args.model == 'mhim':

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'mask_ratio' : args.mask_ratio, # 全局掩码比例
            'n_classes': args.n_classes,
            'temp_t': args.temp_t, # 温度参数
            'act': args.act, # 激活函数类型
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion, # 多头注意力的融合方式
            'da_act': args.da_act, # 数据增强相关的激活函数
            'attn_layer': args.attn_layer, # 注意力层的配置
            'use_human_mask': args.use_human_mask,
            'use_annotation_loss': args.use_annotation_loss,
            'use_attention_loss': args.use_attention_loss,
        }
        
        model = mhim.MHIM(**model_params).to(device)
    elif args.model == 'pure': # 最最简单的模型
        model = mhim.MHIM(select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)
        
    pre_dict = torch.load(args.explained_model)
    if 'model' in pre_dict:
        pre_dict = pre_dict['model'] # 取出实际参数
    info = model.load_state_dict(pre_dict,strict=False)
    print(info)
    
    return model

def main(args):
    print("start running", flush=True)
    if "fold_0" in args.explained_model: fold = "fold_0"
    elif "fold_1" in args.explained_model: fold = "fold_1"
    elif "fold_2" in args.explained_model: fold = "fold_2"
    
    if "transmil" in args.explained_model: model_name = "transmil"
    elif "dsmil" in args.explained_model: model_name = "dsmil"
    elif "abmil" in args.explained_model: model_name = "abmil"
    
    output_dir = f"/u/jcai1/code/usefulxai/code/results/explanations/{model_name}_{fold}_{args.explanation}_{args.search_num}"
    os.makedirs(output_dir, exist_ok=True)
    
    seed_torch(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # *** Load dataset ***
    if "h5" in args.h5_folder_name: args.use_h5=True
    else: args.use_h5=False
    print("args.h5_folder_name",args.h5_folder_name,"args.use_h5",args.use_h5)
    
    if args.datasets == 'camelyon16':
        args.n_classes=2
        args.task = "task_1_tumor_vs_normal"
        dataset, slide_id2_list = get_data_list_camelyon16(args)
    
    # args.split_dir
    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    else:
        args.split_dir = os.path.join('splits', args.split_dir)
    print('split_dir: ', args.split_dir)
    assert os.path.isdir(args.split_dir)
    
    # *** Load Trained Model *** 
    model = load_trained_model(args, device)
    
    # *** Generate Explanation *** 
    if args.explanation == "perturbation":
        ex = Explainer_perturbation(device)
    elif args.explanation == "shap1":
        ex = Explainer_shap(device)
    i = 0
    for slide_id in slide_id2_list:
        slide2patch_scores = {}
        # Check whether heapmap already exists
        output_file = f"{slide_id}.pt"
        output_file = os.path.join(output_dir, output_file)
        
        if os.path.exists(output_file):
            print(f"[INFO] Skipping {slide_id}, already explained.", flush=True)
            continue

        print(slide_id)
        slide_data = dataset.get_data_by_slide_id(slide_id)
        if args.explanation == "perturbation":
            patch_scores = ex.explain(slide_data, model, 'drop') #drop/keep
        elif args.explanation == "shap1":
            patch_scores = ex.explain(slide_data, model, args.search_num)
        patch_scores = patch_scores[0] if len(patch_scores.shape) > 1 else patch_scores
        
        patch_scores = patch_scores.squeeze()
        slide2patch_scores[slide_id] = patch_scores
        
        # *** Visualize patch scores in heatmap ***
        # patch_scores_plot, _ = clean_outliers_fliers(patch_scores)
        # _ = slide_heatmap_thumbnail()
        torch.save(slide2patch_scores, output_file)

        
if __name__ == '__main__':
    
    args = parse_arguments()
    # args.model_path = args.model_path + "/" + args.project
    print(args)

    localtime = time.asctime( time.localtime(time.time()) ) 
    print(localtime)
    main(args=args)
