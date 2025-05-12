import time
import os

import numpy as np
import torch
import torch.nn as nn

from args_parser import parse_arguments

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from dataset_modules.dataset_generic_her2 import Generic_MIL_Dataset_Her2
from loop_in_fold import one_fold

def seed_torch(seed=2021):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

def main(args):
    seed_torch(args.seed)
    
    # *** Load dataset ***
    if "h5" in args.h5_folder_name: args.use_h5=True
    else: args.use_h5=False
    print("args.h5_folder_name",args.h5_folder_name,"args.use_h5",args.use_h5)
    
    if args.datasets == 'camelyon16':
        args.n_classes=2
        # args.task = "task_1_tumor_vs_normal"
        dataset = Generic_MIL_Dataset(csv_path = '/u/jcai1/code/usefulxai/code/CLAM/dataset_csv/tumor_vs_normal_dummy_my.csv',
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
        args.split_dir = "/u/jcai1/code/usefulxai/code/my_method/splits/camelyon16_3fold"
    elif args.datasets == 'her2':
        args.n_classes=2
        dataset = Generic_MIL_Dataset_Her2(csv_path = '/u/jcai1/code/usefulxai/code/CLAM/dataset_csv/her2_my.csv',
                                data_dir= '/u/jcai1/code/usefulxai/data/Her2/Yale_HER2_cohort/yale_her2',
                                h5_folder_name = args.h5_folder_name,
                                use_h5 = args.use_h5,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'her2_negative':0, 'her2_positive':1},
                                patient_strat=False,
                                ignore=[],)
        # dataset.load_from_h5(True)
        print("use h5 files:",dataset.use_h5,"h5_folder_name:", dataset.h5_folder_name)
        args.split_dir = "/u/jcai1/code/usefulxai/code/my_method/splits/her2_3fold"
    print('split_dir: ', args.split_dir)
    assert os.path.isdir(args.split_dir)
    
    
    # *** Evaluation matrix *** 
    # acs   List of Accuracy scores: records the accuracy for each fold in cross-validation
    # pre   List of Precision scores: records the precision for each fold
    # rec   List of Recall scores: records the recall for each fold
    # fs    List of F1 Scores: records the F1 score for each fold
    # auc   List of AUC scores (Area Under Curve on the validation set): records the AUC for each fold
    # te_auc  List of Test AUC scores: records the AUC on the test set for each fold
    # te_fs   List of Test F1 Scores: records the F1 score on the test set for each fold
    acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]
    
    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        args.fold_start = ckp['k']
        acs, pre, rec,fs,auc,te_auc,te_fs = ckp['ckc_metric']
        print("auto_resume")
        
    # folds: Allows the user to train on a subset of folds by specifying --k_start and --k_end parameters (defaults to all folds)
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k_fold
    else:
        end = args.k_end
    folds = np.arange(start, end)
    
    # *** Train model for each fold *** 
    for i in folds:
        print('*************************')
        print('Start %d-fold cross validation: fold %d ' % (args.k_fold, i))
        ckc_metric = one_fold(args, i, ckc_metric, dataset)
        print('*************************')

        
    print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
    print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
    print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
    print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
    print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))
    
if __name__ == '__main__':
    
    args = parse_arguments()
    args.model_path = args.model_path + "/" + args.project
    print(args)

    localtime = time.asctime( time.localtime(time.time()) ) 
    print(localtime)
    main(args=args)
