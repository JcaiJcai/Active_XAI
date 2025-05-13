import time
import torch
import wandb
import numpy as np
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
# from dataloader import *
from torch.utils.data import DataLoader, RandomSampler
import argparse, os
from torch.nn.functional import one_hot
# from torch.cuda.amp import GradScaler
from contextlib import suppress
import time
import random
import copy

from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict

from utils_clam.utils import get_split_loader
from utils_clam import *
from utils_mhim import *
from uncertainty import *

from modules import attmil,clam,mhim,dsmil,transmil,mean_max
from explanation import shapley

import pdb
from scipy import stats
from sklearn.metrics import pairwise_distances


def seed_torch(seed=2021):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

def one_fold(args,fold_k,ckc_metric,dataset):
    seed_torch(args.seed)
    # loss_scaler = GradScaler() if args.amp else None # AMP (Automatic Mixed Precision Training)
    loss_scaler = None
    # amp_autocast = torch.cuda.amp.autocast if args.amp else suppress # Autocast under mixed precision
    amp_autocast = suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # TODO
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric
    
    # *** Load Data ***
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, use_h5=args.use_h5, h5_folder_name=args.h5_folder_name, csv_path='{}/splits_{}.csv'.format(args.split_dir, fold_k))
    print('\nInit Loaders...', end=' ')
    
    print("train_dataset",train_dataset)
    print("val_dataset",val_dataset)
    print("test_dataset",test_dataset)
    # ! Note: Need to check whether we have a fixed seed
    train_loader = get_split_loader(train_dataset, args.batch_size, training=True, testing = args.testing, weighted = args.weighted_sample) # batch_size = 1
    val_loader = get_split_loader(val_dataset, args.batch_size, testing = args.testing)
    test_loader = get_split_loader(test_dataset, args.batch_size, testing = args.testing)
    print('Done!')
    print(len(val_loader))
    
    mm_sche = None
    # Load the previously trained model from a specific fold as the initial teacher model
    if not args.teacher_init.endswith('.pt'):
        _str = 'fold_{fold}_model_best_auc.pt'.format(fold=fold_k)
        _teacher_init = os.path.join(args.teacher_init,_str)
    else:
        _teacher_init =args.teacher_init
    
    # ** Load model **
    if args.model == 'mhim':
        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t, # Temperature parameter
            'act': args.act, # Activation function
            'head': args.n_heads,
            'da_act': args.da_act, # Activation function related to data augmentation
            'use_human_mask': args.use_human_mask,
            'use_annotation_loss': args.use_annotation_loss,
            'use_attention_loss': args.use_attention_loss,
        }
        
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.num_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)# mm scheduler using cosine decay; may start with warmup (start_warmup_value)

        model = mhim.MHIM(**model_params).to(device)
    elif args.model == 'pure': # pure model
        model = mhim.MHIM(select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)
    elif args.model =='clam_sb':
        # model = CLAM_SB(**model_dict,  instance_loss_fn=instance_loss_fn)
        model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'clam_mb':
        # model = CLAM_MB(**model_dict,instance_loss_fn=instance_loss_fn)
        model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)        
    
    # Initialize the student model parameters
    if args.strategy != 'ours':
        pre_dict = torch.load(_teacher_init)
        if 'model' in pre_dict:
            pre_dict = pre_dict['model'] # Extract actual model parameters
        info = model.load_state_dict(pre_dict,strict=False) # Initialize the entire model
        if not args.no_log:
            print(info)
            
    if args.init_stu_type != 'none':
        if not args.no_log:
            print('######### Model Initializing.....')
        pre_dict = torch.load(_teacher_init)
        if 'model' in pre_dict:
            pre_dict = pre_dict['model'] # Extract actual model parameters
            
        new_state_dict ={}
        if args.init_stu_type == 'fc': # Initialize only the patch_to_emb module
        # only patch_to_emb
            for _k,v in pre_dict.items():
                _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                new_state_dict[_k]=v
            info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
        else: 
        # init all
            info = model.load_state_dict(pre_dict,strict=False) # Initialize the entire model
        if not args.no_log:
            print(info)
            
    # *** Init teacher ***
    if args.model == 'mhim':
        model_tea = deepcopy(model) # Create a teacher model: directly copy the current student model as the initial version (deep copy, independent weights)
        if not args.no_tea_init and args.tea_type != 'same': # !!
            print('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(_teacher_init)
                if 'model' in pre_dict:
                    pre_dict = pre_dict['model']
                info = model_tea.load_state_dict(pre_dict,strict=False) # Loading weights from a checkpoint into model_tea; allow partial loading
                if not args.no_log:
                    print(info)
            except:
                if not args.no_log:
                    print('########## Init Error')
        opt_evid = torch.optim.Adam(model_tea.evidence_head.parameters(), lr=1e-4)

    else:
        model_tea = None
        opt_evid = None
    
    # set the teacher to None in the active learning baseline
    model_tea = None

    # *** Loss
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    
    # *** Optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        
    # *** Learning Rate Scheduler: dynamically adjusts the learning rate of the optimizer during training
    if args.lr_sche == 'cosine': # 	Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step': # Step decay	
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const': # Constant learning rate
        scheduler = None

    # *** Early stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    epoch_start = 0
    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_thr,opt_epoch = 0, 0, 0, 0,0,0,0
    opt_te_auc,opt_tea_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs  = 0., 0., 0., 0., 0.
    
    train_time_meter = AverageMeter() # Timer
    fixed_topk_ids = None
    opt_uncertainty = None
    opt_features = None
    opt_probs = None
    for epoch in range(epoch_start, args.num_epoch):
        # ****** TRAIN (Teacher and Student)******
        # uncertainty: 当前模型计算出来的uncertainty
        train_loss, start, end, fixed_topk_ids, uncertainty, features, probs = train_loop(args,model,model_tea,train_loader,optimizer,opt_evid, device,amp_autocast,criterion,loss_scaler,scheduler,fold_k,mm_sche,epoch, fixed_topk_ids, opt_uncertainty, opt_features, opt_probs)
        train_time_meter.update(end-start) # Training time
        
        if epoch % 20 == 0:
            print(f"[Epoch {epoch}] fixed_topk_ids:", fixed_topk_ids)
            print(f"[Epoch {epoch}] uncertainty:", uncertainty) # fixed_topk_ids和uncertainty在固定epoch之后应该保持不变
        # ****** EVALUATE (Student) ******
        # stop: whether early stopping was triggered; threshold_optimal: the optimal classification threshold (for binary classification)
        stop,accuracy, auc_value, precision, recall, fscore, test_loss, threshold_optimal = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch,model_tea)
        
        # # ****** EVALUATE (Teacher) ******
        # if model_tea is not None: # If a teacher model exists
        #     # Perform a validation run on the teacher model
                        

        #     _,accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea,_ = val_loop(args,model_tea,val_loader,device,criterion,None,epoch,model_tea)
            
        if auc_value > opt_tea_auc: # If the current teacher model's AUC is better than the previous best, update it
            opt_tea_auc = auc_value
            opt_uncertainty = copy.deepcopy(uncertainty) # !opt_uncertainty: The uncertainty of the current optimal model. Once fixed_topk_ids is fixed, data will no longer be selected based on it
            opt_features = copy.deepcopy(features)
            opt_probs = copy.deepcopy(probs)
        # ********* TEST (Student) *********
        if args.always_test:
            # Test student model
            _te_accuracy, _te_auc_value, _te_precision, _te_recall, _te_fscore,_te_test_loss_log = test(args,model,test_loader,device,criterion,model_tea)

            if _te_auc_value > opt_te_auc:
                opt_te_auc = _te_auc_value
                opt_te_fs = _te_fscore
            
            if model_tea is not None: # Test teacher model
                _te_tea_accuracy, _te_tea_auc_value, _te_tea_precision, _te_tea_recall, _te_tea_fscore,_te_tea_test_loss_log = test(args,model_tea,test_loader,device,criterion,model_tea)
            
                if _te_tea_auc_value > opt_te_tea_auc:
                    opt_te_tea_auc = _te_tea_auc_value
                    opt_te_tea_fs = _te_tea_fscore

        print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))
        
        # When the AUC on the validation set exceeds the historical best and training has reached a certain stage, save the current model as the "best model".
        # args.save_best_model_stage is a float between 0 and 1 (e.g., 0.5), indicating that the "best model" will only be saved after reaching that proportion of the total training epochs.
        if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.num_epoch:
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_thr = threshold_optimal
            opt_epoch = epoch

        if not os.path.exists(args.model_path):
            os.mkdir(args.model_path)
            
        best_pt = {
            'model': model.state_dict(), # student 模型的参数
            'teacher': model_tea.state_dict() if model_tea is not None else None, # 如果有 teacher 模型，就把它的参数也保存；否则设为 None
        }
        torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=fold_k))) # 保存这组参数到本地文件系统
        
        # save checkpoint 保存随机状态（用于完全复现）
        random_state = {
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'py': random.getstate(),
            'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
        }
        ckp = { # 构造完整 checkpoint 信息
            'model': model.state_dict(), # 当前 student 模型参数
            'lr_sche': scheduler.state_dict(), # 学习率调度器状态
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1, # 当前轮次 +1，resume 时从下轮开始
            'k': fold_k, # 当前是第几折 fold
            'early_stop': early_stopping.state_dict(), # # early stopping 的内部状态
            'random': random_state,
            'ckc_metric': [acs,pre,rec,fs,auc,te_auc,te_fs], #  # 当前累计的各类交叉验证指标
            'val_best_metric': [optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch], # 验证集最优结果
            'te_best_metric': [opt_te_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs], # 测试集最优结果
            'wandb_id': wandb.run.id if args.wandb else '', # wandb 的 run ID，用于恢复日志连接
        }
        
        torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))
        if stop: break
        
    # 加载当前折的最优模型参数文件，并把它恢复到 student 模型 model 和（如果有）teacher 模型 model_tea 中
    best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=fold_k))) # 加载当前第 k 折交叉验证的最优模型权重文件
    info = model.load_state_dict(best_std['model']) # 将保存的 student 模型参数加载到当前模型 model 中
    print(info)
    if model_tea is not None and best_std['teacher'] is not None:
        info = model_tea.load_state_dict(best_std['teacher'])
        print(info)
    
    accuracy, auc_value, precision, recall, fscore, test_loss_log = test(args,model,test_loader,device,criterion,model_tea,opt_thr)
    
    print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
    
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)
    
    if args.always_test: # 如果开启 --always_test，也保存额外的测试指标（例如 teacher 的）
        te_auc.append(opt_te_auc)
        te_fs.append(opt_te_fs)
        
    return [acs,pre,rec,fs,auc,te_auc,te_fs]

def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll

def train_loop(args,model,model_tea,loader,optimizer,optimizer_teacher,device,amp_autocast,criterion,loss_scaler,scheduler,fold_k,mm_sche,epoch, fixed_topk_ids=None, opt_uncertainty=None, opt_features=None, opt_probs=None):
    # ! opt_uncertainty是用于选topk annotation的
    start = time.time()

    loss_cls_meter = AverageMeter() # logit loss
    loss_cl_meter = AverageMeter() # cls_loss
    patch_num_meter = AverageMeter() # 输入的 patch 数
    keep_num_meter = AverageMeter() # “保留”的 patch 数（例如某些掩码操作后剩下的）
    mm_meter = AverageMeter() # EMA momentum value
    
    train_loss_log = 0. # 最终返回的平均损失
    
    # 将主模型和 teacher 模型都设置为训练模式
    model.train()
    if model_tea is not None:
        model_tea.train()
        
    all_uncertainties = {}
    all_features = {}
    all_probs = {}
    if args.uncertainty:
        if epoch<args.start_using_annotation: 
            topk_ids = None
        elif epoch==args.start_using_annotation:
            print("\n" + "="*30)
            print(f"[START ANNOTATION] Epoch {epoch}: Human annotation begins on selected samples!")
            if args.strategy == 'ours':
                # Choose top-k data according to current_uncertainties for human annotation
                topk_slide_ids = sorted(opt_uncertainty.items(), key=lambda x: x[1], reverse=True)[:args.top_k_for_annotation]
                topk_ids = [slide_id for slide_id, uncertainty in topk_slide_ids]
            elif args.strategy == 'random': # Random assign annotations
                topk_ids = np.random.choice(list(opt_uncertainty.keys()), size=args.top_k_for_annotation, replace=False)
            elif args.strategy == 'entropy':
                all_slide_id = []
                all_entorpy = []
                for slide_id, slide_logits in opt_uncertainty.items():
                    probs = F.softmax(slide_logits, dim=1)
                    log_probs = torch.log(probs)
                    entorpy = (probs*log_probs).sum(1)
                    all_slide_id.append(slide_id)
                    all_entorpy.append(entorpy)
                topk_ids = np.array(all_slide_id)[torch.tensor(all_entorpy).sort()[1][:args.top_k_for_annotation]]
            elif args.strategy == 'coreset':
                all_slide_id = []
                all_features_list = []
                for slide_id, feature in opt_features.items():
                    all_features_list.append(feature.numpy())
                    all_slide_id.append(slide_id)
                all_features_list = np.concatenate(all_features_list)
                all_slide_id = np.array(all_slide_id)
                
                m = np.shape(all_features_list)[0]
                min_dist = np.tile(float("inf"), m)
                idxs = []
                for i in range(args.top_k_for_annotation):
                    idx = min_dist.argmax()
                    idxs.append(idx)
                    dist_new_ctr = pairwise_distances(all_features_list, all_features_list[[idx], :])
                    for j in range(m):
                        min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
                topk_ids = all_slide_id[idxs]
            elif args.strategy == 'badge':
                all_slide_id = []
                embDim = list(opt_features.values())[0].shape[-1]
                embedding = np.zeros([len(opt_uncertainty), embDim * args.n_classes])
                for i, (slide_id, slide_logits) in enumerate(opt_uncertainty.items()):
                    features = opt_features[slide_id].reshape(-1)
                    batchProbs = F.softmax(slide_logits, dim=1).numpy().reshape(-1)
                    maxInds = np.argmax(batchProbs)
                    for c in range(args.n_classes):
                        if c == maxInds:
                            embedding[i][embDim * c : embDim * (c+1)] = deepcopy(features) * (1 - batchProbs[c])
                        else:
                            embedding[i][embDim * c : embDim * (c+1)] = deepcopy(features) * (-1 * batchProbs[c])
                    all_slide_id.append(slide_id)
                indices = init_centers(embedding, args.top_k_for_annotation)
                topk_ids = np.array(all_slide_id)[indices]
            elif args.strategy == 'bald':
                all_slides = []
                all_prob_list = []
                for slide_id, prob in opt_probs.items():
                    all_prob_list.append(prob)
                    all_slides.append(slide_id)
                probs = torch.stack(all_prob_list, dim=1)
                pb = probs.mean(0)
                entropy1 = (-pb*torch.log(pb)).sum(1)
                entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
                entorpy_diff = entropy2 - entropy1
                topk_ids = np.array(all_slides)[entorpy_diff.sort()[1][:args.top_k_for_annotation]]
            print("Used ncertainties:", opt_uncertainty)
            print("Selected indices:", topk_ids)
        elif epoch>args.start_using_annotation:
            topk_ids=fixed_topk_ids

    for i, data in enumerate(loader): # i - batch index
        # data[0]: features(36710, 1024)
        # data[1]: label(1)
        # data[2]: coords(36710, 2)
        # data[3]: labels_mask(36710,)
        
        optimizer.zero_grad() # 清除上一个 batch 的梯度
        if optimizer_teacher is not None:
            optimizer_teacher.zero_grad()

        # bag: 是一个 batch 的 feature（MIL 模型中，一张 WSI = 一个 bag，bag 内有多个 patch）
        bag=data[0].to(device)  # b*n*1024
        if bag.ndim == 2: # if batch_size==1
            bag = bag.unsqueeze(0)
        batch_size=bag.size(0)
        
        label=data[1].to(device)
        # coords=data[2].to(device)
        labels_mask=data[3].to(device)
        slide_id2=data[4][0]

        with amp_autocast(): # AMP
            # shuffle patches
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)
            
            logit_loss = None
            
            # model_tea & model: MHIM
            if args.model == 'mhim':
                if model_tea is not None:
                    # ***** Teacher Model (generate prediction and attention without back propagation) *****
                    # ! attn -> explanation
                    if args.explanation == "attention":
                        if args.uncertainty: # and epoch>0.5*args.num_epoch-1:
                            # Use teacher model to generate prediction and attention
                            alpha, cls_tea, attn = model_tea.forward_teacher_edl(bag)
                            K = alpha.shape[1] # 类别数
                            S = torch.sum(alpha, dim=1, keepdim=True)  # 每个样本的总证据强度 S_i
                            uncertainty = K / (S + 1e-8) # [B, 1] # uncertainty 越大，模型越不确定
                            # print(slide_id2,uncertainty.item())
                            all_uncertainties[slide_id2] = uncertainty.item()
                        else:
                            cls_tea, attn = model_tea.forward_teacher(bag)
                        # attn_sum = attn[1].sum(dim=-1)
                        # print("!!!!!!!attn_sum",attn_sum) # attention is not normalized
                    elif args.explanation == "shap1": # 用教师模型生成shap解释
                        # cls_tea, attn = model_tea.forward_teacher(bag)
                        # # print("attn",len(attn), attn) # len是2说明这个attention对应的是两层的
                        # attn_avg_lastlayer = attn[-1].mean(dim=1).view(-1)
                        # attn_index = np.argsort(-attn_avg_lastlayer.detach().cpu().numpy())
                        # score = shapley.shapley_value(attn_index, bag, label, model_tea, device, args.baseline, subset_num=10).to(attn[0].device) # (len(search_indices), )
                        # attn = [score.unsqueeze(0).unsqueeze(0).expand(1, 8, -1) for _ in range(2)] # （1，8, 16124）
                        pt_path = "/u/jcai1/code/usefulxai/code/results/explanations/transmil_fold_"+str(fold_k)+"_shap1_2000/"+slide_id2+".pt"
                        # print(pt_path)
                        shap_score = torch.load(pt_path, map_location='cpu', weights_only=False)
                        shap_score = list(shap_score.values())[0]
                        shap_score = torch.tensor(shap_score, device=device)
                        shap_attn = shap_score.unsqueeze(0).unsqueeze(0).expand(1, 8, -1)
                        attn = [shap_attn.clone() for _ in range(2)]
                        cls_tea, _ = model_tea.forward_teacher(bag)
                else:
                    attn,cls_tea = None, None
                    
                cls_tea = None if args.cl_alpha == 0. else cls_tea

                # ***** Student Model: Forward *****
                if args.use_attention_loss or args.use_annotation_loss:
                    if args.baseline == 'dsmil':
                        # logits 是 DSMIL 的主类预测 + instance-level 预测。用两个都计算 loss
                        logits, cls_loss, attn_loss, annotation_loss, patch_num, keep_num = model.forward_with_distill_loss(bag,attn,labels_mask,args.anno_loss_type,cls_tea[0],i=epoch*len(loader)+i) # !!!!!
                        logit_loss = 0.5*criterion(logits[0].view(batch_size,-1),label) + 0.5*criterion(logits[1].view(batch_size,-1),label)
                    else:
                        logits, cls_loss, attn_loss, annotation_loss, patch_num, keep_num = model.forward_with_distill_loss(bag,attn,labels_mask,args.anno_loss_type,cls_tea,i=epoch*len(loader)+i)
                    
                else:
                    if args.baseline == 'dsmil':
                        # logits 是 DSMIL 的主类预测 + instance-level 预测。用两个都计算 loss
                        logits, cls_loss,patch_num,keep_num = model(bag,attn,labels_mask,cls_tea[0],i=epoch*len(loader)+i) # !!!!!
                        logit_loss = 0.5*criterion(logits[0].view(batch_size,-1),label) + 0.5*criterion(logits[1].view(batch_size,-1),label)
                    else:
                        logits, cls_loss,patch_num,keep_num = model(bag,attn,labels_mask,cls_tea,i=epoch*len(loader)+i)
                 
            elif args.model == 'pure':
                if args.baseline == 'dsmil':
                    logits, cls_loss, patch_num, keep_num, attn, features = model.pure(bag)
                    logit_loss = 0.5*criterion(logits[0].view(batch_size,-1),label) + 0.5*criterion(logits[1].view(batch_size,-1),label)
                    logits = logits[0]                    
                else:
                    logits, cls_loss,patch_num,keep_num, attn, features = model.pure(bag)
                all_uncertainties[slide_id2] = logits.detach().cpu()
                all_features[slide_id2] = features.detach().cpu()
                if args.strategy == 'bald':
                    n_drop = 10
                    probs = torch.zeros([n_drop, args.n_classes])
                    with torch.no_grad():
                        for i in range(n_drop):
                            bal_logits, _, _, _, _, _ = model.pure(bag)
                            if args.baseline == 'dsmil':
                                bal_logits = bal_logits[0]
                            probs[i] += F.softmax(bal_logits, dim=1).cpu().data.reshape(-1)
                    all_probs[slide_id2] = probs

            elif args.model in ('clam_sb','clam_mb','dsmil'):
                logits,cls_loss,patch_num = model(bag,label,criterion)
                keep_num = patch_num
            else:
                logits = model(bag)
                cls_loss,patch_num,keep_num = 0.,0.,0.
            
            # classification loss
            if logit_loss is None:
                if args.loss == 'ce':
                    logit_loss = criterion(logits.view(batch_size,-1),label)
                elif args.loss == 'bce':
                    logit_loss = criterion(logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))

        # Overall Loss
        # ! cls_loss is not used anymore in our code!!!!
        
        # * Annotation_loss & attention_loss. No uncertainty
        if not args.uncertainty:
            if args.use_attention_loss or args.use_annotation_loss:
                # print("logit_loss",logit_loss)
                # print("cls_loss",cls_loss)
                # print("attn_loss",attn_loss)
                # train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha + attn_loss*attn_alpha
                if not args.use_attention_loss: 
                    args.attn_alpha = 0.
                if not args.annotation_alpha: 
                    args.annotation_alpha = 0.
                train_loss = args.cls_alpha * logit_loss + attn_loss*args.attn_alpha + annotation_loss*args.annotation_alpha
            else:
                train_loss = args.cls_alpha * logit_loss
        
        # * Use uncertainty to combine annotation_loss & attention_loss
        elif args.uncertainty:
        
            # In the earlier epochs, we only learned with explanation.
            if epoch<args.start_using_annotation: # In the earlier epochs, we only learned with explanation.
                train_loss = args.cls_alpha * logit_loss
                
            # At a predifined epoch, choose top-k data according to current_uncertainties for human annotation.
            elif epoch==args.start_using_annotation:
                if slide_id2 in topk_ids:
                    # print("use annotation_loss",slide_id2)
                    annotation_loss = model.forward_annotation_loss(attn, labels_mask, args.anno_loss_type)
                    train_loss = args.cls_alpha * logit_loss + annotation_loss * args.annotation_alpha
                    print(f"Logits: {logit_loss}, Annotation: {annotation_loss}, Both: {train_loss}")

                else:
                    train_loss = args.cls_alpha * logit_loss
            
            # After certain epochs, we use fixed_topk_ids
            else: 
                if slide_id2 in fixed_topk_ids:
                    # print("use annotation_loss",slide_id2)
                    annotation_loss = model.forward_annotation_loss(attn, labels_mask, args.anno_loss_type)

                    train_loss = args.cls_alpha * logit_loss + annotation_loss * args.annotation_alpha
                    print(f"Logits: {logit_loss}, Annotation: {annotation_loss}, Both: {train_loss}")
                else:
                    train_loss = args.cls_alpha * logit_loss
               

        train_loss = train_loss / args.accumulation_steps
        
        # clip gradient
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        #if (i+1) % args.accumulation_steps == 0:
        train_loss.backward() # 对本 batch 的损失 train_loss 进行 反向传播，计算梯度
        optimizer.step() # 用计算出的梯度对模型参数进行更新（执行一次梯度下降）
        
        with torch.no_grad():
            model.alphas.clamp_(-5.0, 5.0)
        
        # 如果开启了 --lr_supi（表示 每个 iteration 都要 step 一次学习率），就在这里更新学习率
        if args.lr_supi and scheduler is not None:
            scheduler.step()
            
        if args.model == 'mhim':
            if mm_sche is not None: # 如果传入了 mm_sche（即 EMA momentum 调度器），就根据当前 step（epoch*len(loader)+i）取得当前步的 momentum 值 mm
                mm = mm_sche[epoch*len(loader)+i]
            else:
                mm = args.mm
            if model_tea is not None:
                # *** Teacher Model: Update with EMA ***
                if args.uncertainty: ema_update_edl(model,model_tea,mm)
                else: ema_update(model,model_tea,mm)
                # *** Teacher Model: Training the evidence head ***
                alpha, cls_tea, attn = model_tea.forward_teacher_edl(bag)
                num_classes = 2
                y = one_hot_embedding(label, num_classes)
                edl_loss = edl_my_loss(alpha, y.float(), epoch, num_classes, 10, device)
                
                edl_loss.backward()
                optimizer_teacher.step()
        else:
            mm = 0.

        # 对训练过程中的每个指标做平均
        loss_cls_meter.update(logit_loss,1)
        loss_cl_meter.update(cls_loss,1)
        patch_num_meter.update(patch_num,1)
        keep_num_meter.update(keep_num,1)
        mm_meter.update(mm,1)

        # 日志输出（每隔若干步打印）
        if i % args.log_iter == 0 or i == len(loader)-1:
            # 获取当前学习率（如果 optimizer 里有多个 param group，就取平均）
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('cls_loss',loss_cls_meter.avg),
                ('lr',lr),
                ('cl_loss',loss_cl_meter.avg),
                ('patch_num',patch_num_meter.avg),
                ('keep_num',keep_num_meter.avg),
                ('mm',mm_meter.avg),
            ])
            print('[{}/{}] logit_loss:{}, cls_loss:{},  patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,patch_num_meter.avg, keep_num_meter.avg))

        train_loss_log = train_loss_log + train_loss.item() 


    end = time.time()
    train_loss_log = train_loss_log/len(loader) # 整轮平均训练损失
    if not args.lr_supi and scheduler is not None: # 如果不是按 step 更新，而是按 epoch 更新，就在 epoch 末尾 step 一次
        scheduler.step()
    
    # !返回的all_uncertainties是逐时更新的，我们不用这个来选topk_ids，因为直接用fixed_topk_ids
    return train_loss_log,start,end, topk_ids, all_uncertainties, all_features, all_probs

def val_loop(args,model,loader,device,criterion,early_stopping,epoch,model_tea=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    loss_cls_meter = AverageMeter()
    bag_logit, bag_labels=[], []

    with torch.no_grad(): # 禁用梯度计算，以节省内存和加快验证速度
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            bag=data[0].to(device)  # b*n*1024
            # print("bag", bag.shape)
            if bag.ndim == 2: # 如果batch_size=1的话
                bag = bag.unsqueeze(0)
            batch_size=bag.size(0)

            label=data[1].to(device)
            # 对不同类型模型进行推理
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
                if args.baseline == 'dsmil':
                    test_logits = test_logits[0]
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            else:
                test_logits = model(bag)

            # 计算 loss + 获取预测概率
            if args.loss == 'ce': # 交叉熵损失
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))) or (args.model == 'pure' and args.baseline == 'dsmil'):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label.view(batch_size,-1).float())
                    
                    bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            loss_cls_meter.update(test_loss,1) # 更新损失统计器
    
    # save the log file # 计算最终评估指标
    # print("1111111",bag_labels, bag_logit)
    accuracy, auc_value, precision, recall, fscore, threshold_optimal = five_scores(bag_labels, bag_logit)
    
    # early stop 
    if early_stopping is not None:
        # 根据当前的 AUC 值判断是否要提前停止（AUC 越大越好，因此要用 -auc 做最小化监控）
        early_stopping(epoch,-auc_value,model)
        stop = early_stopping.early_stop
    else:
        stop = False
    return stop,accuracy, auc_value, precision, recall, fscore,loss_cls_meter.avg, threshold_optimal


def test(args,model,loader,device,criterion,model_tea=None,opt_thr=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())
                
            bag=data[0].to(device)  # b*n*1024
            # print("bag", bag.shape)
            if bag.ndim == 2: # 如果batch_size=1的话
                bag = bag.unsqueeze(0)
            batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
                if args.baseline == 'dsmil':
                    test_logits = test_logits[0]
            elif args.model == 'dsmil':
                test_logits,_ = model(bag) # dsmil 返回两个输出（instance-level 和 bag-level），这里只用 bag-level
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple)))or (args.model == 'pure' and args.baseline == 'dsmil'):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    # 使用 DSMIL 或类似双预测头的模型时，对两个 logits 求 softmax，然后平均预测
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average: # DSMIL 且使用双头平均
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label.view(1,-1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    # save the log file
    # cal the best thr with val set
    opt_thr = opt_thr if args.best_thr_val else None # 是否使用验证集算出的最佳阈值
    
    # 计算各类指标和loss
    accuracy, auc_value, precision, recall, fscore, _ = five_scores(bag_labels, bag_logit,threshold_optimal=opt_thr)
    test_loss_log = test_loss_log/len(loader)

    return accuracy, auc_value, precision, recall, fscore,test_loss_log