import time
import torch
import wandb
import numpy as np
from copy import deepcopy
import torch.nn as nn
# from dataloader import *
from torch.utils.data import DataLoader, RandomSampler
import argparse, os
from torch.nn.functional import one_hot
# from torch.cuda.amp import GradScaler
from contextlib import suppress
import time
import random

from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict

from utils_clam.utils import get_split_loader
from utils_clam import *
from utils_mhim import *

from modules import attmil,clam,mhim,dsmil,transmil,mean_max
from explanation import shapley

def seed_torch(seed=2021):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

def one_fold(args,k,ckc_metric,dataset):
    seed_torch(args.seed)
    # loss_scaler = GradScaler() if args.amp else None # AMP（自动混合精度训练）
    loss_scaler = None
    # amp_autocast = torch.cuda.amp.autocast if args.amp else suppress # 混合精度下的自动 autocast
    amp_autocast = suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # TODO
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric
    
    # *** Load Data ***
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, use_h5=args.use_h5, h5_folder_name=args.h5_folder_name, csv_path='{}/splits_{}.csv'.format(args.split_dir, k))
    print('\nInit Loaders...', end=' ')
    
    print("train_dataset",train_dataset)
    print("val_dataset",val_dataset)
    print("test_dataset",test_dataset)
    # !注意这里我们可能没有固定seed
    train_loader = get_split_loader(train_dataset, args.batch_size, training=True, testing = args.testing, weighted = args.weighted_sample) # batch_size都是1
    val_loader = get_split_loader(val_dataset, args.batch_size, testing = args.testing)
    test_loader = get_split_loader(test_dataset, args.batch_size, testing = args.testing)
    print('Done!')
    print(len(val_loader))
    # try:
    #     first_batch = next(iter(val_loader))
    #     print("val_loader 有数据")
    # except StopIteration:
    #     print("[WARNING] val_loader 是空的（StopIteration）")
    
    mm_sche = None
    # 加载之前训练好的某折的模型作为初始的teacher模型
    _teacher_init =args.teacher_init
    
    # ** Load model **
    if args.model == 'mhim':
        if args.mrh_sche:
            # 调度器控制每一步的 mask_ratio_h，随 epoch / iteration 逐步下降（余弦衰减）
            mrh_sche = cosine_scheduler(args.mask_ratio_h, # 初始值
                                        0., # 最终值
                                        epochs=args.num_epoch,
                                        niter_per_ep=len(train_loader)
                                        )
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'mask_ratio' : args.mask_ratio, # 全局掩码比例
            'n_classes': args.n_classes,
            'temp_t': args.temp_t, # 温度参数
            'act': args.act, # 激活函数类型
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion, # 多头注意力的融合方式
            # 高层、恢复、高层再掩码和低层掩码比例
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche, # 掩码调度器
            'da_act': args.da_act, # 数据增强相关的激活函数
            'attn_layer': args.attn_layer, # 注意力层的配置
            'use_human_annotation': args.use_human_annotation,
        }
        
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.num_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)# start_warmup_value前期可能先预热

        model = mhim.MHIM(**model_params).to(device)
    elif args.model == 'pure': # 最最简单的模型
        model = mhim.MHIM(select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)    
    elif args.model =='clam_sb':
        # model = CLAM_SB(**model_dict,  instance_loss_fn=instance_loss_fn)
        model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'clam_mb':
        # model = CLAM_MB(**model_dict,instance_loss_fn=instance_loss_fn)
        model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)        
    
    # 初始化学生模型（student model）参数 
    if args.init_stu_type != 'none':
        if not args.no_log:
            print('######### Model Initializing.....')
        pre_dict = torch.load(_teacher_init)
        if 'model' in pre_dict:
            pre_dict = pre_dict['model'] # 取出实际参数
            
        new_state_dict ={}
        if args.init_stu_type == 'fc': # 只初始化 patch_to_emb 模块
        # only patch_to_emb
            for _k,v in pre_dict.items():
                _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                new_state_dict[_k]=v
            info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
        else: 
        # init all
            info = model.load_state_dict(pre_dict,strict=False) # 初始化整个模型
        if not args.no_log:
            print(info)
            
    # *** Init teacher ***
    if args.model == 'mhim':
        model_tea = deepcopy(model) # 创建一个 teacher 模型：直接复制当前 student 模型作为初始化版本（深拷贝，二者权重独立）
        if not args.no_tea_init and args.tea_type != 'same': # !!
            print('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(_teacher_init)
                if 'model' in pre_dict:
                    pre_dict = pre_dict['model']
                info = model_tea.load_state_dict(pre_dict,strict=False) # 尝试从 checkpoint 加载权重给 model_tea，允许部分加载
                if not args.no_log:
                    print(info)
            except:
                if not args.no_log:
                    print('########## Init Error')
        # if args.tea_type == 'same':
        #     model_tea = model # teacher 就是 student，不需要分开两个模型（共享权重）。
    else:
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
        
    # *** Learning Rate Scheduler，用于动态调整训练过程中优化器的学习率
    if args.lr_sche == 'cosine': # 	余弦退火
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step': # 阶梯衰减	
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const': # 固定学习率
        scheduler = None

    # *** Early stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    epoch_start = 0
    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_thr,opt_epoch = 0, 0, 0, 0,0,0,0
    opt_te_auc,opt_tea_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs  = 0., 0., 0., 0., 0.
    
    train_time_meter = AverageMeter() # 计时器
    
    for epoch in range(epoch_start, args.num_epoch):
        # ****** TRAIN (Teacher and Student)******
        train_loss,start,end = train_loop(args,model,model_tea,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch)
        train_time_meter.update(end-start) # 训练时间
        # stop: early stopping 是否触发了；threshold_optimal: 最优分类阈值（如果是二分类）
        
        # ****** EVALUATE (Student) ******
        stop,accuracy, auc_value, precision, recall, fscore, test_loss, threshold_optimal = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch,model_tea)
        
        # ****** EVALUATE (Teacher) ******
        if model_tea is not None: # 如果有 teacher 模型
            # 对 teacher 模型做一次验证
            _,accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea,_ = val_loop(args,model_tea,val_loader,device,criterion,None,epoch,model_tea)
            
            if auc_value_tea > opt_tea_auc: # 如果当前 teacher 的验证 AUC 比之前最优的还高，就更新
                opt_tea_auc = auc_value_tea

        # ********* TEST *********
        if args.always_test:
            # 测试 student 模型
            _te_accuracy, _te_auc_value, _te_precision, _te_recall, _te_fscore,_te_test_loss_log = test(args,model,test_loader,device,criterion,model_tea)

            if _te_auc_value > opt_te_auc:
                opt_te_auc = _te_auc_value
                opt_te_fs = _te_fscore
            
            if model_tea is not None: # 测试教师模型
                _te_tea_accuracy, _te_tea_auc_value, _te_tea_precision, _te_tea_recall, _te_tea_fscore,_te_tea_test_loss_log = test(args,model_tea,test_loader,device,criterion,model_tea)
            
                if _te_tea_auc_value > opt_te_tea_auc:
                    opt_te_tea_auc = _te_tea_auc_value
                    opt_te_tea_fs = _te_tea_fscore

        print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))
        
        # 在验证集 AUC 超过历史最优值，并且训练进度达到一定阶段后，保存当前模型为“最佳模型”。
        # args.save_best_model_stage 是一个 [0, 1] 之间的浮点数，比如 0.5，表示“从总轮数的一半开始才保存最佳模型”。
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
        torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k))) # 保存这组参数到本地文件系统
        
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
            'k': k, # 当前是第几折 fold
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
    best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k))) # 加载当前第 k 折交叉验证的最优模型权重文件
    info = model.load_state_dict(best_std['model']) # 将保存的 student 模型参数加载到当前模型 model 中
    print(info)
    if model_tea is not None and best_std['teacher'] is not None:
        info = model_tea.load_state_dict(best_std['teacher'])
        print(info)
    
    accuracy, auc_value, precision, recall, fscore,test_loss_log = test(args,model,test_loader,device,criterion,model_tea,opt_thr)
    
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

def train_loop(args,model,model_tea,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch):
    start = time.time()
    # 记录训练过程中平均指标值
    loss_cls_meter = AverageMeter() # 分类损失（logit loss）
    loss_cl_meter = AverageMeter() # 对比损失或蒸馏损失（cls_loss）
    patch_num_meter = AverageMeter() # 输入的 patch 数
    keep_num_meter = AverageMeter() # “保留”的 patch 数（例如某些掩码操作后剩下的）
    mm_meter = AverageMeter() # EMA momentum 值
    train_loss_log = 0. # 最终返回的平均损失
    
    # 将主模型和 teacher 模型都设置为训练模式
    model.train()
    if model_tea is not None:
        model_tea.train()

    for i, data in enumerate(loader): # i是batch index，batch_size是1时，data是整个train数据集
        # data[0]: features(36710, 1024)
        # data[1]: label(1)
        # data[2]: coords(36710, 2)
        # data[3]: labels_mask(36710,)
        # print("data",data[0].shape, data) # torch.Size([55340, 1024])
        optimizer.zero_grad() # 清除上一个 batch 的梯度

        # bag: 是一个 batch 的 feature（MIL 模型中，一张 WSI = 一个 bag，bag 内有多个 patch）
        # batch_size: 当前 batch 内有多少张图（每张图是一个 bag）
        bag=data[0].to(device)  # b*n*1024
        # print("bag", bag.shape)
        if bag.ndim == 2: # 如果batch_size=1的话
            bag = bag.unsqueeze(0)
        batch_size=bag.size(0)
        
        label=data[1].to(device)
        # coords=data[2].to(device) #暂时用不到
        labels_mask=data[3].to(device)
        

        with amp_autocast(): # 开启 AMP
            # 打乱patch
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)
            
            # 初始化分类损失
            logit_loss = None
            
            # model_tea和model都是MHIM
            if args.model == 'mhim':
                if model_tea is not None:
                    # ***** 教师模型 *****
                    # ! attn -> explanation
                    if args.explanation == "attention":
                        cls_tea, attn = model_tea.forward_teacher(bag) # 用教师模型生成 attention 和预测概率
                        # attn_sum = attn[1].sum(dim=-1)
                        # print("!!!!!!!attn_sum",attn_sum) # attention并不是归一化之后的
                    elif args.explanation == "shap-approximate": 
                        cls_tea, attn = model_tea.forward_teacher(bag)
                        # 用教师模型生成shap解释
                        # print("attn",len(attn), attn) # len是2说明这个attention对应的是两层的
                        attn_avg_lastlayer = attn[-1].mean(dim=1).view(-1)
                        # print("attn_avg_lastlayer",attn_avg_lastlayer.shape)
                        attn_index = np.argsort(-attn_avg_lastlayer.detach().cpu().numpy())
                        # print("attn_index",attn_index)
                        # search_num = min(100, len(attn_index) // 2)
                        # search_indices = attn_index[:search_num]
                        # print("search_indices",search_indices)
                        score = shapley.shapley_value(attn_index, bag, label, model_tea, device, args.baseline, subset_num=10).to(attn[0].device) # (len(search_indices), )
                        attn = [score.unsqueeze(0).unsqueeze(0).expand(1, 8, -1) for _ in range(2)] # （1，8, 16124）
                else:
                    attn,cls_tea = None,None
                    
                
                cls_tea = None if args.cl_alpha == 0. else cls_tea

                # ***** 学生模型 *****
                if args.use_attn_loss == True:
                    if args.baseline == 'dsmil':
                        # logits 是 DSMIL 的主类预测 + instance-level 预测。用两个都计算 loss
                        logits, cls_loss, attn_loss, patch_num, keep_num = model.forward_with_distill_loss(bag,attn,labels_mask,cls_tea[0],i=epoch*len(loader)+i) # !!!!!
                        logit_loss = 0.5*criterion(logits[0].view(batch_size,-1),label) + 0.5*criterion(logits[1].view(batch_size,-1),label)
                    else:
                        logits, cls_loss, attn_loss, patch_num, keep_num = model.forward_with_distill_loss(bag,attn,labels_mask,cls_tea,i=epoch*len(loader)+i)
                    
                else:
                    if args.baseline == 'dsmil':
                        # logits 是 DSMIL 的主类预测 + instance-level 预测。用两个都计算 loss
                        logits, cls_loss,patch_num,keep_num = model(bag,attn,labels_mask,cls_tea[0],i=epoch*len(loader)+i) # !!!!!
                        logit_loss = 0.5*criterion(logits[0].view(batch_size,-1),label) + 0.5*criterion(logits[1].view(batch_size,-1),label)
                    else:
                        logits, cls_loss,patch_num,keep_num = model(bag,attn,labels_mask,cls_tea,i=epoch*len(loader)+i)

            elif args.model == 'pure':
                if args.baseline == 'dsmil':
                    logits, cls_loss,patch_num,keep_num = model.pure(bag)
                    logit_loss = 0.5*criterion(logits[0].view(batch_size,-1),label) + 0.5*criterion(logits[1].view(batch_size,-1),label)
                else:
                    logits, cls_loss,patch_num,keep_num = model.pure(bag)
            elif args.model in ('clam_sb','clam_mb','dsmil'):
                logits,cls_loss,patch_num = model(bag,label,criterion)
                keep_num = patch_num
            else:
                logits = model(bag)
                cls_loss,patch_num,keep_num = 0.,0.,0.
            
            # 分类损失计算
            if logit_loss is None:
                if args.loss == 'ce':
                    logit_loss = criterion(logits.view(batch_size,-1),label)
                elif args.loss == 'bce':
                    logit_loss = criterion(logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))

        # 总Loss
        if args.use_attn_loss == True:
            # print("logit_loss",logit_loss)
            # print("cls_loss",cls_loss)
            # print("attn_loss",attn_loss)
            # train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha + attn_loss*attn_alpha
            train_loss = args.cls_alpha * logit_loss + attn_loss*args.attn_alpha # !试一下只用attn_loss的实验
        else:
            train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha
        train_loss = train_loss / args.accumulation_steps
        
        # 梯度裁剪
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        #if (i+1) % args.accumulation_steps == 0:
        train_loss.backward() # 对本 batch 的损失 train_loss 进行 反向传播，计算梯度
        optimizer.step() # 用计算出的梯度对模型参数进行更新（执行一次梯度下降）
        
        # 如果开启了 --lr_supi（表示 每个 iteration 都要 step 一次学习率），就在这里更新学习率
        if args.lr_supi and scheduler is not None:
            scheduler.step()
            
        if args.model == 'mhim':
            if mm_sche is not None: # 如果传入了 mm_sche（即 EMA momentum 调度器），就根据当前 step（epoch*len(loader)+i）取得当前步的 momentum 值 mm
                mm = mm_sche[epoch*len(loader)+i]
            else:
                mm = args.mm
            if model_tea is not None:
                if args.tea_type == 'same':
                    pass
                else: # 如果存在 teacher 模型，并且 tea_type 不是 'same'，就执行 ema_update()
                    ema_update(model,model_tea,mm)
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
    
    return train_loss_log,start,end

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