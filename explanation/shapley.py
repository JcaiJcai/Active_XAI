import numpy as np
import random
import torch

def shapley_value(
    search_indices,  # 要计算 Shapley 值的一组 patch 下标
    data,            # 当前 bag 的所有 patch，形状为 [1, patch数, 维度]
    label,           # 当前 bag 的 ground truth label
    model,           # MIL 分类模型
    device,          # 使用的设备（如 cuda）
    MIL_model='ABMIL',  # 模型类型（支持 ABMIL 或 CLAM）
    shuffle=False,    # 是否在插入目标 patch 时打乱顺序 # 默认为True
    shuffle_time=2,  # 每个 subset 重复 shuffle 次数（用于做平均）
    subset_num=3     # 将剩余 patch 分成多少个子集来加速计算
):
    
    model.eval()
    with torch.no_grad():
        left_indices = [i for i in range(data.shape[1]) if i not in search_indices]
        random.shuffle(left_indices)
        left_data = data[:, left_indices, :]
        left_logits = []
        subset_data = [left_data[:,i::subset_num,:] for i in range(subset_num)] # 从第 i 个元素开始，每隔 subset_num 个取一个
        for _subset_data in subset_data:
            left_logit, _, _, _ = model(_subset_data.to(device))
            left_logits.append(left_logit.cpu())
        
        cont = torch.zeros((data.shape[1], left_logit.shape[-1])) # cont[i, c] 表示 patch i 在类别 c 上的累计 logit 增益
        for i in search_indices:
            for j, _subset_data in enumerate(subset_data): # 遍历将 left_data 分成的多个 subset（加速计算）
                x = torch.cat((data[:, i, :].unsqueeze(0), _subset_data), axis=1) # 把 patch i 插入到当前的 subset_data 形成新的输入 x，用于模拟“添加 i 后”的预测
                for _ in range(shuffle_time):
                    if shuffle:
                        idx = torch.randperm(x.shape[1]) # 就打乱 patch 的顺序（模拟它在不同位置时的影响）
                        x = x[:, idx, :]
                    logit, _, _, _ = model(x.to(device))
                    cont[i] = cont[i] + logit.cpu() - left_logits[j] # 当前 logit 与 baseline（没加 patch i 时）对比，表示 patch i 的边际增益
        cont = cont / shuffle_time # 取平均得到每个patch在类别上的稳定增益值
        score = cont[search_indices, int(label)] # 提取当前 bag 的 真实类别 label 对应的 logit 增益，作为 Shapley score 的近似
        score = (score - torch.min(score)) / (torch.max(score) - torch.min(score)) # 将 score 归一化到 [0, 1]
    return score # (len(search_indices), )

