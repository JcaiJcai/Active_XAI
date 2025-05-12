import numpy as np
import random
import torch
import torch.nn as nn

class Explainer_shap(nn.Module):
    def __init__(self, device, MIL_model=None, subset_num=3, shuffle=True, shuffle_time=2):
        super().__init__()
        self.device = device
        self.MIL_model = MIL_model
        self.subset_num = subset_num
        self.shuffle = shuffle
        self.shuffle_time = shuffle_time

    def set_explained_class(self, batch):
        return batch['label']

    def explain(self, batch, model, search_num):
        features = batch['features'].unsqueeze(0).to(self.device)  # [1, N, D]
        label = batch['label']  # scalar
        cls_tea, attn = model.forward_teacher(features)
        if self.MIL_model == "dsmil" or self.MIL_model == "abmil":
            attn_avg_lastlayer = attn.squeeze(0)
        elif self.MIL_model == "transmil":
            attn_avg_lastlayer = attn[-1].mean(dim=1).view(-1)
        
        # We ask the shape of attn_avg_lastlayer to be like: torch.Size([7974])
        # print(attn_avg_lastlayer.shape)
        attn_index = np.argsort(-attn_avg_lastlayer.detach().cpu().numpy())
        search_num = min(search_num, len(attn_index))
        search_indices = attn_index[:search_num] #  search_indices 是按注意力值从高到低排序后的前 search_num 个索引，已经排序好了！！
        # search_indices = attn_index
        shapley_score = self.shapley_value(search_indices, features, label, model)
        # print("shapley_score",shapley_score)
        
        # 将Shapley值映射到 attention值 的前search_num个值区间
        attn_top_values = attn_avg_lastlayer[search_indices].to("cpu")
        attn_min, attn_max = attn_top_values.min(), attn_top_values.max()
        shap_norm = (shapley_score - shapley_score.min()) / (shapley_score.max() - shapley_score.min() + 1e-8)
        shap_scaled = shap_norm * (attn_max - attn_min) + attn_min
        # print("shap_scaled",shap_scaled)
        
        final_scores = np.array(attn_avg_lastlayer.to("cpu"))
        final_scores[search_indices] = shap_scaled # 替换对应位置为 shap 值
        return final_scores

    def approximate_shapley_subset(self, search_indices, data, label, model, subset_num):
        left_indices = [i for i in range(data.shape[1]) if i not in search_indices]
        random.shuffle(left_indices)
        left_data = data[:, left_indices, :]
        left_logits = []
        subset_data = [left_data[:,i::subset_num,:] for i in range(subset_num)] # 从第 i 个元素开始，每隔 subset_num 个取一个
        # print("subset_data",subset_data)
        for _subset_data in subset_data:
            # print(_subset_data)
            left_logit, _, _, _ = model(_subset_data.to(self.device))
            if self.MIL_model == "dsmil":
                left_logit = left_logit[0] # 因为dsmil只用第一个logit做预测！
            # We ask the form of left_logit to be like: tensor([[ 5.8495, -6.6218]], device='cuda:0')
            # print("left_logit",left_logit)
            left_logits.append(left_logit.cpu())
        
        cont = torch.zeros((data.shape[1], left_logit.shape[-1])) # cont[i, c] 表示 patch i 在类别 c 上的累计 logit 增益
        # m = 0
        for i in search_indices:
            for j, _subset_data in enumerate(subset_data): # 遍历将 left_data 分成的多个 subset（加速计算）
                x = torch.cat((data[:, i, :].unsqueeze(0), _subset_data), axis=1) # 把 patch i 插入到当前的 subset_data 形成新的输入 x，用于模拟“添加 i 后”的预测
                for _ in range(self.shuffle_time):
                    if self.shuffle:
                        idx = torch.randperm(x.shape[1]) # 就打乱 patch 的顺序（模拟它在不同位置时的影响）
                        x = x[:, idx, :]
                    logit, _, _, _ = model(x.to(self.device))
                    if self.MIL_model == "dsmil":
                        logit = logit[0]
                    cont[i] = cont[i] + logit.cpu() - left_logits[j] # 当前 logit 与 baseline（没加 patch i 时）对比，表示 patch i 的边际增益
            # print(m)
            # m=m+1
        cont = cont / self.shuffle_time # 取平均得到每个patch在类别上的稳定增益值
        score = cont[search_indices, int(label)] # 提取当前 bag 的 真实类别 label 对应的 logit 增益，作为 Shapley score 的近似
        return score
    
    def shapley_value(self, search_indices, data, label, model):
        model.eval()
        with torch.no_grad():
            if data.shape[1]>3000: # 对于包含超过6000个patch的WSI图片
                score = self.approximate_shapley_subset(search_indices, data, label, model, subset_num = 3)
            elif data.shape[1]<=3000: # 对半分search_indices，再传入approximate_shapley_subset，再把score拼起来
                mid = len(search_indices) // 2
                indices_1 = search_indices[:mid]
                indices_2 = search_indices[mid:]

                score_1 = self.approximate_shapley_subset(indices_1, data, label, model, subset_num=3)
                score_2 = self.approximate_shapley_subset(indices_2, data, label, model, subset_num=3)

                score = torch.zeros(len(search_indices), device=score_1.device)
                score[:mid] = score_1
                score[mid:] = score_2
          
            # 将 score 归一化到 [0, 1]    
            score = (score - torch.min(score)) / (torch.max(score) - torch.min(score)) 
        return score # (len(search_indices), )