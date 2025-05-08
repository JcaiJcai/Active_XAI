
import torch
import torch.nn as nn

class Explainer_perturbation(nn.Module):
    # explained_rel: the output relevance. can be:
    #     'logit-diff': the difference of the logits (1st eq. of p. 202 of Montavon et. al., 2019)
    #     'logits': the logits without any change
    def __init__(self,device):
        super().__init__()
        self.explained_rel='logit' # 'logit-diff'
        self.device=device

    def set_explained_class(self, batch): # 确定要解释的是哪个类别
        return batch['label']

    def explain(self, batch, model, perturbation_method):
        # 逐个 patch：要么保留它（keep），要么去掉它（drop）
        # 看bag-level预测分数变化
        # 变化越大，说明这个patch越重要
        def forward_fn(features, _bag_sizes):
            features = features.to(self.device)
            return model(features)

        model.eval()
        explained_class = self.set_explained_class(batch)
        return self.perturbation_scores(batch, perturbation_method, forward_fn, explained_class, self.explained_rel)
    
    @staticmethod
    def perturbation_scores(batch, perturbation_method, forward_fn, explained_class, explained_rel='softmax'):
        # batch: 一个 dict，包含当前样本的 patch 特征、bag 大小等。
        # perturbation_method: 'keep' 或 'drop'，即：
        # 'keep': 只保留当前 patch
        # 'drop': 删除当前 patch
        # forward_fn: 前向推理函数，用来计算预测值
        # explained_class: 想解释的类别索引（如肿瘤 = 1）
        # explained_rel: 'softmax' 表示对 logit 做 softmax 后再解释
        num_patches = batch['bag_size']
        num_batches = 1
        scores = []
        for patch_idx in range(num_patches):
            # print(patch_idx)
            # 构造新的 bag，进行“保留或删除 patch”
            if perturbation_method == 'keep':
                keep_idx = [patch_idx]
                bag_sizes = torch.tensor([1])
            elif perturbation_method == 'drop':
                keep_idx = list(range(patch_idx)) + list(range(patch_idx + 1, num_patches)) # 删除当前patch
                bag_sizes = torch.tensor([num_patches - 1])
            else:
                raise ValueError(f"Unknown perturbation method: {perturbation_method}")
            features = batch['features'].unsqueeze(0)
            features = features[..., keep_idx, :] # # 选取需要保留的 patch 特征
            preds = forward_fn(features, bag_sizes)[0].detach().cpu()
            if explained_rel == 'softmax':
                preds = torch.softmax(preds, dim=-1)
            scores.append(preds[:, explained_class]) # 提取解释目标类的概率
        scores = torch.cat(scores, dim=0)
        if perturbation_method == 'drop':
            features, bag_sizes = batch['features'], batch['bag_size']
            features = batch['features'].unsqueeze(0)
            preds = forward_fn(features, bag_sizes)[0].detach().cpu()# 计算完整 bag 的预测（没删 patch）
            if explained_rel == 'softmax':
                preds = torch.softmax(preds, dim=-1)
            scores = preds[0, explained_class] - scores # 全图预测值 - 删除 patch 后的预测值
        return scores.numpy() # 代表该 patch 的“正向贡献程度”