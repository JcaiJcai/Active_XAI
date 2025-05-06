import torch
import numpy as np
from torch import nn
from modules.datten import *
import torch.nn.functional as F
from modules.satten import *
from modules.clam_mhim import CLAMSB, CLAMMB
from .topk.svm import SmoothTop1SVM


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self,temp_t=1.,temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool= True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t,dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss
        
class MHIM(nn.Module):
    def __init__(self, mlp_dim=512,mask_ratio=0,n_classes=2,temp_t=1.,temp_s=1.,dropout=0.25,act='relu',select_mask=False, use_human_annotation=False, select_inv=False,msa_fusion='vote',mask_ratio_h=0.,mrh_sche=None,mask_ratio_hr=0.,mask_ratio_l=0.,da_act='gelu',baseline='selfattn',head=8,attn_layer=0):
        super(MHIM, self).__init__()
 
        self.mask_ratio = mask_ratio # 总体随机 mask 比例
        self.mask_ratio_h = mask_ratio_h # 高 attention 区域的 mask 比例
        self.mask_ratio_hr = mask_ratio_hr # 高 attention 区域中随机挑选的比例
        self.mask_ratio_l = mask_ratio_l # 低 attention 区域的 mask 比例
        self.select_mask = select_mask # 是否启用 mask 操作
        self.use_human_annotation = use_human_annotation
        self.select_inv = select_inv # 是否对 mask 区域反选（选择需要关注的 patch）
        self.msa_fusion = msa_fusion # 多头 attention 聚合方式
        self.mrh_sche = mrh_sche # 是否使用 mask_ratio_h 的调度器（如余弦变化）
        self.attn_layer = attn_layer # 如果 encoder 有多层 attention，指定取哪一层
        self.baseline = baseline # 主干结构类型（如 selfattn, dsmil, attn）

        self.patch_to_emb = [nn.Linear(1024, 512)] # 用线性层把每个 patch 的原始特征（1024维）映射到 512 维嵌入空间

        # 激活函数
        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]
        # dropout
        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # 封装成一个 Sequential 模块
        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        if baseline == 'selfattn': # Transmil
            self.online_encoder = SAttention(mlp_dim=mlp_dim,head=head)
        elif baseline == 'attn':  # ABMIL
            self.online_encoder = DAttention(mlp_dim,da_act)
        elif baseline == 'dsmil': # DSMIL
            self.online_encoder = DSMIL(mlp_dim=mlp_dim,mask_ratio=mask_ratio)
        elif baseline == 'clam_sb': # SAJ CLAM hardcoded device for now
            self.online_encoder = CLAMSB(l_dim=mlp_dim, d_dim=mlp_dim//2, dropout=False, n_classes=n_classes, device='cuda')
        elif baseline == 'clam_mb': # SAJ CLAM hardcoded device for now
            self.online_encoder = CLAMMB(l_dim=mlp_dim, d_dim=mlp_dim//2, dropout=False, n_classes=n_classes, device='cuda')
        
        self.predictor = nn.Linear(mlp_dim,n_classes)

        # 设置 teacher 和 student 的 softmax 温度（用于 CL）
        self.temp_t = temp_t
        self.temp_s = temp_s

        # 对比学习的 soft cross-entropy 损失函数
        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t,self.temp_s)

        # ???????
        self.predictor_cl = nn.Identity()
        self.target_predictor = nn.Identity()

        # 对模型的所有参数执行初始化操作
        self.apply(initialize_weights)

    # 根据 attention 分数、mask 比例、mask 类型（高置信 or 低置信）等，选择哪些 patch 应该被 mask，用于后续的 masked modeling 或对比学习。
    def select_mask_fn(self,ps,attn,largest,mask_ratio,mask_ids_other=None,len_keep_other=None,cls_attn_topk_idx_other=None,random_ratio=1.,select_inv=False):
        # ps：每个 sample 的 patch 数量。
        # attn：attention 分数。
        # largest：True 选分数高的；False 选分数低的。
        # mask_ratio：要 mask 掉的比例。
        # mask_ids_other：已有的 mask index（如果这是组合 mask）。
        # len_keep_other：已有 mask 里保留的数量（用于计算新 mask）。
        # cls_attn_topk_idx_other：已有的 top-k index（可复用）。
        # random_ratio：是否只从 top-k 里随机选一部分（<1）。
        # select_inv：是否反选。
        
        ps_tmp = ps
        mask_ratio_ori = mask_ratio
        mask_ratio = mask_ratio / random_ratio
        if mask_ratio > 1:
            random_ratio = mask_ratio_ori
            mask_ratio = 1.
            
        # print(attn.size())
        if mask_ids_other is not None: # 如果这是第二次遮盖（组合 mask），剩下能遮盖的 patch 数量要更新
            if cls_attn_topk_idx_other is None:
                cls_attn_topk_idx_other = mask_ids_other[:,len_keep_other:].squeeze()
                ps_tmp = ps - cls_attn_topk_idx_other.size(0)
                
        if len(attn.size()) > 2:
            if self.msa_fusion == 'mean': # 各层取平均，再 top-k
                _,cls_attn_topk_idx = torch.topk(attn,int(np.ceil((ps_tmp*mask_ratio)) // attn.size(1)),largest=largest)
                cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3,-1))
            elif self.msa_fusion == 'vote': # 每层 top-k，最后对出现频率 voting
                vote = attn.clone()
                vote[:] = 0
                
                _,idx = torch.topk(attn,k=int(np.ceil((ps_tmp*mask_ratio))),sorted=False,largest=largest)
                mask = vote.clone() 
                mask = mask.scatter_(2,idx,1) == 1
                vote[mask] = 1
                vote = vote.sum(dim=1)
                _,cls_attn_topk_idx = torch.topk(vote,k=int(np.ceil((ps_tmp*mask_ratio))),sorted=False)
                # print(cls_attn_topk_idx.size())
                cls_attn_topk_idx = cls_attn_topk_idx[0]
        else: # 单层 attention 直接选 top-k
            k = int(np.ceil((ps_tmp*mask_ratio)))
            _,cls_attn_topk_idx = torch.topk(attn,k,largest=largest)
            cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)
        
        # randomly 
        if random_ratio < 1.: # 从 top-k 中随机采样一部分
            random_idx = torch.randperm(cls_attn_topk_idx.size(0),device=cls_attn_topk_idx.device)

            cls_attn_topk_idx = torch.gather(cls_attn_topk_idx,dim=0,index=random_idx[:int(np.ceil((cls_attn_topk_idx.size(0)*random_ratio)))])
        

        # concat other masking idx 合并已有 mask
        if mask_ids_other is not None:
            cls_attn_topk_idx = torch.cat([cls_attn_topk_idx,cls_attn_topk_idx_other]).unique()

        # if cls_attn_topk_idx is not None: 
        len_keep = ps - cls_attn_topk_idx.size(0)
        a = set(cls_attn_topk_idx.tolist())
        b = set(list(range(ps)))
        mask_ids =  torch.tensor(list(b.difference(a)),device=attn.device) # 计算最终的 mask id
        
        # 是否反选
        if select_inv:
            mask_ids = torch.cat([cls_attn_topk_idx,mask_ids]).unsqueeze(0)
            len_keep = ps - len_keep
        else:
            mask_ids = torch.cat([mask_ids,cls_attn_topk_idx]).unsqueeze(0)

        return len_keep,mask_ids # 最终保留的 patch 数，mask patch index

    # ****** 似乎我们根据这个选mask就行 ******
    def get_mask(self,ps,i,attn,mrh=None):
        # ps：每个样本的 patch 数量（patch size）。
        # i：当前训练步数（epoch * iter）用于调节 mask 比例。
        # attn：attention 分数（来自 teacher 模型或学生自己）。
        # mrh：mask_ratio_h（高 attention masking 比例）。
        # 如果传进来的 attn 是多层 attention（如多个 transformer layer），需要从其中一层选择
        if attn is not None and isinstance(attn,(list,tuple)):
            if self.attn_layer == -1:
                attn = attn[1] # 默认用第 1 层 attention
            else:
                attn = attn[self.attn_layer]
        else:
            attn = attn

        # random mask 随机 mask
        if attn is not None and self.mask_ratio > 0.:
            len_keep,mask_ids = self.select_mask_fn(ps,attn,False,self.mask_ratio,select_inv=self.select_inv,random_ratio=0.001)
        else:
            len_keep,mask_ids = ps,None

        # low attention mask 遮盖低 attention patch（低置信区域）
        if attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep,mask_ids = self.select_mask_fn(ps,attn,False,self.mask_ratio_l,select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:,:len_keep].squeeze() if self.select_inv else mask_ids[:,len_keep:].squeeze()
                len_keep,mask_ids = self.select_mask_fn(ps,attn,False,self.mask_ratio_l,select_inv=self.select_inv,mask_ids_other=mask_ids,len_keep_other=ps,cls_attn_topk_idx_other = cls_attn_topk_idx_other)
        
        # high attention mask 遮盖高 attention patch
        mask_ratio_h = self.mask_ratio_h
        if self.mrh_sche is not None:
            mask_ratio_h = self.mrh_sche[i]
        if mrh is not None:
            mask_ratio_h = mrh
        if mask_ratio_h > 0. :
            # mask high conf patch
            if mask_ids is None:
                len_keep,mask_ids = self.select_mask_fn(ps,attn,largest=True,mask_ratio=mask_ratio_h,len_keep_other=ps,random_ratio=self.mask_ratio_hr,select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:,:len_keep].squeeze() if self.select_inv else mask_ids[:,len_keep:].squeeze()
                
                len_keep,mask_ids = self.select_mask_fn(ps,attn,largest=True,mask_ratio=mask_ratio_h,mask_ids_other=mask_ids,len_keep_other=ps,cls_attn_topk_idx_other = cls_attn_topk_idx_other,random_ratio=self.mask_ratio_hr,select_inv=self.select_inv)

        return len_keep,mask_ids
    
    def get_mask_human_annotation(self,ps,i,labels_mask=None,attn=None,mrh=None):
        # mask_ids就是所有保留的id，len_keep就是len(mask_ids)
        # labels_mask的形状是什么？(36710,)对应每个patch(36710, 1024)[i]有一个label
        # print(labels_mask.shape) # torch.Size([36710])
        
        # # 获取标签为2的patch索引
        # mask_ids = torch.nonzero(labels_mask == 2, as_tuple=False).view(-1)

        # # 如果没有标签为2的，则取标签为1的
        # if mask_ids.numel() == 0:
        #     mask_ids = torch.nonzero(labels_mask == 1, as_tuple=False).view(-1)
            
        # 另一种：不管是tumor还是normal，只用标签不为0的
        mask_ids = torch.nonzero(labels_mask != 0, as_tuple=False).view(-1)

        len_keep = mask_ids.shape[0]
        
        return len_keep, mask_ids.unsqueeze(0) # 修改形状以对应一个batch

    @torch.no_grad() 
    def forward_teacher(self,x): # 用于 teacher 模型前向推理

        x = self.patch_to_emb(x) # 将输入的 patch-level 特征 x 映射到嵌入空间（embedding），通常是一个线性层
        x = self.dp(x) # 应用 dropout 以增强模型鲁棒性

        if self.baseline == 'dsmil': # 如果是 DSMIL 模型，encoder 输出三个值，取其中两个
            _,x,attn = self.online_encoder(x,return_attn=True)
        elif self.baseline in ["clam_sb", "clam_mb"]:
            A, h = self.online_encoder(x.squeeze())    
            A = torch.transpose(A, 1, 0)  # KxN
            attn = F.softmax(A, dim=1)  # softmax over N
            x = torch.mm(attn, h) 
        else: # 返回编码后的 bag-level 表示 x 和 attention 分数 attn
            x,attn = self.online_encoder(x,return_attn=True)

        return x, attn
    
    @torch.no_grad()
    def forward_test(self, x, return_attn=False, no_norm=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        if return_attn:
            x, a = self.online_encoder(x, return_attn=True, no_norm=no_norm)
        elif self.baseline in ["clam_sb", "clam_mb"]:
            x, a = self.online_encoder(x, instance_eval=False)
        else:
            x = self.online_encoder(x)

        if self.baseline == 'dsmil':
            pass
        elif self.baseline in ["clam_sb", "clam_mb"]:
            pass
        else:   
            x = self.predictor(x)

        if return_attn:
            return x, a
        else:
            return x

    def pure(self, x, label=None):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)
        total_inst_loss = 0 
        
        if self.baseline == 'dsmil':
            x,_ = self.online_encoder(x)
        elif self.baseline in ["clam_sb", "clam_mb"]:
            x, total_inst_loss, _ = self.online_encoder(x, label)
        else:
            x = self.online_encoder(x)
            x = self.predictor(x)

        if self.training:
            return x, 0, ps, ps, total_inst_loss
        else:
            return x

    # 是原来的forward_loss
    def forward_cls_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat,teacher_cls_feat.detach())
        else:
            cls_loss = 0.
        
        return cls_loss
    
    def forward_attn_loss(self, student_attn, teacher_attn, mask_ids):
        # print("student_attn.shape",student_attn[0].shape, student_attn) # student_attn有两层，每层的的 shape 是 [1, 8, 16124]。
        N, L, D = student_attn[0].shape # [batch, head, tokens] e.g.[1, 8, 16124]
        keep_mask = torch.ones(N, D).to(student_attn[0].device)
        keep_mask.scatter_(dim=1, index=mask_ids, value=0)
        keep_mask = keep_mask.unsqueeze(1)  # 加一个维度[N, 1, D]
        keep_mask = keep_mask.expand(-1, L, -1)  # 复制 L 次变成[N, L, D]
        
        distill_loss = 0.

        for i in range(len(student_attn)):
            if teacher_attn is not None:
                # print(teacher_attn[i].shape, student_attn[i].shape, keep_mask.shape)
                teacher_masked = teacher_attn[i] * keep_mask
                student_masked = student_attn[i] * keep_mask
                distill_loss += - (teacher_masked.softmax(dim=-1) * torch.log_softmax(student_masked, dim=-1)).sum(dim=-1).mean()
                # distill_loss += - (teacher_attn[i].softmax(dim=-1) * torch.log_softmax(student_attn[i], dim=-1)).sum(dim=-1).mean()
        return distill_loss

    # ** Just for Student Model **
    def forward(self, x, attn=None,labels_mask=None,teacher_cls_feat=None,i=None):
        # x: 输入的 patch-level 特征（维度: [batch, n_patches, feature_dim]）。
        # attn: 上一步 teacher 模型生成的 attention，用于指导 masking。
        # teacher_cls_feat: teacher 输出的分类特征，用于计算蒸馏 loss。
        # i: 当前 iteration（用于 scheduler 控制 masking 比例变化）。
        # print("MHIM x.shape",x.shape) # torch.Size([1, 7974, 1024])
        x = self.patch_to_emb(x)
        x = self.dp(x)

        ps = x.size(1) # patch size

        # *** Get Hard Mask (mask_ids)  ***
        # ** use human annotation **
        print("use_human_annotation1",self.use_human_annotation)
        if self.use_human_annotation:
            # print("use human annotation")
            len_keep, mask_ids = self.get_mask_human_annotation(ps, i, labels_mask)#! 对于我们来说，mask_ids就是所有保留的id，len_keep就是len(mask_ids)
            # print(len_keep, mask_ids.shape)
        # ****
        # elif self.select_mask:
        #     len_keep, mask_ids = self.get_mask(ps, i, attn) # 返回保留 patch 数量和 mask 索引
        else:
            len_keep,mask_ids = ps,None
            
        # print("len_keep",len_keep) # 7882，也就是只保留mask_ids的前7882个patch
        # print("mask_ids",mask_ids.shape, mask_ids) # torch.Size([1, 40233])
        
        
        if self.baseline == 'dsmil':
            # forward online network
            student_logit,student_cls_feat= self.online_encoder(x,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True)

            # cl loss 计算蒸馏损失：用 student 和 teacher 的分类特征算 soft target cross entropy
            cls_loss= self.forward_cls_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)

            return student_logit, cls_loss, ps, len_keep
        else:
            # forward online network
            student_cls_feat= self.online_encoder(x,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True)

            # prediction
            student_logit = self.predictor(student_cls_feat)

            # cl loss
            cls_loss= self.forward_cls_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)

            return student_logit, cls_loss, ps, len_keep
    
    def forward_with_distill_loss(self, x, attn=None,labels_mask=None,teacher_cls_feat=None,i=None):
        # x: 输入的 patch-level 特征（维度: [batch, n_patches, feature_dim]）。
        # attn: 上一步 teacher 模型生成的 attention，用于指导 masking或者计算蒸馏损失。
        # teacher_cls_feat: teacher 输出的分类特征，用于计算蒸馏 loss。
        # i: 当前 iteration（用于 scheduler 控制 masking 比例变化）。
        # print("MHIM x.shape",x.shape) # torch.Size([1, 7974, 1024])
        x = self.patch_to_emb(x)
        x = self.dp(x)

        ps = x.size(1) # patch size

        # *** Get Hard Mask (mask_ids)  ***
        # ** use human annotation **
        print("use_human_annotation2",self.use_human_annotation)
        if self.use_human_annotation:
            # print("use human annotation")
            len_keep, mask_ids = self.get_mask_human_annotation(ps, i, labels_mask)#! 对于我们来说，mask_ids就是所有保留的id，len_keep就是len(mask_ids)
            # print(len_keep, mask_ids.shape)
        # ****
        # elif self.select_mask:
        #     len_keep, mask_ids = self.get_mask(ps, i, attn) # 返回保留 patch 数量和 mask 索引
        else:
            len_keep,mask_ids = ps, None
            
        # print("len_keep",len_keep) # 7882，也就是只保留mask_ids的前7882个patch
        # print("mask_ids",mask_ids.shape, mask_ids) # torch.Size([1, 40233])
        # 把attn也mask一下
        
        
        if self.baseline == 'dsmil':
            # forward online network
            student_logit,student_cls_feat, student_attn= self.online_encoder(x,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True,return_attn=True)

            # cl loss 计算特征蒸馏损失：用 student 和 teacher 的分类特征算 soft target cross entropy
            cls_loss = self.forward_cls_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)
            
            # attn_loss
            attn_loss = self.forward_attn_loss(student_attn=student_attn, teacher_attn=attn, mask_ids=mask_ids)

            return student_logit, cls_loss, attn_loss, ps, len_keep
        elif self.baseline == 'clam_sb':
            if mask_ids is not None:
                x, _,_ = self.masking(x, mask_ids, len_keep)
            A, h = self.online_encoder(x.squeeze())  # NxK   
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
            student_cls_feat = torch.mm(A, h) 
            student_logit = self.predictor(student_cls_feat)
            cls_loss= self.forward_cls_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)
            attn_loss = self.forward_attn_loss(student_attn=student_attn, teacher_attn=attn, mask_ids=mask_ids)
            return student_logit, cls_loss, attn_loss, ps, len_keep
        elif self.baseline == "clam_mb": 
            A, h = self.online_encoder(x.squeeze())  # NxK        
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N
            M = torch.mm(A, h) 
            device = M.device
            logits = torch.empty(1, len(self.predictor)).float().to(device)
            for c in range(len(self.predictor)):
                logits[0, c] = self.predictor[c](M[c])
            x = logits
        else:
            # forward online network
            student_cls_feat, student_attn = self.online_encoder(x,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True,return_attn=True)

            # prediction
            student_logit = self.predictor(student_cls_feat)

            # cl loss
            cls_loss= self.forward_cls_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)
            
            # attn_loss
            attn_loss = self.forward_attn_loss(student_attn=student_attn, teacher_attn=attn, mask_ids=mask_ids)

            return student_logit, cls_loss, attn_loss, ps, len_keep
