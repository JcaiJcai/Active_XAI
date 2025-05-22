import torch
import numpy as np
from torch import nn
from modules.datten import *
import torch.nn.functional as F
from modules.satten import *

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
    def __init__(self, mlp_dim=512,mask_ratio=0,n_classes=2,temp_t=1.,temp_s=1.,dropout=0.25,act='relu',select_mask=False, use_human_mask=False, use_attention_loss=False, use_annotation_loss=False, select_inv=False,msa_fusion='vote',mask_ratio_h=0.,mrh_sche=None,mask_ratio_hr=0.,mask_ratio_l=0.,da_act='gelu',baseline='selfattn',head=8,attn_layer=0):
        super(MHIM, self).__init__()
 
        self.mask_ratio = mask_ratio #  mask 
        self.mask_ratio_h = mask_ratio_h #  attention  mask 
        self.mask_ratio_hr = mask_ratio_hr #  attention 
        self.mask_ratio_l = mask_ratio_l #  attention  mask 
        self.select_mask = select_mask #  mask 
        self.select_inv = select_inv #  mask （ patch）
        self.msa_fusion = msa_fusion #  attention 
        self.mrh_sche = mrh_sche #  mask_ratio_h （）
        self.attn_layer = attn_layer #  encoder  attention，
        self.baseline = baseline # （ selfattn, dsmil, attn）
        
        self.use_human_mask = use_human_mask
        self.use_attention_loss = use_attention_loss
        self.use_annotation_loss = use_annotation_loss
        
        #  energy_alphas 
        init_alphas=[-1.0, 0.0, 1.0]
        self.alphas = nn.Parameter(torch.tensor(init_alphas, dtype=torch.float32))

        self.patch_to_emb = [nn.Linear(1024, 512)] #  patch （1024） 512 

        # 
        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]
        # dropout
        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        #  Sequential 
        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        if baseline == 'selfattn': # Transmil
            self.online_encoder = SAttention(mlp_dim=mlp_dim,head=head)
        elif baseline == 'attn':  # ABMIL
            self.online_encoder = DAttention(mlp_dim,da_act)
        elif baseline == 'dsmil': # DSMIL
            self.online_encoder = DSMIL(mlp_dim=mlp_dim,mask_ratio=mask_ratio)

        self.predictor = nn.Linear(mlp_dim,n_classes)
        self.evidence_head = nn.Linear(mlp_dim, n_classes)

        #  teacher  student  softmax （ CL）
        self.temp_t = temp_t
        self.temp_s = temp_s

        #  soft cross-entropy 
        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t,self.temp_s)

        self.predictor_cl = nn.Identity()
        self.target_predictor = nn.Identity()

        # 
        self.apply(initialize_weights)

    #  attention 、mask 、mask （ or ）， patch  mask， masked modeling 。
    def select_mask_fn(self,ps,attn,largest,mask_ratio,mask_ids_other=None,len_keep_other=None,cls_attn_topk_idx_other=None,random_ratio=1.,select_inv=False):
        # ps： sample  patch 。
        # attn：attention 。
        # largest：True ；False 。
        # mask_ratio： mask 。
        # mask_ids_other： mask index（ mask）。
        # len_keep_other： mask （ mask）。
        # cls_attn_topk_idx_other： top-k index（）。
        # random_ratio： top-k （<1）。
        # select_inv：。
        
        ps_tmp = ps
        mask_ratio_ori = mask_ratio
        mask_ratio = mask_ratio / random_ratio
        if mask_ratio > 1:
            random_ratio = mask_ratio_ori
            mask_ratio = 1.
            
        # print(attn.size())
        if mask_ids_other is not None: # （ mask）， patch 
            if cls_attn_topk_idx_other is None:
                cls_attn_topk_idx_other = mask_ids_other[:,len_keep_other:].squeeze()
                ps_tmp = ps - cls_attn_topk_idx_other.size(0)
                
        if len(attn.size()) > 2:
            if self.msa_fusion == 'mean': # ， top-k
                _,cls_attn_topk_idx = torch.topk(attn,int(np.ceil((ps_tmp*mask_ratio)) // attn.size(1)),largest=largest)
                cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3,-1))
            elif self.msa_fusion == 'vote': #  top-k， voting
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
        else: #  attention  top-k
            k = int(np.ceil((ps_tmp*mask_ratio)))
            _,cls_attn_topk_idx = torch.topk(attn,k,largest=largest)
            cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)
        
        # randomly 
        if random_ratio < 1.: #  top-k 
            random_idx = torch.randperm(cls_attn_topk_idx.size(0),device=cls_attn_topk_idx.device)

            cls_attn_topk_idx = torch.gather(cls_attn_topk_idx,dim=0,index=random_idx[:int(np.ceil((cls_attn_topk_idx.size(0)*random_ratio)))])
        

        # concat other masking idx  mask
        if mask_ids_other is not None:
            cls_attn_topk_idx = torch.cat([cls_attn_topk_idx,cls_attn_topk_idx_other]).unique()

        # if cls_attn_topk_idx is not None: 
        len_keep = ps - cls_attn_topk_idx.size(0)
        a = set(cls_attn_topk_idx.tolist())
        b = set(list(range(ps)))
        mask_ids =  torch.tensor(list(b.difference(a)),device=attn.device) #  mask id
        
        # 
        if select_inv:
            mask_ids = torch.cat([cls_attn_topk_idx,mask_ids]).unsqueeze(0)
            len_keep = ps - len_keep
        else:
            mask_ids = torch.cat([mask_ids,cls_attn_topk_idx]).unsqueeze(0)

        return len_keep,mask_ids #  patch ，mask patch index

    # ****** mask ******
    def get_mask(self,ps,i,attn,mrh=None):
        # ps： patch （patch size）。
        # i：（epoch * iter） mask 。
        # attn：attention （ teacher ）。
        # mrh：mask_ratio_h（ attention masking ）。
        #  attn  attention（ transformer layer），
        if attn is not None and isinstance(attn,(list,tuple)):
            if self.attn_layer == -1:
                attn = attn[1] #  1  attention
            else:
                attn = attn[self.attn_layer]
        else:
            attn = attn

        # random mask  mask
        if attn is not None and self.mask_ratio > 0.:
            len_keep,mask_ids = self.select_mask_fn(ps,attn,False,self.mask_ratio,select_inv=self.select_inv,random_ratio=0.001)
        else:
            len_keep,mask_ids = ps,None

        # low attention mask  attention patch（）
        if attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep,mask_ids = self.select_mask_fn(ps,attn,False,self.mask_ratio_l,select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:,:len_keep].squeeze() if self.select_inv else mask_ids[:,len_keep:].squeeze()
                len_keep,mask_ids = self.select_mask_fn(ps,attn,False,self.mask_ratio_l,select_inv=self.select_inv,mask_ids_other=mask_ids,len_keep_other=ps,cls_attn_topk_idx_other = cls_attn_topk_idx_other)
        
        # high attention mask  attention patch
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
        # mask_idsid，len_keeplen(mask_ids)
        # labels_mask？(36710,)patch(36710, 1024)[i]label
        # print(labels_mask.shape) # torch.Size([36710])
        
        # # 2patch
        # mask_ids = torch.nonzero(labels_mask == 2, as_tuple=False).view(-1)

        # # 2，1
        # if mask_ids.numel() == 0:
        #     mask_ids = torch.nonzero(labels_mask == 1, as_tuple=False).view(-1)
            
        # ：tumornormal，0
        mask_ids = torch.nonzero(labels_mask != 0, as_tuple=False).view(-1)

        len_keep = mask_ids.shape[0]
        
        return len_keep, mask_ids.unsqueeze(0) # batch

    @torch.no_grad() 
    def forward_teacher(self,x): #  teacher 

        x = self.patch_to_emb(x) #  patch-level  x （embedding），
        x = self.dp(x) #  dropout 

        if self.baseline == 'dsmil': #  DSMIL ，encoder ，
            _,x,attn = self.online_encoder(x,return_attn=True)
        else: #  bag-level  x  attention  attn
            x,attn = self.online_encoder(x,return_attn=True)

        return x, attn
    
    def forward_teacher_edl(self,x): #  teacher 
        with torch.no_grad():
            x = self.patch_to_emb(x) #  patch-level  x （embedding），
            x = self.dp(x) #  dropout 

            if self.baseline == 'dsmil': #  DSMIL ，encoder ，
                _,x,attn = self.online_encoder(x,return_attn=True)
                x = x[:, 0, :]
            else: #  bag-level  x  attention  attn
                x,attn = self.online_encoder(x,return_attn=True)

        # print("x.shape", x.shape) # torch.Size([1, 512])
        evidence = F.relu(self.evidence_head(x))
        alpha = evidence + 1.0
        return alpha, x, attn
    
    @torch.no_grad()
    def forward_test(self,x,return_attn=False,no_norm=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)

        if return_attn:
            # x,a = self.online_encoder(x,return_attn=True,no_norm=no_norm)
            x,a = self.online_encoder(x,return_attn=True)
        else:
            x = self.online_encoder(x)

        if self.baseline == 'dsmil':
            pass
        else:   
            x = self.predictor(x)

        if return_attn:
            return x,a
        else:
            return x

    def pure(self,x):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if self.baseline == 'dsmil':
            x,_ = self.online_encoder(x)
        else:
            x = self.online_encoder(x)
            x = self.predictor(x)

        if self.training:
            return x, 0, ps,ps
        else:
            return x

    # forward_loss
    def forward_cls_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat,teacher_cls_feat.detach())
        else:
            cls_loss = 0.
        
        return cls_loss
    
    
    def forward_attn_loss(self, student_attn, teacher_attn, mask_ids):
        # print("student_attn.shape",student_attn[0].shape, student_attn) # student_attn， shape  [1, 8, 16124]。
        if self.baseline == "selfattn": # Transmil or Abmil
            N, L, D = student_attn[0].shape # [batch, head, tokens] e.g.[1, 8, 16124]
            keep_mask = torch.ones(N, D).to(student_attn[0].device)
            if mask_ids is not None:
                keep_mask.scatter_(dim=1, index=mask_ids, value=0)
                keep_mask = keep_mask.unsqueeze(1)  # [N, 1, D]
                keep_mask = keep_mask.expand(-1, L, -1)  #  L [N, L, D]
            distill_loss = 0.
            for i in range(len(student_attn)):
                if teacher_attn is not None:
                    # print(teacher_attn[i].shape, student_attn[i].shape, keep_mask.shape)
                    # print(teacher_attn[i].device, keep_mask.device)
                    teacher_masked = teacher_attn[i] * keep_mask
                    student_masked = student_attn[i] * keep_mask
                    # [Important] The shape of teacher_masked is [1, 8, 5587]
                    distill_loss += - (teacher_masked.softmax(dim=-1) * torch.log_softmax(student_masked, dim=-1)).sum(dim=-1).mean()
        elif self.baseline == "dsmil" or self.baseline == "attn":
            # print(student_attn.shape, teacher_attn.shape) # [1, 92924], [1, 92924]
            distill_loss = - (student_attn.softmax(dim=-1) * torch.log_softmax(teacher_attn, dim=-1)).sum(dim=-1).mean()
        return distill_loss
    
    def forward_annotation_loss(self, student_attn, annotation, anno_loss_type, mask_ids=None, energy_alphas=(-1.0, 0.0, 1.0)):
        if anno_loss_type == "L1":
            student_attn = student_attn[1] # attention
            student_attn = student_attn.mean(dim=1).squeeze(0) # 
            
            attn_sum = student_attn.sum(dim=-1, keepdim=True)
            attn_norm = student_attn / attn_sum
            
            annot_norm = annotation.float() / 2.0
            l1_loss = torch.nn.L1Loss(reduction='mean')
            annotation_loss = l1_loss(attn_norm, annot_norm)
        elif anno_loss_type == "L2":
            student_attn = student_attn[1] # attention
            student_attn = student_attn.mean(dim=1).squeeze(0) # 
            
            attn_sum = student_attn.sum(dim=-1, keepdim=True)
            attn_norm = student_attn / attn_sum
            
            annot_norm = annotation.float() / 2.0
            l2_loss = torch.nn.L2Loss(reduction='mean')
            
            annotation_loss = l2_loss(attn_norm, annot_norm)
        elif anno_loss_type == "entropy":
            student_attn = student_attn[1] # attention
            student_attn = student_attn.mean(dim=1).squeeze(0) # 
            
            attn_sum = student_attn.sum(dim=-1, keepdim=True)
            attn_norm = student_attn / attn_sum
            
            annot_norm = annotation.float() / 2.0
            
            bce_loss = torch.nn.BCELoss(reduction='mean')
            annotation_loss = bce_loss(attn_norm, annot_norm)
        elif anno_loss_type == "energy":
            # annotation: label0, label1, label2
            # print("student_attn",student_attn.shape)
            B, *rest = annotation.shape
            if self.baseline == "selfattn": # transmil or abmil
                student_attn = student_attn[1] # attention
                student_flat = student_attn.view(B, -1)
            elif self.baseline == "dsmil" or self.baseline == "attn":
                # print("student_attn",student_attn.shape) # torch.Size([1, 48354])
                student_flat = student_attn.view(B, -1)
                # print("student_attn2",student_attn.shape) # torch.Size([1, 48354])
            
            # [Important] The shape of student_flat should be [num_of_data, num_of_head]
            # e.g. [47345, 8] for transmil and [47345, 1] for dsmil
            # print("student_flat", student_flat.shape)
            annot_flat   = annotation.view(B, -1)
            sum_attn = student_flat.sum(dim=1, keepdim=True) + 1e-6  # [B,1]

            Ec_terms = []
            # print("self.alphas",self.alphas)
            for c in range(self.alphas.shape[0]):
                mask_c = (annot_flat == c).float()           # [B, N]
                num = (mask_c * student_flat).sum(dim=1)     # [B]
                Ec = num / (student_flat.sum(dim=1) + 1e-6)  # [B]
                Ec_terms.append(self.alphas[c] * Ec)         # alpha[c] * Ec

            Ec_stack = torch.stack(Ec_terms, dim=1)              # [B,3]
            # - sum_c α_c * E_c， batch 
            return -Ec_stack.sum(dim=1).mean()
            
        return annotation_loss
        
        
    # ** Just for Student Model **
    def forward(self, x, attn=None,labels_mask=None,teacher_cls_feat=None,i=None):
        # x:  patch-level （: [batch, n_patches, feature_dim]）。
        # attn:  teacher  attention， masking。
        # teacher_cls_feat: teacher ， loss。
        # i:  iteration（ scheduler  masking ）。
        # print("MHIM x.shape",x.shape) # torch.Size([1, 7974, 1024])
        x = self.patch_to_emb(x)
        x = self.dp(x)

        ps = x.size(1) # patch size

        # *** Get Hard Mask (mask_ids)  ***
        # ** use human annotation **
        # print("use_human_annotation1",self.use_human_annotation)
        if self.use_human_mask:
            # print("use human annotation")
            len_keep, mask_ids = self.get_mask_human_annotation(ps, i, labels_mask)#! ，mask_idsid，len_keepl(mask_ids)
        else:
            len_keep,mask_ids = ps,None
            
        # print("len_keep",len_keep) # 7882，mask_ids7882patch
        # print("mask_ids",mask_ids.shape, mask_ids) # torch.Size([1, 40233])
        
        
        if self.baseline == 'dsmil':
            # forward online network
            student_logit,student_cls_feat= self.online_encoder(x,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True)

            # cl loss ： student  teacher  soft target cross entropy
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
        
    def forward_with_distill_loss(self, x, attn=None,labels_mask=None, anno_loss_type=None, teacher_cls_feat=None,i=None):
        # x:  patch-level （: [batch, n_patches, feature_dim]）。
        # attn:  teacher  attention， masking。
        # teacher_cls_feat: teacher ， loss。
        # i:  iteration（ scheduler  masking ）。
        # print("MHIM x.shape",x.shape) # torch.Size([1, 7974, 1024])
        x = self.patch_to_emb(x)
        x = self.dp(x)

        ps = x.size(1) # patch size

        # *** Get Hard Mask (mask_ids)  ***
        # ** use human annotation **
        # print("use_human_annotation2",self.use_human_annotation)
        if self.use_human_mask:
            # print("use human annotation")
            len_keep, mask_ids = self.get_mask_human_annotation(ps, i, labels_mask)#! ，mask_idsid，len_keeplen(mask_ids)
        else:
            len_keep,mask_ids = ps,None
            
        # print("len_keep",len_keep) # 7882，mask_ids7882patch
        # print("mask_ids",mask_ids.shape, mask_ids) # torch.Size([1, 40233])
        # attnmask
        
        if self.baseline == 'dsmil':
            # forward online network
            student_logit,student_cls_feat, student_attn= self.online_encoder(x,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True,return_attn=True)

            # cl loss ： student  teacher  soft target cross entropy
            cls_loss = self.forward_cls_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)
            
            # attn_loss
            if self.use_attention_loss == True:
                attn_loss = self.forward_attn_loss(student_attn=student_attn, teacher_attn=attn, mask_ids=mask_ids)
            else: attn_loss = 0.
            
            if self.use_annotation_loss == True:
                annotation_loss = self.forward_annotation_loss(student_attn=student_attn, annotation=labels_mask, anno_loss_type=anno_loss_type, mask_ids=mask_ids)
            else: annotation_loss = 0.

            return student_logit, cls_loss, attn_loss, annotation_loss, ps, len_keep
        else:
            # forward online network
            student_cls_feat, student_attn = self.online_encoder(x,len_keep=len_keep,mask_ids=mask_ids,mask_enable=True,return_attn=True)

            # prediction
            student_logit = self.predictor(student_cls_feat)

            # cl loss
            cls_loss= self.forward_cls_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)
            
            # attn_loss
            if self.use_attention_loss == True:
                attn_loss = self.forward_attn_loss(student_attn=student_attn, teacher_attn=attn, mask_ids=mask_ids)
            else: attn_loss = 0.
            
            if self.use_annotation_loss == True:
                annotation_loss = self.forward_annotation_loss(student_attn=student_attn, annotation=labels_mask, anno_loss_type=anno_loss_type, mask_ids=mask_ids)
            else: annotation_loss = 0.
            
            return student_logit, cls_loss, attn_loss, annotation_loss, ps, len_keep
