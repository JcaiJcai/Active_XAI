import torch
import torch.nn as nn
from einops import repeat
from .nystrom_attention import NystromAttention
from modules.emb_position import *

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = head,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x, need_attn=False):
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
            x = x+z
            return x,attn
        else:
            x = x + self.attn(self.norm(x))
            return x  

# Transmil
class SAttention(nn.Module):
    # mlp_dim: 输入的特征维度
    # pos: 选择哪种 positional encoding（如 ppeg、sincos、peg）
    # peg_k: PEG 的卷积核大小
    # head: Multi-Head Attention 的头数
    # pos_pos: 是否加 positional encoding（没用到）
    def __init__(self,mlp_dim=512,pos_pos=0,pos='ppeg',peg_k=7,head=8):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)
        # 创建一个 可学习的 class token，形状为 [1, 1, 512]，用于聚合整个序列的信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
        # 定义两个 Transformer 层，通常由 Self-Attention 和 FFN 构成
        self.layer1 = TransLayer(dim=mlp_dim,head=head)
        self.layer2 = TransLayer(dim=mlp_dim,head=head)

        # 位置编码方法
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k) # Partial Positional Encoding via Grouped Convolution
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim) # 固定的 sin/cos 编码
        elif pos == 'peg':
            self.pos_embedding = PEG(512,k=peg_k) # Positional Encoding via Convolution
        else:
            self.pos_embedding = nn.Identity() # 默认什么也不加

        self.pos_pos = pos_pos

    # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None,len_keep=None): # 原来的方法
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # print("x.shape",x.shape,"ids_shuffle.shape",ids_shuffle.shape)
        assert ids_shuffle is not None

        # 不关心排序后的值 values，只关心排序后对应的【原始索引】
        _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # 取前 len_keep 个 patch 的索引
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # 按这些索引去 x 中提取对应的 patch，最后得到的x_masked.shape = [B, len_keep, D]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device) # 全部标记为 "要mask"（值为1）
        mask[:, :len_keep] = 0 # 前 len_keep 个位置标记为 0（表示保留）
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # 把 mask 顺序“还原”回原始 patch 顺序

        # x_masked: 被保留的 patch 特征
        # mask: 原始 patch 顺序下的 0/1 二值 mask，0 表示保留，1 表示遮住
        # ids_restore: 恢复顺序用的索引
        return x_masked, mask, ids_restore
    
    def soft_masking(self, x, ids_shuffle=None, len_keep=None): # 我们修改过的方法。不真正去掉patches，而是保留它们的位置，以获取对应attention。
        """
        Soft token masking by setting unselected tokens' features to zero (instead of dropping them).
        
        Args:
            x: Tensor of shape [N, L, D] - input token features
            ids_shuffle: Tensor of shape [N, L] - shuffled token indices for each sample
            len_keep: int - number of tokens to keep per sample

        Returns:
            x_soft_masked: [N, L, D] - same shape as input, unkept tokens are zeroed
            mask: [N, L] - binary mask, 0 for keep, 1 for masked
            ids_restore: [N, L] - indices to restore original order
        """
        N, L, D = x.shape # [batch_size, num_tokens, feature_dim]
        assert ids_shuffle is not None

        _, ids_restore = ids_shuffle.sort() # ids_shuffle 是提前打乱后的索引，我们通过 .sort() 恢复原顺序

        # 取前 len_keep 个 patch 的索引作为保留的 token 索引
        ids_keep = ids_shuffle[:, :len_keep]  # [N, len_keep]

        # 构造 mask: [N, L]，初始全为 0
        keep_mask = torch.zeros(N, L, device=x.device)  # 0 表示不保留
        keep_mask.scatter_(dim=1, index=ids_keep, value=1)  # 把保留的 token 标记为 1

        keep_mask = keep_mask.unsqueeze(-1)  # 加一个维度[N, L, 1]
        keep_mask = keep_mask.expand(-1, -1, D)  # 复制 D 次变成[N, L, D]

        # 用 keep_mask 乘以原来的 x。保留的 token 特征保持不变（乘以1）。被 mask 的 token 特征全部归零（乘以0）。
        x_soft_masked = x * keep_mask  # [N, L, D]

        # 生成 0/1 的 mask（0 表保留，1 表屏蔽），再恢复顺序
        binary_mask = torch.ones([N, L], device=x.device)
        binary_mask[:, :len_keep] = 0
        # 把 binary_mask 按照 ids_restore 重新排列，还原成输入原顺序对应的 mask
        mask = torch.gather(binary_mask, dim=1, index=ids_restore)

        return x_soft_masked, mask, ids_restore


    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,mask_enable=False):
        # x: 输入特征，shape 是 [batch_size, num_patches, feature_dim]。
        # mask_ids, len_keep: 如果启用了 Mask，指定要保留哪些 patch。
        # return_attn: 是否返回 attention（用于解释）。
        # mask_enable: 是否启用 masking。
        batch, num_patches, C = x.shape 
        
        attn = []

        # 如果 pos_pos 设置为 -2，先加位置编码（比较少见，表示在 mask 前加位置编码）
        if self.pos_pos == -2:
            x = self.pos_embedding(x)
        
        # masking
        if mask_enable and mask_ids is not None:
            # x, _, _ = self.masking(x,mask_ids,len_keep)
            x, _, _ = self.soft_masking(x,mask_ids,len_keep) # 保留被mask掉的patch，对应特征设为0

        # cls_token 加入 class token（类似 ViT），用于最终聚合。repeat 是 einops 语法，把 [1, 1, d] 扩展成 [batch, 1, d]。
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 如果位置编码位置为 -1（表示在拼接 class token 之后加）加位置编码
        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        # translayer1
        if return_attn:
            x,_attn = self.layer1(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)

        # add pos embedding
        # 如果位置编码的位置是 0，就只对 patch（不包括 cls token）加位置编码
        if self.pos_pos == 0:
            x[:,1:,:] = self.pos_embedding(x[:,1:,:])
        
        # translayer2
        if return_attn:
            x,_attn = self.layer2(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer2(x)

        #---->cls_token
        x = self.norm(x)
        # 只取 cls_token 这一位作为最终输出特征（用于分类）
        logits = x[:,0,:]
 
        if return_attn:
            _a = attn
            return logits ,_a
        else:
            return logits
    