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
    # mlp_dim: 
    # pos:  positional encoding（ ppeg、sincos、peg）
    # peg_k: PEG 
    # head: Multi-Head Attention 
    # pos_pos:  positional encoding（）
    def __init__(self,mlp_dim=512,pos_pos=0,pos='ppeg',peg_k=7,head=8):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)
        #   class token， [1, 1, 512]，
        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
        #  Transformer ， Self-Attention  FFN 
        self.layer1 = TransLayer(dim=mlp_dim,head=head)
        self.layer2 = TransLayer(dim=mlp_dim,head=head)

        # 
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k) # Partial Positional Encoding via Grouped Convolution
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim) #  sin/cos 
        elif pos == 'peg':
            self.pos_embedding = PEG(512,k=peg_k) # Positional Encoding via Convolution
        else:
            self.pos_embedding = nn.Identity() # 

        self.pos_pos = pos_pos

    # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None,len_keep=None): # 
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        # print("x.shape",x.shape,"ids_shuffle.shape",ids_shuffle.shape)
        assert ids_shuffle is not None

        #  values，【】
        _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] #  len_keep  patch 
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #  x  patch，x_masked.shape = [B, len_keep, D]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device) #  "mask"（1）
        mask[:, :len_keep] = 0 #  len_keep  0（）
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) #  mask “” patch 

        # x_masked:  patch 
        # mask:  patch  0/1  mask，0 ，1 
        # ids_restore: 
        return x_masked, mask, ids_restore
    
    def soft_masking(self, x, ids_shuffle=None, len_keep=None): # 。patches，，attention。
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

        _, ids_restore = ids_shuffle.sort() # ids_shuffle ， .sort() 

        #  len_keep  patch  token 
        ids_keep = ids_shuffle[:, :len_keep]  # [N, len_keep]

        #  mask: [N, L]， 0
        keep_mask = torch.zeros(N, L, device=x.device)  # 0 
        keep_mask.scatter_(dim=1, index=ids_keep, value=1)  #  token  1

        keep_mask = keep_mask.unsqueeze(-1)  # [N, L, 1]
        keep_mask = keep_mask.expand(-1, -1, D)  #  D [N, L, D]

        #  keep_mask  x。 token （1）。 mask  token （0）。
        x_soft_masked = x * keep_mask  # [N, L, D]

        #  0/1  mask（0 ，1 ），
        binary_mask = torch.ones([N, L], device=x.device)
        binary_mask[:, :len_keep] = 0
        #  binary_mask  ids_restore ， mask
        mask = torch.gather(binary_mask, dim=1, index=ids_restore)

        return x_soft_masked, mask, ids_restore


    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,mask_enable=False):
        # x: ，shape  [batch_size, num_patches, feature_dim]。
        # mask_ids, len_keep:  Mask， patch。
        # return_attn:  attention（）。
        # mask_enable:  masking。
        batch, num_patches, C = x.shape 
        
        attn = []

        #  pos_pos  -2，（， mask ）
        if self.pos_pos == -2:
            x = self.pos_embedding(x)
        
        # masking
        if mask_enable and mask_ids is not None:
            # x, _, _ = self.masking(x,mask_ids,len_keep)
            x, _, _ = self.soft_masking(x,mask_ids,len_keep) # maskpatch，0

        # cls_token  class token（ ViT），。repeat  einops ， [1, 1, d]  [batch, 1, d]。
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
        x = torch.cat((cls_tokens, x), dim=1)
        
        #  -1（ class token ）
        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        # translayer1
        if return_attn:
            x,_attn = self.layer1(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)

        # add pos embedding
        #  0， patch（ cls token）
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
        #  cls_token （）
        logits = x[:,0,:]
 
        if return_attn:
            _a = attn
            return logits ,_a
        else:
            return logits
    