import openslide
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from modules.mhim import MHIM
import cv2
import random
import seaborn as sns
import os

import openslide
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_features_and_coords(h5_path):
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:])  # shape: [N, D]
        coords = f['coords'][:]  # shape: [N, 2], each is [x, y]
    print(f"Loaded {features.shape[0]} patches, feature dim: {features.shape[1]}, coords shape: {coords.shape}")
    return features, coords

def show_attention_on_region(svs_path, coords, coords_attn, 
                             patch_size=256, level=0, alpha=0.5,
                             region=None,
                             mask_path=None, mask_level=8):  # <<< 新增参数
    # * -------------- 读取原图片对应region部分（level0） --------------
    slide = openslide.OpenSlide(svs_path)
    
    if region is None:
        left, top = 0, 0
        right, bottom = slide.level_dimensions[level]
    else:
        left, top, right, bottom = region

    w, h = right - left, bottom - top
    base_img = np.array(slide.read_region((left, top), level, (w, h)).convert("RGB"))
    heatmap = base_img.copy()
    print(f"[INFO] Reading image region at level {level}")
    print(f"        Region coordinates (level 0): left={left}, top={top}, width={w}, height={h}, right={right}, bottom={bottom}")
    print(f"[INFO] Image region loaded. Shape: {heatmap.shape}")

    # normalize attention
    print("[INFO] Attn min:", attn_norm.min(), "max:", attn_norm.max(),"attn_norm shape",attn_norm.shape)
    
    for i, (x, y) in enumerate(coords): # x,x+256,y,y+256对应原图在level0的一个小方块
        if region and not (left <= x < right and top <= y < bottom):
            continue  # skip patches outside region
        rel_x = int(x - left)
        rel_y = int(y - top)
        patch_w = patch_h = max(int(patch_size), 1)
        # print("pt1", (rel_x, rel_y), "pt2", (rel_x + patch_w, rel_y + patch_h))
        # pt1 (144, 400) pt2 (400, 656)
        # pt1 (13968, 7568) pt2 (14224, 7824)
        color = plt.cm.jet(coords_attn[(x,y)])[:3]
        color = tuple(int(255 * c) for c in color)
        # 为每个 patch 用其 attention score 映射的颜色，画出对应的矩形区域，从而形成整个 attention heatmap 叠加图
        cv2.rectangle(
            img=heatmap,                          # 要画的图像（NumPy array，通常是 RGB）
            pt1=(rel_x, rel_y),                   # 左上角坐标
            pt2=(rel_x + patch_w, rel_y + patch_h), # 右下角坐标
            color=color,                          # 矩形颜色（B, G, R）格式
            thickness=-1                          # 填充整个矩形（如果是 2 表示边框宽度为 2）
        )
        break

    blended = cv2.addWeighted(heatmap, alpha, base_img, 1 - alpha, 0)
    
    # * -------------- 加载mask并绘制轮廓 --------------
    # if mask_path is not None:
    #     mask_slide = openslide.OpenSlide(mask_path)
    #     mask_downsample = mask_slide.level_downsamples[mask_level]
    #     width, height = mask_slide.level_dimensions[mask_level]

    #     # 将 region 从 level 0 映射到 mask_level=8（用于读取子图）
    #     w_mask = int(w / mask_downsample)
    #     h_mask = int(h / mask_downsample)
        
    #     # 读取指定区域的局部 mask（坐标必须是 level 0 下的）
    #     # print(f"[INFO] Reading mask region at level {mask_level}: origin=({left}, {top}), size=({w_mask}, {h_mask})")
    #     mask_img = mask_slide.read_region((left, top), 0, (w, h)).convert("L")
    #     mask_arr = np.array(mask_img)
        
    #     print(f"[INFO] Partial mask shape: {mask_arr.shape}, unique labels: {np.unique(mask_arr)}")
        
    #     # 提取 tumor 区域并绘制边界（仅 label==2）
    #     tumor_mask = (mask_arr == 2).astype(np.uint8)
        
    #     # Important! 确保tumor_mask和heatmap对的上！！！
    #     print(f"  tumor_mask shape: {tumor_mask.shape}")
    #     print(f"  heatmap shape   : {heatmap.shape}")
        
        
    #     contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     print(f"[INFO] Found {len(contours)} contours")
        
    #     # 将轮廓绘制到heatmap（黑色）
    #     cv2.drawContours(
    #         image=blended,        # 要绘制轮廓的图像（NumPy array，通常是 RGB）
    #         contours=contours,    # 一个 list，每个元素是一组轮廓点（坐标数组）
    #         contourIdx=-1,        # -1 表示画所有轮廓（不是只画第几个）
    #         color=(0, 0, 0),      # 绘制轮廓的颜色（BGR）→ 这是黑色
    #         thickness=200           # 轮廓线宽为 2 像素
    #     )
                
    # ------------------------------------------------

    # show image
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Attention Heatmap + Tumor Mask")
    # plt.show()
    
    plt.savefig("heatmap_mask.jpg", format='jpg', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
features, coords = load_features_and_coords("/u/jcai1/code/usefulxai/data/camelyon16-clam/tumor_vs_normal_resnet_features/h5_files_labels/normal_001.h5")

# Attention for TransMIL
pure = MHIM(da_act='relu',baseline='selfattn').eval()
pure.requires_grad_(False)

pure_cpt = torch.load('/u/jcai1/code/usefulxai/paper_results/pure_model/transmil_0507_1818/fold_0_model_best_auc.pt',map_location='cpu')

pure.load_state_dict(pure_cpt['model'], strict=False)
pure.eval()

with torch.no_grad():
    _, attns = pure.forward_test(features.unsqueeze(0),return_attn=True)
    attn = attns[-1].mean(dim=1).squeeze(0) # Take last layer
    print(attn.shape)
attn_scores = attn.cpu().numpy()
print(attn_scores)

# 得到每个坐标的attention
attn_norm = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-6)
coords_attn = {
    tuple(coords[i]): float(attn_norm[i]) for i in range(len(coords))
}

# region = (70144, 122624, 73472, 134656)  # 在 level=0 下的 region
region = (60144, 112624, 74216, 144400)
show_attention_on_region(
    svs_path="/u/jcai1/code/usefulxai/data/camelyon16/images/tumor_001.tif",
    coords=coords,
    coords_attn=coords_attn,
    region=region,
    mask_path="/u/jcai1/code/usefulxai/data/camelyon16/masks/tumor_001_mask.tif"
)
