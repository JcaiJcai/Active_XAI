"""
main_draw_attention.py

Generate attention heatmaps for Whole Slide Images (WSI) using TransMIL or DSMIL models.
"""
import argparse
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

# Import your model definitions here
# from models.transmil import TransMIL
# from models.dsmil import MILNet, IClassifier, BClassifier


def load_features(h5_path: str, device: torch.device) -> torch.Tensor:
    """Load patch features from HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        feats = f['features'][:]
    return torch.from_numpy(feats).unsqueeze(0).to(device)


def load_coords(csv_path: str) -> np.ndarray:
    """Load patch (row,col) coordinates from CSV."""
    df = pd.read_csv(csv_path)
    return df[['row', 'col']].values


def extract_attention_transmil(model, features: torch.Tensor) -> np.ndarray:
    """Extract attention scores from a TransMIL model."""
    model.eval()
    with torch.no_grad():
        _, attns, _ = model(features)
        # Take last layer, first head, skip CLS token
        attn = attns[-1][0, 0, 1:]
    return attn.cpu().numpy()


def extract_attention_dsmil(model, features: torch.Tensor) -> np.ndarray:
    """Extract attention scores from a DSMIL model."""
    model.eval()
    with torch.no_grad():
        feats, _ = model.i_classifier(features[0])
        _, A, _ = model.b_classifier(feats, None)
    return A.cpu().squeeze(-1).numpy()


def extract_attention(args):
    # *** Load model (pure) 
    model = mhim.MHIM(select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)
    
    # *** Initialize the model parameters (未完成)
    print('######### Model Initializing.....')
    pre_dict = torch.load(trained_model_path)
    if 'model' in pre_dict:
        pre_dict = pre_dict['model'] # 取出实际参数
    info = model.load_state_dict(pre_dict,strict=False)
    print(info)
    
    # *** Get attention metrix (未完成)
    if args.model_type == 'transmil':
        model = TransMIL(input_dim=features.shape[-1]).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        attn = extract_attention_transmil(model, features)
    else:
        i_clf = IClassifier(...).to(device)  # adjust params
        b_clf = BClassifier(...).to(device)
        model = MILNet(i_clf, b_clf).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        attn = extract_attention_dsmil(model, features)
    
    return attn


def draw_attention_map(
    attn: np.ndarray,
    coords: np.ndarray,
    thumbnail: str,
    patch_size: int,
    downsample: int,
    output_path: str
) -> None:
    """Overlay attention heatmap onto the slide thumbnail and save."""
    img = cv2.imread(thumbnail)
    h, w = img.shape[:2]
    mask = np.zeros((h//downsample, w//downsample), dtype=float)
    block = patch_size // downsample

    for (r, c), score in zip(coords, attn):
        y, x = int(r // downsample), int(c // downsample)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            mask[y:y+block, x:x+block] = score

    # Normalize mask to [0,1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"Saved heatmap: {output_path}")
    

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate WSI attention heatmap using TransMIL or DSMIL."
    )
    parser.add_argument('--model_type', choices=['transmil', 'dsmil'], required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--h5_path', required=True)
    parser.add_argument('--coords_csv', required=True)
    parser.add_argument('--thumbnail', required=True)
    parser.add_argument('--output', default='attention_overlay.png')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--downsample', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    return args


def main():
    
    args = parse_arguments()
    
    # *** Load data ***
    device = torch.device(args.device)
    features = load_features(args.h5_path, device)
    coords = load_coords(args.coords_csv)

    # *** Load model and get attention ***
    attn = extract_attention()      
    
    # *** Draw attention map ***
    draw_attention_map(
        attn, coords, args.thumbnail,
        patch_size=args.patch_size,
        downsample=args.downsample,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
