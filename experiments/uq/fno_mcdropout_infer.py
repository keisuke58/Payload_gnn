import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from models_fno import FNO2d

class FNO2dDropout(FNO2d):
    def __init__(self, modes1=12, modes2=12, width=32, in_channels=4, out_channels=1, p=0.2):
        super().__init__(modes1, modes2, width, in_channels, out_channels)
        self.dropout = nn.Dropout(p)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.dropout(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

def save_field_png(field, path):
    arr = field.squeeze().detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    img = (arr * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser(description='FNO MC Dropout UQ')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--modes1', type=int, default=12)
    parser.add_argument('--modes2', type=int, default=12)
    parser.add_argument('--dropout_p', type=float, default=0.2)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--grid', type=int, nargs=2, default=[64, 64])
    parser.add_argument('--output_dir', type=str, default='experiments/uq/results_fno')
    parser.add_argument('--input_npy', type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNO2dDropout(args.modes1, args.modes2, args.width, args.in_channels, args.out_channels, args.dropout_p).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt if isinstance(ckpt, dict) else None)
        if isinstance(sd, dict):
            model.load_state_dict(sd, strict=False)
    model.train()
    if args.input_npy and os.path.exists(args.input_npy):
        arr = np.load(args.input_npy)
        x = torch.from_numpy(arr).float().unsqueeze(0).to(device)
    else:
        x = torch.randn(1, args.in_channels, args.grid[0], args.grid[1], device=device)
    preds = []
    for _ in range(args.T):
        y = model(x)
        preds.append(y.detach().cpu())
    preds = torch.stack(preds, dim=0)
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)
    save_field_png(mean[0], os.path.join(args.output_dir, 'mean.png'))
    save_field_png(std[0], os.path.join(args.output_dir, 'std.png'))
    stats = {
        'mean_min': float(mean.min()),
        'mean_max': float(mean.max()),
        'std_mean': float(std.mean()),
        'std_max': float(std.max()),
        'T': args.T,
        'dropout_p': args.dropout_p
    }
    with open(os.path.join(args.output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
