
import os
import sys
import argparse
import time
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models_point import SimplePointTransformer, SimplePointNetSeg

# Re-use FocalLoss and Metrics from train.py (or copy them here for standalone)
# For simplicity, I'll copy them to avoid import issues if train.py changes
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, num_classes=2):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        else:
            if num_classes == 2:
                self.register_buffer('alpha', torch.tensor([1 - alpha, alpha], dtype=torch.float))
            else:
                self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        return focal.mean()

def compute_metrics(logits, targets, num_classes=2):
    preds = logits.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    acc = (preds == targets_np).mean()
    
    if num_classes == 2:
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        f1 = f1_score(targets_np, preds, zero_division=0)
        prec = precision_score(targets_np, preds, zero_division=0)
        rec = recall_score(targets_np, preds, zero_division=0)
        try:
            auc = roc_auc_score(targets_np, probs)
        except:
            auc = 0.0
    else:
        f1 = f1_score(targets_np, preds, average='macro', zero_division=0)
        prec = precision_score(targets_np, preds, average='macro', zero_division=0)
        rec = recall_score(targets_np, preds, average='macro', zero_division=0)
        auc = 0.0

    return {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'auc': auc}

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Prepare input for PointTransformer (Batch, N, Feat)
        x_dense, mask = to_dense_batch(batch.x, batch.batch)
        
        # Forward
        out = model(x_dense) # (Batch, Classes, N)
        
        # Flatten for loss calculation
        # out: (Batch, C, N) -> (Batch, N, C) -> (Batch*N, C)
        out = out.permute(0, 2, 1) 
        
        # Mask valid nodes
        valid_out = out[mask] # (Total_Nodes, C)
        valid_y = batch.y # (Total_Nodes)
        
        loss = criterion(valid_out, valid_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        all_logits.append(valid_out.detach())
        all_targets.append(valid_y.detach())
        
    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat)
    metrics['loss'] = avg_loss
    return metrics

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    
    for batch in loader:
        batch = batch.to(device)
        
        x_dense, mask = to_dense_batch(batch.x, batch.batch)
        out = model(x_dense)
        
        out = out.permute(0, 2, 1)
        valid_out = out[mask]
        valid_y = batch.y
        
        loss = criterion(valid_out, valid_y)
        
        total_loss += loss.item() * batch.num_graphs
        all_logits.append(valid_out)
        all_targets.append(valid_y)
        
    avg_loss = total_loss / len(loader.dataset)
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat)
    metrics['loss'] = avg_loss
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed_25mm_100')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2) # PointNet takes memory
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='runs')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    train_path = os.path.join(args.data_dir, 'train.pt')
    val_path = os.path.join(args.data_dir, 'val.pt')
    
    if not os.path.exists(train_path):
        print(f"Data not found: {train_path}")
        return
        
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Init Model
    sample = train_data[0]
    in_channels = sample.x.shape[1]
    num_classes = 2
    
    model = SimplePointNetSeg(in_channels=in_channels, num_classes=num_classes).to(device)
    print(f"Model: SimplePointNetSeg | In: {in_channels} | Params: {sum(p.numel() for p in model.parameters())}")
    
    # Loss & Optimizer
    # Class weighting
    all_labels = torch.cat([d.y for d in train_data])
    n_total = len(all_labels)
    n_pos = (all_labels == 1).sum().item()
    n_neg = n_total - n_pos
    alpha = n_neg / n_total # Inverse freq
    print(f"Class balance: Pos={n_pos} ({n_pos/n_total:.2%}), Neg={n_neg}")
    
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Logging
    run_name = f"pointnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'log.csv')
    
    with open(log_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_f1', 'val_loss', 'val_f1', 'val_auc'])
        
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:3d} | Train F1: {train_m['f1']:.4f} Loss: {train_m['loss']:.4f} | "
              f"Val F1: {val_m['f1']:.4f} AUC: {val_m['auc']:.4f} | {time.time()-t0:.1f}s")
              
        with open(log_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_m['loss'], train_m['f1'], val_m['loss'], val_m['f1'], val_m['auc']])
            
        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))
            
    print(f"Best Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
