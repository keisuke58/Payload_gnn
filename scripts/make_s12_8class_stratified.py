#!/usr/bin/env python3
"""
Stratified resplit of processed_s12_czm_96 → processed_s12_czm_96_8class

Each graph has exactly one defect class (the unique non-zero label in g.y).
We stratify on that class to guarantee all 7 defect types appear in val.

Usage:
  python scripts/make_s12_8class_stratified.py
"""
import os, sys, math, collections
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(PROJECT_ROOT, "data", "processed_s12_czm_96")
OUT_DIR  = os.path.join(PROJECT_ROOT, "data", "processed_s12_czm_96_8class")
VAL_RATIO = 0.2
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ── load all graphs (train + val from old split) ──────────────────────────────
print("Loading existing graphs from", SRC_DIR)
old_train = torch.load(os.path.join(SRC_DIR, "train.pt"), weights_only=False)
old_val   = torch.load(os.path.join(SRC_DIR, "val.pt"),   weights_only=False)
all_graphs = old_train + old_val
print(f"  Total: {len(all_graphs)} graphs")

# ── determine defect class per graph ─────────────────────────────────────────
def graph_defect_class(g):
    """Return the unique non-zero label (defect type). 0 if healthy-only."""
    nonzero = g.y[g.y > 0].unique()
    if len(nonzero) == 0:
        return 0
    return int(nonzero[0].item())

classes = [graph_defect_class(g) for g in all_graphs]
class_counter = collections.Counter(classes)
print("Defect class distribution:", dict(sorted(class_counter.items())))

# ── stratified split ──────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
train_idx, val_idx = [], []

for cls, count in sorted(class_counter.items()):
    idx_cls = [i for i, c in enumerate(classes) if c == cls]
    rng.shuffle(idx_cls)
    n_val = max(1, math.ceil(count * VAL_RATIO))   # at least 1 per class in val
    val_idx.extend(idx_cls[:n_val])
    train_idx.extend(idx_cls[n_val:])

train_data = [all_graphs[i] for i in train_idx]
val_data   = [all_graphs[i] for i in val_idx]

# ── norm stats from train only ────────────────────────────────────────────────
x_cat = torch.cat([g.x for g in train_data], dim=0)
mean = x_cat.mean(dim=0)
std  = x_cat.std(dim=0)
std[std < 1e-8] = 1.0
norm_stats = {"mean": mean, "std": std}

# ── save ─────────────────────────────────────────────────────────────────────
torch.save(train_data, os.path.join(OUT_DIR, "train.pt"))
torch.save(val_data,   os.path.join(OUT_DIR, "val.pt"))
torch.save(norm_stats, os.path.join(OUT_DIR, "norm_stats.pt"))

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\nSaved to {OUT_DIR}")
print(f"  Train: {len(train_data)}  Val: {len(val_data)}")

val_classes = collections.Counter(graph_defect_class(g) for g in val_data)
trn_classes = collections.Counter(graph_defect_class(g) for g in train_data)
print("\nPer-class split:")
all_cls = sorted(class_counter.keys())
print(f"  {'class':>6}  {'total':>6}  {'train':>6}  {'val':>4}")
for c in all_cls:
    print(f"  {c:6d}  {class_counter[c]:6d}  {trn_classes[c]:6d}  {val_classes.get(c,0):4d}")

# Verify all defect classes (1-7) are in val
val_defect = {c for c in val_classes if c > 0}
all_defect = {c for c in class_counter if c > 0}
missing = all_defect - val_defect
if missing:
    print(f"\nWARNING: classes {missing} still missing from val!")
else:
    print(f"\nAll {len(all_defect)} defect classes represented in val.")

all_labels = torch.cat([g.y for g in all_graphs])
num_cls = int(all_labels.max()) + 1
defect_pct = 100 * (all_labels > 0).float().mean().item()
print(f"num_classes={num_cls}  defect%={defect_pct:.3f}")
