# -*- coding: utf-8 -*-
"""
Prediction API for Fairing Defect Localization

Provides a simple REST API (FastAPI) and a Python-callable interface
for running inference on new FEM results.

REST API Usage:
    uvicorn src.predict_api:app --host 0.0.0.0 --port 8000

    POST /predict
    Body: { "nodes_csv": "path/to/nodes.csv", "elements_csv": "path/to/elements.csv" }
    Response: { "defect_nodes": [...], "centroid": [x,y,z], "confidence": 0.95, ... }

Python Usage:
    from predict_api import FairingPredictor
    predictor = FairingPredictor("runs/gat_xxx/best_model.pt")
    result = predictor.predict("sample_dir/")
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from preprocess_fairing_data import process_single_sample, load_baseline_dspss


class FairingPredictor:
    """Inference wrapper for defect localization."""

    def __init__(self, checkpoint_path, baseline_dir=None, device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.args = argparse.Namespace(**ckpt['args'])

        # Load normalization stats
        data_dir = self.args.data_dir
        stats_path = os.path.join(data_dir, 'norm_stats.pt')
        if os.path.exists(stats_path):
            self.norm_stats = torch.load(stats_path, map_location=self.device,
                                         weights_only=False)
        else:
            self.norm_stats = None

        # Load baseline DSPSS
        self.baseline_dspss = None
        if baseline_dir:
            self.baseline_dspss = load_baseline_dspss(baseline_dir)

        # Build model
        # We need to know in_channels; read from checkpoint args or default
        in_channels = ckpt.get('in_channels', 10)
        edge_attr_dim = ckpt.get('edge_attr_dim', 4)

        self.model = build_model(
            self.args.arch, in_channels, edge_attr_dim,
            hidden_channels=self.args.hidden, num_layers=self.args.layers,
            dropout=0.0, num_classes=2,
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    @torch.no_grad()
    def predict(self, sample_dir, threshold=0.5):
        """
        Run inference on a single sample directory.

        Returns dict with:
            - defect_nodes: list of node indices predicted as defect
            - defect_probs: per-node defect probability
            - centroid: [x, y, z] of predicted defect centroid
            - confidence: mean probability of predicted defect nodes
            - n_defect_nodes: count
        """
        data = process_single_sample(
            sample_dir, self.baseline_dspss,
            mesh_size=getattr(self.args, 'mesh_size', 50.0),
            height=getattr(self.args, 'height', 5000.0),
        )
        if data is None:
            return {'error': 'Failed to process sample directory'}

        # Normalize
        if self.norm_stats:
            data.x = (data.x - self.norm_stats['mean'].to(self.device)) / \
                     self.norm_stats['std'].to(self.device)

        data = data.to(self.device)
        logits = self.model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

        defect_mask = probs >= threshold
        defect_indices = np.where(defect_mask)[0].tolist()
        pos_np = data.pos.cpu().numpy()

        if len(defect_indices) > 0:
            centroid = pos_np[defect_mask].mean(axis=0).tolist()
            confidence = float(probs[defect_mask].mean())
        else:
            centroid = [0.0, 0.0, 0.0]
            confidence = 0.0

        return {
            'defect_nodes': defect_indices,
            'defect_probs': probs.tolist(),
            'centroid': centroid,
            'confidence': confidence,
            'n_defect_nodes': len(defect_indices),
            'n_total_nodes': len(probs),
            'threshold': threshold,
        }


# =========================================================================
# FastAPI REST endpoint (optional, only if fastapi is installed)
# =========================================================================
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Fairing Defect Localization API")

    _predictor = None

    class PredictRequest(BaseModel):
        sample_dir: str
        threshold: float = 0.5

    class PredictResponse(BaseModel):
        defect_nodes: list
        centroid: list
        confidence: float
        n_defect_nodes: int
        n_total_nodes: int

    @app.on_event("startup")
    def startup():
        global _predictor
        ckpt = os.environ.get('MODEL_CHECKPOINT', 'runs/best_model.pt')
        baseline = os.environ.get('BASELINE_DIR', 'dataset_output/healthy_baseline')
        _predictor = FairingPredictor(ckpt, baseline_dir=baseline)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": _predictor.args.arch if _predictor else "not loaded"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        result = _predictor.predict(req.sample_dir, req.threshold)
        return result

except ImportError:
    app = None


# =========================================================================
# CLI
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run defect prediction')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sample_dir', type=str, required=True)
    parser.add_argument('--baseline_dir', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    predictor = FairingPredictor(args.checkpoint, baseline_dir=args.baseline_dir)
    result = predictor.predict(args.sample_dir, args.threshold)

    print(json.dumps(result, indent=2, default=str))
