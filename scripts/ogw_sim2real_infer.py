#!/usr/bin/env python3
"""Sim-to-real: run a FEM-GW-trained classifier on real OGW sensor graphs.

Loads a graph-level GW classifier checkpoint (trained on FEM gw sensor data) and
applies it to the OGW omega-stringer scenarios (Intact / local debond / large
debond) built with the SAME feature_set, reporting predicted class + probability.

Usage:
    python3 scripts/ogw_sim2real_infer.py --ckpt runs/sage_XXXX/best_model.pt \
        --csv_dir data/external/ogw_stringer/csv9
"""
from __future__ import annotations
import argparse, os, sys, glob
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from build_gw_graph import build_gw_graph        # noqa: E402
from train_gw import build_graph_level_model     # noqa: E402

FEAT = {3: "baseline", 10: "extended", 15: "full", 24: "comprehensive"}
SCEN = [("Intact", 0), ("FirstImpact", 1), ("SecondImpact", 1)]  # true labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv_dir", default="data/external/ogw_stringer/csv9")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    in_ch = ck.get("in_channels")
    arch = args.get("arch", "sage")
    feat = FEAT.get(in_ch, "comprehensive")
    print(f"checkpoint: {a.ckpt}\n  arch={arch} in_channels={in_ch} feature_set={feat} "
          f"val_f1={ck.get('val_f1')}")

    model = build_graph_level_model(arch, in_ch, edge_attr_dim=ck.get("edge_attr_dim", 0),
                                    num_classes=2)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    print("\n=== OGW sim-to-real ===")
    n_correct = 0
    for tag, ytrue in SCEN:
        cands = glob.glob(os.path.join(a.csv_dir, f"OGW_{tag}_*.csv"))
        if not cands:
            print(f"  {tag}: CSV not found"); continue
        g = build_gw_graph(cands[0], label=ytrue, feature_set=feat)
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            out = model(g)
            prob = torch.softmax(out, dim=-1).flatten()
            pred = int(prob.argmax())
        ok = "OK" if pred == ytrue else "X"
        n_correct += int(pred == ytrue)
        print(f"  {tag:13s} true={ytrue} pred={pred} p(defect)={prob[1]:.3f}  [{ok}]")
    print(f"\nsim-to-real accuracy: {n_correct}/{len(SCEN)}")


if __name__ == "__main__":
    main()
