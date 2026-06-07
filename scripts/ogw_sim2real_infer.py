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

    import json
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck

    # config: prefer sibling results.json, else the dict checkpoint, else infer
    cfg = {}
    rj = os.path.join(os.path.dirname(a.ckpt), "results.json")
    if os.path.exists(rj):
        cfg = json.load(open(rj)).get("args", {})
    elif isinstance(ck, dict):
        cfg = ck.get("args", {})
    # in_channels & num_layers inferred from the state_dict (robust)
    conv_w = [v for k, v in sd.items() if k.endswith("convs.0.lin_l.weight")]
    in_ch = conv_w[0].shape[1] if conv_w else ck.get("in_channels", 24)
    n_layers = cfg.get("num_layers") or len({k.split(".")[2] for k in sd if ".convs." in k})
    hidden = cfg.get("hidden", 128)
    arch = cfg.get("arch", "sage")
    feat = FEAT.get(in_ch, "comprehensive")
    print(f"checkpoint: {a.ckpt}\n  arch={arch} in={in_ch} hidden={hidden} layers={n_layers} "
          f"feature_set={feat} best_val_f1={json.load(open(rj)).get('best_val_f1') if os.path.exists(rj) else '?'}")

    model = build_graph_level_model(arch, in_ch, edge_attr_dim=0, hidden=hidden,
                                    num_classes=2, num_layers=n_layers)
    model.load_state_dict(sd)
    model.eval()

    print("\n=== OGW sim-to-real ===")
    n_correct = 0
    for tag, ytrue in SCEN:
        cands = glob.glob(os.path.join(a.csv_dir, f"OGW_{tag}_*.csv"))
        if not cands:
            print(f"  {tag}: CSV not found"); continue
        g = build_gw_graph(cands[0], label=ytrue, feature_set=feat)
        batch = torch.zeros(g.x.size(0), dtype=torch.long)
        ea = getattr(g, "edge_attr", None)
        with torch.no_grad():
            out = model(g.x, g.edge_index, ea, batch)
            prob = torch.softmax(out, dim=-1).flatten()
            pred = int(prob.argmax())
        ok = "OK" if pred == ytrue else "X"
        n_correct += int(pred == ytrue)
        print(f"  {tag:13s} true={ytrue} pred={pred} p(defect)={prob[1]:.3f}  [{ok}]")
    print(f"\nsim-to-real accuracy: {n_correct}/{len(SCEN)}")


if __name__ == "__main__":
    main()
