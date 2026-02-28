#!/usr/bin/env python3
"""
Compare mesh convergence: 50mm, 25mm, 12mm (same defect: theta=30, z=2500, r=50).
10mm failed with numerical singularity.
"""
import os
import pandas as pd

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same defect for all
samples = [
    ("50mm", "dataset_output_ideal_50mm/sample_0000"),  # need to run
    ("25mm", "dataset_output_ideal/sample_0000"),
    ("12mm", "dataset_output_ideal_12mm/sample_0000"),
]

def main():
    print("Mesh Convergence Check (defect: theta=30, z=2500, r=50)")
    print("-" * 60)
    rows = []
    for name, rel_path in samples:
        path = os.path.join(PROJECT, rel_path, "nodes.csv")
        if not os.path.exists(path):
            print(f"  {name}: (not found)")
            continue
        df = pd.read_csv(path)
        n_nodes = len(df)
        n_defect = (df["defect_label"] == 1).sum() if "defect_label" in df.columns else 0
        smises_max = df["smises"].max() if "smises" in df.columns else float("nan")
        rows.append((name, n_nodes, n_defect, smises_max))
        sm_str = f", smises_max={smises_max:.1f}" if pd.notna(smises_max) else ""
        print(f"  {name}: {n_nodes:,} nodes, {n_defect} defect nodes{sm_str}")

    if rows:
        print("-" * 60)
        print("Convergence: finer mesh → more nodes, more defect resolution")
        h_vals = {"50mm": 50, "25mm": 25, "12mm": 12}
        for name, n_nodes, n_defect, _ in rows:
            h = h_vals.get(name, 0)
            if h:
                print(f"  h={h}mm: N={n_nodes:,}, n_defect={n_defect}")
        print("")
        print("10mm: numerical singularity (solver failed)")

if __name__ == "__main__":
    main()
