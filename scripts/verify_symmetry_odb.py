#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify sector boundary symmetry from ODB results.

Compare Mises stress at theta~0 vs theta~60 deg for each instance.
Run: abaqus python scripts/verify_symmetry_odb.py <odb_path>

For a correct sector model with symmetry BCs, stress at theta=0
should be very close to stress at theta=60 at the same (r, y) position.
"""
import sys
import math
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL

SECTOR_ANGLE = 60.0
THETA_RAD = math.radians(SECTOR_ANGLE)

# Tolerance for boundary identification
THETA_TOL = 0.05  # radians (~3 degrees)


def get_nodal_stress(odb):
    """Extract Mises stress at each node (ELEMENT_NODAL -> averaged)."""
    step = odb.steps.values()[-1]
    frame = step.frames[-1]
    stress = frame.fieldOutputs['S']

    node_stress = {}  # {(inst_name, node_label): mises}
    for inst_name in odb.rootAssembly.instances.keys():
        inst = odb.rootAssembly.instances[inst_name]
        if len(inst.nodes) == 0:
            continue
        try:
            sub = stress.getSubset(region=inst, position=ELEMENT_NODAL)
        except Exception:
            continue
        if len(sub.values) == 0:
            continue

        # Average stress contributions at each node
        node_sum = {}
        node_cnt = {}
        for v in sub.values:
            nl = v.nodeLabel
            m = v.mises
            if nl in node_sum:
                node_sum[nl] += m
                node_cnt[nl] += 1
            else:
                node_sum[nl] = m
                node_cnt[nl] = 1

        for nl in node_sum:
            node_stress[(inst_name, nl)] = node_sum[nl] / node_cnt[nl]

    return node_stress


def classify_boundary_nodes(odb):
    """Classify nodes near theta=0 and theta=60 boundaries."""
    theta0_nodes = {}   # {(inst, label): (r, y, mises)}
    theta60_nodes = {}

    for inst_name in odb.rootAssembly.instances.keys():
        inst = odb.rootAssembly.instances[inst_name]
        for node in inst.nodes:
            x, y, z = node.coordinates
            r = math.sqrt(x**2 + z**2)
            if r < 10.0:
                continue
            theta = math.atan2(z, x)

            if abs(theta) < THETA_TOL:
                theta0_nodes[(inst_name, node.label)] = (r, y)
            elif abs(theta - THETA_RAD) < THETA_TOL:
                theta60_nodes[(inst_name, node.label)] = (r, y)

    return theta0_nodes, theta60_nodes


def find_matching_pairs(theta0_nodes, theta60_nodes, node_stress,
                        r_tol=5.0, y_tol=5.0):
    """Find matching node pairs and compare stress."""
    # Group theta0 by instance
    inst_theta0 = {}
    for (inst, nl), (r, y) in theta0_nodes.items():
        inst_theta0.setdefault(inst, []).append((r, y, nl))

    inst_theta60 = {}
    for (inst, nl), (r, y) in theta60_nodes.items():
        inst_theta60.setdefault(inst, []).append((r, y, nl))

    results = {}  # inst_name -> list of (stress0, stress60, r, y)
    for inst in sorted(set(inst_theta0.keys()) & set(inst_theta60.keys())):
        pairs = []
        nodes0 = inst_theta0[inst]
        nodes60 = inst_theta60[inst]

        for r0, y0, nl0 in nodes0:
            s0 = node_stress.get((inst, nl0))
            if s0 is None:
                continue
            # Find closest match at theta=60
            best_dist = 1e9
            best_s60 = None
            for r60, y60, nl60 in nodes60:
                dr = abs(r0 - r60)
                dy = abs(y0 - y60)
                if dr < r_tol and dy < y_tol:
                    dist = math.sqrt(dr**2 + dy**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_s60 = node_stress.get((inst, nl60))

            if best_s60 is not None:
                pairs.append((s0, best_s60, r0, y0))

        results[inst] = pairs

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: abaqus python verify_symmetry_odb.py <odb_path>")
        sys.exit(1)

    odb = openOdb(sys.argv[1], readOnly=True)

    print("=" * 60)
    print("Symmetry Verification: %s" % sys.argv[1])
    print("=" * 60)

    # Get stress
    node_stress = get_nodal_stress(odb)
    print("Total nodes with stress: %d" % len(node_stress))

    # Classify boundary nodes
    theta0_nodes, theta60_nodes = classify_boundary_nodes(odb)
    print("Nodes near theta=0: %d" % len(theta0_nodes))
    print("Nodes near theta=60: %d" % len(theta60_nodes))

    # Match and compare
    results = find_matching_pairs(theta0_nodes, theta60_nodes, node_stress)

    print("\n" + "-" * 60)
    print("Symmetry comparison (Mises stress):")
    print("-" * 60)

    for inst in sorted(results.keys()):
        pairs = results[inst]
        if not pairs:
            print("\n%s: no matching pairs found" % inst)
            continue

        # Compute relative error
        rel_errors = []
        for s0, s60, r, y in pairs:
            avg = (abs(s0) + abs(s60)) / 2.0
            if avg > 0.001:
                rel_errors.append(abs(s0 - s60) / avg)

        n = len(pairs)
        n_err = len(rel_errors)
        if n_err == 0:
            print("\n%s: %d pairs, all near zero stress" % (inst, n))
            continue

        mean_err = sum(rel_errors) / n_err
        max_err = max(rel_errors)
        pct_good = sum(1 for e in rel_errors if e < 0.05) * 100.0 / n_err

        print("\n%s:" % inst)
        print("  Matched pairs: %d" % n)
        print("  Mean relative error: %.4f (%.1f%%)" % (mean_err, mean_err * 100))
        print("  Max  relative error: %.4f (%.1f%%)" % (max_err, max_err * 100))
        print("  Pairs with <5%% error: %.1f%%" % pct_good)

        # Sample pairs (sorted by y position)
        pairs_sorted = sorted(pairs, key=lambda p: p[3])
        print("  Sample pairs (r, y, stress_0, stress_60, rel_err):")
        step = max(1, len(pairs_sorted) // 8)
        for i in range(0, len(pairs_sorted), step):
            s0, s60, r, y = pairs_sorted[i]
            avg = (abs(s0) + abs(s60)) / 2.0
            err = abs(s0 - s60) / avg if avg > 0.001 else 0.0
            print("    r=%7.1f y=%7.1f  S0=%8.3f  S60=%8.3f  err=%.1f%%" %
                  (r, y, s0, s60, err * 100))

    odb.close()
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
