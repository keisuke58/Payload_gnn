#!/usr/bin/env python3
"""
Extract CompDam_DGD damage results from ODB and compute effective stiffness reduction.

Run with:  abaqus python scripts/extract_compdam_results.py

Reads: abaqus_work/compdam_flatplate/compdam_flatplate_impact.odb
Writes: abaqus_work/compdam_flatplate/compdam_damage_results.csv
        abaqus_work/compdam_flatplate/compdam_stiffness_reduction.json
"""

import sys
import os
import json
import csv

# Abaqus Python (Python 3 in Abaqus 2024)
from odbAccess import openOdb
import numpy as np


def extract_damage(odb_path):
    """Extract CompDam SDV damage variables from ODB."""

    odb = openOdb(odb_path, readOnly=True)

    # Get last frame
    step = odb.steps['Impact']
    last_frame = step.frames[-1]

    # SDV field names — Abaqus uses named labels from *Depvar
    sdv_keys = {
        'CDM_d2': 'SDV_CDM_D2',
        'CDM_FIm': 'SDV_CDM_FIM',
        'CDM_STATUS': 'SDV_CDM_STATUS',
        'CDM_FIfT': 'SDV_CDM_FIFT',
        'CDM_FIfC': 'SDV_CDM_FIFC',
        'CDM_d1T': 'SDV_CDM_D1T',
        'CDM_d1C': 'SDV_CDM_D1C',
    }

    # List available field outputs
    available = list(last_frame.fieldOutputs.keys())
    print("  Available fields: {}".format(', '.join(available[:20])))

    # Collect SDVs per element
    results = {}
    for sdv_name, sdv_label in sdv_keys.items():
        if sdv_label not in last_frame.fieldOutputs:
            print("  WARNING: {} not found".format(sdv_label))
            continue
        field = last_frame.fieldOutputs[sdv_label]
        for val in field.values:
            inst = val.instance.name if val.instance else ''
            key = (inst, val.elementLabel)
            if key not in results:
                results[key] = {'instance': inst, 'elem': val.elementLabel}
            try:
                results[key][sdv_name] = float(val.data)
            except (TypeError, AttributeError):
                results[key][sdv_name] = val.data

    # Extract stress S11 for stiffness estimation
    if 'S' in last_frame.fieldOutputs:
        stress_field = last_frame.fieldOutputs['S']
        for val in stress_field.values:
            inst = val.instance.name if val.instance else ''
            key = (inst, val.elementLabel)
            if key in results:
                results[key]['S11'] = float(val.data[0])
                results[key]['S22'] = float(val.data[1])
                results[key]['S33'] = float(val.data[2])
                results[key]['S12'] = float(val.data[3])

    odb.close()
    return list(results.values())


def compute_stiffness_reduction(results):
    """
    Compute effective stiffness reduction from CompDam damage variables.

    CompDam damage variables:
    - CDM_d2: matrix damage (0=undamaged, 1=fully damaged)
    - CDM_d1T: fiber tension damage (0-1)
    - CDM_d1C: fiber compression damage (0-1)
    - CDM_STATUS: element status (1=active, 0=deleted)

    Effective stiffness reduction:
    - E1_eff = E1 * (1 - d1T) * (1 - d1C)
    - E2_eff = E2 * (1 - d2)
    - G12_eff = G12 * (1 - d2) * (1 - d1T) (approximate)
    """

    # Classify elements by section/ply
    ply_stats = {}
    deleted_count = 0
    total_cfrp = 0

    for r in results:
        elem = r['elem']
        d2 = r.get('CDM_d2', 0.0)
        d1T = r.get('CDM_d1T', 0.0)
        d1C = r.get('CDM_d1C', 0.0)
        status = r.get('CDM_STATUS', 1)
        fi_m = r.get('CDM_FIm', 0.0)

        total_cfrp += 1
        if status == 0:
            deleted_count += 1

        # Determine ply from element number
        # Elements are numbered: layer_idx * 2500 + local_eid (1-based)
        # 2500 = 50 * 50 elements per layer
        elems_per_layer = 2500
        layer_idx = (elem - 1) // elems_per_layer

        if layer_idx not in ply_stats:
            ply_stats[layer_idx] = {
                'n_elements': 0,
                'n_damaged_matrix': 0,
                'n_damaged_fiber_T': 0,
                'n_damaged_fiber_C': 0,
                'n_deleted': 0,
                'd2_max': 0.0,
                'd1T_max': 0.0,
                'd1C_max': 0.0,
                'd2_sum': 0.0,
                'd1T_sum': 0.0,
                'd1C_sum': 0.0,
            }

        stats = ply_stats[layer_idx]
        stats['n_elements'] += 1
        stats['d2_sum'] += d2
        stats['d1T_sum'] += d1T
        stats['d1C_sum'] += d1C
        stats['d2_max'] = max(stats['d2_max'], d2)
        stats['d1T_max'] = max(stats['d1T_max'], d1T)
        stats['d1C_max'] = max(stats['d1C_max'], d1C)
        if d2 > 0.01:
            stats['n_damaged_matrix'] += 1
        if d1T > 0.01:
            stats['n_damaged_fiber_T'] += 1
        if d1C > 0.01:
            stats['n_damaged_fiber_C'] += 1
        if status == 0:
            stats['n_deleted'] += 1

    # Compute averages and effective stiffness reduction
    E1_ref = 171420.0  # IM7-8552
    E2_ref = 9080.0
    G12_ref = 5290.0

    summary = {
        'total_cfrp_elements': total_cfrp,
        'deleted_elements': deleted_count,
        'plies': {},
    }

    for layer_idx in sorted(ply_stats.keys()):
        stats = ply_stats[layer_idx]
        n = stats['n_elements']
        d2_avg = stats['d2_sum'] / n if n > 0 else 0
        d1T_avg = stats['d1T_sum'] / n if n > 0 else 0
        d1C_avg = stats['d1C_sum'] / n if n > 0 else 0

        # Effective stiffness ratios
        E1_ratio = (1.0 - d1T_avg) * (1.0 - d1C_avg)
        E2_ratio = 1.0 - d2_avg
        G12_ratio = (1.0 - d2_avg) * max(1.0 - d1T_avg, 1.0 - d1C_avg)

        # Ply classification
        if layer_idx < 8:
            ply_name = "InnerPly{}".format(layer_idx + 1)
        elif layer_idx < 13:
            ply_name = "Core{}".format(layer_idx - 7)
            continue  # skip core
        else:
            ply_name = "OuterPly{}".format(layer_idx - 12)

        summary['plies'][ply_name] = {
            'layer_idx': layer_idx,
            'n_elements': n,
            'd2_avg': round(d2_avg, 6),
            'd2_max': round(stats['d2_max'], 6),
            'd1T_avg': round(d1T_avg, 6),
            'd1T_max': round(stats['d1T_max'], 6),
            'd1C_avg': round(d1C_avg, 6),
            'd1C_max': round(stats['d1C_max'], 6),
            'n_damaged_matrix': stats['n_damaged_matrix'],
            'n_damaged_fiber_T': stats['n_damaged_fiber_T'],
            'n_damaged_fiber_C': stats['n_damaged_fiber_C'],
            'n_deleted': stats['n_deleted'],
            'E1_eff_ratio': round(E1_ratio, 4),
            'E2_eff_ratio': round(E2_ratio, 4),
            'G12_eff_ratio': round(G12_ratio, 4),
        }

    # Overall stiffness reduction (weighted average across all plies)
    all_d2 = []
    all_d1T = []
    all_d1C = []
    for r in results:
        all_d2.append(r.get('CDM_d2', 0.0))
        all_d1T.append(r.get('CDM_d1T', 0.0))
        all_d1C.append(r.get('CDM_d1C', 0.0))

    if all_d2:
        d2_global = sum(all_d2) / len(all_d2)
        d1T_global = sum(all_d1T) / len(all_d1T)
        d1C_global = sum(all_d1C) / len(all_d1C)
    else:
        d2_global = d1T_global = d1C_global = 0.0

    summary['global'] = {
        'd2_avg': round(d2_global, 6),
        'd1T_avg': round(d1T_global, 6),
        'd1C_avg': round(d1C_global, 6),
        'E1_eff_ratio': round((1 - d1T_global) * (1 - d1C_global), 4),
        'E2_eff_ratio': round(1 - d2_global, 4),
        'G12_eff_ratio': round((1 - d2_global) * max(1 - d1T_global, 1 - d1C_global), 4),
    }

    # Comparison with project empirical factors
    summary['comparison'] = {
        'project_impact_E1_factor': 0.7,
        'project_impact_E2_factor': 0.3,
        'project_delamination_E1_factor': 0.9,
        'compdam_E1_factor': summary['global']['E1_eff_ratio'],
        'compdam_E2_factor': summary['global']['E2_eff_ratio'],
    }

    return summary


def main():
    odb_dir = 'abaqus_work/compdam_flatplate'
    odb_path = os.path.join(odb_dir, 'compdam_flatplate_impact.odb')

    if not os.path.exists(odb_path):
        print("ERROR: ODB not found: {}".format(odb_path))
        sys.exit(1)

    print("Extracting CompDam results from: {}".format(odb_path))

    # Extract damage
    results = extract_damage(odb_path)
    print("  Extracted {} element records".format(len(results)))

    # Write CSV
    csv_path = os.path.join(odb_dir, 'compdam_damage_results.csv')
    if results:
        keys = ['instance', 'elem', 'CDM_d2', 'CDM_FIm', 'CDM_STATUS',
                'CDM_FIfT', 'CDM_FIfC', 'CDM_d1T', 'CDM_d1C',
                'S11', 'S22', 'S33', 'S12']
        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print("  CSV written: {}".format(csv_path))

    # Compute stiffness reduction
    summary = compute_stiffness_reduction(results)

    # Write JSON
    json_path = os.path.join(odb_dir, 'compdam_stiffness_reduction.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print("  JSON written: {}".format(json_path))

    # Print summary
    print("\n" + "=" * 60)
    print("CompDam Damage Summary")
    print("=" * 60)
    print("Total CFRP elements: {}".format(summary['total_cfrp_elements']))
    print("Deleted elements:    {}".format(summary['deleted_elements']))

    print("\nPer-ply damage:")
    for name, data in sorted(summary['plies'].items()):
        print("  {}: d2_max={:.4f}, d1T_max={:.4f}, d1C_max={:.4f}, "
              "damaged_matrix={}, damaged_fiber_T={}"
              .format(name, data['d2_max'], data['d1T_max'], data['d1C_max'],
                      data['n_damaged_matrix'], data['n_damaged_fiber_T']))

    print("\nGlobal stiffness reduction:")
    g = summary['global']
    print("  E1_eff/E1 = {:.4f}".format(g['E1_eff_ratio']))
    print("  E2_eff/E2 = {:.4f}".format(g['E2_eff_ratio']))
    print("  G12_eff/G12 = {:.4f}".format(g['G12_eff_ratio']))

    print("\nComparison with project empirical factors:")
    c = summary['comparison']
    print("  E1: CompDam={:.4f} vs Project(impact)={:.2f}".format(
        c['compdam_E1_factor'], c['project_impact_E1_factor']))
    print("  E2: CompDam={:.4f} vs Project(impact)={:.2f}".format(
        c['compdam_E2_factor'], c['project_impact_E2_factor']))


if __name__ == '__main__':
    main()
