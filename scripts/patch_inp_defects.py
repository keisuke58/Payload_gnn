#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_inp_defects.py — INPテンプレートに L1 欠陥パッチを適用

AdhesiveInner/AdhesiveOuter インスタンスの C3D8R 要素セットを分割し、
欠陥ゾーンの要素に剛性低下材料を割り当てる。

Usage:
    python scripts/patch_inp_defects.py \
        --template abaqus_work/Job-CZM-S12-Test.inp \
        --defect_json sample_defect.json \
        --output abaqus_work/Job-S12-D001.inp
"""

import argparse
import json
import math
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Defect type → target adhesive layer(s)
LAYER_TARGETS = {
    'debonding': ['Outer'],
    'fod': ['Outer'],
    'thermal_progression': ['Outer'],
    'acoustic_fatigue': ['Outer'],
    'inner_debond': ['Inner'],
    'impact': ['Inner', 'Outer'],
    'delamination': ['Inner', 'Outer'],
}

# Defect type → degraded isotropic material for C3D8R solid elements
# debonding/inner_debond: E/1000 (complete disbond)
# others: E/10 (partial damage)
DEFECT_MATERIAL = {
    'debonding': 'MAT-ADH-DEBONDED',
    'inner_debond': 'MAT-ADH-DEBONDED',
    'impact': 'MAT-ADH-PARTIAL',
    'delamination': 'MAT-ADH-PARTIAL',
    'fod': 'MAT-ADH-PARTIAL',
    'thermal_progression': 'MAT-ADH-PARTIAL',
    'acoustic_fatigue': 'MAT-ADH-PARTIAL',
}

# New isotropic material definitions (matching MAT-ADHESIVE-SOLID format)
# MAT-ADHESIVE-SOLID: E=20000, v=0.3, density=1200e-12, CTE=45e-6
MATERIAL_DEFS = {
    'MAT-ADH-DEBONDED': [
        '*Material, name=MAT-ADH-DEBONDED',
        '*Density',
        '1200e-12,',
        '*Elastic',
        ' 20., 0.3',
        '*Expansion, zero=20.',
        ' 45e-6,',
    ],
    'MAT-ADH-PARTIAL': [
        '*Material, name=MAT-ADH-PARTIAL',
        '*Density',
        '1200e-12,',
        '*Elastic',
        ' 2000., 0.3',
        '*Expansion, zero=20.',
        ' 45e-6,',
    ],
}

# Instance name patterns
INSTANCE_NAMES = {
    'Inner': 'Part-AdhesiveInner-1',
    'Outer': 'Part-AdhesiveOuter-1',
}


# ---------------------------------------------------------------------------
# Defect zone geometry (same formula as extract_odb_results.py:_is_node_in_defect)
# ---------------------------------------------------------------------------

def _is_in_defect_zone(x, y, z, defect_params):
    """Check if point (x,y,z) lies inside the circular defect zone.

    Abaqus Revolve coordinate system:
        Y = axial, XZ = radial plane
        r = sqrt(x^2 + z^2), theta = atan2(z, x)
    """
    theta_center = math.radians(defect_params['theta_deg'])
    z_center = defect_params['z_center']
    radius = defect_params['radius']

    r_node = math.sqrt(x * x + z * z)
    if r_node < 1.0:
        return False
    theta_node = math.atan2(z, x)
    arc_mm = r_node * abs(theta_node - theta_center)
    dy = y - z_center
    dist = math.sqrt(arc_mm * arc_mm + dy * dy)
    return dist <= radius


# ---------------------------------------------------------------------------
# INP parsing
# ---------------------------------------------------------------------------

def _parse_instance(lines, start):
    """Parse nodes and elements from an adhesive instance block.

    Args:
        lines: full INP file lines (list of str)
        start: line index of '*Instance, name=...' line

    Returns:
        nodes: {node_id: (x, y, z)}
        elements: {elem_id: (n1, n2, ..., n8)}
        elset_data_end: line index after Elset generate data (insert point)
        end_instance: line index of '*End Instance'
    """
    nodes = {}
    elements = {}
    elset_data_end = None
    end_instance = None

    mode = None  # 'node' or 'element'
    i = start + 1

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == '*End Instance':
            end_instance = i
            break

        # Mode switches on keyword lines
        if stripped == '*Node':
            mode = 'node'
            i += 1
            continue

        if stripped.startswith('*Element, type='):
            mode = 'element'
            i += 1
            continue

        if stripped.startswith('*'):
            mode = None

        if stripped.startswith('*Elset, elset=Set-All'):
            # Next line is the generate data, we insert after it
            elset_data_end = i + 2
            i += 2
            continue

        # Parse data lines
        if mode == 'node' and stripped and not stripped.startswith('*'):
            parts = stripped.split(',')
            if len(parts) >= 4:
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                nodes[nid] = (x, y, z)

        elif mode == 'element' and stripped and not stripped.startswith('*'):
            parts = stripped.split(',')
            if len(parts) >= 9:
                eid = int(parts[0])
                nids = tuple(int(p) for p in parts[1:9])
                elements[eid] = nids

        i += 1

    return nodes, elements, elset_data_end, end_instance


def _format_elset(name, elem_ids):
    """Format element IDs as an Abaqus *Elset block (16 per line)."""
    result = ['*Elset, elset=%s' % name]
    ids = sorted(elem_ids)
    for i in range(0, len(ids), 16):
        chunk = ids[i:i + 16]
        line = ', '.join(str(e) for e in chunk)
        if i + 16 < len(ids):
            line += ','
        result.append(' ' + line)
    return result


# ---------------------------------------------------------------------------
# Main patch logic
# ---------------------------------------------------------------------------

def patch_inp(template_path, defect_params, output_path):
    """Apply L1 defect patch to INP template.

    Args:
        template_path: path to healthy template INP
        defect_params: dict with theta_deg, z_center, radius, defect_type
        output_path: path for patched INP output

    Returns:
        dict with patch statistics
    """
    defect_type = defect_params['defect_type']
    targets = LAYER_TARGETS.get(defect_type)
    if targets is None:
        raise ValueError("Unknown defect_type: %s" % defect_type)

    mat_name = DEFECT_MATERIAL[defect_type]

    print("Patching INP: %s" % template_path)
    print("  Defect: %s at theta=%.1f deg, z=%.0f mm, r=%.0f mm" % (
        defect_type, defect_params['theta_deg'],
        defect_params['z_center'], defect_params['radius']))
    print("  Target layers: %s, material: %s" % (targets, mat_name))

    with open(template_path, 'r') as f:
        lines = f.readlines()

    # Find adhesive instance start lines
    instance_starts = {}
    for i, line in enumerate(lines):
        for layer, inst_name in INSTANCE_NAMES.items():
            if line.strip().startswith('*Instance, name=%s' % inst_name):
                instance_starts[layer] = i

    # Collect patches (layer, insert_point, end_point, new_content)
    stats = {}
    patches = []

    for layer in targets:
        if layer not in instance_starts:
            print("  WARNING: %s instance not found, skipping" % layer)
            continue

        nodes, elements, elset_data_end, end_instance = \
            _parse_instance(lines, instance_starts[layer])

        print("  %s: %d nodes, %d elements" % (layer, len(nodes), len(elements)))

        # Classify elements by centroid position
        defect_ids = []
        healthy_ids = []
        for eid, nids in elements.items():
            cx = sum(nodes[n][0] for n in nids) / 8.0
            cy = sum(nodes[n][1] for n in nids) / 8.0
            cz = sum(nodes[n][2] for n in nids) / 8.0
            if _is_in_defect_zone(cx, cy, cz, defect_params):
                defect_ids.append(eid)
            else:
                healthy_ids.append(eid)

        print("  %s: %d defect, %d healthy elements" % (
            layer, len(defect_ids), len(healthy_ids)))

        if not defect_ids:
            print("  WARNING: No elements in defect zone for %s" % layer)
            continue

        suffix = layer.upper()

        # Build new lines to replace the section block
        new_lines = []
        new_lines.extend(_format_elset('DEFECT-%s' % suffix, defect_ids))
        new_lines.extend(_format_elset('HEALTHY-%s' % suffix, healthy_ids))
        new_lines.append('** Section: Section-Adhesive-Healthy-%s' % suffix)
        new_lines.append(
            '*Solid Section, elset=HEALTHY-%s, material=MAT-ADHESIVE-SOLID' % suffix)
        new_lines.append(',')
        new_lines.append('** Section: Section-Adhesive-Defect-%s' % suffix)
        new_lines.append(
            '*Solid Section, elset=DEFECT-%s, material=%s' % (suffix, mat_name))
        new_lines.append(',')

        # Replace range: from after Elset data to before *End Instance
        patches.append((elset_data_end, end_instance, new_lines))
        stats[layer] = {'defect': len(defect_ids), 'healthy': len(healthy_ids)}

    if not patches:
        print("WARNING: No patches applied, copying template as-is")
        with open(output_path, 'w') as f:
            f.writelines(lines)
        return stats

    # Apply patches from end to start (preserve line indices)
    patches.sort(key=lambda p: p[0], reverse=True)
    for replace_start, replace_end, new_lines in patches:
        lines[replace_start:replace_end] = [l + '\n' for l in new_lines]

    # Add material definitions if not already present
    existing_mats = set()
    for line in lines:
        if line.strip().lower().startswith('*material, name='):
            name = line.strip().split('=', 1)[1].strip().upper()
            existing_mats.add(name)

    if mat_name.upper() not in existing_mats:
        # Find last material block end
        last_mat_idx = -1
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('*material, name='):
                last_mat_idx = i

        if last_mat_idx >= 0:
            # Scan forward past material sub-keywords and data lines
            mat_sub = ('*density', '*elastic', '*expansion', '*damage')
            j = last_mat_idx + 1
            while j < len(lines):
                s = lines[j].strip().lower()
                if not s or s.startswith('**'):
                    j += 1
                    continue
                if s.startswith('*') and not any(s.startswith(k) for k in mat_sub):
                    break
                j += 1

            # Insert new material block
            insert_lines = ['**\n']
            for ml in MATERIAL_DEFS[mat_name]:
                insert_lines.append(ml + '\n')
            for k, il in enumerate(insert_lines):
                lines.insert(j + k, il)

    with open(output_path, 'w') as f:
        f.writelines(lines)

    print("  Output: %s (%d lines)" % (output_path, len(lines)))
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Apply L1 defect patch to CZM sector12 INP template')
    parser.add_argument('--template', required=True,
                        help='Template INP file path')
    parser.add_argument('--defect_json', required=True,
                        help='JSON file with defect parameters')
    parser.add_argument('--output', required=True,
                        help='Output patched INP file path')
    args = parser.parse_args()

    with open(args.defect_json, 'r') as f:
        defect_params = json.load(f)

    # Support nested defect_params key (DOE output format)
    if 'defect_params' in defect_params and 'theta_deg' not in defect_params:
        defect_params = defect_params['defect_params']

    stats = patch_inp(args.template, defect_params, args.output)

    if not stats:
        print("No defect elements found — check defect zone parameters")
        sys.exit(1)

    for layer, s in stats.items():
        print("  %s: %d defect / %d healthy" % (layer, s['defect'], s['healthy']))


if __name__ == '__main__':
    main()
