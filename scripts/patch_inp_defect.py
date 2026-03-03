#!/usr/bin/env python3
"""Patch INP file to apply defect section assignments based on element centroids.

Usage: python3 patch_inp_defect.py <inp_path> <defect_json_path>

Modifies the INP in-place:
1. Reads node coordinates and element connectivity from *Instance blocks
2. Identifies elements in the defect zone by centroid
3. Creates new element sets and section overrides (injected before *End Instance)
4. Adds damaged solid-adhesive material if needed (before ** BOUNDARY CONDITIONS)

Works with dependent=OFF models where mesh data is in *Instance, not *Part.
"""
import sys
import json
import math
import re
import shutil
from collections import defaultdict

# Ply sequence matching Section-CFRP-Skin in generate_ground_truth.py
PLY_ANGLES = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
PLY_THICKNESS = 0.125  # mm per ply (1.0mm / 8 plies)


def parse_defect_json(path):
    with open(path) as f:
        return json.load(f)


def point_in_defect_zone(x, y, z, defect_params):
    """Check if point (INP coords: y=axial) is in circular defect zone.

    defect_params['z_center'] = axial position (maps to INP y)
    defect_params['theta_deg'] = angular position in x-z plane
    defect_params['radius'] = defect radius on surface (mm)
    """
    z_c = defect_params['z_center']
    theta_deg = defect_params['theta_deg']
    r_def = defect_params['radius']

    # Quick axial check
    dy = y - z_c
    if abs(dy) > r_def * 1.5:
        return False

    # Radial distance from axis
    r_local = math.sqrt(x * x + z * z)
    if r_local < 1.0:
        return False

    # Angular distance on surface
    theta_pt = math.atan2(z, x)
    theta_center = math.radians(theta_deg)
    arc = r_local * abs(theta_pt - theta_center)

    dist = math.sqrt(arc * arc + dy * dy)
    return dist <= r_def * 1.01


def parse_inp_instances(inp_path):
    """Parse INP to extract nodes and elements per *Instance block."""
    with open(inp_path) as f:
        lines = f.readlines()

    nodes = {}       # (instance_name, node_id) -> (x, y, z)
    elements = {}    # (instance_name, elem_id) -> [node_ids]
    current_inst = None
    current_block = None  # 'node' or 'element'
    elem_type = None

    for line in lines:
        stripped = line.strip()

        # Track instance boundaries
        m = re.match(r'\*Instance,\s*name=(\S+)', stripped, re.I)
        if m:
            current_inst = m.group(1).rstrip(',')
            current_block = None
            continue

        if stripped.upper().startswith('*END INSTANCE'):
            current_inst = None
            current_block = None
            continue

        if current_inst is None:
            continue

        # Detect *Node block
        if stripped.upper().startswith('*NODE') and \
           not stripped.upper().startswith('*NODE OUTPUT') and \
           not stripped.upper().startswith('*NODE PRINT'):
            current_block = 'node'
            continue

        # Detect *Element block
        m = re.match(r'\*Element,\s*type=(\S+)', stripped, re.I)
        if m:
            current_block = 'element'
            elem_type = m.group(1)
            continue

        # Any other keyword ends the current data block
        if stripped.startswith('*'):
            current_block = None
            continue

        # Parse node data
        if current_block == 'node':
            parts = stripped.split(',')
            if len(parts) >= 4:
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                nodes[(current_inst, nid)] = (x, y, z)

        # Parse element data
        elif current_block == 'element':
            parts = [p.strip() for p in stripped.split(',') if p.strip()]
            if parts:
                eid = int(parts[0])
                nids = [int(p) for p in parts[1:]]
                elements[(current_inst, eid)] = nids

    return lines, nodes, elements


def find_defect_elements(nodes, elements, defect_params):
    """Identify elements whose centroid falls within the defect zone."""
    defect_elems = defaultdict(list)  # instance_name -> [elem_ids]

    for (inst, eid), nids in elements.items():
        coords = [nodes.get((inst, n)) for n in nids]
        coords = [c for c in coords if c is not None]
        if not coords:
            continue
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        if point_in_defect_zone(cx, cy, cz, defect_params):
            defect_elems[inst].append(eid)

    return defect_elems


def _elset_lines(set_name, eids):
    """Generate *Elset data lines (16 IDs per line)."""
    lines = ['*Elset, elset=%s\n' % set_name]
    sorted_eids = sorted(eids)
    for j in range(0, len(sorted_eids), 16):
        batch = sorted_eids[j:j + 16]
        lines.append(', '.join(str(e) for e in batch) + '\n')
    return lines


def _debonded_skin_section(set_name):
    """Generate *Shell Section for debonded outer skin (CFRP_DEBONDED)."""
    lines = ['** Defect override: debonded outer skin\n']
    lines.append('*Shell Section, elset=%s, composite\n' % set_name)
    for ang in PLY_ANGLES:
        lines.append('%.3f, 3, CFRP_DEBONDED, %.1f\n' % (PLY_THICKNESS, ang))
    return lines


def _damaged_adhesive_section(set_name):
    """Generate *Solid Section for damaged solid adhesive."""
    lines = ['** Defect override: damaged outer adhesive\n']
    lines.append(
        '*Solid Section, elset=%s, material=MAT-ADHESIVE-SOLID-DAMAGED\n'
        % set_name)
    lines.append(',\n')
    return lines


# Material definition for damaged solid adhesive (E/1000 of healthy)
MAT_ADHESIVE_SOLID_DAMAGED = """\
*Material, name=MAT-ADHESIVE-SOLID-DAMAGED
*Density
1200e-12,
*Elastic
20., 0.3
*Expansion, zero=20.
45e-6,
"""


def main():
    if len(sys.argv) < 3:
        print("Usage: %s <inp_path> <defect_json>" % sys.argv[0])
        sys.exit(1)

    inp_path = sys.argv[1]
    defect_json = sys.argv[2]
    dp = parse_defect_json(defect_json)
    defect_type = dp.get('defect_type', 'debonding')

    print("=" * 60)
    print("Patching INP for defect: %s" % defect_type)
    print("  theta=%.1f deg, z_center=%.0f mm, radius=%.0f mm" % (
        dp['theta_deg'], dp['z_center'], dp['radius']))
    print("=" * 60)

    # Backup original
    backup_path = inp_path + '.bak'
    shutil.copy2(inp_path, backup_path)
    print("Backup: %s" % backup_path)

    lines, nodes, elements = parse_inp_instances(inp_path)
    print("Parsed %d nodes, %d elements across instances" % (
        len(nodes), len(elements)))

    defect_elems = find_defect_elements(nodes, elements, dp)

    total = 0
    for inst in sorted(defect_elems.keys()):
        eids = defect_elems[inst]
        print("  %-30s: %d elements in defect zone" % (inst, len(eids)))
        total += len(eids)

    if total == 0:
        print("WARNING: No elements found in defect zone! INP not modified.")
        sys.exit(0)

    # Build patched INP
    output_lines = []
    current_inst = None
    need_damaged_adhesive_mat = False

    for line in lines:
        stripped = line.strip()

        # Track current instance
        m = re.match(r'\*Instance,\s*name=(\S+)', stripped, re.I)
        if m:
            current_inst = m.group(1).rstrip(',')

        # Before *End Instance, inject defect elset + section override
        # Only inject for instances where we have a meaningful section override
        if stripped.upper().startswith('*END INSTANCE') and current_inst:
            eids = defect_elems.get(current_inst, [])
            if eids:
                inst_upper = current_inst.upper()
                set_name = 'DEFECT-ZONE'

                # Section override based on instance type and defect type
                if defect_type == 'debonding':
                    if 'OUTERSKIN' in inst_upper:
                        output_lines.extend(_elset_lines(set_name, eids))
                        output_lines.extend(
                            _debonded_skin_section(set_name))
                        print("  -> Injected debonded shell section for %s "
                              "(%d elems)" % (current_inst, len(eids)))

                    elif 'ADHESIVEOUTER' in inst_upper or \
                         ('ADHESIVE' in inst_upper and
                          'OUTER' in inst_upper):
                        output_lines.extend(_elset_lines(set_name, eids))
                        output_lines.extend(
                            _damaged_adhesive_section(set_name))
                        need_damaged_adhesive_mat = True
                        print("  -> Injected damaged solid section for %s "
                              "(%d elems)" % (current_inst, len(eids)))

            current_inst = None

        # Before ** BOUNDARY CONDITIONS, inject damaged material if needed
        if need_damaged_adhesive_mat and \
           stripped.startswith('** BOUNDARY CONDITIONS'):
            output_lines.append(MAT_ADHESIVE_SOLID_DAMAGED)
            need_damaged_adhesive_mat = False
            print("  -> Injected MAT-ADHESIVE-SOLID-DAMAGED material")

        output_lines.append(line)

    # If material wasn't injected (no BOUNDARY CONDITIONS found), append
    if need_damaged_adhesive_mat:
        # Find last *Material block and insert after
        output_lines.append(MAT_ADHESIVE_SOLID_DAMAGED)
        print("  -> Injected MAT-ADHESIVE-SOLID-DAMAGED material (at end)")

    # Write patched INP
    with open(inp_path, 'w') as f:
        f.writelines(output_lines)

    print("=" * 60)
    print("INP patched successfully: %s" % inp_path)
    print("  Total defect elements: %d" % total)
    print("=" * 60)


if __name__ == '__main__':
    main()
