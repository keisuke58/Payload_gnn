#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch Abaqus INP file:
  1. Add thermal load and BC if missing
  2. Add LE (strain) to field output
  3. Apply z-dependent temperature profile (aerodynamic heating)

Run: python scripts/patch_inp_thermal.py Job-Realistic.inp
"""
import sys
import re

TEMP_INITIAL = 20.0

# z-dependent aerodynamic heating profile for outer skin
# T_outer(z) models Max-Q heating: base ~100C, barrel ~120C, ogive ~130-220C
TOTAL_HEIGHT = 10400.0  # mm
BARREL_H = 5000.0

# Inner skin / core / frames remain uniform
TEMP_INNER = 20.0
TEMP_CORE = 70.0


def t_outer_profile(z):
    """Outer skin temperature as a function of axial position z (mm).

    Physical basis: aerodynamic heating during Max Q
      - Base (z=0): ~100 C  (attached to vehicle, moderate heating)
      - Barrel mid (z=2500): ~120 C
      - Ogive transition (z=5000): ~130 C
      - Ogive mid (z=7500): ~160 C
      - Nose (z=10400): ~220 C (stagnation point)
    """
    zn = max(0.0, min(z / TOTAL_HEIGHT, 1.0))  # normalized 0-1
    # Barrel: gentle linear (100 -> 130)
    # Ogive: quadratic rise to stagnation temp
    if z <= BARREL_H:
        return 100.0 + 30.0 * (z / BARREL_H)
    else:
        z_ogive = (z - BARREL_H) / (TOTAL_HEIGHT - BARREL_H)
        return 130.0 + 90.0 * z_ogive ** 1.5


def parse_nodes_and_instances(content):
    """Parse node coordinates and their instance membership from INP file."""
    # Find all node blocks per instance
    instance_nodes = {}  # {instance_name: {node_id: (x, y, z)}}

    # Pattern: *Instance, name=Part-OuterSkin-1, part=Part-OuterSkin
    inst_pattern = re.compile(
        r'\*Instance,\s*name=([^,\n]+)',
        re.IGNORECASE)
    node_block_pattern = re.compile(
        r'\*Node\s*\n((?:\s*\d+.*\n)+)',
        re.IGNORECASE)

    # Split content by instances
    parts = re.split(r'(\*Instance,\s*name=[^,\n]+[^\n]*\n)', content)
    current_inst = None
    for i, part in enumerate(parts):
        m = inst_pattern.search(part)
        if m:
            current_inst = m.group(1).strip()
            instance_nodes[current_inst] = {}
        elif current_inst:
            # Look for *Node blocks in this instance section
            for nm in node_block_pattern.finditer(part):
                node_text = nm.group(1)
                for line in node_text.strip().split('\n'):
                    line = line.strip()
                    if not line or line.startswith('*'):
                        break
                    tokens = line.split(',')
                    if len(tokens) >= 4:
                        try:
                            nid = int(tokens[0].strip())
                            x = float(tokens[1].strip())
                            y = float(tokens[2].strip())
                            z = float(tokens[3].strip())
                            instance_nodes[current_inst][nid] = (x, y, z)
                        except ValueError:
                            pass
            # Stop at *End Instance
            if '*End Instance' in part:
                current_inst = None

    return instance_nodes


def build_z_dependent_temperature_block(content):
    """Build *Temperature block with z-dependent outer skin temperature."""
    instance_nodes = parse_nodes_and_instances(content)

    temp_lines = []
    # Outer skin: z-dependent profile
    for inst_name, nodes in instance_nodes.items():
        inst_lower = inst_name.lower()
        if 'outerskin' in inst_lower:
            for nid, (x, y, z) in sorted(nodes.items()):
                # In Abaqus, y-axis is the axial direction for this model
                t = t_outer_profile(y)
                temp_lines.append('%s.%d, %.1f' % (inst_name, nid, t))
        elif 'innerskin' in inst_lower:
            # Inner skin: uniform inner temperature
            if nodes:
                temp_lines.append('%s.Set-All, %.1f' % (inst_name, TEMP_INNER))
        elif 'core' in inst_lower:
            # Core: uniform core temperature
            if nodes:
                temp_lines.append('%s.Set-All, %.1f' % (inst_name, TEMP_CORE))
        elif 'frame' in inst_lower:
            # Ring frames: inner temperature
            if nodes:
                temp_lines.append('%s.Set-All, %.1f' % (inst_name, TEMP_INNER))

    return temp_lines


def patch_inp(inp_path):
    with open(inp_path, 'r') as f:
        content = f.read()

    modified = False

    # ── 1. Initial Conditions ──
    if 'type=TEMPERATURE' not in content:
        insert_marker = '** \n** MATERIALS'
        if insert_marker in content:
            has_core = 'Part-Core-1.Set-All' in content or 'PART-CORE-1_SET-ALL' in content
            ic_lines = [
                'Part-InnerSkin-1.Set-All, %g' % TEMP_INITIAL,
                'Part-OuterSkin-1.Set-All, %g' % TEMP_INITIAL,
            ]
            if has_core:
                ic_lines.append('Part-Core-1.Set-All, %g' % TEMP_INITIAL)
            frame_sets = re.findall(r'(Part-Frame-\d+-1\.Set-All)', content)
            for fs in frame_sets:
                ic_lines.append('%s, %g' % (fs, TEMP_INITIAL))
            ic_block = ('\n** \n** PATCHED: Initial temperature\n'
                        '*Initial Conditions, type=TEMPERATURE\n'
                        + '\n'.join(ic_lines) + '\n')
            content = content.replace(insert_marker, ic_block + insert_marker)
            modified = True

    # ── 2. Add LE (strain) to Element Output ──
    # The field output line may be: S, U, RF, TEMP or S, TEMP, etc.
    if '*Element Output' in content and ', LE' not in content and 'LE,' not in content:
        # Insert LE before TEMP in element output
        # Match any line after *Element Output containing S
        pattern = r'(\*Element Output[^\n]*\n\s*)(S(?:,\s*\w+)*)'
        match = re.search(pattern, content)
        if match:
            old_vars = match.group(2)
            new_vars = old_vars.rstrip() + ', LE'
            content = content[:match.start(2)] + new_vars + content[match.end(2):]
            modified = True
            print("  Added LE to Element Output: %s -> %s" % (old_vars.strip(), new_vars.strip()))

    # ── 3. Add NT (nodal temperature) to ALL Node Output blocks ──
    if '*Node Output' in content:
        for pat, repl in [
            (r'(\*Node Output\s*\n\s*RF,\s*U)\s*\n', r'\1, NT\n'),
            (r'(\*Node Output\s*\n\s*U,\s*RF)\s*\n', r'\1, NT\n'),
        ]:
            new_content = re.sub(pat, repl, content)  # no count limit
            if new_content != content:
                content = new_content
                modified = True
                break

    # ── 4. z-dependent thermal load ──
    if '*Step, name=Step-1' in content and '** PATCHED: Thermal' not in content:
        # Build z-dependent temperature for outer skin
        temp_lines = build_z_dependent_temperature_block(content)

        if temp_lines:
            # Insert after *Static block
            pattern = r'(\*Static\s*\n\s*[^\n]+\.\s*\n)'
            thermal_block = (
                r'\1** PATCHED: Thermal load (z-dependent outer skin)\n'
                r'*Temperature\n'
                + r'\n'.join(temp_lines) + r'\n'
            )
            new_content = re.sub(pattern, thermal_block, content, count=1)
            if new_content != content:
                content = new_content
                modified = True
                print("  Applied z-dependent temperature profile (%d lines)" % len(temp_lines))

            # Remove CAE-generated uniform outer skin temperature (would override per-node)
            outer_temp_pat = re.compile(
                r'\*\*\s*Name:\s*Temp_Outer[^\n]*\n\*Temperature\s*\n'
                r'Part-OuterSkin-1\.Set-All,\s*\d+\.?\d*\s*\n',
                re.IGNORECASE)
            cleaned = outer_temp_pat.sub('', content)
            if cleaned != content:
                content = cleaned
                print("  Removed CAE uniform outer skin temperature (overridden by z-dependent)")
        else:
            # Fallback: uniform temperature if node parsing fails
            print("  Warning: Could not parse nodes, falling back to uniform temp")
            has_core = 'Part-Core-1.Set-All' in content
            temp_lines_fallback = [
                'Part-OuterSkin-1.Set-All, 120',
                'Part-InnerSkin-1.Set-All, 20',
            ]
            if has_core:
                temp_lines_fallback.append('Part-Core-1.Set-All, 70')
            frame_sets = re.findall(r'(Part-Frame-\d+-1\.Set-All)', content)
            for fs in frame_sets:
                temp_lines_fallback.append('%s, 20' % fs)
            pattern = r'(\*Static\s*\n\s*[^\n]+\.\s*\n)'
            thermal_block = (
                r'\1** PATCHED: Thermal load\n*Temperature\n'
                + r'\n'.join(temp_lines_fallback) + r'\n'
            )
            new_content = re.sub(pattern, thermal_block, content, count=1)
            if new_content != content:
                content = new_content
                modified = True

    if modified:
        with open(inp_path, 'w') as f:
            f.write(content)
        print("Patched %s" % inp_path)
    else:
        print("No patch needed for %s" % inp_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python patch_inp_thermal.py <inp_file>")
        sys.exit(1)
    patch_inp(sys.argv[1])
