#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch Abaqus INP file to add thermal load and BC if missing.
Run: python scripts/patch_inp_thermal.py Job-Verify-Defect.inp
"""
import sys
import re

TEMP_INITIAL = 20.0
TEMP_OUTER = 120.0
TEMP_INNER = 20.0
TEMP_CORE = 70.0


def patch_inp(inp_path):
    with open(inp_path, 'r') as f:
        content = f.read()
    
    modified = False

    # Check if *Initial Conditions, type=TEMPERATURE exists
    if 'type=TEMPERATURE' not in content:
        insert_marker = '** \n** MATERIALS'
        if insert_marker in content:
            # Include Core if Part-Core-1 has nodes in the INP
            has_core = 'Part-Core-1' in content and '*Node' in content.split('Part-Core-1')[0].split('*End Instance')[0] if 'Part-Core-1' in content else False
            # Simple check: core is present if Part-Core-1 has a Set-All
            has_core = 'Part-Core-1.Set-All' in content or 'PART-CORE-1_SET-ALL' in content
            ic_lines = [
                'Part-InnerSkin-1.Set-All, %g' % TEMP_INITIAL,
                'Part-OuterSkin-1.Set-All, %g' % TEMP_INITIAL,
            ]
            if has_core:
                ic_lines.append('Part-Core-1.Set-All, %g' % TEMP_INITIAL)
            ic_block = '\n** \n** PATCHED: Initial temperature\n*Initial Conditions, type=TEMPERATURE\n' + '\n'.join(ic_lines) + '\n'
            content = content.replace(insert_marker, ic_block + insert_marker)
            modified = True
    
    # Add NT (nodal temperature) to Node Output for extraction
    if '*Node Output' in content and 'RF, U, NT' not in content and 'U, RF, NT' not in content:
        # Match RF, U or U, RF (Abaqus may write either order)
        for pat, repl in [
            (r'(\*Node Output\s*\n\s*RF,\s*U)\s*\n', r'\1, NT\n'),
            (r'(\*Node Output\s*\n\s*U,\s*RF)\s*\n', r'\1, NT\n'),
        ]:
            new_content = re.sub(pat, repl, content, count=1)
            if new_content != content:
                content = new_content
                modified = True
                break

    # Check if *Temperature exists in Step-1 (thermal load)
    if '*Step, name=Step-1' in content and '** PATCHED: Thermal' not in content:
        has_core = 'Part-Core-1.Set-All' in content or 'PART-CORE-1_SET-ALL' in content
        # Insert *Temperature after *Static block
        # Match *Static + newline + data line (e.g. "1., 1., 1e-05, 1.")
        pattern = r'(\*Static\s*\n\s*[^\n]+\.\s*\n)'
        temp_lines = [
            'Part-OuterSkin-1.Set-All, %g' % TEMP_OUTER,
            'Part-InnerSkin-1.Set-All, %g' % TEMP_INNER,
        ]
        if has_core:
            temp_lines.append('Part-Core-1.Set-All, %g' % TEMP_CORE)
        thermal_block = (
            r'\1** PATCHED: Thermal load\n*Temperature\n'
            + r'\n'.join(temp_lines) + r'\n'
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
