# -*- coding: utf-8 -*-
"""
extract_pce_qoi.py  -- Abaqus Python script
PCE UQ用 QoI (Quantity of Interest) 抽出スクリプト

Extracts from Step-2 (Thermal + Mechanical, Max-Q) of the static .odb:
  - max_smises : max von Mises stress in CFRP face sheets [MPa]
  - max_sdeg   : max scalar damage (SDEG) in adhesive/cohesive elements
  - max_disp   : max displacement magnitude U_magnitude [mm]

Usage (must be run via Abaqus Python, not system Python):
    abaqus python extract_pce_qoi.py --odb <path>.odb --output <path>_qoi.json
"""

from __future__ import print_function
import sys
import os
import json
import argparse

# odbAccess is only available inside Abaqus Python
try:
    from odbAccess import openOdb
    from abaqusConstants import NODAL, INTEGRATION_POINT
except ImportError:
    sys.exit("ERROR: Run this script with 'abaqus python', not system Python.")


def extract_qoi(odb_path, output_json):
    """Open ODB and extract max_smises, max_sdeg, max_disp from Step-2."""

    if not os.path.exists(odb_path):
        print("ERROR: ODB not found: %s" % odb_path)
        return False

    odb = openOdb(odb_path, readOnly=True)

    try:
        # Target step: Step-2 (Thermal + Mechanical)
        step_name = 'Step-2'
        if step_name not in odb.steps:
            # Fallback: use last step
            step_name = list(odb.steps.keys())[-1]
            print("  [INFO] Step-2 not found, using: %s" % step_name)

        step   = odb.steps[step_name]
        frames = step.frames
        if not frames:
            print("ERROR: No frames in %s" % step_name)
            odb.close()
            return False

        last_frame = frames[-1]

        # ---- (A) Max von Mises stress in CFRP skins (element output) ----
        max_smises = 0.0
        if 'S' in last_frame.fieldOutputs:
            s_field = last_frame.fieldOutputs['S']
            # Filter to CFRP skin instances
            skin_names = ['Part-InnerSkin-1', 'Part-OuterSkin-1']
            for inst_name in skin_names:
                if inst_name in odb.rootAssembly.instances:
                    inst   = odb.rootAssembly.instances[inst_name]
                    subset = s_field.getSubset(region=inst)
                    for val in subset.values:
                        if val.mises > max_smises:
                            max_smises = val.mises
        # Fallback: global max
        if max_smises == 0.0 and 'S' in last_frame.fieldOutputs:
            for val in last_frame.fieldOutputs['S'].values:
                if val.mises > max_smises:
                    max_smises = val.mises

        # ---- (B) Max damage variable SDEG in adhesive layers ------------
        max_sdeg = 0.0
        if 'SDEG' in last_frame.fieldOutputs:
            sdeg_field = last_frame.fieldOutputs['SDEG']
            adh_names  = ['Part-AdhesiveInner-1', 'Part-AdhesiveOuter-1']
            for inst_name in adh_names:
                if inst_name in odb.rootAssembly.instances:
                    inst   = odb.rootAssembly.instances[inst_name]
                    subset = sdeg_field.getSubset(region=inst)
                    for val in subset.values:
                        if val.data > max_sdeg:
                            max_sdeg = val.data
            # Fallback: global
            if max_sdeg == 0.0:
                for val in sdeg_field.values:
                    if val.data > max_sdeg:
                        max_sdeg = val.data

        # ---- (C) Max displacement magnitude (nodal) ---------------------
        max_disp = 0.0
        if 'U' in last_frame.fieldOutputs:
            u_field = last_frame.fieldOutputs['U']
            for val in u_field.values:
                mag = val.magnitude
                if mag > max_disp:
                    max_disp = mag

        result = {
            'max_smises': float(max_smises),
            'max_sdeg':   float(max_sdeg),
            'max_disp':   float(max_disp),
            'step':       step_name,
            'frame':      len(frames) - 1,
        }

        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        print("  QoI: smises=%.1f MPa  sdeg=%.4f  disp=%.3f mm" % (
            max_smises, max_sdeg, max_disp))
        return True

    finally:
        odb.close()


def main():
    parser = argparse.ArgumentParser(
        description='Extract QoI from Abaqus ODB for PCE UQ')
    parser.add_argument('--odb',    required=True, help='Path to .odb file')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    ok = extract_qoi(args.odb, args.output)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
