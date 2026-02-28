#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify ODB thermal and displacement results.
Checks: (1) Step-1 temperature on outer skin ~120C, (2) non-zero displacement.
Run: abaqus python scripts/verify_odb_thermal.py --odb Job-Verify-Defect.odb
"""
import sys
import os
import argparse

try:
    from odbAccess import openOdb
except ImportError:
    print("Run with: abaqus python scripts/verify_odb_thermal.py --odb <odb>")
    sys.exit(1)


def verify_odb(odb_path):
    if not os.path.exists(odb_path):
        print("ODB not found: %s" % odb_path)
        return False
    print("Opening ODB: %s" % odb_path)
    odb = openOdb(path=odb_path)
    step_keys = list(odb.steps.keys())
    if not step_keys:
        print("Error: ODB has no steps")
        return False
    step_name = step_keys[0]
    for k in step_keys:
        if k.upper() != 'INITIAL':
            step_name = k
            break
    frame = odb.steps[step_name].frames[-1]
    print("Step: %s, Frame: %s" % (step_name, frame.frameId))

    instance_name = 'PART-OUTERSKIN-1'
    if instance_name not in odb.rootAssembly.instances.keys():
        instance_name = list(odb.rootAssembly.instances.keys())[0]
    instance = odb.rootAssembly.instances[instance_name]

    disp_field = frame.fieldOutputs['U'] if 'U' in frame.fieldOutputs else None
    temp_field = frame.fieldOutputs['TEMP'] if 'TEMP' in frame.fieldOutputs else None
    if not disp_field:
        print("ERROR: No displacement field (U) in ODB")
        return False
    if not temp_field:
        print("WARNING: No temperature field (TEMP) in ODB - thermal output may not be requested")

    disp_sub = disp_field.getSubset(region=instance)
    temp_sub = temp_field.getSubset(region=instance) if temp_field else None

    disp_vals = list(disp_sub.values)
    temp_vals = list(temp_sub.values) if temp_sub else []

    ux_list = [v.data[0] for v in disp_vals]
    uy_list = [v.data[1] for v in disp_vals]
    uz_list = [v.data[2] for v in disp_vals]
    temp_list = []
    for v in temp_vals:
        d = v.data
        temp_list.append(d if isinstance(d, (int, float)) else (d[0] if d else 0))

    def stats(name, vals):
        v = [x for x in vals if x is not None]
        v = [x for x in v if abs(x) < 1e30]
        if not v:
            return 0.0, 0.0, 0.0
        return min(v), max(v), sum(v) / len(v)

    ux_min, ux_max, ux_mean = stats("Ux", ux_list)
    uy_min, uy_max, uy_mean = stats("Uy", uy_list)
    uz_min, uz_max, uz_mean = stats("Uz", uz_list)
    print("")
    print("=== Displacement (U) ===")
    print("  Ux: min=%.6e max=%.6e mean=%.6e" % (ux_min, ux_max, ux_mean))
    print("  Uy: min=%.6e max=%.6e mean=%.6e" % (uy_min, uy_max, uy_mean))
    print("  Uz: min=%.6e max=%.6e mean=%.6e" % (uz_min, uz_max, uz_mean))

    if temp_list:
        t_min, t_max, t_mean = stats("TEMP", temp_list)
        print("")
        print("=== Temperature (TEMP) ===")
        print("  min=%.2f C  max=%.2f C  mean=%.2f C" % (t_min, t_max, t_mean))
        outer_expected = 120.0
        inner_expected = 20.0
        if abs(t_max - outer_expected) < 5:
            print("  [OK] Outer skin ~120C applied")
        else:
            print("  [CHECK] Expected outer~120C, got max=%.1f" % t_max)
    else:
        print("")
        print("=== Temperature: NOT in ODB ===")

    odb.close()
    disp_nonzero = any(abs(x) > 1e-10 for x in ux_list + uy_list + uz_list)
    return disp_nonzero


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--odb', type=str, required=True)
    args = parser.parse_args()
    ok = verify_odb(args.odb)
    sys.exit(0 if ok else 1)
