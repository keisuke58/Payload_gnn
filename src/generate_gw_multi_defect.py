# -*- coding: utf-8 -*-
"""
Generate Multi-Defect GW Fairing Model (Abaqus/Explicit)

Wraps generate_gw_fairing.py to place multiple defects in a single model.

Strategy:
  - For each defect: partition zone, create defect materials, assign sections
  - CZM surfaces: classify each element face against ALL defect zones
  - Sensors: 100-sensor grid to capture local scattering at each defect

Key modification vs single-defect:
  - _classify_core_surfaces_mesh checks against LIST of defect_params
  - Each defect gets unique surface names (Surf-Core-Outer-Defect-0, -1, ...)
  - General Contact assigns CZM-Damaged to all defect surfaces

Usage (Abaqus Python):
  abaqus cae noGUI=src/generate_gw_multi_defect.py -- \
    --doe doe_gw_multi_defect.json --model_idx 0
"""

from __future__ import print_function
import json
import math
import os
import sys

# Must be run under Abaqus Python
try:
    from abaqus import *
    from abaqusConstants import *
    from caeModules import *
except ImportError:
    print("ERROR: Must be run under 'abaqus cae noGUI=' or 'abaqus python'")
    sys.exit(1)

# Import the base generator
sys.path.insert(0, os.path.dirname(__file__))
from generate_gw_fairing import (
    generate_model,
    partition_defect_zone,
    create_defect_materials,
    create_defect_sections,
    _assign_defect_zone_sections,
    _point_in_defect_zone,
    _defect_damaged_interface,
    _create_elem_surface,
    RADIUS, CORE_T,
)


def _classify_core_surfaces_multi(assembly, inst_core, defect_list):
    """Classify core surfaces for multiple defects.

    For each element face, check against ALL defect zones.
    Creates per-defect damaged surfaces.

    Returns:
        surf_inner_h: healthy inner surface
        surf_outer_h: healthy outer surface
        defect_surfaces: list of (defect_idx, surf_inner_d, surf_outer_d)
    """
    r_inner = RADIUS
    r_outer = RADIUS + CORE_T
    tol_r = CORE_T * 0.3

    inner_h_elems = []
    outer_h_elems = []
    # Per-defect damaged elements
    inner_d_by_defect = {i: [] for i in range(len(defect_list))}
    outer_d_by_defect = {i: [] for i in range(len(defect_list))}

    for elem in inst_core.elements:
        nodes = elem.connectivity
        node_coords = [inst_core.nodes[n].coordinates for n in nodes]

        n_nodes = len(nodes)
        if n_nodes == 8:
            face_defs = [
                (0, 1, 2, 3, 'S1'), (4, 5, 6, 7, 'S2'),
                (0, 1, 5, 4, 'S3'), (1, 2, 6, 5, 'S4'),
                (2, 3, 7, 6, 'S5'), (3, 0, 4, 7, 'S6')]
        elif n_nodes == 6:
            face_defs = [
                (0, 1, 2, -1, 'S1'), (3, 4, 5, -1, 'S2'),
                (0, 1, 4, 3, 'S3'), (1, 2, 5, 4, 'S4'),
                (0, 2, 5, 3, 'S5')]
        elif n_nodes == 4:
            face_defs = [
                (0, 1, 2, -1, 'S1'), (0, 1, 3, -1, 'S2'),
                (1, 2, 3, -1, 'S3'), (0, 2, 3, -1, 'S4')]
        else:
            continue

        for fd in face_defs:
            fn = fd[-1]
            face_nodes = [i for i in fd[:-1] if i >= 0]
            if len(face_nodes) < 3:
                continue
            cx = sum([node_coords[i][0] for i in face_nodes]) / len(face_nodes)
            cy = sum([node_coords[i][1] for i in face_nodes]) / len(face_nodes)
            cz = sum([node_coords[i][2] for i in face_nodes]) / len(face_nodes)
            r_face = math.sqrt(cx * cx + cz * cz)

            # Check against all defects
            assigned_defect = None
            for di, dp in enumerate(defect_list):
                if _point_in_defect_zone(cx, cy, cz, dp):
                    damaged_iface = _defect_damaged_interface(dp)
                    if abs(r_face - r_inner) < tol_r and damaged_iface == 'inner':
                        assigned_defect = ('inner', di)
                        break
                    elif abs(r_face - r_outer) < tol_r and damaged_iface == 'outer':
                        assigned_defect = ('outer', di)
                        break

            if assigned_defect is not None:
                iface, di = assigned_defect
                if iface == 'inner':
                    inner_d_by_defect[di].append((elem.label, fn))
                else:
                    outer_d_by_defect[di].append((elem.label, fn))
            else:
                if abs(r_face - r_inner) < tol_r:
                    inner_h_elems.append((elem.label, fn))
                elif abs(r_face - r_outer) < tol_r:
                    outer_h_elems.append((elem.label, fn))

    # Create surfaces
    surf_inner_h = None
    surf_outer_h = None
    if inner_h_elems:
        surf_inner_h = _create_elem_surface(
            assembly, inst_core, 'Surf-Core-Inner-Healthy', inner_h_elems)
    if outer_h_elems:
        surf_outer_h = _create_elem_surface(
            assembly, inst_core, 'Surf-Core-Outer-Healthy', outer_h_elems)

    defect_surfaces = []
    for di in range(len(defect_list)):
        sid = None
        sod = None
        if inner_d_by_defect[di]:
            sid = _create_elem_surface(
                assembly, inst_core,
                'Surf-Core-Inner-Defect-%d' % di, inner_d_by_defect[di])
        if outer_d_by_defect[di]:
            sod = _create_elem_surface(
                assembly, inst_core,
                'Surf-Core-Outer-Defect-%d' % di, outer_d_by_defect[di])
        defect_surfaces.append((di, sid, sod))

    n_d = sum(1 for _, s1, s2 in defect_surfaces if s1 or s2)
    print("Multi-defect surfaces: %d defects with damaged interfaces" % n_d)

    return surf_inner_h, surf_outer_h, defect_surfaces


def create_interactions_multi(model, assembly,
                               inst_inner, inst_core, inst_outer,
                               defect_list):
    """Create General Contact with multiple CZM defect zones."""
    surf_inner_h, surf_outer_h, defect_surfaces = \
        _classify_core_surfaces_multi(assembly, inst_core, defect_list)

    # Skin surfaces
    surf_inner_skin = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')
    surf_outer_skin = assembly.Surface(
        side1Faces=inst_outer.faces, name='Surf-OuterSkin')

    # General Contact
    gc = model.ContactExp(name='GeneralContact', createStepName='Initial')
    gc.includedPairs.setValuesInStep(stepName='Initial', useAllstar=ON)
    gc.contactPropertyAssignments.appendInStep(
        stepName='Initial',
        assignments=((GLOBAL, SELF, 'IntProp-Default'),))

    pair_assignments = []

    # Healthy interfaces
    if surf_inner_h is not None:
        pair_assignments.append(
            (assembly.surfaces['Surf-Core-Inner-Healthy'],
             surf_inner_skin, 'IntProp-CZM-Healthy'))
    if surf_outer_h is not None:
        pair_assignments.append(
            (assembly.surfaces['Surf-Core-Outer-Healthy'],
             surf_outer_skin, 'IntProp-CZM-Healthy'))

    # Per-defect damaged interfaces
    for di, surf_id, surf_od in defect_surfaces:
        if surf_id is not None:
            pair_assignments.append(
                (assembly.surfaces['Surf-Core-Inner-Defect-%d' % di],
                 surf_inner_skin, 'IntProp-CZM-Damaged'))
        if surf_od is not None:
            pair_assignments.append(
                (assembly.surfaces['Surf-Core-Outer-Defect-%d' % di],
                 surf_outer_skin, 'IntProp-CZM-Damaged'))

    if pair_assignments:
        gc.contactPropertyAssignments.appendInStep(
            stepName='Initial',
            assignments=tuple(pair_assignments))

    print("General Contact: %d CZM pairs (%d defect zones)" % (
        len(pair_assignments), len(defect_list)))


def generate_multi_defect_model(doe_path, model_idx=0,
                                  no_run=True, n_sensors=100):
    """Generate a multi-defect fairing GW model.

    Uses the base generate_model for geometry/mesh/step setup,
    then replaces defect zone handling with multi-defect logic.
    """
    with open(doe_path) as f:
        doe = json.load(f)

    if model_idx >= len(doe['models']):
        print("ERROR: model_idx %d >= %d models" % (model_idx, len(doe['models'])))
        return

    model_spec = doe['models'][model_idx]
    job_name = model_spec['job_name']
    defect_list = model_spec['defects']
    n_defects = len(defect_list)

    print("=" * 60)
    print("Multi-Defect Model: %s (%d defects)" % (job_name, n_defects))
    print("=" * 60)

    # Step 1: Generate base model WITHOUT defect (healthy geometry)
    # We'll add defects manually after
    generate_model(
        job_name=job_name,
        defect_params=None,  # No defect initially
        n_sensors=n_sensors,
        sensor_layout='grid',
        no_run=True,  # Don't run yet
    )

    # Step 2: Re-open model and add multi-defect features
    mdb_path = job_name + '.cae'
    if not os.path.exists(mdb_path):
        print("ERROR: CAE not found: %s" % mdb_path)
        return

    mdb = openMdb(mdb_path)
    model = mdb.models['Model-1']

    # Get parts
    p_core = model.parts['Part-Core']
    p_outer = model.parts['Part-OuterSkin']

    # Step 3: Partition and assign materials for each defect
    for di, dp in enumerate(defect_list):
        print("\nDefect %d/%d: z=%.0f, theta=%.1f, r=%.0f, type=%s" % (
            di + 1, n_defects, dp['z_center'], dp['theta_deg'],
            dp['radius'], dp.get('defect_type', 'debonding')))

        # Partition
        partition_defect_zone(p_core, p_outer, dp)

        # Create defect-specific materials
        skin_mat, core_mat = create_defect_materials(model, dp)

        # Assign sections to defect zone elements
        _assign_defect_zone_sections(p_core, p_outer, dp, skin_mat, core_mat)

    # Step 4: Re-mesh (partitioning may have changed geometry)
    a = model.rootAssembly
    a.regenerate()

    # Step 5: Re-create interactions with multi-defect surfaces
    # Delete old GeneralContact if exists
    if 'GeneralContact' in model.interactions:
        del model.interactions['GeneralContact']

    inst_inner = a.instances['Part-InnerSkin-1']
    inst_core = a.instances['Part-Core-1']
    inst_outer = a.instances['Part-OuterSkin-1']

    create_interactions_multi(model, a, inst_inner, inst_core, inst_outer,
                               defect_list)

    # Step 6: Save and write INP
    mdb.saveAs(pathName=job_name + '.cae')
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)

    print("\n" + "=" * 60)
    print("Multi-defect INP written: %s.inp" % job_name)
    print("  Defects: %d" % n_defects)
    print("  Sensors: %d (grid)" % n_sensors)
    print("=" * 60)


# ==============================================================================
# CLI
# ==============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate multi-defect GW fairing model')
    parser.add_argument('--doe', type=str, required=True,
                        help='Path to multi-defect DOE JSON')
    parser.add_argument('--model_idx', type=int, default=0,
                        help='Model index in DOE (default: 0)')
    parser.add_argument('--n_sensors', type=int, default=100,
                        help='Number of sensors (default: 100)')
    parser.add_argument('--no_run', action='store_true', default=True,
                        help='Write INP only (default: True)')

    args, _ = parser.parse_known_args()

    generate_multi_defect_model(
        doe_path=args.doe,
        model_idx=args.model_idx,
        no_run=args.no_run,
        n_sensors=args.n_sensors,
    )
