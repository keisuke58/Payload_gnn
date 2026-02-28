# -*- coding: utf-8 -*-
# generate_realistic_dataset.py
# Realistic H3 Type-S fairing FEM with openings, ring frames, Tie constraints,
# AND multi-type defect insertion for ML dataset generation.
#
# Combines generate_realistic_fairing.py (openings, ring frames, Tie)
# with generate_fairing_dataset.py (7 defect types, partitioning, section assignment).
#
# 3-tier section assignment: healthy -> void (openings) -> defect (override)
#
# Usage:
#   abaqus cae noGUI=generate_realistic_dataset.py -- --job <name> --param_file <json>
#   abaqus cae noGUI=generate_realistic_dataset.py -- --job <name> --phase 2 --global_seed 25

import sys
import os
import math
import json
import argparse
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup

executeOnCaeStartup()

# ==============================================================================
# GEOMETRY PARAMETERS (H3 Type-S Fairing)
# ==============================================================================
RADIUS = 2600.0       # mm (phi 5.2 m)
H_BARREL = 5000.0     # mm
H_NOSE = 5400.0       # mm
TOTAL_HEIGHT = H_BARREL + H_NOSE  # 10400 mm

# Tangent Ogive
OGIVE_RHO = (RADIUS**2 + H_NOSE**2) / (2.0 * RADIUS)
OGIVE_XC = RADIUS - OGIVE_RHO

# Thicknesses
FACE_T = 1.0          # mm (CFRP skin)
CORE_T = 38.0         # mm (Al honeycomb core)

# ==============================================================================
# MATERIAL PROPERTIES
# ==============================================================================
# CFRP Face Sheets (Toray T1000G)
E1 = 160000.0         # MPa
E2 = 10000.0
NU12 = 0.3
G12 = 5000.0
G13 = 5000.0
G23 = 3000.0
CFRP_DENSITY = 1600e-12   # tonne/mm^3
CFRP_CTE = 2e-6           # /C

# Aluminum Honeycomb Core (5052)
E_CORE_1 = 1.0
E_CORE_2 = 1.0
E_CORE_3 = 1000.0
NU_CORE_12 = 0.01
NU_CORE_13 = 0.01
NU_CORE_23 = 0.01
G_CORE_12 = 1.0
G_CORE_13 = 400.0
G_CORE_23 = 240.0
CORE_DENSITY = 50e-12
CORE_CTE = 23e-6

# ==============================================================================
# THERMAL LOAD
# ==============================================================================
TEMP_INITIAL = 20.0
TEMP_FINAL_OUTER = 120.0
TEMP_FINAL_INNER = 20.0
TEMP_FINAL_CORE = 70.0

# ==============================================================================
# MESH
# ==============================================================================
GLOBAL_SEED = 25.0     # mm
DEFECT_SEED = 10.0     # mm (around defect zone)
OPENING_SEED = 10.0    # mm (around openings)
FRAME_SEED = 15.0      # mm (ring frames)
BOUNDARY_SEED_RATIO = 0.4  # partition boundary seed = DEFECT_SEED * ratio
BOUNDARY_MARGIN = 30.0     # mm — refinement width around partition boundaries

# ==============================================================================
# OPENING DEFINITIONS
# ==============================================================================
OPENINGS_PHASE1 = [
    {
        'name': 'AccessDoor',
        'theta_deg': 30.0,
        'z_center': 1500.0,
        'diameter': 1300.0,
    },
]

OPENINGS_PHASE2 = [
    {
        'name': 'AccessDoor',
        'theta_deg': 30.0,
        'z_center': 1500.0,
        'diameter': 1300.0,
    },
    {
        'name': 'HVAC_Door',
        'theta_deg': 20.0,
        'z_center': 2500.0,
        'diameter': 400.0,
    },
    {
        'name': 'RF_Window',
        'theta_deg': 40.0,
        'z_center': 4000.0,
        'diameter': 400.0,
    },
    {
        'name': 'Vent_1',
        'theta_deg': 15.0,
        'z_center': 300.0,
        'diameter': 100.0,
    },
    {
        'name': 'Vent_2',
        'theta_deg': 45.0,
        'z_center': 300.0,
        'diameter': 100.0,
    },
]

# ==============================================================================
# RING FRAME DEFINITIONS
# ==============================================================================
RING_FRAME_Z_POSITIONS = [500.0, 1000.0, 1500.0, 2000.0, 2500.0,
                          3000.0, 3500.0, 4000.0, 4500.0]
RING_FRAME_HEIGHT = 50.0     # mm (radial extent, inward from inner skin)
RING_FRAME_THICKNESS = 3.0   # mm (shell thickness of frame web)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_radius_at_z(z):
    """Outer radius at axial position z (mm). Y=axial in Abaqus."""
    if z < 0:
        return RADIUS
    elif z <= H_BARREL:
        return RADIUS
    elif z > TOTAL_HEIGHT:
        return 0.0
    else:
        z_local = z - H_BARREL
        term = OGIVE_RHO**2 - z_local**2
        if term < 0:
            return 0.0
        return OGIVE_XC + math.sqrt(term)


def _is_point_in_opening(x, y, z, opening, radius_offset=0.0):
    """Check if point (x,y,z) is inside a circular opening."""
    z_c = opening['z_center']
    r_half = opening['diameter'] / 2.0
    if abs(y - z_c) > r_half * 1.5:
        return False
    r_local = math.sqrt(x * x + z * z)
    if r_local < 1.0:
        return False
    theta_rad_pt = math.atan2(z, x)
    theta_center_rad = math.radians(opening['theta_deg'])
    arc_mm = r_local * abs(theta_rad_pt - theta_center_rad)
    dy = y - z_c
    dist = math.sqrt(arc_mm * arc_mm + dy * dy)
    return dist <= r_half * 1.05


def _is_point_in_any_opening(x, y, z, openings, radius_offset=0.0):
    """Check if point is inside ANY opening."""
    for op in openings:
        if _is_point_in_opening(x, y, z, op, radius_offset):
            return True
    return False


def _point_in_defect_zone(x, y, z, defect_params):
    """Check if point (x,y,z) is inside the circular defect zone on the fairing surface."""
    z_c = defect_params['z_center']
    theta_deg = defect_params['theta_deg']
    r_def = defect_params['radius']
    if abs(y - z_c) > r_def * 1.5:
        return False
    r_local = math.sqrt(x * x + z * z)
    if r_local < 1.0:
        return False
    theta_rad_pt = math.atan2(z, x)
    theta_center_rad = math.radians(theta_deg)
    arc_mm = r_local * abs(theta_rad_pt - theta_center_rad)
    dy = y - z_c
    dist = math.sqrt(arc_mm * arc_mm + dy * dy)
    return dist <= r_def * 1.01


def is_face_in_defect_zone(face, defect_params):
    """Checks if a face's centroid is within the defect zone."""
    if not defect_params:
        return False
    pt = face.pointOn[0]
    return _point_in_defect_zone(pt[0], pt[1], pt[2], defect_params)


def is_cell_in_defect_zone(cell, defect_params, openings=None):
    """Checks if a solid cell's centroid is within the defect zone (excludes openings)."""
    if not defect_params:
        return False
    pt = cell.pointOn[0]
    if openings and _is_point_in_any_opening(pt[0], pt[1], pt[2], openings):
        return False
    return _point_in_defect_zone(pt[0], pt[1], pt[2], defect_params)


# ==============================================================================
# MATERIALS AND SECTIONS
# ==============================================================================

def create_materials(model):
    """Define CFRP, Honeycomb, frame, and void materials."""
    # CFRP T1000G
    mat = model.Material(name='CFRP_T1000G')
    mat.Elastic(type=LAMINA, table=((E1, E2, NU12, G12, G13, G23),))
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Expansion(table=((CFRP_CTE, CFRP_CTE, 0.0),))

    # Aluminum Honeycomb Core
    mat = model.Material(name='AL_HONEYCOMB')
    mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
        E_CORE_1, E_CORE_2, E_CORE_3,
        NU_CORE_12, NU_CORE_13, NU_CORE_23,
        G_CORE_12, G_CORE_13, G_CORE_23
    ),))
    mat.Density(table=((CORE_DENSITY,),))
    mat.Expansion(table=((CORE_CTE,),))

    # CFRP for ring frame (isotropic simplification)
    mat = model.Material(name='CFRP_FRAME')
    mat.Elastic(type=ISOTROPIC, table=((70000.0, 0.3),))
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Expansion(table=((CFRP_CTE,),))

    # Void material for opening regions
    mat = model.Material(name='VOID')
    mat.Elastic(type=ISOTROPIC, table=((1.0, 0.3),))
    mat.Density(table=((1e-20,),))
    mat.Expansion(table=((0.0,),))


def create_sections(model):
    """Create shell and solid sections for healthy, void, and frame."""
    # Composite layup [45/0/-45/90]s (8 plies)
    angles = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
    entries = [section.SectionLayer(
        thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_T1000G')
        for ang in angles]
    model.CompositeShellSection(
        name='Section-CFRP-Skin', preIntegrate=OFF,
        idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
        thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        temperature=GRADIENT, integrationRule=SIMPSON)

    # Solid core
    model.HomogeneousSolidSection(
        name='Section-Core', material='AL_HONEYCOMB', thickness=None)

    # Ring frame shell
    model.HomogeneousShellSection(
        name='Section-Frame', material='CFRP_FRAME',
        thickness=RING_FRAME_THICKNESS)

    # Void sections for opening regions
    model.HomogeneousShellSection(
        name='Section-Void-Shell', material='VOID', thickness=0.01)
    model.HomogeneousSolidSection(
        name='Section-Void-Solid', material='VOID')


def create_defect_materials(model, defect_params):
    """Create defect-type-specific modified materials."""
    defect_type = defect_params.get('defect_type', 'debonding')

    if defect_type == 'debonding':
        mat = model.Material(name='CFRP_DEBONDED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.01, E2 * 0.01, NU12,
            G12 * 0.01, G13 * 0.01, G23 * 0.01
        ),))
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Expansion(table=((CFRP_CTE, CFRP_CTE, 0.0),))

    elif defect_type == 'fod':
        sf = defect_params.get('stiffness_factor', 10.0)
        mat = model.Material(name='AL_HONEYCOMB_FOD')
        mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1 * sf, E_CORE_2 * sf, E_CORE_3 * sf,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12 * sf, G_CORE_13 * sf, G_CORE_23 * sf
        ),))
        mat.Density(table=((200e-12,),))
        mat.Expansion(table=((12e-6,),))

    elif defect_type == 'impact':
        dr = defect_params.get('damage_ratio', 0.3)
        mat_skin = model.Material(name='CFRP_IMPACT_DAMAGED')
        mat_skin.Elastic(type=LAMINA, table=((
            E1 * 0.7, E2 * dr, NU12,
            G12 * dr, G13 * dr, G23 * dr
        ),))
        mat_skin.Density(table=((CFRP_DENSITY,),))
        mat_skin.Expansion(table=((CFRP_CTE, CFRP_CTE, 0.0),))

        mat_core = model.Material(name='AL_HONEYCOMB_CRUSHED')
        mat_core.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1 * 0.5, E_CORE_2 * 0.5, E_CORE_3 * 0.1,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12 * 0.5, G_CORE_13 * 0.1, G_CORE_23 * 0.1
        ),))
        mat_core.Density(table=((CORE_DENSITY,),))
        mat_core.Expansion(table=((CORE_CTE,),))

    elif defect_type == 'delamination':
        depth = defect_params.get('delam_depth', 0.5)
        shear_red = max(0.05, 1.0 - depth)
        mat = model.Material(name='CFRP_DELAMINATED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.9, E2 * (0.3 + 0.5 * (1 - depth)), NU12,
            G12 * shear_red, G13 * shear_red, G23 * shear_red
        ),))
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Expansion(table=((CFRP_CTE, CFRP_CTE, 0.0),))

    elif defect_type == 'inner_debond':
        mat = model.Material(name='CFRP_INNER_DEBONDED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.01, E2 * 0.01, NU12,
            G12 * 0.01, G13 * 0.01, G23 * 0.01
        ),))
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Expansion(table=((CFRP_CTE, CFRP_CTE, 0.0),))

    elif defect_type == 'thermal_progression':
        mat = model.Material(name='CFRP_THERMAL_DAMAGED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.05, E2 * 0.05, NU12,
            G12 * 0.05, G13 * 0.05, G23 * 0.05
        ),))
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Expansion(table=((8e-6, 8e-6, 0.0),))

    elif defect_type == 'acoustic_fatigue':
        sev = defect_params.get('fatigue_severity', 0.35)
        mat = model.Material(name='CFRP_ACOUSTIC_FATIGUED')
        mat.Elastic(type=LAMINA, table=((
            E1 * (0.2 + 0.5 * (1 - sev)), E2 * sev, NU12,
            G12 * sev, G13 * sev, G23 * sev
        ),))
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Expansion(table=((CFRP_CTE, CFRP_CTE, 0.0),))

    print("Defect materials created: type=%s" % defect_type)


def create_defect_sections(model, defect_params):
    """Create sections for defect-zone materials."""
    defect_type = defect_params.get('defect_type', 'debonding')
    angles = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]

    if defect_type == 'debonding':
        entries = [section.SectionLayer(
            thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_DEBONDED')
            for ang in angles]
        model.CompositeShellSection(
            name='Section-CFRP-Debonded', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

    elif defect_type == 'fod':
        model.HomogeneousSolidSection(
            name='Section-Core-FOD', material='AL_HONEYCOMB_FOD', thickness=None)

    elif defect_type == 'impact':
        entries = [section.SectionLayer(
            thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_IMPACT_DAMAGED')
            for ang in angles]
        model.CompositeShellSection(
            name='Section-CFRP-Impact', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)
        model.HomogeneousSolidSection(
            name='Section-Core-Crushed', material='AL_HONEYCOMB_CRUSHED', thickness=None)

    elif defect_type == 'delamination':
        entries = [section.SectionLayer(
            thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_DELAMINATED')
            for ang in angles]
        model.CompositeShellSection(
            name='Section-CFRP-Delaminated', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

    elif defect_type == 'inner_debond':
        entries = [section.SectionLayer(
            thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_INNER_DEBONDED')
            for ang in angles]
        model.CompositeShellSection(
            name='Section-CFRP-InnerDebonded', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

    elif defect_type == 'thermal_progression':
        entries = [section.SectionLayer(
            thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_THERMAL_DAMAGED')
            for ang in angles]
        model.CompositeShellSection(
            name='Section-CFRP-ThermalDamaged', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

    elif defect_type == 'acoustic_fatigue':
        entries = [section.SectionLayer(
            thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_ACOUSTIC_FATIGUED')
            for ang in angles]
        model.CompositeShellSection(
            name='Section-CFRP-AcousticFatigued', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)


# ==============================================================================
# GEOMETRY: BASE PARTS
# ==============================================================================

def create_base_parts(model):
    """Create barrel + ogive nose parts via revolve. Returns (p_inner, p_core, p_outer)."""

    # --- Inner Skin (Shell) ---
    s1 = model.ConstrainedSketch(name='profile_inner', sheetSize=20000.0)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))
    s1.Line(point1=(RADIUS, 0.0), point2=(RADIUS, H_BARREL))
    s1.Line(point1=(RADIUS, H_BARREL), point2=(0.0, TOTAL_HEIGHT))

    p_inner = model.Part(name='Part-InnerSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=60.0, flipRevolveDirection=OFF)

    # --- Core (Solid) ---
    s2 = model.ConstrainedSketch(name='profile_core', sheetSize=20000.0)
    s2.setPrimaryObject(option=STANDALONE)
    s2.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))

    rho_outer = OGIVE_RHO + CORE_T
    z_tip_outer = H_BARREL + math.sqrt(rho_outer**2 - OGIVE_XC**2)

    s2.Line(point1=(RADIUS, 0.0), point2=(RADIUS, H_BARREL))
    s2.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(RADIUS, H_BARREL),
        point2=(0.0, TOTAL_HEIGHT),
        direction=COUNTERCLOCKWISE)
    s2.Line(point1=(0.0, TOTAL_HEIGHT), point2=(0.0, z_tip_outer))
    s2.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(0.0, z_tip_outer),
        point2=(RADIUS + CORE_T, H_BARREL),
        direction=CLOCKWISE)
    s2.Line(point1=(RADIUS + CORE_T, H_BARREL), point2=(RADIUS + CORE_T, 0.0))
    s2.Line(point1=(RADIUS + CORE_T, 0.0), point2=(RADIUS, 0.0))

    p_core = model.Part(name='Part-Core', dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=s2, angle=60.0, flipRevolveDirection=OFF)

    # --- Outer Skin (Shell) ---
    s3 = model.ConstrainedSketch(name='profile_outer', sheetSize=20000.0)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))
    s3.Line(point1=(RADIUS + CORE_T, 0.0), point2=(RADIUS + CORE_T, H_BARREL))
    s3.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(RADIUS + CORE_T, H_BARREL),
        point2=(0.0, z_tip_outer),
        direction=COUNTERCLOCKWISE)

    p_outer = model.Part(name='Part-OuterSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=60.0, flipRevolveDirection=OFF)

    print("Base parts created: InnerSkin, Core, OuterSkin")
    return p_inner, p_core, p_outer


# ==============================================================================
# PARTITIONING: OPENINGS
# ==============================================================================

def partition_opening(part, opening, geom_type):
    """Partition a part at the opening boundary using 4 datum planes."""
    z_c = opening['z_center']
    r_half = opening['diameter'] / 2.0
    theta_rad = math.radians(opening['theta_deg'])

    r_shell = get_radius_at_z(z_c) + CORE_T
    if r_shell < 1.0:
        r_shell = RADIUS + CORE_T
    d_theta = min(r_half / r_shell, math.radians(25.0))
    t1 = theta_rad - d_theta
    t2 = theta_rad + d_theta

    dp_z1 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
                                             offset=z_c - r_half)
    dp_z2 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
                                             offset=z_c + r_half)
    dp_t1 = part.DatumPlaneByThreePoints(
        point1=(0, 0, 0), point2=(0, 100, 0),
        point3=(math.cos(t1), 0, math.sin(t1)))
    dp_t2 = part.DatumPlaneByThreePoints(
        point1=(0, 0, 0), point2=(0, 100, 0),
        point3=(math.cos(t2), 0, math.sin(t2)))

    for dp_id in [dp_z1.id, dp_z2.id, dp_t1.id, dp_t2.id]:
        try:
            if geom_type == 'shell':
                if len(part.faces) > 0:
                    part.PartitionFaceByDatumPlane(
                        datumPlane=part.datums[dp_id], faces=part.faces)
            else:
                if len(part.cells) > 0:
                    part.PartitionCellByDatumPlane(
                        datumPlane=part.datums[dp_id], cells=part.cells)
        except Exception as e:
            print("  Partition warning (%s, %s): %s" % (
                opening['name'], geom_type, str(e)[:60]))

    print("  Opening partitioned: %s (z=%.0f, theta=%.1f, D=%.0f)" % (
        opening['name'], z_c, opening['theta_deg'], opening['diameter']))


def partition_all_openings(p_inner, p_core, p_outer, openings):
    """Partition all 3 parts for each opening."""
    print("Partitioning openings (%d total)..." % len(openings))
    for opening in openings:
        partition_opening(p_inner, opening, 'shell')
        partition_opening(p_core, opening, 'solid')
        partition_opening(p_outer, opening, 'shell')


# ==============================================================================
# PARTITIONING: DEFECT ZONE
# ==============================================================================

def partition_defect_zone(parts_to_partition, defect_params):
    """Partitions parts at the defect zone for section reassignment."""
    z_c = defect_params['z_center']
    theta_deg = defect_params['theta_deg']
    r_def = defect_params['radius']

    r_local = get_radius_at_z(z_c) + CORE_T
    if r_local < 1.0:
        r_local = RADIUS
    theta_rad = math.radians(theta_deg)
    d_theta = min(r_def / r_local, math.radians(30.0))
    t1 = theta_rad - d_theta
    t2 = theta_rad + d_theta

    for geom_type, part in parts_to_partition:
        dp_z1 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=z_c - r_def)
        dp_z2 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=z_c + r_def)
        dp_t1 = part.DatumPlaneByThreePoints(
            point1=(0, 0, 0), point2=(0, 100, 0),
            point3=(math.cos(t1), 0, math.sin(t1)))
        dp_t2 = part.DatumPlaneByThreePoints(
            point1=(0, 0, 0), point2=(0, 100, 0),
            point3=(math.cos(t2), 0, math.sin(t2)))

        for dp_id in [dp_z1.id, dp_z2.id, dp_t1.id, dp_t2.id]:
            try:
                if geom_type == 'shell':
                    faces = part.faces
                    if len(faces) > 0:
                        part.PartitionFaceByDatumPlane(
                            datumPlane=part.datums[dp_id], faces=faces)
                else:
                    cells = part.cells
                    if len(cells) > 0:
                        part.PartitionCellByDatumPlane(
                            datumPlane=part.datums[dp_id], cells=cells)
            except Exception as e:
                print("  Part partition warning (%s): %s" % (geom_type, str(e)[:80]))

    print("Defect zone partitioned: z=%.0f theta=%.1f r=%.0f" % (z_c, theta_deg, r_def))


# ==============================================================================
# SECTION ASSIGNMENT: 3-TIER (HEALTHY -> VOID -> DEFECT)
# ==============================================================================

def assign_sections_3tier(p_inner, p_core, p_outer, openings, defect_params):
    """
    3-tier section assignment:
      Tier 1: Healthy baseline (all faces/cells)
      Tier 2: Void override (opening regions)
      Tier 3: Defect override (defect zone, excluding openings)

    Abaqus: last SectionAssignment wins for any given face/cell.
    """
    # ---- Tier 1: Healthy baseline ----
    region = p_inner.Set(faces=p_inner.faces, name='Set-All')
    p_inner.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_inner.MaterialOrientation(region=region, orientationType=GLOBAL,
                                axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                localCsys=None)

    region = p_core.Set(cells=p_core.cells, name='Set-All')
    p_core.SectionAssignment(region=region, sectionName='Section-Core')
    p_core.MaterialOrientation(region=region, orientationType=GLOBAL,
                                axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                localCsys=None)

    region = p_outer.Set(faces=p_outer.faces, name='Set-All')
    p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_outer.MaterialOrientation(region=region, orientationType=GLOBAL,
                                axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                localCsys=None)

    print("Tier 1: Healthy baseline assigned to all parts")

    # ---- Tier 2: Void override (openings) ----
    if openings:
        # Inner Skin
        opening_faces_inner = [f for f in p_inner.faces
                               if _is_point_in_any_opening(
                                   f.pointOn[0][0], f.pointOn[0][1],
                                   f.pointOn[0][2], openings)]
        if opening_faces_inner:
            pts = tuple((f.pointOn[0],) for f in opening_faces_inner)
            face_seq = p_inner.faces.findAt(*pts)
            void_reg = p_inner.Set(faces=face_seq, name='Set-Opening')
            p_inner.SectionAssignment(region=void_reg, sectionName='Section-Void-Shell')
        print("  InnerSkin: %d void faces (openings)" % len(opening_faces_inner))

        # Core
        opening_cells = [c for c in p_core.cells
                         if _is_point_in_any_opening(
                             c.pointOn[0][0], c.pointOn[0][1],
                             c.pointOn[0][2], openings)]
        if opening_cells:
            pts = tuple((c.pointOn[0],) for c in opening_cells)
            cell_seq = p_core.cells.findAt(*pts)
            void_reg = p_core.Set(cells=cell_seq, name='Set-Opening')
            p_core.SectionAssignment(region=void_reg, sectionName='Section-Void-Solid')
        print("  Core: %d void cells (openings)" % len(opening_cells))

        # Outer Skin
        opening_faces_outer = [f for f in p_outer.faces
                               if _is_point_in_any_opening(
                                   f.pointOn[0][0], f.pointOn[0][1],
                                   f.pointOn[0][2], openings, CORE_T)]
        if opening_faces_outer:
            pts = tuple((f.pointOn[0],) for f in opening_faces_outer)
            face_seq = p_outer.faces.findAt(*pts)
            void_reg = p_outer.Set(faces=face_seq, name='Set-Opening-Outer')
            p_outer.SectionAssignment(region=void_reg, sectionName='Section-Void-Shell')
        print("  OuterSkin: %d void faces (openings)" % len(opening_faces_outer))

    # ---- Tier 3: Defect override ----
    if not defect_params:
        return

    defect_type = defect_params.get('defect_type', 'debonding')

    # Outer skin defects
    outer_skin_defects = ('debonding', 'impact', 'delamination',
                          'thermal_progression', 'acoustic_fatigue')
    if defect_type in outer_skin_defects:
        section_map = {
            'debonding': 'Section-CFRP-Debonded',
            'impact': 'Section-CFRP-Impact',
            'delamination': 'Section-CFRP-Delaminated',
            'thermal_progression': 'Section-CFRP-ThermalDamaged',
            'acoustic_fatigue': 'Section-CFRP-AcousticFatigued',
        }
        section_name = section_map[defect_type]
        defect_faces = [f for f in p_outer.faces
                        if is_face_in_defect_zone(f, defect_params)
                        and not _is_point_in_any_opening(
                            f.pointOn[0][0], f.pointOn[0][1],
                            f.pointOn[0][2], openings or [], CORE_T)]
        if defect_faces:
            pts = tuple((f.pointOn[0],) for f in defect_faces)
            face_seq = p_outer.faces.findAt(*pts)
            region_d = p_outer.Set(faces=face_seq, name='Set-DefectZone-Skin')
            p_outer.SectionAssignment(region=region_d, sectionName=section_name)
            p_outer.MaterialOrientation(region=region_d, orientationType=GLOBAL,
                                        axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                        localCsys=None)
            print("  Tier 3: %s -> %d outer skin faces -> %s" % (
                defect_type, len(defect_faces), section_name))
        else:
            print("  Warning: no outer skin faces found in defect zone")

    # Inner skin defects
    if defect_type == 'inner_debond':
        defect_faces_inner = [f for f in p_inner.faces
                              if is_face_in_defect_zone(f, defect_params)
                              and not _is_point_in_any_opening(
                                  f.pointOn[0][0], f.pointOn[0][1],
                                  f.pointOn[0][2], openings or [])]
        if defect_faces_inner:
            pts = tuple((f.pointOn[0],) for f in defect_faces_inner)
            face_seq = p_inner.faces.findAt(*pts)
            region_d = p_inner.Set(faces=face_seq, name='Set-DefectZone-InnerSkin')
            p_inner.SectionAssignment(region=region_d, sectionName='Section-CFRP-InnerDebonded')
            p_inner.MaterialOrientation(region=region_d, orientationType=GLOBAL,
                                        axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                        localCsys=None)
            print("  Tier 3: inner_debond -> %d inner skin faces" % len(defect_faces_inner))
        else:
            print("  Warning: no inner skin faces found in defect zone")

    # Core defects (fod, impact)
    if defect_type in ('fod', 'impact'):
        section_name = ('Section-Core-FOD' if defect_type == 'fod'
                        else 'Section-Core-Crushed')
        defect_cells = [c for c in p_core.cells
                        if is_cell_in_defect_zone(c, defect_params, openings)]
        if defect_cells:
            pts = tuple((c.pointOn[0],) for c in defect_cells)
            cell_seq = p_core.cells.findAt(*pts)
            region_d = p_core.Set(cells=cell_seq, name='Set-DefectZone-Core')
            p_core.SectionAssignment(region=region_d, sectionName=section_name)
            print("  Tier 3: %s -> %d core cells -> %s" % (
                defect_type, len(defect_cells), section_name))
        else:
            print("  Warning: no core cells found in defect zone")


# ==============================================================================
# RING FRAMES
# ==============================================================================

def create_ring_frame_parts(model, z_positions):
    """Create ring frame parts as shell revolve arcs (1/6 sector)."""
    frame_parts = []
    for i, z_pos in enumerate(z_positions):
        name = 'Part-Frame-%d' % i
        s = model.ConstrainedSketch(
            name='profile_frame_%d' % i, sheetSize=20000.0)
        s.setPrimaryObject(option=STANDALONE)
        s.ConstructionLine(point1=(0.0, -100.0),
                           point2=(0.0, TOTAL_HEIGHT + 1000.0))

        r_at_z = get_radius_at_z(z_pos)
        r_inner = r_at_z - RING_FRAME_HEIGHT
        if r_inner < 10.0:
            r_inner = 10.0

        s.Line(point1=(r_inner, z_pos), point2=(r_at_z, z_pos))

        p = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        p.BaseShellRevolve(sketch=s, angle=60.0, flipRevolveDirection=OFF)

        region = p.Set(faces=p.faces, name='Set-All')
        p.SectionAssignment(region=region, sectionName='Section-Frame')

        frame_parts.append(p)

    print("Ring frames created: %d at z = %s" % (
        len(frame_parts),
        ', '.join(['%.0f' % z for z in z_positions])))
    return frame_parts


# ==============================================================================
# ASSEMBLY
# ==============================================================================

def create_assembly(model, p_inner, p_core, p_outer, frame_parts):
    """Instance all parts into the assembly."""
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)

    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)

    frame_instances = []
    for i, fp in enumerate(frame_parts):
        inst = a.Instance(name='Part-Frame-%d-1' % i, part=fp, dependent=OFF)
        frame_instances.append(inst)

    print("Assembly: 3 base parts + %d ring frames" % len(frame_instances))
    return a, inst_inner, inst_core, inst_outer, frame_instances


# ==============================================================================
# TIE CONSTRAINTS
# ==============================================================================

def create_tie_constraints(model, assembly,
                           inst_inner, inst_core, inst_outer,
                           frame_instances):
    """Create Tie constraints: InnerSkin<->Core, Core<->OuterSkin, Frames<->InnerSkin."""
    surf_inner = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')
    surf_outer = assembly.Surface(
        side1Faces=inst_outer.faces, name='Surf-OuterSkin')

    # Core face classification
    core_inner_pts = []
    core_outer_pts = []
    for f in inst_core.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0]**2 + pt[2]**2)
        r_inner = get_radius_at_z(pt[1])
        r_outer = r_inner + CORE_T
        if abs(r - r_inner) < CORE_T * 0.3:
            core_inner_pts.append((pt,))
        elif abs(r - r_outer) < CORE_T * 0.3:
            core_outer_pts.append((pt,))

    if core_inner_pts:
        core_inner_seq = inst_core.faces.findAt(*core_inner_pts)
        surf_core_inner = assembly.Surface(
            side1Faces=core_inner_seq, name='Surf-Core-Inner')
        model.Tie(name='Tie-InnerSkin-Core', main=surf_core_inner,
                  secondary=surf_inner,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("  Tie: InnerSkin <-> Core (inner), %d core faces" % len(core_inner_pts))

    if core_outer_pts:
        core_outer_seq = inst_core.faces.findAt(*core_outer_pts)
        surf_core_outer = assembly.Surface(
            side1Faces=core_outer_seq, name='Surf-Core-Outer')
        model.Tie(name='Tie-Core-OuterSkin', main=surf_core_outer,
                  secondary=surf_outer,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("  Tie: Core (outer) <-> OuterSkin, %d core faces" % len(core_outer_pts))

    for i, inst_frame in enumerate(frame_instances):
        if len(inst_frame.faces) == 0:
            continue
        surf_f = assembly.Surface(
            side1Faces=inst_frame.faces, name='Surf-Frame-%d' % i)
        model.Tie(name='Tie-Frame-%d' % i, main=surf_inner,
                  secondary=surf_f,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
    if frame_instances:
        print("  Tie: %d ring frames <-> InnerSkin" % len(frame_instances))


# ==============================================================================
# BOUNDARY CONDITIONS
# ==============================================================================

def apply_boundary_conditions(model, assembly,
                               inst_inner, inst_core, inst_outer,
                               frame_instances):
    """Fix bottom (y=0) for all instances."""
    r_box = RADIUS + CORE_T + 500.0
    set_kwargs = {}

    edge_seq = None
    for inst in [inst_inner, inst_outer]:
        try:
            edges = inst.edges.getByBoundingBox(
                xMin=-r_box, xMax=r_box,
                yMin=-0.1, yMax=0.1,
                zMin=-r_box, zMax=r_box)
            if len(edges) > 0:
                edge_seq = edges if edge_seq is None else edge_seq + edges
        except Exception as e:
            print("  BC edge warning (%s): %s" % (inst.name, str(e)[:60]))
    if edge_seq is not None and len(edge_seq) > 0:
        set_kwargs['edges'] = edge_seq

    core_bot_pts = []
    for f in inst_core.faces:
        try:
            pt = f.pointOn[0]
            if abs(pt[1]) < 1.0:
                core_bot_pts.append((pt,))
        except:
            pass
    if core_bot_pts:
        face_seq = inst_core.faces.findAt(*core_bot_pts)
        if len(face_seq) > 0:
            set_kwargs['faces'] = face_seq

    if set_kwargs:
        bot_set = assembly.Set(name='BC_Bottom', **set_kwargs)
        model.DisplacementBC(name='Fix_Bottom', createStepName='Initial',
                             region=bot_set, u1=0, u2=0, u3=0)
        print("BC: Fixed at y=0 (%s)" % ', '.join(
            '%s=%d' % (k, len(v)) for k, v in set_kwargs.items()))
    else:
        print("Warning: No BC geometry found at y=0")


# ==============================================================================
# MESH
# ==============================================================================

def generate_mesh(assembly, inst_inner, inst_core, inst_outer,
                  frame_instances, openings, defect_params,
                  global_seed, opening_seed, defect_seed, frame_seed):
    """Multi-resolution mesh with 4 seed tiers."""
    all_skin_insts = (inst_inner, inst_core, inst_outer)

    # 1. Global seed
    assembly.seedPartInstance(regions=all_skin_insts, size=global_seed,
                              deviationFactor=0.1)

    # 2. Frame seed
    for inst in frame_instances:
        assembly.seedPartInstance(regions=(inst,), size=frame_seed,
                                  deviationFactor=0.1)

    # 3. Opening local refinement
    for opening in openings:
        z_c = opening['z_center']
        r_half = opening['diameter'] / 2.0
        margin = max(100.0, r_half * 0.3)
        z1 = max(1.0, z_c - r_half - margin)
        z2 = min(TOTAL_HEIGHT - 1.0, z_c + r_half + margin)
        r_box = RADIUS + CORE_T + 200
        for inst in all_skin_insts:
            try:
                edges = inst.edges.getByBoundingBox(
                    xMin=-r_box, xMax=r_box,
                    yMin=z1, yMax=z2,
                    zMin=-r_box, zMax=r_box)
                if len(edges) > 0:
                    assembly.seedEdgeBySize(edges=edges, size=opening_seed,
                                            constraint=FINER)
            except Exception as e:
                print("  Mesh refinement warning (%s): %s" % (
                    opening['name'], str(e)[:60]))
        print("  Local mesh: %s zone z=[%.0f, %.0f] seed=%.0f mm" % (
            opening['name'], z1, z2, opening_seed))

    # 4. Defect local refinement
    if defect_params:
        z_c = defect_params['z_center']
        r_def = defect_params['radius']
        margin = 150.0
        z1 = max(1.0, z_c - r_def - margin)
        z2 = min(TOTAL_HEIGHT - 1.0, z_c + r_def + margin)
        r_box = RADIUS + CORE_T + 200
        for inst in all_skin_insts:
            try:
                edges = inst.edges.getByBoundingBox(
                    xMin=-r_box, xMax=r_box,
                    yMin=z1, yMax=z2,
                    zMin=-r_box, zMax=r_box)
                if len(edges) > 0:
                    assembly.seedEdgeBySize(edges=edges, size=defect_seed,
                                            constraint=FINER)
            except Exception as e:
                print("  Defect mesh warning: %s" % str(e)[:60])
        print("  Local mesh: defect zone z=[%.0f, %.0f] seed=%.0f mm" % (
            z1, z2, defect_seed))

        # 4b. Defect boundary refinement (partition edge vicinity)
        #     Finer seed near the 4 partition planes to prevent sliver elements
        boundary_seed = max(defect_seed * BOUNDARY_SEED_RATIO, 3.0)
        bm = BOUNDARY_MARGIN
        theta_deg = defect_params['theta_deg']

        # Z-boundary planes: z_c - r_def and z_c + r_def
        for z_plane in [z_c - r_def, z_c + r_def]:
            z1b = max(1.0, z_plane - bm)
            z2b = min(TOTAL_HEIGHT - 1.0, z_plane + bm)
            for inst in all_skin_insts:
                try:
                    edges = inst.edges.getByBoundingBox(
                        xMin=-r_box, xMax=r_box,
                        yMin=z1b, yMax=z2b,
                        zMin=-r_box, zMax=r_box)
                    if len(edges) > 0:
                        assembly.seedEdgeBySize(
                            edges=edges, size=boundary_seed,
                            constraint=FINER)
                except Exception:
                    pass

        # Theta-boundary planes: theta +/- d_theta
        r_local = get_radius_at_z(z_c) + CORE_T
        if r_local < 1.0:
            r_local = RADIUS
        d_theta = min(r_def / r_local, math.radians(30.0))
        theta_rad = math.radians(theta_deg)
        z1_def = max(1.0, z_c - r_def - bm)
        z2_def = min(TOTAL_HEIGHT - 1.0, z_c + r_def + bm)
        for t_plane in [theta_rad - d_theta, theta_rad + d_theta]:
            x_mid = r_local * math.cos(t_plane)
            z_mid = r_local * math.sin(t_plane)
            for inst in all_skin_insts:
                try:
                    edges = inst.edges.getByBoundingBox(
                        xMin=x_mid - bm, xMax=x_mid + bm,
                        yMin=z1_def, yMax=z2_def,
                        zMin=z_mid - bm, zMax=z_mid + bm)
                    if len(edges) > 0:
                        assembly.seedEdgeBySize(
                            edges=edges, size=boundary_seed,
                            constraint=FINER)
                except Exception:
                    pass

        print("  Local mesh: defect boundary seed=%.1f mm (margin=%.0f mm)" % (
            boundary_seed, bm))

    # 5. Generate mesh
    regions_to_mesh = list(all_skin_insts) + list(frame_instances)
    try:
        assembly.generateMesh(regions=tuple(regions_to_mesh))
    except Exception as e:
        print("  Bulk mesh failed, trying per-instance: %s" % str(e)[:80])
        for inst in regions_to_mesh:
            try:
                assembly.generateMesh(regions=(inst,))
            except Exception as e2:
                print("    Mesh failed for %s: %s" % (inst.name, str(e2)[:60]))

    # Report
    total_nodes = 0
    total_elems = 0
    for inst in regions_to_mesh:
        total_nodes += len(inst.nodes)
        total_elems += len(inst.elements)
    print("Mesh: %d nodes, %d elements (global=%.0f mm)" % (
        total_nodes, total_elems, global_seed))


# ==============================================================================
# THERMAL LOAD
# ==============================================================================

def apply_thermal_load(model, assembly,
                       inst_inner, inst_core, inst_outer,
                       frame_instances):
    """Apply thermal gradient: outer 120C, inner 20C, core 70C, frames 20C."""
    try:
        reg_inner = inst_inner.sets['Set-All']
        reg_outer = inst_outer.sets['Set-All']
        reg_core = inst_core.sets['Set-All']

        # Initial temperature
        model.Temperature(name='Temp_IC_Inner', createStepName='Initial',
                          region=reg_inner, distributionType=UNIFORM,
                          magnitudes=(TEMP_INITIAL,))
        model.Temperature(name='Temp_IC_Outer', createStepName='Initial',
                          region=reg_outer, distributionType=UNIFORM,
                          magnitudes=(TEMP_INITIAL,))
        model.Temperature(name='Temp_IC_Core', createStepName='Initial',
                          region=reg_core, distributionType=UNIFORM,
                          magnitudes=(TEMP_INITIAL,))

        # Step-1 thermal load
        # NOTE: Outer skin temp is applied as z-dependent profile by
        # patch_inp_thermal.py AFTER INP generation. Do NOT set uniform
        # outer temperature here — it would override the gradient.
        model.Temperature(name='Temp_Inner_Step1', createStepName='Step-1',
                          region=reg_inner, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_INNER,))
        model.Temperature(name='Temp_Core_Step1', createStepName='Step-1',
                          region=reg_core, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_CORE,))

        # Ring frame temperatures (attached to inner skin -> inner temperature)
        for i, inst_frame in enumerate(frame_instances):
            if 'Set-All' in inst_frame.sets:
                reg_f = inst_frame.sets['Set-All']
                model.Temperature(name='Temp_IC_Frame_%d' % i,
                                  createStepName='Initial',
                                  region=reg_f, distributionType=UNIFORM,
                                  magnitudes=(TEMP_INITIAL,))
                model.Temperature(name='Temp_Frame_%d_Step1' % i,
                                  createStepName='Step-1',
                                  region=reg_f, distributionType=UNIFORM,
                                  magnitudes=(TEMP_FINAL_INNER,))

        print("Thermal load: outer=%.0fC, inner=%.0fC, core=%.0fC, frames=%.0fC" % (
            TEMP_FINAL_OUTER, TEMP_FINAL_INNER, TEMP_FINAL_CORE, TEMP_FINAL_INNER))
    except Exception as e:
        print("Warning: Thermal load via API: %s" % str(e)[:80])
        print("  -> patch_inp_thermal.py will inject thermal load into INP")


# ==============================================================================
# JOB
# ==============================================================================

def create_and_run_job(model, job_name, no_run=False, project_root=None):
    """Create Abaqus job, write INP, optionally patch and run."""
    mdb.Job(name=job_name, model='Model-1', type=ANALYSIS, resultsFormat=ODB,
            numCpus=4, numDomains=4, multiprocessingMode=DEFAULT)
    mdb.saveAs(pathName=job_name + '.cae')

    print("Writing INP: %s.inp" % job_name)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    inp_path = os.path.abspath(job_name + '.inp')

    if no_run:
        print("INP written. Skipping execution (--no_run)")
        return

    # Patch INP for thermal load
    patch_script = None
    proj_root = (project_root or os.environ.get('PROJECT_ROOT')
                 or os.environ.get('PAYLOAD2026_ROOT'))
    if proj_root:
        patch_script = os.path.join(proj_root, 'scripts', 'patch_inp_thermal.py')
    if not patch_script or not os.path.exists(patch_script):
        inp_dir = os.path.dirname(inp_path)
        for root in [inp_dir, os.path.dirname(inp_dir),
                     os.path.dirname(os.path.dirname(inp_dir))]:
            if root:
                p = os.path.join(root, 'scripts', 'patch_inp_thermal.py')
                if os.path.exists(p):
                    patch_script = p
                    break
    if patch_script and os.path.exists(patch_script):
        import subprocess
        r = subprocess.call([sys.executable, patch_script, inp_path],
                            cwd=os.path.dirname(inp_path))
        if r == 0:
            print("INP patched for thermal load")

    print("Running job '%s'..." % job_name)
    import subprocess
    cwd = os.path.dirname(inp_path) or '.'
    r = subprocess.call(
        ['abaqus', 'job=' + job_name, 'input=' + job_name + '.inp', 'cpus=4'],
        cwd=cwd)
    if r == 0:
        print("Job COMPLETED: %s.odb" % job_name)
    else:
        print("Job FAILED (exit code %d)" % r)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def generate_realistic_dataset(job_name, defect_params=None, phase=2,
                                global_seed=None, defect_seed=None,
                                opening_seed=None, frame_seed=None,
                                no_run=False, project_root=None):
    """
    Main entry point: realistic fairing model with defect insertion.

    Args:
        job_name: Abaqus job name
        defect_params: dict with defect_type, theta_deg, z_center, radius, ...
        phase: 1 (access door only) or 2 (all openings)
        global_seed: mesh size (mm), default 25
        defect_seed: local mesh for defect zone (mm), default 10
        opening_seed: local mesh around openings (mm), default 10
        frame_seed: ring frame mesh (mm), default 15
        no_run: if True, only write INP
        project_root: project root for patch script
    """
    g_seed = global_seed if global_seed is not None else GLOBAL_SEED
    d_seed = defect_seed if defect_seed is not None else DEFECT_SEED
    o_seed = opening_seed if opening_seed is not None else OPENING_SEED
    f_seed = frame_seed if frame_seed is not None else FRAME_SEED

    defect_type = defect_params.get('defect_type', 'debonding') if defect_params else None

    print("=" * 70)
    print("REALISTIC FAIRING DATASET — Phase %d" % phase)
    print("  Mesh: global=%.0f, defect=%.0f, opening=%.0f, frame=%.0f mm" % (
        g_seed, d_seed, o_seed, f_seed))
    if defect_params:
        print("  Defect: %s | theta=%.1f z=%.0f r=%.0f" % (
            defect_type, defect_params['theta_deg'],
            defect_params['z_center'], defect_params['radius']))
    else:
        print("  Mode: healthy (no defect)")
    print("=" * 70)

    Mdb()
    model = mdb.models['Model-1']

    # Select openings
    openings = OPENINGS_PHASE2 if phase >= 2 else OPENINGS_PHASE1
    print("Openings: %d (%s)" % (
        len(openings), ', '.join([o['name'] for o in openings])))

    # Ring frame positions (avoid collision with openings)
    frame_z_positions = []
    for z_pos in RING_FRAME_Z_POSITIONS:
        skip = False
        for op in openings:
            if abs(z_pos - op['z_center']) < op['diameter'] / 2.0 + 50:
                print("  Skipping frame at z=%.0f (collides with %s)" % (
                    z_pos, op['name']))
                skip = True
                break
        if not skip:
            frame_z_positions.append(z_pos)
    print("Ring frames: %d at z = %s" % (
        len(frame_z_positions),
        ', '.join(['%.0f' % z for z in frame_z_positions])))

    # 1. Materials & Sections
    create_materials(model)
    create_sections(model)
    if defect_params:
        create_defect_materials(model, defect_params)
        create_defect_sections(model, defect_params)

    # 2. Base geometry
    p_inner, p_core, p_outer = create_base_parts(model)

    # 3. Partition openings (at part level, before assembly)
    if openings:
        partition_all_openings(p_inner, p_core, p_outer, openings)

    # 4. Partition defect zone
    if defect_params:
        parts_to_partition = [
            ('shell', p_inner),
            ('solid', p_core),
            ('shell', p_outer),
        ]
        partition_defect_zone(parts_to_partition, defect_params)

    # 5. 3-tier section assignment
    assign_sections_3tier(p_inner, p_core, p_outer, openings, defect_params)

    # 6. Ring frames
    frame_parts = create_ring_frame_parts(model, frame_z_positions)

    # 7. Assembly
    a, inst_inner, inst_core, inst_outer, frame_insts = \
        create_assembly(model, p_inner, p_core, p_outer, frame_parts)

    # 8. Tie constraints
    create_tie_constraints(model, a, inst_inner, inst_core, inst_outer,
                           frame_insts)

    # 9. Boundary conditions
    apply_boundary_conditions(model, a, inst_inner, inst_core, inst_outer,
                              frame_insts)

    # 10. Step + Field Output
    model.StaticStep(name='Step-1', previous='Initial')
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'U', 'RF', 'TEMP', 'LE'))

    # 11. Mesh
    generate_mesh(a, inst_inner, inst_core, inst_outer, frame_insts,
                  openings, defect_params,
                  g_seed, o_seed, d_seed, f_seed)

    # 12. Thermal load
    apply_thermal_load(model, a, inst_inner, inst_core, inst_outer,
                       frame_insts)

    # 13. Job
    create_and_run_job(model, job_name, no_run, project_root)

    print("=" * 70)
    print("DONE: %s (Phase %d, %s)" % (
        job_name, phase, defect_type or 'healthy'))
    print("=" * 70)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate realistic H3 fairing FEM with defect insertion')
    parser.add_argument('--job', type=str, default='Job-Realistic-Dataset',
                        help='Job name')
    parser.add_argument('--job_name', type=str, default=None,
                        help='Alias for --job (run_batch compatibility)')
    parser.add_argument('--defect', type=str, default=None,
                        help='JSON string or path to JSON with defect params')
    parser.add_argument('--param_file', type=str, default=None,
                        help='Path to JSON file with defect params (run_batch compat)')
    parser.add_argument('--phase', type=int, default=2, choices=[1, 2],
                        help='Phase: 1=access door only, 2=all openings')
    parser.add_argument('--global_seed', type=float, default=None,
                        help='Global mesh seed (mm), default 25')
    parser.add_argument('--defect_seed', type=float, default=None,
                        help='Defect zone mesh seed (mm), default 10')
    parser.add_argument('--opening_seed', type=float, default=None,
                        help='Opening zone mesh seed (mm), default 10')
    parser.add_argument('--no_run', action='store_true',
                        help='Write INP only, do not run solver')
    parser.add_argument('--project_root', type=str, default=None,
                        help='Project root for patch script')

    args, _ = parser.parse_known_args()

    job_name = args.job_name if args.job_name is not None else args.job

    defect_data = None
    if args.param_file:
        if os.path.exists(args.param_file):
            with open(args.param_file, 'r') as f:
                defect_data = json.load(f)
        else:
            print("Param file not found: %s" % args.param_file)
    elif args.defect:
        if os.path.exists(args.defect):
            with open(args.defect, 'r') as f:
                defect_data = json.load(f)
        else:
            try:
                defect_data = json.loads(args.defect)
            except:
                print("Invalid defect JSON or file path")

    if defect_data and isinstance(defect_data, dict):
        if 'defect_params' in defect_data:
            defect_data = defect_data['defect_params']

    project_root = (args.project_root or os.environ.get('PROJECT_ROOT')
                    or os.environ.get('PAYLOAD2026_ROOT'))

    generate_realistic_dataset(
        job_name=job_name,
        defect_params=defect_data,
        phase=args.phase,
        global_seed=args.global_seed,
        defect_seed=args.defect_seed,
        opening_seed=args.opening_seed,
        no_run=getattr(args, 'no_run', False),
        project_root=project_root,
    )
