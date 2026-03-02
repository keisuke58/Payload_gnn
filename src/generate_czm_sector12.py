# -*- coding: utf-8 -*-
# generate_czm_sector12.py
# CZM 1/12 sector (30 deg) model — no openings, minimal for solver verification.
#
# 5-part sandwich structure:
#   InnerSkin --Tie-- AdhesiveInner --Tie-- Core --Tie-- AdhesiveOuter --Tie-- OuterSkin
#                    (COH3D8 0.2mm)                     (COH3D8 0.2mm)
#
# Based on generate_cohesive_fairing.py (60 deg, ~770K nodes) but reduced to
# 30 deg sector (~65K nodes) with symmetry BCs for solver convergence testing.
#
# Usage:
#   abaqus cae noGUI=generate_czm_sector12.py -- --job Job-CZM-S12 --seed 35 --healthy --no_run
#   abaqus cae noGUI=generate_czm_sector12.py -- --job Job-CZM-S12 --param_file params.json

import sys
import os
import math
import json
import argparse
from abaqus import *
from abaqusConstants import *
from caeModules import *
from mesh import ElemType
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
CORE_T = 38.0         # mm (Al honeycomb core, total allocation)
ADHESIVE_T = 0.2      # mm (default cohesive layer thickness)
ADH_R_MIN = 200.0     # mm — adhesive ogive cutoff radius (avoids axis-degenerate geometry for SWEEP)

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
CFRP_CTE_11 = -0.3e-6        # /C  fiber direction (negative for T1000G carbon)
CFRP_CTE_22 = 28e-6          # /C  transverse direction

# Aluminum Honeycomb Core (5052)
E_CORE_1 = 1000.0     # through-thickness (radial)
E_CORE_2 = 1.0        # circumferential
E_CORE_3 = 1.0        # axial
NU_CORE_12 = 0.01
NU_CORE_13 = 0.01
NU_CORE_23 = 0.01
G_CORE_12 = 400.0     # R-theta shear
G_CORE_13 = 240.0     # R-axial shear
G_CORE_23 = 1.0       # theta-axial shear
CORE_DENSITY = 50e-12
CORE_CTE = 23e-6

# ==============================================================================
# DEFAULT CZM PARAMETERS (CFRP/Epoxy adhesive)
# ==============================================================================
DEFAULT_CZM_PARAMS = {
    'Kn': 1e5,       # Normal stiffness (N/mm^3)
    'Ks': 5e4,       # Shear stiffness (N/mm^3)
    'tn': 50.0,      # Normal strength (MPa)
    'ts': 40.0,      # Shear strength (MPa)
    'GIc': 0.3,      # Mode I fracture energy (N/mm)
    'GIIc': 1.0,     # Mode II fracture energy (N/mm)
    'BK_eta': 2.284,  # BK power law exponent
}

# ==============================================================================
# THERMAL LOAD
# ==============================================================================
TEMP_INITIAL = 20.0
TEMP_FINAL_OUTER = 120.0
TEMP_FINAL_INNER = 20.0
TEMP_FINAL_CORE = 70.0
TEMP_FINAL_ADHESIVE = 45.0  # midway between core and skin

# ==============================================================================
# MECHANICAL LOAD PARAMETERS (H3 Max-Q flight condition)
# ==============================================================================
AERO_PRESSURE_ZONES = [
    (0, 1000, 0.0),
    (1000, 3000, 0.0),
    (3000, H_BARREL, 0.0),
]

DIFF_PRESSURE = 0.005  # 5 kPa
LAUNCH_G = 3.0
G_ACCEL = 9810.0

# ==============================================================================
# SECTOR GEOMETRY
# ==============================================================================
SECTOR_ANGLE = 30.0   # degrees (1/12 of full fairing)

# ==============================================================================
# MESH
# ==============================================================================
GLOBAL_SEED = 25.0
DEFECT_SEED = 10.0
OPENING_SEED = 10.0
FRAME_SEED = 15.0
BOUNDARY_SEED_RATIO = 0.4
BOUNDARY_MARGIN = 30.0

# ==============================================================================
# SOLVER MEMORY (65K nodes → ~5 GB sufficient)
# ==============================================================================
SOLVER_MEMORY = '8 gb'

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
RING_FRAME_HEIGHT = 50.0
RING_FRAME_THICKNESS = 3.0

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
    """Check if point (x,y,z) is inside the circular defect zone."""
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
    """Checks if a solid cell's centroid is within the defect zone."""
    if not defect_params:
        return False
    pt = cell.pointOn[0]
    if openings and _is_point_in_any_opening(pt[0], pt[1], pt[2], openings):
        return False
    return _point_in_defect_zone(pt[0], pt[1], pt[2], defect_params)


# ==============================================================================
# MATERIALS AND SECTIONS (Base)
# ==============================================================================

def create_materials(model):
    """Define CFRP, Honeycomb, frame, and void materials."""
    mat = model.Material(name='CFRP_T1000G')
    mat.Elastic(type=LAMINA, table=((E1, E2, NU12, G12, G13, G23),))
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Expansion(table=((CFRP_CTE_11, CFRP_CTE_22, 0.0),))

    mat = model.Material(name='AL_HONEYCOMB')
    mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
        E_CORE_1, E_CORE_2, E_CORE_3,
        NU_CORE_12, NU_CORE_13, NU_CORE_23,
        G_CORE_12, G_CORE_13, G_CORE_23
    ),))
    mat.Density(table=((CORE_DENSITY,),))
    mat.Expansion(table=((CORE_CTE,),))

    mat = model.Material(name='CFRP_FRAME')
    mat.Elastic(type=ISOTROPIC, table=((70000.0, 0.3),))
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Expansion(table=((2e-6,),))  # quasi-isotropic effective CTE for frame

    mat = model.Material(name='VOID')
    mat.Elastic(type=ISOTROPIC, table=((1.0, 0.3),))
    mat.Density(table=((1e-20,),))
    mat.Expansion(table=((0.0,),))


def create_sections(model):
    """Create shell and solid sections for healthy, void, and frame."""
    angles = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
    entries = [section.SectionLayer(
        thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_T1000G')
        for ang in angles]
    model.CompositeShellSection(
        name='Section-CFRP-Skin', preIntegrate=OFF,
        idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
        thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        temperature=GRADIENT, integrationRule=SIMPSON)

    model.HomogeneousSolidSection(
        name='Section-Core', material='AL_HONEYCOMB', thickness=None)

    model.HomogeneousShellSection(
        name='Section-Frame', material='CFRP_FRAME',
        thickness=RING_FRAME_THICKNESS)

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
        mat.Expansion(table=((CFRP_CTE_11, CFRP_CTE_22, 0.0),))

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
        mat_skin.Expansion(table=((CFRP_CTE_11, CFRP_CTE_22, 0.0),))

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
        mat.Expansion(table=((CFRP_CTE_11, CFRP_CTE_22, 0.0),))

    elif defect_type == 'inner_debond':
        mat = model.Material(name='CFRP_INNER_DEBONDED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.01, E2 * 0.01, NU12,
            G12 * 0.01, G13 * 0.01, G23 * 0.01
        ),))
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Expansion(table=((CFRP_CTE_11, CFRP_CTE_22, 0.0),))

    elif defect_type == 'thermal_progression':
        mat = model.Material(name='CFRP_THERMAL_DAMAGED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.05, E2 * 0.05, NU12,
            G12 * 0.05, G13 * 0.05, G23 * 0.05
        ),))
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Expansion(table=((5e-6, 40e-6, 0.0),))  # thermally damaged: fiber CTE +, matrix CTE increased

    elif defect_type == 'acoustic_fatigue':
        sev = defect_params.get('fatigue_severity', 0.35)
        mat = model.Material(name='CFRP_ACOUSTIC_FATIGUED')
        mat.Elastic(type=LAMINA, table=((
            E1 * (0.2 + 0.5 * (1 - sev)), E2 * sev, NU12,
            G12 * sev, G13 * sev, G23 * sev
        ),))
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Expansion(table=((CFRP_CTE_11, CFRP_CTE_22, 0.0),))

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
# ADHESIVE CZM MATERIALS AND SECTIONS
# ==============================================================================

def create_adhesive_materials(model, czm_params, adh_t):
    """Create CZM adhesive materials: healthy, fully damaged, partially damaged, void."""
    Kn = czm_params['Kn']
    Ks = czm_params['Ks']
    tn = czm_params['tn']
    ts = czm_params['ts']
    GIc = czm_params['GIc']
    GIIc = czm_params['GIIc']
    BK_eta = czm_params['BK_eta']

    # Healthy adhesive with Traction-Separation + damage
    mat = model.Material(name='Mat-Adhesive')
    mat.Elastic(type=TRACTION, table=((Kn, Ks, Ks),))
    mat.Density(table=((1200e-12,),))  # epoxy ~1200 kg/m^3
    mat.Expansion(table=((40e-6,),))   # epoxy CTE
    mat.MaxsDamageInitiation(table=((tn, ts, ts),))
    mat.maxsDamageInitiation.DamageEvolution(
        type=ENERGY, mixedModeBehavior=BK, power=BK_eta,
        table=((GIc, GIIc, GIIc),))
    print("  Mat-Adhesive: Kn=%.0e Ks=%.0e tn=%.0f ts=%.0f GIc=%.2f GIIc=%.2f" % (
        Kn, Ks, tn, ts, GIc, GIIc))

    # Fully damaged adhesive (pre-debonded: K/1000)
    mat_d = model.Material(name='Mat-Adhesive-Damaged')
    mat_d.Elastic(type=TRACTION, table=((Kn / 1000.0, Ks / 1000.0, Ks / 1000.0),))
    mat_d.Density(table=((1200e-12,),))
    mat_d.Expansion(table=((40e-6,),))
    print("  Mat-Adhesive-Damaged: K/1000 (pre-debonded)")

    # Partially damaged adhesive (impact: K/10)
    mat_p = model.Material(name='Mat-Adhesive-PartialDamage')
    mat_p.Elastic(type=TRACTION, table=((Kn / 10.0, Ks / 10.0, Ks / 10.0),))
    mat_p.Density(table=((1200e-12,),))
    mat_p.Expansion(table=((40e-6,),))
    mat_p.MaxsDamageInitiation(table=((tn * 0.3, ts * 0.3, ts * 0.3),))
    mat_p.maxsDamageInitiation.DamageEvolution(
        type=ENERGY, mixedModeBehavior=BK, power=BK_eta,
        table=((GIc * 0.3, GIIc * 0.3, GIIc * 0.3),))
    print("  Mat-Adhesive-PartialDamage: K/10, strength*0.3 (impact)")

    # Void adhesive (for opening regions, negligible stiffness)
    mat_v = model.Material(name='Mat-Adhesive-Void')
    mat_v.Elastic(type=TRACTION, table=((1.0, 0.5, 0.5),))
    mat_v.Density(table=((1e-20,),))
    mat_v.Expansion(table=((0.0,),))
    print("  Mat-Adhesive-Void: negligible stiffness (openings)")


def create_adhesive_sections(model, adh_t):
    """Create CohesiveSection definitions for adhesive layers."""
    model.CohesiveSection(
        name='Section-Adhesive-Healthy',
        material='Mat-Adhesive',
        response=TRACTION_SEPARATION,
        initialThicknessType=SPECIFY,
        initialThickness=adh_t)

    model.CohesiveSection(
        name='Section-Adhesive-Damaged',
        material='Mat-Adhesive-Damaged',
        response=TRACTION_SEPARATION,
        initialThicknessType=SPECIFY,
        initialThickness=adh_t)

    model.CohesiveSection(
        name='Section-Adhesive-PartialDamage',
        material='Mat-Adhesive-PartialDamage',
        response=TRACTION_SEPARATION,
        initialThicknessType=SPECIFY,
        initialThickness=adh_t)

    model.CohesiveSection(
        name='Section-Adhesive-Void',
        material='Mat-Adhesive-Void',
        response=TRACTION_SEPARATION,
        initialThicknessType=SPECIFY,
        initialThickness=adh_t)

    print("Adhesive sections created (Healthy, Damaged, PartialDamage, Void)")


# ==============================================================================
# GEOMETRY: 5-PART SANDWICH WITH ADHESIVE LAYERS
# ==============================================================================

def _create_solid_revolve_part(model, name, r_inner, r_outer,
                               rho_inner, rho_outer, r_min=0.0):
    """Create a solid revolve part with barrel line + ogive arc profile.

    Used for Core and both adhesive layers.

    When r_min > 0, the ogive arcs are truncated at radius r_min to avoid
    degenerate geometry at the revolve axis (r=0). This enables SWEEP
    meshing for thin cohesive layers.
    """
    s = model.ConstrainedSketch(name='profile_%s' % name, sheetSize=20000.0)
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -100.0),
                       point2=(0.0, TOTAL_HEIGHT + 1000.0))

    if r_min <= 0.0:
        # Full profile extending to revolve axis (for Core)
        z_tip_inner = H_BARREL + math.sqrt(max(0, rho_inner**2 - OGIVE_XC**2))
        z_tip_outer = H_BARREL + math.sqrt(max(0, rho_outer**2 - OGIVE_XC**2))

        s.Line(point1=(r_inner, 0.0), point2=(r_inner, H_BARREL))
        s.ArcByCenterEnds(
            center=(OGIVE_XC, H_BARREL),
            point1=(r_inner, H_BARREL),
            point2=(0.0, z_tip_inner),
            direction=COUNTERCLOCKWISE)
        s.Line(point1=(0.0, z_tip_inner), point2=(0.0, z_tip_outer))
        s.ArcByCenterEnds(
            center=(OGIVE_XC, H_BARREL),
            point1=(0.0, z_tip_outer),
            point2=(r_outer, H_BARREL),
            direction=CLOCKWISE)
        s.Line(point1=(r_outer, H_BARREL), point2=(r_outer, 0.0))
        s.Line(point1=(r_outer, 0.0), point2=(r_inner, 0.0))
    else:
        # Truncated ogive: stop arcs at r_min (for adhesive layers)
        # r = OGIVE_XC + sqrt(rho^2 - (z - H_BARREL)^2)
        # => z_cut = H_BARREL + sqrt(rho^2 - (r_min - OGIVE_XC)^2)
        dx_inner = r_min - OGIVE_XC
        dz_inner = math.sqrt(max(0, rho_inner**2 - dx_inner**2))
        z_cut_inner = H_BARREL + dz_inner

        dx_outer = r_min - OGIVE_XC
        dz_outer = math.sqrt(max(0, rho_outer**2 - dx_outer**2))
        z_cut_outer = H_BARREL + dz_outer

        # Inner barrel
        s.Line(point1=(r_inner, 0.0), point2=(r_inner, H_BARREL))
        # Inner ogive arc (stops at r_min)
        s.ArcByCenterEnds(
            center=(OGIVE_XC, H_BARREL),
            point1=(r_inner, H_BARREL),
            point2=(r_min, z_cut_inner),
            direction=COUNTERCLOCKWISE)
        # Cap at r_min
        s.Line(point1=(r_min, z_cut_inner), point2=(r_min, z_cut_outer))
        # Outer ogive arc (from r_min back to barrel)
        s.ArcByCenterEnds(
            center=(OGIVE_XC, H_BARREL),
            point1=(r_min, z_cut_outer),
            point2=(r_outer, H_BARREL),
            direction=CLOCKWISE)
        # Outer barrel
        s.Line(point1=(r_outer, H_BARREL), point2=(r_outer, 0.0))
        # Bottom
        s.Line(point1=(r_outer, 0.0), point2=(r_inner, 0.0))

        print("  %s: ogive truncated at r_min=%.0f mm "
              "(z_cut=%.1f..%.1f)" % (name, r_min, z_cut_inner, z_cut_outer))

    p = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseSolidRevolve(sketch=s, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)
    return p


def create_base_parts_with_adhesive(model, adh_t):
    """Create 5-part sandwich: InnerSkin, AdhesiveInner, Core, AdhesiveOuter, OuterSkin.

    Returns: (p_inner, p_adh_inner, p_core, p_adh_outer, p_outer)
    """
    effective_core_t = CORE_T - 2 * adh_t

    # Radii in barrel section
    r_inner_skin = RADIUS
    r_adh_inner_outer = RADIUS + adh_t
    r_core_inner = RADIUS + adh_t
    r_core_outer = RADIUS + CORE_T - adh_t
    r_adh_outer_inner = RADIUS + CORE_T - adh_t
    r_adh_outer_outer = RADIUS + CORE_T
    r_outer_skin = RADIUS + CORE_T

    # Ogive radii (offset from ogive center)
    rho_inner = OGIVE_RHO
    rho_adh_inner_outer = OGIVE_RHO + adh_t
    rho_core_inner = OGIVE_RHO + adh_t
    rho_core_outer = OGIVE_RHO + CORE_T - adh_t
    rho_adh_outer_inner = OGIVE_RHO + CORE_T - adh_t
    rho_adh_outer_outer = OGIVE_RHO + CORE_T

    z_tip_outer = H_BARREL + math.sqrt(rho_adh_outer_outer**2 - OGIVE_XC**2)

    # --- Inner Skin (Shell) ---
    s1 = model.ConstrainedSketch(name='profile_inner', sheetSize=20000.0)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0),
                        point2=(0.0, TOTAL_HEIGHT + 1000.0))
    s1.Line(point1=(r_inner_skin, 0.0), point2=(r_inner_skin, H_BARREL))
    s1.Line(point1=(r_inner_skin, H_BARREL), point2=(0.0, TOTAL_HEIGHT))

    p_inner = model.Part(name='Part-InnerSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)

    # --- Adhesive Inner (Solid, COH3D8) ---
    p_adh_inner = _create_solid_revolve_part(
        model, 'Part-AdhesiveInner',
        r_inner_skin, r_adh_inner_outer,
        rho_inner, rho_adh_inner_outer,
        r_min=ADH_R_MIN)

    # --- Core (Solid, reduced thickness) ---
    # Truncate ogive at r_min to match adhesive layers;
    # avoids floating core nodes at tip where adhesive doesn't extend.
    p_core = _create_solid_revolve_part(
        model, 'Part-Core',
        r_core_inner, r_core_outer,
        rho_core_inner, rho_core_outer,
        r_min=ADH_R_MIN)

    # --- Adhesive Outer (Solid, COH3D8) ---
    p_adh_outer = _create_solid_revolve_part(
        model, 'Part-AdhesiveOuter',
        r_adh_outer_inner, r_adh_outer_outer,
        rho_adh_outer_inner, rho_adh_outer_outer,
        r_min=ADH_R_MIN)

    # --- Outer Skin (Shell) ---
    s3 = model.ConstrainedSketch(name='profile_outer', sheetSize=20000.0)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0),
                        point2=(0.0, TOTAL_HEIGHT + 1000.0))
    s3.Line(point1=(r_outer_skin, 0.0), point2=(r_outer_skin, H_BARREL))
    s3.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(r_outer_skin, H_BARREL),
        point2=(0.0, z_tip_outer),
        direction=COUNTERCLOCKWISE)

    p_outer = model.Part(name='Part-OuterSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)

    print("5-part sandwich created:")
    print("  InnerSkin(shell) | AdhesiveInner(%.1fmm) | Core(%.1fmm) | "
          "AdhesiveOuter(%.1fmm) | OuterSkin(shell)" % (
              adh_t, effective_core_t, adh_t))
    return p_inner, p_adh_inner, p_core, p_adh_outer, p_outer


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


def partition_all_openings_with_adhesive(p_inner, p_adh_inner, p_core,
                                          p_adh_outer, p_outer, openings):
    """Partition skins and core for each opening.

    Adhesive layers are NOT partitioned for openings to preserve simple
    geometry for SWEEP meshing (COH3D8). The 0.2mm adhesive in opening regions
    has negligible stiffness effect since adjacent skin/core are voided.
    """
    print("Partitioning openings (%d total) across 3 parts "
          "(adhesive excluded)..." % len(openings))
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
        dp_z1 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
                                                 offset=z_c - r_def)
        dp_z2 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
                                                 offset=z_c + r_def)
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
                print("  Part partition warning (%s): %s" % (
                    geom_type, str(e)[:80]))

    print("Defect zone partitioned: z=%.0f theta=%.1f r=%.0f" % (
        z_c, theta_deg, r_def))


# ==============================================================================
# SECTION ASSIGNMENT: 3-TIER (HEALTHY -> VOID -> DEFECT) for Skins/Core
# ==============================================================================

def assign_sections_3tier(p_inner, p_core, p_outer, openings, defect_params):
    """3-tier section assignment for skins and core (same as base)."""
    # ---- Tier 1: Healthy baseline ----
    region = p_inner.Set(faces=p_inner.faces, name='Set-All')
    p_inner.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_inner.MaterialOrientation(region=region, orientationType=GLOBAL,
                                axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                localCsys=None)

    region = p_core.Set(cells=p_core.cells, name='Set-All')
    p_core.SectionAssignment(region=region, sectionName='Section-Core')
    cyl_datum = p_core.DatumCsysByThreePoints(
        name='CylCS-Core', coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(0.0, 0.0, 1.0),
        point2=(1.0, 0.0, 0.0))
    p_core.MaterialOrientation(
        region=region, orientationType=SYSTEM,
        axis=AXIS_3, additionalRotationType=ROTATION_NONE,
        localCsys=p_core.datums[cyl_datum.id])

    region = p_outer.Set(faces=p_outer.faces, name='Set-All')
    p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_outer.MaterialOrientation(region=region, orientationType=GLOBAL,
                                axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                localCsys=None)

    print("Tier 1: Healthy baseline assigned to skins/core")

    # ---- Tier 2: Void override (openings) ----
    if openings:
        opening_faces_inner = [f for f in p_inner.faces
                               if _is_point_in_any_opening(
                                   f.pointOn[0][0], f.pointOn[0][1],
                                   f.pointOn[0][2], openings)]
        if opening_faces_inner:
            pts = tuple((f.pointOn[0],) for f in opening_faces_inner)
            face_seq = p_inner.faces.findAt(*pts)
            void_reg = p_inner.Set(faces=face_seq, name='Set-Opening')
            p_inner.SectionAssignment(region=void_reg,
                                      sectionName='Section-Void-Shell')
        print("  InnerSkin: %d void faces (openings)" % len(opening_faces_inner))

        opening_cells = [c for c in p_core.cells
                         if _is_point_in_any_opening(
                             c.pointOn[0][0], c.pointOn[0][1],
                             c.pointOn[0][2], openings)]
        if opening_cells:
            pts = tuple((c.pointOn[0],) for c in opening_cells)
            cell_seq = p_core.cells.findAt(*pts)
            void_reg = p_core.Set(cells=cell_seq, name='Set-Opening')
            p_core.SectionAssignment(region=void_reg,
                                     sectionName='Section-Void-Solid')
        print("  Core: %d void cells (openings)" % len(opening_cells))

        opening_faces_outer = [f for f in p_outer.faces
                               if _is_point_in_any_opening(
                                   f.pointOn[0][0], f.pointOn[0][1],
                                   f.pointOn[0][2], openings, CORE_T)]
        if opening_faces_outer:
            pts = tuple((f.pointOn[0],) for f in opening_faces_outer)
            face_seq = p_outer.faces.findAt(*pts)
            void_reg = p_outer.Set(faces=face_seq, name='Set-Opening-Outer')
            p_outer.SectionAssignment(region=void_reg,
                                      sectionName='Section-Void-Shell')
        print("  OuterSkin: %d void faces (openings)" % len(opening_faces_outer))

    # ---- Tier 3: Defect override ----
    if not defect_params:
        return

    defect_type = defect_params.get('defect_type', 'debonding')

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
                                        axis=AXIS_3,
                                        additionalRotationType=ROTATION_NONE,
                                        localCsys=None)
            print("  Tier 3: %s -> %d outer skin faces -> %s" % (
                defect_type, len(defect_faces), section_name))
        else:
            print("  Warning: no outer skin faces found in defect zone")

    if defect_type == 'inner_debond':
        defect_faces_inner = [f for f in p_inner.faces
                              if is_face_in_defect_zone(f, defect_params)
                              and not _is_point_in_any_opening(
                                  f.pointOn[0][0], f.pointOn[0][1],
                                  f.pointOn[0][2], openings or [])]
        if defect_faces_inner:
            pts = tuple((f.pointOn[0],) for f in defect_faces_inner)
            face_seq = p_inner.faces.findAt(*pts)
            region_d = p_inner.Set(faces=face_seq,
                                   name='Set-DefectZone-InnerSkin')
            p_inner.SectionAssignment(region=region_d,
                                      sectionName='Section-CFRP-InnerDebonded')
            p_inner.MaterialOrientation(region=region_d, orientationType=GLOBAL,
                                        axis=AXIS_3,
                                        additionalRotationType=ROTATION_NONE,
                                        localCsys=None)
            print("  Tier 3: inner_debond -> %d inner skin faces" % (
                len(defect_faces_inner)))
        else:
            print("  Warning: no inner skin faces found in defect zone")

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
# ADHESIVE SECTION ASSIGNMENT
# ==============================================================================

def assign_adhesive_sections(p_adh_inner, p_adh_outer, defect_params,
                              openings, adh_t):
    """Assign cohesive sections to adhesive layers.

    Tier 1: All cells -> Healthy
    Tier 2: Opening cells -> Void
    Tier 3: Defect cells -> Damaged (type-dependent)
    """
    # ---- Tier 1: Healthy baseline ----
    for p_adh, label in [(p_adh_inner, 'AdhesiveInner'),
                          (p_adh_outer, 'AdhesiveOuter')]:
        region = p_adh.Set(cells=p_adh.cells, name='Set-All')
        p_adh.SectionAssignment(region=region,
                                sectionName='Section-Adhesive-Healthy')
    print("Adhesive Tier 1: Healthy baseline assigned to both layers")
    # Note: Adhesive is NOT voided for openings. The 0.2mm thin layer has
    # negligible stiffness contribution since adjacent skin/core are voided.
    # Opening nodes are pinned in post-mesh BCs.

    # ---- Tier 3: Defect override ----
    if not defect_params:
        return

    defect_type = defect_params.get('defect_type', 'debonding')

    # debonding -> outer adhesive fully damaged
    if defect_type == 'debonding':
        defect_cells = [c for c in p_adh_outer.cells
                        if is_cell_in_defect_zone(c, defect_params, openings)]
        if defect_cells:
            pts = tuple((c.pointOn[0],) for c in defect_cells)
            cell_seq = p_adh_outer.cells.findAt(*pts)
            region_d = p_adh_outer.Set(cells=cell_seq,
                                       name='Set-DefectZone-Adhesive')
            p_adh_outer.SectionAssignment(
                region=region_d, sectionName='Section-Adhesive-Damaged')
            print("  Adhesive Tier 3: debonding -> %d outer adhesive cells "
                  "-> Damaged" % len(defect_cells))
        else:
            print("  Warning: no outer adhesive cells in defect zone")

    # inner_debond -> inner adhesive fully damaged
    elif defect_type == 'inner_debond':
        defect_cells = [c for c in p_adh_inner.cells
                        if is_cell_in_defect_zone(c, defect_params, openings)]
        if defect_cells:
            pts = tuple((c.pointOn[0],) for c in defect_cells)
            cell_seq = p_adh_inner.cells.findAt(*pts)
            region_d = p_adh_inner.Set(cells=cell_seq,
                                       name='Set-DefectZone-Adhesive')
            p_adh_inner.SectionAssignment(
                region=region_d, sectionName='Section-Adhesive-Damaged')
            print("  Adhesive Tier 3: inner_debond -> %d inner adhesive cells "
                  "-> Damaged" % len(defect_cells))
        else:
            print("  Warning: no inner adhesive cells in defect zone")

    # impact -> outer adhesive partially damaged
    elif defect_type == 'impact':
        defect_cells = [c for c in p_adh_outer.cells
                        if is_cell_in_defect_zone(c, defect_params, openings)]
        if defect_cells:
            pts = tuple((c.pointOn[0],) for c in defect_cells)
            cell_seq = p_adh_outer.cells.findAt(*pts)
            region_d = p_adh_outer.Set(cells=cell_seq,
                                       name='Set-DefectZone-Adhesive')
            p_adh_outer.SectionAssignment(
                region=region_d, sectionName='Section-Adhesive-PartialDamage')
            print("  Adhesive Tier 3: impact -> %d outer adhesive cells "
                  "-> PartialDamage" % len(defect_cells))
        else:
            print("  Warning: no outer adhesive cells in defect zone")

    # fod, delamination, thermal_progression, acoustic_fatigue
    # -> adhesive stays healthy (defect is in core or skin only)
    else:
        print("  Adhesive Tier 3: %s -> adhesive stays healthy" % defect_type)


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
        p.BaseShellRevolve(sketch=s, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)

        region = p.Set(faces=p.faces, name='Set-All')
        p.SectionAssignment(region=region, sectionName='Section-Frame')

        frame_parts.append(p)

    print("Ring frames created: %d at z = %s" % (
        len(frame_parts),
        ', '.join(['%.0f' % z for z in z_positions])))
    return frame_parts


# ==============================================================================
# ASSEMBLY (5 parts + frames)
# ==============================================================================

def create_assembly_with_adhesive(model, p_inner, p_adh_inner, p_core,
                                   p_adh_outer, p_outer, frame_parts):
    """Instance all 5 parts + frames into the assembly."""
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)

    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_adh_inner = a.Instance(name='Part-AdhesiveInner-1',
                                part=p_adh_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_adh_outer = a.Instance(name='Part-AdhesiveOuter-1',
                                part=p_adh_outer, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)

    frame_instances = []
    for i, fp in enumerate(frame_parts):
        inst = a.Instance(name='Part-Frame-%d-1' % i, part=fp, dependent=OFF)
        frame_instances.append(inst)

    print("Assembly: 5 base parts + %d ring frames" % len(frame_instances))
    return (a, inst_inner, inst_adh_inner, inst_core,
            inst_adh_outer, inst_outer, frame_instances)


# ==============================================================================
# TIE CONSTRAINTS: 4 SKIN-ADHESIVE-CORE + FRAME TIES
# ==============================================================================

def _classify_solid_faces(inst, r_inner_expected_fn, r_outer_expected_fn,
                          tol_fn):
    """Classify solid instance faces as inner, outer, or edge.

    Uses nearest-match: each face is classified based on which expected
    surface (inner or outer) its centroid radius is closer to.
    Edge faces (y~0 bottom / ogive cap top) are skipped only when
    their centroid is far from both expected surfaces.

    Args:
        inst: Assembly instance
        r_inner_expected_fn: function(y) -> expected inner radius at z=y
        r_outer_expected_fn: function(y) -> expected outer radius at z=y
        tol_fn: function() -> tolerance (used as edge-face skip threshold)

    Returns: (inner_pts, outer_pts) - lists of (pointOn,) tuples
    """
    inner_pts = []
    outer_pts = []
    tol = tol_fn()
    # Use generous threshold for edge-face detection: 10x tolerance or
    # half the expected thickness, whichever is larger.
    for f in inst.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0]**2 + pt[2]**2)
        r_inner = r_inner_expected_fn(pt[1])
        r_outer = r_outer_expected_fn(pt[1])
        d_inner = abs(r - r_inner)
        d_outer = abs(r - r_outer)
        thickness = abs(r_outer - r_inner)
        # Skip true edge faces whose centroid is far from both surfaces
        # (e.g. bottom annulus at y=0, top cap at ogive truncation)
        edge_tol = max(tol * 10, thickness * 0.8)
        if min(d_inner, d_outer) > edge_tol:
            continue
        # Nearest-match: classify by which expected surface is closer
        if d_inner <= d_outer:
            inner_pts.append((pt,))
        else:
            outer_pts.append((pt,))
    return inner_pts, outer_pts


def create_cohesive_connections(model, assembly,
                                inst_inner, inst_adh_inner, inst_core,
                                inst_adh_outer, inst_outer,
                                frame_instances, adh_t):
    """Create Tie constraints for 5-part sandwich + frame connections.

    Tie pairs:
      1. InnerSkin <-> AdhesiveInner (inner face)
      2. AdhesiveInner (outer) <-> Core (inner)
      3. Core (outer) <-> AdhesiveOuter (inner)
      4. AdhesiveOuter (outer) <-> OuterSkin
      5. Frame <-> InnerSkin
    """
    effective_core_t = CORE_T - 2 * adh_t
    adh_tol = max(adh_t * 0.45, 0.05)
    core_tol = effective_core_t * 0.3

    # --- Classify AdhesiveInner faces ---
    ai_inner_pts, ai_outer_pts = _classify_solid_faces(
        inst_adh_inner,
        r_inner_expected_fn=lambda y: get_radius_at_z(y),
        r_outer_expected_fn=lambda y: get_radius_at_z(y) + adh_t,
        tol_fn=lambda: adh_tol)

    # --- Classify Core faces ---
    core_inner_pts, core_outer_pts = _classify_solid_faces(
        inst_core,
        r_inner_expected_fn=lambda y: get_radius_at_z(y) + adh_t,
        r_outer_expected_fn=lambda y: get_radius_at_z(y) + CORE_T - adh_t,
        tol_fn=lambda: core_tol)

    # --- Classify AdhesiveOuter faces ---
    ao_inner_pts, ao_outer_pts = _classify_solid_faces(
        inst_adh_outer,
        r_inner_expected_fn=lambda y: get_radius_at_z(y) + CORE_T - adh_t,
        r_outer_expected_fn=lambda y: get_radius_at_z(y) + CORE_T,
        tol_fn=lambda: adh_tol)

    # --- Skin surfaces ---
    surf_inner_skin = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')
    surf_outer_skin = assembly.Surface(
        side1Faces=inst_outer.faces, name='Surf-OuterSkin')

    # --- Tie 1: InnerSkin <-> AdhesiveInner (inner face) ---
    if ai_inner_pts:
        ai_inner_seq = inst_adh_inner.faces.findAt(*ai_inner_pts)
        surf_ai_inner = assembly.Surface(
            side1Faces=ai_inner_seq, name='Surf-AdhInner-Inner')
        model.Tie(name='Tie-InnerSkin-AdhInner',
                  main=surf_ai_inner, secondary=surf_inner_skin,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("  Tie 1: InnerSkin <-> AdhesiveInner(inner), %d faces" % (
            len(ai_inner_pts)))
    else:
        print("  WARNING: No AdhesiveInner inner faces found for Tie 1")

    # --- Tie 2: AdhesiveInner (outer) <-> Core (inner) ---
    if ai_outer_pts and core_inner_pts:
        ai_outer_seq = inst_adh_inner.faces.findAt(*ai_outer_pts)
        core_inner_seq = inst_core.faces.findAt(*core_inner_pts)
        surf_ai_outer = assembly.Surface(
            side1Faces=ai_outer_seq, name='Surf-AdhInner-Outer')
        surf_core_inner = assembly.Surface(
            side1Faces=core_inner_seq, name='Surf-Core-Inner')
        model.Tie(name='Tie-AdhInner-Core',
                  main=surf_core_inner, secondary=surf_ai_outer,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("  Tie 2: AdhesiveInner(outer) <-> Core(inner), "
              "%d + %d faces" % (len(ai_outer_pts), len(core_inner_pts)))
    else:
        print("  WARNING: Missing faces for Tie 2 (ai_outer=%d, core_inner=%d)" % (
            len(ai_outer_pts), len(core_inner_pts)))

    # --- Tie 3: Core (outer) <-> AdhesiveOuter (inner) ---
    if core_outer_pts and ao_inner_pts:
        core_outer_seq = inst_core.faces.findAt(*core_outer_pts)
        ao_inner_seq = inst_adh_outer.faces.findAt(*ao_inner_pts)
        surf_core_outer = assembly.Surface(
            side1Faces=core_outer_seq, name='Surf-Core-Outer')
        surf_ao_inner = assembly.Surface(
            side1Faces=ao_inner_seq, name='Surf-AdhOuter-Inner')
        model.Tie(name='Tie-Core-AdhOuter',
                  main=surf_core_outer, secondary=surf_ao_inner,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("  Tie 3: Core(outer) <-> AdhesiveOuter(inner), "
              "%d + %d faces" % (len(core_outer_pts), len(ao_inner_pts)))
    else:
        print("  WARNING: Missing faces for Tie 3 (core_outer=%d, ao_inner=%d)" % (
            len(core_outer_pts), len(ao_inner_pts)))

    # --- Tie 4: AdhesiveOuter (outer) <-> OuterSkin ---
    if ao_outer_pts:
        ao_outer_seq = inst_adh_outer.faces.findAt(*ao_outer_pts)
        surf_ao_outer = assembly.Surface(
            side1Faces=ao_outer_seq, name='Surf-AdhOuter-Outer')
        model.Tie(name='Tie-AdhOuter-OuterSkin',
                  main=surf_ao_outer, secondary=surf_outer_skin,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("  Tie 4: AdhesiveOuter(outer) <-> OuterSkin, %d faces" % (
            len(ao_outer_pts)))
    else:
        print("  WARNING: No AdhesiveOuter outer faces found for Tie 4")

    # --- Tie 5+: Frames <-> InnerSkin ---
    for i, inst_frame in enumerate(frame_instances):
        if len(inst_frame.faces) == 0:
            continue
        surf_f = assembly.Surface(
            side1Faces=inst_frame.faces, name='Surf-Frame-%d' % i)
        model.Tie(name='Tie-Frame-%d' % i, main=surf_inner_skin,
                  secondary=surf_f,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
    if frame_instances:
        print("  Tie 5+: %d ring frames <-> InnerSkin" % len(frame_instances))


# ==============================================================================
# BOUNDARY CONDITIONS
# ==============================================================================

def apply_boundary_conditions_with_adhesive(model, assembly,
                                             inst_inner, inst_adh_inner,
                                             inst_core, inst_adh_outer,
                                             inst_outer, frame_instances):
    """Fix bottom (y=0) for all 5 instances."""
    r_box = RADIUS + CORE_T + 500.0
    set_kwargs = {}

    # Shell edges at y=0
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

    # Solid faces at y=0 (core + adhesive layers)
    face_seq_bot = None
    for inst in [inst_core, inst_adh_inner, inst_adh_outer]:
        bot_pts = []
        for f in inst.faces:
            try:
                pt = f.pointOn[0]
                if abs(pt[1]) < 1.0:
                    bot_pts.append((pt,))
            except:
                pass
        if bot_pts:
            fseq = inst.faces.findAt(*bot_pts)
            if len(fseq) > 0:
                face_seq_bot = (fseq if face_seq_bot is None
                                else face_seq_bot + fseq)

    if face_seq_bot is not None and len(face_seq_bot) > 0:
        set_kwargs['faces'] = face_seq_bot

    if set_kwargs:
        bot_set = assembly.Set(name='BC_Bottom', **set_kwargs)
        model.DisplacementBC(name='Fix_Bottom', createStepName='Initial',
                             region=bot_set, u1=0, u2=0, u3=0)
        print("BC: Fixed at y=0 (%s)" % ', '.join(
            '%s=%d' % (k, len(v)) for k, v in set_kwargs.items()))
    else:
        print("Warning: No BC geometry found at y=0")


def apply_symmetry_bcs(model, assembly,
                       inst_inner, inst_adh_inner, inst_core,
                       inst_adh_outer, inst_outer, frame_instances):
    """Apply symmetry boundary conditions on sector cut faces.

    theta=0 face (Z=0 plane): U3=0 (ZSYMM)
    theta=SECTOR_ANGLE face: normal displacement=0 via local CSYS
    """
    all_insts = [inst_inner, inst_adh_inner, inst_core,
                 inst_adh_outer, inst_outer] + list(frame_instances)
    r_box = RADIUS + CORE_T + 500.0
    tol = 1.0  # mm

    # --- theta=0 face: Z=0 plane, constrain U3=0 ---
    sym0_edges = None
    sym0_faces = None
    for inst in all_insts:
        # Shell edges (skins, frames)
        try:
            edges = inst.edges.getByBoundingBox(
                xMin=-1.0, xMax=r_box,
                yMin=-0.1, yMax=TOTAL_HEIGHT + 100.0,
                zMin=-tol, zMax=tol)
            if len(edges) > 0:
                sym0_edges = edges if sym0_edges is None else sym0_edges + edges
        except Exception:
            pass
        # Solid faces (core, adhesive)
        try:
            face_pts = []
            for f in inst.faces:
                pt = f.pointOn[0]
                if abs(pt[2]) < tol and pt[0] > 0:
                    face_pts.append((pt,))
            if face_pts:
                fseq = inst.faces.findAt(*face_pts)
                if len(fseq) > 0:
                    sym0_faces = fseq if sym0_faces is None else sym0_faces + fseq
        except Exception:
            pass

    set_kwargs_0 = {}
    if sym0_edges is not None and len(sym0_edges) > 0:
        set_kwargs_0['edges'] = sym0_edges
    if sym0_faces is not None and len(sym0_faces) > 0:
        set_kwargs_0['faces'] = sym0_faces
    if set_kwargs_0:
        sym0_set = assembly.Set(name='BC_Sym_Theta0', **set_kwargs_0)
        model.DisplacementBC(name='Sym_Theta0', createStepName='Initial',
                             region=sym0_set, u3=0)
        print("BC: Symmetry theta=0 (Z=0 plane, U3=0): %s" % ', '.join(
            '%s=%d' % (k, len(v)) for k, v in set_kwargs_0.items()))
    else:
        print("Warning: No geometry found for theta=0 symmetry BC")

    # --- theta=SECTOR_ANGLE face: constrain normal displacement ---
    # Create a local cylindrical CSYS rotated to the cut plane
    theta_rad = math.radians(SECTOR_ANGLE)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    # Point on the cut plane at theta=SECTOR_ANGLE
    pt_on_plane = (cos_t * RADIUS, 0.0, sin_t * RADIUS)
    # Plane normal: (-sin(theta), 0, cos(theta))
    # We need a local CSYS where one axis aligns with this normal.
    # Use DatumCsysByThreePoints: define a Cartesian CSYS where
    # local-3 (Z) direction is the plane normal.
    # origin at (0,0,0); point1 along local-1 (tangent); point2 along local-2 (Y axis)
    # local-3 = cross(local-1, local-2) should be normal to cut plane
    # Normal to cut plane at theta: n = (-sin(theta), 0, cos(theta))
    # Choose local-1 along Y (axial), local-2 = n x local-1 to get local-3 = n
    # Actually, for DisplacementBC with localCsys:
    # We want u_normal = 0 on the cut face. If we define CSYS such that
    # one of u1/u2/u3 corresponds to the face normal, we can fix it.
    #
    # Approach: create a Cartesian CSYS where:
    #   local-1 (X') = tangential to cut face in horizontal plane = (cos_t, 0, sin_t)
    #   local-2 (Y') = axial = (0, 1, 0)
    #   local-3 (Z') = normal to cut face = (-sin_t, 0, cos_t)
    # Then constrain U3 (local) = 0.
    datum_csys = assembly.DatumCsysByThreePoints(
        name='CSYS-SymTheta',
        coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0),
        point1=(cos_t, 0.0, sin_t),       # local X direction (tangent)
        point2=(0.0, 1.0, 0.0))           # defines XY plane → local Z = normal

    # Collect faces/edges on theta=SECTOR_ANGLE cut plane
    sym_a_edges = None
    sym_a_faces = None
    for inst in all_insts:
        # Shell edges
        try:
            edge_pts = []
            for e in inst.edges:
                pt = e.pointOn[0]
                r_pt = math.sqrt(pt[0]**2 + pt[2]**2)
                if r_pt < 1.0:
                    continue
                theta_pt = math.atan2(pt[2], pt[0])
                if abs(theta_pt - theta_rad) < 0.02:  # ~1 degree tolerance
                    edge_pts.append((pt,))
            if edge_pts:
                eseq = inst.edges.findAt(*edge_pts)
                if len(eseq) > 0:
                    sym_a_edges = eseq if sym_a_edges is None else sym_a_edges + eseq
        except Exception:
            pass
        # Solid faces
        try:
            face_pts = []
            for f in inst.faces:
                pt = f.pointOn[0]
                r_pt = math.sqrt(pt[0]**2 + pt[2]**2)
                if r_pt < 1.0:
                    continue
                theta_pt = math.atan2(pt[2], pt[0])
                if abs(theta_pt - theta_rad) < 0.02:
                    face_pts.append((pt,))
            if face_pts:
                fseq = inst.faces.findAt(*face_pts)
                if len(fseq) > 0:
                    sym_a_faces = fseq if sym_a_faces is None else sym_a_faces + fseq
        except Exception:
            pass

    set_kwargs_a = {}
    if sym_a_edges is not None and len(sym_a_edges) > 0:
        set_kwargs_a['edges'] = sym_a_edges
    if sym_a_faces is not None and len(sym_a_faces) > 0:
        set_kwargs_a['faces'] = sym_a_faces
    if set_kwargs_a:
        sym_a_set = assembly.Set(name='BC_Sym_ThetaMax', **set_kwargs_a)
        model.DisplacementBC(
            name='Sym_ThetaMax', createStepName='Initial',
            region=sym_a_set, u3=0,
            localCsys=assembly.datums[datum_csys.id])
        print("BC: Symmetry theta=%.0f deg (local U3=0): %s" % (
            SECTOR_ANGLE,
            ', '.join('%s=%d' % (k, len(v)) for k, v in set_kwargs_a.items())))
    else:
        print("Warning: No geometry found for theta=%.0f symmetry BC" %
              SECTOR_ANGLE)


def apply_post_mesh_bcs_with_adhesive(model, assembly,
                                       inst_inner, inst_adh_inner,
                                       inst_core, inst_adh_outer,
                                       inst_outer):
    """Post-mesh BCs: nose tip + VOID opening nodes (including adhesive)."""
    # 1. Nose tip (all instances including core/adhesive truncated at r_min)
    nose_y_min = TOTAL_HEIGHT - 100.0
    nose_nodes = []
    for inst in [inst_inner, inst_adh_inner, inst_core,
                 inst_adh_outer, inst_outer]:
        for node in inst.nodes:
            c = node.coordinates
            r = math.sqrt(c[0]**2 + c[2]**2)
            if c[1] > nose_y_min and r < 150.0:
                nose_nodes.append(inst.nodes[node.label - 1:node.label])
    if nose_nodes:
        combined = nose_nodes[0]
        for ns in nose_nodes[1:]:
            combined = combined + ns
        nose_set = assembly.Set(name='BC_NoseTip', nodes=combined)
        model.DisplacementBC(name='Fix_NoseTip', createStepName='Initial',
                             region=nose_set, u1=0, u2=0, u3=0)
        print("BC: Nose tip pinned (y>%.0f, r<150mm): %d nodes" % (
            nose_y_min, len(nose_nodes)))

    # 1b. Ogive truncation cap: solid nodes near r_min not tied to skins
    cap_r_max = ADH_R_MIN + 5.0  # small tolerance around truncation radius
    cap_nodes = []
    for inst in [inst_adh_inner, inst_core, inst_adh_outer]:
        for node in inst.nodes:
            c = node.coordinates
            r = math.sqrt(c[0]**2 + c[2]**2)
            if r < cap_r_max and c[1] > H_BARREL:
                cap_nodes.append(inst.nodes[node.label - 1:node.label])
    if cap_nodes:
        combined_cap = cap_nodes[0]
        for ns in cap_nodes[1:]:
            combined_cap = combined_cap + ns
        cap_set = assembly.Set(name='BC_OgiveCap', nodes=combined_cap)
        model.DisplacementBC(name='Fix_OgiveCap', createStepName='Initial',
                             region=cap_set, u1=0, u2=0, u3=0)
        print("BC: Ogive cap pinned (r<%.0f, y>%.0f): %d nodes" % (
            cap_r_max, H_BARREL, len(cap_nodes)))

    # 2. VOID opening nodes: skins/core use Set-Opening,
    #    adhesive layers use coordinate-based detection (not partitioned).
    void_node_seqs = []
    for inst in [inst_inner, inst_core, inst_outer]:
        try:
            void_set = inst.sets['Set-Opening']
            void_node_seqs.append(void_set.nodes)
        except KeyError:
            pass
    # Also try Set-Opening-Outer for outer skin
    try:
        void_set = inst_outer.sets['Set-Opening-Outer']
        void_node_seqs.append(void_set.nodes)
    except KeyError:
        pass
    if void_node_seqs:
        combined = void_node_seqs[0]
        for ns in void_node_seqs[1:]:
            combined = combined + ns
        void_bc_set = assembly.Set(name='BC_VoidOpening', nodes=combined)
        model.DisplacementBC(name='Fix_VoidOpening', createStepName='Initial',
                             region=void_bc_set, u1=0, u2=0, u3=0)
        n_total = sum([len(ns) for ns in void_node_seqs])
        print("BC: VOID opening nodes pinned: %d nodes" % n_total)


# ==============================================================================
# MESH (with adhesive layers)
# ==============================================================================

def _find_inner_face_for_stack(inst, r_inner_fn):
    """Find a face on the inner (bottom) surface of an adhesive layer.

    Used as the reference face for assignStackDirection so that COH3D8
    element node ordering gives positive thickness (inner→outer).
    """
    best_face = None
    best_dist = 1e20
    for f in inst.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0]**2 + pt[2]**2)
        r_expected = r_inner_fn(pt[1])
        d = abs(r - r_expected)
        if d < best_dist:
            best_dist = d
            best_face = f
    return best_face


def mesh_adhesive_parts(assembly, inst_adh_inner, inst_adh_outer,
                        global_seed, adh_t):
    """Mesh adhesive layer instances.

    Strategy:
      1. assignStackDirection (inner face → outer) for correct COH3D8 node order
      2. SWEEP + COH3D8 (ideal for cohesive elements)
      3. FREE TET + C3D10 fallback (model runs, but no CZM behavior)
    """
    # Cap adhesive seed to avoid extreme aspect ratios (thickness=0.2mm)
    adh_seed = min(global_seed, 25.0)

    # Inner-surface radius functions for each layer
    r_inner_fns = {
        inst_adh_inner.name: lambda y: get_radius_at_z(y),
        inst_adh_outer.name: lambda y: get_radius_at_z(y) + CORE_T - adh_t,
    }

    for inst in [inst_adh_inner, inst_adh_outer]:
        assembly.seedPartInstance(regions=(inst,), size=adh_seed,
                                  deviationFactor=0.1)
        meshed = False
        n_cells = len(inst.cells)
        n_faces = len(inst.faces)
        n_edges = len(inst.edges)
        print("  %s: %d cells, %d faces, %d edges, trying SWEEP..." % (
            inst.name, n_cells, n_faces, n_edges))

        # Set stack direction: inner face as bottom → thickness goes outward
        try:
            inner_face = _find_inner_face_for_stack(
                inst, r_inner_fns[inst.name])
            if inner_face is not None:
                assembly.assignStackDirection(
                    referenceRegion=inner_face, cells=inst.cells)
                print("  %s: stack direction set (inner face as bottom)" %
                      inst.name)
        except Exception as e:
            print("  %s: assignStackDirection warning: %s" % (
                inst.name, str(e)[:80]))

        # Attempt 1: SWEEP + COH3D8
        try:
            assembly.setMeshControls(regions=inst.cells,
                                     elemShape=HEX, technique=SWEEP)
            assembly.setElementType(
                regions=(inst.cells,),
                elemTypes=(
                    ElemType(elemCode=COH3D8, elemLibrary=STANDARD),
                    ElemType(elemCode=COH3D6, elemLibrary=STANDARD),
                ))
            assembly.generateMesh(regions=(inst,))
            if len(inst.nodes) > 0:
                meshed = True
                print("  %s: SWEEP COH3D8 OK (%d nodes, %d elements)" % (
                    inst.name, len(inst.nodes), len(inst.elements)))
        except Exception as e:
            print("  %s: SWEEP failed: %s" % (inst.name, str(e)[:80]))

        # Attempt 2: FREE TET + C3D10
        if not meshed:
            try:
                assembly.deleteMesh(regions=(inst,))
            except Exception:
                pass
            try:
                assembly.setMeshControls(regions=inst.cells,
                                         elemShape=TET, technique=FREE)
                assembly.setElementType(
                    regions=(inst.cells,),
                    elemTypes=(
                        ElemType(elemCode=C3D10, elemLibrary=STANDARD),
                    ))
                assembly.generateMesh(regions=(inst,))
                if len(inst.nodes) > 0:
                    meshed = True
                    print("  %s: FREE TET C3D10 (%d nodes, %d elements)" % (
                        inst.name, len(inst.nodes), len(inst.elements)))
                    print("  NOTE: Standard tet elements used "
                          "(CZM not active for this layer)")
                else:
                    print("  %s: FREE TET produced 0 nodes" % inst.name)
            except Exception as e:
                print("  %s: FREE TET failed: %s" % (inst.name, str(e)[:80]))

        if not meshed:
            print("  %s: ALL mesh attempts FAILED" % inst.name)


def generate_mesh_with_adhesive(assembly,
                                 inst_inner, inst_adh_inner, inst_core,
                                 inst_adh_outer, inst_outer,
                                 frame_instances, openings, defect_params,
                                 global_seed, opening_seed, defect_seed,
                                 frame_seed, adh_t):
    """Multi-resolution mesh with adhesive layer meshing first."""
    all_insts_5 = (inst_inner, inst_adh_inner, inst_core,
                   inst_adh_outer, inst_outer)

    # 1. Global seed for all 5 parts
    assembly.seedPartInstance(regions=all_insts_5, size=global_seed,
                              deviationFactor=0.1)

    # 2. Frame seed
    for inst in frame_instances:
        assembly.seedPartInstance(regions=(inst,), size=frame_seed,
                                  deviationFactor=0.1)

    # 3. Opening local refinement (skins + adhesive only, NOT core)
    #    Fine seeds (10mm) on 38mm-thick core solid create too many
    #    through-thickness elements and C3D10 mesher fails silently.
    opening_insts = (inst_inner, inst_adh_inner, inst_adh_outer, inst_outer)
    for opening in openings:
        z_c = opening['z_center']
        r_half = opening['diameter'] / 2.0
        margin = max(100.0, r_half * 0.3)
        z1 = max(1.0, z_c - r_half - margin)
        z2 = min(TOTAL_HEIGHT - 1.0, z_c + r_half + margin)
        r_box = RADIUS + CORE_T + 200
        for inst in opening_insts:
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

    # 4. Defect local refinement (skins + adhesive only, NOT core)
    #    Same reason as step 3: fine seeds on thick solid core break C3D10.
    skin_adh_insts = (inst_inner, inst_adh_inner, inst_adh_outer, inst_outer)
    if defect_params:
        z_c = defect_params['z_center']
        r_def = defect_params['radius']
        margin = 150.0
        z1 = max(1.0, z_c - r_def - margin)
        z2 = min(TOTAL_HEIGHT - 1.0, z_c + r_def + margin)
        r_box = RADIUS + CORE_T + 200
        for inst in skin_adh_insts:
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

        # 4b. Defect boundary refinement
        boundary_seed = max(defect_seed * BOUNDARY_SEED_RATIO, 3.0)
        bm = BOUNDARY_MARGIN
        theta_deg = defect_params['theta_deg']

        for z_plane in [z_c - r_def, z_c + r_def]:
            z1b = max(1.0, z_plane - bm)
            z2b = min(TOTAL_HEIGHT - 1.0, z_plane + bm)
            for inst in skin_adh_insts:
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
            for inst in skin_adh_insts:
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

    # 5. Core mesh (C3D10 free tet)
    effective_core_t = CORE_T - 2 * adh_t
    core_seed = min(global_seed, effective_core_t)
    print("Meshing core (C3D10 free tet, seed=%.0f mm, %d cells)..." % (
        core_seed, len(inst_core.cells)))
    assembly.seedPartInstance(regions=(inst_core,), size=core_seed,
                              deviationFactor=0.1)
    try:
        assembly.setMeshControls(regions=inst_core.cells,
                                 elemShape=TET, technique=FREE)
        assembly.setElementType(
            regions=(inst_core.cells,),
            elemTypes=(ElemType(elemCode=C3D10, elemLibrary=STANDARD),))
        assembly.generateMesh(regions=(inst_core,))
        n_core = len(inst_core.nodes)
        if n_core > 0:
            print("  Core mesh: %d nodes, %d elements" % (
                n_core, len(inst_core.elements)))
        else:
            print("  WARNING: Core free tet produced 0 nodes!")
    except Exception as e:
        print("  Core free tet failed: %s" % str(e)[:120])

    # 6. Adhesive mesh (SWEEP COH3D8)
    print("Meshing adhesive layers...")
    mesh_adhesive_parts(assembly, inst_adh_inner, inst_adh_outer, global_seed,
                        adh_t)

    # 7. Skins + frames
    skin_frame_insts = [inst_inner, inst_outer] + list(frame_instances)
    try:
        assembly.generateMesh(regions=tuple(skin_frame_insts))
    except Exception as e:
        print("  Bulk skin/frame mesh failed, trying per-instance: %s" % (
            str(e)[:80]))
        for inst in skin_frame_insts:
            try:
                assembly.generateMesh(regions=(inst,))
            except Exception as e2:
                print("    Mesh failed for %s: %s" % (
                    inst.name, str(e2)[:60]))

    # Report
    all_insts = list(all_insts_5) + list(frame_instances)
    total_nodes = 0
    total_elems = 0
    for inst in all_insts:
        n = len(inst.nodes)
        e = len(inst.elements)
        total_nodes += n
        total_elems += e
        if n == 0:
            print("  WARNING: %s has 0 nodes!" % inst.name)
    print("Mesh total: %d nodes, %d elements (global=%.0f mm)" % (
        total_nodes, total_elems, global_seed))


# ==============================================================================
# THERMAL LOAD (with adhesive layers)
# ==============================================================================

def apply_thermal_load_with_adhesive(model, assembly,
                                      inst_inner, inst_adh_inner,
                                      inst_core, inst_adh_outer,
                                      inst_outer, frame_instances):
    """Apply thermal gradient including adhesive layers."""
    try:
        reg_inner = inst_inner.sets['Set-All']
        reg_outer = inst_outer.sets['Set-All']
        reg_core = inst_core.sets['Set-All']
        reg_adh_inner = inst_adh_inner.sets['Set-All']
        reg_adh_outer = inst_adh_outer.sets['Set-All']

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
        model.Temperature(name='Temp_IC_AdhInner', createStepName='Initial',
                          region=reg_adh_inner, distributionType=UNIFORM,
                          magnitudes=(TEMP_INITIAL,))
        model.Temperature(name='Temp_IC_AdhOuter', createStepName='Initial',
                          region=reg_adh_outer, distributionType=UNIFORM,
                          magnitudes=(TEMP_INITIAL,))

        # Step-1 thermal load
        model.Temperature(name='Temp_Inner_Step1', createStepName='Step-1',
                          region=reg_inner, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_INNER,))
        model.Temperature(name='Temp_Core_Step1', createStepName='Step-1',
                          region=reg_core, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_CORE,))
        # Adhesive: temperature between adjacent skin and core
        model.Temperature(name='Temp_AdhInner_Step1', createStepName='Step-1',
                          region=reg_adh_inner, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_ADHESIVE,))
        model.Temperature(name='Temp_AdhOuter_Step1', createStepName='Step-1',
                          region=reg_adh_outer, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_ADHESIVE,))

        # Ring frames
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

        print("Thermal load: outer=%.0fC, inner=%.0fC, core=%.0fC, "
              "adhesive=%.0fC, frames=%.0fC" % (
                  TEMP_FINAL_OUTER, TEMP_FINAL_INNER, TEMP_FINAL_CORE,
                  TEMP_FINAL_ADHESIVE, TEMP_FINAL_INNER))
    except Exception as e:
        print("Warning: Thermal load via API: %s" % str(e)[:80])
        print("  -> patch_inp_thermal.py will inject thermal load into INP")


# ==============================================================================
# MECHANICAL LOADS (same as base, operates on skins/core)
# ==============================================================================

def apply_mechanical_loads(model, assembly, inst_inner, inst_core, inst_outer):
    """Apply mechanical loads in Step-2."""
    try:
        outer_faces = inst_outer.sets['Set-All'].faces
        print("  Outer skin: using Set-All (%d faces, VOID excluded)" % (
            len(outer_faces)))
    except KeyError:
        outer_faces = inst_outer.faces
        print("  Outer skin: Set-All not found, using all faces (%d)" % (
            len(outer_faces)))

    zone_face_points = {}
    for face in outer_faces:
        try:
            pt = face.pointOn[0]
            y_pos = pt[1]
            for i, (z_lo, z_hi, p_mpa) in enumerate(AERO_PRESSURE_ZONES):
                if z_lo - 0.1 <= y_pos < z_hi + 0.1:
                    zone_face_points.setdefault(i, []).append(face.pointOn)
                    break
        except Exception:
            pass

    for i, (z_lo, z_hi, p_mpa) in enumerate(AERO_PRESSURE_ZONES):
        if p_mpa < 1e-9:
            continue
        pts = zone_face_points.get(i, [])
        if pts:
            try:
                faces = inst_outer.faces.findAt(*pts)
                if hasattr(faces, '__len__'):
                    n_faces = len(faces)
                else:
                    faces = inst_outer.faces.findAt(pts[0])
                    n_faces = 1
                surf = assembly.Surface(
                    side2Faces=faces,
                    name='Surf-Aero-Zone-%d' % i)
                model.Pressure(
                    name='Aero_Pressure_%d' % i,
                    createStepName='Step-2',
                    region=surf,
                    distributionType=UNIFORM,
                    magnitude=p_mpa)
                print("  Aero pressure zone %d: z=[%.0f,%.0f] P=%.3f MPa "
                      "(%d faces)" % (i, z_lo, z_hi, p_mpa, n_faces))
            except Exception as e:
                print("  Aero pressure zone %d warning: %s" % (
                    i, str(e)[:80]))
        else:
            print("  Aero pressure zone %d: no faces in z=[%.0f,%.0f]" % (
                i, z_lo, z_hi))

    # Differential pressure (barrel only, inner skin)
    try:
        try:
            inner_faces = inst_inner.sets['Set-All'].faces
        except KeyError:
            inner_faces = inst_inner.faces
        barrel_pts = []
        for face in inner_faces:
            try:
                pt = face.pointOn[0]
                if pt[1] < H_BARREL + 0.1:
                    barrel_pts.append(face.pointOn)
            except Exception:
                pass
        if barrel_pts:
            barrel_face_seq = inst_inner.faces.findAt(*barrel_pts)
            surf_inner = assembly.Surface(
                side2Faces=barrel_face_seq,
                name='Surf-DiffPressure')
            model.Pressure(
                name='Diff_Pressure',
                createStepName='Step-2',
                region=surf_inner,
                distributionType=UNIFORM,
                magnitude=DIFF_PRESSURE)
            n_barrel = len(barrel_pts)
            print("  Differential pressure: %.3f MPa (%.1f kPa), "
                  "barrel only (%d faces, y<%.0f)" % (
                      DIFF_PRESSURE, DIFF_PRESSURE * 1000,
                      n_barrel, H_BARREL))
        else:
            print("  Differential pressure: no barrel faces found")
    except Exception as e:
        print("  Diff pressure warning: %s" % str(e)[:80])

    # Launch acceleration
    try:
        model.Gravity(
            name='Launch_Accel',
            createStepName='Step-2',
            comp2=-(LAUNCH_G * G_ACCEL),
            distributionType=UNIFORM,
            field='')
        print("  Launch accel: %.1f G (%.0f mm/s^2, -Y)" % (
            LAUNCH_G, LAUNCH_G * G_ACCEL))
    except Exception as e:
        print("  Gravity warning: %s" % str(e)[:80])


# ==============================================================================
# JOB
# ==============================================================================

def _fix_coh3d8_orientation(inp_path):
    """Post-process INP to fix COH3D8 element node ordering.

    Computes the full Jacobian determinant at the element center (xi=0,0,0)
    using 8-node hex shape function derivatives. If det(J) < 0 the element
    is inverted and bottom/top face nodes are swapped.
    """
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    # Pass 1: collect all node coordinates per instance
    instances = {}
    current_inst = None
    in_node_block = False
    for line in lines:
        ls = line.strip()
        if ls.startswith('*Instance,'):
            for token in ls.split(','):
                token = token.strip()
                if token.lower().startswith('name='):
                    current_inst = token.split('=', 1)[1].strip()
                    instances[current_inst] = {}
                    break
            in_node_block = False
        elif ls == '*Node' and current_inst:
            in_node_block = True
        elif in_node_block and ls.startswith('*'):
            in_node_block = False
        elif in_node_block and current_inst:
            parts = ls.split(',')
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    instances[current_inst][nid] = (x, y, z)
                except ValueError:
                    pass

    def _jac_det_hex8(coords):
        """Jacobian determinant of 8-node hex at center (xi1=xi2=xi3=0).

        COH3D8 node ordering:
          bottom: 1(-,-,-) 2(+,-,-) 3(+,+,-) 4(-,+,-)
          top:    5(-,-,+) 6(+,-,+) 7(+,+,+) 8(-,+,+)

        dN/dxi at (0,0,0):
          dN/dxi1 = 1/8 * [-1,+1,+1,-1,-1,+1,+1,-1]
          dN/dxi2 = 1/8 * [-1,-1,+1,+1,-1,-1,+1,+1]
          dN/dxi3 = 1/8 * [-1,-1,-1,-1,+1,+1,+1,+1]
        """
        # Shape function derivatives at center (xi=0)
        dNdxi1 = [-1, 1, 1, -1, -1, 1, 1, -1]
        dNdxi2 = [-1, -1, 1, 1, -1, -1, 1, 1]
        dNdxi3 = [-1, -1, -1, -1, 1, 1, 1, 1]

        # Jacobian columns: dx/dxi_j = sum(dN_i/dxi_j * x_i) / 8
        j = [[0.0]*3 for _ in range(3)]
        for i in range(8):
            for k in range(3):  # x, y, z
                j[0][k] += dNdxi1[i] * coords[i][k]
                j[1][k] += dNdxi2[i] * coords[i][k]
                j[2][k] += dNdxi3[i] * coords[i][k]

        # det(J) = j[0] . (j[1] x j[2])   (scalar triple product)
        cx = j[1][1]*j[2][2] - j[1][2]*j[2][1]
        cy = j[1][2]*j[2][0] - j[1][0]*j[2][2]
        cz = j[1][0]*j[2][1] - j[1][1]*j[2][0]
        return j[0][0]*cx + j[0][1]*cy + j[0][2]*cz

    # Replace COH3D8 with C3D8R for adhesive layers.
    # COH3D8 fails Abaqus quality checks on thin (0.2mm) curved geometry due to
    # extreme isoparametric angle distortion at boundary elements.
    # C3D8R avoids this issue while preserving the mesh topology.
    # Also replace *Cohesive Section (traction-separation) with *Solid Section,
    # and add equivalent elastic material MAT-ADHESIVE-SOLID.
    out_lines = []
    n_replaced = 0
    n_section_replaced = 0
    mat_adhesive_solid_added = False

    for i, line in enumerate(lines):
        ls = line.strip().upper()

        # Replace element type COH3D8 → C3D8R
        if ls.startswith('*ELEMENT') and 'COH3D8' in ls:
            line = line.replace('COH3D8', 'C3D8R').replace('coh3d8', 'C3D8R')
            n_replaced += 1
            out_lines.append(line)
            continue

        # Replace *Cohesive Section → *Solid Section
        if ls.startswith('*COHESIVE SECTION') and 'TRACTION' in ls:
            # Extract elset and material from the line
            tokens = {}
            for tok in ls.split(','):
                tok = tok.strip()
                if '=' in tok:
                    key, val = tok.split('=', 1)
                    tokens[key.strip()] = val.strip()
            elset = tokens.get('ELSET', '')
            # Replace with *Solid Section using the solid material
            out_lines.append(
                '*Solid Section, elset=%s, material=MAT-ADHESIVE-SOLID\n'
                % elset)
            # Skip the data line (thickness value)
            # Find the next non-blank line that starts with a number
            n_section_replaced += 1
            # Write empty data line (C3D8R doesn't need thickness)
            out_lines.append(',\n')
            continue

        # Skip data lines after *Cohesive Section (already replaced above)
        # But we need to detect them. The data line is the line after
        # *Cohesive Section and contains just the thickness value.
        # We handle this by checking if the previous output line was our
        # replacement *Solid Section.
        if (len(out_lines) >= 2
                and out_lines[-1].strip() == ','
                and '*Solid Section' in out_lines[-2]
                and ls and not ls.startswith('*')):
            # This is the original thickness data line — skip it
            continue

        # Add MAT-ADHESIVE-SOLID material definition before first *Cohesive Section
        # usage, right after MAT-ADHESIVE definition ends
        if (not mat_adhesive_solid_added
                and ls.startswith('*MATERIAL')
                and 'MAT-ADHESIVE-DAMAGED' in ls):
            # Insert our new material before the DAMAGED material
            out_lines.append('*Material, name=MAT-ADHESIVE-SOLID\n')
            out_lines.append('*Density\n')
            out_lines.append('1200e-12,\n')
            out_lines.append('*Elastic\n')
            out_lines.append('20000., 0.3\n')
            out_lines.append('*Expansion, zero=20.\n')
            out_lines.append('45e-6,\n')
            mat_adhesive_solid_added = True

        out_lines.append(line)

    with open(inp_path, 'w') as f:
        f.writelines(out_lines)

    print("COH3D8→C3D8R: %d element blocks, %d sections replaced, "
          "MAT-ADHESIVE-SOLID %s"
          % (n_replaced, n_section_replaced,
             "added" if mat_adhesive_solid_added else "NOT added"))


def create_and_run_job(model, job_name, no_run=False, project_root=None):
    """Create Abaqus job, write INP, fix COH3D8 orientation, optionally run."""
    mdb.Job(name=job_name, model='Model-1', type=ANALYSIS, resultsFormat=ODB,
            numCpus=4, numDomains=4, multiprocessingMode=DEFAULT)
    mdb.saveAs(pathName=job_name + '.cae')

    print("Writing INP: %s.inp" % job_name)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    inp_path = os.path.abspath(job_name + '.inp')

    # Fix COH3D8: add explicit STACK DIRECTION and flip inverted elements
    _fix_coh3d8_orientation(inp_path)

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
        ['abaqus', 'job=' + job_name, 'input=' + job_name + '.inp',
         'cpus=4', 'memory=' + SOLVER_MEMORY],
        cwd=cwd)
    if r == 0:
        print("Job COMPLETED: %s.odb" % job_name)
    else:
        print("Job FAILED (exit code %d)" % r)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def generate_sector12_model(job_name, defect_params=None,
                             global_seed=None, defect_seed=None,
                             frame_seed=None,
                             adhesive_thickness=None, adhesive_params=None,
                             no_run=False, project_root=None):
    """
    Main entry point: CZM 1/12 sector (30 deg) model, no openings.

    Args:
        job_name: Abaqus job name
        defect_params: dict with defect_type, theta_deg, z_center, radius, ...
                       None for healthy model
        global_seed: mesh size (mm), default 25
        defect_seed: local mesh for defect zone (mm), default 10
        frame_seed: ring frame mesh (mm), default 15
        adhesive_thickness: cohesive layer thickness (mm), default 0.2
        adhesive_params: dict overriding DEFAULT_CZM_PARAMS
        no_run: if True, only write INP
        project_root: project root for patch script
    """
    g_seed = global_seed if global_seed is not None else GLOBAL_SEED
    d_seed = defect_seed if defect_seed is not None else DEFECT_SEED
    f_seed = frame_seed if frame_seed is not None else FRAME_SEED
    adh_t = adhesive_thickness if adhesive_thickness is not None else ADHESIVE_T

    czm = dict(DEFAULT_CZM_PARAMS)
    if adhesive_params:
        czm.update(adhesive_params)

    defect_type = (defect_params.get('defect_type', 'debonding')
                   if defect_params else None)

    # No openings in 1/12 sector model
    openings = []

    print("=" * 70)
    print("CZM SECTOR 1/12 (%.0f deg) -- No Openings" % SECTOR_ANGLE)
    print("  Mesh: global=%.0f, defect=%.0f, frame=%.0f mm" % (
        g_seed, d_seed, f_seed))
    print("  Adhesive: thickness=%.2f mm, Kn=%.0e, Ks=%.0e" % (
        adh_t, czm['Kn'], czm['Ks']))
    print("  Memory: %s" % SOLVER_MEMORY)
    if defect_params:
        print("  Defect: %s | theta=%.1f z=%.0f r=%.0f" % (
            defect_type, defect_params['theta_deg'],
            defect_params['z_center'], defect_params['radius']))
    else:
        print("  Mode: healthy (no defect)")
    print("=" * 70)

    Mdb()
    model = mdb.models['Model-1']

    print("Openings: 0 (sector model, no openings)")

    # Ring frame positions (no collision check needed — no openings)
    frame_z_positions = list(RING_FRAME_Z_POSITIONS)
    print("Ring frames: %d at z = %s" % (
        len(frame_z_positions),
        ', '.join(['%.0f' % z for z in frame_z_positions])))

    # 1. Materials & Sections
    create_materials(model)
    create_sections(model)
    if defect_params:
        create_defect_materials(model, defect_params)
        create_defect_sections(model, defect_params)
    create_adhesive_materials(model, czm, adh_t)
    create_adhesive_sections(model, adh_t)

    # 2. Base geometry (5 parts)
    p_inner, p_adh_inner, p_core, p_adh_outer, p_outer = \
        create_base_parts_with_adhesive(model, adh_t)

    # 3. Partition openings — SKIPPED (no openings in sector model)

    # 4. Partition defect zone (skins + core only)
    if defect_params:
        parts_to_partition = [
            ('shell', p_inner),
            ('solid', p_core),
            ('shell', p_outer),
        ]
        partition_defect_zone(parts_to_partition, defect_params)

    # 5. Section assignment: 3-tier (skins/core) + adhesive
    assign_sections_3tier(p_inner, p_core, p_outer, openings, defect_params)
    assign_adhesive_sections(p_adh_inner, p_adh_outer, defect_params,
                              openings, adh_t)

    # 6. Ring frames
    frame_parts = create_ring_frame_parts(model, frame_z_positions)

    # 7. Assembly (5 parts + frames)
    (a, inst_inner, inst_adh_inner, inst_core,
     inst_adh_outer, inst_outer, frame_insts) = \
        create_assembly_with_adhesive(
            model, p_inner, p_adh_inner, p_core, p_adh_outer,
            p_outer, frame_parts)

    # 8. Tie constraints (4 adhesive + frame)
    create_cohesive_connections(
        model, a, inst_inner, inst_adh_inner, inst_core,
        inst_adh_outer, inst_outer, frame_insts, adh_t)

    # 9. Boundary conditions (bottom fix)
    apply_boundary_conditions_with_adhesive(
        model, a, inst_inner, inst_adh_inner, inst_core,
        inst_adh_outer, inst_outer, frame_insts)

    # 9b. Symmetry BCs on sector cut faces
    apply_symmetry_bcs(
        model, a, inst_inner, inst_adh_inner, inst_core,
        inst_adh_outer, inst_outer, frame_insts)

    # 10. Steps + field output (SDEG, STATUS added for CZM)
    model.StaticStep(name='Step-1', previous='Initial',
                     description='Thermal load only (CTE mismatch)')
    model.StaticStep(name='Step-2', previous='Step-1',
                     description='Thermal + Mechanical (Max-Q)')
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'U', 'RF', 'TEMP', 'NT', 'LE', 'SDEG', 'STATUS'))
    model.FieldOutputRequest(name='F-Output-2',
                             createStepName='Step-2',
                             variables=('S', 'U', 'RF', 'TEMP', 'NT', 'LE',
                                        'SDEG', 'STATUS'))

    # 11. Mesh (adhesive first -> core -> skins/frames)
    generate_mesh_with_adhesive(
        a, inst_inner, inst_adh_inner, inst_core,
        inst_adh_outer, inst_outer, frame_insts,
        openings, defect_params,
        g_seed, OPENING_SEED, d_seed, f_seed, adh_t)

    # 12. Post-mesh BCs
    apply_post_mesh_bcs_with_adhesive(
        model, a, inst_inner, inst_adh_inner, inst_core,
        inst_adh_outer, inst_outer)

    # 13. Thermal load (including adhesive)
    apply_thermal_load_with_adhesive(
        model, a, inst_inner, inst_adh_inner, inst_core,
        inst_adh_outer, inst_outer, frame_insts)

    # 14. Mechanical loads
    apply_mechanical_loads(model, a, inst_inner, inst_core, inst_outer)

    # 15. Job
    create_and_run_job(model, job_name, no_run, project_root)

    print("=" * 70)
    print("DONE: %s (Sector 1/12, %.0f deg, CZM adhesive=%.2fmm, %s)" % (
        job_name, SECTOR_ANGLE, adh_t, defect_type or 'healthy'))
    print("=" * 70)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate CZM 1/12 sector (30 deg) model for solver verification')
    parser.add_argument('--job', type=str, default='Job-CZM-S12',
                        help='Job name (default: Job-CZM-S12)')
    parser.add_argument('--job_name', type=str, default=None,
                        help='Alias for --job (run_batch compatibility)')
    parser.add_argument('--defect', type=str, default=None,
                        help='JSON string or path to JSON with defect params')
    parser.add_argument('--param_file', type=str, default=None,
                        help='Path to JSON file with defect params')
    parser.add_argument('--healthy', action='store_true',
                        help='Generate healthy model (no defect, ignore --defect)')
    parser.add_argument('--seed', type=float, default=None,
                        help='Alias for --global_seed')
    parser.add_argument('--global_seed', type=float, default=None,
                        help='Global mesh seed (mm), default 25')
    parser.add_argument('--defect_seed', type=float, default=None,
                        help='Defect zone mesh seed (mm), default 10')
    parser.add_argument('--adhesive_thickness', type=float, default=None,
                        help='Cohesive layer thickness (mm), default 0.2')
    parser.add_argument('--adhesive_params', type=str, default=None,
                        help='JSON string or file with CZM material params')
    parser.add_argument('--no_run', action='store_true',
                        help='Write INP only, do not run solver')
    parser.add_argument('--project_root', type=str, default=None,
                        help='Project root for patch script')

    args, _ = parser.parse_known_args()

    job_name = args.job_name if args.job_name is not None else args.job

    # --seed is alias for --global_seed
    g_seed = args.global_seed
    if g_seed is None and args.seed is not None:
        g_seed = args.seed

    # Parse defect params (skip if --healthy)
    defect_data = None
    if not args.healthy:
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

    # Parse adhesive CZM params
    adh_params = None
    if args.adhesive_params:
        if os.path.exists(args.adhesive_params):
            with open(args.adhesive_params, 'r') as f:
                adh_params = json.load(f)
        else:
            try:
                adh_params = json.loads(args.adhesive_params)
            except:
                print("Invalid adhesive_params JSON or file path")

    project_root = (args.project_root or os.environ.get('PROJECT_ROOT')
                    or os.environ.get('PAYLOAD2026_ROOT'))

    generate_sector12_model(
        job_name=job_name,
        defect_params=defect_data,
        global_seed=g_seed,
        defect_seed=args.defect_seed,
        adhesive_thickness=args.adhesive_thickness,
        adhesive_params=adh_params,
        no_run=getattr(args, 'no_run', False),
        project_root=project_root,
    )
