# -*- coding: utf-8 -*-
# generate_fairing_dataset.py
# Abaqus Python script to generate H3 Type-S fairing FEM model with multiple defect types
#
# Supported defect types (academic justification: docs/DEFECT_MODELS_ACADEMIC.md):
#   debonding          — Outer skin-core delamination. Ref: NASA NTRS 20160005994.
#   fod                — FOD / hard inclusion in core. Ref: MDPI Appl. Sci. 2024 16(3):1459.
#   impact             — BVID: matrix degradation + core crush. Ref: Composites Part B 2017, ASTM D7136.
#   delamination       — Inter-ply delamination. Ref: Compos. Sci. Technol. 2006, MDPI Materials 2019.
#   inner_debond       — Inner skin-core debonding. Ref: NASA NTRS, DEFECT_PLAN.
#   thermal_progression— CTE mismatch (CFRP vs Al) interface damage. Ref: Composites Part B 2018.
#   acoustic_fatigue   — 147–148 dB launch fatigue. Ref: UTIAS Acoustic Fatigue 2019.
#
# Usage: abaqus cae noGUI=generate_fairing_dataset.py -- --job <job_name> --defect <defect_params_json>

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
# PARAMETERS (JAXA H3 Type-S Fairing Dimensions)
# ==============================================================================
# Geometry
RADIUS = 2600.0  # mm (Approximate H3 fairing radius)
H_BARREL = 5000.0 # mm (Barrel section height)
H_NOSE = 5400.0   # mm (Ogive nose cone height)
TOTAL_HEIGHT = H_BARREL + H_NOSE

# Tangent Ogive Calculation
# R = Radius of base (RADIUS)
# L = Length of nose (H_NOSE)
# rho = (R^2 + L^2) / (2*R)  (Ogive radius of curvature)
# center of curvature = (xc, yc) = (0, R - rho) relative to nose base center
# Here we define the profile in (r, z) coordinates.
# The arc center is at (R - rho, 0) in local coords if z starts at 0.
OGIVE_RHO = (RADIUS**2 + H_NOSE**2) / (2.0 * RADIUS)
OGIVE_XC = RADIUS - OGIVE_RHO  # X-coordinate of arc center (relative to axis)

# Material Properties (Representative Values)
# CFRP Face Sheets (Toray T1000G or similar)
E1 = 160000.0 # MPa
E2 = 10000.0  # MPa
NU12 = 0.3
G12 = 5000.0  # MPa
G13 = 5000.0
G23 = 3000.0

# Aluminum Honeycomb Core — Cylindrical CSYS convention (1=R=through-thickness, 2=θ, 3=Z)
E_CORE_1 = 1000.0  # MPa (R: high out-of-plane / through-thickness stiffness)
E_CORE_2 = 1.0     # MPa (θ: very low in-plane stiffness)
E_CORE_3 = 1.0     # MPa (Z: very low in-plane stiffness)
NU_CORE_12 = 0.01
NU_CORE_13 = 0.01
NU_CORE_23 = 0.01
G_CORE_12 = 400.0  # MPa (R-θ: through-thickness shear L-dir)
G_CORE_13 = 240.0  # MPa (R-Z: through-thickness shear W-dir)
G_CORE_23 = 1.0    # MPa (θ-Z: in-plane shear)

# Thicknesses
FACE_T = 1.0   # mm (CFRP Face Sheet Thickness)
CORE_T = 38.0  # mm (Honeycomb Core Thickness) — H3: パネル総厚~40mm, スキン2×1mm → コア~38mm

# Mesh Size — docs/MESH_DEFECT_ANALYSIS.md: h≤D/2 for defect resolution
# 50 mm: resolves Medium+ (r≥25mm), ~2 min/sample, 100 samples ~3.5 h
GLOBAL_SEED = 50.0  # mm (was 200: only Critical resolvable)
DEFECT_SEED = 15.0  # mm (Local refinement around defect)

# Thermal Load
TEMP_INITIAL = 20.0 # C
TEMP_FINAL_OUTER = 120.0 # C (Ascent heating)
TEMP_FINAL_INNER = 20.0  # C
TEMP_FINAL_CORE = 70.0   # C (Approx average)

# Opening zone (access door exclusion) — None = no opening
OPENING_PARAMS = None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_radius_at_z(z):
    """
    Returns the outer radius of the fairing at a given Z-coordinate.
    Z=0 is the base of the barrel.
    """
    if z < 0:
        return RADIUS
    elif z <= H_BARREL:
        return RADIUS
    elif z > TOTAL_HEIGHT:
        return 0.0
    else:
        # Ogive section
        # Local z in nose cone
        z_local = z - H_BARREL
        term = OGIVE_RHO**2 - z_local**2
        if term < 0:
            return 0.0
        return OGIVE_XC + math.sqrt(term)

def _point_in_opening_zone(x, y, z, opening_params):
    """Check if point is inside the opening (access door, etc.). Exclude from defect zone."""
    if not opening_params:
        return False
    z_c = opening_params['z_center']
    theta_deg = opening_params['theta_deg']
    r_open = opening_params['radius']
    if abs(y - z_c) > r_open * 1.5:
        return False
    r_local = math.sqrt(x * x + z * z)
    if r_local < 1.0:
        return False
    theta_rad_pt = math.atan2(z, x)
    theta_center_rad = math.radians(theta_deg)
    arc_mm = r_local * abs(theta_rad_pt - theta_center_rad)
    dy = y - z_c
    dist = math.sqrt(arc_mm * arc_mm + dy * dy)
    return dist <= r_open * 1.01

def _point_in_defect_zone(x, y, z, defect_params):
    """
    Check if point (x,y,z) is inside the circular defect zone on the fairing surface.
    Abaqus revolve convention: Y=axial, XZ=radial plane.
    Same formula as extract_odb_results._is_node_in_defect().
    """
    z_c = defect_params['z_center']
    theta_deg = defect_params['theta_deg']
    r_def = defect_params['radius']

    # Axial quick reject
    if abs(y - z_c) > r_def * 1.5:
        return False

    # Radial distance in XZ plane
    r_local = math.sqrt(x * x + z * z)
    if r_local < 1.0:
        return False

    # Circumferential angle in XZ plane (not XY!)
    theta_rad_pt = math.atan2(z, x)
    theta_center_rad = math.radians(theta_deg)
    arc_mm = r_local * abs(theta_rad_pt - theta_center_rad)
    dy = y - z_c
    dist = math.sqrt(arc_mm * arc_mm + dy * dy)
    return dist <= r_def * 1.01  # 1% tolerance

def is_face_in_defect_zone(face, defect_params):
    """
    Checks if a face's centroid is within the defect zone.
    defect_params: {z_center, theta_deg, radius}
    Uses pointOn (reliable in Abaqus) for face position.
    """
    if not defect_params:
        return False
    pt = face.pointOn[0]
    return _point_in_defect_zone(pt[0], pt[1], pt[2], defect_params)

def is_cell_in_defect_zone(cell, defect_params, opening_params=None):
    """
    Checks if a solid cell's centroid is within the defect zone (for core).
    Excludes cells inside the opening.
    """
    if not defect_params:
        return False
    pt = cell.pointOn[0]
    if opening_params and _point_in_opening_zone(pt[0], pt[1], pt[2], opening_params):
        return False
    return _point_in_defect_zone(pt[0], pt[1], pt[2], defect_params)

def create_materials(model):
    """Defines CFRP and Honeycomb materials in the Abaqus model."""
    # CFRP
    mat_cfrp = model.Material(name='CFRP_T1000G')
    mat_cfrp.Elastic(type=LAMINA, table=((E1, E2, NU12, G12, G13, G23), ))
    mat_cfrp.Density(table=((1600e-12, ), )) # tonne/mm^3
    mat_cfrp.Expansion(table=((-0.3e-6, 28e-6, 0.0), )) # Alpha11, Alpha22, Alpha12 (local)

    # Honeycomb
    mat_core = model.Material(name='AL_HONEYCOMB')
    mat_core.Elastic(type=ENGINEERING_CONSTANTS, table=((
        E_CORE_1, E_CORE_2, E_CORE_3,
        NU_CORE_12, NU_CORE_13, NU_CORE_23,
        G_CORE_12, G_CORE_13, G_CORE_23
    ), ))
    mat_core.Density(table=((50e-12, ), ))
    mat_core.Expansion(table=((23e-6, ), )) # Isotropic-ish

def create_sections(model):
    """Creates shell and solid sections."""
    # Composite Layup for Shells
    # [45/0/-45/90]s -> 8 plies
    
    layup_orientation = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
    entries = []
    for ang in layup_orientation:
        entries.append(section.SectionLayer(
            thickness=FACE_T/8.0, orientAngle=ang, material='CFRP_T1000G'))
    
    model.CompositeShellSection(
        name='Section-CFRP-Skin', preIntegrate=OFF, 
        idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF, 
        thicknessType=UNIFORM, poissonDefinition=DEFAULT, 
        temperature=GRADIENT, integrationRule=SIMPSON)

    # Solid Section for Core
    model.HomogeneousSolidSection(
        name='Section-Core', material='AL_HONEYCOMB', thickness=None)

def create_defect_materials(model, defect_params):
    """Create defect-type-specific modified materials."""
    defect_type = defect_params.get('defect_type', 'debonding')

    if defect_type == 'debonding':
        # Near-zero stiffness skin (delaminated region loses load transfer)
        mat = model.Material(name='CFRP_DEBONDED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.01, E2 * 0.01, NU12,
            G12 * 0.01, G13 * 0.01, G23 * 0.01
        ), ))
        mat.Density(table=((1600e-12, ), ))
        mat.Expansion(table=((-0.3e-6, 28e-6, 0.0), ))

    elif defect_type == 'fod':
        # Stiff FOD inclusion. MDPI Appl. Sci. 2024 16(3):1459. CTE 12e-6 for metallic FOD.
        sf = defect_params.get('stiffness_factor', 10.0)
        mat = model.Material(name='AL_HONEYCOMB_FOD')
        mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1 * sf, E_CORE_2 * sf, E_CORE_3 * sf,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12 * sf, G_CORE_13 * sf, G_CORE_23 * sf
        ), ))
        mat.Density(table=((200e-12, ), ))  # Heavier inclusion
        mat.Expansion(table=((12e-6, ), ))  # Different CTE

    elif defect_type == 'impact':
        # Matrix-damaged skin (fiber partly intact, matrix severely degraded)
        dr = defect_params.get('damage_ratio', 0.3)
        mat_skin = model.Material(name='CFRP_IMPACT_DAMAGED')
        mat_skin.Elastic(type=LAMINA, table=((
            E1 * 0.7,     # Fiber slightly degraded
            E2 * dr,       # Matrix severely degraded
            NU12,
            G12 * dr,      # Shear degraded
            G13 * dr,
            G23 * dr
        ), ))
        mat_skin.Density(table=((1600e-12, ), ))
        mat_skin.Expansion(table=((-0.3e-6, 28e-6, 0.0), ))

        # Crushed honeycomb. Composites Part A 2019: cell buckling under impact.
        # Cylindrical convention: 1=R(through-thickness), 2=θ(in-plane), 3=Z(in-plane)
        mat_core = model.Material(name='AL_HONEYCOMB_CRUSHED')
        mat_core.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1 * 0.1, E_CORE_2 * 0.5, E_CORE_3 * 0.5,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12 * 0.1, G_CORE_13 * 0.1, G_CORE_23 * 0.5
        ), ))
        mat_core.Density(table=((50e-12, ), ))
        mat_core.Expansion(table=((23e-6, ), ))

    elif defect_type == 'delamination':
        # Inter-ply delamination: reduced shear (G12, G13, G23) simulates layer separation
        depth = defect_params.get('delam_depth', 0.5)  # 0.2-0.8 fraction of plies
        shear_red = max(0.05, 1.0 - depth)
        mat = model.Material(name='CFRP_DELAMINATED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.9,           # Fiber mostly intact
            E2 * (0.3 + 0.5 * (1 - depth)),  # Matrix degraded by delam
            NU12,
            G12 * shear_red, G13 * shear_red, G23 * shear_red
        ), ))
        mat.Density(table=((1600e-12, ), ))
        mat.Expansion(table=((-0.3e-6, 28e-6, 0.0), ))

    elif defect_type == 'inner_debond':
        # Inner skin-core. Same mechanics as debonding. NASA NTRS, DEFECT_PLAN.
        mat = model.Material(name='CFRP_INNER_DEBONDED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.01, E2 * 0.01, NU12,
            G12 * 0.01, G13 * 0.01, G23 * 0.01
        ), ))
        mat.Density(table=((1600e-12, ), ))
        mat.Expansion(table=((-0.3e-6, 28e-6, 0.0), ))

    elif defect_type == 'thermal_progression':
        # CTE mismatch (CFRP -0.3 vs Al 23e-6) induced interface damage
        mat = model.Material(name='CFRP_THERMAL_DAMAGED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.05, E2 * 0.05, NU12,
            G12 * 0.05, G13 * 0.05, G23 * 0.05
        ), ))
        mat.Density(table=((1600e-12, ), ))
        # Higher effective CTE simulates thermally-opened interface
        mat.Expansion(table=((5e-6, 40e-6, 0.0), ))  # thermally damaged: fiber CTE +, matrix CTE increased

    elif defect_type == 'acoustic_fatigue':
        # 147-148 dB launch acoustic fatigue. UTIAS 2019: interface weakening.
        sev = defect_params.get('fatigue_severity', 0.35)
        mat = model.Material(name='CFRP_ACOUSTIC_FATIGUED')
        mat.Elastic(type=LAMINA, table=((
            E1 * (0.2 + 0.5 * (1 - sev)), E2 * sev, NU12,
            G12 * sev, G13 * sev, G23 * sev
        ), ))
        mat.Density(table=((1600e-12, ), ))
        mat.Expansion(table=((-0.3e-6, 28e-6, 0.0), ))

    print("Defect materials created: type=%s" % defect_type)

def create_defect_sections(model, defect_params):
    """Create sections for defect-zone materials."""
    defect_type = defect_params.get('defect_type', 'debonding')
    layup_orientation = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]

    if defect_type == 'debonding':
        entries = [section.SectionLayer(
            thickness=FACE_T/8.0, orientAngle=ang, material='CFRP_DEBONDED')
            for ang in layup_orientation]
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
            thickness=FACE_T/8.0, orientAngle=ang, material='CFRP_IMPACT_DAMAGED')
            for ang in layup_orientation]
        model.CompositeShellSection(
            name='Section-CFRP-Impact', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)
        model.HomogeneousSolidSection(
            name='Section-Core-Crushed', material='AL_HONEYCOMB_CRUSHED', thickness=None)

    elif defect_type == 'delamination':
        entries = [section.SectionLayer(
            thickness=FACE_T/8.0, orientAngle=ang, material='CFRP_DELAMINATED')
            for ang in layup_orientation]
        model.CompositeShellSection(
            name='Section-CFRP-Delaminated', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

    elif defect_type == 'inner_debond':
        entries = [section.SectionLayer(
            thickness=FACE_T/8.0, orientAngle=ang, material='CFRP_INNER_DEBONDED')
            for ang in layup_orientation]
        model.CompositeShellSection(
            name='Section-CFRP-InnerDebonded', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

    elif defect_type == 'thermal_progression':
        entries = [section.SectionLayer(
            thickness=FACE_T/8.0, orientAngle=ang, material='CFRP_THERMAL_DAMAGED')
            for ang in layup_orientation]
        model.CompositeShellSection(
            name='Section-CFRP-ThermalDamaged', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

    elif defect_type == 'acoustic_fatigue':
        entries = [section.SectionLayer(
            thickness=FACE_T/8.0, orientAngle=ang, material='CFRP_ACOUSTIC_FATIGUED')
            for ang in layup_orientation]
        model.CompositeShellSection(
            name='Section-CFRP-AcousticFatigued', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

def create_parts(model):
    """Creates the geometry parts (Inner Skin, Core, Outer Skin)."""
    
    # ---------------------------------------------------------
    # Part 1: Inner Skin
    # ---------------------------------------------------------
    s1 = model.ConstrainedSketch(name='profile_inner', sheetSize=20000.0)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))  # Revolve axis
    
    # Barrel
    s1.Line(point1=(RADIUS, 0.0), point2=(RADIUS, H_BARREL))
    
    # Ogive (conical approximation - more robust than arc for some Abaqus versions)
    s1.Line(point1=(RADIUS, H_BARREL), point2=(0.0, TOTAL_HEIGHT))
    
    p_inner = model.Part(name='Part-InnerSkin', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=60.0, flipRevolveDirection=OFF)  # 1/6 section
    
    # ---------------------------------------------------------
    # Part 2: Core (Solid)
    # ---------------------------------------------------------
    s2 = model.ConstrainedSketch(name='profile_core', sheetSize=20000.0)
    s2.setPrimaryObject(option=STANDALONE)
    s2.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))
    
    rho_outer = OGIVE_RHO + CORE_T
    z_tip_outer = H_BARREL + math.sqrt(rho_outer**2 - OGIVE_XC**2)

    # Closed loop with Arcs
    s2.Line(point1=(RADIUS, 0.0), point2=(RADIUS, H_BARREL))
    s2.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(RADIUS, H_BARREL),
        point2=(0.0, TOTAL_HEIGHT),
        direction=COUNTERCLOCKWISE
    )
    s2.Line(point1=(0.0, TOTAL_HEIGHT), point2=(0.0, z_tip_outer))
    s2.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(0.0, z_tip_outer),
        point2=(RADIUS + CORE_T, H_BARREL),
        direction=CLOCKWISE
    )
    s2.Line(point1=(RADIUS + CORE_T, H_BARREL), point2=(RADIUS + CORE_T, 0.0))
    s2.Line(point1=(RADIUS + CORE_T, 0.0), point2=(RADIUS, 0.0))
    
    p_core = model.Part(name='Part-Core', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=s2, angle=60.0, flipRevolveDirection=OFF)  # 1/6 section
    
    # ---------------------------------------------------------
    # Part 3: Outer Skin (Shell)
    # ---------------------------------------------------------
    s3 = model.ConstrainedSketch(name='profile_outer', sheetSize=20000.0)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))  # Revolve axis
    
    s3.Line(point1=(RADIUS + CORE_T, 0.0), point2=(RADIUS + CORE_T, H_BARREL))
    s3.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(RADIUS + CORE_T, H_BARREL),
        point2=(0.0, z_tip_outer),
        direction=COUNTERCLOCKWISE
    )
    
    p_outer = model.Part(name='Part-OuterSkin', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=60.0, flipRevolveDirection=OFF)  # 1/6 section

    return p_inner, p_core, p_outer

def partition_defect_zone(parts_to_partition, defect_params):
    """
    Partitions parts at the defect zone for section reassignment.
    Works at PART level (before assembly) so that section assignments
    carry through to instances.

    Args:
        parts_to_partition: list of (geom_type, part) tuples
            geom_type: 'shell' or 'solid'
        defect_params: {z_center, theta_deg, radius, defect_type, ...}
    """
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

    # Bounding box for selecting entities near defect (with margin)
    z1_bb = max(1.0, z_c - r_def - 50)
    z2_bb = min(TOTAL_HEIGHT - 1.0, z_c + r_def + 50)
    r_max = RADIUS + CORE_T + 100
    x_min = -r_max - 100
    x_max = r_max + 100

    for geom_type, part in parts_to_partition:
        # Create datum planes on the part
        dp_z1 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=z_c - r_def)
        dp_z2 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=z_c + r_def)
        # Theta cutting planes pass through Y-axis (fairing axial axis)
        # theta is measured in XZ plane: x=R*cos(theta), z=R*sin(theta)
        dp_t1 = part.DatumPlaneByThreePoints(
            point1=(0, 0, 0), point2=(0, 100, 0),
            point3=(math.cos(t1), 0, math.sin(t1)))
        dp_t2 = part.DatumPlaneByThreePoints(
            point1=(0, 0, 0), point2=(0, 100, 0),
            point3=(math.cos(t2), 0, math.sin(t2)))

        for dp_id in [dp_z1.id, dp_z2.id, dp_t1.id, dp_t2.id]:
            try:
                if geom_type == 'shell':
                    # Pass ALL faces — datum plane defines the cut location
                    faces = part.faces
                    if len(faces) > 0:
                        part.PartitionFaceByDatumPlane(
                            datumPlane=part.datums[dp_id], faces=faces)
                else:  # solid
                    cells = part.cells
                    if len(cells) > 0:
                        part.PartitionCellByDatumPlane(
                            datumPlane=part.datums[dp_id], cells=cells)
            except Exception as e:
                print("  Part partition warning (%s): %s" % (geom_type, str(e)[:80]))

    print("Defect zone partitioned: z=%.0f theta=%.1f r=%.0f" % (z_c, theta_deg, r_def))

def _create_cylindrical_csys(part, name='CylCS'):
    """Create a cylindrical CSYS aligned with the fairing axis (Y).

    Convention: CSYS-1=R (radial), CSYS-2=theta (circumferential), CSYS-3=Z (axial=Y).
    Used for the core solid.
    """
    datum = part.DatumCsysByThreePoints(
        name=name, coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(1.0, 0.0, 0.0),   # R at theta=0 -> X direction
        point2=(0.0, 1.0, 0.0))   # R-Z plane -> Z = Y direction (axial)
    return datum


def _create_shell_orientation_csys(part, name='ShellCS'):
    """Create a Cartesian CSYS with Y (axial) as the primary direction.

    When used with axis=AXIS_3 on shells:
      material-3 = shell normal
      material-1 = projection of CSYS-1 (Y=axial) onto shell = axial direction
      material-2 = normal x material-1 = circumferential direction
    This gives consistent fiber orientation everywhere on the cylindrical/ogive shell.
    """
    datum = part.DatumCsysByThreePoints(
        name=name, coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0),
        point1=(0.0, 1.0, 0.0),   # local 1 = Y direction (axial)
        point2=(1.0, 0.0, 0.0))   # local 1-2 plane
    return datum


def assign_all_sections(p_inner, p_core, p_outer, defect_params):
    """
    Assign sections to all parts, including defect-zone overrides.
    Must be called AFTER partition_defect_zone().

    Material orientation:
      Shells: Cartesian CSYS (Y=axial primary), axis=AXIS_3.
              material-1=axial, material-2=circumferential, material-3=shell normal
      Core:   Cylindrical CSYS, 1=R(through-thickness), 2=theta, 3=Z(axial)
    """
    # Shell orientation: Cartesian with Y-axial primary (consistent on curved shell)
    shell_cs_inner = _create_shell_orientation_csys(p_inner, 'ShellCS-Inner')
    shell_cs_outer = _create_shell_orientation_csys(p_outer, 'ShellCS-Outer')
    # Core orientation: Cylindrical (1=R, 2=theta, 3=Z)
    cyl_core = _create_cylindrical_csys(p_core, 'CylCS-Core')

    # Inner skin: always healthy CFRP
    region = p_inner.Set(faces=p_inner.faces, name='Set-All')
    p_inner.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_inner.MaterialOrientation(region=region, orientationType=SYSTEM, axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE,
                                localCsys=p_inner.datums[shell_cs_inner.id])

    # Core: healthy baseline for all cells
    region = p_core.Set(cells=p_core.cells, name='Set-All')
    p_core.SectionAssignment(region=region, sectionName='Section-Core')
    p_core.MaterialOrientation(region=region, orientationType=SYSTEM, axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE,
                                localCsys=p_core.datums[cyl_core.id])

    # Outer skin: healthy baseline for all faces
    region = p_outer.Set(faces=p_outer.faces, name='Set-All')
    p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_outer.MaterialOrientation(region=region, orientationType=SYSTEM, axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE,
                                localCsys=p_outer.datums[shell_cs_outer.id])

    if not defect_params:
        return

    defect_type = defect_params.get('defect_type', 'debonding')

    # --- Defect-zone section overrides ---
    # Outer skin defects: debonding, impact, delamination, thermal_progression, acoustic_fatigue
    outer_skin_defects = ('debonding', 'impact', 'delamination', 'thermal_progression', 'acoustic_fatigue')
    if defect_type in outer_skin_defects:
        section_map = {
            'debonding': 'Section-CFRP-Debonded',
            'impact': 'Section-CFRP-Impact',
            'delamination': 'Section-CFRP-Delaminated',
            'thermal_progression': 'Section-CFRP-ThermalDamaged',
            'acoustic_fatigue': 'Section-CFRP-AcousticFatigued',
        }
        section_name = section_map.get(defect_type, 'Section-CFRP-Debonded')
        defect_faces = [f for f in p_outer.faces
                        if is_face_in_defect_zone(f, defect_params)]
        if defect_faces:
            pts = tuple((f.pointOn[0],) for f in defect_faces)
            face_seq = p_outer.faces.findAt(*pts)
            region_d = p_outer.Set(faces=face_seq, name='Set-DefectZone-Skin')
            p_outer.SectionAssignment(region=region_d, sectionName=section_name)
            p_outer.MaterialOrientation(region=region_d, orientationType=SYSTEM, axis=AXIS_3,
                                        additionalRotationType=ROTATION_NONE,
                                        localCsys=p_outer.datums[shell_cs_outer.id])
            print("  %s: %d outer skin faces -> %s" % (defect_type, len(defect_faces), section_name))
        else:
            print("  Warning: no outer skin faces found in defect zone")

    # Inner skin defects: inner_debond
    if defect_type == 'inner_debond':
        defect_faces_inner = [f for f in p_inner.faces
                              if is_face_in_defect_zone(f, defect_params)]
        if defect_faces_inner:
            pts = tuple((f.pointOn[0],) for f in defect_faces_inner)
            face_seq = p_inner.faces.findAt(*pts)
            region_d = p_inner.Set(faces=face_seq, name='Set-DefectZone-InnerSkin')
            p_inner.SectionAssignment(region=region_d, sectionName='Section-CFRP-InnerDebonded')
            p_inner.MaterialOrientation(region=region_d, orientationType=SYSTEM, axis=AXIS_3,
                                        additionalRotationType=ROTATION_NONE,
                                        localCsys=p_inner.datums[shell_cs_inner.id])
            print("  inner_debond: %d inner skin faces -> Section-CFRP-InnerDebonded" % len(defect_faces_inner))
        else:
            print("  Warning: no inner skin faces found in defect zone")

    if defect_type in ('fod', 'impact'):
        # Override core in defect zone
        section_name = ('Section-Core-FOD' if defect_type == 'fod'
                        else 'Section-Core-Crushed')
        opening = defect_params.get('_opening_params') or OPENING_PARAMS
        defect_cells = [c for c in p_core.cells
                        if is_cell_in_defect_zone(c, defect_params, opening)]
        if defect_cells:
            # Use findAt to get proper CellArray type for Set()
            pts = tuple((c.pointOn[0],) for c in defect_cells)
            cell_seq = p_core.cells.findAt(*pts)
            region_d = p_core.Set(cells=cell_seq, name='Set-DefectZone-Core')
            p_core.SectionAssignment(region=region_d, sectionName=section_name)
            print("  %s: %d core cells -> %s" % (defect_type, len(defect_cells), section_name))
        else:
            print("  Warning: no core cells found in defect zone")

# ==============================================================================
# TIE CONSTRAINTS (skin-core coupling)
# ==============================================================================


def _classify_core_faces(inst_core):
    """Classify core solid faces as inner (R ~ RADIUS) or outer (R ~ RADIUS+CORE_T).

    Uses nearest-match: each face is classified based on which expected
    surface its centroid radius is closer to.  Edge faces (bottom, top,
    sector cuts) are skipped when their centroid is far from both surfaces.
    """
    inner_pts = []
    outer_pts = []
    theta_60 = math.radians(60.0)

    for f in inst_core.faces:
        try:
            pt = f.pointOn[0]
        except Exception:
            continue
        x, y, z = pt
        r = math.sqrt(x**2 + z**2)
        theta = math.atan2(z, x)

        # Skip bottom/top annulus faces
        if y < 1.0 or y > TOTAL_HEIGHT - 10:
            continue
        # Skip sector cut faces (theta ~ 0 or ~ 60 deg)
        if r > 10.0 and (abs(theta) < 0.02 or abs(theta - theta_60) < 0.02):
            continue

        # Expected inner/outer radii at this height
        r_inner = get_radius_at_z(y)
        r_outer = r_inner + CORE_T
        d_inner = abs(r - r_inner)
        d_outer = abs(r - r_outer)

        # Skip faces far from both surfaces (edge artifacts)
        if min(d_inner, d_outer) > CORE_T * 0.8:
            continue

        if d_inner <= d_outer:
            inner_pts.append((pt,))
        else:
            outer_pts.append((pt,))

    return inner_pts, outer_pts


def apply_tie_constraints(model, assembly, inst_inner, inst_core, inst_outer):
    """Tie inner/outer skins to core for proper thermal-structural coupling.

    Tie pairs:
      1. InnerSkin (SPOS) <-> Core inner face
      2. Core outer face <-> OuterSkin (SNEG)
    """
    # Classify core faces
    core_inner_pts, core_outer_pts = _classify_core_faces(inst_core)

    # Skin surfaces (SPOS = outward normal for revolved shells)
    surf_inner_skin = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')
    # OuterSkin SNEG = inward (toward core)
    surf_outer_skin = assembly.Surface(
        side2Faces=inst_outer.faces, name='Surf-OuterSkin')

    # --- Tie 1: InnerSkin <-> Core inner ---
    if core_inner_pts:
        core_inner_seq = inst_core.faces.findAt(*core_inner_pts)
        surf_core_inner = assembly.Surface(
            side1Faces=core_inner_seq, name='Surf-Core-Inner')
        model.Tie(name='Tie-InnerSkin-Core',
                  main=surf_core_inner, secondary=surf_inner_skin,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie 1: InnerSkin <-> Core(inner): %d core faces" %
              len(core_inner_pts))
    else:
        print("WARNING: No core inner faces found for Tie 1")

    # --- Tie 2: Core outer <-> OuterSkin ---
    if core_outer_pts:
        core_outer_seq = inst_core.faces.findAt(*core_outer_pts)
        surf_core_outer = assembly.Surface(
            side1Faces=core_outer_seq, name='Surf-Core-Outer')
        model.Tie(name='Tie-Core-OuterSkin',
                  main=surf_core_outer, secondary=surf_outer_skin,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie 2: Core(outer) <-> OuterSkin: %d core faces" %
              len(core_outer_pts))
    else:
        print("WARNING: No core outer faces found for Tie 2")


# ==============================================================================
# BOUNDARY CONDITIONS (symmetry + bottom fixity)
# ==============================================================================
SECTOR_ANGLE = 60.0  # degrees


def apply_bottom_fixity(model, assembly, inst_inner, inst_core, inst_outer):
    """Fix bottom (y=0) for all instances: u1=u2=u3=0."""
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

    # Solid faces at y=0 (core)
    face_seq = None
    bot_pts = []
    for f in inst_core.faces:
        try:
            pt = f.pointOn[0]
            if abs(pt[1]) < 1.0:
                bot_pts.append((pt,))
        except:
            pass
    if bot_pts:
        fseq = inst_core.faces.findAt(*bot_pts)
        if len(fseq) > 0:
            face_seq = fseq
    if face_seq is not None and len(face_seq) > 0:
        set_kwargs['faces'] = face_seq

    if set_kwargs:
        bot_set = assembly.Set(name='BC_Bottom', **set_kwargs)
        model.DisplacementBC(name='Fix_Bottom', createStepName='Initial',
                             region=bot_set, u1=0, u2=0, u3=0)
        print("BC: Fixed at y=0 (%s)" % ', '.join(
            '%s=%d' % (k, len(v)) for k, v in set_kwargs.items()))
    else:
        print("Warning: No BC geometry found at y=0")


def apply_symmetry_bcs(model, assembly, inst_inner, inst_core, inst_outer):
    """Apply symmetry BCs on sector cut faces.

    theta=0 face (Z=0 plane): U3=0 (ZSYMM-like)
    theta=SECTOR_ANGLE face: normal displacement=0 via local CSYS
    """
    all_insts = [inst_inner, inst_core, inst_outer]
    r_box = RADIUS + CORE_T + 500.0
    tol = 1.0  # mm

    # --- theta=0 face: Z=0 plane, constrain U3=0 ---
    sym0_edges = None
    sym0_faces = None
    for inst in all_insts:
        # Shell edges (skins)
        try:
            edges = inst.edges.getByBoundingBox(
                xMin=-1.0, xMax=r_box,
                yMin=-0.1, yMax=TOTAL_HEIGHT + 100.0,
                zMin=-tol, zMax=tol)
            if len(edges) > 0:
                sym0_edges = edges if sym0_edges is None else sym0_edges + edges
        except Exception:
            pass
        # Solid faces (core)
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
    theta_rad = math.radians(SECTOR_ANGLE)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Local Cartesian CSYS:
    #   local-1 (X') = tangential = (cos_t, 0, sin_t)
    #   local-2 (Y') = axial = (0, 1, 0)
    #   local-3 (Z') = normal to cut face = (-sin_t, 0, cos_t)
    # Constrain local U3 = 0.
    datum_csys = assembly.DatumCsysByThreePoints(
        name='CSYS-SymTheta',
        coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0),
        point1=(cos_t, 0.0, sin_t),       # local X direction (tangent)
        point2=(0.0, 1.0, 0.0))           # defines XY plane -> local Z = normal

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


def generate_model(job_name, defect_params=None, project_root=None,
                  global_seed=None, defect_seed=None, no_run=False,
                  opening_params=None):
    """Main function to generate the model."""
    g_seed = global_seed if global_seed is not None else GLOBAL_SEED
    d_seed = defect_seed if defect_seed is not None else DEFECT_SEED
    if g_seed != GLOBAL_SEED or d_seed != DEFECT_SEED:
        print("Mesh seeds: GLOBAL=%.0f mm, DEFECT=%.0f mm (override)" % (g_seed, d_seed))
    Mdb() # Clear
    model = mdb.models['Model-1']
    
    defect_type = defect_params.get('defect_type', 'debonding') if defect_params else None
    opening = opening_params or OPENING_PARAMS
    if opening and defect_params:
        defect_params = dict(defect_params)
        defect_params['_opening_params'] = opening
    if defect_params:
        print("Defect type: %s | theta=%.1f z=%.0f r=%.0f" % (
            defect_type, defect_params['theta_deg'],
            defect_params['z_center'], defect_params['radius']))
    if opening:
        print("Opening zone: z=%.0f theta=%.1f r=%.0f (defect excluded)" % (
            opening['z_center'], opening['theta_deg'], opening['radius']))

    # 1. Materials & Sections (healthy + defect-type-specific)
    create_materials(model)
    create_sections(model)
    if defect_params:
        create_defect_materials(model, defect_params)
        create_defect_sections(model, defect_params)

    # 2. Parts (geometry)
    p_inner, p_core, p_outer = create_parts(model)

    # 3. Partition parts at defect zone (BEFORE assembly, for section assignment)
    # All three parts are ALWAYS partitioned to ensure dependent=OFF instances
    # export mesh data to INP. Section overrides are applied only where needed.
    if defect_params:
        parts_to_partition = [
            ('shell', p_inner),
            ('solid', p_core),
            ('shell', p_outer),
        ]
        partition_defect_zone(parts_to_partition, defect_params)

    # 4. Assign sections (healthy baseline + defect-zone overrides)
    assign_all_sections(p_inner, p_core, p_outer, defect_params)

    # 5. Assembly
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)
    
    # 6. Interaction — Tie constraints (skin-core coupling)
    apply_tie_constraints(model, a, inst_inner, inst_core, inst_outer)

    # 7. Boundary Conditions (bottom fixity + sector symmetry)
    apply_bottom_fixity(model, a, inst_inner, inst_core, inst_outer)
    apply_symmetry_bcs(model, a, inst_inner, inst_core, inst_outer)

    # 8. Step & Loads
    model.StaticStep(name='Step-1', previous='Initial')
    # Thermal load: ascent heating (outer 120°C, inner 20°C) — applied in Step-1 after mesh
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'RF', 'TEMP'))

    # 9. Mesh
    a.seedPartInstance(regions=(inst_inner, inst_core, inst_outer), size=g_seed, deviationFactor=0.1)
    # Local refinement around defect (h≤D/2 for physical resolution)
    if defect_params:
        z_c, r_def = defect_params['z_center'], defect_params['radius']
        margin = 150.0
        z1, z2 = max(1.0, z_c - r_def - margin), min(TOTAL_HEIGHT - 1.0, z_c + r_def + margin)
        r_box = RADIUS + CORE_T + 200
        try:
            for inst in (inst_outer, inst_core, inst_inner):
                edges = inst.edges.getByBoundingBox(
                    xMin=-r_box, xMax=r_box, yMin=z1, yMax=z2, zMin=-r_box, zMax=r_box)
                if len(edges) > 0:
                    a.seedEdgeBySize(edges=edges, size=d_seed, constraint=FINER)
            print("Local mesh refinement: DEFECT_SEED=%.0f mm in defect zone (z=%.0f–%.0f)" % (d_seed, z1, z2))
        except Exception as e:
            print("Warning: Local seed skipped: %s" % str(e)[:60])
    a.generateMesh(regions=(inst_inner, inst_core, inst_outer))
    # Temperature IC and thermal load are handled by patch_inp_thermal.py post-processing.

    # 10. Job
    mdb.Job(name=job_name, model='Model-1', type=ANALYSIS, resultsFormat=ODB,
            numCpus=4, numDomains=4, multiprocessingMode=DEFAULT)
    
    # Save CAE
    mdb.saveAs(pathName=job_name + '.cae')

    # Write INP (patch and run done by run_batch when --no_run)
    print("Writing INP for job '%s'..." % job_name)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    inp_path = os.path.abspath(job_name + '.inp')
    if no_run:
        print("Skipping patch and job execution (--no_run)")
        return
    if os.path.exists(inp_path):
        # Apply thermal patch directly (import patch_inp function)
        patch_script = None
        proj_root = project_root or os.environ.get('PROJECT_ROOT') or os.environ.get('PAYLOAD2026_ROOT')
        if proj_root:
            patch_script = os.path.join(proj_root, 'scripts', 'patch_inp_thermal.py')
        if not patch_script or not os.path.exists(patch_script):
            inp_dir = os.path.dirname(inp_path)
            for _root in [inp_dir, os.path.dirname(inp_dir), os.path.dirname(os.path.dirname(inp_dir))]:
                if _root:
                    p = os.path.join(_root, 'scripts', 'patch_inp_thermal.py')
                    if os.path.exists(p):
                        patch_script = p
                        break
        if patch_script and os.path.exists(patch_script):
            try:
                _g = {'__name__': '_patch_module', '__file__': patch_script}
                exec(open(patch_script).read(), _g)
                _g['patch_inp'](inp_path)
                print("INP patched for thermal load")
            except Exception as e:
                print("Warning: INP patch failed: %s" % str(e)[:80])
    print("Running job '%s' with patched INP..." % job_name)
    import subprocess
    cwd = os.path.dirname(inp_path) or '.'
    r = subprocess.call(['abaqus', 'job=' + job_name, 'input=' + job_name + '.inp', 'cpus=4'], cwd=cwd)
    if r == 0:
        print("Job COMPLETED: %s.odb" % job_name)
    else:
        print("Job FAILED (exit code %d)" % r)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default='Job-H3-Fairing',
                        help='Job name (also used for --job_name by run_batch)')
    parser.add_argument('--job_name', type=str, default=None,
                        help='Alias for --job (run_batch compatibility)')
    parser.add_argument('--defect', type=str, default=None, help='JSON string or path to JSON')
    parser.add_argument('--param_file', type=str, default=None,
                        help='Path to JSON file with defect params (run_batch compatibility)')
    parser.add_argument('--project_root', type=str, default=None,
                        help='Project root for patch script (run_batch sets via env)')
    parser.add_argument('--global_seed', type=float, default=None,
                        help='Override GLOBAL_SEED (mm). Fine mesh: 12–15.')
    parser.add_argument('--defect_seed', type=float, default=None,
                        help='Override DEFECT_SEED (mm). Fine mesh: 5–8.')
    
    # When run via abaqus cae noGUI=script.py -- args
    args, unknown = parser.parse_known_args()
    
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
    
    opening_params = None
    if getattr(args, 'opening', None):
        try:
            opening_params = json.loads(args.opening) if isinstance(args.opening, str) else args.opening
        except (json.JSONDecodeError, TypeError):
            pass
    if defect_data and isinstance(defect_data, dict):
        if 'opening_params' in defect_data:
            opening_params = defect_data.get('opening_params') or opening_params
        if 'defect_params' in defect_data:
            defect_data = defect_data['defect_params']
    project_root = args.project_root or os.environ.get('PROJECT_ROOT') or os.environ.get('PAYLOAD2026_ROOT')
    generate_model(job_name, defect_data, project_root=project_root,
                   global_seed=args.global_seed, defect_seed=args.defect_seed,
                   no_run=getattr(args, 'no_run', False), opening_params=opening_params)
