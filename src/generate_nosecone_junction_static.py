# -*- coding: utf-8 -*-
# generate_nosecone_junction.py
# Abaqus Python script: H3 fairing barrel-nosecone junction FEM model
#
# Geometry: z = 3500-6500 mm (barrel upper 1500mm + ogive lower 1500mm)
# Sector: 30 deg (1/12 of full circumference)
# Key feature: curvature transition (single -> double curvature) at z = 5000 mm
#              + junction flange (Al-7075 bolted ring)
#
# Defect types: same 7 types as barrel model (generate_fairing_dataset.py)
# Materials: same CFRP/Al-HC sandwich + Al-7075 flange
#
# Usage:
#   abaqus cae noGUI=generate_nosecone_junction.py -- --job <name> [--defect <json>]
#
# Academic justification: barrel-nosecone junction is a critical zone for
# debonding due to curvature change, stress concentration, and manufacturing
# complexity (two-shell assembly).

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
# PARAMETERS
# ==============================================================================

# --- Full fairing geometry (for ogive calculations) ---
RADIUS = 2600.0       # mm — H3 fairing barrel inner radius
H_BARREL = 5000.0     # mm — full barrel height (z=0 to z=5000)
H_NOSE = 5400.0       # mm — full ogive nose cone height
TOTAL_HEIGHT = H_BARREL + H_NOSE  # 10400 mm

# Tangent ogive
OGIVE_RHO = (RADIUS**2 + H_NOSE**2) / (2.0 * RADIUS)
OGIVE_XC = RADIUS - OGIVE_RHO

# --- Junction model bounds ---
Z_BOTTOM = 3500.0     # mm — model lower boundary (barrel region)
Z_TOP = 6500.0        # mm — model upper boundary (ogive region)
Z_JUNCTION = H_BARREL  # 5000 mm — barrel-nosecone interface
SECTOR_ANGLE = 30.0   # deg (1/12)

# --- Sandwich structure ---
FACE_T = 1.0          # mm — CFRP face sheet thickness
CORE_T = 38.0         # mm — Al honeycomb core thickness

# --- CFRP (Toray T1000G class) ---
E1 = 160000.0         # MPa — fiber direction
E2 = 10000.0          # MPa — transverse
NU12 = 0.3
G12 = 5000.0          # MPa
G13 = 5000.0
G23 = 3000.0
RHO_CFRP = 1600e-12   # tonne/mm^3
CTE_CFRP = (-0.3e-6, 28e-6, 0.0)  # alpha11, alpha22, alpha12

# --- Al honeycomb core (cylindrical convention: 1=R, 2=theta, 3=Z) ---
E_CORE_1 = 1000.0     # through-thickness
E_CORE_2 = 1.0        # in-plane (theta)
E_CORE_3 = 1.0        # in-plane (Z)
NU_CORE_12 = 0.01
NU_CORE_13 = 0.01
NU_CORE_23 = 0.01
G_CORE_12 = 400.0     # R-theta shear
G_CORE_13 = 240.0     # R-Z shear
G_CORE_23 = 1.0       # theta-Z shear
RHO_CORE = 50e-12

# --- Al-7075 (junction flange) ---
E_AL7075 = 71700.0    # MPa
NU_AL7075 = 0.33
RHO_AL7075 = 2810e-12 # tonne/mm^3

# --- CFRP frame (quasi-isotropic equivalent) ---
E_FRAME = 60000.0     # MPa
NU_FRAME = 0.3
RHO_FRAME = 1600e-12

# --- Junction flange dimensions ---
FLANGE_Z = Z_JUNCTION  # 5000 mm
FLANGE_HEIGHT = 80.0   # mm (radial extent outward from outer skin)
FLANGE_THICK = 8.0     # mm (shell thickness)

# --- Ring frames ---
FRAME_POSITIONS = [4000.0, 4500.0, 5500.0, 6000.0]  # z positions (mm)
FRAME_HEIGHT = 50.0    # mm (radial extent)
FRAME_THICK = 3.0      # mm (shell thickness)

# --- Mesh ---
GLOBAL_SEED = 7.5      # mm
DEFECT_SEED = 5.0      # mm

# --- Thermal ---
TEMP_INITIAL = 20.0
TEMP_FINAL_OUTER = 120.0
TEMP_FINAL_INNER = 20.0
TEMP_FINAL_CORE = 70.0

# --- Layup ---
LAYUP = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]  # [45/0/-45/90]s


# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================

def get_radius_at_z(z):
    """Fairing inner radius at axial position z (ogive profile)."""
    if z <= H_BARREL:
        return RADIUS
    z_local = z - H_BARREL
    term = OGIVE_RHO**2 - z_local**2
    if term < 0:
        return 0.0
    return OGIVE_XC + math.sqrt(term)


def _cone_radius_at_z(z):
    """Conical approximation radius (used for inner skin, consistent with barrel model)."""
    if z <= H_BARREL:
        return RADIUS
    t = (z - H_BARREL) / H_NOSE
    return RADIUS * (1.0 - t)


def _ogive_radius_at_z(z, offset=0.0):
    """True ogive radius with optional radial offset (for core/outer skin)."""
    if z <= H_BARREL:
        return RADIUS + offset
    z_local = z - H_BARREL
    rho = OGIVE_RHO + offset
    term = rho**2 - z_local**2
    if term < 0:
        return 0.0
    return OGIVE_XC + math.sqrt(term)


# ==============================================================================
# DEFECT ZONE DETECTION
# ==============================================================================

def _point_in_defect_zone(x, y, z, defect_params):
    """Check if point (x,y,z) is inside the circular defect zone on the surface.
    Abaqus revolve convention: Y=axial, XZ=radial plane.
    """
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
    if not defect_params:
        return False
    pt = face.pointOn[0]
    return _point_in_defect_zone(pt[0], pt[1], pt[2], defect_params)


def is_cell_in_defect_zone(cell, defect_params):
    if not defect_params:
        return False
    pt = cell.pointOn[0]
    return _point_in_defect_zone(pt[0], pt[1], pt[2], defect_params)


# ==============================================================================
# MATERIALS
# ==============================================================================

def create_materials(model):
    """CFRP, honeycomb core, Al-7075 flange, CFRP frame materials."""
    # CFRP face sheets
    mat = model.Material(name='CFRP_T1000G')
    mat.Elastic(type=LAMINA, table=((E1, E2, NU12, G12, G13, G23), ))
    mat.Density(table=((RHO_CFRP, ), ))
    mat.Expansion(table=(CTE_CFRP, ))

    # Al honeycomb core
    mat = model.Material(name='AL_HONEYCOMB')
    mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
        E_CORE_1, E_CORE_2, E_CORE_3,
        NU_CORE_12, NU_CORE_13, NU_CORE_23,
        G_CORE_12, G_CORE_13, G_CORE_23), ))
    mat.Density(table=((RHO_CORE, ), ))
    mat.Expansion(table=((23e-6, ), ))

    # Al-7075 junction flange
    mat = model.Material(name='AL_7075')
    mat.Elastic(type=ISOTROPIC, table=((E_AL7075, NU_AL7075), ))
    mat.Density(table=((RHO_AL7075, ), ))
    mat.Expansion(table=((23.6e-6, ), ))

    # CFRP frame (quasi-isotropic equivalent)
    mat = model.Material(name='CFRP_FRAME')
    mat.Elastic(type=ISOTROPIC, table=((E_FRAME, NU_FRAME), ))
    mat.Density(table=((RHO_FRAME, ), ))
    mat.Expansion(table=((2e-6, ), ))


def create_sections(model):
    """Shell and solid sections for healthy structure."""
    # CFRP composite layup [45/0/-45/90]s
    entries = [section.SectionLayer(
        thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_T1000G')
        for ang in LAYUP]
    model.CompositeShellSection(
        name='Section-CFRP-Skin', preIntegrate=OFF,
        idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
        thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        temperature=GRADIENT, integrationRule=SIMPSON)

    # Core solid
    model.HomogeneousSolidSection(
        name='Section-Core', material='AL_HONEYCOMB', thickness=None)

    # Junction flange shell
    model.HomogeneousShellSection(
        name='Section-Flange', material='AL_7075',
        thickness=FLANGE_THICK, idealization=NO_IDEALIZATION,
        poissonDefinition=DEFAULT, integrationRule=SIMPSON,
        numIntPts=5)

    # Ring frame shell
    model.HomogeneousShellSection(
        name='Section-Frame', material='CFRP_FRAME',
        thickness=FRAME_THICK, idealization=NO_IDEALIZATION,
        poissonDefinition=DEFAULT, integrationRule=SIMPSON,
        numIntPts=5)


# ==============================================================================
# DEFECT MATERIALS & SECTIONS (7 types, same as barrel model)
# ==============================================================================

def create_defect_materials(model, defect_params):
    """Create defect-type-specific modified materials."""
    dt = defect_params.get('defect_type', 'debonding')

    if dt == 'debonding':
        mat = model.Material(name='CFRP_DEBONDED')
        mat.Elastic(type=LAMINA, table=((
            E1*0.01, E2*0.01, NU12, G12*0.01, G13*0.01, G23*0.01), ))
        mat.Density(table=((RHO_CFRP, ), ))
        mat.Expansion(table=(CTE_CFRP, ))

    elif dt == 'fod':
        sf = defect_params.get('stiffness_factor', 10.0)
        mat = model.Material(name='AL_HONEYCOMB_FOD')
        mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1*sf, E_CORE_2*sf, E_CORE_3*sf,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12*sf, G_CORE_13*sf, G_CORE_23*sf), ))
        mat.Density(table=((200e-12, ), ))
        mat.Expansion(table=((12e-6, ), ))

    elif dt == 'impact':
        dr = defect_params.get('damage_ratio', 0.3)
        mat = model.Material(name='CFRP_IMPACT_DAMAGED')
        mat.Elastic(type=LAMINA, table=((
            E1*0.7, E2*dr, NU12, G12*dr, G13*dr, G23*dr), ))
        mat.Density(table=((RHO_CFRP, ), ))
        mat.Expansion(table=(CTE_CFRP, ))
        mat = model.Material(name='AL_HONEYCOMB_CRUSHED')
        mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1*0.1, E_CORE_2*0.5, E_CORE_3*0.5,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12*0.1, G_CORE_13*0.1, G_CORE_23*0.5), ))
        mat.Density(table=((RHO_CORE, ), ))
        mat.Expansion(table=((23e-6, ), ))

    elif dt == 'delamination':
        depth = defect_params.get('delam_depth', 0.5)
        sr = max(0.05, 1.0 - depth)
        mat = model.Material(name='CFRP_DELAMINATED')
        mat.Elastic(type=LAMINA, table=((
            E1*0.9, E2*(0.3 + 0.5*(1-depth)), NU12,
            G12*sr, G13*sr, G23*sr), ))
        mat.Density(table=((RHO_CFRP, ), ))
        mat.Expansion(table=(CTE_CFRP, ))

    elif dt == 'inner_debond':
        mat = model.Material(name='CFRP_INNER_DEBONDED')
        mat.Elastic(type=LAMINA, table=((
            E1*0.01, E2*0.01, NU12, G12*0.01, G13*0.01, G23*0.01), ))
        mat.Density(table=((RHO_CFRP, ), ))
        mat.Expansion(table=(CTE_CFRP, ))

    elif dt == 'thermal_progression':
        mat = model.Material(name='CFRP_THERMAL_DAMAGED')
        mat.Elastic(type=LAMINA, table=((
            E1*0.05, E2*0.05, NU12, G12*0.05, G13*0.05, G23*0.05), ))
        mat.Density(table=((RHO_CFRP, ), ))
        mat.Expansion(table=((5e-6, 40e-6, 0.0), ))

    elif dt == 'acoustic_fatigue':
        sev = defect_params.get('fatigue_severity', 0.35)
        mat = model.Material(name='CFRP_ACOUSTIC_FATIGUED')
        mat.Elastic(type=LAMINA, table=((
            E1*(0.2 + 0.5*(1-sev)), E2*sev, NU12,
            G12*sev, G13*sev, G23*sev), ))
        mat.Density(table=((RHO_CFRP, ), ))
        mat.Expansion(table=(CTE_CFRP, ))

    print("Defect materials created: type=%s" % dt)


def create_defect_sections(model, defect_params):
    """Create sections for defect-zone materials."""
    dt = defect_params.get('defect_type', 'debonding')

    # Skin defect section names
    _skin_section_map = {
        'debonding': ('CFRP_DEBONDED', 'Section-CFRP-Debonded'),
        'impact': ('CFRP_IMPACT_DAMAGED', 'Section-CFRP-Impact'),
        'delamination': ('CFRP_DELAMINATED', 'Section-CFRP-Delaminated'),
        'inner_debond': ('CFRP_INNER_DEBONDED', 'Section-CFRP-InnerDebonded'),
        'thermal_progression': ('CFRP_THERMAL_DAMAGED', 'Section-CFRP-ThermalDamaged'),
        'acoustic_fatigue': ('CFRP_ACOUSTIC_FATIGUED', 'Section-CFRP-AcousticFatigued'),
    }
    if dt in _skin_section_map:
        mat_name, sec_name = _skin_section_map[dt]
        entries = [section.SectionLayer(
            thickness=FACE_T/8.0, orientAngle=ang, material=mat_name)
            for ang in LAYUP]
        model.CompositeShellSection(
            name=sec_name, preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)

    # Core defect sections
    if dt == 'fod':
        model.HomogeneousSolidSection(
            name='Section-Core-FOD', material='AL_HONEYCOMB_FOD', thickness=None)
    elif dt == 'impact':
        model.HomogeneousSolidSection(
            name='Section-Core-Crushed', material='AL_HONEYCOMB_CRUSHED', thickness=None)


# ==============================================================================
# GEOMETRY: JUNCTION MODEL PARTS
# ==============================================================================

def create_parts(model):
    """Create inner skin, core, outer skin for z=Z_BOTTOM..Z_TOP."""

    # Pre-compute ogive end points at Z_TOP
    r_cone_top = _cone_radius_at_z(Z_TOP)  # conical approx for inner skin
    r_ogive_top = _ogive_radius_at_z(Z_TOP, offset=0.0)  # true ogive inner
    r_ogive_top_outer = _ogive_radius_at_z(Z_TOP, offset=CORE_T)  # true ogive outer

    print("Junction geometry: z=[%.0f, %.0f], junction=%.0f" % (Z_BOTTOM, Z_TOP, Z_JUNCTION))
    print("  R_barrel=%.1f, R_cone_top=%.1f, R_ogive_top=%.1f, R_ogive_outer_top=%.1f" %
          (RADIUS, r_cone_top, r_ogive_top, r_ogive_top_outer))

    # ---------------------------------------------------------
    # Part 1: Inner Skin (Shell)
    #   Barrel: vertical line at r=RADIUS
    #   Ogive: conical line (consistent with barrel model)
    # ---------------------------------------------------------
    s1 = model.ConstrainedSketch(name='profile_inner', sheetSize=20000.0)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))
    s1.Line(point1=(RADIUS, Z_BOTTOM), point2=(RADIUS, Z_JUNCTION))
    s1.Line(point1=(RADIUS, Z_JUNCTION), point2=(r_cone_top, Z_TOP))

    p_inner = model.Part(name='Part-InnerSkin', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)

    # ---------------------------------------------------------
    # Part 2: Core (Solid)
    #   Closed cross-section between inner and outer ogive arcs
    # ---------------------------------------------------------
    s2 = model.ConstrainedSketch(name='profile_core', sheetSize=20000.0)
    s2.setPrimaryObject(option=STANDALONE)
    s2.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))

    # Bottom closure (horizontal at z=Z_BOTTOM)
    s2.Line(point1=(RADIUS, Z_BOTTOM), point2=(RADIUS + CORE_T, Z_BOTTOM))
    # Outer barrel (vertical)
    s2.Line(point1=(RADIUS + CORE_T, Z_BOTTOM), point2=(RADIUS + CORE_T, Z_JUNCTION))
    # Outer ogive arc (from barrel top to Z_TOP)
    rho_outer = OGIVE_RHO + CORE_T
    s2.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(RADIUS + CORE_T, Z_JUNCTION),
        point2=(r_ogive_top_outer, Z_TOP),
        direction=COUNTERCLOCKWISE)
    # Top closure (horizontal at z=Z_TOP)
    s2.Line(point1=(r_ogive_top_outer, Z_TOP), point2=(r_ogive_top, Z_TOP))
    # Inner ogive arc (reverse, from Z_TOP back to barrel top)
    s2.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(r_ogive_top, Z_TOP),
        point2=(RADIUS, Z_JUNCTION),
        direction=CLOCKWISE)
    # Inner barrel (vertical, downward)
    s2.Line(point1=(RADIUS, Z_JUNCTION), point2=(RADIUS, Z_BOTTOM))

    p_core = model.Part(name='Part-Core', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=s2, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)

    # ---------------------------------------------------------
    # Part 3: Outer Skin (Shell)
    #   Uses true ogive arc (same as barrel model)
    # ---------------------------------------------------------
    s3 = model.ConstrainedSketch(name='profile_outer', sheetSize=20000.0)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))
    s3.Line(point1=(RADIUS + CORE_T, Z_BOTTOM), point2=(RADIUS + CORE_T, Z_JUNCTION))
    s3.ArcByCenterEnds(
        center=(OGIVE_XC, H_BARREL),
        point1=(RADIUS + CORE_T, Z_JUNCTION),
        point2=(r_ogive_top_outer, Z_TOP),
        direction=COUNTERCLOCKWISE)

    p_outer = model.Part(name='Part-OuterSkin', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)

    print("Parts created: InnerSkin, Core, OuterSkin (sector=%.0f deg)" % SECTOR_ANGLE)
    return p_inner, p_core, p_outer


def create_junction_flange(model):
    """Create junction flange part — thick Al-7075 ring at z=Z_JUNCTION.
    Extends radially outward from outer skin surface."""
    r_base = RADIUS + CORE_T  # outer skin radius at barrel
    s = model.ConstrainedSketch(name='profile_flange', sheetSize=20000.0)
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))
    s.Line(point1=(r_base, FLANGE_Z), point2=(r_base + FLANGE_HEIGHT, FLANGE_Z))

    p = model.Part(name='Part-Flange', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseShellRevolve(sketch=s, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)

    # Section assignment
    region = p.Set(faces=p.faces, name='Set-All')
    p.SectionAssignment(region=region, sectionName='Section-Flange')
    print("Junction flange: z=%.0f, height=%.0f mm, t=%.1f mm (Al-7075)" %
          (FLANGE_Z, FLANGE_HEIGHT, FLANGE_THICK))
    return p


def create_ring_frames(model):
    """Create CFRP ring frames at specified axial positions."""
    frame_parts = []
    for i, z_pos in enumerate(FRAME_POSITIONS):
        r_base = _ogive_radius_at_z(z_pos, offset=CORE_T)
        name = 'Part-Frame-%d' % (i + 1)

        s = model.ConstrainedSketch(name='profile_frame_%d' % i, sheetSize=20000.0)
        s.setPrimaryObject(option=STANDALONE)
        s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT + 1000.0))
        s.Line(point1=(r_base, z_pos), point2=(r_base + FRAME_HEIGHT, z_pos))

        p = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        p.BaseShellRevolve(sketch=s, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)

        region = p.Set(faces=p.faces, name='Set-All')
        p.SectionAssignment(region=region, sectionName='Section-Frame')
        frame_parts.append(p)

    print("Ring frames: %d frames at z=%s" %
          (len(FRAME_POSITIONS), [int(z) for z in FRAME_POSITIONS]))
    return frame_parts


# ==============================================================================
# PARTITIONING & SECTION ASSIGNMENT
# ==============================================================================

def _create_cylindrical_csys(part, name='CylCS'):
    """Cylindrical CSYS: 1=R, 2=theta, 3=Z(=Y axial).
    Same convention as barrel model."""
    datum = part.DatumCsysByThreePoints(
        name=name, coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(0.0, 0.0, 1.0),
        point2=(1.0, 0.0, 0.0))
    return datum


def partition_defect_zone(parts_to_partition, defect_params):
    """Partition parts at the defect zone boundaries for section reassignment."""
    z_c = defect_params['z_center']
    theta_deg = defect_params['theta_deg']
    r_def = defect_params['radius']

    r_local = get_radius_at_z(z_c) + CORE_T
    if r_local < 1.0:
        r_local = RADIUS
    theta_rad = math.radians(theta_deg)
    d_theta = min(r_def / r_local, math.radians(15.0))
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
                print("  Partition warning (%s): %s" % (geom_type, str(e)[:80]))

    print("Defect zone partitioned: z=%.0f theta=%.1f r=%.0f" %
          (z_c, theta_deg, r_def))


def assign_all_sections(p_inner, p_core, p_outer, defect_params):
    """Assign sections with cylindrical material orientation."""
    cyl_core = _create_cylindrical_csys(p_core, 'CylCS-Core')
    cyl_inner = _create_cylindrical_csys(p_inner, 'CylCS-InnerSkin')
    cyl_outer = _create_cylindrical_csys(p_outer, 'CylCS-OuterSkin')

    # Inner skin
    region = p_inner.Set(faces=p_inner.faces, name='Set-All')
    p_inner.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_inner.MaterialOrientation(region=region, orientationType=SYSTEM,
                                axis=AXIS_1, additionalRotationType=ROTATION_NONE,
                                localCsys=p_inner.datums[cyl_inner.id])

    # Core
    region = p_core.Set(cells=p_core.cells, name='Set-All')
    p_core.SectionAssignment(region=region, sectionName='Section-Core')
    p_core.MaterialOrientation(region=region, orientationType=SYSTEM, axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE,
                                localCsys=p_core.datums[cyl_core.id])

    # Outer skin
    region = p_outer.Set(faces=p_outer.faces, name='Set-All')
    p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_outer.MaterialOrientation(region=region, orientationType=SYSTEM,
                                axis=AXIS_1, additionalRotationType=ROTATION_NONE,
                                localCsys=p_outer.datums[cyl_outer.id])

    if not defect_params:
        return

    dt = defect_params.get('defect_type', 'debonding')

    # --- Outer skin defect overrides ---
    outer_defects = ('debonding', 'impact', 'delamination',
                     'thermal_progression', 'acoustic_fatigue')
    if dt in outer_defects:
        sec_map = {
            'debonding': 'Section-CFRP-Debonded',
            'impact': 'Section-CFRP-Impact',
            'delamination': 'Section-CFRP-Delaminated',
            'thermal_progression': 'Section-CFRP-ThermalDamaged',
            'acoustic_fatigue': 'Section-CFRP-AcousticFatigued',
        }
        sec_name = sec_map[dt]
        dfaces = [f for f in p_outer.faces if is_face_in_defect_zone(f, defect_params)]
        if dfaces:
            pts = tuple((f.pointOn[0],) for f in dfaces)
            fseq = p_outer.faces.findAt(*pts)
            reg = p_outer.Set(faces=fseq, name='Set-DefectZone-Skin')
            p_outer.SectionAssignment(region=reg, sectionName=sec_name)
            p_outer.MaterialOrientation(region=reg, orientationType=SYSTEM,
                                        axis=AXIS_1, additionalRotationType=ROTATION_NONE,
                                        localCsys=p_outer.datums[cyl_outer.id])
            print("  %s: %d outer skin faces -> %s" % (dt, len(dfaces), sec_name))
        else:
            print("  Warning: no outer skin faces in defect zone")

    # --- Inner skin defect (inner_debond) ---
    if dt == 'inner_debond':
        dfaces = [f for f in p_inner.faces if is_face_in_defect_zone(f, defect_params)]
        if dfaces:
            pts = tuple((f.pointOn[0],) for f in dfaces)
            fseq = p_inner.faces.findAt(*pts)
            reg = p_inner.Set(faces=fseq, name='Set-DefectZone-InnerSkin')
            p_inner.SectionAssignment(region=reg, sectionName='Section-CFRP-InnerDebonded')
            p_inner.MaterialOrientation(region=reg, orientationType=SYSTEM,
                                        axis=AXIS_1, additionalRotationType=ROTATION_NONE,
                                        localCsys=p_inner.datums[cyl_inner.id])
            print("  inner_debond: %d inner skin faces" % len(dfaces))

    # --- Core defects (fod, impact) ---
    if dt in ('fod', 'impact'):
        sec_name = 'Section-Core-FOD' if dt == 'fod' else 'Section-Core-Crushed'
        dcells = [c for c in p_core.cells if is_cell_in_defect_zone(c, defect_params)]
        if dcells:
            pts = tuple((c.pointOn[0],) for c in dcells)
            cseq = p_core.cells.findAt(*pts)
            reg = p_core.Set(cells=cseq, name='Set-DefectZone-Core')
            p_core.SectionAssignment(region=reg, sectionName=sec_name)
            print("  %s: %d core cells -> %s" % (dt, len(dcells), sec_name))


# ==============================================================================
# TIE CONSTRAINTS
# ==============================================================================

def _classify_core_faces(inst_core):
    """Identify core inner and outer surface faces using known geometry points."""
    theta_mid = math.radians(SECTOR_ANGLE / 2.0)
    cos_t = math.cos(theta_mid)
    sin_t = math.sin(theta_mid)

    inner_pts = []
    outer_pts = []

    # Barrel portion (z < Z_JUNCTION)
    y_barrel = (Z_BOTTOM + Z_JUNCTION) / 2.0
    inner_pts.append(((RADIUS * cos_t, y_barrel, RADIUS * sin_t),))
    outer_pts.append((((RADIUS + CORE_T) * cos_t, y_barrel, (RADIUS + CORE_T) * sin_t),))

    # Ogive portion (z > Z_JUNCTION)
    for frac in [0.3, 0.7]:
        y_og = Z_JUNCTION + (Z_TOP - Z_JUNCTION) * frac
        z_local = y_og - H_BARREL
        # Inner ogive
        term_in = OGIVE_RHO**2 - z_local**2
        if term_in > 0:
            r_in = OGIVE_XC + math.sqrt(term_in)
            if r_in > 50:
                inner_pts.append(((r_in * cos_t, y_og, r_in * sin_t),))
        # Outer ogive
        rho_outer = OGIVE_RHO + CORE_T
        term_out = rho_outer**2 - z_local**2
        if term_out > 0:
            r_out = OGIVE_XC + math.sqrt(term_out)
            if r_out > 50:
                outer_pts.append(((r_out * cos_t, y_og, r_out * sin_t),))

    return inner_pts, outer_pts


def apply_tie_constraints(model, assembly, inst_inner, inst_core, inst_outer,
                          inst_flange, inst_frames):
    """Tie skins to core, and flange/frames to outer skin."""
    # --- Skin-core ties ---
    core_inner_pts, core_outer_pts = _classify_core_faces(inst_core)

    surf_inner_skin = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')
    surf_outer_skin = assembly.Surface(
        side2Faces=inst_outer.faces, name='Surf-OuterSkin')

    if core_inner_pts:
        cseq = inst_core.faces.findAt(*core_inner_pts)
        surf = assembly.Surface(side1Faces=cseq, name='Surf-Core-Inner')
        model.Tie(name='Tie-InnerSkin-Core',
                  main=surf, secondary=surf_inner_skin,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie: InnerSkin <-> Core(inner): %d faces" % len(core_inner_pts))

    if core_outer_pts:
        cseq = inst_core.faces.findAt(*core_outer_pts)
        surf = assembly.Surface(side1Faces=cseq, name='Surf-Core-Outer')
        model.Tie(name='Tie-Core-OuterSkin',
                  main=surf, secondary=surf_outer_skin,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie: Core(outer) <-> OuterSkin: %d faces" % len(core_outer_pts))

    # --- Flange tie (flange inner edge to outer skin at z=FLANGE_Z) ---
    if inst_flange is not None:
        surf_flange = assembly.Surface(
            side2Faces=inst_flange.faces, name='Surf-Flange')
        surf_outer_at_junc = assembly.Surface(
            side1Faces=inst_outer.faces, name='Surf-OuterSkin-Flange')
        model.Tie(name='Tie-Flange-OuterSkin',
                  main=surf_outer_at_junc, secondary=surf_flange,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie: Flange <-> OuterSkin")

    # --- Frame ties ---
    for i, inst_frame in enumerate(inst_frames):
        surf_frame = assembly.Surface(
            side2Faces=inst_frame.faces, name='Surf-Frame-%d' % (i+1))
        model.Tie(name='Tie-Frame-%d' % (i+1),
                  main=surf_outer_skin, secondary=surf_frame,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
    if inst_frames:
        print("Tie: %d ring frames <-> OuterSkin" % len(inst_frames))


# ==============================================================================
# BOUNDARY CONDITIONS
# ==============================================================================

def apply_bottom_fixity(model, assembly, inst_inner, inst_core, inst_outer):
    """Fix bottom face (y=Z_BOTTOM) — clamped boundary for truncated model."""
    r_box = RADIUS + CORE_T + 500.0
    y_tol = 1.0
    set_kwargs = {}

    # Shell edges at y=Z_BOTTOM
    edge_seq = None
    for inst in [inst_inner, inst_outer]:
        try:
            edges = inst.edges.getByBoundingBox(
                xMin=-r_box, xMax=r_box,
                yMin=Z_BOTTOM - y_tol, yMax=Z_BOTTOM + y_tol,
                zMin=-r_box, zMax=r_box)
            if len(edges) > 0:
                edge_seq = edges if edge_seq is None else edge_seq + edges
        except Exception as e:
            print("  BC edge warning: %s" % str(e)[:60])
    if edge_seq is not None and len(edge_seq) > 0:
        set_kwargs['edges'] = edge_seq

    # Core faces at y=Z_BOTTOM
    bot_pts = []
    for f in inst_core.faces:
        try:
            pt = f.pointOn[0]
            if abs(pt[1] - Z_BOTTOM) < y_tol:
                bot_pts.append((pt,))
        except:
            pass
    if bot_pts:
        fseq = inst_core.faces.findAt(*bot_pts)
        if len(fseq) > 0:
            set_kwargs['faces'] = fseq

    if set_kwargs:
        bot_set = assembly.Set(name='BC_Bottom', **set_kwargs)
        model.DisplacementBC(name='Fix_Bottom', createStepName='Initial',
                             region=bot_set, u1=0, u2=0, u3=0)
        print("BC: Fixed at y=%.0f" % Z_BOTTOM)
    else:
        print("Warning: No geometry found at y=%.0f for bottom BC" % Z_BOTTOM)


def apply_top_fixity(model, assembly, inst_inner, inst_core, inst_outer):
    """Fix top face (y=Z_TOP) — clamped boundary for truncated model."""
    r_box = RADIUS + CORE_T + 500.0
    y_tol = 1.0
    set_kwargs = {}

    edge_seq = None
    for inst in [inst_inner, inst_outer]:
        try:
            edges = inst.edges.getByBoundingBox(
                xMin=-r_box, xMax=r_box,
                yMin=Z_TOP - y_tol, yMax=Z_TOP + y_tol,
                zMin=-r_box, zMax=r_box)
            if len(edges) > 0:
                edge_seq = edges if edge_seq is None else edge_seq + edges
        except Exception as e:
            print("  BC edge warning: %s" % str(e)[:60])
    if edge_seq is not None and len(edge_seq) > 0:
        set_kwargs['edges'] = edge_seq

    top_pts = []
    for f in inst_core.faces:
        try:
            pt = f.pointOn[0]
            if abs(pt[1] - Z_TOP) < y_tol:
                top_pts.append((pt,))
        except:
            pass
    if top_pts:
        fseq = inst_core.faces.findAt(*top_pts)
        if len(fseq) > 0:
            set_kwargs['faces'] = fseq

    if set_kwargs:
        top_set = assembly.Set(name='BC_Top', **set_kwargs)
        model.DisplacementBC(name='Fix_Top', createStepName='Initial',
                             region=top_set, u1=0, u2=0, u3=0)
        print("BC: Fixed at y=%.0f" % Z_TOP)


def _select_boundary_nodes(inst, theta_target, tol_coord=1.0):
    """Select mesh nodes on a sector boundary by coordinate check."""
    labels = []
    theta_tol = 0.02
    for n in inst.nodes:
        x, y, z = n.coordinates
        r = math.sqrt(x**2 + z**2)
        if r < 1.0:
            continue
        if theta_target < 0.01:
            if abs(z) < tol_coord and x > 0:
                labels.append(n.label)
        else:
            theta = math.atan2(z, x)
            if abs(theta - theta_target) < theta_tol:
                labels.append(n.label)
    return labels


def apply_symmetry_bcs(model, assembly, all_insts):
    """Symmetry BCs on sector cut faces (theta=0 and theta=SECTOR_ANGLE)."""
    theta_rad = math.radians(SECTOR_ANGLE)

    # theta=0: ZSYMM (U3=UR1=UR2=0)
    total_0 = 0
    for i, inst in enumerate(all_insts):
        labels = _select_boundary_nodes(inst, 0.0)
        if labels:
            seq = inst.nodes.sequenceFromLabels(tuple(labels))
            nset = assembly.Set(name='BC_Sym_T0_%s' % inst.name, nodes=seq)
            model.DisplacementBC(name='Sym_T0_%d' % i, createStepName='Initial',
                                 region=nset, u3=0, ur1=0, ur2=0)
            total_0 += len(labels)
    print("BC: Symmetry theta=0: %d nodes" % total_0)

    # theta=SECTOR_ANGLE: local CSYS symmetry
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    datum_csys = assembly.DatumCsysByThreePoints(
        name='CSYS-SymTheta',
        coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0),
        point1=(cos_t, 0.0, sin_t),
        point2=(0.0, 1.0, 0.0))

    total_a = 0
    for i, inst in enumerate(all_insts):
        labels = _select_boundary_nodes(inst, theta_rad)
        if labels:
            seq = inst.nodes.sequenceFromLabels(tuple(labels))
            nset = assembly.Set(name='BC_Sym_Ta_%s' % inst.name, nodes=seq)
            model.DisplacementBC(name='Sym_Ta_%d' % i, createStepName='Initial',
                                 region=nset, u3=0, ur1=0, ur2=0,
                                 localCsys=assembly.datums[datum_csys.id])
            total_a += len(labels)
    print("BC: Symmetry theta=%.0f: %d nodes" % (SECTOR_ANGLE, total_a))


# ==============================================================================
# MAIN MODEL GENERATION
# ==============================================================================

def generate_model(job_name, defect_params=None, project_root=None,
                   global_seed=None, defect_seed=None, no_run=False):
    """Generate junction FEM model."""
    g_seed = global_seed if global_seed is not None else GLOBAL_SEED
    d_seed = defect_seed if defect_seed is not None else DEFECT_SEED
    print("=" * 60)
    print("JUNCTION MODEL: %s" % job_name)
    print("  z=[%.0f, %.0f], sector=%.0f deg" % (Z_BOTTOM, Z_TOP, SECTOR_ANGLE))
    print("  mesh: global=%.1f mm, defect=%.1f mm" % (g_seed, d_seed))
    print("=" * 60)

    dt = defect_params.get('defect_type', 'debonding') if defect_params else None
    if defect_params:
        print("Defect: %s | theta=%.1f z=%.0f r=%.0f" % (
            dt, defect_params['theta_deg'],
            defect_params['z_center'], defect_params['radius']))

    Mdb()
    model = mdb.models['Model-1']

    # 1. Materials & Sections
    create_materials(model)
    create_sections(model)
    if defect_params:
        create_defect_materials(model, defect_params)
        create_defect_sections(model, defect_params)

    # 2. Parts
    p_inner, p_core, p_outer = create_parts(model)
    p_flange = create_junction_flange(model)
    p_frames = create_ring_frames(model)

    # 3. Partition defect zone
    if defect_params:
        partition_defect_zone([
            ('shell', p_inner),
            ('solid', p_core),
            ('shell', p_outer),
        ], defect_params)

    # 4. Section assignment
    assign_all_sections(p_inner, p_core, p_outer, defect_params)

    # 5. Assembly
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)
    inst_flange = a.Instance(name='Part-Flange-1', part=p_flange, dependent=OFF)
    inst_frames = []
    for i, p_frame in enumerate(p_frames):
        inst = a.Instance(name='Part-Frame-%d-1' % (i+1), part=p_frame, dependent=OFF)
        inst_frames.append(inst)

    # 6. Mesh
    all_shell_insts = (inst_inner, inst_outer, inst_flange) + tuple(inst_frames)
    a.seedPartInstance(regions=all_shell_insts + (inst_core,), size=g_seed, deviationFactor=0.1)

    # Local refinement around defect
    if defect_params:
        z_c, r_def = defect_params['z_center'], defect_params['radius']
        margin = 100.0
        z1 = max(Z_BOTTOM + 1.0, z_c - r_def - margin)
        z2 = min(Z_TOP - 1.0, z_c + r_def + margin)
        r_box = RADIUS + CORE_T + 200
        try:
            for inst in (inst_outer, inst_core, inst_inner):
                edges = inst.edges.getByBoundingBox(
                    xMin=-r_box, xMax=r_box, yMin=z1, yMax=z2,
                    zMin=-r_box, zMax=r_box)
                if len(edges) > 0:
                    a.seedEdgeBySize(edges=edges, size=d_seed, constraint=FINER)
            print("Local refinement: seed=%.0f in z=[%.0f, %.0f]" % (d_seed, z1, z2))
        except Exception as e:
            print("Warning: local seed skipped: %s" % str(e)[:60])

    # Core mesh: SWEEP C3D8R -> FREE C3D4 fallback
    core_seed = max(g_seed, CORE_T * 4)
    a.seedPartInstance(regions=(inst_core,), size=core_seed, deviationFactor=0.1)
    core_meshed = False
    try:
        a.setMeshControls(regions=inst_core.cells, elemShape=HEX, technique=SWEEP)
        a.setElementType(
            regions=(inst_core.cells,),
            elemTypes=(ElemType(elemCode=C3D8R, elemLibrary=STANDARD),
                       ElemType(elemCode=C3D6, elemLibrary=STANDARD)))
        a.generateMesh(regions=(inst_core,))
        if len(inst_core.nodes) > 0:
            core_meshed = True
            print("Core mesh: SWEEP C3D8R seed=%.0f (%d nodes, %d elems)" %
                  (core_seed, len(inst_core.nodes), len(inst_core.elements)))
    except Exception as e:
        print("Core SWEEP failed: %s" % str(e)[:60])

    if not core_meshed:
        try:
            a.deleteMesh(regions=(inst_core,))
        except:
            pass
        a.seedPartInstance(regions=(inst_core,), size=core_seed, deviationFactor=0.1)
        a.setMeshControls(regions=inst_core.cells, elemShape=TET, technique=FREE)
        a.setElementType(
            regions=(inst_core.cells,),
            elemTypes=(ElemType(elemCode=C3D4, elemLibrary=STANDARD),))
        a.generateMesh(regions=(inst_core,))
        if len(inst_core.nodes) > 0:
            core_meshed = True
            print("Core mesh: FREE C3D4 seed=%.0f (%d nodes, %d elems)" %
                  (core_seed, len(inst_core.nodes), len(inst_core.elements)))

    # Shell meshes
    a.generateMesh(regions=all_shell_insts)
    print("Shell mesh: InnerSkin=%d, OuterSkin=%d, Flange=%d nodes" %
          (len(inst_inner.nodes), len(inst_outer.nodes), len(inst_flange.nodes)))
    for i, inst_f in enumerate(inst_frames):
        print("  Frame-%d: %d nodes" % (i+1, len(inst_f.nodes)))

    # 7. Tie constraints
    apply_tie_constraints(model, a, inst_inner, inst_core, inst_outer,
                          inst_flange, inst_frames)

    # 8. Boundary conditions
    all_insts_for_bc = [inst_inner, inst_core, inst_outer,
                        inst_flange] + inst_frames
    apply_bottom_fixity(model, a, inst_inner, inst_core, inst_outer)
    apply_top_fixity(model, a, inst_inner, inst_core, inst_outer)
    apply_symmetry_bcs(model, a, all_insts_for_bc)

    # 9. Step & output
    model.StaticStep(name='Step-1', previous='Initial')
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'U', 'RF', 'TEMP'))

    # 10. Job
    mdb.Job(name=job_name, model='Model-1', type=ANALYSIS, resultsFormat=ODB,
            numCpus=4, numDomains=4, multiprocessingMode=DEFAULT)

    # Save CAE
    mdb.saveAs(pathName=job_name + '.cae')

    # Write INP
    print("Writing INP for job '%s'..." % job_name)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    inp_path = os.path.abspath(job_name + '.inp')

    if no_run:
        print("INP written. Skipping job execution (--no_run).")
        return

    # Apply thermal patch
    proj_root = project_root or os.environ.get('PROJECT_ROOT') or os.environ.get('PAYLOAD2026_ROOT')
    patch_script = None
    if proj_root:
        patch_script = os.path.join(proj_root, 'scripts', 'patch_inp_thermal.py')
    if not patch_script or not os.path.exists(patch_script):
        for _root in [os.path.dirname(inp_path),
                      os.path.dirname(os.path.dirname(inp_path)),
                      os.path.dirname(os.path.dirname(os.path.dirname(inp_path)))]:
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

    # Run job
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
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='H3 fairing barrel-nosecone junction FEM model generator')
    parser.add_argument('--job', type=str, default='Job-Junction',
                        help='Job name')
    parser.add_argument('--job_name', type=str, default=None,
                        help='Alias for --job')
    parser.add_argument('--defect', type=str, default=None,
                        help='JSON string or path to JSON file with defect params')
    parser.add_argument('--param_file', type=str, default=None,
                        help='Path to JSON file with defect params')
    parser.add_argument('--project_root', type=str, default=None,
                        help='Project root for patch script location')
    parser.add_argument('--global_seed', type=float, default=None,
                        help='Override GLOBAL_SEED (mm)')
    parser.add_argument('--defect_seed', type=float, default=None,
                        help='Override DEFECT_SEED (mm)')
    parser.add_argument('--no_run', action='store_true',
                        help='Write INP only, do not run solver')

    args, unknown = parser.parse_known_args()
    job_name = args.job_name if args.job_name is not None else args.job

    defect_data = None
    if args.param_file:
        if os.path.exists(args.param_file):
            with open(args.param_file, 'r') as f:
                defect_data = json.load(f)
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

    project_root = (args.project_root
                    or os.environ.get('PROJECT_ROOT')
                    or os.environ.get('PAYLOAD2026_ROOT'))

    generate_model(job_name, defect_data, project_root=project_root,
                   global_seed=args.global_seed, defect_seed=args.defect_seed,
                   no_run=args.no_run)
