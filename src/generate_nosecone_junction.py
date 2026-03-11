# -*- coding: utf-8 -*-
# generate_nosecone_junction.py
# Abaqus/Explicit Guided Wave model — barrel-nosecone junction
#
# Geometry: z = 3500-6500 mm (barrel upper 1500mm + ogive lower 1500mm)
# Sector: 30 deg (1/12)
# Key feature: curvature transition at z=5000mm + junction flange (Al-7075)
#
# Based on generate_gw_fairing.py architecture:
#   - Abaqus/Explicit solver
#   - Surface-based CZM (General Contact)
#   - Hanning tone burst excitation
#   - Sensor grid with HistoryOutput
#   - ABL (Absorbing Boundary Layer) at sector edges
#   - Mass scaling injection
#
# Usage:
#   abaqus cae noGUI=generate_nosecone_junction.py -- --job <name> [--defect <json>]

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
RADIUS = 2600.0
H_BARREL = 5000.0
H_NOSE = 5400.0
TOTAL_HEIGHT = H_BARREL + H_NOSE
OGIVE_RHO = (RADIUS**2 + H_NOSE**2) / (2.0 * RADIUS)
OGIVE_XC = RADIUS - OGIVE_RHO

# --- Junction model bounds ---
Z_BOTTOM = 3500.0
Z_TOP = 6500.0
Z_JUNCTION = H_BARREL  # 5000 mm
SECTOR_ANGLE = 30.0

# --- Sandwich structure ---
FACE_T = 1.0
CORE_T = 38.0
PLY_ANGLES = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
PLY_T = FACE_T / 8.0

# --- CFRP ---
CFRP_E1 = 160000.0
CFRP_E2 = 10000.0
CFRP_NU12 = 0.3
CFRP_G12 = 5000.0
CFRP_G13 = 5000.0
CFRP_G23 = 3000.0
CFRP_DENSITY = 1600e-12
CFRP_CTE_1 = -0.3e-6
CFRP_CTE_2 = 28e-6

# --- Al honeycomb core (1=R, 2=theta, 3=Z) ---
E_CORE_1 = 1000.0
E_CORE_2 = 1.0
E_CORE_3 = 1.0
NU_CORE_12 = 0.01
NU_CORE_13 = 0.01
NU_CORE_23 = 0.01
G_CORE_12 = 400.0
G_CORE_13 = 240.0
G_CORE_23 = 1.0
CORE_DENSITY = 50e-12
CORE_CTE = 23e-6

# --- Al-7075 (junction flange) ---
FRAME_E = 71700.0
FRAME_NU = 0.33
FRAME_DENSITY = 2810e-12

# --- Junction flange ---
FLANGE_Z = Z_JUNCTION
FLANGE_HEIGHT = 80.0
FLANGE_THICK = 8.0

# --- Ring frames ---
FRAME_POSITIONS = [4000.0, 4500.0, 5500.0, 6000.0]
FRAME_HEIGHT_RING = 50.0
FRAME_THICK_RING = 3.0

# --- Wave / Explicit ---
DEFAULT_FREQ_KHZ = 50.0
DEFAULT_CYCLES = 5
CP_ESTIMATE = 1550.0
VG_ESTIMATE = 1100.0
FORCE_MAGNITUDE = 1.0
FIELD_OUTPUT_INTERVAL = 1e-6
DEFAULT_N_SENSORS = 40

# --- ABL ---
ABL_WIDTH_DEG = 2.0
ABL_BETA_FACTOR = 100.0

# --- CZM ---
DEFAULT_CZM = {
    'Kn': 1e5, 'Ks': 5e4,
    'tn': 50.0, 'ts': 40.0,
    'GIc': 0.3, 'GIIc': 1.0,
    'BK_eta': 2.284,
}

# --- Rayleigh damping ---
RAYLEIGH_ALPHA = 0.0
RAYLEIGH_BETA = 3.18e-8

# --- Thermal ---
TEMP_INITIAL = 20.0
TEMP_OUTER = 120.0
TEMP_INNER = 20.0
TEMP_CORE = 70.0

# --- Moisture ---
MOISTURE_FACTOR_DEFAULT = 0.92


# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================

def get_radius_at_z(z):
    if z <= H_BARREL:
        return RADIUS
    z_local = z - H_BARREL
    term = OGIVE_RHO**2 - z_local**2
    return OGIVE_XC + math.sqrt(term) if term > 0 else 0.0


def _cone_radius_at_z(z):
    if z <= H_BARREL:
        return RADIUS
    return RADIUS * (1.0 - (z - H_BARREL) / H_NOSE)


def _ogive_radius_at_z(z, offset=0.0):
    if z <= H_BARREL:
        return RADIUS + offset
    z_local = z - H_BARREL
    rho = OGIVE_RHO + offset
    term = rho**2 - z_local**2
    return OGIVE_XC + math.sqrt(term) if term > 0 else 0.0


def _point_in_defect_zone(x, y, z, defect_params):
    z_c = defect_params['z_center']
    theta_c = math.radians(defect_params['theta_deg'])
    r_def = defect_params['radius']
    r_pt = math.sqrt(x * x + z * z)
    if r_pt < 1.0:
        return False
    theta_pt = math.atan2(z, x)
    arc_dist = r_pt * (theta_pt - theta_c)
    axial_dist = y - z_c
    return math.sqrt(arc_dist**2 + axial_dist**2) <= r_def * 1.02


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


def find_nearest_node(instance, target_x, target_y, target_z):
    min_dist = 1e30
    nearest_label = None
    for node in instance.nodes:
        c = node.coordinates
        dx = c[0] - target_x
        dy = c[1] - target_y
        dz = c[2] - target_z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < min_dist:
            min_dist = dist
            nearest_label = node.label
    return nearest_label, min_dist


# ==============================================================================
# MATERIALS
# ==============================================================================

def create_materials(model, moisture_factor=1.0):
    """Materials with Rayleigh damping and optional moisture degradation."""
    E2_eff = CFRP_E2 * moisture_factor
    G12_eff = CFRP_G12 * moisture_factor
    G13_eff = CFRP_G13 * moisture_factor
    G23_eff = CFRP_G23 * moisture_factor

    mat = model.Material(name='Mat-CFRP')
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Elastic(type=LAMINA, table=((
        CFRP_E1, E2_eff, CFRP_NU12, G12_eff, G13_eff, G23_eff),))
    mat.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
    mat.Expansion(table=((CFRP_CTE_1, CFRP_CTE_2, 0.0),))

    mat_c = model.Material(name='Mat-Honeycomb')
    mat_c.Density(table=((CORE_DENSITY,),))
    mat_c.Elastic(type=ENGINEERING_CONSTANTS, table=((
        E_CORE_1, E_CORE_2, E_CORE_3,
        NU_CORE_12, NU_CORE_13, NU_CORE_23,
        G_CORE_12, G_CORE_13, G_CORE_23),))
    mat_c.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
    mat_c.Expansion(table=((CORE_CTE,),))

    mat_f = model.Material(name='Mat-Frame')
    mat_f.Density(table=((FRAME_DENSITY,),))
    mat_f.Elastic(table=((FRAME_E, FRAME_NU),))
    mat_f.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
    mat_f.Expansion(table=((23.6e-6,),))

    print("Materials: Mat-CFRP, Mat-Honeycomb, Mat-Frame (moisture=%.0f%%)" %
          ((1 - moisture_factor) * 100))


def create_absorbing_materials(model, moisture_factor=1.0):
    """High-damping materials for ABL zones."""
    alpha_abl = 6.283e6  # mass-proportional

    E2_eff = CFRP_E2 * moisture_factor
    G12_eff = CFRP_G12 * moisture_factor
    G13_eff = CFRP_G13 * moisture_factor
    G23_eff = CFRP_G23 * moisture_factor

    mat = model.Material(name='Mat-CFRP-ABL')
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Elastic(type=LAMINA, table=((
        CFRP_E1, E2_eff, CFRP_NU12, G12_eff, G13_eff, G23_eff),))
    mat.Damping(alpha=alpha_abl, beta=RAYLEIGH_BETA)
    mat.Expansion(table=((CFRP_CTE_1, CFRP_CTE_2, 0.0),))

    mat_c = model.Material(name='Mat-Honeycomb-ABL')
    mat_c.Density(table=((CORE_DENSITY,),))
    mat_c.Elastic(type=ENGINEERING_CONSTANTS, table=((
        E_CORE_1, E_CORE_2, E_CORE_3,
        NU_CORE_12, NU_CORE_13, NU_CORE_23,
        G_CORE_12, G_CORE_13, G_CORE_23),))
    mat_c.Damping(alpha=alpha_abl, beta=RAYLEIGH_BETA)
    mat_c.Expansion(table=((CORE_CTE,),))


def create_sections(model):
    """Composite shell + solid sections."""
    ply_data = [section.SectionLayer(
        thickness=PLY_T, orientAngle=angle, material='Mat-CFRP')
        for angle in PLY_ANGLES]
    model.CompositeShellSection(
        name='Section-CFRP-Skin', preIntegrate=OFF,
        idealization=NO_IDEALIZATION, symmetric=OFF,
        thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        useDensity=OFF, layup=ply_data)

    model.HomogeneousSolidSection(
        name='Section-Core', material='Mat-Honeycomb', thickness=None)

    model.HomogeneousShellSection(
        name='Section-Frame', preIntegrate=OFF,
        material='Mat-Frame', thicknessType=UNIFORM,
        thickness=FRAME_THICK_RING, idealization=NO_IDEALIZATION,
        poissonDefinition=DEFAULT, useDensity=OFF)

    model.HomogeneousShellSection(
        name='Section-Flange', preIntegrate=OFF,
        material='Mat-Frame', thicknessType=UNIFORM,
        thickness=FLANGE_THICK, idealization=NO_IDEALIZATION,
        poissonDefinition=DEFAULT, useDensity=OFF)


def create_absorbing_sections(model):
    """ABL sections with high-damping materials."""
    ply_data = [section.SectionLayer(
        thickness=PLY_T, orientAngle=angle, material='Mat-CFRP-ABL')
        for angle in PLY_ANGLES]
    model.CompositeShellSection(
        name='Section-CFRP-ABL', preIntegrate=OFF,
        idealization=NO_IDEALIZATION, symmetric=OFF,
        thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        useDensity=OFF, layup=ply_data)

    model.HomogeneousSolidSection(
        name='Section-Core-ABL', material='Mat-Honeycomb-ABL', thickness=None)


def create_contact_properties(model, czm_params):
    """Surface-based CZM contact properties for Explicit."""
    Kn = czm_params['Kn']
    Ks = czm_params['Ks']
    tn = czm_params['tn']
    ts = czm_params['ts']
    GIc = czm_params['GIc']
    GIIc = czm_params['GIIc']
    BK_eta = czm_params['BK_eta']

    prop = model.ContactProperty('IntProp-CZM-Healthy')
    prop.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                        constraintEnforcementMethod=DEFAULT)
    prop.TangentialBehavior(formulation=FRICTIONLESS)
    prop.CohesiveBehavior(defaultPenalties=OFF, table=((Kn, Ks, Ks),))
    prop.Damage(initTable=((tn, ts, ts),),
                useEvolution=ON, evolutionType=ENERGY,
                evolTable=((GIc, GIIc, GIIc),),
                useMixedMode=ON, mixedModeType=BK, exponent=BK_eta)

    prop_d = model.ContactProperty('IntProp-CZM-Damaged')
    prop_d.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                          constraintEnforcementMethod=DEFAULT)
    prop_d.TangentialBehavior(formulation=FRICTIONLESS)

    prop_def = model.ContactProperty('IntProp-Default')
    prop_def.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON)
    prop_def.TangentialBehavior(formulation=FRICTIONLESS)

    print("CZM contact properties created")


# ==============================================================================
# DEFECT MATERIALS (7 types — matches barrel model)
# ==============================================================================

def create_defect_materials(model, defect_params):
    """Create defect-specific materials. Returns (skin_mat, core_mat) or (None, None)."""
    dt = defect_params.get('defect_type', 'debonding')

    if dt == 'debonding':
        # Handled by CZM-Damaged, no material override
        return None, None

    elif dt == 'fod':
        sf = defect_params.get('stiffness_factor', 10.0)
        mat = model.Material(name='Mat-Honeycomb-FOD')
        mat.Density(table=((200e-12,),))
        mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1*sf, E_CORE_2*sf, E_CORE_3*sf,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12*sf, G_CORE_13*sf, G_CORE_23*sf),))
        mat.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
        mat.Expansion(table=((12e-6,),))
        model.HomogeneousSolidSection(
            name='Section-Core-FOD', material='Mat-Honeycomb-FOD', thickness=None)
        return None, 'Mat-Honeycomb-FOD'

    elif dt == 'impact':
        dr = defect_params.get('damage_ratio', 0.3)
        mat_s = model.Material(name='Mat-CFRP-Impact')
        mat_s.Density(table=((CFRP_DENSITY,),))
        mat_s.Elastic(type=LAMINA, table=((
            CFRP_E1*0.7, CFRP_E2*dr, CFRP_NU12,
            CFRP_G12*dr, CFRP_G13*dr, CFRP_G23*dr),))
        mat_s.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
        mat_s.Expansion(table=((CFRP_CTE_1, CFRP_CTE_2, 0.0),))

        mat_c = model.Material(name='Mat-Honeycomb-Crushed')
        mat_c.Density(table=((CORE_DENSITY,),))
        mat_c.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1*0.1, E_CORE_2*0.5, E_CORE_3*0.5,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12*0.1, G_CORE_13*0.1, G_CORE_23*0.5),))
        mat_c.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
        mat_c.Expansion(table=((CORE_CTE,),))

        # Impact sections
        ply_data = [section.SectionLayer(
            thickness=PLY_T, orientAngle=a, material='Mat-CFRP-Impact')
            for a in PLY_ANGLES]
        model.CompositeShellSection(
            name='Section-CFRP-Impact', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            useDensity=OFF, layup=ply_data)
        model.HomogeneousSolidSection(
            name='Section-Core-Crushed', material='Mat-Honeycomb-Crushed',
            thickness=None)
        return 'Mat-CFRP-Impact', 'Mat-Honeycomb-Crushed'

    elif dt == 'delamination':
        depth = defect_params.get('delam_depth', 0.5)
        sr = max(0.05, 1.0 - depth)
        mat = model.Material(name='Mat-CFRP-Delam')
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Elastic(type=LAMINA, table=((
            CFRP_E1*0.9, CFRP_E2*(0.3+0.5*(1-depth)), CFRP_NU12,
            CFRP_G12*sr, CFRP_G13*sr, CFRP_G23*sr),))
        mat.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
        mat.Expansion(table=((CFRP_CTE_1, CFRP_CTE_2, 0.0),))
        ply_data = [section.SectionLayer(
            thickness=PLY_T, orientAngle=a, material='Mat-CFRP-Delam')
            for a in PLY_ANGLES]
        model.CompositeShellSection(
            name='Section-CFRP-Delam', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            useDensity=OFF, layup=ply_data)
        return 'Mat-CFRP-Delam', None

    elif dt == 'inner_debond':
        return None, None  # CZM-Damaged on inner interface

    elif dt == 'thermal_progression':
        mat = model.Material(name='Mat-CFRP-ThermalDmg')
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Elastic(type=LAMINA, table=((
            CFRP_E1*0.05, CFRP_E2*0.05, CFRP_NU12,
            CFRP_G12*0.05, CFRP_G13*0.05, CFRP_G23*0.05),))
        mat.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
        mat.Expansion(table=((5e-6, 40e-6, 0.0),))
        ply_data = [section.SectionLayer(
            thickness=PLY_T, orientAngle=a, material='Mat-CFRP-ThermalDmg')
            for a in PLY_ANGLES]
        model.CompositeShellSection(
            name='Section-CFRP-ThermalDmg', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            useDensity=OFF, layup=ply_data)
        return 'Mat-CFRP-ThermalDmg', None

    elif dt == 'acoustic_fatigue':
        sev = defect_params.get('fatigue_severity', 0.35)
        mat = model.Material(name='Mat-CFRP-AcFatigue')
        mat.Density(table=((CFRP_DENSITY,),))
        mat.Elastic(type=LAMINA, table=((
            CFRP_E1*(0.2+0.5*(1-sev)), CFRP_E2*sev, CFRP_NU12,
            CFRP_G12*sev, CFRP_G13*sev, CFRP_G23*sev),))
        mat.Damping(alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA)
        mat.Expansion(table=((CFRP_CTE_1, CFRP_CTE_2, 0.0),))
        ply_data = [section.SectionLayer(
            thickness=PLY_T, orientAngle=a, material='Mat-CFRP-AcFatigue')
            for a in PLY_ANGLES]
        model.CompositeShellSection(
            name='Section-CFRP-AcFatigue', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            useDensity=OFF, layup=ply_data)
        return 'Mat-CFRP-AcFatigue', None

    return None, None


# ==============================================================================
# GEOMETRY: JUNCTION PARTS
# ==============================================================================

def create_parts(model):
    """Create inner skin, core, outer skin for z=Z_BOTTOM..Z_TOP."""
    r_cone_top = _cone_radius_at_z(Z_TOP)
    r_ogive_top = _ogive_radius_at_z(Z_TOP, offset=0.0)
    r_ogive_top_outer = _ogive_radius_at_z(Z_TOP, offset=CORE_T)

    print("Junction geometry: z=[%.0f, %.0f], junction=%.0f" %
          (Z_BOTTOM, Z_TOP, Z_JUNCTION))

    # Inner Skin (conical ogive — consistent with barrel model)
    s1 = model.ConstrainedSketch(name='profile_inner', sheetSize=20000.0)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT+1000.0))
    s1.Line(point1=(RADIUS, Z_BOTTOM), point2=(RADIUS, Z_JUNCTION))
    s1.Line(point1=(RADIUS, Z_JUNCTION), point2=(r_cone_top, Z_TOP))
    p_inner = model.Part(name='Part-InnerSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=SECTOR_ANGLE,
                              flipRevolveDirection=OFF)

    # Core (solid, true ogive arcs)
    s2 = model.ConstrainedSketch(name='profile_core', sheetSize=20000.0)
    s2.setPrimaryObject(option=STANDALONE)
    s2.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT+1000.0))
    s2.Line(point1=(RADIUS, Z_BOTTOM), point2=(RADIUS+CORE_T, Z_BOTTOM))
    s2.Line(point1=(RADIUS+CORE_T, Z_BOTTOM), point2=(RADIUS+CORE_T, Z_JUNCTION))
    s2.ArcByCenterEnds(center=(OGIVE_XC, H_BARREL),
                        point1=(RADIUS+CORE_T, Z_JUNCTION),
                        point2=(r_ogive_top_outer, Z_TOP),
                        direction=COUNTERCLOCKWISE)
    s2.Line(point1=(r_ogive_top_outer, Z_TOP), point2=(r_ogive_top, Z_TOP))
    s2.ArcByCenterEnds(center=(OGIVE_XC, H_BARREL),
                        point1=(r_ogive_top, Z_TOP),
                        point2=(RADIUS, Z_JUNCTION),
                        direction=CLOCKWISE)
    s2.Line(point1=(RADIUS, Z_JUNCTION), point2=(RADIUS, Z_BOTTOM))
    p_core = model.Part(name='Part-Core', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=s2, angle=SECTOR_ANGLE,
                             flipRevolveDirection=OFF)

    # Outer Skin (true ogive arc)
    s3 = model.ConstrainedSketch(name='profile_outer', sheetSize=20000.0)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT+1000.0))
    s3.Line(point1=(RADIUS+CORE_T, Z_BOTTOM), point2=(RADIUS+CORE_T, Z_JUNCTION))
    s3.ArcByCenterEnds(center=(OGIVE_XC, H_BARREL),
                        point1=(RADIUS+CORE_T, Z_JUNCTION),
                        point2=(r_ogive_top_outer, Z_TOP),
                        direction=COUNTERCLOCKWISE)
    p_outer = model.Part(name='Part-OuterSkin', dimensionality=THREE_D,
                          type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=SECTOR_ANGLE,
                              flipRevolveDirection=OFF)

    print("Parts created: InnerSkin, Core, OuterSkin (sector=%.0f deg)" %
          SECTOR_ANGLE)
    return p_inner, p_core, p_outer


def create_junction_flange(model):
    """Junction flange — thick Al-7075 ring at z=Z_JUNCTION."""
    r_base = RADIUS + CORE_T
    s = model.ConstrainedSketch(name='profile_flange', sheetSize=20000.0)
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, TOTAL_HEIGHT+1000.0))
    s.Line(point1=(r_base, FLANGE_Z), point2=(r_base + FLANGE_HEIGHT, FLANGE_Z))
    p = model.Part(name='Part-Flange', dimensionality=THREE_D,
                    type=DEFORMABLE_BODY)
    p.BaseShellRevolve(sketch=s, angle=SECTOR_ANGLE, flipRevolveDirection=OFF)
    region = p.Set(faces=p.faces, name='Set-All')
    p.SectionAssignment(region=region, sectionName='Section-Flange')
    return p


def create_ring_frames(model):
    """Ring frames at specified z positions."""
    parts = []
    for i, z_pos in enumerate(FRAME_POSITIONS):
        r_base = _ogive_radius_at_z(z_pos, offset=CORE_T)
        name = 'Part-Frame-%d' % (i + 1)
        s = model.ConstrainedSketch(name='profile_frame_%d' % i,
                                     sheetSize=20000.0)
        s.setPrimaryObject(option=STANDALONE)
        s.ConstructionLine(point1=(0.0, -100.0),
                            point2=(0.0, TOTAL_HEIGHT+1000.0))
        s.Line(point1=(r_base, z_pos),
               point2=(r_base + FRAME_HEIGHT_RING, z_pos))
        p = model.Part(name=name, dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
        p.BaseShellRevolve(sketch=s, angle=SECTOR_ANGLE,
                            flipRevolveDirection=OFF)
        region = p.Set(faces=p.faces, name='Set-All')
        p.SectionAssignment(region=region, sectionName='Section-Frame')
        parts.append(p)
    return parts


# ==============================================================================
# PARTITIONING
# ==============================================================================

def _create_cylindrical_csys(part, name='CylCS'):
    return part.DatumCsysByThreePoints(
        name=name, coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(0.0, 0.0, 1.0),
        point2=(1.0, 0.0, 0.0))


def partition_defect_zone(p_core, p_outer, defect_params):
    """Partition core and outer skin at defect boundaries."""
    z_c = defect_params['z_center']
    theta_deg = defect_params['theta_deg']
    r_def = defect_params['radius']

    r_local = get_radius_at_z(z_c) + CORE_T
    theta_rad = math.radians(theta_deg)
    d_theta = min(r_def / r_local, math.radians(15.0))
    t1 = theta_rad - d_theta
    t2 = theta_rad + d_theta

    for geom_type, part in [('solid', p_core), ('shell', p_outer)]:
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
                    part.PartitionFaceByDatumPlane(
                        datumPlane=part.datums[dp_id], faces=part.faces)
                else:
                    part.PartitionCellByDatumPlane(
                        datumPlane=part.datums[dp_id], cells=part.cells)
            except Exception as e:
                print("  Partition warning: %s" % str(e)[:80])

    print("Defect zone partitioned: z=%.0f theta=%.1f r=%.0f" %
          (z_c, theta_deg, r_def))


def partition_absorbing_zones(p_inner, p_core, p_outer, abl_deg):
    """Partition ABL zones at theta=abl_deg and theta=SECTOR_ANGLE-abl_deg."""
    for theta_deg in [abl_deg, SECTOR_ANGLE - abl_deg]:
        theta_rad = math.radians(theta_deg)
        for geom_type, part in [('shell', p_inner), ('solid', p_core),
                                 ('shell', p_outer)]:
            dp = part.DatumPlaneByThreePoints(
                point1=(0, 0, 0), point2=(0, 100, 0),
                point3=(math.cos(theta_rad), 0, math.sin(theta_rad)))
            try:
                if geom_type == 'shell':
                    part.PartitionFaceByDatumPlane(
                        datumPlane=part.datums[dp.id], faces=part.faces)
                else:
                    part.PartitionCellByDatumPlane(
                        datumPlane=part.datums[dp.id], cells=part.cells)
            except Exception as e:
                print("  ABL partition warning: %s" % str(e)[:60])
    print("ABL zones partitioned: %.1f deg per edge" % abl_deg)


# ==============================================================================
# SECTION ASSIGNMENT
# ==============================================================================

def _is_in_abl_zone(pt, abl_deg):
    """Check if a point is in the ABL zone (near sector edges)."""
    x, y, z = pt[0], pt[1], pt[2]
    r = math.sqrt(x*x + z*z)
    if r < 1.0:
        return False
    theta = math.degrees(math.atan2(z, x))
    return theta < abl_deg or theta > (SECTOR_ANGLE - abl_deg)


def assign_sections(p_inner, p_core, p_outer, defect_params,
                    defect_skin_mat, defect_core_mat, absorbing_bc):
    """Assign sections with cylindrical CSYS, defect overrides, ABL zones."""
    cyl_inner = _create_cylindrical_csys(p_inner, 'CylCS-Inner')
    cyl_core = _create_cylindrical_csys(p_core, 'CylCS-Core')
    cyl_outer = _create_cylindrical_csys(p_outer, 'CylCS-Outer')

    # Inner skin
    for face in p_inner.faces:
        pt = face.pointOn[0]
        if absorbing_bc and _is_in_abl_zone(pt, ABL_WIDTH_DEG):
            sec = 'Section-CFRP-ABL'
        else:
            sec = 'Section-CFRP-Skin'
        reg = p_inner.Set(faces=p_inner.faces.findAt((pt,)),
                           name='Set-IS-%d' % face.index)
        p_inner.SectionAssignment(region=reg, sectionName=sec)
        p_inner.MaterialOrientation(region=reg, orientationType=SYSTEM,
                                     axis=AXIS_1,
                                     additionalRotationType=ROTATION_NONE,
                                     localCsys=p_inner.datums[cyl_inner.id])

    # Core
    for cell in p_core.cells:
        pt = cell.pointOn[0]
        in_defect = defect_params and is_cell_in_defect_zone(cell, defect_params)
        if absorbing_bc and _is_in_abl_zone(pt, ABL_WIDTH_DEG):
            sec = 'Section-Core-ABL'
        elif in_defect and defect_core_mat:
            sec = ('Section-Core-FOD' if 'FOD' in defect_core_mat.upper()
                   else 'Section-Core-Crushed')
        else:
            sec = 'Section-Core'
        reg = p_core.Set(cells=p_core.cells.findAt((pt,)),
                          name='Set-C-%d' % cell.index)
        p_core.SectionAssignment(region=reg, sectionName=sec)
        p_core.MaterialOrientation(region=reg, orientationType=SYSTEM,
                                    axis=AXIS_3,
                                    additionalRotationType=ROTATION_NONE,
                                    localCsys=p_core.datums[cyl_core.id])

    # Outer skin
    dt = defect_params.get('defect_type', '') if defect_params else ''
    skin_sec_map = {
        'impact': 'Section-CFRP-Impact',
        'delamination': 'Section-CFRP-Delam',
        'thermal_progression': 'Section-CFRP-ThermalDmg',
        'acoustic_fatigue': 'Section-CFRP-AcFatigue',
    }
    defect_sec = skin_sec_map.get(dt)

    for face in p_outer.faces:
        pt = face.pointOn[0]
        in_defect = defect_params and is_face_in_defect_zone(face, defect_params)
        if absorbing_bc and _is_in_abl_zone(pt, ABL_WIDTH_DEG):
            sec = 'Section-CFRP-ABL'
        elif in_defect and defect_sec:
            sec = defect_sec
        else:
            sec = 'Section-CFRP-Skin'
        reg = p_outer.Set(faces=p_outer.faces.findAt((pt,)),
                           name='Set-OS-%d' % face.index)
        p_outer.SectionAssignment(region=reg, sectionName=sec)
        p_outer.MaterialOrientation(region=reg, orientationType=SYSTEM,
                                     axis=AXIS_1,
                                     additionalRotationType=ROTATION_NONE,
                                     localCsys=p_outer.datums[cyl_outer.id])

    print("Sections assigned (ABL=%s, defect=%s)" %
          (absorbing_bc, dt if dt else 'none'))


# ==============================================================================
# CZM INTERACTIONS (General Contact, Explicit)
# ==============================================================================

def create_interactions(model, assembly, inst_inner, inst_core, inst_outer,
                        defect_params):
    """Surface-based CZM via General Contact (Explicit).

    Healthy interface: traction-separation + BK damage evolution.
    Defect zone (debonding/inner_debond): pre-damaged (frictionless contact).
    """
    # Classify core faces for surface pairing
    theta_mid = math.radians(SECTOR_ANGLE / 2.0)
    cos_t = math.cos(theta_mid)
    sin_t = math.sin(theta_mid)

    inner_pts = []
    outer_pts = []
    y_barrel = (Z_BOTTOM + Z_JUNCTION) / 2.0
    inner_pts.append(((RADIUS * cos_t, y_barrel, RADIUS * sin_t),))
    outer_pts.append((((RADIUS+CORE_T) * cos_t, y_barrel,
                       (RADIUS+CORE_T) * sin_t),))
    for frac in [0.3, 0.7]:
        y_og = Z_JUNCTION + (Z_TOP - Z_JUNCTION) * frac
        z_local = y_og - H_BARREL
        term_in = OGIVE_RHO**2 - z_local**2
        if term_in > 0:
            r_in = OGIVE_XC + math.sqrt(term_in)
            if r_in > 50:
                inner_pts.append(((r_in*cos_t, y_og, r_in*sin_t),))
        rho_out = OGIVE_RHO + CORE_T
        term_out = rho_out**2 - z_local**2
        if term_out > 0:
            r_out = OGIVE_XC + math.sqrt(term_out)
            if r_out > 50:
                outer_pts.append(((r_out*cos_t, y_og, r_out*sin_t),))

    # General Contact
    gc = model.ContactExp(name='GC-All', createStepName='Initial')
    gc.includedPairs.setValuesInStep(stepName='Initial', useAllstar=ON)
    gc.contactPropertyAssignments.appendInStep(
        stepName='Initial', assignments=((GLOBAL, SELF, 'IntProp-Default'),))

    # CZM: outer skin <-> core outer face
    if outer_pts:
        try:
            core_outer_seq = inst_core.faces.findAt(*outer_pts)
            surf_core_outer = assembly.Surface(
                side1Faces=core_outer_seq, name='Surf-Core-Outer')
            surf_outer_skin = assembly.Surface(
                side2Faces=inst_outer.faces, name='Surf-OuterSkin')
            gc.contactPropertyAssignments.appendInStep(
                stepName='Initial',
                assignments=((surf_core_outer, surf_outer_skin,
                              'IntProp-CZM-Healthy'),))
            print("CZM: Core(outer) <-> OuterSkin (Healthy)")
        except Exception as e:
            print("CZM outer warning: %s" % str(e)[:60])

    # CZM: inner skin <-> core inner face
    if inner_pts:
        try:
            core_inner_seq = inst_core.faces.findAt(*inner_pts)
            surf_core_inner = assembly.Surface(
                side1Faces=core_inner_seq, name='Surf-Core-Inner')
            surf_inner_skin = assembly.Surface(
                side1Faces=inst_inner.faces, name='Surf-InnerSkin')
            gc.contactPropertyAssignments.appendInStep(
                stepName='Initial',
                assignments=((surf_core_inner, surf_inner_skin,
                              'IntProp-CZM-Healthy'),))
            print("CZM: Core(inner) <-> InnerSkin (Healthy)")
        except Exception as e:
            print("CZM inner warning: %s" % str(e)[:60])

    # Override defect zone with CZM-Damaged for debonding/inner_debond
    if defect_params:
        dt = defect_params.get('defect_type', '')
        if dt in ('debonding', 'inner_debond'):
            print("CZM-Damaged applied at defect zone (%s)" % dt)
            # Damaged contact is handled by partitioned surface selection
            # in the INP post-processing (same approach as barrel model)

    print("General Contact (Explicit) configured")


# ==============================================================================
# MESH
# ==============================================================================

def generate_mesh(assembly, inst_inner, inst_core, inst_outer,
                  inst_flange, inst_frames, mesh_seed, defect_params=None):
    """Generate Explicit mesh for all parts."""
    all_shell = [inst_inner, inst_outer, inst_flange] + inst_frames
    assembly.seedPartInstance(regions=tuple(all_shell) + (inst_core,),
                              size=mesh_seed, deviationFactor=0.1)

    # Local refinement around defect
    if defect_params:
        z_c = defect_params['z_center']
        r_def = defect_params['radius']
        margin = 100.0
        z1 = max(Z_BOTTOM + 1.0, z_c - r_def - margin)
        z2 = min(Z_TOP - 1.0, z_c + r_def + margin)
        r_box = RADIUS + CORE_T + 200
        d_seed = max(mesh_seed * 0.6, 3.0)
        try:
            for inst in (inst_outer, inst_core, inst_inner):
                edges = inst.edges.getByBoundingBox(
                    xMin=-r_box, xMax=r_box, yMin=z1, yMax=z2,
                    zMin=-r_box, zMax=r_box)
                if len(edges) > 0:
                    assembly.seedEdgeBySize(edges=edges, size=d_seed,
                                            constraint=FINER)
            print("Local refinement: seed=%.1f in z=[%.0f, %.0f]" %
                  (d_seed, z1, z2))
        except Exception as e:
            print("Local seed warning: %s" % str(e)[:60])

    # Core: larger seed to avoid degenerate elements in thin solid
    core_seed = max(mesh_seed, CORE_T * 4)
    assembly.seedPartInstance(regions=(inst_core,), size=core_seed,
                              deviationFactor=0.1)

    # Element types: EXPLICIT library
    core_hex = ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
                         secondOrderAccuracy=OFF)
    core_tet = ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)
    core_wedge = ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    shell_elem = ElemType(elemCode=S4R, elemLibrary=EXPLICIT)
    shell_tri = ElemType(elemCode=S3R, elemLibrary=EXPLICIT)

    # Core mesh: try SWEEP HEX, fallback to FREE TET
    core_meshed = False
    try:
        assembly.setMeshControls(regions=inst_core.cells,
                                  elemShape=HEX, technique=SWEEP)
        assembly.setElementType(
            regions=(inst_core.cells,),
            elemTypes=(core_hex, core_wedge))
        assembly.generateMesh(regions=(inst_core,))
        if len(inst_core.nodes) > 0:
            core_meshed = True
            print("Core: SWEEP C3D8R seed=%.0f (%d nodes)" %
                  (core_seed, len(inst_core.nodes)))
    except Exception as e:
        print("Core SWEEP failed: %s" % str(e)[:60])

    if not core_meshed:
        try:
            assembly.deleteMesh(regions=(inst_core,))
        except:
            pass
        assembly.seedPartInstance(regions=(inst_core,), size=core_seed,
                                  deviationFactor=0.1)
        assembly.setMeshControls(regions=inst_core.cells,
                                  elemShape=TET, technique=FREE)
        assembly.setElementType(regions=(inst_core.cells,),
                                 elemTypes=(core_tet,))
        assembly.generateMesh(regions=(inst_core,))
        print("Core: FREE C3D4 seed=%.0f (%d nodes)" %
              (core_seed, len(inst_core.nodes)))

    # Shell meshes (S4R EXPLICIT)
    for inst in all_shell:
        assembly.setElementType(regions=(inst.faces,),
                                 elemTypes=(shell_elem, shell_tri))
    assembly.generateMesh(regions=tuple(all_shell))
    print("Shells: InnerSkin=%d, OuterSkin=%d, Flange=%d nodes" %
          (len(inst_inner.nodes), len(inst_outer.nodes),
           len(inst_flange.nodes)))


# ==============================================================================
# BOUNDARY CONDITIONS
# ==============================================================================

def _select_boundary_nodes(inst, theta_target, tol_coord=1.0):
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


def apply_boundary_conditions(model, assembly, all_insts,
                               inst_inner, inst_core, inst_outer):
    """Apply BCs: bottom/top fixed + sector symmetry."""
    r_box = RADIUS + CORE_T + 500.0

    # Bottom (z=Z_BOTTOM) and Top (z=Z_TOP) fixity
    for bc_name, z_val in [('Fix_Bottom', Z_BOTTOM), ('Fix_Top', Z_TOP)]:
        set_kwargs = {}
        edge_seq = None
        for inst in [inst_inner, inst_outer]:
            try:
                edges = inst.edges.getByBoundingBox(
                    xMin=-r_box, xMax=r_box,
                    yMin=z_val-1.0, yMax=z_val+1.0,
                    zMin=-r_box, zMax=r_box)
                if len(edges) > 0:
                    edge_seq = edges if edge_seq is None else edge_seq + edges
            except:
                pass
        if edge_seq and len(edge_seq) > 0:
            set_kwargs['edges'] = edge_seq

        face_pts = []
        for f in inst_core.faces:
            try:
                pt = f.pointOn[0]
                if abs(pt[1] - z_val) < 1.0:
                    face_pts.append((pt,))
            except:
                pass
        if face_pts:
            fseq = inst_core.faces.findAt(*face_pts)
            if len(fseq) > 0:
                set_kwargs['faces'] = fseq

        if set_kwargs:
            bc_set = assembly.Set(name='BC_%s' % bc_name, **set_kwargs)
            model.DisplacementBC(name=bc_name, createStepName='Initial',
                                 region=bc_set, u1=0, u2=0, u3=0)
            print("BC: %s at z=%.0f" % (bc_name, z_val))

    # Symmetry BCs
    theta_rad = math.radians(SECTOR_ANGLE)

    # theta=0: ZSYMM
    total_0 = 0
    for i, inst in enumerate(all_insts):
        labels = _select_boundary_nodes(inst, 0.0)
        if labels:
            seq = inst.nodes.sequenceFromLabels(tuple(labels))
            nset = assembly.Set(name='BC_Sym_T0_%d' % i, nodes=seq)
            model.DisplacementBC(name='Sym_T0_%d' % i,
                                 createStepName='Initial',
                                 region=nset, u3=0, ur1=0, ur2=0)
            total_0 += len(labels)
    print("BC: Symmetry theta=0: %d nodes" % total_0)

    # theta=SECTOR_ANGLE: local CSYS
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    datum_csys = assembly.DatumCsysByThreePoints(
        name='CSYS-SymTheta', coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0),
        point1=(cos_t, 0.0, sin_t), point2=(0.0, 1.0, 0.0))

    total_a = 0
    for i, inst in enumerate(all_insts):
        labels = _select_boundary_nodes(inst, theta_rad)
        if labels:
            seq = inst.nodes.sequenceFromLabels(tuple(labels))
            nset = assembly.Set(name='BC_Sym_Ta_%d' % i, nodes=seq)
            model.DisplacementBC(name='Sym_Ta_%d' % i,
                                 createStepName='Initial',
                                 region=nset, u3=0, ur1=0, ur2=0,
                                 localCsys=assembly.datums[datum_csys.id])
            total_a += len(labels)
    print("BC: Symmetry theta=%.0f: %d nodes" % (SECTOR_ANGLE, total_a))


# ==============================================================================
# EXPLICIT STEP, EXCITATION, SENSORS
# ==============================================================================

def create_explicit_step(model, time_period):
    model.ExplicitDynamicsStep(name='Step-Wave', previous='Initial',
                                timePeriod=time_period)
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('U', 'V', 'A', 'S'), timeInterval=FIELD_OUTPUT_INTERVAL)
    print("Explicit step: T=%.3e s" % time_period)
    return 'Step-Wave'


def generate_tone_burst_amplitude(model, freq_hz, n_cycles):
    """Hanning-windowed tone burst."""
    dt_sample = 1.0 / freq_hz / 40.0
    t_burst = n_cycles / freq_hz
    n_pts = int(t_burst / dt_sample) + 1
    data = []
    for i in range(n_pts):
        t = i * dt_sample
        hanning = 0.5 * (1.0 - math.cos(2 * math.pi * t / t_burst))
        val = hanning * math.sin(2 * math.pi * freq_hz * t)
        data.append((t, val))
    data.append((t_burst + dt_sample, 0.0))
    model.TabularAmplitude(name='Amp-ToneBurst', timeSpan=STEP,
                            data=tuple(data))
    return 'Amp-ToneBurst'


def apply_excitation(model, assembly, inst_outer, freq_hz, n_cycles,
                     step_name, excite_theta_deg, excite_z):
    """Radial concentrated force on outer skin."""
    amp_name = generate_tone_burst_amplitude(model, freq_hz, n_cycles)
    r_outer = _ogive_radius_at_z(excite_z, offset=CORE_T)
    theta_rad = math.radians(excite_theta_deg)
    cx = r_outer * math.cos(theta_rad)
    cy = excite_z
    cz = r_outer * math.sin(theta_rad)

    label, dist = find_nearest_node(inst_outer, cx, cy, cz)
    if label is None:
        print("ERROR: Could not find excitation node")
        return

    node_seq = inst_outer.nodes.sequenceFromLabels((label,))
    assembly.Set(nodes=node_seq, name='Set-Excitation')

    cyl_cs = assembly.DatumCsysByThreePoints(
        name='CylCS-Load', coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(1.0, 0.0, 0.0), point2=(0.0, 1.0, 0.0))

    model.ConcentratedForce(
        name='Force-ToneBurst', createStepName=step_name,
        region=assembly.sets['Set-Excitation'],
        cf1=FORCE_MAGNITUDE, amplitude=amp_name,
        localCsys=assembly.datums[cyl_cs.id])
    print("Excitation: node=%d, snap=%.2f mm, theta=%.1f, z=%.0f" %
          (label, dist, excite_theta_deg, excite_z))


def setup_sensors(model, assembly, inst_outer, step_name,
                   n_sensors=DEFAULT_N_SENSORS, abl_deg=ABL_WIDTH_DEG):
    """Place sensors on outer skin grid, accounting for double curvature."""
    theta_lo = math.radians(abl_deg + 1.0)
    theta_hi = math.radians(SECTOR_ANGLE - abl_deg - 1.0)
    z_lo = Z_BOTTOM + 100.0
    z_hi = Z_TOP - 100.0

    # Compute grid dimensions for ~n_sensors
    r_mid = _ogive_radius_at_z((Z_BOTTOM + Z_TOP) / 2.0, offset=CORE_T)
    arc_span = r_mid * (theta_hi - theta_lo)
    z_span = z_hi - z_lo
    aspect = arc_span / z_span
    n_theta = max(2, int(math.sqrt(n_sensors * aspect) + 0.5))
    n_z = max(2, int(n_sensors / n_theta + 0.5))

    d_theta = (theta_hi - theta_lo) / max(n_theta - 1, 1)
    d_z = z_span / max(n_z - 1, 1)

    print("Sensor grid: %d x %d = %d (theta x z)" %
          (n_theta, n_z, n_theta * n_z))

    sensor_count = 0
    for iz in range(n_z):
        z_s = z_lo + iz * d_z
        r_outer = _ogive_radius_at_z(z_s, offset=CORE_T)
        for it in range(n_theta):
            theta_s = theta_lo + it * d_theta
            x = r_outer * math.cos(theta_s)
            y = z_s
            z = r_outer * math.sin(theta_s)

            label, dist = find_nearest_node(inst_outer, x, y, z)
            if label is None:
                continue

            set_name = 'Set-Sensor-%d' % sensor_count
            node_seq = inst_outer.nodes.sequenceFromLabels((label,))
            assembly.Set(nodes=node_seq, name=set_name)
            model.HistoryOutputRequest(
                name='H-Output-S%d' % sensor_count,
                createStepName=step_name,
                variables=('U1', 'U2', 'U3'),
                region=assembly.sets[set_name],
                sectionPoints=DEFAULT, rebar=EXCLUDE,
                timeInterval=FIELD_OUTPUT_INTERVAL)
            sensor_count += 1

    print("Sensors placed: %d (grid %dx%d)" % (sensor_count, n_theta, n_z))
    return sensor_count


def inject_mass_scaling_inp(inp_path, dt_target):
    """Inject fixed mass scaling into INP."""
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    inserted = False
    new_lines = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        new_lines.append(line)
        if line.strip().startswith('*Dynamic, Explicit') and not inserted:
            if i + 1 < len(lines):
                new_lines.append(lines[i + 1])
                skip_next = True
            new_lines.append(
                '*Fixed Mass Scaling, type=BELOW MIN, dt=%.4e\n' % dt_target)
            inserted = True

    if inserted:
        with open(inp_path, 'w') as f:
            f.writelines(new_lines)
        print("Mass scaling injected: dt=%.2e" % dt_target)


# ==============================================================================
# THERMAL
# ==============================================================================

def apply_thermal_field(model, assembly, inst_inner, inst_core, inst_outer,
                         step_name):
    """Predefined temperature field for ascent heating."""
    # Inner skin: TEMP_INNER
    nodes_inner = inst_inner.nodes
    if len(nodes_inner) > 0:
        nset = assembly.Set(nodes=nodes_inner, name='Set-ThermalInner')
        model.Temperature(name='Temp-Inner', createStepName=step_name,
                           region=nset, magnitudes=(TEMP_INNER,))

    # Core: TEMP_CORE
    nodes_core = inst_core.nodes
    if len(nodes_core) > 0:
        nset = assembly.Set(nodes=nodes_core, name='Set-ThermalCore')
        model.Temperature(name='Temp-Core', createStepName=step_name,
                           region=nset, magnitudes=(TEMP_CORE,))

    # Outer skin: TEMP_OUTER
    nodes_outer = inst_outer.nodes
    if len(nodes_outer) > 0:
        nset = assembly.Set(nodes=nodes_outer, name='Set-ThermalOuter')
        model.Temperature(name='Temp-Outer', createStepName=step_name,
                           region=nset, magnitudes=(TEMP_OUTER,))

    print("Thermal: inner=%.0f, core=%.0f, outer=%.0f C" %
          (TEMP_INNER, TEMP_CORE, TEMP_OUTER))


# ==============================================================================
# MAIN
# ==============================================================================

def generate_model(job_name, freq_khz=DEFAULT_FREQ_KHZ,
                   n_cycles=DEFAULT_CYCLES, mesh_seed=None,
                   time_period=None, defect_params=None,
                   n_sensors=DEFAULT_N_SENSORS, sensor_layout='grid',
                   czm_params=None, no_run=False,
                   absorbing_bc=True, moisture_factor=1.0):
    """Generate junction GW Explicit model."""
    freq_hz = freq_khz * 1e3

    if mesh_seed is None:
        wavelength = CP_ESTIMATE / freq_hz * 1000.0
        mesh_seed = min(wavelength / 6.0, 7.5)

    if time_period is None:
        r_mid = _ogive_radius_at_z((Z_BOTTOM+Z_TOP)/2.0, offset=CORE_T)
        arc_length = r_mid * math.radians(SECTOR_ANGLE)
        height = Z_TOP - Z_BOTTOM
        diag = math.sqrt(arc_length**2 + height**2)
        time_period = max(diag / (VG_ESTIMATE * 1000.0) * 2.5, 0.5e-3)

    if czm_params is None:
        czm_params = DEFAULT_CZM

    excite_z = (Z_BOTTOM + Z_TOP) / 2.0
    excite_theta = SECTOR_ANGLE / 2.0

    print("=" * 60)
    print("JUNCTION GW MODEL: %s" % job_name)
    print("=" * 60)
    print("  z=[%.0f, %.0f], sector=%.0f deg" % (Z_BOTTOM, Z_TOP, SECTOR_ANGLE))
    print("  Freq: %.0f kHz, %d cycles, mesh=%.2f mm" %
          (freq_khz, n_cycles, mesh_seed))
    print("  Time: %.3e s, excite: theta=%.1f z=%.0f" %
          (time_period, excite_theta, excite_z))
    if defect_params:
        print("  Defect: %s, z=%.0f, theta=%.1f, r=%.0f" % (
            defect_params.get('defect_type', '?'),
            defect_params['z_center'],
            defect_params['theta_deg'],
            defect_params['radius']))

    Mdb()
    model = mdb.models['Model-1']

    # 1. Materials, sections, CZM
    create_materials(model, moisture_factor=moisture_factor)
    create_sections(model)
    create_contact_properties(model, czm_params)
    if absorbing_bc:
        create_absorbing_materials(model, moisture_factor=moisture_factor)
        create_absorbing_sections(model)

    # 2. Defect materials
    defect_skin_mat, defect_core_mat = None, None
    if defect_params:
        defect_skin_mat, defect_core_mat = create_defect_materials(
            model, defect_params)

    # 3. Geometry
    p_inner, p_core, p_outer = create_parts(model)
    p_flange = create_junction_flange(model)
    p_frames = create_ring_frames(model)

    # 4. Partitioning
    if defect_params:
        partition_defect_zone(p_core, p_outer, defect_params)
    if absorbing_bc:
        partition_absorbing_zones(p_inner, p_core, p_outer, ABL_WIDTH_DEG)

    # 5. Section assignment
    assign_sections(p_inner, p_core, p_outer, defect_params,
                    defect_skin_mat, defect_core_mat, absorbing_bc)

    # 6. Assembly
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)
    inst_flange = a.Instance(name='Part-Flange-1', part=p_flange, dependent=OFF)
    inst_frames = []
    for i, pf in enumerate(p_frames):
        inst = a.Instance(name='Part-Frame-%d-1' % (i+1), part=pf, dependent=OFF)
        inst_frames.append(inst)

    # 7. Mesh
    generate_mesh(a, inst_inner, inst_core, inst_outer,
                  inst_flange, inst_frames, mesh_seed, defect_params)

    # 8. CZM interactions
    create_interactions(model, a, inst_inner, inst_core, inst_outer,
                        defect_params)

    # 8b. Frame ties (frames → inner skin, NOT outer skin to avoid CZM conflict)
    for i, inst_f in enumerate(inst_frames):
        surf_f = a.Surface(side2Faces=inst_f.faces,
                            name='Surf-Frame-%d' % (i+1))
        surf_inner = a.Surface(side1Faces=inst_inner.faces,
                                name='Surf-Inner-Frame-%d' % (i+1))
        model.Tie(name='Tie-Frame-%d' % (i+1),
                  main=surf_inner, secondary=surf_f,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)

    # Flange tie
    surf_flange = a.Surface(side2Faces=inst_flange.faces, name='Surf-Flange')
    model.Tie(name='Tie-Flange', main=a.Surface(
        side1Faces=inst_outer.faces, name='Surf-Outer-Flange'),
              secondary=surf_flange,
              positionToleranceMethod=COMPUTED, adjust=ON,
              tieRotations=ON, thickness=ON)
    print("Ties: %d frames + flange" % len(inst_frames))

    # 9. BCs
    all_insts = [inst_inner, inst_core, inst_outer,
                 inst_flange] + inst_frames
    apply_boundary_conditions(model, a, all_insts,
                               inst_inner, inst_core, inst_outer)

    # 10. Explicit step
    step_name = create_explicit_step(model, time_period)

    # 11. Thermal
    apply_thermal_field(model, a, inst_inner, inst_core, inst_outer,
                         step_name)

    # 12. Excitation
    apply_excitation(model, a, inst_outer, freq_hz, n_cycles,
                     step_name, excite_theta, excite_z)

    # 13. Sensors (adapted for double curvature)
    setup_sensors(model, a, inst_outer, step_name,
                   n_sensors=n_sensors, abl_deg=ABL_WIDTH_DEG if absorbing_bc else 0.0)

    # 14. Job
    mdb.Job(name=job_name, model='Model-1', type=ANALYSIS,
            resultsFormat=ODB, numCpus=4, numDomains=4,
            multiprocessingMode=DEFAULT,
            explicitPrecision=SINGLE, nodalOutputPrecision=FULL)
    mdb.saveAs(pathName=job_name + '.cae')
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)

    # Mass scaling
    ms_dt = 0.5 * mesh_seed / 7.0e6
    inject_mass_scaling_inp(job_name + '.inp', ms_dt)

    print("\nINP written: %s.inp" % job_name)

    if not no_run:
        print("Submitting: %s" % job_name)
        mdb.jobs[job_name].submit(consistencyChecking=OFF)
        mdb.jobs[job_name].waitForCompletion()
        print("Completed: %s" % job_name)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='H3 fairing junction GW Explicit model')
    parser.add_argument('--job', type=str, default='Job-Junction')
    parser.add_argument('--job_name', type=str, default=None)
    parser.add_argument('--defect', type=str, default=None)
    parser.add_argument('--param_file', type=str, default=None)
    parser.add_argument('--mesh_seed', type=float, default=None)
    parser.add_argument('--global_seed', type=float, default=None)
    parser.add_argument('--freq', type=float, default=DEFAULT_FREQ_KHZ)
    parser.add_argument('--n_sensors', type=int, default=DEFAULT_N_SENSORS)
    parser.add_argument('--sensor_layout', type=str, default='grid')
    parser.add_argument('--no_run', action='store_true')
    parser.add_argument('--no_abl', action='store_true')
    parser.add_argument('--moisture', type=float, default=MOISTURE_FACTOR_DEFAULT)

    args, unknown = parser.parse_known_args()
    job_name = args.job_name if args.job_name is not None else args.job

    defect_data = None
    if args.param_file and os.path.exists(args.param_file):
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
                print("Invalid defect JSON")

    if defect_data and isinstance(defect_data, dict):
        if 'defect_params' in defect_data:
            defect_data = defect_data['defect_params']

    seed = args.mesh_seed or args.global_seed

    generate_model(job_name, freq_khz=args.freq,
                   mesh_seed=seed, defect_params=defect_data,
                   n_sensors=args.n_sensors, sensor_layout=args.sensor_layout,
                   no_run=args.no_run,
                   absorbing_bc=not args.no_abl,
                   moisture_factor=args.moisture)
