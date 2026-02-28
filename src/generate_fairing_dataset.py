# -*- coding: utf-8 -*-
# generate_fairing_dataset.py
# Abaqus Python script to generate H3 Type-S fairing FEM model with multiple defect types
#
# Supported defect types:
#   debonding — Skin-core delamination (stiffness loss in outer skin)
#   fod       — Foreign Object Debris / hard spot (core stiffening)
#   impact    — Impact damage (matrix degradation + core crushing)
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

# Aluminum Honeycomb Core
E_CORE_1 = 1.0     # MPa (very low in-plane stiffness)
E_CORE_2 = 1.0     # MPa
E_CORE_3 = 1000.0  # MPa (high out-of-plane stiffness)
NU_CORE_12 = 0.01
NU_CORE_13 = 0.01
NU_CORE_23 = 0.01
G_CORE_12 = 1.0    # MPa
G_CORE_13 = 400.0  # MPa (Shear stiffness L-dir)
G_CORE_23 = 240.0  # MPa (Shear stiffness W-dir)

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
    mat_cfrp.Expansion(table=((2e-6, 2e-6, 0.0), )) # Alpha11, Alpha22, Alpha33 (local)

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
        mat.Expansion(table=((2e-6, 2e-6, 0.0), ))

    elif defect_type == 'fod':
        # Stiff foreign object inclusion in core
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
        mat_skin.Expansion(table=((2e-6, 2e-6, 0.0), ))

        # Crushed honeycomb core
        mat_core = model.Material(name='AL_HONEYCOMB_CRUSHED')
        mat_core.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1 * 0.5, E_CORE_2 * 0.5, E_CORE_3 * 0.1,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12 * 0.5, G_CORE_13 * 0.1, G_CORE_23 * 0.1
        ), ))
        mat_core.Density(table=((50e-12, ), ))
        mat_core.Expansion(table=((23e-6, ), ))

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

def assign_all_sections(p_inner, p_core, p_outer, defect_params):
    """
    Assign sections to all parts, including defect-zone overrides.
    Must be called AFTER partition_defect_zone().
    """
    # Inner skin: always healthy CFRP
    region = p_inner.Set(faces=p_inner.faces, name='Set-All')
    p_inner.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_inner.MaterialOrientation(region=region, orientationType=GLOBAL, axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE, localCsys=None)

    # Core: healthy baseline for all cells
    region = p_core.Set(cells=p_core.cells, name='Set-All')
    p_core.SectionAssignment(region=region, sectionName='Section-Core')

    # Outer skin: healthy baseline for all faces
    region = p_outer.Set(faces=p_outer.faces, name='Set-All')
    p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_outer.MaterialOrientation(region=region, orientationType=GLOBAL, axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE, localCsys=None)

    if not defect_params:
        return

    defect_type = defect_params.get('defect_type', 'debonding')

    # --- Defect-zone section overrides ---
    if defect_type in ('debonding', 'impact'):
        # Override outer skin in defect zone
        section_name = ('Section-CFRP-Debonded' if defect_type == 'debonding'
                        else 'Section-CFRP-Impact')
        defect_faces = [f for f in p_outer.faces
                        if is_face_in_defect_zone(f, defect_params)]
        if defect_faces:
            # Use findAt to get proper FaceArray type for Set()
            pts = tuple((f.pointOn[0],) for f in defect_faces)
            face_seq = p_outer.faces.findAt(*pts)
            region_d = p_outer.Set(faces=face_seq, name='Set-DefectZone-Skin')
            p_outer.SectionAssignment(region=region_d, sectionName=section_name)
            p_outer.MaterialOrientation(region=region_d, orientationType=GLOBAL, axis=AXIS_3,
                                        additionalRotationType=ROTATION_NONE, localCsys=None)
            print("  %s: %d skin faces -> %s" % (defect_type, len(defect_faces), section_name))
        else:
            print("  Warning: no outer skin faces found in defect zone")

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
    if defect_params:
        parts_to_partition = []
        if defect_type in ('debonding', 'impact'):
            parts_to_partition.append(('shell', p_outer))
        if defect_type in ('fod', 'impact'):
            parts_to_partition.append(('solid', p_core))
        # Also partition for mesh refinement (always partition outer skin and core)
        if defect_type == 'fod' and ('shell', p_outer) not in parts_to_partition:
            parts_to_partition.append(('shell', p_outer))
        partition_defect_zone(parts_to_partition, defect_params)

    # 4. Assign sections (healthy baseline + defect-zone overrides)
    assign_all_sections(p_inner, p_core, p_outer, defect_params)

    # 5. Assembly
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)
    
    # 6. Interaction — Tie constraints are NOT used.
    # With dependent=OFF instances, assembly-level surfaces don't export to INP correctly.
    # The model solves under thermal load only (each part deforms independently).
    # Defect physics come from material property differences in the defect zone.
    # (Tie + BC improvement is a future TODO for physically coupled model.)
    
    # 7. BCs - fix bottom (y=0) for static equilibrium — Abaqus: Y=axial
    try:
        bottom_faces = []
        for inst in (inst_inner, inst_core, inst_outer):
            for f in inst.faces:
                if f.pointOn[0][1] < 1.0:  # y < 1
                    bottom_faces.append(f)
        if bottom_faces:
            bot_set = a.Set(faces=bottom_faces, name='BC_Bottom')
            model.DisplacementBC(name='Fix_Bottom', createStepName='Initial',
                                region=bot_set, u1=0, u2=0, u3=0)
    except Exception as e:
        print("Warning: BC: %s" % str(e))

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
    
    # 9. Temperature IC — single set for all nodes (avoids assembly set naming issues)
    try:
        all_nodes = []
        for inst in (inst_inner, inst_core, inst_outer):
            all_nodes.extend(list(inst.nodes))
        set_all = a.Set(nodes=all_nodes, name='TempSet_All')
        model.Temperature(name='Temp_IC', createStepName='Initial',
                         region=set_all, distributionType=UNIFORM, magnitudes=(TEMP_INITIAL,))
    except Exception as e:
        print("Warning: Temperature IC skipped: %s" % str(e)[:60])
    
    # 9b. Thermal load in Step-1 (ascent heating: outer 120°C, inner 20°C, core gradient)
    try:
        set_outer = a.Set(nodes=list(inst_outer.nodes), name='TempSet_Outer')
        set_inner = a.Set(nodes=list(inst_inner.nodes), name='TempSet_Inner')
        set_core = a.Set(nodes=list(inst_core.nodes), name='TempSet_Core')
        model.Temperature(name='Temp_Outer_Step1', createStepName='Step-1',
                         region=set_outer, distributionType=UNIFORM, magnitudes=(TEMP_FINAL_OUTER,))
        model.Temperature(name='Temp_Inner_Step1', createStepName='Step-1',
                         region=set_inner, distributionType=UNIFORM, magnitudes=(TEMP_FINAL_INNER,))
        temp_core = (TEMP_FINAL_OUTER + TEMP_FINAL_INNER) / 2.0
        model.Temperature(name='Temp_Core_Step1', createStepName='Step-1',
                         region=set_core, distributionType=UNIFORM, magnitudes=(temp_core,))
        print("Thermal load applied: outer=%g C, inner=%g C, core=%g C" % (TEMP_FINAL_OUTER, TEMP_FINAL_INNER, temp_core))
    except Exception as e:
        print("Warning: Thermal load skipped: %s" % str(e)[:80])
    
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
        # Find patch script: project_root arg, env, or search upward from inp dir
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
            import subprocess
            r = subprocess.call([sys.executable, patch_script, inp_path], cwd=os.path.dirname(inp_path))
            if r == 0:
                print("INP patched for thermal load")
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
