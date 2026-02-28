# -*- coding: utf-8 -*-
# generate_realistic_fairing.py
# Realistic H3 Type-S fairing FEM with openings, ring frames, local mesh refinement.
#
# Phase 1: Access door (phi 1300 mm) + 9 ring frames + thermal load
# Phase 2: + HVAC door, RF window, vent holes (future)
# Phase 3: + Doublers, defect integration (future)
#
# Usage:
#   abaqus cae noGUI=generate_realistic_fairing.py -- --job Job-Realistic --phase 1

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
# GEOMETRY PARAMETERS (H3 Type-S Fairing — identical to generate_fairing_dataset.py)
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
GLOBAL_SEED = 50.0     # mm
OPENING_SEED = 15.0    # mm (around openings)
FRAME_SEED = 25.0      # mm (ring frames)

# ==============================================================================
# OPENING DEFINITIONS (Phase 1: access door only)
# ==============================================================================
OPENINGS_PHASE1 = [
    {
        'name': 'AccessDoor',
        'theta_deg': 30.0,       # center angle in 1/6 sector (0-60)
        'z_center': 1500.0,      # mm from base
        'diameter': 1300.0,      # mm (phi 1300)
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
    """
    Check if point (x,y,z) in Abaqus coords is inside a circular opening.
    Y = axial, XZ = radial. radius_offset: 0 for inner skin, CORE_T for outer.
    """
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
    return dist <= r_half * 1.05  # 5% tolerance for partition alignment


def _is_point_near_opening(x, y, z, opening, margin, radius_offset=0.0):
    """Check if point is within margin mm of the opening edge (for local mesh)."""
    z_c = opening['z_center']
    r_half = opening['diameter'] / 2.0
    r_outer = r_half + margin
    if abs(y - z_c) > r_outer * 1.5:
        return False
    r_local = math.sqrt(x * x + z * z)
    if r_local < 1.0:
        return False
    theta_rad_pt = math.atan2(z, x)
    theta_center_rad = math.radians(opening['theta_deg'])
    arc_mm = r_local * abs(theta_rad_pt - theta_center_rad)
    dy = y - z_c
    dist = math.sqrt(arc_mm * arc_mm + dy * dy)
    return dist <= r_outer and dist >= r_half * 0.9


# ==============================================================================
# MATERIALS AND SECTIONS
# ==============================================================================

def create_materials(model):
    """Define CFRP, Honeycomb, and frame materials."""
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

    # CFRP for ring frame (isotropic simplification for thin frame web)
    mat = model.Material(name='CFRP_FRAME')
    mat.Elastic(type=ISOTROPIC, table=((70000.0, 0.3),))
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Expansion(table=((CFRP_CTE,),))

    # Void material for opening regions (isotropic, negligible stiffness)
    mat = model.Material(name='VOID')
    mat.Elastic(type=ISOTROPIC, table=((1.0, 0.3),))
    mat.Density(table=((1e-20,),))
    mat.Expansion(table=((0.0,),))


def create_sections(model):
    """Create shell and solid sections."""
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


# ==============================================================================
# GEOMETRY: BASE PARTS (from generate_fairing_dataset.py)
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
# OPENINGS: PARTITION + SECTION EXCLUSION
# ==============================================================================

def partition_opening(part, opening, geom_type):
    """
    Partition a part at the opening boundary using 4 datum planes.
    Same technique as partition_defect_zone() in generate_fairing_dataset.py.
    """
    z_c = opening['z_center']
    r_half = opening['diameter'] / 2.0
    theta_rad = math.radians(opening['theta_deg'])

    # Compute angular extent on the shell surface
    r_shell = get_radius_at_z(z_c) + CORE_T
    if r_shell < 1.0:
        r_shell = RADIUS + CORE_T
    d_theta = min(r_half / r_shell, math.radians(25.0))
    t1 = theta_rad - d_theta
    t2 = theta_rad + d_theta

    # Axial planes
    dp_z1 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
                                             offset=z_c - r_half)
    dp_z2 = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
                                             offset=z_c + r_half)
    # Theta planes through Y-axis
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


def assign_sections_with_openings(p_inner, p_core, p_outer, openings):
    """
    Assign sections to ALL faces/cells:
    - Healthy regions: CFRP skin / honeycomb core sections
    - Opening regions: void (negligible stiffness) sections
    This ensures all meshed elements have property definitions.
    """
    # --- Helpers: classify faces/cells ---
    def _classify_faces(part, radius_offset):
        healthy, opening = [], []
        for face in part.faces:
            pt = face.pointOn[0]
            in_any = False
            for op in openings:
                if _is_point_in_opening(pt[0], pt[1], pt[2], op, radius_offset):
                    in_any = True
                    break
            (opening if in_any else healthy).append(face)
        return healthy, opening

    def _classify_cells(part):
        healthy, opening = [], []
        for cell in part.cells:
            pt = cell.pointOn[0]
            in_any = False
            for op in openings:
                if _is_point_in_opening(pt[0], pt[1], pt[2], op):
                    in_any = True
                    break
            (opening if in_any else healthy).append(cell)
        return healthy, opening

    # --- Inner Skin ---
    healthy, opening_f = _classify_faces(p_inner, 0.0)
    if healthy:
        pts = tuple((f.pointOn[0],) for f in healthy)
        face_seq = p_inner.faces.findAt(*pts)
        region = p_inner.Set(faces=face_seq, name='Set-All')
        p_inner.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
        p_inner.MaterialOrientation(region=region, orientationType=GLOBAL,
                                    axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                    localCsys=None)
    if opening_f:
        pts_v = tuple((f.pointOn[0],) for f in opening_f)
        void_seq = p_inner.faces.findAt(*pts_v)
        void_reg = p_inner.Set(faces=void_seq, name='Set-Opening')
        p_inner.SectionAssignment(region=void_reg, sectionName='Section-Void-Shell')
    print("  InnerSkin: %d CFRP + %d void faces" % (len(healthy), len(opening_f)))

    # --- Core ---
    healthy_c, opening_c = _classify_cells(p_core)
    if healthy_c:
        pts = tuple((c.pointOn[0],) for c in healthy_c)
        cell_seq = p_core.cells.findAt(*pts)
        region = p_core.Set(cells=cell_seq, name='Set-All')
        p_core.SectionAssignment(region=region, sectionName='Section-Core')
        p_core.MaterialOrientation(region=region, orientationType=GLOBAL,
                                   axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                   localCsys=None)
    if opening_c:
        pts_v = tuple((c.pointOn[0],) for c in opening_c)
        void_seq = p_core.cells.findAt(*pts_v)
        void_reg = p_core.Set(cells=void_seq, name='Set-Opening')
        p_core.SectionAssignment(region=void_reg, sectionName='Section-Void-Solid')
    print("  Core: %d honeycomb + %d void cells" % (len(healthy_c), len(opening_c)))

    # --- Outer Skin ---
    healthy, opening_f = _classify_faces(p_outer, CORE_T)
    if healthy:
        pts = tuple((f.pointOn[0],) for f in healthy)
        face_seq = p_outer.faces.findAt(*pts)
        region = p_outer.Set(faces=face_seq, name='Set-All')
        p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
        p_outer.MaterialOrientation(region=region, orientationType=GLOBAL,
                                    axis=AXIS_3, additionalRotationType=ROTATION_NONE,
                                    localCsys=None)
    if opening_f:
        pts_v = tuple((f.pointOn[0],) for f in opening_f)
        void_seq = p_outer.faces.findAt(*pts_v)
        void_reg = p_outer.Set(faces=void_seq, name='Set-Opening')
        p_outer.SectionAssignment(region=void_reg, sectionName='Section-Void-Shell')
    print("  OuterSkin: %d CFRP + %d void faces" % (len(healthy), len(opening_f)))


# ==============================================================================
# RING FRAMES
# ==============================================================================

def create_ring_frame_parts(model, z_positions):
    """
    Create ring frame parts as shell revolve arcs (1/6 sector).
    Each frame is a thin radial web extending inward from the inner skin.
    """
    frame_parts = []
    for i, z_pos in enumerate(z_positions):
        name = 'Part-Frame-%d' % i
        s = model.ConstrainedSketch(
            name='profile_frame_%d' % i, sheetSize=20000.0)
        s.setPrimaryObject(option=STANDALONE)
        s.ConstructionLine(point1=(0.0, -100.0),
                           point2=(0.0, TOTAL_HEIGHT + 1000.0))

        # Frame at this z: radial line from (RADIUS - FRAME_HEIGHT) to RADIUS
        r_at_z = get_radius_at_z(z_pos)
        r_inner = r_at_z - RING_FRAME_HEIGHT
        if r_inner < 10.0:
            r_inner = 10.0

        s.Line(point1=(r_inner, z_pos), point2=(r_at_z, z_pos))

        p = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        p.BaseShellRevolve(sketch=s, angle=60.0, flipRevolveDirection=OFF)

        # Section assignment
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
    """
    Create Tie constraints:
    1. InnerSkin <-> Core (inner surface)
    2. Core (outer surface) <-> OuterSkin
    3. Each RingFrame <-> InnerSkin

    Uses findAt() to build proper Abaqus FaceArray sequences (not Python lists).
    """
    # --- Shell surfaces (full face sets) ---
    surf_inner = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')
    surf_outer = assembly.Surface(
        side1Faces=inst_outer.faces, name='Surf-OuterSkin')

    # --- Core face classification using findAt ---
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

    # Tie: InnerSkin <-> Core inner face
    if core_inner_pts:
        core_inner_seq = inst_core.faces.findAt(*core_inner_pts)
        surf_core_inner = assembly.Surface(
            side1Faces=core_inner_seq, name='Surf-Core-Inner')
        model.Tie(name='Tie-InnerSkin-Core', main=surf_core_inner,
                  secondary=surf_inner,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("  Tie: InnerSkin <-> Core (inner), %d core faces" % len(core_inner_pts))

    # Tie: Core outer face <-> OuterSkin
    if core_outer_pts:
        core_outer_seq = inst_core.faces.findAt(*core_outer_pts)
        surf_core_outer = assembly.Surface(
            side1Faces=core_outer_seq, name='Surf-Core-Outer')
        model.Tie(name='Tie-Core-OuterSkin', main=surf_core_outer,
                  secondary=surf_outer,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("  Tie: Core (outer) <-> OuterSkin, %d core faces" % len(core_outer_pts))

    # --- Ring Frame Ties ---
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
    """Fix bottom (y=0) for all instances using proper Abaqus sequences."""
    r_box = RADIUS + CORE_T + 500.0
    set_kwargs = {}

    # Shell instances: select bottom edges at y=0 via getByBoundingBox
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

    # Core (solid): select bottom face at y=0 via findAt
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
                  frame_instances, openings, global_seed, opening_seed):
    """Multi-resolution mesh with local refinement around openings."""
    all_skin_insts = (inst_inner, inst_core, inst_outer)

    # Global seed
    assembly.seedPartInstance(regions=all_skin_insts, size=global_seed,
                              deviationFactor=0.1)

    # Frame seed
    for inst in frame_instances:
        assembly.seedPartInstance(regions=(inst,), size=FRAME_SEED,
                                  deviationFactor=0.1)

    # Local refinement around openings
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

    # Generate mesh
    regions_to_mesh = list(all_skin_insts) + list(frame_instances)
    try:
        assembly.generateMesh(regions=tuple(regions_to_mesh))
    except Exception as e:
        # Fallback: mesh each instance separately
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
    print("Mesh: %d nodes, %d elements" % (total_nodes, total_elems))


# ==============================================================================
# THERMAL LOAD
# ==============================================================================

def apply_thermal_load(model, assembly,
                       inst_inner, inst_core, inst_outer):
    """
    Apply thermal gradient: outer 120C, inner 20C, core 70C.
    Uses instance-level Set-All regions (created during section assignment).
    Falls back to patch_inp_thermal.py if Python API fails.
    """
    try:
        # Access instance-level copies of part Set-All
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
        model.Temperature(name='Temp_Outer_Step1', createStepName='Step-1',
                          region=reg_outer, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_OUTER,))
        model.Temperature(name='Temp_Inner_Step1', createStepName='Step-1',
                          region=reg_inner, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_INNER,))
        model.Temperature(name='Temp_Core_Step1', createStepName='Step-1',
                          region=reg_core, distributionType=UNIFORM,
                          magnitudes=(TEMP_FINAL_CORE,))
        print("Thermal load: outer=%.0fC, inner=%.0fC, core=%.0fC" % (
            TEMP_FINAL_OUTER, TEMP_FINAL_INNER, TEMP_FINAL_CORE))
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
# MAIN
# ==============================================================================

def generate_realistic_model(job_name, phase=1, global_seed=None,
                              opening_seed=None, no_run=False,
                              project_root=None):
    """
    Main entry point for realistic fairing model generation.

    Args:
        job_name: Abaqus job name
        phase: 1 (access door + frames), 2 (all openings), 3 (future: doublers)
        global_seed: mesh size (mm), default 50
        opening_seed: mesh size around openings (mm), default 15
        no_run: if True, only write INP
        project_root: project root for patch script
    """
    g_seed = global_seed if global_seed is not None else GLOBAL_SEED
    o_seed = opening_seed if opening_seed is not None else OPENING_SEED

    print("=" * 70)
    print("H3 REALISTIC FAIRING MODEL — Phase %d" % phase)
    print("  Mesh: global=%.0f mm, opening=%.0f mm" % (g_seed, o_seed))
    print("=" * 70)

    Mdb()
    model = mdb.models['Model-1']

    # Select openings for this phase
    if phase >= 2:
        openings = OPENINGS_PHASE2
    else:
        openings = OPENINGS_PHASE1
    print("Openings: %d (%s)" % (
        len(openings), ', '.join([o['name'] for o in openings])))

    # Select ring frame positions (skip frames that collide with openings)
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

    # 2. Base geometry
    p_inner, p_core, p_outer = create_base_parts(model)

    # 3. Partition openings
    if openings:
        partition_all_openings(p_inner, p_core, p_outer, openings)

    # 4. Assign sections (exclude opening faces)
    assign_sections_with_openings(p_inner, p_core, p_outer, openings)

    # 5. Ring frames
    frame_parts = create_ring_frame_parts(model, frame_z_positions)

    # 6. Assembly
    a, inst_inner, inst_core, inst_outer, frame_insts = \
        create_assembly(model, p_inner, p_core, p_outer, frame_parts)

    # 7. Tie constraints
    create_tie_constraints(model, a, inst_inner, inst_core, inst_outer,
                           frame_insts)

    # 8. Boundary conditions
    apply_boundary_conditions(model, a, inst_inner, inst_core, inst_outer,
                              frame_insts)

    # 9. Step
    model.StaticStep(name='Step-1', previous='Initial')
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'U', 'RF', 'TEMP'))

    # 10. Mesh
    generate_mesh(a, inst_inner, inst_core, inst_outer, frame_insts,
                  openings, g_seed, o_seed)

    # 11. Thermal load (after mesh)
    apply_thermal_load(model, a, inst_inner, inst_core, inst_outer)

    # 12. Job
    create_and_run_job(model, job_name, no_run, project_root)

    print("=" * 70)
    print("DONE: %s (Phase %d)" % (job_name, phase))
    print("=" * 70)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate realistic H3 fairing FEM model')
    parser.add_argument('--job', type=str, default='Job-Realistic',
                        help='Job name')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3],
                        help='Phase: 1=access door+frames, 2=all openings, 3=doublers')
    parser.add_argument('--global_seed', type=float, default=None,
                        help='Global mesh seed (mm)')
    parser.add_argument('--opening_seed', type=float, default=None,
                        help='Mesh seed around openings (mm)')
    parser.add_argument('--no_run', action='store_true',
                        help='Write INP only, do not run solver')
    parser.add_argument('--project_root', type=str, default=None,
                        help='Project root for patch script')

    args, _ = parser.parse_known_args()

    project_root = (args.project_root or os.environ.get('PROJECT_ROOT')
                    or os.environ.get('PAYLOAD2026_ROOT'))

    generate_realistic_model(
        job_name=args.job,
        phase=args.phase,
        global_seed=args.global_seed,
        opening_seed=args.opening_seed,
        no_run=args.no_run,
        project_root=project_root,
    )
