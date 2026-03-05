# -*- coding: utf-8 -*-
"""
Realistic Fairing Guided Wave Model — Abaqus/Explicit

Generates a 30-degree (1/12) barrel sector of the H3 fairing with:
  - CFRP/Al-Honeycomb sandwich structure (3-part)
  - Surface-based CZM adhesive (ContactExp + CohesiveBehavior)
  - Openings (access door, HVAC, vents) with void sections
  - Ring frames tied to inner skin
  - Cylindrical CSYS material orientation
  - Symmetry BCs at sector edges
  - Hanning tone burst excitation on outer skin
  - 10 sensors in cross pattern (5 circumferential + 5 axial)
  - Abaqus/Explicit solver (no mass scaling)

Usage:
  abaqus cae noGUI=src/generate_gw_fairing.py -- --job Job-GW-Fair-H --no_run
  abaqus cae noGUI=src/generate_gw_fairing.py -- --job Job-GW-Fair-D \\
      --defect '{"z_center":1800,"theta_deg":15,"radius":50}' --no_run
"""

from __future__ import print_function
import sys
import os
import math
import json

# Abaqus imports
from abaqus import *
from abaqusConstants import *
from caeModules import *
import mesh

# ==============================================================================
# CONSTANTS — Geometry
# ==============================================================================
RADIUS = 2600.0          # mm, inner skin radius (barrel section)
CORE_T = 38.0            # mm, honeycomb core thickness
FACE_T = 1.0             # mm, CFRP skin thickness (each)
ADHESIVE_T = 0.0         # mm, surface-based CZM (no physical thickness)
DEFAULT_SECTOR_ANGLE = 30.0  # degrees (1/12 of circumference)
DEFAULT_Z_MIN = 500.0    # mm, barrel bottom cutoff
DEFAULT_Z_MAX = 2500.0   # mm, barrel top cutoff

# Ring frames
RING_FRAME_Z = [500, 1000, 1500, 2000, 2500]  # standard positions
RING_FRAME_HEIGHT = 50.0  # mm, radial extent inward
RING_FRAME_T = 3.0        # mm, shell thickness

# ==============================================================================
# CONSTANTS — Materials
# ==============================================================================
# CFRP T1000G/Epoxy
CFRP_E1 = 160000.0    # MPa
CFRP_E2 = 10000.0
CFRP_NU12 = 0.3
CFRP_G12 = 5000.0
CFRP_G13 = 5000.0
CFRP_G23 = 3000.0
CFRP_DENSITY = 1600e-12  # tonne/mm^3

# Al-Honeycomb 5052 (cylindrical CSYS: 1=R, 2=theta, 3=axial)
E_CORE_1 = 1000.0     # MPa (radial / through-thickness)
E_CORE_2 = 1.0        # theta
E_CORE_3 = 1.0        # axial
NU_CORE_12 = 0.001
NU_CORE_13 = 0.001
NU_CORE_23 = 0.001
G_CORE_12 = 400.0     # R-theta shear
G_CORE_13 = 240.0     # R-axial shear
G_CORE_23 = 1.0       # theta-axial
CORE_DENSITY = 50e-12

# Frame (Al-7075)
FRAME_E = 71700.0
FRAME_NU = 0.33
FRAME_DENSITY = 2810e-12

# Layup: [45/0/-45/90]s  (8 plies, each FACE_T/8)
PLY_ANGLES = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
PLY_T = FACE_T / 8.0

# ==============================================================================
# CONSTANTS — CZM (surface-based cohesive)
# ==============================================================================
DEFAULT_CZM = {
    'Kn': 1e5,       # N/mm^3 (normal penalty stiffness)
    'Ks': 5e4,       # N/mm^3 (shear penalty stiffness)
    'tn': 50.0,      # MPa (normal strength)
    'ts': 40.0,      # MPa (shear strength)
    'GIc': 0.3,      # N/mm (mode I fracture energy)
    'GIIc': 1.0,     # N/mm (mode II fracture energy)
    'BK_eta': 2.284, # Benzeggagh-Kenane exponent
}

# ==============================================================================
# CONSTANTS — Wave / Explicit
# ==============================================================================
DEFAULT_FREQ_KHZ = 50.0
DEFAULT_CYCLES = 5
CP_ESTIMATE = 1550.0      # m/s, A0 mode phase velocity estimate
FORCE_MAGNITUDE = 1.0     # N, concentrated force
FIELD_OUTPUT_INTERVAL = 1e-6  # 1 microsecond
SENSOR_SPACING = 30.0     # mm between sensors
DEFAULT_N_SENSORS = 10    # 5 circumferential + 5 axial

# ==============================================================================
# CONSTANTS — Openings (filtered to sector range at runtime)
# ==============================================================================
OPENINGS_PHASE1 = [
    {'name': 'HVAC_Door', 'z_center': 2500.0, 'theta_deg': 20.0,
     'diameter': 400.0},
]
OPENINGS_PHASE2 = OPENINGS_PHASE1 + [
    {'name': 'Vent_1', 'z_center': 600.0, 'theta_deg': 15.0,
     'diameter': 100.0},
]
# Note: Access door (D=1300mm at θ=30°) is too large for 30° sector
# HVAC door (D=400mm at θ=20°) fits well within the 30° sector


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_radius_at_z(z):
    """Return outer radius of fairing at axial position z.
    Barrel section: constant radius. Ogive: not used here."""
    return RADIUS


def _point_in_defect_zone(x, y, z, defect_params):
    """Check if point (x,y,z) is inside the circular defect zone.

    defect_params: {z_center, theta_deg, radius}
    Point is in cylindrical coords: x=R*cos(theta), y=z_axial, z=R*sin(theta).
    """
    z_c = defect_params['z_center']
    theta_c = math.radians(defect_params['theta_deg'])
    r_def = defect_params['radius']

    r_pt = math.sqrt(x * x + z * z)
    if r_pt < 1.0:
        return False
    theta_pt = math.atan2(z, x)

    # Arc distance + axial distance
    arc_dist = r_pt * (theta_pt - theta_c)
    axial_dist = y - z_c
    dist = math.sqrt(arc_dist * arc_dist + axial_dist * axial_dist)
    return dist <= r_def * 1.02


def _is_point_in_opening(x, y, z, opening, radius_offset=0.0):
    """Check if point is inside an opening zone.

    opening: {z_center, theta_deg, diameter, name}
    """
    z_c = opening['z_center']
    theta_c = math.radians(opening['theta_deg'])
    r_opening = opening['diameter'] / 2.0

    r_pt = math.sqrt(x * x + z * z) + radius_offset
    if r_pt < 1.0:
        return False
    theta_pt = math.atan2(z, x)

    arc_dist = r_pt * abs(theta_pt - theta_c)
    axial_dist = abs(y - z_c)
    dist = math.sqrt(arc_dist * arc_dist + axial_dist * axial_dist)
    return dist <= r_opening * 1.05


def _filter_openings_in_range(openings, z_min, z_max, theta_min, theta_max):
    """Filter openings that fall within the sector range."""
    filtered = []
    for op in openings:
        z_c = op['z_center']
        r_op = op['diameter'] / 2.0
        theta_c = op['theta_deg']
        # Check if opening center is within z and theta range (with margin)
        if (z_min - r_op) < z_c < (z_max + r_op):
            if (theta_min - 5.0) < theta_c < (theta_max + 5.0):
                filtered.append(op)
    return filtered


def find_nearest_node(instance, target_x, target_y, target_z):
    """Find mesh node closest to target coordinates."""
    min_dist = 1e30
    nearest_label = None
    for node in instance.nodes:
        c = node.coordinates
        dx = c[0] - target_x
        dy = c[1] - target_y
        dz = c[2] - target_z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < min_dist:
            min_dist = dist
            nearest_label = node.label
    return nearest_label, min_dist


# ==============================================================================
# MATERIALS & SECTIONS
# ==============================================================================

def create_materials(model):
    """Create all materials (no CTE — dynamic analysis only)."""
    # CFRP
    mat = model.Material(name='Mat-CFRP')
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Elastic(
        type=LAMINA,
        table=((CFRP_E1, CFRP_E2, CFRP_NU12,
                CFRP_G12, CFRP_G13, CFRP_G23),))

    # Al-Honeycomb (engineering constants in cylindrical CSYS)
    mat_c = model.Material(name='Mat-Honeycomb')
    mat_c.Density(table=((CORE_DENSITY,),))
    mat_c.Elastic(
        type=ENGINEERING_CONSTANTS,
        table=((E_CORE_1, E_CORE_2, E_CORE_3,
                NU_CORE_12, NU_CORE_13, NU_CORE_23,
                G_CORE_12, G_CORE_13, G_CORE_23),))

    # Frame (isotropic Al-7075)
    mat_f = model.Material(name='Mat-Frame')
    mat_f.Density(table=((FRAME_DENSITY,),))
    mat_f.Elastic(table=((FRAME_E, FRAME_NU),))

    # Void (for openings)
    mat_v = model.Material(name='Mat-Void')
    mat_v.Density(table=((1e-15,),))
    mat_v.Elastic(table=((1.0, 0.3),))

    print("Materials created: CFRP, Honeycomb, Frame, Void")


def create_sections(model):
    """Create composite shell + solid sections."""
    # CFRP skin: [45/0/-45/90]s
    ply_data = [section.SectionLayer(
        thickness=PLY_T, orientAngle=angle, material='Mat-CFRP')
        for angle in PLY_ANGLES]
    model.CompositeShellSection(
        name='Section-CFRP-Skin',
        preIntegrate=OFF, idealization=NO_IDEALIZATION,
        symmetric=OFF, thicknessType=UNIFORM,
        poissonDefinition=DEFAULT, useDensity=OFF,
        layup=ply_data)

    # Core solid
    model.HomogeneousSolidSection(
        name='Section-Core', material='Mat-Honeycomb', thickness=None)

    # Frame shell
    model.HomogeneousShellSection(
        name='Section-Frame', preIntegrate=OFF,
        material='Mat-Frame', thicknessType=UNIFORM,
        thickness=RING_FRAME_T, idealization=NO_IDEALIZATION,
        poissonDefinition=DEFAULT, useDensity=OFF)

    # Void sections
    model.HomogeneousShellSection(
        name='Section-Void-Shell', preIntegrate=OFF,
        material='Mat-Void', thicknessType=UNIFORM,
        thickness=0.01, idealization=NO_IDEALIZATION,
        poissonDefinition=DEFAULT, useDensity=OFF)
    model.HomogeneousSolidSection(
        name='Section-Void-Solid', material='Mat-Void', thickness=None)

    print("Sections created: CFRP-Skin [45/0/-45/90]s, Core, Frame, Voids")


def create_contact_properties(model, czm_params):
    """Create ContactProperty definitions for surface-based CZM."""
    Kn = czm_params['Kn']
    Ks = czm_params['Ks']
    tn = czm_params['tn']
    ts = czm_params['ts']
    GIc = czm_params['GIc']
    GIIc = czm_params['GIIc']
    BK_eta = czm_params['BK_eta']

    # Healthy CZM: traction-separation + BK damage
    prop = model.ContactProperty('IntProp-CZM-Healthy')
    prop.NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT)
    prop.TangentialBehavior(formulation=FRICTIONLESS)
    prop.CohesiveBehavior(
        defaultPenalties=OFF,
        table=((Kn, Ks, Ks),))
    prop.Damage(
        initTable=((tn, ts, ts),),
        useEvolution=ON,
        evolutionType=ENERGY,
        evolTable=((GIc, GIIc, GIIc),),
        useMixedMode=ON,
        mixedModeType=BK,
        exponent=BK_eta)
    print("  IntProp-CZM-Healthy: Kn=%.0e Ks=%.0e tn=%.0f ts=%.0f" % (
        Kn, Ks, tn, ts))

    # Damaged: frictionless hard contact (pre-debonded zone)
    prop_d = model.ContactProperty('IntProp-CZM-Damaged')
    prop_d.NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON,
        constraintEnforcementMethod=DEFAULT)
    prop_d.TangentialBehavior(formulation=FRICTIONLESS)
    print("  IntProp-CZM-Damaged: frictionless (pre-debonded)")

    # Default: hard contact for self-contact etc.
    prop_def = model.ContactProperty('IntProp-Default')
    prop_def.NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON)
    prop_def.TangentialBehavior(formulation=FRICTIONLESS)

    print("Surface-based CZM contact properties created")


# ==============================================================================
# GEOMETRY
# ==============================================================================

def create_sector_parts(model, z_min, z_max, sector_angle):
    """Create 3-part barrel sector via revolve."""
    r_inner = RADIUS
    r_outer = RADIUS + CORE_T

    # --- Inner Skin (shell) ---
    s1 = model.ConstrainedSketch(name='sk_inner', sheetSize=20000.0)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s1.Line(point1=(r_inner, z_min), point2=(r_inner, z_max))

    p_inner = model.Part(name='Part-InnerSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=sector_angle,
                             flipRevolveDirection=OFF)
    s1.unsetPrimaryObject()

    # --- Core (solid) ---
    s2 = model.ConstrainedSketch(name='sk_core', sheetSize=20000.0)
    s2.setPrimaryObject(option=STANDALONE)
    s2.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s2.Line(point1=(r_inner, z_min), point2=(r_inner, z_max))
    s2.Line(point1=(r_inner, z_max), point2=(r_outer, z_max))
    s2.Line(point1=(r_outer, z_max), point2=(r_outer, z_min))
    s2.Line(point1=(r_outer, z_min), point2=(r_inner, z_min))

    p_core = model.Part(name='Part-Core', dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=s2, angle=sector_angle,
                            flipRevolveDirection=OFF)
    s2.unsetPrimaryObject()

    # --- Outer Skin (shell) ---
    s3 = model.ConstrainedSketch(name='sk_outer', sheetSize=20000.0)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s3.Line(point1=(r_outer, z_min), point2=(r_outer, z_max))

    p_outer = model.Part(name='Part-OuterSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=sector_angle,
                             flipRevolveDirection=OFF)
    s3.unsetPrimaryObject()

    print("Sector parts created: %.0f deg, z=[%.0f, %.0f] mm" % (
        sector_angle, z_min, z_max))
    print("  InnerSkin: R=%.0f mm" % r_inner)
    print("  Core: R=%.0f to %.0f mm (t=%.0f)" % (r_inner, r_outer, CORE_T))
    print("  OuterSkin: R=%.0f mm" % r_outer)

    return p_inner, p_core, p_outer


def create_ring_frame_parts(model, z_positions, sector_angle):
    """Create ring frame shell parts within the sector."""
    frame_parts = []
    for i, z_pos in enumerate(z_positions):
        name = 'Part-Frame-%d' % i
        s = model.ConstrainedSketch(
            name='profile_frame_%d' % i, sheetSize=20000.0)
        s.setPrimaryObject(option=STANDALONE)
        s.ConstructionLine(point1=(0.0, -100.0),
                           point2=(0.0, z_pos + 1000.0))

        r_at_z = get_radius_at_z(z_pos)
        r_inner_frame = r_at_z - RING_FRAME_HEIGHT
        if r_inner_frame < 10.0:
            r_inner_frame = 10.0
        s.Line(point1=(r_inner_frame, z_pos), point2=(r_at_z, z_pos))

        p = model.Part(name=name, dimensionality=THREE_D,
                       type=DEFORMABLE_BODY)
        p.BaseShellRevolve(sketch=s, angle=sector_angle,
                           flipRevolveDirection=OFF)
        s.unsetPrimaryObject()

        region = p.Set(faces=p.faces, name='Set-All')
        p.SectionAssignment(region=region, sectionName='Section-Frame')

        frame_parts.append(p)

    if z_positions:
        print("Ring frames: %d at z = %s" % (
            len(frame_parts),
            ', '.join(['%.0f' % z for z in z_positions])))
    return frame_parts


# ==============================================================================
# PARTITIONING
# ==============================================================================

def partition_opening(part, opening, geom_type='shell'):
    """Partition a part around an opening using datum planes."""
    z_c = opening['z_center']
    theta_c = math.radians(opening['theta_deg'])
    r_op = opening['diameter'] / 2.0

    # Bounding box in axial direction
    try:
        part.DatumPlaneByPrincipalPlane(
            principalPlane=XZPLANE, offset=z_c - r_op)
        part.DatumPlaneByPrincipalPlane(
            principalPlane=XZPLANE, offset=z_c + r_op)

        # Bounding planes in circumferential (approximate with X-Z rotated planes)
        r_ref = RADIUS + CORE_T / 2.0
        arc_half = r_op / r_ref
        theta_lo = theta_c - arc_half
        theta_hi = theta_c + arc_half

        # Angled datum planes for circumferential bounds
        for theta in [theta_lo, theta_hi]:
            nx = -math.sin(theta)
            nz = math.cos(theta)
            pt = (r_ref * math.cos(theta), z_c, r_ref * math.sin(theta))
            try:
                part.DatumPlaneByPointNormal(
                    point=pt, normal=(nx, 0.0, nz))
            except Exception:
                pass

        # Partition using datum planes
        datum_ids = sorted(part.datums.keys())
        for did in datum_ids:
            datum = part.datums[did]
            if hasattr(datum, 'pointOn'):
                continue
            try:
                if geom_type == 'solid':
                    part.PartitionCellByDatumPlane(
                        datumPlane=datum, cells=part.cells)
                else:
                    part.PartitionFaceByDatumPlane(
                        datumPlane=datum, faces=part.faces)
            except Exception:
                pass

    except Exception as e:
        print("  Warning: partition failed for %s: %s" % (
            opening['name'], str(e)[:60]))

    print("  Opening partitioned: %s (z=%.0f, theta=%.1f, D=%.0f)" % (
        opening['name'], z_c, opening['theta_deg'], opening['diameter']))


def partition_all_openings(p_inner, p_core, p_outer, openings):
    """Partition all 3 parts for each opening."""
    if not openings:
        return
    print("Partitioning openings (%d)..." % len(openings))
    for op in openings:
        partition_opening(p_inner, op, 'shell')
        partition_opening(p_core, op, 'solid')
        partition_opening(p_outer, op, 'shell')


def partition_defect_zone(p_core, p_outer, defect_params):
    """Partition core and outer skin at defect boundary."""
    if not defect_params:
        return

    z_c = defect_params['z_center']
    theta_c = math.radians(defect_params['theta_deg'])
    r_def = defect_params['radius']
    r_ref = RADIUS + CORE_T / 2.0

    # Axial bounds
    for part in [p_core, p_outer]:
        try:
            part.DatumPlaneByPrincipalPlane(
                principalPlane=XZPLANE, offset=z_c - r_def)
            part.DatumPlaneByPrincipalPlane(
                principalPlane=XZPLANE, offset=z_c + r_def)
        except Exception:
            pass

        # Circumferential bounds
        arc_half = r_def / r_ref
        for theta in [theta_c - arc_half, theta_c + arc_half]:
            pt = (r_ref * math.cos(theta), z_c, r_ref * math.sin(theta))
            nx = -math.sin(theta)
            nz = math.cos(theta)
            try:
                part.DatumPlaneByPointNormal(
                    point=pt, normal=(nx, 0.0, nz))
            except Exception:
                pass

        # Partition
        datum_ids = sorted(part.datums.keys())
        for did in datum_ids:
            datum = part.datums[did]
            if hasattr(datum, 'pointOn'):
                continue
            try:
                if part.cells:
                    part.PartitionCellByDatumPlane(
                        datumPlane=datum, cells=part.cells)
                else:
                    part.PartitionFaceByDatumPlane(
                        datumPlane=datum, faces=part.faces)
            except Exception:
                pass

    print("Defect zone partitioned: z=%.0f, theta=%.1f deg, r=%.0f mm" % (
        z_c, defect_params['theta_deg'], r_def))


# ==============================================================================
# SECTION ASSIGNMENT (with cylindrical CSYS)
# ==============================================================================

def _create_cylindrical_csys(part, name='CylCS'):
    """Create cylindrical CSYS: 1=R, 2=theta, 3=axial(Y)."""
    datum = part.DatumCsysByThreePoints(
        name=name, coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(0.0, 0.0, 1.0),
        point2=(1.0, 0.0, 0.0))
    return datum


def assign_sections(p_inner, p_core, p_outer, openings, defect_params):
    """Assign sections with cylindrical CSYS orientation."""
    # Create CSYS for each part
    cyl_inner = _create_cylindrical_csys(p_inner, 'CylCS-InnerSkin')
    cyl_outer = _create_cylindrical_csys(p_outer, 'CylCS-OuterSkin')
    cyl_core = _create_cylindrical_csys(p_core, 'CylCS-Core')

    # --- Classify and assign faces/cells ---
    def _in_any_opening(x, y, z, r_off=0.0):
        for op in (openings or []):
            if _is_point_in_opening(x, y, z, op, r_off):
                return True
        return False

    # Inner Skin
    healthy_f, void_f = [], []
    for face in p_inner.faces:
        pt = face.pointOn[0]
        if _in_any_opening(pt[0], pt[1], pt[2], 0.0):
            void_f.append(face)
        else:
            healthy_f.append(face)

    if healthy_f:
        pts = tuple((f.pointOn[0],) for f in healthy_f)
        region = p_inner.Set(
            faces=p_inner.faces.findAt(*pts), name='Set-Healthy')
        p_inner.SectionAssignment(
            region=region, sectionName='Section-CFRP-Skin')
        p_inner.MaterialOrientation(
            region=region, orientationType=SYSTEM,
            axis=AXIS_1, additionalRotationType=ROTATION_NONE,
            localCsys=p_inner.datums[cyl_inner.id])
    if void_f:
        pts = tuple((f.pointOn[0],) for f in void_f)
        reg = p_inner.Set(
            faces=p_inner.faces.findAt(*pts), name='Set-Opening')
        p_inner.SectionAssignment(
            region=reg, sectionName='Section-Void-Shell')
    print("  InnerSkin: %d CFRP + %d void faces" % (
        len(healthy_f), len(void_f)))

    # Core
    healthy_c, void_c = [], []
    for cell in p_core.cells:
        pt = cell.pointOn[0]
        if _in_any_opening(pt[0], pt[1], pt[2]):
            void_c.append(cell)
        else:
            healthy_c.append(cell)

    if healthy_c:
        pts = tuple((c.pointOn[0],) for c in healthy_c)
        region = p_core.Set(
            cells=p_core.cells.findAt(*pts), name='Set-Healthy')
        p_core.SectionAssignment(
            region=region, sectionName='Section-Core')
        p_core.MaterialOrientation(
            region=region, orientationType=SYSTEM,
            axis=AXIS_3, additionalRotationType=ROTATION_NONE,
            localCsys=p_core.datums[cyl_core.id])
    if void_c:
        pts = tuple((c.pointOn[0],) for c in void_c)
        reg = p_core.Set(
            cells=p_core.cells.findAt(*pts), name='Set-Opening')
        p_core.SectionAssignment(
            region=reg, sectionName='Section-Void-Solid')
    print("  Core: %d honeycomb + %d void cells" % (
        len(healthy_c), len(void_c)))

    # Outer Skin
    healthy_f, void_f = [], []
    for face in p_outer.faces:
        pt = face.pointOn[0]
        if _in_any_opening(pt[0], pt[1], pt[2], CORE_T):
            void_f.append(face)
        else:
            healthy_f.append(face)

    if healthy_f:
        pts = tuple((f.pointOn[0],) for f in healthy_f)
        region = p_outer.Set(
            faces=p_outer.faces.findAt(*pts), name='Set-Healthy')
        p_outer.SectionAssignment(
            region=region, sectionName='Section-CFRP-Skin')
        p_outer.MaterialOrientation(
            region=region, orientationType=SYSTEM,
            axis=AXIS_1, additionalRotationType=ROTATION_NONE,
            localCsys=p_outer.datums[cyl_outer.id])
    if void_f:
        pts = tuple((f.pointOn[0],) for f in void_f)
        reg = p_outer.Set(
            faces=p_outer.faces.findAt(*pts), name='Set-Opening')
        p_outer.SectionAssignment(
            region=reg, sectionName='Section-Void-Shell')
    print("  OuterSkin: %d CFRP + %d void faces" % (
        len(healthy_f), len(void_f)))


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
        name = 'Part-Frame-%d-1' % i
        fi = a.Instance(name=name, part=fp, dependent=OFF)
        frame_instances.append(fi)

    print("Assembly: 3 sandwich parts + %d frames" % len(frame_instances))
    return a, inst_inner, inst_core, inst_outer, frame_instances


# ==============================================================================
# INTERACTIONS — Surface-based CZM (General Contact for Explicit)
# ==============================================================================

def _classify_core_surfaces_mesh(assembly, inst_core, defect_params):
    """Create mesh-based surfaces for core inner/outer faces.

    Uses element face normals to identify inner (R=RADIUS) and outer
    (R=RADIUS+CORE_T) surfaces, then classifies outer faces into
    healthy vs defect zones.

    Returns (surf_inner, surf_outer_healthy, surf_outer_defect).
    Must be called AFTER meshing.
    """
    r_inner = RADIUS
    r_outer = RADIUS + CORE_T
    tol_r = CORE_T * 0.3  # generous tolerance

    inner_elems = []
    outer_h_elems = []
    outer_d_elems = []

    for elem in inst_core.elements:
        nodes = elem.connectivity
        node_coords = [inst_core.nodes[n].coordinates for n in nodes]

        # Check each element face (for C3D8R: 6 faces)
        # Face normals computed from face center
        n_nodes = len(nodes)
        if n_nodes == 8:
            # C3D8R faces: (0123), (4567), (0154), (1265), (2376), (3047)
            face_defs = [
                (0, 1, 2, 3, 'S1'), (4, 5, 6, 7, 'S2'),
                (0, 1, 5, 4, 'S3'), (1, 2, 6, 5, 'S4'),
                (2, 3, 7, 6, 'S5'), (3, 0, 4, 7, 'S6')]
        elif n_nodes == 6:
            # C3D6 (wedge) faces
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

            if abs(r_face - r_inner) < tol_r:
                inner_elems.append((elem.label, fn))
            elif abs(r_face - r_outer) < tol_r:
                if defect_params and _point_in_defect_zone(
                        cx, cy, cz, defect_params):
                    outer_d_elems.append((elem.label, fn))
                else:
                    outer_h_elems.append((elem.label, fn))

    print("  Core mesh surfaces: %d inner, %d outer healthy, %d outer defect" % (
        len(inner_elems), len(outer_h_elems), len(outer_d_elems)))

    # Create element-based surfaces
    surf_inner = None
    surf_outer_h = None
    surf_outer_d = None

    if inner_elems:
        surf_inner = _create_elem_surface(assembly, inst_core,
                                           'Surf-Core-Inner', inner_elems)
    if outer_h_elems:
        surf_outer_h = _create_elem_surface(assembly, inst_core,
                                             'Surf-Core-Outer-Healthy', outer_h_elems)
    if outer_d_elems:
        surf_outer_d = _create_elem_surface(assembly, inst_core,
                                             'Surf-Core-Outer-Defect', outer_d_elems)

    return surf_inner, surf_outer_h, surf_outer_d


def _create_elem_surface(assembly, inst, name, elem_face_list):
    """Create element-based surface using faceNElements keyword args.

    elem_face_list: [(label, 'S1'), (label, 'S2'), ...]
    Abaqus API: assembly.Surface(name=..., face1Elements=arr, face2Elements=arr, ...)
    """
    from collections import defaultdict
    by_face = defaultdict(list)
    for label, face_id in elem_face_list:
        by_face[face_id].append(label)

    face_kw_map = {
        'S1': 'face1Elements', 'S2': 'face2Elements',
        'S3': 'face3Elements', 'S4': 'face4Elements',
        'S5': 'face5Elements', 'S6': 'face6Elements',
    }

    kwargs = {'name': name}
    for face_id, labels in by_face.items():
        kw = face_kw_map.get(face_id)
        if kw and labels:
            kwargs[kw] = inst.elements.sequenceFromLabels(labels)

    return assembly.Surface(**kwargs)


def create_interactions(model, assembly,
                        inst_inner, inst_core, inst_outer,
                        defect_params):
    """Create General Contact (Explicit) with surface-based CZM.

    CohesiveBehavior requires General Contact in Explicit — cannot use
    SurfaceToSurfaceContactExp (pair-based).
    Strategy:
      - Create named surfaces for core inner/outer, inner/outer skin
      - Use ContactExp (General Contact) with surface pair assignments:
        * Core-Inner ↔ InnerSkin: CZM-Healthy
        * Core-Outer-Healthy ↔ OuterSkin: CZM-Healthy
        * Core-Outer-Defect ↔ OuterSkin: CZM-Damaged (frictionless)
        * All other: default (hard + frictionless)
    """
    # --- Create mesh-based surfaces ---
    # Core surfaces (inner/outer healthy/defect) from element faces
    surf_core_inner, surf_core_outer_h, surf_core_outer_d = \
        _classify_core_surfaces_mesh(assembly, inst_core, defect_params)

    # Inner skin all faces
    surf_inner = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')

    # Outer skin all faces
    surf_outer = assembly.Surface(
        side1Faces=inst_outer.faces, name='Surf-OuterSkin')

    # --- General Contact (Explicit) ---
    gc = model.ContactExp(name='GeneralContact', createStepName='Initial')

    # Include all surfaces with self-contact
    gc.includedPairs.setValuesInStep(
        stepName='Initial', useAllstar=ON)

    # Default property: hard + frictionless
    gc.contactPropertyAssignments.appendInStep(
        stepName='Initial',
        assignments=((GLOBAL, SELF, 'IntProp-Default'),))

    # Surface pair assignments for CZM
    pair_assignments = []

    # Inner interface: CZM-Healthy
    if surf_core_inner is not None:
        pair_assignments.append(
            (assembly.surfaces['Surf-Core-Inner'],
             assembly.surfaces['Surf-InnerSkin'],
             'IntProp-CZM-Healthy'))

    # Outer interface healthy: CZM-Healthy
    if surf_core_outer_h is not None:
        pair_assignments.append(
            (assembly.surfaces['Surf-Core-Outer-Healthy'],
             assembly.surfaces['Surf-OuterSkin'],
             'IntProp-CZM-Healthy'))

    # Outer interface defect: CZM-Damaged (frictionless, no CZM)
    if surf_core_outer_d is not None:
        pair_assignments.append(
            (assembly.surfaces['Surf-Core-Outer-Defect'],
             assembly.surfaces['Surf-OuterSkin'],
             'IntProp-CZM-Damaged'))

    if pair_assignments:
        gc.contactPropertyAssignments.appendInStep(
            stepName='Initial',
            assignments=tuple(pair_assignments))

    n_h = 1 if surf_core_outer_h else 0
    n_d = 1 if surf_core_outer_d else 0
    print("General Contact: %d CZM pairs (%d outer healthy, %d outer defect)" % (
        len(pair_assignments), n_h, n_d))


def create_frame_ties(model, assembly, inst_inner, frame_instances):
    """Tie ring frames to inner skin."""
    for i, fi in enumerate(frame_instances):
        surf_frame = assembly.Surface(
            side1Faces=fi.faces, name='Surf-Frame-%d' % i)
        model.Tie(
            name='Tie-Frame-%d' % i,
            main=surf_frame,
            secondary=assembly.Surface(
                side1Faces=inst_inner.faces,
                name='Surf-InnerForFrame-%d' % i),
            positionToleranceMethod=COMPUTED,
            adjust=ON, tieRotations=ON)
    if frame_instances:
        print("Frame ties: %d" % len(frame_instances))


# ==============================================================================
# MESH
# ==============================================================================

def generate_mesh(assembly, inst_inner, inst_core, inst_outer,
                  frame_instances, mesh_seed):
    """Generate EXPLICIT mesh for all parts."""
    all_insts = [inst_inner, inst_core, inst_outer] + list(frame_instances)

    # Global seed
    assembly.seedPartInstance(regions=tuple(all_insts), size=mesh_seed,
                              deviationFactor=0.1)

    # Core: override thickness-direction edges to 2 divisions
    # (prevents excessive through-thickness refinement)
    n_overridden = 0
    for edge in inst_core.edges:
        try:
            verts = edge.getVertices()
            if len(verts) == 2:
                n0 = inst_core.vertices[verts[0]].pointOn[0]
                n1 = inst_core.vertices[verts[1]].pointOn[0]
                # Radial direction: same theta, same y, different r
                r0 = math.sqrt(n0[0] ** 2 + n0[2] ** 2)
                r1 = math.sqrt(n1[0] ** 2 + n1[2] ** 2)
                dr = abs(r1 - r0)
                dy = abs(n1[1] - n0[1])
                # Thickness edge: radial change ~CORE_T, small axial/circ change
                if dr > CORE_T * 0.5 and dy < 1.0:
                    assembly.seedEdgeByNumber(
                        edges=(edge,), number=2, constraint=FIXED)
                    n_overridden += 1
        except Exception:
            pass
    if n_overridden > 0:
        print("  Core thickness edges: %d overridden to 2 divisions" % n_overridden)

    # Core: SWEEP hex (revolved body) or FREE tet fallback
    try:
        inst_core.setMeshControls(
            regions=inst_core.cells, technique=SWEEP,
            algorithm=ADVANCING_FRONT)
    except Exception:
        try:
            inst_core.setMeshControls(
                regions=inst_core.cells, elemShape=TET,
                technique=FREE, algorithm=DEFAULT)
        except Exception:
            print("  Warning: Core mesh controls failed, using defaults")

    # Element types: EXPLICIT library
    # Core: C3D8R (hex) + C3D6 (wedge) in EXPLICIT
    core_elem = mesh.ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
                               hourglassControl=DEFAULT)
    core_elem_tet = mesh.ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)
    core_elem_wedge = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
    try:
        inst_core.setElementType(
            regions=(inst_core.cells,),
            elemTypes=(core_elem, core_elem_wedge, core_elem_tet))
    except Exception:
        pass

    # Skins: S4R EXPLICIT
    shell_elem = mesh.ElemType(elemCode=S4R, elemLibrary=EXPLICIT)
    shell_elem_tri = mesh.ElemType(elemCode=S3R, elemLibrary=EXPLICIT)
    for inst in [inst_inner, inst_outer]:
        try:
            inst.setElementType(
                regions=(inst.faces,),
                elemTypes=(shell_elem, shell_elem_tri))
        except Exception:
            pass

    # Frames: S4R EXPLICIT
    for fi in frame_instances:
        try:
            fi.setElementType(
                regions=(fi.faces,),
                elemTypes=(shell_elem, shell_elem_tri))
        except Exception:
            pass

    # Generate mesh
    assembly.generateMesh(regions=tuple(all_insts))

    n_total = sum([len(inst.nodes) for inst in all_insts])
    n_elems = sum([len(inst.elements) for inst in all_insts])
    print("Mesh: seed=%.1f mm, %d nodes, %d elements" % (
        mesh_seed, n_total, n_elems))
    for inst in [inst_inner, inst_core, inst_outer]:
        print("  %s: %d nodes, %d elems" % (
            inst.name.split('-1')[0], len(inst.nodes), len(inst.elements)))


# ==============================================================================
# SYMMETRY BOUNDARY CONDITIONS
# ==============================================================================

def apply_symmetry_bcs(model, assembly,
                       inst_inner, inst_core, inst_outer,
                       frame_instances, sector_angle):
    """Apply symmetry BCs on sector cut faces.

    theta=0 (Z=0 plane): U3=UR1=UR2=0 (ZSYMM)
    theta=sector_angle: local CSYS normal constraint
    """
    shell_insts = [inst_inner, inst_outer] + list(frame_instances)
    solid_insts = [inst_core]
    all_insts = shell_insts + solid_insts
    r_box = RADIUS + CORE_T + 500.0
    tol = 1.0

    # --- theta=0 face (Z=0 plane) ---
    # Separate shell and solid for different DOF constraints
    sym0_shell_edges = None
    sym0_shell_faces = None
    sym0_solid_faces = None
    for inst in all_insts:
        is_solid = (inst in solid_insts)
        try:
            if not is_solid:
                edges = inst.edges.getByBoundingBox(
                    xMin=-1.0, xMax=r_box,
                    yMin=-0.1, yMax=10000.0,
                    zMin=-tol, zMax=tol)
                if len(edges) > 0:
                    sym0_shell_edges = edges if sym0_shell_edges is None else sym0_shell_edges + edges
        except Exception:
            pass
        try:
            face_pts = []
            for f in inst.faces:
                pt = f.pointOn[0]
                if abs(pt[2]) < tol and pt[0] > 0:
                    face_pts.append((pt,))
            if face_pts:
                fseq = inst.faces.findAt(*face_pts)
                if len(fseq) > 0:
                    if is_solid:
                        sym0_solid_faces = fseq if sym0_solid_faces is None else sym0_solid_faces + fseq
                    else:
                        sym0_shell_faces = fseq if sym0_shell_faces is None else sym0_shell_faces + fseq
        except Exception:
            pass

    # Shell BC: u3=0, ur1=0, ur2=0 (ZSYMM)
    set_kwargs_shell = {}
    if sym0_shell_edges is not None and len(sym0_shell_edges) > 0:
        set_kwargs_shell['edges'] = sym0_shell_edges
    if sym0_shell_faces is not None and len(sym0_shell_faces) > 0:
        set_kwargs_shell['faces'] = sym0_shell_faces
    if set_kwargs_shell:
        sym0_shell_set = assembly.Set(name='BC_Sym_Theta0_Shell', **set_kwargs_shell)
        model.DisplacementBC(name='Sym_Theta0_Shell', createStepName='Initial',
                             region=sym0_shell_set, u3=0, ur1=0, ur2=0)

    # Solid BC: u3=0 only (no rotational DOF)
    if sym0_solid_faces is not None and len(sym0_solid_faces) > 0:
        sym0_solid_set = assembly.Set(name='BC_Sym_Theta0_Solid', faces=sym0_solid_faces)
        model.DisplacementBC(name='Sym_Theta0_Solid', createStepName='Initial',
                             region=sym0_solid_set, u3=0)

    n_shell = sum([len(v) for v in set_kwargs_shell.values()]) if set_kwargs_shell else 0
    n_solid = len(sym0_solid_faces) if sym0_solid_faces else 0
    print("BC: Symmetry theta=0 (ZSYMM): shell=%d, solid=%d entities" % (n_shell, n_solid))

    # --- theta=sector_angle face ---
    theta_rad = math.radians(sector_angle)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Local Cartesian CSYS: Z' = face normal
    datum_csys = assembly.DatumCsysByThreePoints(
        name='CSYS-SymTheta',
        coordSysType=CARTESIAN,
        origin=(0.0, 0.0, 0.0),
        point1=(cos_t, 0.0, sin_t),     # local X (tangent)
        point2=(0.0, 1.0, 0.0))         # defines XY plane → Z' = normal

    sym_a_shell_edges = None
    sym_a_shell_faces = None
    sym_a_solid_faces = None
    for inst in all_insts:
        is_solid = (inst in solid_insts)
        try:
            if not is_solid:
                edge_pts = []
                for e in inst.edges:
                    pt = e.pointOn[0]
                    r_pt = math.sqrt(pt[0] ** 2 + pt[2] ** 2)
                    if r_pt < 1.0:
                        continue
                    theta_pt = math.atan2(pt[2], pt[0])
                    if abs(theta_pt - theta_rad) < 0.02:
                        edge_pts.append((pt,))
                if edge_pts:
                    eseq = inst.edges.findAt(*edge_pts)
                    if len(eseq) > 0:
                        sym_a_shell_edges = eseq if sym_a_shell_edges is None else sym_a_shell_edges + eseq
        except Exception:
            pass
        try:
            face_pts = []
            for f in inst.faces:
                pt = f.pointOn[0]
                r_pt = math.sqrt(pt[0] ** 2 + pt[2] ** 2)
                if r_pt < 1.0:
                    continue
                theta_pt = math.atan2(pt[2], pt[0])
                if abs(theta_pt - theta_rad) < 0.02:
                    face_pts.append((pt,))
            if face_pts:
                fseq = inst.faces.findAt(*face_pts)
                if len(fseq) > 0:
                    if is_solid:
                        sym_a_solid_faces = fseq if sym_a_solid_faces is None else sym_a_solid_faces + fseq
                    else:
                        sym_a_shell_faces = fseq if sym_a_shell_faces is None else sym_a_shell_faces + fseq
        except Exception:
            pass

    local_csys = assembly.datums[datum_csys.id]

    # Shell BC: u3=0, ur1=0, ur2=0 with local CSYS
    set_kwargs_a_shell = {}
    if sym_a_shell_edges is not None and len(sym_a_shell_edges) > 0:
        set_kwargs_a_shell['edges'] = sym_a_shell_edges
    if sym_a_shell_faces is not None and len(sym_a_shell_faces) > 0:
        set_kwargs_a_shell['faces'] = sym_a_shell_faces
    if set_kwargs_a_shell:
        sym_a_shell_set = assembly.Set(name='BC_Sym_ThetaMax_Shell', **set_kwargs_a_shell)
        model.DisplacementBC(
            name='Sym_ThetaMax_Shell', createStepName='Initial',
            region=sym_a_shell_set, u3=0, ur1=0, ur2=0,
            localCsys=local_csys)

    # Solid BC: u3=0 with local CSYS (no rotational DOF)
    if sym_a_solid_faces is not None and len(sym_a_solid_faces) > 0:
        sym_a_solid_set = assembly.Set(name='BC_Sym_ThetaMax_Solid', faces=sym_a_solid_faces)
        model.DisplacementBC(
            name='Sym_ThetaMax_Solid', createStepName='Initial',
            region=sym_a_solid_set, u3=0,
            localCsys=local_csys)

    n_shell_a = sum([len(v) for v in set_kwargs_a_shell.values()]) if set_kwargs_a_shell else 0
    n_solid_a = len(sym_a_solid_faces) if sym_a_solid_faces else 0
    print("BC: Symmetry theta=%.0f deg (local CSYS): shell=%d, solid=%d entities" % (
        sector_angle, n_shell_a, n_solid_a))


# ==============================================================================
# EXPLICIT STEP & TONE BURST
# ==============================================================================

def create_explicit_step(model, time_period):
    """Create Abaqus/Explicit step (no mass scaling)."""
    model.ExplicitDynamicsStep(
        name='Step-Wave',
        previous='Initial',
        timePeriod=time_period)

    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('U', 'V', 'A', 'S'),
        timeInterval=FIELD_OUTPUT_INTERVAL)

    print("Explicit step: T=%.3e s" % time_period)
    return 'Step-Wave'


def generate_tone_burst_amplitude(model, freq_hz, n_cycles,
                                  amp_name='Amp-ToneBurst'):
    """Hanning-windowed tone burst amplitude."""
    T_burst = float(n_cycles) / freq_hz
    points_per_cycle = 20
    n_points = n_cycles * points_per_cycle
    dt = T_burst / n_points

    amp_data = []
    for i in range(n_points + 1):
        t = i * dt
        hanning = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / T_burst))
        carrier = math.sin(2.0 * math.pi * freq_hz * t)
        amp_data.append((t, hanning * carrier))
    amp_data.append((T_burst + dt, 0.0))

    model.TabularAmplitude(name=amp_name, timeSpan=STEP,
                           smooth=SOLVER_DEFAULT,
                           data=tuple(amp_data))

    print("Tone burst: f=%.0f Hz, %d cycles, T=%.3e s" % (
        freq_hz, n_cycles, T_burst))
    return amp_name


# ==============================================================================
# EXCITATION & SENSORS
# ==============================================================================

def apply_excitation(model, assembly, inst_outer, freq_hz, n_cycles,
                     step_name, excite_theta_deg, excite_z):
    """Apply radial concentrated force on outer skin."""
    amp_name = generate_tone_burst_amplitude(model, freq_hz, n_cycles)

    r_outer = RADIUS + CORE_T
    theta_rad = math.radians(excite_theta_deg)
    cx = r_outer * math.cos(theta_rad)
    cy = excite_z
    cz = r_outer * math.sin(theta_rad)

    center_label, center_dist = find_nearest_node(inst_outer, cx, cy, cz)
    if center_label is None:
        print("ERROR: Could not find excitation node")
        return

    print("Excitation: node=%d, snap=%.2f mm, theta=%.1f deg, z=%.0f mm" % (
        center_label, center_dist, excite_theta_deg, excite_z))

    node_seq = inst_outer.nodes.sequenceFromLabels((center_label,))
    excite_set = assembly.Set(nodes=node_seq, name='Set-Excitation')

    # Cylindrical CSYS for radial force direction
    cyl_load_cs = assembly.DatumCsysByThreePoints(
        name='CylCS-Load', coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(1.0, 0.0, 0.0),
        point2=(0.0, 1.0, 0.0))

    # cf1 = radial (outward) → A0 mode excitation
    model.ConcentratedForce(
        name='Force-ToneBurst',
        createStepName=step_name,
        region=excite_set,
        cf1=FORCE_MAGNITUDE,
        amplitude=amp_name,
        localCsys=assembly.datums[cyl_load_cs.id])
    print("Radial force: %.1f N, amplitude=%s" % (FORCE_MAGNITUDE, amp_name))


def setup_sensor_outputs(model, assembly, inst_outer, step_name,
                         excite_theta_deg, excite_z, sector_angle,
                         n_sensors=10, spacing=SENSOR_SPACING):
    """Place sensors in cross pattern on outer skin.

    5 circumferential (vary theta, fixed z) + 5 axial (vary z, fixed theta).
    """
    r_outer = RADIUS + CORE_T
    n_circ = n_sensors // 2
    n_axial = n_sensors - n_circ
    sensor_count = 0

    # Circumferential sensors
    for i in range(n_circ):
        arc_offset = (i - n_circ // 2) * spacing
        d_theta = arc_offset / r_outer
        theta_s = math.radians(excite_theta_deg) + d_theta
        # Clamp within sector (with margin)
        theta_s = max(0.01, min(theta_s, math.radians(sector_angle) - 0.01))

        x = r_outer * math.cos(theta_s)
        y = excite_z
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
        print("  Sensor %d (circ): arc=%.0f mm, snap=%.1f mm" % (
            sensor_count, arc_offset, dist))
        sensor_count += 1

    # Axial sensors
    for i in range(n_axial):
        z_offset = (i - n_axial // 2) * spacing
        z_s = excite_z + z_offset
        theta_s = math.radians(excite_theta_deg)

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
        print("  Sensor %d (axial): dz=%.0f mm, snap=%.1f mm" % (
            sensor_count, z_offset, dist))
        sensor_count += 1

    print("Sensors placed: %d total (%d circ + %d axial)" % (
        sensor_count, n_circ, n_axial))


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def generate_model(job_name, freq_khz=DEFAULT_FREQ_KHZ, n_cycles=DEFAULT_CYCLES,
                   z_min=DEFAULT_Z_MIN, z_max=DEFAULT_Z_MAX,
                   sector_angle=DEFAULT_SECTOR_ANGLE,
                   mesh_seed=None, time_period=None,
                   defect_params=None, openings_phase=1,
                   include_frames=True, include_openings=True,
                   excite_z=None, excite_theta=None,
                   n_sensors=DEFAULT_N_SENSORS, sensor_spacing=SENSOR_SPACING,
                   czm_params=None, no_run=False):
    """Generate realistic fairing guided wave model."""
    freq_hz = freq_khz * 1e3

    # Auto mesh seed: lambda/8
    if mesh_seed is None:
        wavelength = CP_ESTIMATE / freq_hz * 1000.0
        mesh_seed = min(wavelength / 8.0, 4.0)

    # Auto time period
    if time_period is None:
        arc_length = (RADIUS + CORE_T) * math.radians(sector_angle)
        height = z_max - z_min
        diag = math.sqrt(arc_length ** 2 + height ** 2)
        time_period = max(diag / (CP_ESTIMATE * 1000.0) * 2.5, 0.5e-3)

    if czm_params is None:
        czm_params = DEFAULT_CZM

    # Default excitation position: sector center
    if excite_z is None:
        excite_z = (z_min + z_max) / 2.0
    if excite_theta is None:
        excite_theta = sector_angle / 2.0

    print("=" * 60)
    print("Guided Wave Fairing Model: %s" % job_name)
    print("=" * 60)
    print("  Sector: %.0f deg, z=[%.0f, %.0f] mm" % (
        sector_angle, z_min, z_max))
    print("  Freq: %.0f kHz, %d cycles" % (freq_khz, n_cycles))
    print("  Mesh seed: %.2f mm (lambda=%.1f mm)" % (
        mesh_seed, CP_ESTIMATE / freq_hz * 1000.0))
    print("  Time: %.3e s (%.2f ms)" % (time_period, time_period * 1e3))
    print("  Excitation: theta=%.1f deg, z=%.0f mm" % (
        excite_theta, excite_z))
    if defect_params:
        print("  Defect: z=%.0f, theta=%.1f, r=%.0f" % (
            defect_params['z_center'],
            defect_params['theta_deg'],
            defect_params['radius']))

    Mdb()
    model = mdb.models['Model-1']

    # 1. Materials, sections, contact properties
    create_materials(model)
    create_sections(model)
    create_contact_properties(model, czm_params)

    # 2. Geometry
    p_inner, p_core, p_outer = create_sector_parts(
        model, z_min, z_max, sector_angle)

    # 3. Ring frames
    frame_parts = []
    if include_frames:
        frame_z = [z for z in RING_FRAME_Z if z_min < z < z_max]
        frame_parts = create_ring_frame_parts(model, frame_z, sector_angle)

    # 4. Openings
    openings = []
    if include_openings:
        src = OPENINGS_PHASE2 if openings_phase >= 2 else OPENINGS_PHASE1
        openings = _filter_openings_in_range(
            src, z_min, z_max, 0.0, sector_angle)
        if openings:
            partition_all_openings(p_inner, p_core, p_outer, openings)
            print("Openings in range: %s" % ', '.join(
                o['name'] for o in openings))
        else:
            print("No openings in z/theta range")

    # 5. Defect partitioning
    if defect_params:
        partition_defect_zone(p_core, p_outer, defect_params)

    # 6. Section assignment (cylindrical CSYS)
    assign_sections(p_inner, p_core, p_outer, openings, defect_params)

    # 7. Assembly
    a, inst_inner, inst_core, inst_outer, frame_insts = create_assembly(
        model, p_inner, p_core, p_outer, frame_parts)

    # 8. Mesh (before interactions — need nodes for surfaces)
    generate_mesh(a, inst_inner, inst_core, inst_outer,
                  frame_insts, mesh_seed)

    # 9. Surface-based CZM interactions
    create_interactions(model, a, inst_inner, inst_core, inst_outer,
                        defect_params)

    # 10. Frame ties
    if frame_insts:
        create_frame_ties(model, a, inst_inner, frame_insts)

    # 11. Symmetry BCs
    apply_symmetry_bcs(model, a, inst_inner, inst_core, inst_outer,
                       frame_insts, sector_angle)

    # 12. Explicit step
    step_name = create_explicit_step(model, time_period)

    # 13. Excitation
    apply_excitation(model, a, inst_outer, freq_hz, n_cycles,
                     step_name, excite_theta, excite_z)

    # 14. Sensors
    setup_sensor_outputs(model, a, inst_outer, step_name,
                         excite_theta, excite_z, sector_angle,
                         n_sensors=n_sensors, spacing=sensor_spacing)

    # 15. Job
    mdb.Job(name=job_name, model='Model-1', type=ANALYSIS,
            resultsFormat=ODB, numCpus=4, numDomains=4,
            multiprocessingMode=DEFAULT,
            explicitPrecision=SINGLE, nodalOutputPrecision=FULL)
    mdb.saveAs(pathName=job_name + '.cae')
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
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
    import argparse

    parser = argparse.ArgumentParser(
        description='Realistic fairing guided wave model (Abaqus/Explicit)')
    parser.add_argument('--job', type=str, default='Job-GW-Fairing',
                        help='Job name')
    parser.add_argument('--freq', type=float, default=DEFAULT_FREQ_KHZ,
                        help='Frequency in kHz (default: 50)')
    parser.add_argument('--cycles', type=int, default=DEFAULT_CYCLES,
                        help='Tone burst cycles (default: 5)')
    parser.add_argument('--z_min', type=float, default=DEFAULT_Z_MIN,
                        help='Barrel z_min in mm (default: 500)')
    parser.add_argument('--z_max', type=float, default=DEFAULT_Z_MAX,
                        help='Barrel z_max in mm (default: 2500)')
    parser.add_argument('--sector_angle', type=float,
                        default=DEFAULT_SECTOR_ANGLE,
                        help='Sector angle in degrees (default: 30)')
    parser.add_argument('--mesh_seed', type=float, default=None,
                        help='Mesh seed in mm (auto: lambda/8)')
    parser.add_argument('--time', type=float, default=None,
                        help='Analysis time in seconds (auto)')
    parser.add_argument('--defect', type=str, default=None,
                        help='Defect JSON: {"z_center":X,"theta_deg":Y,"radius":Z}')
    parser.add_argument('--excite_z', type=float, default=None,
                        help='Excitation axial position in mm (auto: center)')
    parser.add_argument('--excite_theta', type=float, default=None,
                        help='Excitation angle in degrees (auto: center)')
    parser.add_argument('--n_sensors', type=int, default=DEFAULT_N_SENSORS,
                        help='Number of sensors (default: 10)')
    parser.add_argument('--sensor_spacing', type=float, default=SENSOR_SPACING,
                        help='Sensor spacing in mm (default: 30)')
    parser.add_argument('--phase', type=int, default=1,
                        help='Opening set: 1 or 2 (default: 1)')
    parser.add_argument('--no_openings', action='store_true',
                        help='Disable openings')
    parser.add_argument('--no_frames', action='store_true',
                        help='Disable ring frames')
    parser.add_argument('--no_run', action='store_true',
                        help='Write INP only, do not run solver')

    # Parse args — use parse_known_args to ignore Abaqus internal flags
    args, _ = parser.parse_known_args()

    # Parse defect JSON
    defect_data = None
    if args.defect:
        try:
            defect_data = json.loads(args.defect)
        except (ValueError, TypeError):
            print("ERROR: Invalid defect JSON: %s" % args.defect)
            sys.exit(1)

    generate_model(
        job_name=args.job,
        freq_khz=args.freq,
        n_cycles=args.cycles,
        z_min=args.z_min,
        z_max=args.z_max,
        sector_angle=args.sector_angle,
        mesh_seed=args.mesh_seed,
        time_period=args.time,
        defect_params=defect_data,
        openings_phase=args.phase,
        include_frames=not args.no_frames,
        include_openings=not args.no_openings,
        excite_z=args.excite_z,
        excite_theta=args.excite_theta,
        n_sensors=args.n_sensors,
        sensor_spacing=args.sensor_spacing,
        no_run=args.no_run,
    )
