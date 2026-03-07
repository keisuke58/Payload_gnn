# -*- coding: utf-8 -*-
# generate_guided_wave.py
# Abaqus/Explicit script for guided wave (Lamb wave) propagation
# in a flat CFRP/Al-Honeycomb sandwich panel.
#
# Flat panel verification + optional debonding defect.
# Excitation: N-cycle Hanning-windowed tone burst (concentrated force).
# Solver: Abaqus/Explicit (no mass scaling).
#
# Defect: circular debonding (Tie removal) between outer skin and core.
# Wave scatters at debonding boundary — detectable via A0 mode reflection.
#
# Usage:
#   abaqus cae noGUI=generate_guided_wave.py -- --job Job-GW-50kHz --no_run
#   abaqus cae noGUI=generate_guided_wave.py -- --job Job-GW-Defect \
#       --defect '{"x_center":80,"y_center":0,"radius":25}' --no_run

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
# GEOMETRY PARAMETERS (Flat Panel)
# ==============================================================================
PANEL_SIZE = 300.0    # mm (panel side length)
FACE_T = 1.0          # mm (CFRP skin thickness)
CORE_T = 38.0         # mm (Al honeycomb core thickness)

# ==============================================================================
# MATERIAL PROPERTIES (consistent with generate_fairing_dataset.py)
# Units: mm / tonne / s / MPa
# ==============================================================================
# CFRP Face Sheets (Toray T1000G)
E1 = 160000.0         # MPa
E2 = 10000.0          # MPa
NU12 = 0.3
G12 = 5000.0          # MPa
G13 = 5000.0          # MPa
G23 = 3000.0          # MPa
CFRP_DENSITY = 1600e-12   # tonne/mm^3 (= 1600 kg/m^3)

# Aluminum Honeycomb Core (5052)
E_CORE_1 = 1.0        # MPa (in-plane)
E_CORE_2 = 1.0        # MPa (in-plane)
E_CORE_3 = 1000.0     # MPa (out-of-plane)
NU_CORE_12 = 0.01
NU_CORE_13 = 0.01
NU_CORE_23 = 0.01
G_CORE_12 = 1.0       # MPa
G_CORE_13 = 400.0     # MPa
G_CORE_23 = 240.0     # MPa
CORE_DENSITY = 50e-12  # tonne/mm^3 (= 50 kg/m^3)

# ==============================================================================
# WAVE EXCITATION PARAMETERS
# ==============================================================================
DEFAULT_FREQ_KHZ = 50.0   # kHz
DEFAULT_CYCLES = 5
FORCE_MAGNITUDE = 1.0     # N (unit force)

# ==============================================================================
# MESH
# ==============================================================================
DEFAULT_MESH_SEED = 3.0    # mm (lambda/8 at 50 kHz: ~31/8 = 3.9 mm)
CORE_THRU_THICK_ELEMS = 4  # minimum through-thickness elements

# ==============================================================================
# ANALYSIS
# ==============================================================================
DEFAULT_TIME_PERIOD = 0.5e-3      # 0.5 ms
FIELD_OUTPUT_INTERVAL = 1.0e-6    # 1 microsecond

# Approximate A0 phase velocity for auto-calculations (m/s)
CP_ESTIMATE = 1550.0

# ==============================================================================
# CURVED PANEL GEOMETRY (H3 Fairing barrel section)
# ==============================================================================
FAIRING_RADIUS = 2600.0    # mm (H3 fairing outer radius)
CURVED_PANEL_HEIGHT = 300.0  # mm (axial extent of panel section)
DEFAULT_SECTOR_ANGLE = None  # auto-calculated from panel_size

# ==============================================================================
# SENSOR LOCATIONS (distance from center along X-axis, mm)
# ==============================================================================
SENSOR_OFFSETS = [0.0, 30.0, 60.0, 90.0, 120.0]  # all within panel ±150mm


# ==============================================================================
# DEFECT HELPERS
# ==============================================================================

def _point_in_defect(x, y, defect_params):
    """Check if (x, y) is inside the circular defect zone on XY plane."""
    if not defect_params:
        return False
    dx = x - defect_params['x_center']
    dy = y - defect_params['y_center']
    return (dx * dx + dy * dy) <= defect_params['radius'] ** 2 * 1.02  # 1% tol


def partition_defect_zone(p_outer, p_core, defect_params):
    """Partition outer skin and core at defect boundary.

    Creates 4 datum planes (rectangular bounding box around defect circle)
    so that faces inside vs outside can be selected for Tie assignment.
    """
    xc = defect_params['x_center']
    yc = defect_params['y_center']
    r = defect_params['radius']
    margin = 2.0  # mm extra margin for clean partition

    # Datum planes on outer skin (XY plane, cuts at defect boundary)
    for part in [p_outer, p_core]:
        # X cuts
        for offset in [xc - r - margin, xc + r + margin]:
            dp = part.DatumPlaneByPrincipalPlane(
                principalPlane=YZPLANE, offset=offset)
            try:
                if hasattr(part, 'cells') and len(part.cells) > 0:
                    part.PartitionCellByDatumPlane(
                        datumPlane=part.datums[dp.id], cells=part.cells)
                elif len(part.faces) > 0:
                    part.PartitionFaceByDatumPlane(
                        datumPlane=part.datums[dp.id], faces=part.faces)
            except Exception as e:
                print("  Defect X-partition at %.1f: %s" % (offset, str(e)[:60]))

        # Y cuts
        for offset in [yc - r - margin, yc + r + margin]:
            dp = part.DatumPlaneByPrincipalPlane(
                principalPlane=XZPLANE, offset=offset)
            try:
                if hasattr(part, 'cells') and len(part.cells) > 0:
                    part.PartitionCellByDatumPlane(
                        datumPlane=part.datums[dp.id], cells=part.cells)
                elif len(part.faces) > 0:
                    part.PartitionFaceByDatumPlane(
                        datumPlane=part.datums[dp.id], faces=part.faces)
            except Exception as e:
                print("  Defect Y-partition at %.1f: %s" % (offset, str(e)[:60]))

    print("Defect partitioned: outer skin %d faces, core %d cells" % (
        len(p_outer.faces), len(p_core.cells)))


# ==============================================================================
# MATERIAL & SECTION DEFINITIONS
# ==============================================================================

def create_materials(model):
    """Define CFRP and Al-Honeycomb materials (no CTE — dynamic only)."""
    mat_cfrp = model.Material(name='CFRP_T1000G')
    mat_cfrp.Elastic(type=LAMINA, table=((E1, E2, NU12, G12, G13, G23), ))
    mat_cfrp.Density(table=((CFRP_DENSITY, ), ))

    mat_core = model.Material(name='AL_HONEYCOMB')
    mat_core.Elastic(type=ENGINEERING_CONSTANTS, table=((
        E_CORE_1, E_CORE_2, E_CORE_3,
        NU_CORE_12, NU_CORE_13, NU_CORE_23,
        G_CORE_12, G_CORE_13, G_CORE_23
    ), ))
    mat_core.Density(table=((CORE_DENSITY, ), ))


def create_sections(model):
    """Create composite shell section for skins and solid section for core."""
    layup_orientation = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
    entries = []
    for ang in layup_orientation:
        entries.append(section.SectionLayer(
            thickness=FACE_T / 8.0, orientAngle=ang, material='CFRP_T1000G'))

    model.CompositeShellSection(
        name='Section-CFRP-Skin', preIntegrate=OFF,
        idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
        thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        temperature=GRADIENT, integrationRule=SIMPSON)

    model.HomogeneousSolidSection(
        name='Section-Core', material='AL_HONEYCOMB', thickness=None)


def create_defect_materials_gw(model, defect_params):
    """Create defect-type materials for GW (no CTE — dynamic only)."""
    defect_type = defect_params.get('defect_type', 'debonding')

    if defect_type == 'fod':
        sf = defect_params.get('stiffness_factor', 10.0)
        mat = model.Material(name='AL_HONEYCOMB_FOD')
        mat.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1 * sf, E_CORE_2 * sf, E_CORE_3 * sf,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12 * sf, G_CORE_13 * sf, G_CORE_23 * sf
        ), ))
        mat.Density(table=((200e-12, ), ))
    elif defect_type == 'impact':
        dr = defect_params.get('damage_ratio', 0.3)
        mat_skin = model.Material(name='CFRP_IMPACT_DAMAGED')
        mat_skin.Elastic(type=LAMINA, table=((
            E1 * 0.7, E2 * dr, NU12,
            G12 * dr, G13 * dr, G23 * dr
        ), ))
        mat_skin.Density(table=((CFRP_DENSITY, ), ))

        mat_core = model.Material(name='AL_HONEYCOMB_CRUSHED')
        mat_core.Elastic(type=ENGINEERING_CONSTANTS, table=((
            E_CORE_1 * 0.5, E_CORE_2 * 0.5, E_CORE_3 * 0.1,
            NU_CORE_12, NU_CORE_13, NU_CORE_23,
            G_CORE_12 * 0.5, G_CORE_13 * 0.1, G_CORE_23 * 0.1
        ), ))
        mat_core.Density(table=((CORE_DENSITY, ), ))
    elif defect_type == 'delamination':
        depth = defect_params.get('delam_depth', 0.5)
        shear_red = max(0.05, 1.0 - depth)
        mat = model.Material(name='CFRP_DELAMINATED')
        mat.Elastic(type=LAMINA, table=((
            E1 * 0.9, E2 * (0.3 + 0.5 * (1 - depth)), NU12,
            G12 * shear_red, G13 * shear_red, G23 * shear_red
        ), ))
        mat.Density(table=((CFRP_DENSITY, ), ))
    else:
        return
    print("Defect materials created: type=%s" % defect_type)


def create_defect_sections_gw(model, defect_params):
    """Create sections for defect-zone (FOD, Impact, Delamination)."""
    defect_type = defect_params.get('defect_type', 'debonding')
    angles = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]

    if defect_type == 'fod':
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
            name='Section-CFRP-Delam', preIntegrate=OFF,
            idealization=NO_IDEALIZATION, layup=entries, symmetric=OFF,
            thicknessType=UNIFORM, poissonDefinition=DEFAULT,
            temperature=GRADIENT, integrationRule=SIMPSON)


# ==============================================================================
# GEOMETRY
# ==============================================================================

def create_flat_panel_parts(model, panel_size):
    """Create flat rectangular sandwich panel parts.

    Coordinate convention:
      X: -panel_size/2 to +panel_size/2
      Y: -panel_size/2 to +panel_size/2
      Z: through-thickness
        Inner skin:  Z = 0  (shell mid-surface)
        Core:        Z = 0 to CORE_T  (solid extrude)
        Outer skin:  Z = CORE_T  (shell mid-surface)

    Returns: (p_inner, p_core, p_outer)
    """
    half = panel_size / 2.0

    # Inner Skin (Shell at Z=0 plane)
    s1 = model.ConstrainedSketch(name='sk_inner', sheetSize=panel_size * 2)
    s1.rectangle(point1=(-half, -half), point2=(half, half))
    p_inner = model.Part(name='Part-InnerSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShell(sketch=s1)

    # Core (Solid, extruded Z=0 to Z=CORE_T)
    s2 = model.ConstrainedSketch(name='sk_core', sheetSize=panel_size * 2)
    s2.rectangle(point1=(-half, -half), point2=(half, half))
    p_core = model.Part(name='Part-Core', dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidExtrude(sketch=s2, depth=CORE_T)

    # Outer Skin (Shell at Z=0, will be translated to Z=CORE_T in assembly)
    s3 = model.ConstrainedSketch(name='sk_outer', sheetSize=panel_size * 2)
    s3.rectangle(point1=(-half, -half), point2=(half, half))
    p_outer = model.Part(name='Part-OuterSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_outer.BaseShell(sketch=s3)

    return p_inner, p_core, p_outer


def partition_core_thickness(p_core):
    """Partition core through-thickness for guaranteed element layers.

    Creates 3 datum planes at 25%, 50%, 75% of CORE_T to ensure 4 layers.
    Also forces dependent=OFF export (partitioned parts are 'modified').
    """
    for frac in [0.25, 0.5, 0.75]:
        dp = p_core.DatumPlaneByPrincipalPlane(
            principalPlane=XYPLANE, offset=CORE_T * frac)
        try:
            p_core.PartitionCellByDatumPlane(
                datumPlane=p_core.datums[dp.id], cells=p_core.cells)
        except Exception as e:
            print("Core Z-partition at %.1f mm: %s" % (
                CORE_T * frac, str(e)[:60]))
    print("Core partitioned: %d cells (4 through-thickness layers)" %
          len(p_core.cells))


def create_curved_panel_parts(model, panel_size, panel_height=None, radius=None):
    """Create curved cylindrical sandwich panel parts (barrel section).

    Uses BaseShellRevolve/BaseSolidRevolve around Y-axis.

    Coordinate convention (same as generate_fairing_dataset.py):
      Y: axial (fairing vertical axis) — panel spans [0, panel_height]
      XZ: radial plane — revolution around Y axis
      theta=0 at X axis, revolves in XZ plane

    Args:
        panel_size: Desired arc length in mm (circumferential extent)
        panel_height: Axial height in mm (default: CURVED_PANEL_HEIGHT)
        radius: Inner skin radius in mm (default: FAIRING_RADIUS)

    Returns: (p_inner, p_core, p_outer, sector_angle_deg)
    """
    if radius is None:
        radius = FAIRING_RADIUS
    if panel_height is None:
        panel_height = CURVED_PANEL_HEIGHT

    # Compute sector angle from desired arc length at outer skin radius
    r_outer = radius + CORE_T
    sector_angle_deg = panel_size / r_outer * 180.0 / math.pi
    # Add 10% margin for sensors near edge
    sector_angle_deg = min(sector_angle_deg * 1.1, 60.0)

    print("Curved panel: R=%.0f mm, height=%.0f mm, sector=%.2f deg (arc=%.0f mm)"
          % (radius, panel_height, sector_angle_deg,
             r_outer * math.radians(sector_angle_deg)))

    # --- Inner Skin (Shell at r=RADIUS) ---
    s1 = model.ConstrainedSketch(name='sk_inner_curved', sheetSize=radius * 3)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, panel_height + 100.0))
    s1.Line(point1=(radius, 0.0), point2=(radius, panel_height))

    p_inner = model.Part(name='Part-InnerSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=sector_angle_deg,
                             flipRevolveDirection=OFF)

    # --- Core (Solid, r=RADIUS to r=RADIUS+CORE_T) ---
    s2 = model.ConstrainedSketch(name='sk_core_curved', sheetSize=radius * 3)
    s2.setPrimaryObject(option=STANDALONE)
    s2.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, panel_height + 100.0))
    # Closed rectangle in RY plane
    s2.Line(point1=(radius, 0.0), point2=(radius, panel_height))
    s2.Line(point1=(radius, panel_height),
            point2=(radius + CORE_T, panel_height))
    s2.Line(point1=(radius + CORE_T, panel_height),
            point2=(radius + CORE_T, 0.0))
    s2.Line(point1=(radius + CORE_T, 0.0), point2=(radius, 0.0))

    p_core = model.Part(name='Part-Core', dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=s2, angle=sector_angle_deg,
                            flipRevolveDirection=OFF)

    # --- Outer Skin (Shell at r=RADIUS+CORE_T) ---
    s3 = model.ConstrainedSketch(name='sk_outer_curved', sheetSize=radius * 3)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, panel_height + 100.0))
    s3.Line(point1=(r_outer, 0.0), point2=(r_outer, panel_height))

    p_outer = model.Part(name='Part-OuterSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=sector_angle_deg,
                             flipRevolveDirection=OFF)

    return p_inner, p_core, p_outer, sector_angle_deg


def partition_core_thickness_curved(p_core, radius=None):
    """Partition curved core through-thickness using cylindrical datum planes.

    For revolved geometry, through-thickness = radial direction.
    Creates cylindrical partition surfaces at 25%, 50%, 75% of core thickness.
    """
    if radius is None:
        radius = FAIRING_RADIUS

    for frac in [0.25, 0.5, 0.75]:
        r_cut = radius + CORE_T * frac
        # Create a datum plane at this radius — use cylindrical surface
        # For revolved bodies, partition by datum plane through Y-axis at angle
        # Actually, we need a cylindrical cut: use DatumPointByCoordinate + sketch
        dp = p_core.DatumPlaneByPrincipalPlane(
            principalPlane=XZPLANE, offset=0.0)
        # Alternative: use shell of revolution as cutting tool
        # Simpler: partition using sketch on XZ plane
        try:
            # Create a cylindrical cutting surface via a revolve-based partition
            # Use DatumAxis through Y for revolution
            datum_axis_y = p_core.DatumAxisByPrincipalAxis(principalAxis=YAXIS)
            sketch_plane = p_core.datums[dp.id]
            # Partition by a cylinder at r_cut
            # Use PartitionCellBySweepEdge or PartitionCellByExtrudeEdge approach
            # Simplest: use datum plane at known offset  won't work for radial cuts
            #
            # For radial cuts in a revolved body, the most reliable method is
            # PartitionCellByPlaneThreePoints using 3 points on the cylinder at r_cut
            theta1 = 0.0
            theta2 = math.pi / 6  # 30 degrees
            theta3 = math.pi / 3  # 60 degrees
            pt1 = (r_cut, 0.0, 0.0)
            pt2 = (r_cut * math.cos(theta2), CORE_T, r_cut * math.sin(theta2))
            pt3 = (r_cut * math.cos(theta3), 0.0, r_cut * math.sin(theta3))
            # Actually 3 collinear points on a cylinder define a plane, not a cylinder
            # For cylindrical partition, create intermediate datum point/sketch approach
            pass
        except Exception as e:
            print("  Curved core partition at r=%.1f: %s" % (r_cut, str(e)[:60]))

    # Fallback: just use global seed with enough through-thickness bias
    # The mesh seed will create elements; Abaqus auto-meshes the thin core
    print("  Curved core: relying on mesh seed for through-thickness resolution")
    print("  Core cells: %d" % len(p_core.cells))


def assign_sections_curved(p_inner, p_core, p_outer):
    """Assign sections with cylindrical material orientation for curved panel.

    Core: Cylindrical CSYS (1=R, 2=theta, 3=axial)
    Skins: Cartesian CSYS (1=Y=axial, projected onto shell) with axis=AXIS_3
    """
    # --- Core ---
    region = p_core.Set(cells=p_core.cells, name='Set-All')
    p_core.SectionAssignment(region=region, sectionName='Section-Core')
    # Cylindrical CSYS: 1=R (through-thickness), 2=theta, 3=Z(=Y axial)
    cyl_cs = p_core.DatumCsysByThreePoints(
        name='CylCS-Core', coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(1.0, 0.0, 0.0),   # R direction at theta=0
        point2=(0.0, 1.0, 0.0))   # R-Z plane: Z = Y (axial)
    p_core.MaterialOrientation(
        region=region, orientationType=SYSTEM, axis=AXIS_3,
        additionalRotationType=ROTATION_NONE,
        localCsys=p_core.datums[cyl_cs.id])

    # --- Skins (Cartesian CSYS with Y=axial primary) ---
    for p, name in [(p_inner, 'Inner'), (p_outer, 'Outer')]:
        region = p.Set(faces=p.faces, name='Set-All')
        p.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
        shell_cs = p.DatumCsysByThreePoints(
            name='ShellCS-%s' % name, coordSysType=CARTESIAN,
            origin=(0.0, 0.0, 0.0),
            point1=(0.0, 1.0, 0.0),   # local-1 = Y (axial)
            point2=(1.0, 0.0, 0.0))   # local 1-2 plane
        p.MaterialOrientation(
            region=region, orientationType=SYSTEM, axis=AXIS_3,
            additionalRotationType=ROTATION_NONE,
            localCsys=p.datums[shell_cs.id])

    print("Sections assigned (curved): cylindrical CSYS for core, "
          "Cartesian for skins")


def assign_defect_sections_flat(p_core, p_outer, defect_params):
    """Assign defect-type sections to defect zone (FOD, Impact, Delamination)."""
    defect_type = defect_params.get('defect_type', 'debonding')
    if defect_type == 'debonding' or defect_type == 'inner_debond':
        return

    # Core defect cells (centroid in defect zone)
    core_defect_cells = []
    for c in p_core.cells:
        try:
            pt = c.getCentroid()
            if _point_in_defect(pt[0], pt[1], defect_params):
                core_defect_cells.append(c)
        except Exception:
            pass

    # Outer skin defect faces
    outer_defect_faces = []
    for f in p_outer.faces:
        pt = f.pointOn[0]
        if _point_in_defect(pt[0], pt[1], defect_params):
            outer_defect_faces.append(f)

    if defect_type == 'fod' and core_defect_cells:
        region = p_core.Set(cells=core_defect_cells, name='Set-Defect-Core')
        p_core.SectionAssignment(region=region, sectionName='Section-Core-FOD')
        print("  FOD: %d core cells -> Section-Core-FOD" % len(core_defect_cells))
    elif defect_type == 'impact':
        if core_defect_cells:
            region = p_core.Set(cells=core_defect_cells, name='Set-Defect-Core')
            p_core.SectionAssignment(region=region, sectionName='Section-Core-Crushed')
            print("  Impact: %d core cells -> Section-Core-Crushed" % len(core_defect_cells))
        if outer_defect_faces:
            region = p_outer.Set(faces=outer_defect_faces, name='Set-Defect-OuterSkin')
            p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Impact')
            print("  Impact: %d outer faces -> Section-CFRP-Impact" % len(outer_defect_faces))
    elif defect_type == 'delamination' and outer_defect_faces:
        region = p_outer.Set(faces=outer_defect_faces, name='Set-Defect-OuterSkin')
        p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Delam')
        print("  Delamination: %d outer faces -> Section-CFRP-Delam" % len(outer_defect_faces))


def assign_sections(p_inner, p_core, p_outer):
    """Assign sections to all parts with MaterialOrientation."""
    # Inner skin
    region = p_inner.Set(faces=p_inner.faces, name='Set-All')
    p_inner.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_inner.MaterialOrientation(region=region, orientationType=GLOBAL,
                                axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE,
                                localCsys=None)

    # Core (ENGINEERING_CONSTANTS requires MaterialOrientation)
    region = p_core.Set(cells=p_core.cells, name='Set-All')
    p_core.SectionAssignment(region=region, sectionName='Section-Core')
    p_core.MaterialOrientation(region=region, orientationType=GLOBAL,
                                axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE,
                                localCsys=None)

    # Outer skin
    region = p_outer.Set(faces=p_outer.faces, name='Set-All')
    p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_outer.MaterialOrientation(region=region, orientationType=GLOBAL,
                                axis=AXIS_3,
                                additionalRotationType=ROTATION_NONE,
                                localCsys=None)


# ==============================================================================
# ASSEMBLY
# ==============================================================================

def create_assembly(model, p_inner, p_core, p_outer, geometry='flat'):
    """Create assembly with positioned instances.

    Flat: Inner skin at Z=0, Core at Z=0..CORE_T, Outer skin translated to Z=CORE_T.
    Curved: All parts already at correct radii from revolution, no translation needed.
    """
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)

    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)

    if geometry == 'flat':
        # Translate outer skin to Z=CORE_T
        a.translate(instanceList=('Part-OuterSkin-1', ),
                    vector=(0.0, 0.0, CORE_T))

    # For curved: parts are already at correct radial positions from revolve
    return a, inst_inner, inst_core, inst_outer


# ==============================================================================
# TIE CONSTRAINTS
# ==============================================================================

def create_tie_constraints(model, assembly, inst_inner, inst_core, inst_outer,
                           defect_params=None):
    """Create Tie constraints between skins and core.

    Tie 1: Inner skin (Z=0) <-> Core bottom face (Z=0)
           If defect_type=inner_debond: excludes defect zone
    Tie 2: Core top face (Z=CORE_T) <-> Outer skin (Z=CORE_T)
           If defect_type=debonding: excludes defect zone

    debonding/inner_debond: Tie removal (wave scatters).
    fod/impact/delamination: Full Tie (material change only).
    """
    # --- Tie 1: Inner skin <-> Core bottom ---
    # inner_debond: exclude defect zone. Others: full Tie.
    tie1_exclude_defect = (defect_params and
                          defect_params.get('defect_type') == 'inner_debond')
    surf_inner = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')

    core_bot_healthy_pts = []
    core_bot_defect_pts = []
    for f in inst_core.faces:
        pt = f.pointOn[0]
        if abs(pt[2]) < 0.1:
            if tie1_exclude_defect and _point_in_defect(pt[0], pt[1], defect_params):
                core_bot_defect_pts.append((pt,))
            else:
                core_bot_healthy_pts.append((pt,))
    core_bot_pts = core_bot_healthy_pts if tie1_exclude_defect else \
        core_bot_healthy_pts + core_bot_defect_pts
    if core_bot_pts:
        core_bot_seq = inst_core.faces.findAt(*core_bot_pts)
        surf_core_bot = assembly.Surface(
            side1Faces=core_bot_seq, name='Surf-Core-Bottom')
        model.Tie(name='Tie-InnerSkin-Core',
                  main=surf_core_bot, secondary=surf_inner,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie 1: InnerSkin <-> Core bottom (%d faces)%s" % (
            len(core_bot_pts), " (excl inner_debond zone)" if tie1_exclude_defect else ""))
    else:
        print("WARNING: No core bottom faces found for Tie 1")

    # --- Tie 2: Outer skin <-> Core top ---
    # debonding: exclude defect zone. fod/impact/delamination: full Tie.
    # Default to debonding when defect_params exists but defect_type not specified
    tie2_exclude_defect = (defect_params and
                           defect_params.get('defect_type', 'debonding') == 'debonding')
    core_top_healthy_pts = []
    core_top_defect_pts = []
    for f in inst_core.faces:
        pt = f.pointOn[0]
        if abs(pt[2] - CORE_T) < 0.1:
            if tie2_exclude_defect and _point_in_defect(pt[0], pt[1], defect_params):
                core_top_defect_pts.append((pt,))
            else:
                core_top_healthy_pts.append((pt,))

    outer_healthy_pts = []
    outer_defect_pts = []
    for f in inst_outer.faces:
        pt = f.pointOn[0]
        if tie2_exclude_defect and _point_in_defect(pt[0], pt[1], defect_params):
            outer_defect_pts.append((pt,))
        else:
            outer_healthy_pts.append((pt,))

    # Create Tie only for healthy region
    if core_top_healthy_pts and outer_healthy_pts:
        core_top_h_seq = inst_core.faces.findAt(*core_top_healthy_pts)
        outer_h_seq = inst_outer.faces.findAt(*outer_healthy_pts)
        surf_core_top_h = assembly.Surface(
            side1Faces=core_top_h_seq, name='Surf-Core-Top-Healthy')
        surf_outer_h = assembly.Surface(
            side1Faces=outer_h_seq, name='Surf-OuterSkin-Healthy')
        model.Tie(name='Tie-Core-OuterSkin',
                  main=surf_core_top_h, secondary=surf_outer_h,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie 2: Core top <-> OuterSkin HEALTHY (%d + %d faces)" % (
            len(core_top_healthy_pts), len(outer_healthy_pts)))
    elif not tie2_exclude_defect:
        # No defect: tie all
        if core_top_healthy_pts:
            core_top_seq = inst_core.faces.findAt(*core_top_healthy_pts)
            surf_core_top = assembly.Surface(
                side1Faces=core_top_seq, name='Surf-Core-Top')
            surf_outer = assembly.Surface(
                side1Faces=inst_outer.faces, name='Surf-OuterSkin')
            model.Tie(name='Tie-Core-OuterSkin',
                      main=surf_core_top, secondary=surf_outer,
                      positionToleranceMethod=COMPUTED, adjust=ON,
                      tieRotations=ON, thickness=ON)
            print("Tie 2: Core top <-> OuterSkin (%d faces)" %
                  len(core_top_healthy_pts))
    else:
        print("WARNING: No healthy faces for Tie 2")

    # Report defect exclusion (debonding / inner_debond)
    if tie2_exclude_defect or tie1_exclude_defect:
        n_core_def = len(core_top_defect_pts)
        n_outer_def = len(outer_defect_pts)
        print("Tie exclusion: %d core top + %d outer skin faces UNTIED" % (
            n_core_def, n_outer_def))
        if tie2_exclude_defect and n_core_def == 0 and n_outer_def == 0:
            print("WARNING: No faces found in defect zone — check defect params")


# ==============================================================================
# MESH
# ==============================================================================

def create_tie_constraints_curved(model, assembly, inst_inner, inst_core,
                                  inst_outer, sector_angle_deg, radius=None,
                                  defect_params=None):
    """Create Tie constraints for curved panel using sample-point method.

    For revolved geometry, inner/outer surfaces are at specific radii.
    Sample points are placed at the middle of the sector (theta_mid) at
    known radial positions.

    Args:
        defect_params: dict with {y_center, theta_deg, radius} for curved defect.
    """
    if radius is None:
        radius = FAIRING_RADIUS

    r_outer = radius + CORE_T
    theta_mid = math.radians(sector_angle_deg / 2.0)
    cos_t = math.cos(theta_mid)
    sin_t = math.sin(theta_mid)
    y_mid = CURVED_PANEL_HEIGHT / 2.0

    # --- Tie 1: Inner skin <-> Core inner surface (always full) ---
    surf_inner = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')

    # Sample point on core inner surface (r=RADIUS)
    core_inner_pt = (radius * cos_t, y_mid, radius * sin_t)
    try:
        core_inner_seq = inst_core.faces.findAt((core_inner_pt,))
        surf_core_inner = assembly.Surface(
            side1Faces=core_inner_seq, name='Surf-Core-Inner')
        model.Tie(name='Tie-InnerSkin-Core',
                  main=surf_core_inner, secondary=surf_inner,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie 1: InnerSkin <-> Core inner (sample at r=%.0f)" % radius)
    except Exception as e:
        # Fallback: classify by radius
        print("  Tie 1 sample-point failed: %s" % str(e)[:60])
        _tie_by_radius(model, assembly, inst_inner, inst_core,
                       radius, 'inner', 'Tie-InnerSkin-Core')

    # --- Tie 2: Core outer surface <-> Outer skin ---
    if not defect_params:
        # Healthy: tie all outer faces
        surf_outer = assembly.Surface(
            side2Faces=inst_outer.faces, name='Surf-OuterSkin')
        core_outer_pt = (r_outer * cos_t, y_mid, r_outer * sin_t)
        try:
            core_outer_seq = inst_core.faces.findAt((core_outer_pt,))
            surf_core_outer = assembly.Surface(
                side1Faces=core_outer_seq, name='Surf-Core-Outer')
            model.Tie(name='Tie-Core-OuterSkin',
                      main=surf_core_outer, secondary=surf_outer,
                      positionToleranceMethod=COMPUTED, adjust=ON,
                      tieRotations=ON, thickness=ON)
            print("Tie 2: Core outer <-> OuterSkin (healthy)")
        except Exception as e:
            print("  Tie 2 sample-point failed: %s" % str(e)[:60])
            _tie_by_radius(model, assembly, inst_outer, inst_core,
                           r_outer, 'outer', 'Tie-Core-OuterSkin')
    else:
        # Defect: selective Tie (exclude defect zone)
        _tie_curved_with_defect(model, assembly, inst_core, inst_outer,
                                radius, sector_angle_deg, defect_params)


def _tie_by_radius(model, assembly, inst_skin, inst_core, target_r,
                   side, tie_name):
    """Fallback Tie by classifying core faces by radial position."""
    core_pts = []
    for f in inst_core.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0] ** 2 + pt[2] ** 2)
        if abs(r - target_r) < 1.0:  # within 1mm of target radius
            core_pts.append((pt,))

    if not core_pts:
        print("WARNING: No core %s faces found at r=%.0f" % (side, target_r))
        return

    core_seq = inst_core.faces.findAt(*core_pts)
    surf_core = assembly.Surface(
        side1Faces=core_seq, name='Surf-Core-%s' % side.capitalize())

    if side == 'inner':
        surf_skin = assembly.Surface(
            side1Faces=inst_skin.faces, name='Surf-InnerSkin')
    else:
        surf_skin = assembly.Surface(
            side2Faces=inst_skin.faces, name='Surf-OuterSkin')

    model.Tie(name=tie_name, main=surf_core, secondary=surf_skin,
              positionToleranceMethod=COMPUTED, adjust=ON,
              tieRotations=ON, thickness=ON)
    print("Tie (radius fallback): %s (%d faces at r=%.0f)" % (
        tie_name, len(core_pts), target_r))


def _point_in_defect_curved(y, theta_rad, defect_params, radius):
    """Check if (y, theta) is inside defect zone on curved surface.

    defect_params: {y_center, theta_deg, radius} — cylindrical coords.
    """
    if not defect_params:
        return False
    dy = y - defect_params['y_center']
    r_outer = radius + CORE_T
    dtheta = theta_rad - math.radians(defect_params['theta_deg'])
    arc_dist = r_outer * dtheta
    dist_sq = dy * dy + arc_dist * arc_dist
    return dist_sq <= defect_params['radius'] ** 2 * 1.02


def partition_defect_zone_curved(p_outer, p_core, defect_params, radius=None,
                                 sector_angle_deg=None):
    """Partition curved panel at defect boundary using Y-planes and theta-planes."""
    if radius is None:
        radius = FAIRING_RADIUS

    yc = defect_params['y_center']
    theta_deg = defect_params['theta_deg']
    r_def = defect_params['radius']
    r_outer = radius + CORE_T
    margin = 2.0

    # Angular extent of defect
    d_theta_rad = r_def / r_outer
    theta_rad = math.radians(theta_deg)
    t1 = theta_rad - d_theta_rad
    t2 = theta_rad + d_theta_rad

    for part in [p_outer, p_core]:
        is_solid = hasattr(part, 'cells') and len(part.cells) > 0

        # Y cuts (axial bounding planes)
        for offset in [yc - r_def - margin, yc + r_def + margin]:
            if offset < 0 or offset > CURVED_PANEL_HEIGHT:
                continue
            dp = part.DatumPlaneByPrincipalPlane(
                principalPlane=XZPLANE, offset=offset)
            try:
                if is_solid:
                    part.PartitionCellByDatumPlane(
                        datumPlane=part.datums[dp.id], cells=part.cells)
                else:
                    part.PartitionFaceByDatumPlane(
                        datumPlane=part.datums[dp.id], faces=part.faces)
            except Exception as e:
                print("  Defect Y-partition at %.1f: %s" % (offset, str(e)[:60]))

        # Theta cuts (radial planes through Y-axis)
        for theta in [t1, t2]:
            dp = part.DatumPlaneByThreePoints(
                point1=(0, 0, 0), point2=(0, 100, 0),
                point3=(math.cos(theta), 0, math.sin(theta)))
            try:
                if is_solid:
                    part.PartitionCellByDatumPlane(
                        datumPlane=part.datums[dp.id], cells=part.cells)
                else:
                    part.PartitionFaceByDatumPlane(
                        datumPlane=part.datums[dp.id], faces=part.faces)
            except Exception as e:
                print("  Defect theta-partition: %s" % str(e)[:60])

    print("Defect partitioned (curved): y=%.0f theta=%.1f r=%.0f" % (
        yc, theta_deg, r_def))


def _tie_curved_with_defect(model, assembly, inst_core, inst_outer,
                            radius, sector_angle_deg, defect_params):
    """Create Tie for curved panel with debonding defect."""
    r_outer = radius + CORE_T

    core_top_healthy_pts = []
    core_top_defect_pts = []
    for f in inst_core.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0] ** 2 + pt[2] ** 2)
        if abs(r - r_outer) < 1.0:
            theta = math.atan2(pt[2], pt[0])
            if _point_in_defect_curved(pt[1], theta, defect_params, radius):
                core_top_defect_pts.append((pt,))
            else:
                core_top_healthy_pts.append((pt,))

    outer_healthy_pts = []
    outer_defect_pts = []
    for f in inst_outer.faces:
        pt = f.pointOn[0]
        theta = math.atan2(pt[2], pt[0])
        if _point_in_defect_curved(pt[1], theta, defect_params, radius):
            outer_defect_pts.append((pt,))
        else:
            outer_healthy_pts.append((pt,))

    if core_top_healthy_pts and outer_healthy_pts:
        core_h_seq = inst_core.faces.findAt(*core_top_healthy_pts)
        outer_h_seq = inst_outer.faces.findAt(*outer_healthy_pts)
        surf_core_h = assembly.Surface(
            side1Faces=core_h_seq, name='Surf-Core-Outer-Healthy')
        surf_outer_h = assembly.Surface(
            side2Faces=outer_h_seq, name='Surf-OuterSkin-Healthy')
        model.Tie(name='Tie-Core-OuterSkin',
                  main=surf_core_h, secondary=surf_outer_h,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie 2: Core outer <-> OuterSkin HEALTHY (%d + %d faces)" % (
            len(core_top_healthy_pts), len(outer_healthy_pts)))
    else:
        print("WARNING: No healthy faces for curved Tie 2")

    n_core_def = len(core_top_defect_pts)
    n_outer_def = len(outer_defect_pts)
    print("Debonding (curved): %d core + %d outer faces UNTIED" % (
        n_core_def, n_outer_def))


def generate_mesh(assembly, inst_inner, inst_core, inst_outer, mesh_seed,
                  geometry='flat'):
    """Generate mesh for all parts.

    Shells: S4R (EXPLICIT library)
    Core:   C3D8R (EXPLICIT, hourglass control)
    """
    # Global seed
    assembly.seedPartInstance(
        regions=(inst_inner, inst_core, inst_outer),
        size=mesh_seed, deviationFactor=0.1)

    # Core mesh strategy
    if geometry == 'flat':
        # Flat: structured hex (reliable for rectangular cross-section)
        assembly.setMeshControls(regions=inst_core.cells,
                                 elemShape=HEX, technique=STRUCTURED)
    else:
        # Curved: try SWEEP first (works for revolved bodies), fallback to FREE
        try:
            assembly.setMeshControls(regions=inst_core.cells,
                                     elemShape=HEX, technique=SWEEP)
        except Exception:
            print("  SWEEP failed for curved core, using FREE tet")
            assembly.setMeshControls(regions=inst_core.cells,
                                     elemShape=TET, technique=FREE)

    assembly.setElementType(
        regions=(inst_core.cells,),
        elemTypes=(
            ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
                     hourglassControl=DEFAULT),
            ElemType(elemCode=C3D6, elemLibrary=EXPLICIT),))

    # Skins: S4R
    for inst in [inst_inner, inst_outer]:
        assembly.setElementType(
            regions=(inst.faces,),
            elemTypes=(
                ElemType(elemCode=S4R, elemLibrary=EXPLICIT),
                ElemType(elemCode=S3, elemLibrary=EXPLICIT),))

    # Generate mesh
    assembly.generateMesh(regions=(inst_inner, inst_core, inst_outer))

    n_inner = len(inst_inner.nodes)
    n_core = len(inst_core.nodes)
    n_outer = len(inst_outer.nodes)
    print("Mesh: seed=%.1f mm (%s)" % (mesh_seed, geometry))
    print("  InnerSkin: %d nodes, %d elems" % (n_inner, len(inst_inner.elements)))
    print("  Core:      %d nodes, %d elems" % (n_core, len(inst_core.elements)))
    print("  OuterSkin: %d nodes, %d elems" % (n_outer, len(inst_outer.elements)))
    print("  Total:     %d nodes" % (n_inner + n_core + n_outer))


# ==============================================================================
# EXPLICIT STEP
# ==============================================================================

def create_explicit_step(model, time_period):
    """Create Abaqus/Explicit dynamic step (no mass scaling)."""
    model.ExplicitDynamicsStep(
        name='Step-Wave',
        previous='Initial',
        timePeriod=time_period)

    # Field output: U, V, A, S at regular intervals
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('U', 'V', 'A', 'S'),
        timeInterval=FIELD_OUTPUT_INTERVAL)

    print("Explicit step: T=%.3e s" % time_period)
    return 'Step-Wave'


# ==============================================================================
# TONE BURST AMPLITUDE
# ==============================================================================

def generate_tone_burst_amplitude(model, freq_hz, n_cycles,
                                  amp_name='Amp-ToneBurst'):
    """Generate Hanning-windowed tone burst amplitude table.

    A(t) = 0.5 * (1 - cos(2*pi*t/T_burst)) * sin(2*pi*f*t)  for 0 <= t <= T_burst
    A(t) = 0                                                   for t > T_burst

    Uses 20 points per cycle for smooth representation.
    """
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

    # Zero after burst ends
    amp_data.append((T_burst + dt, 0.0))

    model.TabularAmplitude(name=amp_name, timeSpan=STEP,
                           smooth=SOLVER_DEFAULT,
                           data=tuple(amp_data))

    print("Tone burst: f=%.0f Hz, %d cycles, T_burst=%.3e s, %d points" % (
        freq_hz, n_cycles, T_burst, len(amp_data)))

    return amp_name


# ==============================================================================
# EXCITATION & SENSORS
# ==============================================================================

def find_nearest_node(instance, target_x, target_y, target_z):
    """Find mesh node closest to target coordinates.

    Returns: (node_label, distance)
    """
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


def apply_excitation(model, assembly, inst_outer, freq_hz, n_cycles, step_name,
                     geometry='flat', sector_angle_deg=None, radius=None):
    """Apply concentrated force at panel center on outer skin.

    Flat: Force in Z-direction (out-of-plane) excites A0 mode.
    Curved: Force in radial direction (out-of-plane for cylinder).
    """
    amp_name = generate_tone_burst_amplitude(model, freq_hz, n_cycles)

    if geometry == 'flat':
        center_label, center_dist = find_nearest_node(
            inst_outer, 0.0, 0.0, CORE_T)
    else:
        # Curved: center at mid-height, mid-angle on outer skin
        if radius is None:
            radius = FAIRING_RADIUS
        if sector_angle_deg is None:
            sector_angle_deg = 10.0
        r_outer = radius + CORE_T
        theta_mid = math.radians(sector_angle_deg / 2.0)
        y_mid = CURVED_PANEL_HEIGHT / 2.0
        cx = r_outer * math.cos(theta_mid)
        cy = y_mid
        cz = r_outer * math.sin(theta_mid)
        center_label, center_dist = find_nearest_node(
            inst_outer, cx, cy, cz)

    if center_label is None:
        print("ERROR: Could not find center node on outer skin")
        return

    print("Excitation node: label=%d, snap_dist=%.3f mm" % (
        center_label, center_dist))

    node_seq = inst_outer.nodes.sequenceFromLabels((center_label,))
    excite_set = assembly.Set(nodes=node_seq, name='Set-Excitation')

    if geometry == 'flat':
        model.ConcentratedForce(
            name='Force-ToneBurst',
            createStepName=step_name,
            region=excite_set,
            cf3=FORCE_MAGNITUDE,
            amplitude=amp_name)
        print("Concentrated force: %.1f N in Z, amplitude=%s" % (
            FORCE_MAGNITUDE, amp_name))
    else:
        # Curved: apply radial force using cylindrical CSYS
        # Create a cylindrical CSYS at the assembly level for the load
        cyl_load_cs = assembly.DatumCsysByThreePoints(
            name='CylCS-Load', coordSysType=CYLINDRICAL,
            origin=(0.0, 0.0, 0.0),
            point1=(1.0, 0.0, 0.0),
            point2=(0.0, 1.0, 0.0))
        # cf1 = radial (outward), which is the "out-of-plane" direction
        model.ConcentratedForce(
            name='Force-ToneBurst',
            createStepName=step_name,
            region=excite_set,
            cf1=FORCE_MAGNITUDE,
            amplitude=amp_name,
            localCsys=assembly.datums[cyl_load_cs.id])
        print("Concentrated force: %.1f N radial (curved), amplitude=%s" % (
            FORCE_MAGNITUDE, amp_name))


def setup_sensor_outputs(model, assembly, inst_outer, step_name,
                         geometry='flat', sector_angle_deg=None, radius=None):
    """Create history output at sensor locations on outer skin.

    Flat: sensors along X-axis at y=0.
    Curved: sensors along circumferential arc at mid-height.
    """
    sensor_info = []

    if geometry == 'flat':
        for i, offset in enumerate(SENSOR_OFFSETS):
            label, dist = find_nearest_node(inst_outer, offset, 0.0, CORE_T)
            if label is None:
                print("WARNING: Sensor %d at offset=%.0f mm: no node" % (
                    i, offset))
                continue
            sensor_info.append((i, offset, label, dist))
    else:
        # Curved: place sensors along circumferential arc
        if radius is None:
            radius = FAIRING_RADIUS
        if sector_angle_deg is None:
            sector_angle_deg = 10.0
        r_outer = radius + CORE_T
        theta_mid = math.radians(sector_angle_deg / 2.0)
        y_mid = CURVED_PANEL_HEIGHT / 2.0

        for i, arc_offset in enumerate(SENSOR_OFFSETS):
            # Convert arc offset to angular offset from center
            d_theta = arc_offset / r_outer
            theta = theta_mid + d_theta
            # Don't exceed sector boundary
            if theta < 0 or theta > math.radians(sector_angle_deg):
                print("WARNING: Sensor %d at arc=%.0f mm outside sector" % (
                    i, arc_offset))
                # Clamp to boundary
                theta = max(0.01, min(theta,
                                      math.radians(sector_angle_deg) - 0.01))
            sx = r_outer * math.cos(theta)
            sy = y_mid
            sz = r_outer * math.sin(theta)
            label, dist = find_nearest_node(inst_outer, sx, sy, sz)
            if label is None:
                print("WARNING: Sensor %d at arc=%.0f mm: no node" % (
                    i, arc_offset))
                continue
            sensor_info.append((i, arc_offset, label, dist))

    for i, offset, label, dist in sensor_info:
        node_seq = inst_outer.nodes.sequenceFromLabels((label,))
        set_name = 'Set-Sensor-%d' % i
        sensor_set = assembly.Set(nodes=node_seq, name=set_name)

        # For curved geometry, use U magnitude instead of U3
        # (radial displacement doesn't align with global Z)
        out_vars = ('U3', ) if geometry == 'flat' else ('U1', 'U2', 'U3')
        model.HistoryOutputRequest(
            name='H-Output-Sensor-%d' % i,
            createStepName=step_name,
            variables=out_vars,
            region=sensor_set,
            sectionPoints=DEFAULT,
            rebar=EXCLUDE,
            timeInterval=FIELD_OUTPUT_INTERVAL)

        print("Sensor %d: offset=%.0f mm, node=%d, snap=%.3f mm" % (
            i, offset, label, dist))

    print("History output: %d sensors, %s at dt=%.1e s" % (
        len(sensor_info),
        'U3' if geometry == 'flat' else 'U1,U2,U3',
        FIELD_OUTPUT_INTERVAL))
    return sensor_info


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def generate_model(job_name, freq_khz=DEFAULT_FREQ_KHZ, n_cycles=DEFAULT_CYCLES,
                   panel_size=PANEL_SIZE, mesh_seed=None, time_period=None,
                   defect_params=None, no_run=False, geometry='flat',
                   fairing_radius=None):
    """Generate guided wave model (Abaqus/Explicit).

    Args:
        geometry: 'flat' for rectangular panel, 'curved' for cylindrical section.
        defect_params: dict for debonding.
            Flat: {x_center, y_center, radius}
            Curved: {y_center, theta_deg, radius}
        fairing_radius: Inner skin radius for curved geometry (mm).
    """
    freq_hz = freq_khz * 1e3

    # Auto mesh seed: lambda/8
    if mesh_seed is None:
        wavelength = CP_ESTIMATE / freq_hz * 1000.0  # mm
        mesh_seed = min(wavelength / 8.0, DEFAULT_MESH_SEED)

    # Auto time period: 2.5x diagonal traversal
    if time_period is None:
        diag = panel_size * math.sqrt(2.0)
        t_traverse = diag / (CP_ESTIMATE * 1000.0)  # s (cp in m/s -> mm/s)
        time_period = max(t_traverse * 2.5, DEFAULT_TIME_PERIOD)

    sector_angle_deg = None
    radius = fairing_radius or FAIRING_RADIUS

    print("=" * 70)
    print("Guided Wave Model: %s (%s)" % (job_name, geometry.upper()))
    if geometry == 'flat':
        print("  Panel: %.0f x %.0f mm, CFRP/Al-HC/CFRP (t=%.0f/%.0f/%.0f mm)"
              % (panel_size, panel_size, FACE_T, CORE_T, FACE_T))
    else:
        r_outer = radius + CORE_T
        sector_angle_deg_est = panel_size / r_outer * 180.0 / math.pi * 1.1
        print("  Curved: R=%.0f mm, arc~%.0f mm, height=%.0f mm" % (
            radius, panel_size, CURVED_PANEL_HEIGHT))
        print("  Sandwich: CFRP/Al-HC/CFRP (t=%.0f/%.0f/%.0f mm)" % (
            FACE_T, CORE_T, FACE_T))
    print("  Excitation: %d-cycle Hanning tone burst at %.0f kHz" % (
        n_cycles, freq_khz))
    print("  Mesh seed: %.2f mm (lambda=%.1f mm)" % (
        mesh_seed, CP_ESTIMATE / freq_hz * 1000.0))
    print("  Time: %.3e s (%.3f ms)" % (time_period, time_period * 1e3))
    if defect_params:
        dtype = defect_params.get('defect_type', 'debonding')
        if geometry == 'flat':
            print("  Defect: %s at (%.0f, %.0f) r=%.0f mm" % (
                dtype.upper(),
                defect_params['x_center'], defect_params['y_center'],
                defect_params['radius']))
        else:
            print("  Defect: %s at y=%.0f theta=%.1f deg r=%.0f mm" % (
                dtype.upper(),
                defect_params['y_center'], defect_params['theta_deg'],
                defect_params['radius']))
    else:
        print("  Defect: NONE (healthy)")
    print("=" * 70)

    Mdb()
    model = mdb.models['Model-1']

    # 1. Materials & Sections
    create_materials(model)
    create_sections(model)
    if defect_params and defect_params.get('defect_type') in ('fod', 'impact', 'delamination'):
        create_defect_materials_gw(model, defect_params)
        create_defect_sections_gw(model, defect_params)

    if geometry == 'flat':
        # 2. Parts
        p_inner, p_core, p_outer = create_flat_panel_parts(model, panel_size)

        # 3. Partition core
        partition_core_thickness(p_core)
        if defect_params:
            partition_defect_zone(p_outer, p_core, defect_params)

        # 4. Assign sections
        assign_sections(p_inner, p_core, p_outer)
        if defect_params and defect_params.get('defect_type') in ('fod', 'impact', 'delamination'):
            assign_defect_sections_flat(p_core, p_outer, defect_params)

        # 5. Assembly
        a, inst_inner, inst_core, inst_outer = create_assembly(
            model, p_inner, p_core, p_outer, geometry='flat')

        # 6. Tie
        create_tie_constraints(model, a, inst_inner, inst_core, inst_outer,
                               defect_params)

        # 7. Mesh
        generate_mesh(a, inst_inner, inst_core, inst_outer, mesh_seed,
                      geometry='flat')

    else:  # curved
        # 2. Parts (revolution-based)
        p_inner, p_core, p_outer, sector_angle_deg = \
            create_curved_panel_parts(model, panel_size, radius=radius)

        # 3. Core partition (curved uses mesh seed, no explicit partition)
        partition_core_thickness_curved(p_core, radius=radius)
        if defect_params:
            partition_defect_zone_curved(p_outer, p_core, defect_params,
                                        radius=radius,
                                        sector_angle_deg=sector_angle_deg)

        # 4. Assign sections (cylindrical CSYS)
        assign_sections_curved(p_inner, p_core, p_outer)

        # 5. Assembly (no translation needed for revolved parts)
        a, inst_inner, inst_core, inst_outer = create_assembly(
            model, p_inner, p_core, p_outer, geometry='curved')

        # 6. Tie (sample-point method for curved)
        create_tie_constraints_curved(
            model, a, inst_inner, inst_core, inst_outer,
            sector_angle_deg, radius=radius,
            defect_params=defect_params)

        # 7. Mesh
        generate_mesh(a, inst_inner, inst_core, inst_outer, mesh_seed,
                      geometry='curved')

    # 8. Explicit step
    step_name = create_explicit_step(model, time_period)

    # 9. Excitation
    apply_excitation(model, a, inst_outer, freq_hz, n_cycles, step_name,
                     geometry=geometry, sector_angle_deg=sector_angle_deg,
                     radius=radius)

    # 10. Sensor outputs
    setup_sensor_outputs(model, a, inst_outer, step_name,
                         geometry=geometry, sector_angle_deg=sector_angle_deg,
                         radius=radius)

    # 11. Job
    mdb.Job(name=job_name, model='Model-1', type=ANALYSIS, resultsFormat=ODB,
            numCpus=4, numDomains=4, multiprocessingMode=DEFAULT,
            explicitPrecision=SINGLE, nodalOutputPrecision=FULL)

    mdb.saveAs(pathName=job_name + '.cae')

    print("Writing INP: %s.inp" % job_name)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)

    if no_run:
        print("INP written. Skipping run (--no_run)")
        return

    print("Submitting Explicit job: %s ..." % job_name)
    import subprocess
    inp_path = os.path.abspath(job_name + '.inp')
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
        description='Guided wave model (Abaqus/Explicit) — flat or curved panel')
    parser.add_argument('--job', type=str, default='Job-GuidedWave',
                        help='Job name (default: Job-GuidedWave)')
    parser.add_argument('--geometry', type=str, default='flat',
                        choices=['flat', 'curved'],
                        help='Panel geometry: flat or curved (default: flat)')
    parser.add_argument('--freq', type=float, default=DEFAULT_FREQ_KHZ,
                        help='Center frequency in kHz (default: 50)')
    parser.add_argument('--cycles', type=int, default=DEFAULT_CYCLES,
                        help='Number of burst cycles (default: 5)')
    parser.add_argument('--panel_size', type=float, default=PANEL_SIZE,
                        help='Panel size in mm: side(flat) or arc(curved) (default: 300)')
    parser.add_argument('--radius', type=float, default=None,
                        help='Fairing inner radius in mm (curved only, default: 2600)')
    parser.add_argument('--mesh_seed', type=float, default=None,
                        help='Mesh seed in mm (auto: lambda/8)')
    parser.add_argument('--time', type=float, default=None,
                        help='Analysis time period in seconds (auto)')
    parser.add_argument('--defect', type=str, default=None,
                        help='Defect JSON. Flat: {"x_center":80,"y_center":0,"radius":25}. '
                             'Add "defect_type":"fod"|"impact"|"delamination"|"inner_debond" for other defects. '
                             'Curved: {"y_center":150,"theta_deg":5,"radius":25}')
    parser.add_argument('--no_run', action='store_true',
                        help='Write INP only, do not run')

    args, _ = parser.parse_known_args()

    defect_data = None
    if args.defect:
        if os.path.exists(args.defect):
            with open(args.defect, 'r') as f:
                defect_data = json.load(f)
        else:
            try:
                defect_data = json.loads(args.defect)
            except (ValueError, TypeError):
                print("Invalid defect JSON: %s" % args.defect)

    generate_model(
        job_name=args.job,
        freq_khz=args.freq,
        n_cycles=args.cycles,
        panel_size=args.panel_size,
        mesh_seed=args.mesh_seed,
        time_period=args.time,
        defect_params=defect_data,
        no_run=args.no_run,
        geometry=args.geometry,
        fairing_radius=args.radius)
