# -*- coding: utf-8 -*-
"""
Fairing Separation Dynamics Model — Abaqus/Explicit

Generates a quarter (90-deg) sector of H3 fairing with separation mechanism:
  - Two 90-degree quarter-shells meeting at a vertical seam
  - Horizontal separation: frangible bolts (beam elements, MODEL CHANGE)
  - Vertical separation: pyro-cord beams along the seam
  - Opening springs with preload at bottom
  - Symmetry BCs at free circumferential edges
  - Abaqus/Explicit dynamic analysis

Symmetry rationale:
  Full fairing = 2 half-shells, each symmetric about its own midplane.
  We model one half-shell split into two 90-deg quarters at the seam.
  → 1/4 of full model, with symmetry BCs at theta=0 and theta=180.
  This captures the seam separation + bolt release physics at 1/4 cost.

Based on:
  - JAXA H3-8 investigation (2025/12): 6-7Hz anomalous vibration at separation
  - KTH thesis: FE modeling of fairing separation with beam element removal
  - Kawasaki: H3 frangible bolt (V-notch) mechanism

Usage:
  abaqus cae noGUI=src/generate_fairing_separation.py -- --job Sep-Normal --no_run
  abaqus cae noGUI=src/generate_fairing_separation.py -- --job Sep-Abnormal \\
      --n_stuck_bolts 3 --spring_stiffness 300 --no_run
"""

from __future__ import print_function
import sys
import os
import math
import json

from abaqus import *
from abaqusConstants import *
from caeModules import *
from section import SectionLayer
import mesh
import regionToolset

# ==============================================================================
# CONSTANTS — Geometry (H3 short fairing barrel)
# ==============================================================================
RADIUS = 2600.0           # mm, inner skin radius
CORE_T = 38.0             # mm, honeycomb core thickness
FACE_T = 1.0              # mm, CFRP skin thickness (each)
BARREL_Z_MIN = 0.0        # mm, fairing bottom (adapter interface)
BARREL_Z_MAX = 5000.0     # mm, barrel top
FAIRING_DIAMETER = 5200.0 # mm

# Adapter ring
ADAPTER_HEIGHT = 100.0    # mm
ADAPTER_T = 10.0          # mm, shell thickness

# PSS (Payload Support Structure) — CFRP/Al-HC sandwich cylinder
PSS_RADIUS = 833.0        # mm, inner radius (PAF-1666SC)
PSS_HEIGHT = 600.0        # mm, cylinder height
PSS_CORE_T = 25.0         # mm, honeycomb core thickness
PSS_FACE_T = 1.0          # mm, CFRP skin thickness (each)
PSS_Z_BOTTOM = 200.0      # mm, bottom of PSS above adapter base

# ==============================================================================
# CONSTANTS — Materials (same as generate_gw_fairing.py)
# ==============================================================================
# CFRP T1000G/Epoxy
CFRP_E1 = 160000.0       # MPa
CFRP_E2 = 10000.0
CFRP_NU12 = 0.3
CFRP_G12 = 5000.0
CFRP_G13 = 5000.0
CFRP_G23 = 3000.0
CFRP_DENSITY = 1600e-12  # tonne/mm^3

# Al-Honeycomb 5052
E_CORE_1 = 1000.0        # MPa (radial)
E_CORE_2 = 10.0          # theta
E_CORE_3 = 10.0          # axial
NU_CORE_12 = 0.001
NU_CORE_13 = 0.001
NU_CORE_23 = 0.001
G_CORE_12 = 400.0
G_CORE_13 = 240.0
G_CORE_23 = 5.0
CORE_DENSITY = 50e-12

# Al-7075 (adapter ring, ring frames)
AL_E = 71700.0
AL_NU = 0.33
AL_DENSITY = 2810e-12

# Steel (bolt material — for reference, used in connector section)
BOLT_E = 200000.0
BOLT_NU = 0.3
BOLT_DENSITY = 7850e-12

# Layup
PLY_ANGLES = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
PLY_T = FACE_T / 8.0

# ==============================================================================
# CONSTANTS — Separation mechanism
# ==============================================================================
N_BOLTS_FULL = 72         # total frangible bolts (full 360)
N_BOLTS_QUARTER = 18      # bolts in 90-deg sector
BOLT_SPACING_DEG = 360.0 / N_BOLTS_FULL  # 5 degrees apart
DEFAULT_SECTOR_ANGLE = 90.0  # degrees (quarter model)

# Vertical pyro-cord
PYRO_SPACING = 200.0      # mm, connector spacing along vertical seam

# Opening spring
N_SPRINGS_PER_HALF = 4    # springs per half-shell
SPRING_STIFFNESS = 500.0  # N/mm, linear spring stiffness (default)
SPRING_PRELOAD = 5000.0   # N, precompression force per spring

# ==============================================================================
# CONSTANTS — Analysis
# ==============================================================================
T_PRELOAD = 0.005         # s, quasi-static preload
T_SEPARATION = 0.200      # s, separation dynamics (enough for ~20deg opening)
MESH_SEED_DEFAULT = 50.0  # mm, coarse mesh for separation dynamics
GRAVITY = 29430.0         # mm/s^2 (3g axial, launch condition)

# Mass scaling for preload step
DT_PRELOAD_SCALE = 1e-4   # target stable time increment for preload


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _bolt_positions(n_bolts):
    """Return list of (theta_deg, theta_rad) for bolt positions."""
    positions = []
    for i in range(n_bolts):
        theta_deg = i * 360.0 / n_bolts
        theta_rad = math.radians(theta_deg)
        positions.append((theta_deg, theta_rad))
    return positions


def _pyro_positions(z_min, z_max, spacing):
    """Return list of z positions for pyro-cord connectors."""
    positions = []
    z = z_min + spacing / 2.0
    while z < z_max:
        positions.append(z)
        z += spacing
    return positions


def _node_at_angle_z(instance, theta_rad, z_target, r_target, tol=None):
    """Find the node closest to (r*cos(theta), z, r*sin(theta))."""
    x_t = r_target * math.cos(theta_rad)
    y_t = z_target  # Abaqus Y = axial
    z_t = r_target * math.sin(theta_rad)

    best_node = None
    best_dist = 1e30
    for node in instance.nodes:
        coords = node.coordinates
        dx = coords[0] - x_t
        dy = coords[1] - y_t
        dz = coords[2] - z_t
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < best_dist:
            best_dist = dist
            best_node = node
    if tol is not None and best_dist > tol:
        return None
    return best_node


# ==============================================================================
# MATERIAL & SECTION CREATION
# ==============================================================================

def create_materials(model):
    """Create all materials for separation model."""
    # CFRP skin
    mat = model.Material(name='Mat-CFRP')
    mat.Density(table=((CFRP_DENSITY,),))
    mat.Elastic(
        type=ENGINEERING_CONSTANTS,
        table=((CFRP_E1, CFRP_E2, CFRP_E2,
                CFRP_NU12, CFRP_NU12, 0.3,
                CFRP_G12, CFRP_G13, CFRP_G23),))

    # Honeycomb core — isotropic approximation for separation dynamics
    # (Anisotropic ENGINEERING_CONSTANTS requires MaterialOrientation on each
    #  core part; we simplify here since separation kinematics dominate.)
    mat_core = model.Material(name='Mat-Honeycomb')
    mat_core.Density(table=((CORE_DENSITY,),))
    mat_core.Elastic(table=((E_CORE_1, NU_CORE_12),))

    # Aluminum (adapter, frames)
    mat_al = model.Material(name='Mat-Al7075')
    mat_al.Density(table=((AL_DENSITY,),))
    mat_al.Elastic(table=((AL_E, AL_NU),))

    # Steel (for bolt connector reference)
    mat_bolt = model.Material(name='Mat-Steel')
    mat_bolt.Density(table=((BOLT_DENSITY,),))
    mat_bolt.Elastic(table=((BOLT_E, BOLT_NU),))

    # Structural damping (Rayleigh) — higher than GW model
    for m in [mat, mat_core, mat_al]:
        m.Damping(alpha=100.0, beta=0.0)


def create_sections(model):
    """Create composite shell + solid sections."""
    # Inner skin — composite layup (assigned per part later via assign_sections)
    # Outer skin — composite layup (assigned per part later via assign_sections)
    # Core — solid homogeneous

    model.HomogeneousSolidSection(
        name='Sec-Core', material='Mat-Honeycomb', thickness=None)

    model.HomogeneousShellSection(
        name='Sec-Adapter', material='Mat-Al7075',
        thickness=ADAPTER_T)

    # CFRP shell section (fallback for simple assignment)
    model.HomogeneousShellSection(
        name='Sec-CFRP-Skin', material='Mat-CFRP',
        thickness=FACE_T)


def assign_sections(model, parts_Q1, parts_Q2, p_adapter, p_pss=None):
    """Assign section properties to all parts.

    Must be called AFTER geometry creation and BEFORE meshing/ties.
    Shell skins get CFRP shell section, cores get solid section,
    adapter gets Al shell section.
    """
    # Quarter shell skins and cores
    for qid, parts in [('Q1', parts_Q1), ('Q2', parts_Q2)]:
        p_inner, p_core, p_outer = parts

        # Inner skin — CFRP shell section
        region_inner = p_inner.Set(
            name='Set-All', faces=p_inner.faces)
        p_inner.SectionAssignment(
            region=region_inner, sectionName='Sec-CFRP-Skin',
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)

        # Outer skin — CFRP shell section
        region_outer = p_outer.Set(
            name='Set-All', faces=p_outer.faces)
        p_outer.SectionAssignment(
            region=region_outer, sectionName='Sec-CFRP-Skin',
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)

        # Core — solid section
        region_core = p_core.Set(
            name='Set-All', cells=p_core.cells)
        p_core.SectionAssignment(
            region=region_core, sectionName='Sec-Core',
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)

    # Adapter ring
    region_adapter = p_adapter.Set(
        name='Set-All', faces=p_adapter.faces)
    p_adapter.SectionAssignment(
        region=region_adapter, sectionName='Sec-Adapter',
        offset=0.0, offsetType=MIDDLE_SURFACE,
        offsetField='', thicknessAssignment=FROM_SECTION)

    # PSS (if included)
    if p_pss is not None:
        p_pss_inner, p_pss_core, p_pss_outer = p_pss

        region_pi = p_pss_inner.Set(
            name='Set-All', faces=p_pss_inner.faces)
        p_pss_inner.SectionAssignment(
            region=region_pi, sectionName='Sec-CFRP-Skin',
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)

        region_po = p_pss_outer.Set(
            name='Set-All', faces=p_pss_outer.faces)
        p_pss_outer.SectionAssignment(
            region=region_po, sectionName='Sec-CFRP-Skin',
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)

        region_pc = p_pss_core.Set(
            name='Set-All', cells=p_pss_core.cells)
        p_pss_core.SectionAssignment(
            region=region_pc, sectionName='Sec-Core',
            offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)


def _create_composite_layup(part, section_name, region):
    """Create composite shell section with [45/0/-45/90]s layup."""
    layup_data = []
    for i, angle in enumerate(PLY_ANGLES):
        layup_data.append(
            SectionLayer(
                material='Mat-CFRP', thickness=PLY_T,
                orientAngle=angle,
                plyName='Ply-%d' % (i + 1)))

    part.CompositeLayup(
        name=section_name,
        elementType=SHELL,
        offsetType=MIDDLE_SURFACE,
        symmetric=False,
        thicknessAssignment=FROM_SECTION)

    layup = part.compositeLayups[section_name]
    layup.CompositePly(
        plyTable=tuple(layup_data),
        region=region,
        material='Mat-CFRP',
        thicknessType=SPECIFY_THICKNESS,
        thickness=PLY_T,
        orientationType=SPECIFY_ORIENT,
        orientationValue=0.0,
        numIntPoints=3)


# ==============================================================================
# GEOMETRY CREATION
# ==============================================================================

def create_quarter_shell(model, quarter_id, sweep_angle_deg, z_min, z_max):
    """Create one quarter-fairing (inner skin + core + outer skin).

    Each quarter is a 90-deg sector. Two quarters (Q1, Q2) share a seam
    at their common edge. Symmetry BCs go on the free edges.

    Args:
        quarter_id: 'Q1' or 'Q2'
        sweep_angle_deg: sector angle (default 90)
        z_min, z_max: axial range

    Returns:
        (part_inner, part_core, part_outer) tuple
    """
    sweep_angle = sweep_angle_deg

    r_inner = RADIUS
    r_core_inner = RADIUS + FACE_T
    r_core_outer = RADIUS + FACE_T + CORE_T
    r_outer = r_core_outer

    # --- Inner skin (shell) ---
    name_inner = '%s-InnerSkin' % quarter_id
    s1 = model.ConstrainedSketch(
        name='sk-inner-%s' % quarter_id, sheetSize=20000.0)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s1.Line(point1=(r_inner, z_min), point2=(r_inner, z_max))

    p_inner = model.Part(name=name_inner, dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=sweep_angle,
                             flipRevolveDirection=OFF)
    s1.unsetPrimaryObject()

    # --- Core (solid) ---
    name_core = '%s-Core' % quarter_id
    s2 = model.ConstrainedSketch(
        name='sk-core-%s' % quarter_id, sheetSize=20000.0)
    s2.setPrimaryObject(option=STANDALONE)
    s2.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s2.Line(point1=(r_core_inner, z_min), point2=(r_core_inner, z_max))
    s2.Line(point1=(r_core_inner, z_max), point2=(r_core_outer, z_max))
    s2.Line(point1=(r_core_outer, z_max), point2=(r_core_outer, z_min))
    s2.Line(point1=(r_core_outer, z_min), point2=(r_core_inner, z_min))

    p_core = model.Part(name=name_core, dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=s2, angle=sweep_angle,
                            flipRevolveDirection=OFF)
    s2.unsetPrimaryObject()

    # --- Outer skin (shell) ---
    name_outer = '%s-OuterSkin' % quarter_id
    s3 = model.ConstrainedSketch(
        name='sk-outer-%s' % quarter_id, sheetSize=20000.0)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s3.Line(point1=(r_core_outer, z_min), point2=(r_core_outer, z_max))

    p_outer = model.Part(name=name_outer, dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=sweep_angle,
                             flipRevolveDirection=OFF)
    s3.unsetPrimaryObject()

    return p_inner, p_core, p_outer


def create_adapter_ring(model, z_bottom=0.0, sweep_angle=180.0):
    """Create cylindrical adapter ring covering both quarter sectors.

    For a 90-deg quarter model, the adapter covers 180 degrees
    (both Q1 and Q2 combined range).
    """
    s = model.ConstrainedSketch(name='sk-adapter', sheetSize=20000.0)
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, -500.0), point2=(0.0, 500.0))
    r_mid = RADIUS + FACE_T + CORE_T / 2.0
    s.Line(point1=(r_mid, z_bottom - ADAPTER_HEIGHT),
           point2=(r_mid, z_bottom))

    p_adapter = model.Part(name='Adapter', dimensionality=THREE_D,
                           type=DEFORMABLE_BODY)
    p_adapter.BaseShellRevolve(sketch=s, angle=sweep_angle,
                               flipRevolveDirection=OFF)
    s.unsetPrimaryObject()
    return p_adapter


def create_pss(model, sweep_angle=360.0):
    """Create PSS (Payload Support Structure) as sandwich cylinder.

    The PSS carries the satellite. In the F8 accident, pre-existing
    debonding propagated under separation shock → structural failure
    within 62 ms.

    Returns:
        (p_inner, p_core, p_outer) Part objects
    """
    r_inner = PSS_RADIUS
    r_core_inner = r_inner + PSS_FACE_T
    r_core_outer = r_core_inner + PSS_CORE_T
    z_min = PSS_Z_BOTTOM
    z_max = PSS_Z_BOTTOM + PSS_HEIGHT

    # Inner skin (shell)
    s1 = model.ConstrainedSketch(name='sk-pss-inner', sheetSize=20000.0)
    s1.setPrimaryObject(option=STANDALONE)
    s1.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s1.Line(point1=(r_inner, z_min), point2=(r_inner, z_max))
    p_inner = model.Part(name='PSS-InnerSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShellRevolve(sketch=s1, angle=sweep_angle,
                             flipRevolveDirection=OFF)
    s1.unsetPrimaryObject()

    # Core (solid)
    s2 = model.ConstrainedSketch(name='sk-pss-core', sheetSize=20000.0)
    s2.setPrimaryObject(option=STANDALONE)
    s2.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s2.Line(point1=(r_core_inner, z_min), point2=(r_core_inner, z_max))
    s2.Line(point1=(r_core_inner, z_max), point2=(r_core_outer, z_max))
    s2.Line(point1=(r_core_outer, z_max), point2=(r_core_outer, z_min))
    s2.Line(point1=(r_core_outer, z_min), point2=(r_core_inner, z_min))
    p_core = model.Part(name='PSS-Core', dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=s2, angle=sweep_angle,
                            flipRevolveDirection=OFF)
    s2.unsetPrimaryObject()

    # Outer skin (shell)
    s3 = model.ConstrainedSketch(name='sk-pss-outer', sheetSize=20000.0)
    s3.setPrimaryObject(option=STANDALONE)
    s3.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, z_max + 100.0))
    s3.Line(point1=(r_core_outer, z_min), point2=(r_core_outer, z_max))
    p_outer = model.Part(name='PSS-OuterSkin', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_outer.BaseShellRevolve(sketch=s3, angle=sweep_angle,
                             flipRevolveDirection=OFF)
    s3.unsetPrimaryObject()

    return p_inner, p_core, p_outer


# ==============================================================================
# ASSEMBLY
# ==============================================================================

def create_assembly(model, parts_Q1, parts_Q2, p_adapter, p_pss=None,
                    sector_angle=90.0):
    """Assemble two quarter-shells, adapter ring, and optionally PSS.

    Quarter Q1: theta = [0, sector_angle] (default orientation)
    Quarter Q2: theta = [sector_angle, 2*sector_angle] (rotated)

    The seam (separation line) is at theta = sector_angle (90 deg).
    Symmetry BCs will be applied at theta=0 (Q1 free edge) and
    theta=2*sector_angle (Q2 free edge).

    Returns:
        assembly, dict of instances
    """
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)

    instances = {}

    # Q1 (0 to sector_angle) — default orientation
    for i, suffix in enumerate(['InnerSkin', 'Core', 'OuterSkin']):
        name = 'Q1-%s' % suffix
        inst = assembly.Instance(name=name, part=parts_Q1[i], dependent=ON)
        instances[name] = inst

    # Q2 (sector_angle to 2*sector_angle) — rotate by sector_angle
    for i, suffix in enumerate(['InnerSkin', 'Core', 'OuterSkin']):
        name = 'Q2-%s' % suffix
        inst = assembly.Instance(name=name, part=parts_Q2[i], dependent=ON)
        assembly.rotate(instanceList=(name,),
                        axisPoint=(0.0, 0.0, 0.0),
                        axisDirection=(0.0, 1.0, 0.0),
                        angle=sector_angle)
        instances[name] = inst

    # Adapter ring (full 360 but only contacts quarter sectors)
    inst_adapter = assembly.Instance(name='Adapter', part=p_adapter,
                                     dependent=ON)
    instances['Adapter'] = inst_adapter

    # PSS (Payload Support Structure) — no rotation, sits at center
    if p_pss is not None:
        for i, suffix in enumerate(['InnerSkin', 'Core', 'OuterSkin']):
            name = 'PSS-%s' % suffix
            inst = assembly.Instance(name=name, part=p_pss[i], dependent=ON)
            instances[name] = inst

    return assembly, instances


# ==============================================================================
# TIES (skin-core bonding within each half)
# ==============================================================================

def create_skin_core_ties(model, assembly, instances):
    """Tie inner/outer skins to core for each quarter."""
    for qid in ['Q1', 'Q2']:
        inst_inner = instances['%s-InnerSkin' % qid]
        inst_core = instances['%s-Core' % qid]
        inst_outer = instances['%s-OuterSkin' % qid]

        # Inner skin to core
        surf_inner = assembly.Surface(
            name='Surf-%s-InnerSkin' % qid,
            side1Faces=inst_inner.faces)
        surf_core = assembly.Surface(
            name='Surf-%s-Core' % qid,
            side1Faces=inst_core.faces)

        model.Tie(name='Tie-%s-InnerToCore' % qid,
                  main=surf_core,
                  secondary=surf_inner,
                  positionToleranceMethod=COMPUTED,
                  adjust=ON,
                  tieRotations=ON)

        # Outer skin to core
        surf_outer = assembly.Surface(
            name='Surf-%s-OuterSkin' % qid,
            side1Faces=inst_outer.faces)

        model.Tie(name='Tie-%s-OuterToCore' % qid,
                  main=surf_core,
                  secondary=surf_outer,
                  positionToleranceMethod=COMPUTED,
                  adjust=ON,
                  tieRotations=ON)

    # PSS ties (if PSS exists)
    if 'PSS-Core' in instances:
        inst_pi = instances['PSS-InnerSkin']
        inst_pc = instances['PSS-Core']
        inst_po = instances['PSS-OuterSkin']

        surf_pi = assembly.Surface(name='Surf-PSS-InnerSkin',
                                   side1Faces=inst_pi.faces)
        surf_pc = assembly.Surface(name='Surf-PSS-Core',
                                   side1Faces=inst_pc.faces)
        model.Tie(name='Tie-PSS-InnerToCore',
                  main=surf_pc, secondary=surf_pi,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON)

        surf_po = assembly.Surface(name='Surf-PSS-OuterSkin',
                                   side1Faces=inst_po.faces)
        model.Tie(name='Tie-PSS-OuterToCore',
                  main=surf_pc, secondary=surf_po,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON)


# ==============================================================================
# MESH
# ==============================================================================

def generate_mesh(assembly, instances, mesh_seed):
    """Mesh all parts with given seed size."""
    for name, inst in instances.items():
        part = inst.part

        # Element type selection
        if 'Core' in name:
            # Solid elements for honeycomb core
            part.seedPart(size=mesh_seed, deviationFactor=0.1,
                          minSizeFactor=0.1)
            elemType_solid = mesh.ElemType(
                elemCode=C3D8R, elemLibrary=EXPLICIT,
                hourglassControl=ENHANCED)
            cells = part.cells
            if len(cells) > 0:
                part.setElementType(
                    regions=(cells,),
                    elemTypes=(elemType_solid,
                               mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT),
                               mesh.ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)))
            part.generateMesh()

        elif 'Adapter' in name:
            part.seedPart(size=mesh_seed, deviationFactor=0.1,
                          minSizeFactor=0.1)
            elemType_shell = mesh.ElemType(
                elemCode=S4R, elemLibrary=EXPLICIT,
                hourglassControl=ENHANCED)
            faces = part.faces
            if len(faces) > 0:
                part.setElementType(
                    regions=(faces,),
                    elemTypes=(elemType_shell,
                               mesh.ElemType(elemCode=S3R, elemLibrary=EXPLICIT)))
            part.generateMesh()

        else:
            # Shell elements for skins
            part.seedPart(size=mesh_seed, deviationFactor=0.1,
                          minSizeFactor=0.1)
            elemType_shell = mesh.ElemType(
                elemCode=S4R, elemLibrary=EXPLICIT,
                hourglassControl=ENHANCED)
            faces = part.faces
            if len(faces) > 0:
                part.setElementType(
                    regions=(faces,),
                    elemTypes=(elemType_shell,
                               mesh.ElemType(elemCode=S3R, elemLibrary=EXPLICIT)))
            part.generateMesh()

    assembly.regenerate()


# ==============================================================================
# CONNECTOR ELEMENTS (bolts, pyro-cord, springs)
# ==============================================================================

def create_bolt_connectors(model, assembly, instances, n_bolts,
                           stuck_bolts=None):
    """Create connector elements for frangible bolts at fairing bottom.

    Bolts connect the adapter ring to the inner skins of both halves
    at Z=0 (bottom edge).

    Args:
        stuck_bolts: list of bolt indices that will NOT be removed (abnormal)

    Returns:
        bolt_wire_names: list of wire feature names for INP post-processing
    """
    if stuck_bolts is None:
        stuck_bolts = []

    r_connect = RADIUS + FACE_T + CORE_T / 2.0  # mid-core radius
    z_connect = BARREL_Z_MIN

    # Create connector section (translational + rotational spring)
    model.ConnectorSection(
        name='ConnSec-Bolt',
        translationalType=CARTESIAN,
        rotationalType=ROTATION)

    bolt_info = []
    bolt_positions = _bolt_positions(n_bolts)

    for idx, (theta_deg, theta_rad) in enumerate(bolt_positions):
        x = r_connect * math.cos(theta_rad)
        y = z_connect
        z = r_connect * math.sin(theta_rad)

        # Point on adapter (slightly below)
        x_a = x
        y_a = z_connect - ADAPTER_HEIGHT / 2.0
        z_a = z

        wire_name = 'Wire-Bolt-%03d' % idx
        set_name = 'Set-Bolt-%03d' % idx

        # Create wire (2-point connector)
        assembly.WirePolyLine(
            mergeType=IMPRINT,
            meshable=OFF,
            points=((x, y, z), (x_a, y_a, z_a)))

        # The wire creates edges — find and assign connector section
        # Note: actual connector assignment happens via INP post-processing
        # because CAE API for connectors on assembly wires is fragile

        is_stuck = idx in stuck_bolts
        bolt_info.append({
            'idx': idx,
            'theta_deg': theta_deg,
            'x': x, 'y': y, 'z': z,
            'x_a': x_a, 'y_a': y_a, 'z_a': z_a,
            'stuck': is_stuck,
            'set_name': set_name,
        })

    return bolt_info


def create_pyro_connectors(model, assembly, z_min, z_max, spacing):
    """Create connector elements along vertical seam (theta=0 and theta=180).

    Returns:
        pyro_info: list of dicts with connector positions
    """
    r_connect = RADIUS + FACE_T + CORE_T / 2.0
    z_positions = _pyro_positions(z_min, z_max, spacing)

    pyro_info = []
    for seam_angle in [0.0, math.pi]:  # theta=0 and theta=180
        for zi, z_pos in enumerate(z_positions):
            x = r_connect * math.cos(seam_angle)
            y = z_pos
            z = r_connect * math.sin(seam_angle)

            # Small offset for second point (across seam)
            offset = 1.0  # mm
            x2 = (r_connect) * math.cos(seam_angle + 0.001)
            y2 = z_pos
            z2 = (r_connect) * math.sin(seam_angle + 0.001)

            seam_name = 'Seam0' if seam_angle < 0.1 else 'Seam180'
            set_name = 'Set-Pyro-%s-%03d' % (seam_name, zi)

            pyro_info.append({
                'seam': seam_name,
                'z': z_pos,
                'x1': x, 'y1': y, 'z1': z,
                'x2': x2, 'y2': y2, 'z2': z2,
                'set_name': set_name,
            })

    return pyro_info


# ==============================================================================
# STEPS
# ==============================================================================

def create_steps(model, t_preload, t_separation):
    """Create analysis steps."""
    # Step 1: Preload (gravity + aerodynamic pressure)
    # Mass scaling: target dt = DT_PRELOAD_SCALE for quasi-static preload
    model.ExplicitDynamicsStep(
        name='Step-Preload',
        previous='Initial',
        timePeriod=t_preload,
        massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING,
                       0.0, DT_PRELOAD_SCALE, None, 0, 0, 0.0, 0.0,
                       0, None),),
        description='Quasi-static preload: gravity + pressure')

    # Step 2: Separation (bolt removal + spring release)
    model.ExplicitDynamicsStep(
        name='Step-Separation',
        previous='Step-Preload',
        timePeriod=t_separation,
        description='Fairing separation dynamics')


# ==============================================================================
# LOADS AND BCS
# ==============================================================================

def apply_gravity(model, assembly, step_name='Step-Preload'):
    """Apply axial gravity (3g launch condition)."""
    model.Gravity(
        name='Load-Gravity',
        createStepName=step_name,
        comp2=-GRAVITY)  # Y = axial, negative = downward


def apply_adapter_bc(model, assembly, instances):
    """Fix adapter ring and PSS bottom (represents vehicle body)."""
    inst_adapter = instances['Adapter']
    region = regionToolset.Region(faces=inst_adapter.faces)
    model.EncastreBC(
        name='BC-Adapter-Fixed',
        createStepName='Initial',
        region=region)

    # Fix PSS bottom edge (attached to 2nd stage)
    if 'PSS-Core' in instances:
        for suffix in ['InnerSkin', 'Core', 'OuterSkin']:
            inst = instances['PSS-%s' % suffix]
            bottom_nodes = inst.nodes.getByBoundingBox(
                xMin=-5000.0, xMax=5000.0,
                yMin=PSS_Z_BOTTOM - 1.0, yMax=PSS_Z_BOTTOM + 1.0,
                zMin=-5000.0, zMax=5000.0)
            if len(bottom_nodes) > 0:
                region = regionToolset.Region(nodes=bottom_nodes)
                model.EncastreBC(
                    name='BC-PSS-%s-Fixed' % suffix,
                    createStepName='Initial',
                    region=region)


def apply_symmetry_bcs(model, assembly, instances, sector_angle=90.0):
    """Apply symmetry BCs at free circumferential edges.

    Q1 free edge at theta=0: ZSYMM (uz=0, rotx=0, roty=0)
    Q2 free edge at theta=2*sector_angle: computed symmetry plane

    For 90-deg sectors:
      theta=0 plane: XY plane → Z-symmetry (u3=0, ur1=0, ur2=0)
      theta=180 plane: also XY but mirrored → Z-symmetry
    """
    # Revolve geometry around Y axis:
    #   theta=0   → (x≈0, z=+R)  in YZ plane
    #   theta=90  → (x=+R, z≈0)  seam edge (XY plane)
    #   theta=180 → (x≈0, z=-R)  in YZ plane (Q2 after 90° rotation)
    # Symmetry planes at theta=0 and theta=180 are both YZ plane → XSYMM
    for qid, edge_desc in [('Q1', 'theta0'), ('Q2', 'theta180')]:
        for suffix in ['InnerSkin', 'Core', 'OuterSkin']:
            inst_name = '%s-%s' % (qid, suffix)
            inst = instances[inst_name]

            if edge_desc == 'theta0':
                # Q1 theta=0 edge: x≈0, z>0 (positive Z axis)
                edge_nodes = inst.nodes.getByBoundingBox(
                    xMin=-1.0, xMax=1.0,
                    yMin=BARREL_Z_MIN - 1, yMax=BARREL_Z_MAX + 1,
                    zMin=0.0, zMax=5000.0)
            else:
                # Q2 theta=180 edge (after 90° rotation): x≈0, z<0
                edge_nodes = inst.nodes.getByBoundingBox(
                    xMin=-1.0, xMax=1.0,
                    yMin=BARREL_Z_MIN - 1, yMax=BARREL_Z_MAX + 1,
                    zMin=-5000.0, zMax=0.0)

            if len(edge_nodes) > 0:
                region = regionToolset.Region(nodes=edge_nodes)
                model.XsymmBC(
                    name='BC-Sym-%s-%s' % (qid, suffix),
                    createStepName='Initial',
                    region=region)


# ==============================================================================
# FIELD OUTPUT
# ==============================================================================

def setup_outputs(model):
    """Configure field and history outputs."""
    # Field output for both steps
    for step_name in ['Step-Preload', 'Step-Separation']:
        model.FieldOutputRequest(
            name='F-Output-%s' % step_name,
            createStepName=step_name,
            variables=('U', 'V', 'A', 'S', 'LE', 'RF', 'STATUS'),
            timeInterval=0.002)  # 2ms intervals = 500Hz sampling

    # History output: energy
    model.HistoryOutputRequest(
        name='H-Energy',
        createStepName='Step-Preload',
        variables=('ALLKE', 'ALLSE', 'ALLAE', 'ALLIE', 'ETOTAL'))


# ==============================================================================
# INP POST-PROCESSING — SEPARATION MECHANISM
# ==============================================================================

def add_separation_to_inp(assembly, instances, inp_path,
                           n_stuck_bolts=0, stuck_bolt_indices=None):
    """Post-process INP: add CONN3D2 connector elements for separation.

    Creates:
    1. Seam connectors (Q1 ↔ Q2 at theta=0°) — pyro-cord
    2. Bottom bolt connectors (Q1/Q2 InnerSkin ↔ Adapter) — frangible bolts
    3. MODEL CHANGE in Step-Separation to remove them
    4. CLOAD opening forces in Step-Separation

    Uses CONN3D2 'Join' connectors (rigid link, all DOFs constrained).
    These can be removed via *MODEL CHANGE, TYPE=ELEMENT, REMOVE.
    """

    # --- 1. Find seam node pairs (Q1 ↔ Q2 at shared edge) ---
    # Seam is at theta=0° in global: x ≈ R, z ≈ 0
    seam_pairs = []
    for suffix in ['InnerSkin', 'OuterSkin']:
        q1 = instances['Q1-%s' % suffix]
        q2 = instances['Q2-%s' % suffix]

        # Q1 edge nodes at theta=0°: x > 0.8*R, |z| < mesh_seed
        q1_by_y = {}
        for node in q1.nodes:
            x, y, z = node.coordinates
            if x > RADIUS * 0.8 and abs(z) < MESH_SEED_DEFAULT:
                y_key = int(round(y))
                if y_key not in q1_by_y or abs(z) < abs(q1_by_y[y_key][3]):
                    q1_by_y[y_key] = (node.label, x, y, z)

        # Q2 edge nodes (after 90° rotation, its end is also at x ≈ R, z ≈ 0)
        q2_by_y = {}
        for node in q2.nodes:
            x, y, z = node.coordinates
            if x > RADIUS * 0.8 and abs(z) < MESH_SEED_DEFAULT:
                y_key = int(round(y))
                if y_key not in q2_by_y or abs(z) < abs(q2_by_y[y_key][3]):
                    q2_by_y[y_key] = (node.label, x, y, z)

        for y_key in sorted(q1_by_y.keys()):
            if y_key in q2_by_y:
                seam_pairs.append(('Q1-%s' % suffix, q1_by_y[y_key][0],
                                   'Q2-%s' % suffix, q2_by_y[y_key][0]))

    print("    Seam pairs: %d" % len(seam_pairs))

    # --- 2. Find bolt node pairs (Q1/Q2 bottom ↔ Adapter) ---
    bolt_pairs = []

    if 'Adapter' in instances:
        adapter = instances['Adapter']
        # Adapter top edge (y ≈ BARREL_Z_MIN = 0)
        ad_nodes = []
        for node in adapter.nodes:
            x, y, z = node.coordinates
            if abs(y - BARREL_Z_MIN) < 5.0:
                ad_nodes.append((node.label, x, y, z))

        for qid in ['Q1', 'Q2']:
            inst = instances['%s-InnerSkin' % qid]
            used_adapter = set()
            for node in inst.nodes:
                x, y, z = node.coordinates
                if abs(y - BARREL_Z_MIN) < 5.0:
                    # Find closest adapter node
                    best_d = 1e30
                    best_ad = None
                    for ad_l, ad_x, ad_y, ad_z in ad_nodes:
                        if ad_l in used_adapter:
                            continue
                        d = math.sqrt((x - ad_x)**2 +
                                      (y - ad_y)**2 +
                                      (z - ad_z)**2)
                        if d < best_d:
                            best_d = d
                            best_ad = ad_l
                    if best_ad and best_d < MESH_SEED_DEFAULT * 2:
                        bolt_pairs.append(('%s-InnerSkin' % qid, node.label,
                                          'Adapter', best_ad))
                        used_adapter.add(best_ad)

    print("    Bolt pairs: %d" % len(bolt_pairs))

    # --- 3. Determine stuck bolts ---
    stuck_set = set()
    if n_stuck_bolts > 0 and len(bolt_pairs) > 0:
        if stuck_bolt_indices is None:
            step = max(1, len(bolt_pairs) // (n_stuck_bolts + 1))
            stuck_bolt_indices = [min((i + 1) * step, len(bolt_pairs) - 1)
                                  for i in range(n_stuck_bolts)]
        stuck_set = set(stuck_bolt_indices)
        print("    Stuck bolt indices: %s" % list(stuck_set))

    # --- 4. Find opening force nodes ---
    spring_nodes = []  # (instance_name, node_label, force_z)
    total_force = SPRING_PRELOAD * N_SPRINGS_PER_HALF

    for qid, force_sign in [('Q1', 1.0), ('Q2', -1.0)]:
        inst = instances['%s-InnerSkin' % qid]
        bottom = []
        for node in inst.nodes:
            x, y, z = node.coordinates
            if abs(y - BARREL_Z_MIN) < 5.0:
                theta = math.atan2(z, x)
                bottom.append((node.label, theta, z))

        if bottom:
            # Sort by theta, pick nodes away from seam (near sector midpoint)
            bottom.sort(key=lambda n: n[1])
            n_force = min(4, len(bottom))
            step_f = max(1, len(bottom) // (n_force + 1))
            force_per_node = total_force / n_force
            for i in range(n_force):
                idx = min((i + 1) * step_f, len(bottom) - 1)
                nid = bottom[idx][0]
                spring_nodes.append(('%s-InnerSkin' % qid, nid,
                                     force_sign * force_per_node))

    print("    Opening force nodes: %d (%.0f N total per half)" % (
        len(spring_nodes), total_force))

    # --- 5. Build element definitions ---
    eid = 200001
    all_conn_lines = []
    release_ids = []

    # Bolt elements
    for i, (i1, n1, i2, n2) in enumerate(bolt_pairs):
        all_conn_lines.append('%d, %s.%d, %s.%d\n' % (eid, i1, n1, i2, n2))
        if i not in stuck_set:
            release_ids.append(eid)
        eid += 1

    # Pyro/seam elements (all released)
    for i1, n1, i2, n2 in seam_pairs:
        all_conn_lines.append('%d, %s.%d, %s.%d\n' % (eid, i1, n1, i2, n2))
        release_ids.append(eid)
        eid += 1

    n_total = len(all_conn_lines)
    n_release = len(release_ids)
    print("    Total connectors: %d (release: %d, stuck: %d)" % (
        n_total, n_release, n_total - n_release))

    if n_total == 0:
        print("    WARNING: No connector pairs found!")
        return

    # --- 6. Build INP injection blocks ---

    # Strategy for Abaqus/Explicit (no *MODEL CHANGE):
    #   - Release connectors: use CONN-RELEASE behavior with DAMAGE
    #     (force-based damage initiation → break under opening force)
    #   - Stuck connectors: use CONN-STUCK behavior (no damage, rigid)
    #
    # *Connector Behavior goes at MODEL level (before *Assembly or after *End Assembly)
    # *Connector Section + *Element go at ASSEMBLY level (inside *Assembly block)

    # Block M: Model-level connector behaviors (after *End Assembly)
    block_m = []
    block_m.append('**\n')
    block_m.append('** ========== CONNECTOR BEHAVIORS ==========\n')

    # Releasable behavior: stiff spring + damage
    block_m.append('*Connector Behavior, name=CONN-BEH-RELEASE\n')
    block_m.append('*Connector Elasticity, component=1\n')
    block_m.append('1.0e6\n')
    block_m.append('*Connector Elasticity, component=2\n')
    block_m.append('1.0e6\n')
    block_m.append('*Connector Elasticity, component=3\n')
    block_m.append('1.0e6\n')
    block_m.append('*Connector Damage Initiation, component=1\n')
    block_m.append('100.0\n')
    block_m.append('*Connector Damage Evolution, type=energy\n')
    block_m.append('0.001\n')
    block_m.append('*Connector Damage Initiation, component=2\n')
    block_m.append('100.0\n')
    block_m.append('*Connector Damage Evolution, type=energy\n')
    block_m.append('0.001\n')
    block_m.append('*Connector Damage Initiation, component=3\n')
    block_m.append('100.0\n')
    block_m.append('*Connector Damage Evolution, type=energy\n')
    block_m.append('0.001\n')

    # Stuck behavior: very stiff, no damage
    block_m.append('*Connector Behavior, name=CONN-BEH-STUCK\n')
    block_m.append('*Connector Elasticity, component=1\n')
    block_m.append('1.0e8\n')
    block_m.append('*Connector Elasticity, component=2\n')
    block_m.append('1.0e8\n')
    block_m.append('*Connector Elasticity, component=3\n')
    block_m.append('1.0e8\n')
    block_m.append('** ==========================================\n')

    # Block A: Assembly-level elements + sections (before *End Assembly)
    release_conn_lines = []
    stuck_conn_lines = []
    release_ids_set = set(release_ids)
    eid_check = 200001
    for line in all_conn_lines:
        if eid_check in release_ids_set:
            release_conn_lines.append(line)
        else:
            stuck_conn_lines.append(line)
        eid_check += 1

    block_a = []
    block_a.append('**\n')
    block_a.append('** ========== SEPARATION CONNECTORS ==========\n')

    if release_conn_lines:
        block_a.append('*Element, type=CONN3D2, elset=CONN-RELEASE\n')
        block_a.extend(release_conn_lines)
        block_a.append('*Connector Section, elset=CONN-RELEASE, behavior=CONN-BEH-RELEASE\n')
        block_a.append('Cartesian,\n')

    if stuck_conn_lines:
        block_a.append('*Element, type=CONN3D2, elset=CONN-STUCK\n')
        block_a.extend(stuck_conn_lines)
        block_a.append('*Connector Section, elset=CONN-STUCK, behavior=CONN-BEH-STUCK\n')
        block_a.append('Cartesian,\n')

    block_a.append('** ==========================================\n')

    # Block B: Opening forces in Step-Separation (after *Dynamic)
    # Note: *MODEL CHANGE is NOT available in Abaqus/Explicit.
    # Instead, we use *CONNECTOR DAMAGE to break connectors at separation.
    # The damage is defined in Block A with the connector section.
    block_b = []
    if spring_nodes:
        block_b.append('**\n')
        block_b.append('** --- Opening spring forces ---\n')
        block_b.append('*Cload\n')
        for inst_name, nid, fz in spring_nodes:
            block_b.append('%s.%d, 3, %.1f\n' % (inst_name, nid, fz))
    block_b.append('**\n')

    # --- 7. Read and modify INP ---
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    # Find *End Assembly
    end_asm = None
    for i, line in enumerate(lines):
        if '*End Assembly' in line:
            end_asm = i
            break

    if end_asm is None:
        print("    ERROR: *End Assembly not found")
        return

    # Insert Block A before *End Assembly
    lines = lines[:end_asm] + block_a + lines[end_asm:]

    # Insert Block M (connector behaviors) after *End Assembly
    # *End Assembly is now shifted by len(block_a)
    end_asm_new = end_asm + len(block_a)
    lines = lines[:end_asm_new + 1] + block_m + lines[end_asm_new + 1:]

    # Find Step-Separation and insert Block B after *Dynamic + parameters
    if block_b:
        for i in range(len(lines)):
            if '*Step, name=Step-Separation' in lines[i]:
                for j in range(i + 1, min(i + 15, len(lines))):
                    if lines[j].strip().startswith('*Dynamic'):
                        # Skip past *Dynamic line AND its parameter line(s)
                        # Parameter lines start with comma or number
                        insert_at = j + 1
                        while insert_at < len(lines):
                            stripped = lines[insert_at].strip()
                            if stripped.startswith(',') or (stripped and stripped[0].isdigit()):
                                insert_at += 1
                            else:
                                break
                        lines = lines[:insert_at] + block_b + lines[insert_at:]
                        break
                break

    with open(inp_path, 'w') as f:
        f.writelines(lines)

    print("    INP post-processed: %d connectors, %d release" % (
        n_total, n_release))


def inject_separation_inp(inp_path, bolt_info, pyro_info,
                          separation_params):
    """Post-process INP file to add MODEL CHANGE for bolt/pyro removal.

    This is the key technique: Abaqus/Explicit supports *MODEL CHANGE
    to remove elements at step boundaries. We inject these keywords
    into the INP file after the Step-Separation header.

    Args:
        inp_path: path to .inp file
        bolt_info: list of bolt dicts (from create_bolt_connectors)
        pyro_info: list of pyro dicts
        separation_params: dict with timing, stuck bolts, etc.
    """
    stuck_bolts = separation_params.get('stuck_bolts', [])

    with open(inp_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    in_step_sep = False

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Detect Step-Separation
        if '*Step, name=Step-Separation' in line:
            in_step_sep = True
            continue

        # After the *Dynamic line within Step-Separation, inject MODEL CHANGE
        if in_step_sep and line.strip().startswith('*Dynamic'):
            # Remove bolts (except stuck ones)
            bolt_sets = []
            for b in bolt_info:
                if not b['stuck']:
                    bolt_sets.append(b['set_name'])

            if bolt_sets:
                new_lines.append('**\n')
                new_lines.append('** === Frangible bolt removal ===\n')
                new_lines.append('*MODEL CHANGE, TYPE=ELEMENT, REMOVE\n')
                for bs in bolt_sets:
                    new_lines.append('%s,\n' % bs)

            # Remove pyro-cord connectors
            pyro_sets = [p['set_name'] for p in pyro_info]
            if pyro_sets:
                new_lines.append('**\n')
                new_lines.append('** === Pyro-cord removal ===\n')
                new_lines.append('*MODEL CHANGE, TYPE=ELEMENT, REMOVE\n')
                for ps in pyro_sets:
                    new_lines.append('%s,\n' % ps)

            in_step_sep = False  # only inject once

    with open(inp_path, 'w') as f:
        f.writelines(new_lines)

    print("INP post-processed: %d bolt removals, %d pyro removals" % (
        len([b for b in bolt_info if not b['stuck']]),
        len(pyro_info)))


# ==============================================================================
# MAIN MODEL GENERATION
# ==============================================================================

def generate_model(job_name,
                   z_min=BARREL_Z_MIN, z_max=BARREL_Z_MAX,
                   mesh_seed=MESH_SEED_DEFAULT,
                   t_preload=T_PRELOAD, t_separation=T_SEPARATION,
                   n_bolts=N_BOLTS_FULL,
                   spring_stiffness=SPRING_STIFFNESS,
                   n_stuck_bolts=0, stuck_bolt_indices=None,
                   pyro_asymmetry=1.0,
                   include_pss=False,
                   no_run=False):
    """Generate fairing separation dynamics model.

    Args:
        job_name: Abaqus job name
        z_min, z_max: barrel axial range (mm)
        mesh_seed: element size (mm)
        t_preload, t_separation: step durations (s)
        n_bolts: number of frangible bolts
        spring_stiffness: opening spring stiffness (N/mm)
        n_stuck_bolts: number of bolts that fail to cut (0=normal)
        stuck_bolt_indices: specific bolt indices to keep (auto if None)
        pyro_asymmetry: energy ratio for pyro-cord (1.0=symmetric)
        no_run: if True, write INP but don't submit
    """
    print("=" * 60)
    print("Fairing Separation Model: %s" % job_name)
    print("=" * 60)
    print("  Barrel: z=[%.0f, %.0f] mm, R=%.0f mm" % (z_min, z_max, RADIUS))
    print("  Mesh seed: %.0f mm" % mesh_seed)
    n_bolts_sector = n_bolts // 4  # quarter model
    print("  Bolts: %d in sector (%d full), %d stuck" % (
        n_bolts_sector, n_bolts, n_stuck_bolts))
    print("  Spring stiffness: %.0f N/mm" % spring_stiffness)
    print("  Pyro asymmetry: %.2f" % pyro_asymmetry)
    print("  Time: preload=%.3fs, separation=%.3fs" % (
        t_preload, t_separation))

    # Determine stuck bolt indices
    if stuck_bolt_indices is None and n_stuck_bolts > 0:
        # Distribute stuck bolts evenly (worst case: clustered)
        stuck_bolt_indices = []
        step = n_bolts // (n_stuck_bolts + 1)
        for i in range(n_stuck_bolts):
            stuck_bolt_indices.append((i + 1) * step)
        print("  Stuck bolts at indices: %s" % stuck_bolt_indices)

    Mdb()
    model = mdb.models['Model-1']

    # 1. Materials and sections
    print("\n[1/8] Creating materials and sections...")
    create_materials(model)
    create_sections(model)

    sector_angle = DEFAULT_SECTOR_ANGLE  # 90 degrees

    # 2. Geometry — two quarter-shells (90 deg each)
    print("[2/8] Creating geometry (two 90-deg quarter-shells)...")
    parts_Q1 = create_quarter_shell(model, 'Q1', sector_angle, z_min, z_max)
    parts_Q2 = create_quarter_shell(model, 'Q2', sector_angle, z_min, z_max)

    # 3. Adapter ring (covers both quarters = 2 * sector_angle)
    print("[3/9] Creating adapter ring...")
    p_adapter = create_adapter_ring(model, z_bottom=z_min,
                                     sweep_angle=2.0 * sector_angle)

    # 3b. PSS (Payload Support Structure) — full 360° cylinder
    p_pss = None
    if include_pss:
        print("[3b/9] Creating PSS (R=%.0f mm, H=%.0f mm)..." % (
            PSS_RADIUS, PSS_HEIGHT))
        p_pss = create_pss(model, sweep_angle=360.0)

    # 3c. Assign section properties to all parts
    print("[3c/9] Assigning section properties...")
    assign_sections(model, parts_Q1, parts_Q2, p_adapter, p_pss)

    # 4. Assembly
    print("[4/9] Assembling...")
    assembly, instances = create_assembly(
        model, parts_Q1, parts_Q2, p_adapter, p_pss, sector_angle)

    # 5. Mesh
    print("[5/8] Meshing (seed=%.0f mm)..." % mesh_seed)
    generate_mesh(assembly, instances, mesh_seed)

    # 6. Ties (skin-core bonding)
    print("[6/8] Creating ties and constraints...")
    create_skin_core_ties(model, assembly, instances)

    # 7. Steps, loads, BCs
    print("[7/8] Creating steps, loads, and BCs...")
    create_steps(model, t_preload, t_separation)
    apply_gravity(model, assembly)
    apply_adapter_bc(model, assembly, instances)
    apply_symmetry_bcs(model, assembly, instances, sector_angle)
    setup_outputs(model)

    # 8. Write INP and post-process
    print("[8/8] Writing INP...")
    inp_dir = os.getcwd()
    job = mdb.Job(name=job_name, model='Model-1',
                  description='Fairing separation dynamics',
                  numCpus=4, numDomains=4,
                  explicitPrecision=DOUBLE_PLUS_PACK)

    job.writeInput()
    inp_path = os.path.join(inp_dir, job_name + '.inp')
    print("  INP written: %s" % inp_path)

    # 8b. Post-process INP: add separation mechanism
    print("[8b/9] Adding separation mechanism (INP post-processing)...")
    add_separation_to_inp(assembly, instances, inp_path,
                           n_stuck_bolts, stuck_bolt_indices)

    if not no_run:
        print("\nSubmitting job: %s" % job_name)
        job.submit()
        job.waitForCompletion()
        print("Job completed: %s" % job_name)
    else:
        print("\n--no_run specified. INP file ready at: %s" % inp_path)

    return job_name


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate fairing separation dynamics model')
    parser.add_argument('--job', default='Sep-Test',
                        help='Job name')
    parser.add_argument('--mesh_seed', type=float, default=MESH_SEED_DEFAULT,
                        help='Mesh seed size (mm)')
    parser.add_argument('--z_min', type=float, default=BARREL_Z_MIN)
    parser.add_argument('--z_max', type=float, default=BARREL_Z_MAX)
    parser.add_argument('--n_bolts', type=int, default=N_BOLTS_FULL)
    parser.add_argument('--spring_stiffness', type=float,
                        default=SPRING_STIFFNESS)
    parser.add_argument('--n_stuck_bolts', type=int, default=0,
                        help='Number of bolts that fail to cut (0=normal)')
    parser.add_argument('--pyro_asymmetry', type=float, default=1.0,
                        help='Pyro-cord energy asymmetry (1.0=symmetric)')
    parser.add_argument('--t_separation', type=float, default=T_SEPARATION,
                        help='Separation step duration (s)')
    parser.add_argument('--include_pss', action='store_true',
                        help='Include PSS (Payload Support Structure)')
    parser.add_argument('--no_run', action='store_true',
                        help='Write INP only, do not submit')

    args, _ = parser.parse_known_args()

    generate_model(
        job_name=args.job,
        z_min=args.z_min,
        z_max=args.z_max,
        mesh_seed=args.mesh_seed,
        n_bolts=args.n_bolts,
        spring_stiffness=args.spring_stiffness,
        n_stuck_bolts=args.n_stuck_bolts,
        pyro_asymmetry=args.pyro_asymmetry,
        t_separation=args.t_separation,
        include_pss=args.include_pss,
        no_run=args.no_run,
    )
