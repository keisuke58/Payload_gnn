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

def create_assembly(model, p_inner, p_core, p_outer):
    """Create assembly with positioned instances.

    Inner skin at Z=0, Core at Z=0..CORE_T, Outer skin translated to Z=CORE_T.
    """
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)

    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)

    # Translate outer skin to Z=CORE_T
    a.translate(instanceList=('Part-OuterSkin-1', ),
                vector=(0.0, 0.0, CORE_T))

    return a, inst_inner, inst_core, inst_outer


# ==============================================================================
# TIE CONSTRAINTS
# ==============================================================================

def create_tie_constraints(model, assembly, inst_inner, inst_core, inst_outer,
                           defect_params=None):
    """Create Tie constraints between skins and core.

    Tie 1: Inner skin (Z=0) <-> Core bottom face (Z=0) — always full
    Tie 2: Core top face (Z=CORE_T) <-> Outer skin (Z=CORE_T)
           If defect_params: excludes defect zone (debonding)

    Debonding is modeled by removing Tie in the defect region,
    allowing the wave to scatter at the debonding boundary.
    """
    # --- Tie 1: Inner skin <-> Core bottom (always full, no defect) ---
    surf_inner = assembly.Surface(
        side1Faces=inst_inner.faces, name='Surf-InnerSkin')

    core_bot_pts = []
    for f in inst_core.faces:
        pt = f.pointOn[0]
        if abs(pt[2]) < 0.1:
            core_bot_pts.append((pt,))
    if core_bot_pts:
        core_bot_seq = inst_core.faces.findAt(*core_bot_pts)
        surf_core_bot = assembly.Surface(
            side1Faces=core_bot_seq, name='Surf-Core-Bottom')
        model.Tie(name='Tie-InnerSkin-Core',
                  main=surf_core_bot, secondary=surf_inner,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie 1: InnerSkin <-> Core bottom (%d faces)" % len(core_bot_pts))
    else:
        print("WARNING: No core bottom faces found for Tie 1")

    # --- Tie 2: Outer skin <-> Core top ---
    # Collect core top faces and outer skin faces, excluding defect zone
    core_top_healthy_pts = []
    core_top_defect_pts = []
    for f in inst_core.faces:
        pt = f.pointOn[0]
        if abs(pt[2] - CORE_T) < 0.1:
            if defect_params and _point_in_defect(pt[0], pt[1], defect_params):
                core_top_defect_pts.append((pt,))
            else:
                core_top_healthy_pts.append((pt,))

    outer_healthy_pts = []
    outer_defect_pts = []
    for f in inst_outer.faces:
        pt = f.pointOn[0]
        if defect_params and _point_in_defect(pt[0], pt[1], defect_params):
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
    elif not defect_params:
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

    # Report defect exclusion
    if defect_params:
        n_core_def = len(core_top_defect_pts)
        n_outer_def = len(outer_defect_pts)
        print("Debonding: %d core top + %d outer skin faces UNTIED" % (
            n_core_def, n_outer_def))
        if n_core_def == 0 and n_outer_def == 0:
            print("WARNING: No faces found in defect zone — check defect params")


# ==============================================================================
# MESH
# ==============================================================================

def generate_mesh(assembly, inst_inner, inst_core, inst_outer, mesh_seed):
    """Generate mesh for all parts.

    Shells: S4R (EXPLICIT library)
    Core:   C3D8R (EXPLICIT, hourglass control)
    """
    # Global seed
    assembly.seedPartInstance(
        regions=(inst_inner, inst_core, inst_outer),
        size=mesh_seed, deviationFactor=0.1)

    # Core: structured hex mesh with C3D8R
    assembly.setMeshControls(regions=inst_core.cells,
                             elemShape=HEX, technique=STRUCTURED)
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
    print("Mesh: seed=%.1f mm" % mesh_seed)
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


def apply_excitation(model, assembly, inst_outer, freq_hz, n_cycles, step_name):
    """Apply concentrated force at panel center on outer skin.

    Force in Z-direction (out-of-plane) excites A0 Lamb wave mode.
    """
    amp_name = generate_tone_burst_amplitude(model, freq_hz, n_cycles)

    # Center node on outer skin (0, 0, CORE_T)
    center_label, center_dist = find_nearest_node(inst_outer, 0.0, 0.0, CORE_T)
    if center_label is None:
        print("ERROR: Could not find center node on outer skin")
        return

    print("Excitation node: label=%d, snap_dist=%.3f mm" % (
        center_label, center_dist))

    node_seq = inst_outer.nodes.sequenceFromLabels((center_label,))
    excite_set = assembly.Set(nodes=node_seq, name='Set-Excitation')

    model.ConcentratedForce(
        name='Force-ToneBurst',
        createStepName=step_name,
        region=excite_set,
        cf3=FORCE_MAGNITUDE,
        amplitude=amp_name)

    print("Concentrated force: %.1f N in Z, amplitude=%s" % (
        FORCE_MAGNITUDE, amp_name))


def setup_sensor_outputs(model, assembly, inst_outer, step_name):
    """Create history output at sensor locations along X-axis on outer skin."""
    sensor_info = []
    for i, offset in enumerate(SENSOR_OFFSETS):
        label, dist = find_nearest_node(inst_outer, offset, 0.0, CORE_T)
        if label is None:
            print("WARNING: Sensor %d at offset=%.0f mm: no node" % (i, offset))
            continue

        node_seq = inst_outer.nodes.sequenceFromLabels((label,))
        set_name = 'Set-Sensor-%d' % i
        sensor_set = assembly.Set(nodes=node_seq, name=set_name)

        model.HistoryOutputRequest(
            name='H-Output-Sensor-%d' % i,
            createStepName=step_name,
            variables=('U3', ),
            region=sensor_set,
            sectionPoints=DEFAULT,
            rebar=EXCLUDE,
            timeInterval=FIELD_OUTPUT_INTERVAL)

        sensor_info.append((i, offset, label, dist))
        print("Sensor %d: x=%.0f mm, node=%d, snap=%.3f mm" % (
            i, offset, label, dist))

    print("History output: %d sensors, U3 at dt=%.1e s" % (
        len(sensor_info), FIELD_OUTPUT_INTERVAL))
    return sensor_info


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def generate_model(job_name, freq_khz=DEFAULT_FREQ_KHZ, n_cycles=DEFAULT_CYCLES,
                   panel_size=PANEL_SIZE, mesh_seed=None, time_period=None,
                   defect_params=None, no_run=False):
    """Generate guided wave flat panel model (Abaqus/Explicit).

    Args:
        defect_params: dict with {x_center, y_center, radius} for debonding.
                       None = healthy panel.
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

    print("=" * 70)
    print("Guided Wave Model: %s" % job_name)
    print("  Panel: %.0f x %.0f mm, CFRP/Al-HC/CFRP (t=%.0f/%.0f/%.0f mm)" % (
        panel_size, panel_size, FACE_T, CORE_T, FACE_T))
    print("  Excitation: %d-cycle Hanning tone burst at %.0f kHz" % (
        n_cycles, freq_khz))
    print("  Mesh seed: %.2f mm (lambda=%.1f mm)" % (
        mesh_seed, CP_ESTIMATE / freq_hz * 1000.0))
    print("  Time: %.3e s (%.3f ms)" % (time_period, time_period * 1e3))
    if defect_params:
        print("  Defect: DEBONDING at (%.0f, %.0f) r=%.0f mm" % (
            defect_params['x_center'], defect_params['y_center'],
            defect_params['radius']))
    else:
        print("  Defect: NONE (healthy)")
    print("=" * 70)

    Mdb()
    model = mdb.models['Model-1']

    # 1. Materials & Sections
    create_materials(model)
    create_sections(model)

    # 2. Parts
    p_inner, p_core, p_outer = create_flat_panel_parts(model, panel_size)

    # 3. Partition core for through-thickness elements
    partition_core_thickness(p_core)

    # 3b. Partition defect zone (outer skin + core)
    if defect_params:
        partition_defect_zone(p_outer, p_core, defect_params)

    # 4. Assign sections
    assign_sections(p_inner, p_core, p_outer)

    # 5. Assembly
    a, inst_inner, inst_core, inst_outer = create_assembly(
        model, p_inner, p_core, p_outer)

    # 6. Tie constraints (healthy region only if defect)
    create_tie_constraints(model, a, inst_inner, inst_core, inst_outer,
                           defect_params)

    # 7. Mesh
    generate_mesh(a, inst_inner, inst_core, inst_outer, mesh_seed)

    # 8. Explicit step
    step_name = create_explicit_step(model, time_period)

    # 9. Excitation
    apply_excitation(model, a, inst_outer, freq_hz, n_cycles, step_name)

    # 10. Sensor outputs
    setup_sensor_outputs(model, a, inst_outer, step_name)

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
        description='Guided wave flat panel model (Abaqus/Explicit)')
    parser.add_argument('--job', type=str, default='Job-GuidedWave',
                        help='Job name (default: Job-GuidedWave)')
    parser.add_argument('--freq', type=float, default=DEFAULT_FREQ_KHZ,
                        help='Center frequency in kHz (default: 50)')
    parser.add_argument('--cycles', type=int, default=DEFAULT_CYCLES,
                        help='Number of burst cycles (default: 5)')
    parser.add_argument('--panel_size', type=float, default=PANEL_SIZE,
                        help='Panel side length in mm (default: 300)')
    parser.add_argument('--mesh_seed', type=float, default=None,
                        help='Mesh seed in mm (auto: lambda/8)')
    parser.add_argument('--time', type=float, default=None,
                        help='Analysis time period in seconds (auto)')
    parser.add_argument('--defect', type=str, default=None,
                        help='Defect JSON: {"x_center":80,"y_center":0,"radius":25}')
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
        no_run=args.no_run)
