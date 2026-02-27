# -*- coding: utf-8 -*-
"""
Payload Fairing Dataset Generation — Shell-Solid-Shell Sandwich Model

Three-part sandwich:
  Outer Facesheet  (S4R shell)   CFRP [45/0/-45/90]s
  Honeycomb Core   (C3D8R solid) Orthotropic homogenised
  Inner Facesheet  (S4R shell)   CFRP [45/0/-45/90]s
Connected via Tie Constraints.

Usage:
  cd abaqus_work
  abaqus cae noGUI=../src/generate_fairing_dataset.py
"""

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import section
import interaction
import step
import load
import math
import os
import sys

# =========================================================================
# Configuration
# =========================================================================
MODEL_NAME = 'Fairing_Sandwich'
JOB_NAME = 'Fairing_Healthy'

RADIUS = 2000.0          # mm  outer skin mid-surface radius
HEIGHT = 5000.0          # mm
ANGLE = 60.0             # 1/6 section

# CFRP T300/914
E1, E2, Nu12 = 135000.0, 10000.0, 0.3
G12, G13, G23 = 5000.0, 5000.0, 3000.0

LAYUP = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
PLY_T = 0.15
FACE_T = PLY_T * len(LAYUP)   # 1.2 mm

# Honeycomb core — orthotropic (HexWeb CR-III 3/16-5052-.001)
CORE_T = 20.0
CORE_E = {
    'E1': 10.0, 'E2': 10.0, 'E3': 1380.0,
    'Nu12': 0.001, 'Nu13': 0.001, 'Nu23': 0.001,
    'G12': 3.85, 'G13': 310.0, 'G23': 180.0,
}

MESH_SHELL = 80.0        # mm — coarser for first test
MESH_CORE  = 80.0

# =========================================================================
# Thermal Environment — JAXA Epsilon/H3 Ascent (Literature Values)
# Ref: Epsilon Users Manual, WIKI.md Section 2.4
# =========================================================================
T_REF          = 25.0       # deg C — stress-free reference (pre-launch ambient)
T_OUTER_SKIN   = 150.0      # deg C — outer skin peak (100-200 C range midpoint)
T_INNER_SKIN   = 50.0       # deg C — inner skin (shielded cavity)

# CFRP T300/914 — Thermal Properties
# Units: mm-t-s-C consistent system
CFRP_DENSITY         = 1.58e-9      # t/mm^3  (1580 kg/m^3)
CFRP_CONDUCTIVITY_11 = 7.0e-3       # mW/(mm C) — fiber direction
CFRP_CONDUCTIVITY_22 = 0.8e-3       # mW/(mm C) — transverse
CFRP_SPECIFIC_HEAT   = 9.0e+8       # mm^2/(s^2 C)  (900 J/(kg C))
CFRP_CTE_1           = -0.3e-6      # /C — fiber direction (negative, carbon fiber)
CFRP_CTE_2           = 28.0e-6      # /C — transverse

# Temperature-dependent CFRP elastic properties
# (E1, E2, Nu12, G12, G13, G23, Temperature)
CFRP_ELASTIC_TEMP = [
    (135000., 10000., 0.30, 5000., 5000., 3000.,  25.),
    (133000.,  9000., 0.30, 4500., 4500., 2700., 100.),
    (130000.,  7000., 0.29, 3800., 3800., 2300., 150.),
    (125000.,  5500., 0.28, 3200., 3200., 1800., 200.),
]

# Honeycomb Core (Al 5052) — Thermal Properties
CORE_DENSITY       = 4.8e-11        # t/mm^3  (48 kg/m^3)
CORE_CONDUCTIVITY  = 1.5e-3         # mW/(mm C) — homogenised
CORE_SPECIFIC_HEAT = 9.0e+8         # mm^2/(s^2 C)  (900 J/(kg C))
CORE_CTE           = 23.0e-6        # /C — isotropic (aluminium)

# Radii
R_OUTER   = RADIUS                                 # outer skin mid-surface
R_CORE_O  = RADIUS - FACE_T / 2.0                  # core outer face
R_CORE_I  = RADIUS - FACE_T / 2.0 - CORE_T         # core inner face
R_INNER   = RADIUS - FACE_T - CORE_T               # inner skin mid-surface

# =========================================================================
# Helper — arc end point
# =========================================================================
def arc_end(r, angle_deg):
    return (r * math.cos(math.radians(angle_deg)),
            r * math.sin(math.radians(angle_deg)))

# =========================================================================
# Build model
# =========================================================================
def build_model():
    # Clean
    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]
    model = mdb.Model(name=MODEL_NAME)

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    mat = model.Material(name='CFRP_T300')
    mat.Elastic(type=LAMINA,
                table=((E1, E2, Nu12, G12, G13, G23),))

    mat_c = model.Material(name='HC_Core')
    mat_c.Elastic(type=ENGINEERING_CONSTANTS,
                  table=((CORE_E['E1'], CORE_E['E2'], CORE_E['E3'],
                          CORE_E['Nu12'], CORE_E['Nu13'], CORE_E['Nu23'],
                          CORE_E['G12'], CORE_E['G13'], CORE_E['G23']),))

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    layers = []
    for i, ang in enumerate(LAYUP):
        layers.append(section.SectionLayer(
            material='CFRP_T300', thickness=PLY_T,
            orientAngle=ang, numIntPts=3, plyName='P%d' % i))
    model.CompositeShellSection(
        name='Skin_Sec', preIntegrate=OFF,
        idealization=NO_IDEALIZATION, symmetric=OFF,
        thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        temperature=GRADIENT, integrationRule=SIMPSON,
        layup=layers)

    model.HomogeneousSolidSection(
        name='Core_Sec', material='HC_Core', thickness=None)

    # ------------------------------------------------------------------
    # Parts
    # ------------------------------------------------------------------
    # Outer skin (shell)
    sk = model.ConstrainedSketch(name='sk_outer', sheetSize=6000.0)
    sk.ArcByCenterEnds(center=(0, 0), point1=(R_OUTER, 0),
                       point2=arc_end(R_OUTER, ANGLE),
                       direction=COUNTERCLOCKWISE)
    p_out = model.Part(name='Skin_Outer', dimensionality=THREE_D,
                       type=DEFORMABLE_BODY)
    p_out.BaseShellExtrude(sketch=sk, depth=HEIGHT)

    # Inner skin (shell)
    sk2 = model.ConstrainedSketch(name='sk_inner', sheetSize=6000.0)
    sk2.ArcByCenterEnds(center=(0, 0), point1=(R_INNER, 0),
                        point2=arc_end(R_INNER, ANGLE),
                        direction=COUNTERCLOCKWISE)
    p_in = model.Part(name='Skin_Inner', dimensionality=THREE_D,
                      type=DEFORMABLE_BODY)
    p_in.BaseShellExtrude(sketch=sk2, depth=HEIGHT)

    # Core (solid) — closed cross-section extruded along z
    sk3 = model.ConstrainedSketch(name='sk_core', sheetSize=6000.0)
    # outer arc
    sk3.ArcByCenterEnds(center=(0, 0),
                        point1=(R_CORE_O, 0),
                        point2=arc_end(R_CORE_O, ANGLE),
                        direction=COUNTERCLOCKWISE)
    # line at theta=ANGLE
    sk3.Line(point1=arc_end(R_CORE_O, ANGLE),
             point2=arc_end(R_CORE_I, ANGLE))
    # inner arc (clockwise = towards theta=0)
    sk3.ArcByCenterEnds(center=(0, 0),
                        point1=arc_end(R_CORE_I, ANGLE),
                        point2=(R_CORE_I, 0),
                        direction=CLOCKWISE)
    # line at theta=0
    sk3.Line(point1=(R_CORE_I, 0), point2=(R_CORE_O, 0))

    p_core = model.Part(name='Core', dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidExtrude(sketch=sk3, depth=HEIGHT)

    # ------------------------------------------------------------------
    # Section assignment + material orientation
    # ------------------------------------------------------------------
    for p_s in (p_out, p_in):
        csys = p_s.DatumCsysByThreePoints(
            name='CylCS', coordSysType=CYLINDRICAL,
            origin=(0, 0, 0), point1=(1, 0, 0), point2=(0, 1, 0))
        face_set = p_s.Set(faces=p_s.faces, name='AllFaces')
        p_s.SectionAssignment(
            region=face_set, sectionName='Skin_Sec',
            offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',
            thicknessAssignment=FROM_SECTION)
        p_s.MaterialOrientation(
            region=face_set, orientationType=SYSTEM, axis=AXIS_1,
            localCsys=p_s.datums[csys.id], fieldName='',
            additionalRotationType=ROTATION_NONE, angle=0.0,
            additionalRotationField='', stackDirection=STACK_3)

    csys_c = p_core.DatumCsysByThreePoints(
        name='CylCS_C', coordSysType=CYLINDRICAL,
        origin=(0, 0, 0), point1=(1, 0, 0), point2=(0, 1, 0))
    cell_set = p_core.Set(cells=p_core.cells, name='AllCells')
    p_core.SectionAssignment(
        region=cell_set, sectionName='Core_Sec',
        offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',
        thicknessAssignment=FROM_SECTION)
    p_core.MaterialOrientation(
        region=cell_set, orientationType=SYSTEM, axis=AXIS_3,
        localCsys=p_core.datums[csys_c.id], fieldName='',
        additionalRotationType=ROTATION_NONE, angle=0.0,
        additionalRotationField='', stackDirection=STACK_3)

    # ------------------------------------------------------------------
    # Core face sets (identify outer / inner curved faces)
    # ------------------------------------------------------------------
    tol_r = 2.0   # mm tolerance for radius check
    core_outer_faces = []
    core_inner_faces = []
    for f in p_core.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0]**2 + pt[1]**2)
        if abs(r - R_CORE_O) < tol_r:
            core_outer_faces.append(f)
        elif abs(r - R_CORE_I) < tol_r:
            core_inner_faces.append(f)

    if core_outer_faces:
        p_core.Surface(side1Faces=p_core.faces[
            core_outer_faces[0].index:core_outer_faces[0].index + 1],
            name='Surf_Outer')
    if core_inner_faces:
        p_core.Surface(side1Faces=p_core.faces[
            core_inner_faces[0].index:core_inner_faces[0].index + 1],
            name='Surf_Inner')

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    for p_s in (p_out, p_in):
        p_s.seedPart(size=MESH_SHELL, deviationFactor=0.1,
                     minSizeFactor=0.1)
        et = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD,
                           secondOrderAccuracy=OFF,
                           hourglassControl=DEFAULT)
        et3 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
        p_s.setElementType(
            regions=p_s.sets['AllFaces'], elemTypes=(et, et3))
        p_s.generateMesh()

    p_core.seedPart(size=MESH_CORE, deviationFactor=0.1,
                    minSizeFactor=0.1)
    # Seed through-thickness: 1 element
    for e in p_core.edges:
        pt = e.pointOn[0]
        r = math.sqrt(pt[0]**2 + pt[1]**2)
        # Radial edges connect R_CORE_O to R_CORE_I
        if abs(pt[2]) < 1.0 or abs(pt[2] - HEIGHT) < 1.0:
            nodes_on = e.getNodes()
            # If edge length ~ CORE_T, it's a radial edge
            verts = e.getVertices()
            if len(verts) == 2:
                v0 = p_core.vertices[verts[0]]
                v1 = p_core.vertices[verts[1]]
                p0 = v0.pointOn[0]
                p1 = v1.pointOn[0]
                length = math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 + (p0[2]-p1[2])**2)
                if abs(length - CORE_T) < 5.0:
                    p_core.seedEdgeByNumber(edges=(e,), number=1)

    et_hex = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD,
                           kinematicSplit=AVERAGE_STRAIN,
                           hourglassControl=ENHANCED)
    et_wed = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    et_tet = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
    p_core.setElementType(
        regions=p_core.sets['AllCells'],
        elemTypes=(et_hex, et_wed, et_tet))
    p_core.generateMesh()

    print("Mesh: Outer=%d, Inner=%d, Core=%d nodes" %
          (len(p_out.nodes), len(p_in.nodes), len(p_core.nodes)))

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    i_out  = a.Instance(name='Skin_Outer-1', part=p_out, dependent=ON)
    i_in   = a.Instance(name='Skin_Inner-1', part=p_in,  dependent=ON)
    i_core = a.Instance(name='Core-1',       part=p_core, dependent=ON)

    # ------------------------------------------------------------------
    # Tie constraints — use findAt for face selection
    # ------------------------------------------------------------------
    # Reference points on core curved faces (mid-height, mid-angle)
    theta_mid = math.radians(ANGLE / 2.0)
    z_mid = HEIGHT / 2.0

    pt_core_outer = (R_CORE_O * math.cos(theta_mid),
                     R_CORE_O * math.sin(theta_mid), z_mid)
    pt_core_inner = (R_CORE_I * math.cos(theta_mid),
                     R_CORE_I * math.sin(theta_mid), z_mid)

    # Outer skin surface
    surf_skin_out = a.Surface(
        side1Faces=i_out.faces, name='S_SkinOuter')

    # Core outer curved face
    co_face = i_core.faces.findAt((pt_core_outer,))
    if co_face:
        surf_core_out = a.Surface(
            side1Faces=co_face, name='S_CoreOuter')
        model.Tie(name='Tie_Outer', main=surf_skin_out,
                  secondary=surf_core_out,
                  positionToleranceMethod=COMPUTED,
                  adjust=ON, tieRotations=ON, thickness=ON)
        print("Tie_Outer: skin outer <-> core outer face")

    # Inner skin surface
    surf_skin_in = a.Surface(
        side1Faces=i_in.faces, name='S_SkinInner')

    # Core inner curved face
    ci_face = i_core.faces.findAt((pt_core_inner,))
    if ci_face:
        surf_core_in = a.Surface(
            side1Faces=ci_face, name='S_CoreInner')
        model.Tie(name='Tie_Inner', main=surf_skin_in,
                  secondary=surf_core_in,
                  positionToleranceMethod=COMPUTED,
                  adjust=ON, tieRotations=ON, thickness=ON)
        print("Tie_Inner: skin inner <-> core inner face")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    model.StaticStep(name='Load', previous='Initial',
                     nlgeom=OFF, maxNumInc=200,
                     initialInc=0.1, minInc=1e-8, maxInc=0.5,
                     stabilizationMagnitude=1e-8,
                     stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
                     adaptiveDampingRatio=0.05,
                     continueDampingFactors=False)
    model.FieldOutputRequest(name='F-Output-1',
                             createStepName='Load',
                             variables=('S', 'E', 'U', 'COORD'))

    # ------------------------------------------------------------------
    # BCs — fix z=0 (clamped bottom)
    # ------------------------------------------------------------------
    # Outer skin bottom edge
    pt_out_bot = (R_OUTER * math.cos(theta_mid),
                  R_OUTER * math.sin(theta_mid), 0.0)
    e_out_bot = i_out.edges.findAt((pt_out_bot,))
    if e_out_bot:
        set_out_bot = a.Set(edges=e_out_bot, name='BC_bot_outer')
        model.DisplacementBC(
            name='Fix_Outer', createStepName='Initial',
            region=set_out_bot,
            u1=0, u2=0, u3=0, ur1=0, ur2=0, ur3=0)

    # Inner skin bottom edge
    pt_in_bot = (R_INNER * math.cos(theta_mid),
                 R_INNER * math.sin(theta_mid), 0.0)
    e_in_bot = i_in.edges.findAt((pt_in_bot,))
    if e_in_bot:
        set_in_bot = a.Set(edges=e_in_bot, name='BC_bot_inner')
        model.DisplacementBC(
            name='Fix_Inner', createStepName='Initial',
            region=set_in_bot,
            u1=0, u2=0, u3=0, ur1=0, ur2=0, ur3=0)

    # Core bottom face (z=0)
    r_core_mid = (R_CORE_O + R_CORE_I) / 2.0
    pt_core_bot = (r_core_mid * math.cos(theta_mid),
                   r_core_mid * math.sin(theta_mid), 0.0)
    f_core_bot = i_core.faces.findAt((pt_core_bot,))
    if f_core_bot:
        set_core_bot = a.Set(faces=f_core_bot, name='BC_bot_core')
        model.DisplacementBC(
            name='Fix_Core', createStepName='Initial',
            region=set_core_bot, u1=0, u2=0, u3=0)

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------
    # Axial compression on outer skin top edge (z = HEIGHT)
    pt_out_top = (R_OUTER * math.cos(theta_mid),
                  R_OUTER * math.sin(theta_mid), HEIGHT)
    e_out_top = i_out.edges.findAt((pt_out_top,))
    if e_out_top:
        top_surf = a.Surface(side1Edges=e_out_top, name='Top_Edge')
        model.ShellEdgeLoad(
            name='Axial_Comp', createStepName='Load',
            region=top_surf, magnitude=-50.0,
            directionVector=((0, 0, 0), (0, 0, 1)),
            distributionType=UNIFORM, localCsys=None)

    # External pressure on outer skin face (Aerodynamic Max Q)
    # 30-40 kPa typical for Epsilon/H3
    pt_out_face = (R_OUTER * math.cos(theta_mid),
                   R_OUTER * math.sin(theta_mid), z_mid)
    f_out = i_out.faces.findAt((pt_out_face,))
    if f_out:
        press_surf = a.Surface(side1Faces=f_out, name='Press_Surf')
        model.Pressure(name='ExtPressure', createStepName='Load',
                       region=press_surf, magnitude=0.03, # 30 kPa
                       distributionType=UNIFORM)

    return model

# =========================================================================
# JAXA Specific Load Implementations
# =========================================================================

def apply_acoustic_load(model, magnitude_db=147.0):
    """
    Applies random acoustic pressure field simulating launch environment.
    Reference: JAXA Epsilon Manual (147 dB OASPL).
    P_rms = 20e-6 * 10^(dB/20)  (Pa)
    147 dB -> ~4500 Pa RMS
    """
    # Create Dynamic Step
    model.ExplicitDynamicsStep(
        name='Acoustic_Step', previous='Load',
        timePeriod=0.05,  # Short duration for demo
        maxIncrement=1e-5
    )
    
    # Calculate Pressure Amplitude (Simplified sinusoidal for demo)
    p_rms = 20e-6 * (10**(magnitude_db/20.0))
    p_peak = p_rms * 1.414  # Sqrt(2)
    
    # Apply as periodic pressure on outer skin
    # In reality, this should be a PSD (Random Response), but using 
    # explicit time-domain pressure for direct integration.
    a = model.rootAssembly
    s_name = 'Press_Surf' # Re-use surface from static load
    if s_name in a.surfaces:
        region = a.surfaces[s_name]
        # Create Amplitude (1000 Hz tone for acoustic center freq)
        model.TabularAmplitude(
            name='Ac_Amp', timeSpan=STEP,
            smooth=SOLVER_DEFAULT,
            data=((0.0, 0.0), (0.00025, 1.0), (0.0005, 0.0), (0.00075, -1.0), (0.001, 0.0)) # 1kHz approx
        )
        model.Pressure(
            name='Acoustic_Press', createStepName='Acoustic_Step',
            region=region, magnitude=p_peak/1e6, # Pa -> MPa
            amplitude='Ac_Amp', distributionType=UNIFORM
        )
    print("Applied Acoustic Load: %f dB (%.4f MPa Peak)" % (magnitude_db, p_peak/1e6))

def apply_separation_shock(model, peak_g=2000.0):
    """
    Applies impulsive shock load simulating fairing separation (Notched Bolt).
    Reference: High frequency, high G (>1000G), short duration (<10ms).
    """
    # Create Shock Step (if not already existing, or subsequent)
    step_name = 'Shock_Step'
    if 'Acoustic_Step' in model.steps:
        prev = 'Acoustic_Step'
    else:
        prev = 'Load'
        
    model.ExplicitDynamicsStep(
        name=step_name, previous=prev,
        timePeriod=0.01, # 10ms
        maxIncrement=1e-6
    )
    
    # Apply acceleration to the base
    a = model.rootAssembly
    if 'BC_bot_outer' in a.sets:
        region = a.sets['BC_bot_outer']
        
        # Shock Amplitude: Half-sine pulse, 1ms duration
        model.TabularAmplitude(
            name='Shock_Amp', timeSpan=STEP,
            smooth=SOLVER_DEFAULT,
            data=((0.0, 0.0), (0.0005, 1.0), (0.001, 0.0), (0.01, 0.0))
        )
        
        # 2000G -> ~2000 * 9800 mm/s^2 = 1.96e7 mm/s^2
        acc_mag = peak_g * 9800.0
        
        model.BoundaryCondition(
            name='Shock_Base_Excitation', createStepName=step_name,
            region=region, type=ACCELERATION,
            a1=acc_mag, a2=acc_mag, a3=acc_mag, # Omni-directional shock
            amplitude='Shock_Amp'
        )
    print("Applied Separation Shock: %.1f G" % peak_g)

def introduce_debonding(model, location_z, size_r):
    """
    Simulates Skin-Core Debonding (JAXA primary defect target).
    Removes the 'Tie' constraint in a specific region.
    """
    # This requires partitioning the surface or defining a specific region 
    # where the tie is NOT applied. 
    # For simplicity in this script, we print the strategy.
    print("To implement Debonding: Define a surface region at z=%f with radius %f and exclude it from the Tie constraint." % (location_z, size_r))
    # Actual implementation would require re-defining the Tie surfaces.

# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    model = build_model()

    # Phase 1: Static analysis only (axial compression + external pressure)
    # Acoustic and shock loads require Abaqus/Explicit — add after
    # static validation succeeds.
    #
    # TODO Phase 2: apply_acoustic_load(model, magnitude_db=147.0)
    # TODO Phase 3: apply_separation_shock(model, peak_g=1500.0)

    # Create Job
    mdb.Job(name=JOB_NAME, model=MODEL_NAME,
            description='Fairing Sandwich — static validation',
            numCpus=4, numDomains=4,
            multiprocessingMode=DEFAULT)

    print("\nSubmitting job '%s'..." % JOB_NAME)
    mdb.jobs[JOB_NAME].submit(consistencyChecking=OFF)
    mdb.jobs[JOB_NAME].waitForCompletion()

    # Check status
    if mdb.jobs[JOB_NAME].status == COMPLETED:
        print("\n=== Job COMPLETED ===")
        print("ODB: %s.odb" % JOB_NAME)
    else:
        print("\n=== Job FAILED (status: %s) ===" % mdb.jobs[JOB_NAME].status)
        print("Check %s.dat and %s.msg for details" % (JOB_NAME, JOB_NAME))
