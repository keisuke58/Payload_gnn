# -*- coding: utf-8 -*-
"""
Payload Fairing Dataset Generation — H3 Rocket (CFRP/Al-Honeycomb)

The H3 Launch Vehicle is the first JAXA rocket to adopt a CFRP Skin / Aluminum Honeycomb Core
fairing structure, replacing the Aluminum structures used in H-IIA/B and Epsilon.

Three-part sandwich:
  Outer Facesheet  (S4R shell)   CFRP Toray T1000G [45/0/-45/90]s
  Honeycomb Core   (C3D8R solid) Al-5052 Honeycomb (Orthotropic)
  Inner Facesheet  (S4R shell)   CFRP Toray T1000G [45/0/-45/90]s
Connected via Tie Constraints.

Supports debonding injection: partial Tie removal on the outer skin-core interface.

Usage:
  cd abaqus_work

  # Healthy baseline
  abaqus cae noGUI=../src/generate_fairing_dataset.py

  # With debonding (theta_deg, z_center_mm, radius_mm)
  abaqus cae noGUI=../src/generate_fairing_dataset.py -- --defect 30.0 2500.0 150.0

  # From JSON parameter file
  abaqus cae noGUI=../src/generate_fairing_dataset.py -- --param_file params.json
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
import json

# =========================================================================
# Configuration
# =========================================================================
MODEL_NAME = 'H3_Fairing_CFRP'
JOB_NAME = 'H3_Healthy'

# JAXA H3 Fairing Dimensions (Target for CFRP Study)
# Diameter: 5.2 m (Type-S/L) -> Radius = 2600 mm
# Length: 10.4 m (S) / 16.4 m (L)
# For local SHM simulation, we model a representative section: Barrel + Ogive
RADIUS = 2600.0          # mm  (Base Radius)
H_BARREL = 5000.0        # mm  (Cylindrical Section)
H_NOSE = 5400.0          # mm  (Ogive Section - Type-S approx)
HEIGHT = H_BARREL + H_NOSE # Total Height
ANGLE = 60.0             # 1/6 section (Symmetry)

# Ogive Geometry Calculation (Tangent Ogive Approximation)
# Center of curvature (xc, zc) relative to the start of the ogive (z=H_BARREL)
# The center is on the line z=H_BARREL (horizontal line from transition point)
# because the tangent at transition is vertical.
# Formula: xc = (R^2 - H_nose^2) / (2*R)
# rho (Radius of curvature) = R - xc
OGIVE_XC = (RADIUS**2 - H_NOSE**2) / (2 * RADIUS)
OGIVE_RHO = RADIUS - OGIVE_XC
# Note: xc is typically negative for an ogive.
# Center in Global Coords: (OGIVE_XC, H_BARREL) assuming axis is x=0

# CFRP Toray T1000G (High Performance Aerospace Grade)
# Typically used in H3 Fairing Skins
E1, E2, Nu12 = 160000.0, 10000.0, 0.32  # T1000G class
G12, G13, G23 = 5500.0, 5500.0, 3200.0

LAYUP = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
PLY_T = 0.125                   # Thinner, high-performance ply
FACE_T = PLY_T * len(LAYUP)     # 1.0 mm

# Honeycomb core — orthotropic (HexWeb CR-III 3/16-5052-.001)
# Panel total ~40 mm (KHI public data) - 2 x 1.0 mm skins = 38 mm core
CORE_T = 38.0
CORE_E = {
    'E1': 10.0, 'E2': 10.0, 'E3': 1380.0,
    'Nu12': 0.001, 'Nu13': 0.001, 'Nu23': 0.001,
    'G12': 3.85, 'G13': 310.0, 'G23': 180.0,
}

MESH_SHELL = 50.0        # mm
MESH_CORE  = 50.0

# =========================================================================
# Thermal Environment — JAXA H3 Ascent
# Ref: H3 User's Manual
# =========================================================================
T_REF          = 25.0       # deg C
T_OUTER_SKIN   = 150.0      # deg C — Aerodynamic heating
T_INNER_SKIN   = 50.0       # deg C

# Radii
R_OUTER   = RADIUS
R_CORE_O  = RADIUS - FACE_T / 2.0
R_CORE_I  = RADIUS - FACE_T / 2.0 - CORE_T
R_INNER   = RADIUS - FACE_T - CORE_T

# =========================================================================
# Helper — Geometry
# =========================================================================
def get_radius_at_z(z):
    """
    Calculate the outer radius of the fairing at a given height z.
    """
    if z <= H_BARREL:
        return RADIUS
    else:
        # Ogive section
        # Equation: (x - xc)^2 + (z - H_BARREL)^2 = rho^2
        # x = xc + sqrt(rho^2 - (z - H_BARREL)^2)
        # Note: x is the radius. xc is typically negative.
        dz = z - H_BARREL
        term = OGIVE_RHO**2 - dz**2
        if term < 0:
            return 0.0 # Should not happen within H_NOSE
        return OGIVE_XC + math.sqrt(term)

def get_ogive_tip_z(radius_offset):
    """
    Calculate the Z coordinate where an offset surface (e.g. inner skin)
    intersects the axis (radius=0).
    Offset surface radius: rho_offset = OGIVE_RHO - radius_offset
    Equation: (0 - xc)^2 + (z - H_BARREL)^2 = rho_offset^2
    z = H_BARREL + sqrt(rho_offset^2 - xc^2)
    """
    rho_eff = OGIVE_RHO - radius_offset
    term = rho_eff**2 - OGIVE_XC**2
    if term < 0:
        # Does not reach axis (hole at top)
        return None
    return H_BARREL + math.sqrt(term)

# =========================================================================
# Debonding — Geometry Partitioning
# =========================================================================
def partition_debonding_zone(p_core, defect_params):
    """
    Partition the core solid to isolate the debonding zone.

    Uses 4 Datum Planes (2 theta-cuts + 2 z-cuts).
    """
    theta_c = math.radians(defect_params['theta_deg'])
    z_c = defect_params['z_center']
    r_def = defect_params['radius']

    # Local radius at the defect center (Outer surface of core ~ Outer Skin Radius)
    # The debonding is at the Skin-Core interface.
    # The radius of that interface is approx get_radius_at_z(z_c) - FACE_T/2.0
    # Or simply get_radius_at_z(z_c) is close enough for angular width calc.
    r_local = get_radius_at_z(z_c)
    
    # Convert radius to angular and axial half-extents
    if r_local > 1.0:
        d_theta = r_def / r_local
    else:
        d_theta = math.radians(10.0) # Fallback if near tip

    d_z = r_def                      # half-height (mm)

    theta1 = theta_c - d_theta
    theta2 = theta_c + d_theta
    z1 = z_c - d_z
    z2 = z_c + d_z

    # Clamp to model bounds
    theta_min = math.radians(1.0)
    theta_max = math.radians(ANGLE - 1.0)
    theta1 = max(theta1, theta_min)
    theta2 = min(theta2, theta_max)
    z1 = max(z1, 1.0)
    z2 = min(z2, HEIGHT - 1.0)

    print("Debonding partition: theta=[%.2f, %.2f] deg, z=[%.1f, %.1f] mm" %
          (math.degrees(theta1), math.degrees(theta2), z1, z2))

    # --- Theta-direction cuts (radial planes through z-axis) ---
    # We can use fixed R_CORE_I/O for defining the plane orientation
    for theta in [theta1, theta2]:
        ct, st = math.cos(theta), math.sin(theta)
        p1 = (R_CORE_I * ct, R_CORE_I * st, 0.0)
        p2 = (R_CORE_O * ct, R_CORE_O * st, 0.0)
        p3 = (R_CORE_O * ct, R_CORE_O * st, HEIGHT)
        datum = p_core.DatumPlaneByThreePoints(
            point1=p1, point2=p2, point3=p3)
        try:
            p_core.PartitionCellByDatumPlane(
                datumPlane=p_core.datums[datum.id],
                cells=p_core.cells)
        except Exception as e:
            print("Warning: theta-cut at %.2f deg failed: %s" %
                  (math.degrees(theta), str(e)))

    # --- Z-direction cuts (XY planes) ---
    for z in [z1, z2]:
        datum = p_core.DatumPlaneByPrincipalPlane(
            principalPlane=XYPLANE, offset=z)
        try:
            p_core.PartitionCellByDatumPlane(
                datumPlane=p_core.datums[datum.id],
                cells=p_core.cells)
        except Exception as e:
            print("Warning: z-cut at %.1f mm failed: %s" % (z, str(e)))

    print("Core partitioned: %d cells" % len(p_core.cells))


def _apply_debonding_outer(model, a, i_out, i_core, surf_skin_out,
                           defect_params):
    """
    Apply selective Tie on the outer interface with debonding.

    Bonded region: Tie constraint (same as healthy).
    Debonded region: Frictionless hard contact (prevents penetration).
    """
    theta_c = math.radians(defect_params['theta_deg'])
    z_c = defect_params['z_center']
    r_def = defect_params['radius']
    
    # Local radius for angular extent
    r_local_core = get_radius_at_z(z_c) - FACE_T / 2.0
    if r_local_core > 1.0:
        d_theta = r_def / r_local_core
    else:
        d_theta = math.radians(10.0)

    d_z = r_def

    tol_r = 5.0 # Increased tolerance for curvature approximation
    bonded_faces = None
    debonded_faces = None
    n_bonded = 0
    n_debonded = 0

    for f in i_core.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0]**2 + pt[1]**2)
        z = pt[2]
        
        # Check if face is on the outer surface of the core
        r_target = get_radius_at_z(z) - FACE_T / 2.0
        
        if abs(r - r_target) > tol_r:
            continue

        idx = f.index
        theta = math.atan2(pt[1], pt[0])
        # z is already obtained
        
        # Normalize theta to [0, 2pi] if needed, but our model is [0, 60 deg]
        # atan2 returns [-pi, pi]. Since we are in 1st quadrant (mostly), it's fine.
        
        in_theta = (theta_c - d_theta) <= theta <= (theta_c + d_theta)
        in_z = (z_c - d_z) <= z <= (z_c + d_z)

        if in_theta and in_z:
            if debonded_faces is None:
                debonded_faces = i_core.faces[idx:idx+1]
            else:
                debonded_faces += i_core.faces[idx:idx+1]
            n_debonded += 1
        else:
            if bonded_faces is None:
                bonded_faces = i_core.faces[idx:idx+1]
            else:
                bonded_faces += i_core.faces[idx:idx+1]
            n_bonded += 1

    print("Outer interface: %d bonded faces, %d debonded faces" %
          (n_bonded, n_debonded))

    # --- Bonded surface: Tie constraint ---
    if bonded_faces is not None:
        surf_core_bonded = a.Surface(
            side1Faces=bonded_faces, name='S_CoreOuter_Bonded')
        model.Tie(name='Tie_Outer', main=surf_skin_out,
                  secondary=surf_core_bonded,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)

    # --- Debonded surface: Frictionless hard contact ---
    if debonded_faces is not None:
        surf_core_debond = a.Surface(
            side1Faces=debonded_faces, name='S_CoreOuter_Debond')

        model.ContactProperty('Debond_IntProp')
        model.interactionProperties['Debond_IntProp'].NormalBehavior(
            pressureOverclosure=HARD, allowSeparation=ON,
            constraintEnforcementMethod=DEFAULT)
        model.interactionProperties['Debond_IntProp'].TangentialBehavior(
            formulation=FRICTIONLESS)

        model.SurfaceToSurfaceContactStd(
            name='Debond_Contact', createStepName='Load',
            main=surf_skin_out, secondary=surf_core_debond,
            sliding=SMALL, interactionProperty='Debond_IntProp',
            enforcement=SURFACE_TO_SURFACE)


# =========================================================================
# CLI argument parsing
# =========================================================================
def parse_defect_args():
    """
    Parse command-line arguments for defect parameters.

    Returns:
        defect_params: dict or None
        job_name: str
    """
    args = sys.argv[1:]

    # Find arguments after Abaqus '--' separator
    try:
        sep_idx = args.index('--')
        args = args[sep_idx + 1:]
    except ValueError:
        pass

    defect_params = None
    job_name = JOB_NAME

    i = 0
    while i < len(args):
        if args[i] == '--defect' and i + 3 < len(args):
            defect_params = {
                'theta_deg': float(args[i + 1]),
                'z_center': float(args[i + 2]),
                'radius': float(args[i + 3]),
            }
            i += 4
        elif args[i] == '--param_file' and i + 1 < len(args):
            with open(args[i + 1], 'r') as f:
                defect_params = json.load(f)
            i += 2
        elif args[i] == '--job_name' and i + 1 < len(args):
            job_name = args[i + 1]
            i += 2
        else:
            i += 1

    return defect_params, job_name


# =========================================================================
# Build model
# =========================================================================
def build_model(defect_params=None):
    """
    Build the H3 fairing FEM model.

    Args:
        defect_params: dict with debonding parameters, or None for healthy.
            Keys: 'theta_deg' (float), 'z_center' (float), 'radius' (float)
    """
    # Clean
    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]
    model = mdb.Model(name=MODEL_NAME)

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    mat = model.Material(name='CFRP_T1000G')
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
            material='CFRP_T1000G', thickness=PLY_T,
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
    # Helper to draw ogive profile
    def draw_profile(sk, r_base, z_tip, is_closed=False, r_inner=None, z_tip_inner=None):
        # Barrel line
        sk.Line(point1=(r_base, 0.0), point2=(r_base, H_BARREL))
        
        # Ogive arc (Outer/Main)
        # Center: (OGIVE_XC, H_BARREL)
        # Start: (r_base, H_BARREL)
        # End: (0.0, z_tip)
        sk.ArcByCenterEnds(center=(OGIVE_XC, H_BARREL),
                           point1=(r_base, H_BARREL),
                           point2=(0.0, z_tip),
                           direction=COUNTERCLOCKWISE)
        
        if is_closed:
            # For Core Solid: Close the loop
            # Top Line on Axis
            sk.Line(point1=(0.0, z_tip), point2=(0.0, z_tip_inner))
            
            # Inner Arc (Downwards)
            # Center: (OGIVE_XC, H_BARREL)
            # Start: (0.0, z_tip_inner)
            # End: (r_inner, H_BARREL)
            sk.ArcByCenterEnds(center=(OGIVE_XC, H_BARREL),
                               point1=(0.0, z_tip_inner),
                               point2=(r_inner, H_BARREL),
                               direction=CLOCKWISE)
            
            # Inner Barrel Line
            sk.Line(point1=(r_inner, H_BARREL), point2=(r_inner, 0.0))
            
            # Bottom Line
            sk.Line(point1=(r_inner, 0.0), point2=(r_base, 0.0))

    # Outer skin (shell)
    sk = model.ConstrainedSketch(name='sk_outer', sheetSize=20000.0)
    # R_OUTER is RADIUS
    z_tip_out = H_BARREL + H_NOSE
    draw_profile(sk, R_OUTER, z_tip_out, is_closed=False)
    
    p_out = model.Part(name='Skin_Outer', dimensionality=THREE_D,
                       type=DEFORMABLE_BODY)
    p_out.BaseShellRevolve(sketch=sk, angle=ANGLE, flipRevolveDirection=OFF)

    # Inner skin (shell)
    # Calculate tip Z for inner skin (offset by FACE_T + CORE_T)
    offset_in = FACE_T + CORE_T
    z_tip_in = get_ogive_tip_z(offset_in)
    if z_tip_in is None:
        z_tip_in = H_BARREL # Fallback
        
    sk2 = model.ConstrainedSketch(name='sk_inner', sheetSize=20000.0)
    draw_profile(sk2, R_INNER, z_tip_in, is_closed=False)
    
    p_in = model.Part(name='Skin_Inner', dimensionality=THREE_D,
                      type=DEFORMABLE_BODY)
    p_in.BaseShellRevolve(sketch=sk2, angle=ANGLE, flipRevolveDirection=OFF)

    # Core (solid)
    # Outer Offset: FACE_T/2.0
    # Inner Offset: FACE_T/2.0 + CORE_T
    offset_c_out = FACE_T / 2.0
    offset_c_in = offset_c_out + CORE_T
    
    z_tip_c_out = get_ogive_tip_z(offset_c_out)
    z_tip_c_in = get_ogive_tip_z(offset_c_in)
    
    sk3 = model.ConstrainedSketch(name='sk_core', sheetSize=20000.0)
    draw_profile(sk3, R_CORE_O, z_tip_c_out, is_closed=True,
                 r_inner=R_CORE_I, z_tip_inner=z_tip_c_in)

    p_core = model.Part(name='Core', dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidRevolve(sketch=sk3, angle=ANGLE, flipRevolveDirection=OFF)

    # ------------------------------------------------------------------
    # Debonding partition (before meshing, after part creation)
    # ------------------------------------------------------------------
    if defect_params:
        partition_debonding_zone(p_core, defect_params)

    # ------------------------------------------------------------------
    # Section assignment (works on all cells, including partitioned)
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
    # Mesh
    # ------------------------------------------------------------------
    for p_s in (p_out, p_in):
        p_s.seedPart(size=MESH_SHELL, deviationFactor=0.1, minSizeFactor=0.1)
        et = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD,
                           secondOrderAccuracy=OFF, hourglassControl=DEFAULT)
        et3 = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
        p_s.setElementType(regions=p_s.sets['AllFaces'], elemTypes=(et, et3))
        p_s.generateMesh()

    p_core.seedPart(size=MESH_CORE, deviationFactor=0.1, minSizeFactor=0.1)
    # Seed through-thickness: 1 element for all radial edges (~CORE_T length)
    for e in p_core.edges:
        verts = e.getVertices()
        if len(verts) == 2:
            v0 = p_core.vertices[verts[0]]
            v1 = p_core.vertices[verts[1]]
            p0 = v0.pointOn[0]
            p1 = v1.pointOn[0]
            length = math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 +
                                (p0[2]-p1[2])**2)
            if abs(length - CORE_T) < 5.0:
                p_core.seedEdgeByNumber(edges=(e,), number=1)

    et_hex = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD,
                           kinematicSplit=AVERAGE_STRAIN, hourglassControl=ENHANCED)
    et_wed = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    et_tet = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
    p_core.setElementType(regions=p_core.sets['AllCells'],
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
    # Step (must be defined before contact interactions)
    # ------------------------------------------------------------------
    model.StaticStep(name='Load', previous='Initial',
                     nlgeom=OFF, maxNumInc=200,
                     initialInc=0.1, minInc=1e-8, maxInc=0.5)
    model.FieldOutputRequest(name='F-Output-1',
                             createStepName='Load',
                             variables=('S', 'E', 'U', 'COORD'))

    # ------------------------------------------------------------------
    # Tie constraints (with debonding support)
    # ------------------------------------------------------------------
    theta_mid = math.radians(ANGLE / 2.0)
    z_mid = HEIGHT / 2.0
    tol_r = 2.0

    # --- Inner interface: always fully tied ---
    surf_skin_in = a.Surface(side1Faces=i_in.faces, name='S_SkinInner')
    inner_faces = None
    for f in i_core.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0]**2 + pt[1]**2)
        if abs(r - R_CORE_I) < tol_r:
            idx = f.index
            if inner_faces is None:
                inner_faces = i_core.faces[idx:idx+1]
            else:
                inner_faces += i_core.faces[idx:idx+1]
    if inner_faces is not None:
        surf_core_in = a.Surface(
            side1Faces=inner_faces, name='S_CoreInner')
        model.Tie(name='Tie_Inner', main=surf_skin_in,
                  secondary=surf_core_in,
                  positionToleranceMethod=COMPUTED, adjust=ON,
                  tieRotations=ON, thickness=ON)
        print("Tie_Inner: %d faces" % len(inner_faces))

    # --- Outer interface: selective tie if debonding ---
    surf_skin_out = a.Surface(side1Faces=i_out.faces, name='S_SkinOuter')

    if defect_params:
        _apply_debonding_outer(model, a, i_out, i_core, surf_skin_out,
                               defect_params)
    else:
        # Healthy: full tie on outer interface
        outer_faces = None
        for f in i_core.faces:
            pt = f.pointOn[0]
            r = math.sqrt(pt[0]**2 + pt[1]**2)
            if abs(r - R_CORE_O) < tol_r:
                idx = f.index
                if outer_faces is None:
                    outer_faces = i_core.faces[idx:idx+1]
                else:
                    outer_faces += i_core.faces[idx:idx+1]
        if outer_faces is not None:
            surf_core_out = a.Surface(
                side1Faces=outer_faces, name='S_CoreOuter')
            model.Tie(name='Tie_Outer', main=surf_skin_out,
                      secondary=surf_core_out,
                      positionToleranceMethod=COMPUTED, adjust=ON,
                      tieRotations=ON, thickness=ON)
            print("Tie_Outer (healthy): %d faces" % len(outer_faces))

    # ------------------------------------------------------------------
    # BCs — fix z=0 (clamped bottom)
    # ------------------------------------------------------------------
    # Outer skin bottom edge
    pt_out_bot = (R_OUTER * math.cos(theta_mid), R_OUTER * math.sin(theta_mid), 0.0)
    e_out_bot = i_out.edges.findAt((pt_out_bot,))
    if e_out_bot:
        set_out_bot = a.Set(edges=e_out_bot, name='BC_bot_outer')
        model.DisplacementBC(name='Fix_Outer', createStepName='Initial',
                             region=set_out_bot, u1=0, u2=0, u3=0, ur1=0, ur2=0, ur3=0)

    # Inner skin bottom edge
    pt_in_bot = (R_INNER * math.cos(theta_mid), R_INNER * math.sin(theta_mid), 0.0)
    e_in_bot = i_in.edges.findAt((pt_in_bot,))
    if e_in_bot:
        set_in_bot = a.Set(edges=e_in_bot, name='BC_bot_inner')
        model.DisplacementBC(name='Fix_Inner', createStepName='Initial',
                             region=set_in_bot, u1=0, u2=0, u3=0, ur1=0, ur2=0, ur3=0)

    # Core bottom face(s) (z=0) — may be multiple after partitioning
    core_bot_faces = None
    for f in i_core.faces:
        pt = f.pointOn[0]
        if abs(pt[2]) < 1.0:
            idx = f.index
            if core_bot_faces is None:
                core_bot_faces = i_core.faces[idx:idx+1]
            else:
                core_bot_faces += i_core.faces[idx:idx+1]
    if core_bot_faces is not None:
        set_core_bot = a.Set(faces=core_bot_faces, name='BC_bot_core')
        model.DisplacementBC(name='Fix_Core', createStepName='Initial',
                             region=set_core_bot, u1=0, u2=0, u3=0)

    # ------------------------------------------------------------------
    # BCs — Symmetry on θ=0° and θ=60° edges (1/6 cyclic sector)
    # In cylindrical CSYS: Uθ=0 constrains circumferential displacement
    # ------------------------------------------------------------------
    cyl_csys = a.DatumCsysByThreePoints(
        name='CylCS_Assy', coordSysType=CYLINDRICAL,
        origin=(0, 0, 0), point1=(1, 0, 0), point2=(0, 1, 0))
    cyl_datum = a.datums[cyl_csys.id]

    # --- θ=0° edges (Y=0 plane) ---
    pt_out_t0 = (R_OUTER, 0.0, z_mid)
    e_out_t0 = i_out.edges.findAt((pt_out_t0,))
    if e_out_t0:
        set_out_t0 = a.Set(edges=e_out_t0, name='Sym_t0_outer')
        model.DisplacementBC(name='Sym_t0_Outer', createStepName='Initial',
                             region=set_out_t0, u2=0, ur1=0, ur3=0,
                             localCsys=cyl_datum)

    pt_in_t0 = (R_INNER, 0.0, z_mid)
    e_in_t0 = i_in.edges.findAt((pt_in_t0,))
    if e_in_t0:
        set_in_t0 = a.Set(edges=e_in_t0, name='Sym_t0_inner')
        model.DisplacementBC(name='Sym_t0_Inner', createStepName='Initial',
                             region=set_in_t0, u2=0, ur1=0, ur3=0,
                             localCsys=cyl_datum)

    # Core θ=0 faces (Y≈0 plane) — may be multiple after partitioning
    core_t0_faces = None
    for f in i_core.faces:
        pt = f.pointOn[0]
        if abs(pt[1]) < 1.0:  # Y≈0 → θ≈0
            idx = f.index
            if core_t0_faces is None:
                core_t0_faces = i_core.faces[idx:idx+1]
            else:
                core_t0_faces += i_core.faces[idx:idx+1]
    if core_t0_faces is not None:
        set_core_t0 = a.Set(faces=core_t0_faces, name='Sym_t0_core')
        model.DisplacementBC(name='Sym_t0_Core', createStepName='Initial',
                             region=set_core_t0, u2=0,
                             localCsys=cyl_datum)

    # --- θ=ANGLE edges ---
    cos_a = math.cos(math.radians(ANGLE))
    sin_a = math.sin(math.radians(ANGLE))

    pt_out_ta = (R_OUTER * cos_a, R_OUTER * sin_a, z_mid)
    e_out_ta = i_out.edges.findAt((pt_out_ta,))
    if e_out_ta:
        set_out_ta = a.Set(edges=e_out_ta, name='Sym_ta_outer')
        model.DisplacementBC(name='Sym_ta_Outer', createStepName='Initial',
                             region=set_out_ta, u2=0, ur1=0, ur3=0,
                             localCsys=cyl_datum)

    pt_in_ta = (R_INNER * cos_a, R_INNER * sin_a, z_mid)
    e_in_ta = i_in.edges.findAt((pt_in_ta,))
    if e_in_ta:
        set_in_ta = a.Set(edges=e_in_ta, name='Sym_ta_inner')
        model.DisplacementBC(name='Sym_ta_Inner', createStepName='Initial',
                             region=set_in_ta, u2=0, ur1=0, ur3=0,
                             localCsys=cyl_datum)

    # Core θ=ANGLE faces — may be multiple after partitioning
    core_ta_faces = None
    for f in i_core.faces:
        pt = f.pointOn[0]
        r = math.sqrt(pt[0]**2 + pt[1]**2)
        if r > 1.0:  # exclude origin-adjacent
            theta_f = math.atan2(pt[1], pt[0])
            if abs(theta_f - math.radians(ANGLE)) < math.radians(1.0):
                idx = f.index
                if core_ta_faces is None:
                    core_ta_faces = i_core.faces[idx:idx+1]
                else:
                    core_ta_faces += i_core.faces[idx:idx+1]
    if core_ta_faces is not None:
        set_core_ta = a.Set(faces=core_ta_faces, name='Sym_ta_core')
        model.DisplacementBC(name='Sym_ta_Core', createStepName='Initial',
                             region=set_core_ta, u2=0,
                             localCsys=cyl_datum)

    print("Symmetry BCs applied on theta=0 and theta=%.0f deg edges" % ANGLE)

    # ------------------------------------------------------------------
    # Loads — H3 Max Q external pressure
    # ------------------------------------------------------------------
    pt_out_face = (R_OUTER * math.cos(theta_mid), R_OUTER * math.sin(theta_mid), z_mid)
    f_out = i_out.faces.findAt((pt_out_face,))
    if f_out:
        press_surf = a.Surface(side1Faces=f_out, name='Press_Surf')
        model.Pressure(name='ExtPressure', createStepName='Load',
                       region=press_surf, magnitude=0.05,  # 50 kPa
                       distributionType=UNIFORM)

    return model

# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    defect_params, job_name = parse_defect_args()

    if defect_params:
        print("\n=== Debonding Mode ===")
        print("  theta = %.2f deg" % defect_params['theta_deg'])
        print("  z     = %.1f mm" % defect_params['z_center'])
        print("  radius= %.1f mm" % defect_params['radius'])
    else:
        print("\n=== Healthy Baseline Mode ===")

    model = build_model(defect_params=defect_params)

    desc = 'H3 CFRP/Al-HC Fairing'
    if defect_params:
        desc += ' — debonding theta=%.1f z=%.0f r=%.0f' % (
            defect_params['theta_deg'], defect_params['z_center'],
            defect_params['radius'])
    else:
        desc += ' — healthy baseline'

    mdb.Job(name=job_name, model=MODEL_NAME,
            description=desc,
            numCpus=4, numDomains=4,
            multiprocessingMode=DEFAULT)

    print("\nSubmitting job '%s'..." % job_name)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()

    if mdb.jobs[job_name].status == COMPLETED:
        print("\n=== Job COMPLETED ===")
        print("ODB: %s.odb" % job_name)
    else:
        print("\n=== Job FAILED (status: %s) ===" % mdb.jobs[job_name].status)
        print("Check %s.dat and %s.msg for details" % (job_name, job_name))
