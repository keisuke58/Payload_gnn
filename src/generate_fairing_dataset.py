# -*- coding: utf-8 -*-
# generate_fairing_dataset.py
# Abaqus Python script to generate H3 Type-S fairing FEM model with debonding defects
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

def is_face_in_defect_zone(face, defect_params):
    """
    Checks if a face's centroid is within the defect zone.
    defect_params: {z_center, theta_deg, radius}
    Uses pointOn (reliable in Abaqus) for face position.
    """
    if not defect_params:
        return False
    
    z_c = defect_params['z_center']
    theta_deg = defect_params['theta_deg']
    r_def = defect_params['radius']
    
    # Face position — Abaqus revolve: Y=axial, XZ=radial
    pt = face.pointOn[0]
    x, y, z = pt[0], pt[1], pt[2]
    
    # Check axial (Y)
    if abs(y - z_c) > r_def:
        return False
    
    # Check Theta (radial plane XZ)
    r_local = math.sqrt(x*x + z*z)
    if r_local < 1.0: return False
    
    # Angle of centroid
    # atan2(y, x) returns (-pi, pi)
    # Our model is 0..60 deg (1/6 section).
    # theta_deg is in 0..60? The DOE generates 5..55.
    
    theta_rad_face = math.atan2(y, x)
    if theta_rad_face < 0: theta_rad_face += 2*math.pi
    theta_deg_face = math.degrees(theta_rad_face)
    
    # Arc length difference
    # defect is defined by arc distance from (z_c, theta_c) < r_def
    # approx: d^2 = (z-zc)^2 + (r*dtheta)^2 < r_def^2
    
    d_theta_deg = abs(theta_deg_face - theta_deg)
    # Handle wrap around if needed (but here we are in 0-60 sector)
    
    arc_len = r_local * math.radians(d_theta_deg)
    dist_sq = (z - z_c)**2 + arc_len**2
    
    return dist_sq < (r_def * 1.01)**2 # 1% tolerance

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

def partition_debonding_zone(model, assembly, defect_params):
    """
    Partitions the surfaces to define the debonding area.
    Defect is defined by (z_center, theta_center, radius).
    Uses getByBoundingBox to select only faces that intersect the partition region.
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
    
    z1 = max(1.0, z_c - r_def - 50)
    z2 = min(TOTAL_HEIGHT - 1.0, z_c + r_def + 50)
    r_min = max(100.0, RADIUS - 100)
    r_max = RADIUS + CORE_T + 100
    x_min = -r_max - 100
    x_max = r_max + 100
    
    inst_core = assembly.instances['Part-Core-1']
    inst_outer = assembly.instances['Part-OuterSkin-1']
    
    # Datum planes — XZPLANE gives y=constant (axial direction)
    p_z1 = assembly.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=z_c - r_def)
    p_z2 = assembly.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=z_c + r_def)
    p_t1 = assembly.DatumPlaneByThreePoints(
        point1=(0, 0, 0), point2=(0, 0, 100),
        point3=(math.cos(t1), math.sin(t1), 0))
    p_t2 = assembly.DatumPlaneByThreePoints(
        point1=(0, 0, 0), point2=(0, 0, 100),
        point3=(math.cos(t2), math.sin(t2), 0))
    
    def partition_faces_safe(inst, plane_id):
        try:
            faces = inst.faces.getByBoundingBox(
                xMin=x_min, xMax=x_max, yMin=x_min, yMax=x_max,
                zMin=z1, zMax=z2)
            if len(faces) > 0:
                assembly.PartitionFaceByDatumPlane(
                    datumPlane=assembly.datums[plane_id],
                    faces=faces)
        except Exception as e:
            print("  Partition warning: %s" % str(e)[:80])
    
    partition_faces_safe(inst_outer, p_z1.id)
    partition_faces_safe(inst_outer, p_z2.id)
    partition_faces_safe(inst_outer, p_t1.id)
    partition_faces_safe(inst_outer, p_t2.id)
    partition_faces_safe(inst_core, p_z1.id)
    partition_faces_safe(inst_core, p_z2.id)
    partition_faces_safe(inst_core, p_t1.id)
    partition_faces_safe(inst_core, p_t2.id)

def generate_model(job_name, defect_params=None, project_root=None):
    """Main function to generate the model."""
    Mdb() # Clear
    model = mdb.models['Model-1']
    
    # 1. Materials & Sections
    create_materials(model)
    create_sections(model)
    
    # 2. Parts
    p_inner, p_core, p_outer = create_parts(model)
    
    # Assign Sections
    region = p_inner.Set(faces=p_inner.faces, name='Set-All')
    p_inner.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    p_inner.MaterialOrientation(region=region, orientationType=GLOBAL, axis=AXIS_3, additionalRotationType=ROTATION_NONE, localCsys=None)
    
    region = p_core.Set(cells=p_core.cells, name='Set-All')
    p_core.SectionAssignment(region=region, sectionName='Section-Core')
    
    region = p_outer.Set(faces=p_outer.faces, name='Set-All')
    p_outer.SectionAssignment(region=region, sectionName='Section-CFRP-Skin')
    
    # 3. Assembly
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst_inner = a.Instance(name='Part-InnerSkin-1', part=p_inner, dependent=OFF)
    inst_core = a.Instance(name='Part-Core-1', part=p_core, dependent=OFF)
    inst_outer = a.Instance(name='Part-OuterSkin-1', part=p_outer, dependent=OFF)
    
    # 4. Debonding (Partitioning)
    if defect_params:
        partition_debonding_zone(model, a, defect_params)
    
    # 5. Interaction (Tie Constraints) - EXCLUDE defect zone for physical debonding
    tol_r = 100.0
    # Core radial: original formula (pointOn x,y) worked in prior runs
    core_inner_faces = [f for f in inst_core.faces
                        if abs(math.sqrt(f.pointOn[0][0]**2 + f.pointOn[0][1]**2) - RADIUS) < tol_r]
    core_outer_faces = [f for f in inst_core.faces
                        if abs(math.sqrt(f.pointOn[0][0]**2 + f.pointOn[0][1]**2) - (RADIUS + CORE_T)) < tol_r]
    # Tie exclusion (defect zone): Abaqus Surface(side1Faces=subset) can fail with partitioned geometry.
    # Using all faces for now; physical debonding requires Contact or CZM for partitioned models.
    surf_inner = a.Surface(side1Faces=inst_inner.faces, name='Surf_Inner')
    surf_outer = a.Surface(side1Faces=inst_outer.faces, name='Surf_Outer')
    if not core_inner_faces or not core_outer_faces:
        print("Warning: core faces not found (inner=%d, outer=%d)" % (len(core_inner_faces), len(core_outer_faces)))
    try:
        if core_inner_faces:
            surf_core_in = a.Surface(side1Faces=core_inner_faces, name='Surf_CoreInner')
            model.Tie(name='Tie-Inner-Core', main=surf_core_in, secondary=surf_inner,
                      positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)
        if core_outer_faces:
            surf_core_out = a.Surface(side1Faces=core_outer_faces, name='Surf_CoreOuter')
            model.Tie(name='Tie-Core-Outer', main=surf_core_out, secondary=surf_outer,
                      positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)
    except Exception as e:
        print("Tie from filtered faces failed (%s), using full core" % str(e)[:50])
        surf_core_in = a.Surface(side1Faces=inst_core.faces, name='Surf_CoreInner')
        surf_core_out = a.Surface(side1Faces=inst_core.faces, name='Surf_CoreOuter')
        model.Tie(name='Tie-Inner-Core', main=surf_core_in, secondary=surf_inner,
                  positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)
        model.Tie(name='Tie-Core-Outer', main=surf_core_out, secondary=surf_outer,
                  positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)
    
    # 6. BCs - fix bottom (y=0) for static equilibrium — Abaqus: Y=axial
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

    # 7. Step & Loads
    model.StaticStep(name='Step-1', previous='Initial')
    # Thermal load: ascent heating (outer 120°C, inner 20°C) — applied in Step-1 after mesh
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'RF', 'TEMP'))
    
    # 8. Mesh
    a.seedPartInstance(regions=(inst_inner, inst_core, inst_outer), size=GLOBAL_SEED, deviationFactor=0.1)
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
                    a.seedEdgeBySize(edges=edges, size=DEFECT_SEED, constraint=FINER)
            print("Local mesh refinement: DEFECT_SEED=%.0f mm in defect zone (z=%.0f–%.0f)" % (DEFECT_SEED, z1, z2))
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

    # Write INP, patch thermal/BC if needed, then run (submit overwrites INP, so use input=)
    print("Writing INP for job '%s'..." % job_name)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    inp_path = os.path.abspath(job_name + '.inp')
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
    
    project_root = args.project_root or os.environ.get('PROJECT_ROOT') or os.environ.get('PAYLOAD2026_ROOT')
    generate_model(job_name, defect_data, project_root=project_root)
