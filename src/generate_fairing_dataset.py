# -*- coding: utf-8 -*-
"""
Payload Fairing Dataset Generation — Shell-Solid-Shell Sandwich Model

Models the fairing as a 3-part sandwich:
  - Outer Facesheet : CFRP [45/0/-45/90]s shell (S4R)
  - Honeycomb Core  : Orthotropic solid (C3D8R), offset inward
  - Inner Facesheet : CFRP [45/0/-45/90]s shell (S4R), offset inward

Connected via Tie Constraints (Skin-Core coupling).
Disbond defects are modeled by deactivating Tie regions.

Usage: abaqus cae noGUI=src/generate_fairing_dataset.py
"""

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import section
import math

# =========================================================================
# Configuration
# =========================================================================
MODEL_NAME = 'Fairing_Sandwich'
RADIUS = 2000.0          # mm  (fairing mid-surface radius)
HEIGHT = 5000.0          # mm
ANGLE = 60.0             # degrees — 1/6 section

# Material: CFRP T300/914
E1, E2, Nu12 = 135000.0, 10000.0, 0.3
G12, G13, G23 = 5000.0, 5000.0, 3000.0

# Layup [45/0/-45/90]s per facesheet
LAYUP = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
PLY_T = 0.15             # mm per ply
FACE_T = PLY_T * len(LAYUP)  # total facesheet thickness = 1.2 mm

# Honeycomb core (orthotropic homogenised)
CORE_T = 20.0            # mm
# Ribbon (L) direction along the fairing axis (z)
# Through-thickness (T) direction radial
CORE_PROPS = {
    'E1': 1.0,     'E2': 1.0,     'E3': 1380.0,   # MPa  (L=~0, W=~0, T=stiff)
    'Nu12': 0.99,  'Nu13': 0.01,  'Nu23': 0.01,
    'G12': 0.5,    'G13': 310.0,  'G23': 180.0,    # MPa
}

MESH_SIZE_SHELL = 50.0   # mm — facesheet seed
MESH_SIZE_CORE = 50.0    # mm — core seed (in-plane); 1 element through thickness


def create_sandwich_model():
    """Build the full Shell-Solid-Shell sandwich model."""

    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]
    model = mdb.Model(name=MODEL_NAME)

    # -----------------------------------------------------------------
    # 1. Materials
    # -----------------------------------------------------------------
    mat_cfrp = model.Material(name='CFRP_T300')
    mat_cfrp.Elastic(
        type=LAMINA,
        table=((E1, E2, Nu12, G12, G13, G23),))

    mat_core = model.Material(name='Honeycomb_Core')
    mat_core.Elastic(
        type=ENGINEERING_CONSTANTS,
        table=((
            CORE_PROPS['E1'], CORE_PROPS['E2'], CORE_PROPS['E3'],
            CORE_PROPS['Nu12'], CORE_PROPS['Nu13'], CORE_PROPS['Nu23'],
            CORE_PROPS['G12'], CORE_PROPS['G13'], CORE_PROPS['G23'],
        ),))

    # -----------------------------------------------------------------
    # 2. Sections
    # -----------------------------------------------------------------
    # Facesheet composite shell section
    layup_entries = []
    for i, angle in enumerate(LAYUP):
        layup_entries.append(section.SectionLayer(
            material='CFRP_T300', thickness=PLY_T,
            orientAngle=angle, numIntPts=3,
            plyName='Ply_%d' % i))

    model.CompositeShellSection(
        name='Facesheet_Section',
        preIntegrate=OFF, idealization=NO_IDEALIZATION,
        symmetric=OFF, thicknessType=UNIFORM,
        poissonDefinition=DEFAULT, temperature=GRADIENT,
        integrationRule=SIMPSON, numIntPts=3,
        layup=layup_entries)

    # Solid core section
    model.HomogeneousSolidSection(
        name='Core_Section', material='Honeycomb_Core', thickness=None)

    # -----------------------------------------------------------------
    # 3. Geometry — Outer facesheet (reference surface)
    # -----------------------------------------------------------------
    arc_end_x = RADIUS * math.cos(math.radians(ANGLE))
    arc_end_y = RADIUS * math.sin(math.radians(ANGLE))

    # --- Outer skin ---
    s_out = model.ConstrainedSketch(name='Outer_Profile', sheetSize=6000.0)
    s_out.ArcByCenterEnds(
        center=(0.0, 0.0), point1=(RADIUS, 0.0),
        point2=(arc_end_x, arc_end_y),
        direction=COUNTERCLOCKWISE)
    p_outer = model.Part(name='Skin_Outer', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_outer.BaseShellExtrude(sketch=s_out, depth=HEIGHT)

    # --- Inner skin (offset inward by FACE_T/2 + CORE_T + FACE_T/2) ---
    r_inner = RADIUS - FACE_T - CORE_T
    inner_end_x = r_inner * math.cos(math.radians(ANGLE))
    inner_end_y = r_inner * math.sin(math.radians(ANGLE))

    s_in = model.ConstrainedSketch(name='Inner_Profile', sheetSize=6000.0)
    s_in.ArcByCenterEnds(
        center=(0.0, 0.0), point1=(r_inner, 0.0),
        point2=(inner_end_x, inner_end_y),
        direction=COUNTERCLOCKWISE)
    p_inner = model.Part(name='Skin_Inner', dimensionality=THREE_D,
                         type=DEFORMABLE_BODY)
    p_inner.BaseShellExtrude(sketch=s_in, depth=HEIGHT)

    # --- Core solid (extruded between skins) ---
    # Create a 2D profile: arc at outer skin inner face -> arc at inner skin outer face
    r_core_out = RADIUS - FACE_T / 2.0
    r_core_in = RADIUS - FACE_T - CORE_T + FACE_T / 2.0

    s_core = model.ConstrainedSketch(name='Core_Profile', sheetSize=6000.0)
    # Outer arc
    s_core.ArcByCenterEnds(
        center=(0.0, 0.0),
        point1=(r_core_out, 0.0),
        point2=(r_core_out * math.cos(math.radians(ANGLE)),
                r_core_out * math.sin(math.radians(ANGLE))),
        direction=COUNTERCLOCKWISE)
    # Inner arc
    s_core.ArcByCenterEnds(
        center=(0.0, 0.0),
        point1=(r_core_in * math.cos(math.radians(ANGLE)),
                r_core_in * math.sin(math.radians(ANGLE))),
        point2=(r_core_in, 0.0),
        direction=CLOCKWISE)
    # Close with radial lines
    s_core.Line(point1=(r_core_out, 0.0), point2=(r_core_in, 0.0))
    s_core.Line(
        point1=(r_core_out * math.cos(math.radians(ANGLE)),
                r_core_out * math.sin(math.radians(ANGLE))),
        point2=(r_core_in * math.cos(math.radians(ANGLE)),
                r_core_in * math.sin(math.radians(ANGLE))))

    p_core = model.Part(name='Core', dimensionality=THREE_D,
                        type=DEFORMABLE_BODY)
    p_core.BaseSolidExtrude(sketch=s_core, depth=HEIGHT)

    # -----------------------------------------------------------------
    # 4. Section assignments
    # -----------------------------------------------------------------
    # Cylindrical datum for material orientation
    for p in (p_outer, p_inner):
        datum = p.DatumCsysByThreePoints(
            name='CylCSYS', coordSysType=CYLINDRICAL,
            origin=(0.0, 0.0, 0.0),
            point1=(1.0, 0.0, 0.0),
            point2=(0.0, 1.0, 0.0))
        region = regionToolset.Region(faces=p.faces)
        p.SectionAssignment(
            region=region, sectionName='Facesheet_Section',
            offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',
            thicknessAssignment=FROM_SECTION)
        p.MaterialOrientation(
            region=region, orientationType=SYSTEM, axis=AXIS_1,
            localCsys=p.datums[datum.id], fieldName='',
            additionalRotationType=ROTATION_NONE, angle=0.0,
            additionalRotationField='', stackDirection=STACK_3)

    # Core
    core_region = regionToolset.Region(cells=p_core.cells)
    p_core.SectionAssignment(
        region=core_region, sectionName='Core_Section',
        offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',
        thicknessAssignment=FROM_SECTION)

    core_datum = p_core.DatumCsysByThreePoints(
        name='CylCSYS_Core', coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0),
        point1=(1.0, 0.0, 0.0),
        point2=(0.0, 1.0, 0.0))
    p_core.MaterialOrientation(
        region=core_region, orientationType=SYSTEM, axis=AXIS_3,
        localCsys=p_core.datums[core_datum.id], fieldName='',
        additionalRotationType=ROTATION_NONE, angle=0.0,
        additionalRotationField='', stackDirection=STACK_3)

    # -----------------------------------------------------------------
    # 5. Mesh
    # -----------------------------------------------------------------
    for p_shell in (p_outer, p_inner):
        p_shell.seedPart(size=MESH_SIZE_SHELL, deviationFactor=0.1,
                         minSizeFactor=0.1)
        et = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD,
                           secondOrderAccuracy=OFF,
                           hourglassControl=DEFAULT)
        p_shell.setElementType(
            regions=regionToolset.Region(faces=p_shell.faces),
            elemTypes=(et,))
        p_shell.generateMesh()

    p_core.seedPart(size=MESH_SIZE_CORE, deviationFactor=0.1,
                    minSizeFactor=0.1)
    et_solid = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD,
                             kinematicSplit=AVERAGE_STRAIN,
                             hourglassControl=DEFAULT)
    et_wedge = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    et_tet = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
    p_core.setElementType(
        regions=regionToolset.Region(cells=p_core.cells),
        elemTypes=(et_solid, et_wedge, et_tet))
    p_core.generateMesh()

    # -----------------------------------------------------------------
    # 6. Assembly
    # -----------------------------------------------------------------
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst_outer = a.Instance(name='Skin_Outer-1', part=p_outer, dependent=ON)
    inst_inner = a.Instance(name='Skin_Inner-1', part=p_inner, dependent=ON)
    inst_core = a.Instance(name='Core-1', part=p_core, dependent=ON)

    # -----------------------------------------------------------------
    # 7. Tie Constraints (Skin-Core coupling)
    # -----------------------------------------------------------------
    # Outer skin -> Core outer face
    core_outer_faces = inst_core.faces.getByBoundingCylinder(
        center1=(0.0, 0.0, -1.0), center2=(0.0, 0.0, HEIGHT + 1.0),
        radius=r_core_out + 1.0)
    # Filter to only outer-radius faces (normal pointing outward)
    outer_tie_master = a.Surface(side1Faces=inst_outer.faces, name='Outer_Skin_Surf')
    outer_tie_slave = a.Surface(side1Faces=core_outer_faces, name='Core_Outer_Surf')

    model.Tie(name='Tie_Outer_Skin_Core',
              master=outer_tie_master, slave=outer_tie_slave,
              positionToleranceMethod=COMPUTED,
              adjust=ON, tieRotations=ON, thickness=ON)

    # Inner skin -> Core inner face
    core_inner_faces = inst_core.faces.getByBoundingCylinder(
        center1=(0.0, 0.0, -1.0), center2=(0.0, 0.0, HEIGHT + 1.0),
        radius=r_core_in + 1.0)
    inner_tie_master = a.Surface(side1Faces=inst_inner.faces, name='Inner_Skin_Surf')
    inner_tie_slave = a.Surface(side1Faces=core_inner_faces, name='Core_Inner_Surf')

    model.Tie(name='Tie_Inner_Skin_Core',
              master=inner_tie_master, slave=inner_tie_slave,
              positionToleranceMethod=COMPUTED,
              adjust=ON, tieRotations=ON, thickness=ON)

    # -----------------------------------------------------------------
    # 8. Step, BCs, Load
    # -----------------------------------------------------------------
    model.StaticStep(name='Load_Step', previous='Initial',
                     nlgeom=ON, maxNumInc=100,
                     initialInc=0.1, minInc=1e-8, maxInc=1.0)
    model.FieldOutputRequest(name='F-Output-1',
                             createStepName='Load_Step',
                             variables=('S', 'E', 'U', 'COORD'))

    # Fix bottom edges of all three parts (z = 0)
    tol = MESH_SIZE_SHELL
    for inst in (inst_outer, inst_inner, inst_core):
        bottom = []
        for edge in inst.edges:
            pts = edge.pointOn
            if pts and abs(pts[0][2]) < tol:
                bottom.append(edge)
        if bottom:
            bc_set = a.Set(edges=bottom, name='BC_Bottom_%s' % inst.name)
            model.DisplacementBC(
                name='Fix_%s' % inst.name,
                createStepName='Initial', region=bc_set,
                u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0)

    # Axial compression on outer skin top edge
    top_edges = []
    for edge in inst_outer.edges:
        pts = edge.pointOn
        if pts and abs(pts[0][2] - HEIGHT) < tol:
            top_edges.append(edge)
    if top_edges:
        top_surf = a.Surface(side1Edges=top_edges, name='Top_Load_Surf')
        model.ShellEdgeLoad(
            name='Axial_Compression', createStepName='Load_Step',
            region=top_surf, magnitude=-50.0,
            directionVector=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            distributionType=UNIFORM, localCsys=None)

    # External pressure on outer skin
    pressure_surf = a.Surface(side1Faces=inst_outer.faces,
                              name='Outer_Pressure_Surf')
    model.Pressure(name='Acoustic_Pressure', createStepName='Load_Step',
                   region=pressure_surf, magnitude=0.03,
                   distributionType=UNIFORM)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    n_outer = len(p_outer.nodes)
    n_inner = len(p_inner.nodes)
    n_core = len(p_core.nodes)
    print("=== Sandwich Model Created ===")
    print("  Outer Skin : %d nodes (S4R, [45/0/-45/90]s, t=%.2f mm)" % (n_outer, FACE_T))
    print("  Core       : %d nodes (C3D8R, orthotropic, t=%.1f mm)" % (n_core, CORE_T))
    print("  Inner Skin : %d nodes (S4R, [45/0/-45/90]s, t=%.2f mm)" % (n_inner, FACE_T))
    print("  Total      : %d nodes" % (n_outer + n_inner + n_core))
    print("  Ties       : Outer-Core, Inner-Core")
    print("  Loading    : Axial compression + External pressure")

    return model


if __name__ == '__main__':
    create_sandwich_model()
