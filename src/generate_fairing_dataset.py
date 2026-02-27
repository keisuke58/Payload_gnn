# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
MODEL_NAME = 'Fairing_Model'
PART_NAME = 'Fairing_Panel'
RADIUS = 2000.0       # Radius of the fairing (mm) -> 4m diameter
HEIGHT = 5000.0       # Height of the fairing section (mm)
ANGLE = 60.0          # 1/6th section (360/6 = 60 degrees)

# Material Properties (T300/914 CFRP - Representative)
E1 = 135000.0   # MPa
E2 = 10000.0    # MPa
Nu12 = 0.3
G12 = 5000.0    # MPa
G13 = 5000.0
G23 = 3000.0

# Layup Sequence: Quasi-Isotropic [45/0/-45/90]s
# Note: Angles are relative to the material orientation (Axis 1 = Circumferential or Axial)
LAYUP_SEQUENCE = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]
PLY_THICKNESS = 0.15 # mm

def create_fairing_model():
    # 1. Create Model
    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]
    model = mdb.Model(name=MODEL_NAME)

    # 2. Create Part (Deformable Shell)
    # Using extrusion to create a curved shell section
    s = model.ConstrainedSketch(name='Fairing_Profile', sheetSize=5000.0)
    s.ArcByCenterEnds(center=(0.0, 0.0), point1=(RADIUS, 0.0), point2=(RADIUS * cos(radians(ANGLE)), RADIUS * sin(radians(ANGLE))), direction=COUNTERCLOCKWISE)
    
    p = model.Part(name=PART_NAME, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseShellExtrude(sketch=s, depth=HEIGHT)
    
    # 3. Define Materials
    # CFRP T300
    mat_cfrp = model.Material(name='CFRP_T300')
    mat_cfrp.Elastic(type=LAMINA, table=((E1, E2, Nu12, G12, G13, G23), ))
    
    # Honeycomb Core (Aluminum - Simplified as Isotropic for now, usually Orthotropic)
    mat_core = model.Material(name='Honeycomb_Al')
    mat_core.Elastic(table=((1000.0, 0.3), )) # Simplified low modulus core
    
    # 4. Define Composite Layup
    layup_entries = []
    
    # Facesheet 1 (Outer)
    for i, angle in enumerate(LAYUP_SEQUENCE):
        layup_entries.append(section.SectionLayer(
            material='CFRP_T300', thickness=PLY_THICKNESS, 
            orientAngle=angle, numIntPts=3, plyName='Face1_Ply_%d' % i))
            
    # Core
    layup_entries.append(section.SectionLayer(
        material='Honeycomb_Al', thickness=20.0, 
        orientAngle=0.0, numIntPts=3, plyName='Core'))
        
    # Facesheet 2 (Inner) - Symmetric
    for i, angle in enumerate(reversed(LAYUP_SEQUENCE)):
        layup_entries.append(section.SectionLayer(
            material='CFRP_T300', thickness=PLY_THICKNESS, 
            orientAngle=angle, numIntPts=3, plyName='Face2_Ply_%d' % i))

    # Create Section
    model.CompositeShellSection(
        name='Fairing_Section', preIntegrate=OFF, idealization=NO_IDEALIZATION,
        symmetric=OFF, thicknessType=UNIFORM, poissonDefinition=DEFAULT,
        temperature=GRADIENT, integrationRule=SIMPSON, numIntPts=3, layup=layup_entries)

    # Assign Section
    region = p.Set(cells=p.cells, name='All_Cells')
    p.SectionAssignment(region=region, sectionName='Fairing_Section', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

    # 5. Define Material Orientation
    # Important for curved composites!
    # Axis 1 = Circumferential (Tangent), Axis 2 = Axial (Longitudinal)
    # Using a cylindrical coordinate system
    datum_sys = p.DatumCsysByThreePoints(name='Cylindrical_CSYS', coordSysType=CYLINDRICAL,
        origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0), point2=(0.0, 1.0, 0.0))
    
    p.MaterialOrientation(region=region, orientationType=SYSTEM, axis=AXIS_1, 
        localCsys=p.datums[datum_sys.id], fieldName='', additionalRotationType=ROTATION_NONE, 
        angle=0.0, additionalRotationField='', stackDirection=STACK_3)

    # 6. Mesh
    p.seedPart(size=50.0, deviationFactor=0.1, minSizeFactor=0.1)
    elemType = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD)
    p.setElementType(regions=region, elemTypes=(elemType,))
    p.generateMesh()

    # 7. Assembly
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    a.Instance(name=PART_NAME+'-1', part=p, dependent=ON)

    print("Model '%s' created successfully with [45/0/-45/90]s layup." % MODEL_NAME)

if __name__ == '__main__':
    create_fairing_model()
