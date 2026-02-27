# -*- coding: mbcs -*-
"""
Payload Fairing Dataset Generation Script for Abaqus/Standard
This script generates a parametric 1/6th cylindrical payload fairing model
with realistic Composite Laminate Layup [0/90/45/-45].

Usage: abaqus cae noGUI=src/generate_fairing_dataset.py
"""

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import step
import job
import sketch
import part
import material
import section
import assembly
import interaction
import load
import random
import os
import math

# --- Configuration ---
PROJECT_NAME = "PayloadFairing_GNN"
OUTPUT_DIR = "dataset_output"
NUM_SAMPLES = 1  # Generate 1 sample for test

# Geometry Parameters (Units: mm)
RADIUS = 2000.0
HEIGHT = 5000.0
ANGLE = 60.0  # 1/6th section
PLY_THICKNESS = 0.125  # Standard CFRP ply thickness

# Material Properties (Carbon/Epoxy T300/914 approximation)
# E1=135GPa, E2=10GPa, Nu12=0.3, G12=5GPa
MAT_PROPS = {
    'E1': 135000.0,
    'E2': 10000.0,
    'Nu12': 0.3,
    'G12': 5000.0,
    'G13': 5000.0,
    'G23': 3000.0
}

# Layup Sequence: Quasi-Isotropic [45/0/-45/90]s
LAYUP_SEQUENCE = [45.0, 0.0, -45.0, 90.0, 90.0, -45.0, 0.0, 45.0]

def create_materials(model):
    """Defines Orthotropic CFRP Material."""
    mat = model.Material(name='CFRP_T300')
    mat.Elastic(type=LAMINA, table=((
        MAT_PROPS['E1'], MAT_PROPS['E2'], MAT_PROPS['Nu12'], 
        MAT_PROPS['G12'], MAT_PROPS['G13'], MAT_PROPS['G23']
    ), ))
    
    # Honeycomb Core Material (Simplified)
    mat_core = model.Material(name='Honeycomb_Al')
    mat_core.Elastic(table=((1000.0, 0.3), )) # Effective properties

def create_composite_section(model):
    """Defines Composite Shell Section with [0/90/45/-45] Layup."""
    
    # Define Layup
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

    model.CompositeShellSection(
        name='Fairing_Section', 
        preIntegrate=OFF, 
        idealization=NO_IDEALIZATION, 
        symmetric=OFF, 
        thicknessType=UNIFORM, 
        poissonDefinition=DEFAULT, 
        temperature=GRADIENT, 
        integrationRule=SIMPSON, 
        numIntPts=3,
        layup=layup_entries
    )

def create_geometry(model, part_name):
    """Creates the 1/6th cylindrical shell part."""
    s = model.ConstrainedSketch(name='__profile__', sheetSize=HEIGHT*2)
    
    # Create Arc
    # Note: In Abaqus Scripting, ArcByCenterEnds requires explicit coordinates
    p1 = (RADIUS, 0.0)
    p2_x = RADIUS * math.cos(math.radians(ANGLE))
    p2_y = RADIUS * math.sin(math.radians(ANGLE))
    p2 = (p2_x, p2_y)
    
    s.ArcByCenterEnds(center=(0.0, 0.0), point1=p1, point2=p2, direction=COUNTERCLOCKWISE)
    
    # Extrude
    p = model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseShellExtrude(sketch=s, depth=HEIGHT)
    return p

def assign_section_orientation(model, part):
    """Assigns Material Orientation for Composite."""
    # Define a datum coordinate system (Cylindrical)
    # This ensures 0 degree ply is along the axis or hoop direction correctly
    
    # Create Cylindrical Datum CSYS
    d = part.datums
    # origin=(0,0,0), z-axis=(0,0,1) -> Cylindrical axis is Z
    # But our extrusion is along Z (depth), so the arc is in XY plane.
    # The default extrusion direction is usually Z. 
    # Let's assume global Z is the fairing axis.
    
    # Use default global CSYS or create one
    # For a fairing, usually Axis 1 = Axial, Axis 2 = Hoop.
    
    # Assign section
    region = regionToolset.Region(faces=part.faces)
    part.SectionAssignment(region=region, sectionName='Fairing_Section', 
        offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
        
    # Assign Orientation
    # Normal Direction is crucial for shell
    part.MaterialOrientation(region=region, 
        orientationType=SYSTEM, axis=AXIS_3, 
        localCsys=None, fieldName='', 
        additionalRotationType=ROTATION_NONE, 
        angle=0.0, additionalRotationField='', stackDirection=STACK_3)

def generate_sample(sample_id):
    """Generates a single simulation sample."""
    model_name = 'Model_%d' % sample_id
    if model_name in mdb.models:
        del mdb.models[model_name]
    
    model = mdb.Model(name=model_name)
    create_materials(model)
    create_composite_section(model)
    
    part_name = 'Fairing_Panel'
    p = create_geometry(model, part_name)
    assign_section_orientation(model, p)
    
    # Assembly
    a = model.rootAssembly
    inst = a.Instance(name=part_name+'-1', part=p, dependent=ON)
    
    # Mesh
    p.seedPart(size=100.0, deviationFactor=0.1, minSizeFactor=0.1)
    # Use Shell Elements (S4R)
    elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD, 
        secondOrderAccuracy=OFF, hourglassControl=DEFAULT)
    p.setElementType(regions=regionToolset.Region(faces=p.faces), elemTypes=(elemType1,))
    p.generateMesh()
    
    # Step
    model.StaticStep(name='Load_Step', previous='Initial')
    
    # Job
    job_name = 'Job_%d' % sample_id
    mdb.Job(name=job_name, model=model_name, description='Composite Fairing Sample')
    
    print("Sample %d generated with [0/90/45/-45] Layup." % sample_id)

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for i in range(NUM_SAMPLES):
        generate_sample(i)
