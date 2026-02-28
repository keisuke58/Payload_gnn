#!/bin/bash
# run_pipeline.sh
# End-to-end pipeline: FEM Generation -> Analysis -> Extraction -> GNN Preprocessing

set -e # Exit on error

JOB_NAME="Job-H3-Fairing-Test"
DEFECT_PARAMS='{"z_center": 2500.0, "theta_deg": 45.0, "radius": 200.0}'
OUTPUT_DIR="output_data"
GRAPH_OUTPUT="fairing_graph.pt"

# Clean up previous run
rm -rf *.lck *.odb *.stt *.com *.prt *.sim *.log *.msg *.dat *.sta
rm -rf $OUTPUT_DIR

echo "Creating defect parameters..."
# (Passed as JSON string)

echo "Generating Abaqus Model..."
# Run with Abaqus CAE kernel (Required for MDB access)
# Note: Using '--' to separate script arguments from Abaqus arguments
abaqus cae noGUI=src/generate_fairing_dataset.py -- --job $JOB_NAME --defect "$DEFECT_PARAMS"

echo "Submitting Abaqus Job..."
# Run analysis
abaqus job=$JOB_NAME interactive ask_delete=OFF

echo "Extracting Results..."
# Extract to CSV using Abaqus Python (odbAccess is sufficient here, but viewer is safer)
# Using abaqus viewer noGUI to ensure ODB access works reliably
abaqus viewer noGUI=src/extract_odb_results.py -- --odb "${JOB_NAME}.odb" --output $OUTPUT_DIR

echo "Preprocessing for GNN..."
# Run with System Python (Python 3.x with PyTorch)
python src/preprocess_fairing_data.py $OUTPUT_DIR $GRAPH_OUTPUT

echo "Pipeline Complete. Data saved to $GRAPH_OUTPUT"
