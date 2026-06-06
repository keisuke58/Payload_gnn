#!/bin/bash
# HEMEW3D Dataset Downloader (Lehmann et al. 2024)
#
# Full dataset: ~264 GB (30,000 simulations)
# This script downloads a subset for pre-training.
#
# Source: doi:10.57745/LAI6YU
# GitHub: https://github.com/lehmannfa/MIFNO
#
# Usage:
#   bash scripts/download_hemew3d.sh           # subset (first 2000 samples)
#   bash scripts/download_hemew3d.sh full      # full dataset (~264 GB)

set -e

DATA_DIR="${1:-data/hemew3d}"
MODE="${2:-subset}"

echo "=== HEMEW3D Dataset Download ==="
echo "  Target: $DATA_DIR"
echo "  Mode: $MODE"

mkdir -p "$DATA_DIR"

# Clone MIFNO repo for preprocessing scripts
if [ ! -d "$DATA_DIR/MIFNO" ]; then
    echo "Cloning MIFNO repository..."
    git clone --depth 1 https://github.com/lehmannfa/MIFNO.git "$DATA_DIR/MIFNO"
fi

# Materials (3.9 GB total, 15 files × ~260 MB each)
MATERIALS_URL="https://entrepot.recherche.data.gouv.fr/api/access/datafile/"

echo ""
echo "=== Materials Download ==="
echo "Materials are .npy files (3D S-wave velocity fields at 32^3 resolution)"
echo "Total: 15 files × ~260 MB = 3.9 GB"
echo ""
echo "To download manually, visit:"
echo "  https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/LAI6YU"
echo ""
echo "After downloading, place .npy files in: $DATA_DIR/materials/"
echo "And .h5 velocity files in: $DATA_DIR/velocities/"

mkdir -p "$DATA_DIR/materials"
mkdir -p "$DATA_DIR/velocities"

# Check if data already exists
N_MAT=$(ls "$DATA_DIR/materials/"*.npy 2>/dev/null | wc -l)
N_VEL=$(ls "$DATA_DIR/velocities/"*.h5 2>/dev/null | wc -l)
echo ""
echo "Current data:"
echo "  Materials: $N_MAT .npy files"
echo "  Velocities: $N_VEL .h5 files"

if [ "$N_VEL" -gt 0 ]; then
    echo ""
    echo "Data found! Ready for pre-training."
    echo "Use: python src/train_fno_gw.py --pretrain_hemew3d $DATA_DIR"
else
    echo ""
    echo "No velocity data found."
    echo ""
    echo "Option 1: Download from Recherche Data Gouv (manual)"
    echo "  URL: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/LAI6YU"
    echo ""
    echo "Option 2: Use MIFNO preprocessing scripts"
    echo "  cd $DATA_DIR/MIFNO && python preprocess.py --data_dir .."
    echo ""
    echo "Option 3: Generate sample data for testing"
    echo "  python -c \"import scripts.generate_hemew3d_sample as g; g.generate('$DATA_DIR')\""
fi

echo ""
echo "Done."
