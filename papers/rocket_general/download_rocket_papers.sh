#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== ロケット関連論文ダウンロード ==="

# === 燃焼不安定性 ===
echo "[1/16] Combustion Instability — Progress & Challenges (ICCFD)"
curl -sL -o "CombustionInstability_ICCFD.pdf" \
  "https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-3105_paper.pdf"

# === ノズル設計最適化 ===
echo "[2/16] Rocket Nozzle Multidisciplinary Design Optimization (Lisbon Thesis)"
curl -sL -o "NozzleDesignOptimization_Thesis.pdf" \
  "https://fenix.tecnico.ulisboa.pt/downloadFile/1970719973969901/Thesis_on_Multidisciplinary_Nozzle_Design_Reviewed.pdf"

echo "[3/16] NASA CFD-Based Rocket Injector Optimization"
curl -sL -o "NASA_CFD_Injector_Optimization.pdf" \
  "https://ntrs.nasa.gov/api/citations/20030060421/downloads/20030060421.pdf"

# === フェアリング複合材構造 ===
echo "[4/16] Beyond Gravity — Reusable Fairing MDAO"
curl -sL -o "BeyondGravity_ReusableFairing_MDAO.pdf" \
  "https://europeanspaceflight.com/wp-content/uploads/2023/01/Beyond-Gravitys-approach-to-the-multidisciplinary-design-analysis-and-optimization-of-reusable-payload-fairings.pdf"

echo "[5/16] Composite Payload Fairing Architecture (SLS)"
curl -sL -o "CompositeFairing_SLS_Architecture.pdf" \
  "https://core.ac.uk/download/pdf/10568099.pdf"

# === 推進剤スロッシング ===
echo "[6/16] NASA — Analysis of Propellant Slosh Dynamics"
curl -sL -o "NASA_PropellantSloshDynamics.pdf" \
  "https://ntrs.nasa.gov/api/citations/19680017770/downloads/19680017770.pdf"

echo "[7/16] Open-Source Propellant Sloshing Modeling"
curl -sL -o "OpenSource_PropellantSloshing.pdf" \
  "https://www.researchgate.net/profile/Alvaro-Romero-Calvo/publication/377851094_Open-Source_Propellant_Sloshing_Modeling_and_Simulation/links/65e0444fe7670d36abe61763/Open-Source-Propellant-Sloshing-Modeling-and-Simulation.pdf"

echo "[8/16] Orion Service Module Propellant Sloshing (FLOW-3D)"
curl -sL -o "Orion_PropellantSloshing_FLOW3D.pdf" \
  "https://www.flow3d.com/wp-content/uploads/2014/08/Investigation-of-Propellant-Sloshing-and-Zero-Gravity-Equilibrium-for-the-Orion-Service-Module-Propellant-Tanks.pdf"

# === 再使用ロケット再突入 ===
echo "[9/16] MDO of Reusable Launch Vehicles (AIAA 2024)"
curl -sL -o "AIAA2024_MDO_ReusableLV.pdf" \
  "https://aiaa.org/wp-content/uploads/2024/12/multidisciplinarydesignoptimizationofreusablelaunchvehicles.pdf"

echo "[10/16] NASA LOFTID Aerothermodynamics"
curl -sL -o "NASA_LOFTID_Aerothermo.pdf" \
  "https://ntrs.nasa.gov/api/citations/20230017572/downloads/AIAA%202024_LOFTID_paper_V16.pdf"

# === RL 姿勢制御 ===
echo "[11/16] Deep RL Spacecraft Attitude Control (Glasgow)"
curl -sL -o "DeepRL_SpacecraftAttitude_Glasgow.pdf" \
  "https://eprints.gla.ac.uk/301489/2/303820.pdf"

echo "[12/16] NASA — Autonomous Spacecraft Attitude Control via Deep RL"
curl -sL -o "NASA_AutonomousAttitude_DeepRL.pdf" \
  "https://ntrs.nasa.gov/api/citations/20205008891/downloads/elkins_iac_RLADCS_v2_2_reformat.pdf"

echo "[13/16] Deep RL Attitude with Keep-Out Constraint (arXiv)"
curl -sL -o "DeepRL_Attitude_KeepOut_arXiv.pdf" \
  "https://arxiv.org/pdf/2511.13746"

# === GNN + SHM (プロジェクト直結!) ===
echo "[14/16] GNN for SHM — Delft University"
curl -sL -o "GNN_SHM_Delft.pdf" \
  "https://repository.tudelft.nl/file/File_3aef0521-5de7-4198-931b-2470482d13d6"

echo "[15/16] GNN Aerodynamic Flow Reconstruction (arXiv)"
curl -sL -o "GNN_AeroFlowReconstruction_arXiv.pdf" \
  "https://arxiv.org/pdf/2301.03228"

# === トポロジー最適化 + ML ===
echo "[16/16] Self-Directed ML for Topology Optimization (Nature Comms)"
curl -sL -o "NatureComms_ML_TopologyOpt.pdf" \
  "https://www.nature.com/articles/s41467-021-27713-7.pdf"

echo ""
echo "=== ダウンロード完了 ==="
ls -lhS *.pdf
