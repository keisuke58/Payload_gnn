#!/bin/bash
# SpaceX 関連の主要論文をダウンロード

set -e
cd "$(dirname "$0")"

echo "=== SpaceX 関連論文ダウンロード ==="

# 1. Lossless Convexification (Blackmore, Açıkmeşe) — Falcon 9 着陸の理論的基盤
echo "[1/8] Lossless Convexification of Nonconvex Control Bound (Blackmore+, IEEE TCST 2013)"
curl -sL -o "Blackmore2013_LosslessConvexification_IEEE_TCST.pdf" \
  "http://www.larsblackmore.com/iee_tcst13.pdf"

# 2. Enhancements on Convex Programming PDG (Açıkmeşe+, AAS 2008)
echo "[2/8] Enhancements on Convex Programming Based PDG (Açıkmeşe+, AAS 2008)"
curl -sL -o "Acikmese2008_ConvexPDG_Enhancements.pdf" \
  "http://www.larsblackmore.com/AcikmeseAAS08.pdf"

# 3. Lossless Convexification with Pointing Constraints (Carson+, ACC 2011)
echo "[3/8] Lossless Convexification with Pointing Constraints (Carson+, ACC 2011)"
curl -sL -o "Carson2011_LosslessConvex_PointingConstraints.pdf" \
  "http://www.larsblackmore.com/CarsonAcikmeseBlackmoreACC11.pdf"

# 4. PICA-X Heat Shield (SpaceX Dragon TPS)
echo "[4/8] PICA-X Heat Shield (SpaceX)"
curl -sL -o "SpaceX_PICAX_HeatShield.pdf" \
  "https://spacex.com.pl/files/2017-11/pica-x.pdf"

# 5. Crew Dragon Docking System (Matthews, ESMATS 2020)
echo "[5/8] Crew Dragon Docking System (Matthews, ESMATS 2020)"
curl -sL -o "Matthews2020_CrewDragon_DockingSystem.pdf" \
  "https://esmats.eu/amspapers/pastpapers/pdfs/2020/matthews.pdf"

# 6. Starship Flip-Landing Trajectory Optimization (arXiv 2025)
echo "[6/8] Starship Flip-Landing Trajectory Optimization (arXiv)"
curl -sL -o "Starship_FlipLanding_TrajectoryOpt.pdf" \
  "https://arxiv.org/pdf/2508.06520"

# 7. Rocket Landing RL with Random Annealing Jump Start (arXiv 2024)
echo "[7/8] Rocket Landing RL - RAJS (arXiv 2024)"
curl -sL -o "RocketLanding_RL_RAJS_2024.pdf" \
  "https://arxiv.org/pdf/2407.15083"

# 8. Online Trajectory Optimization for Starship-like Flip Landing (MDPI Mathematics 2023)
echo "[8/8] Online Trajectory Opt for Starship-like Flip Landing (MDPI 2023)"
curl -sL -o "MDPI2023_FlipLanding_OnlineTrajectoryOpt.pdf" \
  "https://www.mdpi.com/2227-7390/11/2/288/pdf"

echo ""
echo "=== ダウンロード完了 ==="
ls -lh *.pdf
