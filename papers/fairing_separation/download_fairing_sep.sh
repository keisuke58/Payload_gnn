#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== フェアリング分離ダイナミクス論文ダウンロード ==="

# 1. JAXA H3 8号機 フェアリング分離異常 調査報告 (2025/12/25)
echo "[1/8] JAXA H3-8 フェアリング分離異常 調査報告 (2025)"
curl -sL -o "JAXA_H3F8_FairingSep_Investigation_2025.pdf" \
  "https://www.mext.go.jp/content/20251225-mxt_uchukai01-000046593_002.pdf"

# 2. KTH — Simulation of payload fairing during separation (Abaqus)
echo "[2/8] KTH — Fairing Separation Simulation (Abaqus)"
curl -sL -o "KTH_FairingSeparation_Simulation.pdf" \
  "https://kth.diva-portal.org/smash/get/diva2:1078063/FULLTEXT01.pdf"

# 3. Kawasaki — Development of Payload Fairings for Launch Vehicle
echo "[3/8] Kawasaki — H3 Fairing Development (CFRP/Honeycomb)"
curl -sL -o "Kawasaki_H3_Fairing_Development.pdf" \
  "https://global.kawasaki.com/en/corp/rd/magazine/179/pdf/n179en06.pdf"

# 4. Reliability assessment of fairing separation (multi-source uncertainties)
echo "[4/8] Fairing Separation Reliability Assessment (Thin-Walled Structures)"
curl -sL -o "FairingSep_Reliability_MultiUncertainty.pdf" \
  "https://jatm.com.br/jatm/article/download/1332/1003/6298"

# 5. Pyroshock Prediction of Ridge-Cut Explosive Bolts (Hydrocodes)
echo "[5/8] Pyroshock Explosive Bolts (Wiley)"
curl -sL -o "Pyroshock_ExplosiveBolts_Hydrocodes.pdf" \
  "https://onlinelibrary.wiley.com/doi/pdf/10.1155/2016/1218767"

# 6. VEGA fairing structural response (modal coupling)
echo "[6/8] VEGA Fairing Structural Response (CEAS)"
curl -sL -o "VEGA_Fairing_ModalCoupling.pdf" \
  "https://link.springer.com/content/pdf/10.1007/s12567-018-0225-5.pdf" 2>/dev/null || echo "  (Springer要認証、スキップ)"

# 7. Fairing aerodynamic study
echo "[7/8] Fairing Aerodynamic Study"
curl -sL -o "Fairing_Aerodynamic_Study.pdf" \
  "https://www.rroij.com/open-access/aerodynamic-study-of-payload-fairing.pdf"

# 8. Mechanical Shock FEA
echo "[8/8] Mechanical Shock FEA (KTH)"
curl -sL -o "MechanicalShock_FEA.pdf" \
  "https://www.diva-portal.org/smash/get/diva2:1555097/FULLTEXT01.pdf"

echo ""
echo "=== ダウンロード完了 ==="
ls -lhS *.pdf
