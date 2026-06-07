# Payload — データ拡充計画 ＆ 定量目標（2026-06-07）

現状データは不足（FEM 214 / OGW実質3標本・健全1）。決定版/高IFには A:実データ追加 ＋ B:FEM拡大 が必須。
両軸を進めつつ、達成すべき**定量目標**を明文化する。

## A. 実データ拡充（最弱点＝最優先）
| データ | 内容 | modality | 用途 | 状態 |
|---|---|---|---|---|
| **OGW long-term / full-scale** (Nature SciData 2025, s41597-025-05300-5) | 160 PZT・翼box実構造・多環境多状態 | **GW** | sim-to-real本命拡充 | 取得候補(repo要特定) |
| OGW omega-stringer (Zenodo 5105861) | 平板+ストリンガ・3状態 | GW | 取得済(3標本) | ✅ |
| Single-stiffener panels (Mendeley ys8r8m7bx2) | ストリンガdisbond+衝撃・run-to-failure | AE | 構造直結(別modality) | 候補 |
| NASA-PCoE composite | コーポン疲労・多サイクル | Lamb | クロス検証 | 既知 |
- 取得は rclone/curl で `data/external/` に貯める（gitignore済）。
- **目標: 独立実標本 ≥10–20**（現状3）。

## B. FEM拡大 DOE（自分で増やせる・要Abaqus Explicit）
構造整合FEM(`FEM_OGW_MATCHED_DESIGN.md`)で生成:
- 因子×水準: 欠陥状態(健全/局所/大) × 位置(5) × 周波数(4: 50/100/200/300kHz) × **健全反復(微小摂動 ≥20)**。
- **目標規模: ≥500 標本**（現状214）。健全を複数にしてcalibration問題を解消。
- ライセンス対策: 領域縮小+吸収境界 / 対称 / cpus=1試走→横展開。

## 定量目標（達成で「決定版」）
### Payload
- FEM in-dist: 二値 F1 ≥0.95（現0.98） / 5クラス局在 macro-F1 ≥0.70（現~0.55, 要データ）。
- **sim-to-real: AUC ≥0.85 ∧ recall ≥0.85 @ FPR ≤0.10 ∧ bal-acc ≥0.85、実標本 ≥10**（現: AUC0.83/recall0.70@FPR0.05/bal0.83/標本3）。
- LGSTA優位がOOD(未知サイズ/位置/温度)に転移するか検証（転移率 ≥in-dist差の50%）。

### Composites B（[WCCM repo `paper_figs/RESULTS_LOG.md`]）
- in-dist: HybridMGN macro-F1 ≥0.80（現0.79）/ defect-F1 ≥0.80（現0.78）/ exact ≥0.85（現0.85✓）。
- **OODギャップ縮小: 非アーキ手段で OOD defect-F1 ≥0.30（現≈0）or OOD detRec ≥0.85 @ FPR ≤0.10**。
- rigor: 勝ち構成3シード±std / アブレ確定 / OOD ≥2軸（サイズ＋位置 or 層）。

## 進め方
A(実データ取得)とB(FEM DOE)を並行。指標は多指標パネル統一([[feedback_eval_metrics]])。結果は各repoのRESULTS_LOG/本書に貯める。
