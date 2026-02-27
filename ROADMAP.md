# Payload Fairing GNN Project Roadmap

このプロジェクトは、ペイロードフェアリングの構造ヘルスモニタリング (SHM) のために、GNNとFEMを統合した欠陥位置特定手法を開発することを目的としています。
現在の最優先事項は、**高品質な学習データセットの作成** です。

## フェーズ 1: データセット生成 (Current Focus)
Abaqus/Pythonスクリプトを使用して、以下の仕様でデータセットを生成します。

### 1. 形状モデリング (Geometry)
- **対象**: ペイロードフェアリング 1/6 円筒バレルセクション
- **構造**: CFRPハニカムサンドイッチパネル
  - Facesheet: CFRP積層板 (Shell Elements: S4R/S3)
  - Core: アルミ/Nomexハニカム (Solid Elements: C3D8R or Equivalent Homogenized Shell)
- **寸法**: 半径 $R$, 高さ $H$, 厚さ $t_{face}$, $t_{core}$ (パラメータ化)

### 2. 欠陥導入 (Defect Injection)
- **種類**:
  - Facesheet-Core Disbond (剥離)
  - Impact Damage (衝撃損傷 - 剛性低下領域としてモデル化)
- **パラメータ**:
  - 位置 $(x, y)$
  - サイズ (半径 $r$)
  - 深刻度 (剛性低減率)

### 3. 荷重条件 (Loading)
- **圧縮荷重 (Axial Compression)**: 座屈挙動を模擬
- **音響圧 (Acoustic Pressure)**: ランダムな圧力分布
- **境界条件**: 下端固定、上端自由または拘束

### 4. FEM解析 & 出力 (Simulation & Output)
- **Solver**: Abaqus/Standard
- **出力データ**:
  - ノード座標 $(x, y, z)$
  - 要素コネクティビティ
  - 表面主応力和 (DSPSS)
  - 歪み分布 (Strain Field)

## フェーズ 2: グラフ構築 (Graph Construction)
- **ノード定義**: FEMメッシュのノード、または要素中心
- **エッジ定義**:
  - メッシュ接続性に基づくエッジ
  - 距離に基づく近傍エッジ (k-NN)
  - 曲率を考慮したエッジ属性 (Geodesic Distance)
- **特徴量**:
  - ノード特徴量: 座標、DSPSS、歪み
  - エッジ特徴量: 相対位置、距離

## フェーズ 3: GNNモデル開発 (Model Development)
- **アーキテクチャ**:
  - Graph Attention Network (GAT)
  - Graph Isomorphism Network (GIN) for topology awareness
- **タスク**: ノード分類 (欠陥あり/なし) または 回帰 (欠陥中心座標の予測)

## フェーズ 4: 検証と実証 (Validation)
- **Split**: Training / Validation / Test sets
- **Metrics**: Accuracy, IoU (Intersection over Union), Localization Error
- **Visualization**: 欠陥予測ヒートマップの可視化

---

## Wiki への追加事項
- [ ] Abaqus Python Scripting Guide for Composites
- [ ] Dataset Format Specification (CSV/HDF5)
- [ ] GNN Architecture Diagrams
