# メッシュ構造 — Mesh Structure

## 結論：4面体ではない

| 部材 | 要素タイプ | 形状 | 備考 |
|------|------------|------|------|
| **外板 (Outer Skin)** | S4R, S3 | 4節点シェル / 3節点シェル | 四辺形 + 三角形（境界） |
| **内板 (Inner Skin)** | S4R, S3 | 同上 | 同上 |
| **コア (Core)** | C3D8R | **8節点六面体** | 四面体ではなく六面体 |

→ **四面体 (C3D4) は使用していない**。コアはリボルブ形状のため Abaqus が六面体 (C3D8R) でスイープメッシュを生成。

---

## メッシュの作り方

### 1. ジオメトリ

```
Inner Skin: BaseShellRevolve (1/6 セクション, 60°)
Core:       BaseSolidRevolve (閉じた断面を回転 → ソリッド)
Outer Skin: BaseShellRevolve (同上)
```

### 2. シード・メッシュ

```python
# generate_fairing_dataset.py
a.seedPartInstance(regions=(inst_inner, inst_core, inst_outer), 
                   size=GLOBAL_SEED, deviationFactor=0.1)  # 50 mm
# 欠陥周辺の局所細分化
a.seedEdgeBySize(edges=edges, size=DEFECT_SEED, constraint=FINER)  # 15 mm
a.generateMesh(regions=(inst_inner, inst_core, inst_outer))
```

- **GLOBAL_SEED**: 50 mm（全体）
- **DEFECT_SEED**: 15 mm（欠陥ゾーン）

### 3. パーティション（欠陥領域の切り出し）

`partition_debonding_zone()` で Datum Plane により欠陥円を近似：

- **XZPLANE** (y=constant): 軸方向の範囲（y = z_c ± r_def）
- **3点平面**: θ 方向の範囲（欠陥中心 ± d_theta）

```python
assembly.PartitionFaceByDatumPlane(datumPlane=..., faces=faces)
```

→ 外板・コアの面を平面で分割し、欠陥ゾーンを定義。Tie 制約の除外に使用（現状は全面 Tie）。

#### 切り方の正当性

| 項目 | 評価 | 備考 |
|------|------|------|
| **実行順序** | ✓ 正当 | パーティション → メッシュ（Abaqus 標準） |
| **手法** | ✓ 正当 | `PartitionFaceByDatumPlane` は Abaqus 標準 API |
| **Datum 平面** | ✓ 正当 | 主平面 + 3点平面で欠陥円を矩形近似 |
| **面の絞り込み** | ✓ 修正済 | `getByBoundingBox` で Y=軸方向(z1..z2)、XZ=半径方向を正しく指定 |

欠陥円を 4 枚の平面で囲む矩形で近似しているため、円形境界ではなく「角ばった」境界になる。物理的デボンディングの厳密表現には CZM 等が必要だが、GNN 用ラベル付与・ゾーン定義としては妥当。

### 4. 抽出対象

`extract_odb_results.py` は **外板 (PART-OUTERSKIN-1) のみ** を抽出。  
コアの六面体メッシュは GNN 入力には使っていない。

---

## 要素タイプ詳細

| Abaqus | 節点数 | 形状 | 用途 |
|--------|--------|------|------|
| S4R | 4 | 四辺形 | シェル（メイン） |
| S3 | 3 | 三角形 | シェル（境界・オジーブ先端） |
| C3D8R | 8 | 六面体 | コア（ソリッド） |

---

## 理想メッシュ（細密版）

```bash
bash scripts/run_ideal_mesh.sh
```

- GLOBAL_SEED=25 mm, DEFECT_SEED=8 mm → 外板 ~43k ノード（通常 50mm の約 4 倍）
- 出力: `dataset_output_ideal/sample_0000/`

## 可視化

```bash
python scripts/visualize_mesh_structure.py --data dataset_output/sample_0001
python scripts/visualize_mesh_structure.py --data dataset_output_ideal/sample_0000
```

出力: `figures/mesh_structure/`
- `01_element_types.png` — S4R/S3 の個数分布
- `02_mesh_wireframe.png` — ワイヤーフレーム（青=S4R 四辺形, 赤=S3 三角形）
