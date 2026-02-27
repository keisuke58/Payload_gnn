[← Home](Home)

# 欠陥生成とラベル付け — Defect Generation & Labeling

> 最終更新: 2026-02-28

H3 フェアリング FEM データセットにおける**デボンディング欠陥**の生成方法と、ノード単位の**defect_label** 付与ルールをまとめる。

---

## 1. 想定する欠陥

### 1.1 欠陥タイプ

| 種類 | 界面 | 発生要因 | 本プロジェクト |
|------|------|----------|-----------------|
| **製造時接着不良** | 外スキン-コア | OoA 接着の不完全固化、異物混入 | **対象** |
| 製造時接着不良 | 内スキン-コア | 同上 | 将来拡張 |
| 熱応力による進展 | 外スキン-コア | CTE 不整合 | 将来拡張 |
| 音響疲労 | 外スキン-コア | 打ち上げ 147–148 dB | 将来拡張 |

**F8 事故の示唆**: 製造検査で PSS 4個全てに想定を超える剥離を確認。製造時接着不良が有力要因。

### 1.2 モデル化

- **界面**: **外スキン-コア (outer)** に限定
- **形状**: **円形パッチ**（製造不良の典型的モデル化）
- **FEM 表現**: パーティショニングで欠陥ゾーンを定義 → Tie 制約の部分解除（計画）

---

## 2. パラメータ生成 (DOE)

### 2.1 円筒座標

欠陥は円筒座標 `(θ, z, r)` で定義:

| パラメータ | 範囲 | 単位 | 根拠 |
|------------|------|------|------|
| **theta_deg** | 5° – 55° | deg | 対称端 (0°, 60°) からマージン |
| **z_center** | 800 – 4200 | mm | クランプ端・上端を回避（Barrel 5 m） |
| **radius** | 20 – 250 | mm | サイズ階層に応じて設定（メッシュ整合） |

### 2.2 サイズ階層（層化サンプリング）

| 階層 | 半径 (mm) | 想定 | 割合 |
|------|-----------|------|------|
| **Small** | 20–50 | 検出限界付近 (h=50mm) | 30% |
| **Medium** | 50–80 | 在役検出可能 | 40% |
| **Large** | 80–150 | 重大、進展リスク | 25% |
| **Critical** | 150–250 | 破壊直前 | 5% |

### 2.3 サンプリング方法

- **θ, z**: Latin Hypercube Sampling (LHS) で一様にカバー
- **radius**: 各階層内で一様乱数

```bash
python src/generate_doe.py --n_samples 100 --output doe_100.json
```

---

## 3. FEM での欠陥挿入

### 3.1 パーティショニング (`generate_fairing_dataset.py`)

1. 欠陥パラメータ `(z_center, theta_deg, radius)` を受け取る
2. Datum Plane 4枚で外板・コアを分割:
   - z = z_center ± radius
   - θ = theta_deg ± Δθ（円弧長から換算）
3. 欠陥ゾーンを矩形近似で定義

### 3.2 境界条件

- **健全領域**: Tie 制約（スキン-コア完全結合）
- **欠陥領域**: Tie 解除 → 摩擦なし接触（計画、実装は進行中）

---

## 4. ラベル付け (defect_label)

### 4.1 幾何学的判定

ノード `(x, y, z)` が欠陥円内かどうかは、**円筒面上の距離**で判定。

**座標系**: Abaqus Revolve では Y=軸方向、X-Z=半径平面。
- `r_node = sqrt(x² + z²)`
- `θ_node = atan2(z, x)` [rad]
- `z_axial = y`（軸方向位置）

```
arc_mm   = r_node × |θ_node − θ_center|
dz       = z_axial − z_center
dist     = sqrt(arc² + dz²)

defect_label = 1  if dist ≤ radius  else 0
```

- **arc_mm**: 円周方向の弧長 (mm)
- **dz**: 軸方向の差 (mm)
- **dist**: 円筒展開面上のユークリッド距離

### 4.2 実装箇所

| スクリプト | 役割 |
|------------|------|
| `extract_odb_results.py` | ODB 抽出時に `--defect_json` から defect_label を付与 |
| `scripts/add_defect_labels.py` | 既存 nodes.csv に後から defect_label を付与 |

### 4.3 出力形式

**nodes.csv**:
```
node_id,x,y,z,ux,uy,uz,temp,defect_label
1,1319.0,5000.0,2284.575,...,0
2,...,1
```

**metadata.csv**:
```
key,value
defect_type,debonding
n_defect_nodes,42
theta_deg,38.1
z_center,895.0
radius,17.0
```

### 4.4 健全ベースライン

`defect_params` が無い場合（healthy_baseline）:
- 全ノード `defect_label = 0`
- `metadata.csv`: defect_type=healthy, n_defect_nodes=0

---

## 5. パイプライン

```
generate_doe.py          → doe_100.json (欠陥パラメータ)
       ↓
run_batch.py             → 各サンプルごとに:
       ├─ generate_fairing_dataset.py  (FEM モデル生成)
       ├─ extract_odb_results.py --defect_json  (ODB → CSV + defect_label)
       └─ metadata.csv 出力
       ↓
build_graph.py           → PyG Data (y = defect_label)
```

### 既存データへのラベル付与

```bash
python scripts/add_defect_labels.py --doe doe_100.json --data_dir dataset_output
```

---

## 6. 参考文献

- [[DEFECT_PLAN]] — 欠陥挿入計画（JAXA 研究者向け）
- [[JAXA-Fairing-Specs]] — H3 フェアリング仕様
- [[Dataset-Format]] — データセット仕様（nodes.csv 列定義）
