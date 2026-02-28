[← Home](Home)

# Architecture: H3 フェアリング GNN-SHM パイプライン

> **日本語概要**: FEM (Abaqus) → CSV 抽出 → 曲率対応グラフ構築 (build_graph.py) → PyG Data → GNN 4 種 (GCN/GAT/GIN/SAGE) 学習 → 推論 API。Shell-Solid-Shell サンドイッチ構造、熱解析 (CTE 不整合) を含む。用語は [用語集](Vocabulary) を参照。

---

## 全体パイプライン

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PIPELINE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: FEM              Phase 2: Graph           Phase 3: GNN   │
│  ┌───────────────┐         ┌──────────────┐        ┌────────────┐  │
│  │ Abaqus/Standard│         │  PyG Graph   │        │  Training  │  │
│  │               │  CSV    │              │  .pt   │            │  │
│  │ Shell-Solid-  │───────→│ Curvature-   │──────→│ GCN / GAT  │  │
│  │ Shell Sandwich│  nodes  │ Aware Graph  │ Data   │ GIN / SAGE │  │
│  │ + Thermal     │  elems  │ Construction │        │            │  │
│  └───────────────┘         └──────────────┘        └─────┬──────┘  │
│         ↑                                                │         │
│         │                                                ↓         │
│  JAXA Literature                                  ┌────────────┐   │
│  (Thermal BCs,                                    │ Inference  │   │
│   Acoustic 147dB,                                 │ FastAPI    │   │
│   Separation Shock)                               │ predict_api│   │
│                                                   └────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## ファイル構成

```
src/
├── generate_fairing_dataset.py   # Abaqus マクロ: FEM モデル生成 + 熱解析 + CSV出力
├── build_graph.py                # 曲率対応グラフ構築 (法線・主曲率・測地線距離)
├── preprocess_fairing_data.py    # FEM CSV → PyG Data 変換 + 正規化 + 分割
├── models.py                     # GNN アーキテクチャ定義 (4種)
├── train.py                      # 学習ループ (Focal Loss, CV, Early Stopping)
├── evaluate.py                   # 評価指標・ヒートマップ・比較
└── predict_api.py                # 推論 API (Python callable + FastAPI REST)
```

## FEM モデル: Shell-Solid-Shell Sandwich

**積層構造（17層）** — 詳細: [Layup-Structure](Layup-Structure)

| 部位 | 層数 | 厚さ | 構成 |
|------|------|------|------|
| Outer Skin | 8 plies | 1.0 mm | CFRP [45/0/-45/90]s (0.125 mm/ply) |
| Core | 1 | 38 mm | Al-5052 ハニカム |
| Inner Skin | 8 plies | 1.0 mm | CFRP [45/0/-45/90]s (0.125 mm/ply) |
| **合計** | **17層** | **40 mm** | |

![Layup Structure](images/layup_structure.png)

```
     Outer Skin (S4RT)          CFRP [45/0/-45/90]s  1.0 mm (8 plies)
     ══════════════════
     ┃  Honeycomb Core  ┃       Al-5052 Orthotropic  38 mm
     ┃   (C3D8RT)       ┃
     ══════════════════
     Inner Skin (S4RT)          CFRP [45/0/-45/90]s  1.0 mm (8 plies)

     接続: Tie Constraints (Skin ↔ Core 接合面)
     温度: 外板 120℃ / 内板 20℃ / コア勾配 / 基準 25℃
```

### 解析ステップ

| Step | Type | 内容 |
|------|------|------|
| Initial | — | 変位BC（下端固定）+ 初期温度 T_REF=25℃ |
| Thermal | CoupledTempDisplacement (Steady) | 温度場適用 → CTE不整合による熱応力 |
| Load | Static | 軸圧縮 + 外圧 30kPa (Max Q) |

### 材料物性

**CFRP T300/914（温度依存）**:

| T (℃) | E1 (GPa) | E2 (GPa) | G12 (GPa) | CTE1 (/℃) | CTE2 (/℃) |
|--------|----------|----------|-----------|-----------|-----------|
| 25 | 135 | 10 | 5.0 | -0.3e-6 | 28e-6 |
| 100 | 133 | 9 | 4.5 | — | — |
| 150 | 130 | 7 | 3.8 | — | — |
| 200 | 125 | 5.5 | 3.2 | — | — |

**ハニカムコア (Al-5052)**: E3=1380 MPa, G13=310 MPa, CTE=23e-6 /℃

## GNN アーキテクチャ

4種のアーキテクチャを実装・比較:

| Model | 特徴 | 適用場面 |
|-------|------|---------|
| **GCN** | シンプルなスペクトル畳み込み | ベースライン |
| **GAT** | Attention機構で隣接ノードの重み付け | 欠陥境界の検出 |
| **GIN** | トポロジ識別力が最も高い | グラフ構造の活用 |
| **GraphSAGE** | サンプリングベース、大規模対応 | スケーラビリティ |

### 共通設定

- **損失関数**: Focal Loss (α=0.25, γ=2.0) — 健全ノード>>欠陥ノードの不均衡対策
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR
- **Early Stopping**: patience=20 (validation F1 based)
- **入力次元**: 動的検出（データから自動判定）

### ノード特徴量

**build_graph.py（曲率対応）: 34次元**

| Index | Feature | Dim | 説明 |
|-------|---------|-----|------|
| 0-2 | Position | 3 | x, y, z |
| 3-5 | Normal | 3 | nx, ny, nz |
| 6-9 | Curvature | 4 | κ1, κ2, H, K |
| 10-13 | Displacement | 4 | ux, uy, uz, u_mag |
| 14 | Temperature | 1 | temp |
| 15-20 | Stress | 6 | s11, s22, s12, smises, principal_stress_sum, thermal_smises |
| 21-23 | Strain | 3 | le11, le22, le12 |
| 24-26 | Fiber orientation | 3 | 周方向単位ベクトル |
| 27-31 | Layup | 5 | layup [0,45,-45,90]°, circum_angle |
| 32-33 | Node Type | 2 | boundary, loaded |

詳細: [Node-Features](Node-Features)

**preprocess_fairing_data.py（簡易版）: 12次元**

| Index | Feature | Dim |
|-------|---------|-----|
| 0-2 | Position | 3 |
| 3-6 | Stress | 4 |
| 7-9 | Node Type | 3 |
| 10-11 | **Thermal** | **2** |

### エッジ特徴量

| Feature | Dim | 説明 |
|---------|-----|------|
| Relative position | 3 | Δx, Δy, Δz |
| Euclidean distance | 1 | ||Δr|| |
| Normal angle | 1 | 法線間角度 |
| Geodesic distance | 1 | メッシュ上最短経路（オプション） |

## 推論 API

```
POST /predict
  Input:  nodes.csv + elements.csv
  Output: {
    "defect_nodes": [node_id, ...],
    "center": [x, y, z],
    "confidence": 0.95,
    "heatmap": [prob_per_node]
  }

GET /health → {"status": "ok"}
```
