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

```
     Outer Skin (S4RT)          CFRP [45/0/-45/90]s  1.2mm
     ══════════════════
     ┃  Honeycomb Core  ┃       Al-5052 Orthotropic  20mm
     ┃   (C3D8RT)       ┃
     ══════════════════
     Inner Skin (S4RT)          CFRP [45/0/-45/90]s  1.2mm

     接続: Tie Constraints (Skin ↔ Core 接合面)
     温度: 外板150℃ / 内板50℃ / コア勾配 / 基準25℃
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

**build_graph.py（曲率対応）: 18次元**

| Index | Feature | Dim | 説明 |
|-------|---------|-----|------|
| 0-2 | Position | 3 | x, y, z |
| 3-5 | Normal | 3 | nx, ny, nz |
| 6-9 | Curvature | 4 | κ1, κ2, H (平均), K (ガウス) |
| 10-13 | Stress | 4 | s11, s22, s12, DSPSS |
| 14-15 | Node Type | 2 | boundary, loaded |
| 16-17 | **Thermal** | **2** | **temperature, thermal_smises** |

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
