[← Home](Home)

# XAI ロードマップ — Explainable AI Roadmap

> 最終更新: 2026-02-28  
> SHAP, LIME 等の説明可能 AI (XAI) を導入し、欠陥検出の根拠を可視化する計画。

---

## 1. 概要

GNN や LightGBM による欠陥検出モデルの**予測根拠を説明**するため、以下を導入予定:

- **SHAP** (SHapley Additive exPlanations): 特徴量の寄与度を Shapley 値で算出
- **LIME** (Local Interpretable Model-agnostic Explanations): 局所線形近似で説明
- **GNN 固有**: アテンション重み、Grad-CAM 系、ノード/エッジ重要度

---

## 2. 導入予定手法

| 手法 | 対象モデル | 説明単位 | 用途 |
|------|------------|----------|------|
| **SHAP** | LightGBM, GNN (集約後) | 特徴量・ノード | グローバル・ローカル寄与度 |
| **LIME** | 任意 (モデル非依存) | サンプル単位 | 個別予測の局所説明 |
| **Attention 可視化** | GAT | エッジ重み | どの隣接ノードを重視したか |
| **GNNExplainer** | GCN/GAT/GIN | サブグラフ | 予測に寄与した部分グラフ |
| **Integrated Gradients** | GNN | ノード・エッジ | 勾配ベースの重要度 |

---

## 3. 実装計画

### Phase A: 樹木モデル向け (LightGBM)

| タスク | 内容 | 難易度 |
|--------|------|--------|
| SHAP 導入 | `shap` ライブラリで feature importance, summary plot | ★ |
| ノード単位説明 | 各ノードの SHAP 値をフェアリング上にマッピング | ★★ |
| LIME | サンプル単位の局所説明 (オプション) | ★ |

### Phase B: GNN 向け

| タスク | 内容 | 難易度 |
|--------|------|--------|
| GNNExplainer | PyG の `torch_geometric.nn.models.GNNExplainer` | ★★ |
| Attention 可視化 | GAT のエッジ重みを抽出・可視化 | ★ |
| ノード重要度 | 勾配または SHAP 風のノード寄与度 | ★★★ |

### Phase C: 可視化・論文用

| タスク | 内容 | 難易度 |
|--------|------|--------|
| 3D マップ | 欠陥検出根拠をフェアリング 3D 上にヒートマップ表示 | ★★ |
| 論文用図 | SHAP summary, 代表サンプルの説明図 | ★ |

---

## 4. 依存関係・ライブラリ

```
# 追加予定
pip install shap          # SHAP (TreeExplainer, KernelExplainer)
pip install lime          # LIME (lime_tabular 等)
# PyG 標準
from torch_geometric.nn.models import GNNExplainer
```

---

## 5. 想定出力

- **SHAP summary plot**: 全特徴量の寄与度ランキング
- **SHAP dependence plot**: 特定特徴量と予測の関係
- **ノード重要度マップ**: フェアリング上で「どのノードが欠陥判定に効いたか」を色で表示
- **GNNExplainer サブグラフ**: 予測に寄与したノード・エッジのハイライト

---

## 6. 関連

| ページ | 内容 |
|--------|------|
| [Roadmap](Roadmap) | 全体ロードマップ (Phase 10) |
| [2-Year-Goals](2-Year-Goals) | 2年目標 |
| [Benchmark-Targets](Benchmark-Targets) | ベンチマーク指標・評価計画 |
| [Architecture](Architecture) | モデル構成 |
| [Ideal-vs-Implementation](Ideal-vs-Implementation) | 理想・難易度マトリクス |
