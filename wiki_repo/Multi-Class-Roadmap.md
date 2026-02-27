[← Home](Home)

# マルチクラス分類ロードマップ — Multi-Class Classification Roadmap

> 最終更新: 2026-02-28  
> debonding のみから **debond / delam / impact / healthy** の 4 クラス分類への拡張計画。  
> **2年研究計画** の一環 → [2-Year-Goals](2-Year-Goals)

---

## 1. 目標

| 現状 | 目標 (2年後) |
|------|---------------|
| 2 クラス (defect / healthy) | **4 クラス** (debond / delam / impact / healthy) |
| デボンディングのみ検出 | 複合材フェアリングの主要損傷を網羅 |
| マクロ F1 | **> 0.70** (4 クラス) |

---

## 2. クラス定義

| クラス ID | 名称 | 説明 | FEM モデル化 |
|-----------|------|------|--------------|
| 0 | **healthy** | 健全（欠陥なし） | 欠陥挿入なし |
| 1 | **debonding** | スキン-コア界面剥離 | 現行（円形デボンディング） |
| 2 | **delamination** | 積層内の層間剥離 | CONTACT / COHESIVE 等で追加 |
| 3 | **impact** | 衝撃損傷 (BVID 等) | 打撃シミュレーション or 剛性低下ゾーン |

---

## 3. 実装フェーズ

### Phase A: データ拡張 (デボンディング + 健全)

| タスク | 内容 | 難易度 |
|--------|------|--------|
| DOE に healthy 追加 | 欠陥なしサンプル 10–20 件 | ★ |
| `defect_type` カラム | 0=healthy, 1=debond | ★ |
| 学習: 2 クラス維持 | healthy vs defect の 2 値 | ★ |

### Phase B: デラミネーション追加

| タスク | 内容 | 難易度 |
|--------|------|--------|
| FEM でデラミネーション | 積層間の COHESIVE 要素 or 剛性低下 | ★★★★ |
| ラベル: defect_type=2 | delam ノードに 2 を付与 | ★★ |
| 学習: 3 クラス | healthy / debond / delam | ★★★ |

### Phase C: 衝撃損傷追加

| タスク | 内容 | 難易度 |
|--------|------|--------|
| FEM で衝撃損傷 | Explicit 動解析 or 等価剛性低下 | ★★★★★ |
| ラベル: defect_type=3 | impact ゾーンノードに 3 を付与 | ★★ |
| 学習: 4 クラス | healthy / debond / delam / impact | ★★★★ |

### Phase D: マルチクラス GNN

| タスク | 内容 | 難易度 |
|--------|------|--------|
| 出力層 | 4 クラス (num_classes=4) | ★ |
| 損失関数 | CrossEntropy + Focal (クラス不均衡) | ★★ |
| 評価指標 | マクロ F1, 混同行列, クラス別 Recall | ★ |

---

## 4. データ形式の変更

### 4.1 nodes.csv

```diff
- defect_label: 0 or 1
+ defect_label: 0 or 1  # 後方互換
+ defect_type: 0=healthy, 1=debond, 2=delam, 3=impact
```

### 4.2 PyG Data

```python
Data(
    y = [N],           # 0, 1, 2, 3
    num_classes = 4,
)
```

### 4.3 モデル出力

```python
# 現行: [N, 2] (binary)
# マルチ: [N, 4] (4-class logits)
out = model(data.x, data.edge_index, data.edge_attr)  # [N, 4]
```

---

## 5. クラス比の目安

| クラス | 目標割合 | 備考 |
|--------|----------|------|
| healthy | 10–20% | 偽陽性低減 |
| debonding | 50–60% | 現行メイン |
| delamination | 15–25% | Phase B で追加 |
| impact | 10–15% | Phase C で追加 |

※ 実データの発生頻度に合わせて調整。

---

## 6. 依存関係

```
Phase A (healthy 追加)
    ↓
Phase B (delamination FEM + ラベル)
    ↓
Phase C (impact FEM + ラベル)
    ↓
Phase D (4 クラス GNN 学習)
```

---

## 7. 関連

| ページ | 内容 |
|--------|------|
| [Ideal-vs-Implementation](Ideal-vs-Implementation) | 損傷タイプ別の難易度・貢献 |
| [Defect-Generation-and-Labeling](Defect-Generation-and-Labeling) | 現行欠陥パラメータ |
| [Roadmap](Roadmap) | 全体フェーズ |
| [Dataset-Format](Dataset-Format) | データ仕様 |
