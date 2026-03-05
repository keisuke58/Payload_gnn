[← Home](Home) | [静動融合ランキング](Static-Dynamic-Fusion-Ranking)

# 静解析・動解析融合 — 関連文献

> 本計画に近い文献・論文の調査結果  
> 最終更新: 2026-03-06

---

## 1. 融合・マルチモーダル SHM

### 1.1 Broer et al. (2021) — 融合ベース損傷診断

| 項目 | 内容 |
|------|------|
| **タイトル** | Fusion-based damage diagnostics for stiffened composite panels |
| **掲載** | Structural Health Monitoring (SAGE), 2021 |
| **DOI** | 10.1177/14759217211007127 |
| **手法** | アコースティックエミッション (AE) + 分散型光ファイバひずみセンサの**特徴レベル融合**と**結果レベル融合** |
| **対象** | 炭素繊維エポキシ、単一ストリンガー、疲労 CAI 試験 |
| **成果** | 検出・位置同定・種別同定・重症度の **4 段階診断** を同時達成。単一手法では不可能な能力を融合で実現。 |
| **本計画との関連** | **後段融合 (Late Fusion)** の根拠。Feature-level + Result-level の併用が有効。 |

### 1.2 橋梁欠陥検出 — マルチモーダル融合

| 項目 | 内容 |
|------|------|
| **出典** | arXiv:2412.17968 (2024) |
| **手法** | Impact Echo (IE) + Ultrasonic Surface Waves (USW) の融合、画像処理 |
| **対象** | 橋梁のデラミネーション・デボンディング |
| **成果** | **F1 = 0.83**、誤検出低減 |
| **本計画との関連** | 異種 NDT の融合で相補的効果。静的な IE と動的な USW の組み合わせ。 |

### 1.3 接着継手 — Feature-Based Data Fusion

| 項目 | 内容 |
|------|------|
| **出典** | J. Nondestructive Evaluation (Springer), 2024 |
| **手法** | 超音波 + X線の特徴ベース融合 |
| **対象** | 接着継手のデラミネーション・介在物 |
| **成果** | 欠陥寸法の絶対誤差 **0.02 mm** |
| **本計画との関連** | 特徴レベル融合の有効性。静的な X線と動的な超音波の相補性。 |

---

## 2. ドメイン適応・Sim-to-Real

### 2.1 Zhang et al. (2022) — 分布適応型転移学習

| 項目 | 内容 |
|------|------|
| **タイトル** | Distribution adaptation deep transfer learning method for cross-structure health monitoring using guided waves |
| **掲載** | Structural Health Monitoring (SAGE), 2022 |
| **DOI** | 10.1177/14759217211010709 |
| **手法** | **Joint Distribution Adaptation (JDA)** — 周辺分布と条件分布の両方を適応。Conv-LSTM。 |
| **対象** | 異なる構造間の GW-SHM 転移（構造 A の単一センサ → 構造 B の多センサ監視） |
| **成果** | PCA, TCA 等を上回る損傷イメージング |
| **本計画との関連** | **DANN** の先行研究。JDA は DANN の代替として検討可能。 |

### 2.2 Exeter — 転移学習 + 数値モデリング

| 項目 | 内容 |
|------|------|
| **出典** | University of Exeter, データベース SHM のデータ不足対策 |
| **手法** | FE シミュレーションで学習し、実験データでテスト。TCA, JDA で特徴マッピング。 |
| **成果** | 実験的損傷データを大幅に削減 |
| **本計画との関連** | FEM → 実データの転移。本プロジェクトの DANN 方針と一致。 |

### 2.3 深層学習 Sim-to-Real (GW)

| 項目 | 内容 |
|------|------|
| **出典** | SSRN (Deep Predictions and Transfer Learning for Simulation-Driven SHM) |
| **手法** | FE シミュで学習 → 特徴増強・最終層ファインチューニング・Boosting で実測へ適応 |
| **成果** | 特徴増強 + 適切な活性化関数が GW 損傷同定で最良 |
| **本計画との関連** | シミュ GW → 実 GW の転移。Year 2 の DANN 実装の参考。 |

---

## 3. GNN + GW / FEM

### 3.1 Frontiers (2025) — FEM + GNN で CFRP 欠陥位置同定

| 項目 | 内容 |
|------|------|
| **タイトル** | Development of defect localization method for perforated CFRP specimens using FEM and GNN |
| **掲載** | Frontiers in Materials, 2025 |
| **手法** | **GAT (Graph Attention Network)** + FEM 応力分布 |
| **対象** | 穴あき CFRP、3D 欠陥位置分類 |
| **成果** | **マクロ F1 = 61%**（3 クラス位置同定） |
| **本計画との関連** | 静解析 FEM + GNN の実績。本プロジェクトの静解析 GNN と同系統。 |

### 3.2 MDPI (2024) — 航空宇宙構造のリアルタイム損傷検出

| 項目 | 内容 |
|------|------|
| **タイトル** | Real-Time Damage Detection and Localization on Aerospace Structures Using Graph Neural Networks |
| **掲載** | MDPI Aerospace, 2024 |
| **手法** | GNN、AOMA (Automated Operational Modal Analysis) のひずみモード形状を入力 |
| **対象** | 複合材翼 |
| **成果** | **AUC 0.97**、**位置同定誤差 約 3%** |
| **本計画との関連** | 動的モーダルデータ + GNN。センサグラフとしての GNN 適用。 |

### 3.3 IEEE — 空間時系列 GCN + 決定融合

| 項目 | 内容 |
|------|------|
| **手法** | マルチセンサ空間時系列特徴 + Deep Graph Convolutional Network + 決定融合 |
| **本計画との関連** | **後段融合**の GNN 版。決定融合 (decision fusion) が本計画の Late Fusion に相当。 |

### 3.4 PMC — GCN + 単一モデル決定融合

| 項目 | 内容 |
|------|------|
| **出典** | PMC (PubMed Central), Application of GCN combined with Decision-Making Fusion |
| **本計画との関連** | GNN と決定融合の組み合わせ。アンサンブル効果。 |

---

## 4. ガイド波・デボンディング

### 4.1 デラミネーション位置同定 — 位相スペクトル

| 項目 | 内容 |
|------|------|
| **出典** | Applied Sciences (MDPI), 2023 |
| **手法** | GW モードの位相速度分散を再構成、最小センサでデラミネーション位置同定 |
| **本計画との関連** | 動解析 GW の信号処理。センサ数制約下での検出。 |

### 4.2 ICAS 2022 — ドメイン適応転移学習

| 項目 | 内容 |
|------|------|
| **対象** | 複合材 GW 損傷モニタリング |
| **成果** | **検出精度 約 85.7%**（ドメイン適応転移学習） |
| **本計画との関連** | Sim-to-Real の定量的目標の参考。 |

---

## 5. 文献から得た計画への示唆

| 示唆 | 文献根拠 |
|------|----------|
| **後段融合は有効** | Broer ら: Feature-level + Result-level で 4 段階診断。橋梁: F1=0.83。 |
| **DANN/JDA は Sim-to-Real の本命** | Zhang ら: JDA で GW 転移。Exeter: TCA/JDA で FEM→実データ。 |
| **GNN + センサグラフは実績あり** | MDPI: AUC 0.97。Frontiers: GAT で F1=61%。 |
| **決定融合・アンサンブル** | IEEE, PMC: GCN + 決定融合で性能向上。 |
| **ペアデータは高コスト** | 文献では単一モダリティ or 転移学習が主流。マルチモーダル融合は少数。 |

---

## 6. 参考文献（BibTeX 形式）

```
@article{broer2021fusion,
  title={Fusion-based damage diagnostics for stiffened composite panels},
  author={Broer, AA and others},
  journal={Structural Health Monitoring},
  year={2021},
  doi={10.1177/14759217211007127}
}

@article{zhang2022distribution,
  title={Distribution adaptation deep transfer learning for cross-structure health monitoring using guided waves},
  author={Zhang, Bin and Hong, Xiaobin and Liu, Yuan},
  journal={Structural Health Monitoring},
  year={2022},
  doi={10.1177/14759217211010709}
}
```

---

## 7. 関連

| ページ | 内容 |
|--------|------|
| [Static-Dynamic-Fusion-Ranking](Static-Dynamic-Fusion-Ranking) | 総合おすすめランキング |
| [Dataset-Survey](Dataset-Survey) | 外部データセット調査 |
| [Literature-Review](Literature-Review) | 文献レビュー（既存） |
