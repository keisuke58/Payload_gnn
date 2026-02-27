[← Home](Home)

# 超最先端 ML — Cutting-Edge ML for H3 Fairing SHM

> 最終更新: 2026-02-28  
> 2024–2025 年の SOTA 手法で本プロジェクトに適用可能なものを整理。

---

## 日本語概要

2024–2025 年の最先端 ML を H3 フェアリング SHM に適用可能な観点で整理。**最優先**: Graph Mamba（長距離依存・大グラフに強い）、E(3)-Equivariant GNN（物理的整合性）。**高優先**: FNO（データ生成加速）、PINN（逆問題）。専門用語は [用語集](Vocabulary) を参照。

---

## 1. 適用可能性マトリクス

| 手法 | 本プロジェクトへの適合度 | 難易度 | 期待効果 | 優先度 |
|------|-------------------------|--------|----------|--------|
| **Graph Mamba** | ★★★★★ | ★★★ | 長距離依存・オーバースクワッシュ解消 | 高 |
| **Graph Transformer** | ★★★★☆ | ★★★ | 大域 Attention、曲率・異方性の表現 | 高 |
| **E(3)-Equivariant GNN** | ★★★★★ | ★★★★ | 回転・並進不変、物理的整合性 | 高 |
| **Graph Foundation Model** | ★★★☆☆ | ★★★★★ | 事前学習・転移、データ効率 | 中 |
| **Neural Operator (FNO)** | ★★★★★ | ★★★★ | データ生成 1000x 加速、解像度非依存 | 高 |
| **Physics-Informed ML** | ★★★★★ | ★★★★ | データ不足補完、逆問題 | 高 |
| **Graph Diffusion** | ★★★☆☆ | ★★★★ | データ拡張・欠陥サンプル生成 | 中 |

---

## 2. Graph Mamba (State Space Model)

### 概要
- **論文**: Graph-Mamba (2024), [arXiv:2402.08678](https://arxiv.org/abs/2402.08678)
- **特徴**: 従来 GNN の **over-squashing** と **長距離依存の弱さ** を解消
- **計算量**: Graph Transformer の O(N²) に対して **O(N)** で線形スケール

### 本プロジェクトでの利点
- フェアリングは **φ5.2 m × 10 m** の巨大構造 → 欠陥とセンサ間の長距離相関が重要
- メッシュ 10 mm で ~250K ノード → Transformer はメモリ逼迫、Mamba は対応可能
- **Selective SSM** で入力依存のコンテキスト選択 → 欠陥近傍に適応的に注目

### 実装
- [Graph-Mamba (GitHub)](https://github.com/bowang-lab/Graph-Mamba)
- GraphGPS の Attention を Mamba Block に置換

---

## 3. Graph Transformer

### 概要
- **動向**: 2024–2025 で Graph Tokenization, Structure-Aware Attention が発展
- **利点**: over-smoothing / over-squashing の克服、大域的な構造の学習

### 本プロジェクトでの利点
- 曲率・法線・異方性を **Positional Encoding** や **Edge Bias** に組み込み可能
- Point Transformer は既にロードマップにあり → Graph Transformer 系も比較対象に

### 注意
- 大グラフでは O(N²) でメモリ逼迫 → 25 mm メッシュ (~40K nodes) が上限目安

---

## 4. E(3)-Equivariant GNN

### 概要
- **代表**: NequIP, SchNet, e3nn
- **特徴**: 3D 回転・並進に対して **等変** → 物理法則と整合

### 本プロジェクトでの利点
- フェアリングは **曲面多様体**。メッシュの向きに依存しない表現が望ましい
- 応力テンソル・変位ベクトルは **幾何量** → Equivariant で自然に扱える
- **データ効率**: NequIP は競合の 1/1000 のデータで同精度を報告

### 実装
- [e3nn](https://github.com/e3nn/e3nn), [NequIP](https://github.com/mir-group/nequip)
- ノード特徴を **球面調和関数** で展開する必要あり

---

## 5. Neural Operator (FNO / DeepONet)

### 概要
- **FNO**: フーリエ空間で積分演算子を学習、解像度非依存
- **DeepONet**: Branch-Trunk で関数空間の写像を学習
- **効果**: 従来ソルバーの **4–5 桁** の高速化

### 本プロジェクトでの利点
- **データ生成加速**: 欠陥マスク → 波動場の写像を学習 → Abaqus の代わりに 1000x 高速
- **Zero-Shot Super-Resolution**: 粗いメッシュで学習 → 細かいメッシュで推論可能
- 波動方程式の **解演算子** を学習 → 物理的整合性が高い

### 実装
- [neuraloperator](https://github.com/neuraloperator/neuraloperator)
- 既に `src/prototype_fno.py` あり → 拡張が現実的

---

## 6. Physics-Informed ML (PINN, Grey-Box)

### 概要
- **PINN**: 損失関数に PDE 残差を組み込み
- **Grey-Box**: 簡易物理モデル + データ駆動の残差学習
- **Full-Waveform Inversion**: NN + 有限差分で超音波ガイド波の逆問題を解く

### 本プロジェクトでの利点
- **データ不足**: 欠陥サンプルが少ない → 物理制約で補完
- **逆問題**: スパースセンサ (20 点) から欠陥位置を推定
- 弾性波動方程式を損失に組み込み → デボンディングによる剛性低下を「発見」

### 実装
- 既に `src/prototype_pinn.py` あり
- [DeepXDE](https://github.com/lululxvi/deepxde) 等のライブラリ

---

## 7. Graph Foundation Model (GFM)

### 概要
- **GIT** (ICML 2025): 30+ グラフで事前学習、Zero-Shot 汎化
- **GFT, AnyGraph, RiemannGFM**: ドメイン横断の転移学習

### 本プロジェクトでの利点
- **データ効率**: 大規模グラフで事前学習 → 少量の H3 データで Fine-Tune
- **Sim-to-Real**: シミュレーショングラフで事前学習 → 実データで適応

### 注意
- 2025 年時点で **メッシュグラフ・SHM 向け** の GFM は未整備
- 分子・タンパク質・ソーシャルグラフが主 → 適応コストが高い

---

## 8. Graph Diffusion

### 概要
- **用途**: グラフ生成、制約付き生成 (ConStruct, TreeDiff)
- **Beta Diffusion**: 離散-連続の混合に対応

### 本プロジェクトでの利点
- **データ拡張**: 欠陥パターンの拡散モデルで新サンプル生成
- **条件付き生成**: 欠陥サイズ・位置を指定してサンプル生成

### 注意
- メッシュグラフの生成は未開拓
- まずは FNO サロゲートでデータ拡張を優先するのが現実的

---

## 9. 推奨導入順序

| 順位 | 手法 | 理由 |
|------|------|------|
| 1 | **Graph Mamba** | 実装が比較的容易、長距離・大グラフに強い。既存 GNN の置き換えとして即効性 |
| 2 | **E(3)-Equivariant GNN** | 物理的整合性が高く、データ効率も良い。論文の新規性が高い |
| 3 | **FNO サロゲート** | データ生成ボトルネック解消。既存プロトタイプを拡張 |
| 4 | **PINN 逆問題** | スパースセンサ対応。既存プロトタイプを拡張 |
| 5 | **Graph Transformer** | Point Transformer と併せて比較。メッシュ 25 mm 以下で検証 |

---

## 10. 難しい用語の補足

| 用語 | 読み・意味 |
|------|------------|
| **over-squashing** | オーバースクワッシュ。GNN で遠くのノード情報が圧縮され失われる現象 |
| **over-smoothing** | オーバースムーシング。層を重ねるとノード表現が似通ってしまう現象 |
| **Equivariant** | 等変。回転・並進に対して出力が対応して変わる性質 |
| **SSM (State Space Model)** | 状態空間モデル。時系列の線形動的システム。Mamba は Selective SSM |
| **FNO** | Fourier Neural Operator。フーリエ変換で PDE の解演算子を学習 |
| **Zero-Shot** | ゼロショット。追加学習なしで未知の条件に汎化すること |

---

## 10. 参考リンク

| 手法 | 論文・リポジトリ |
|------|-----------------|
| Graph Mamba | [arXiv:2402.08678](https://arxiv.org/abs/2402.08678), [GitHub](https://github.com/bowang-lab/Graph-Mamba) |
| Graph Foundation Models | [arXiv:2505.15116](https://arxiv.org/abs/2505.15116) |
| E(3)-Equivariant | [e3nn](https://github.com/e3nn/e3nn), [NequIP](https://github.com/mir-group/nequip) |
| Neural Operator | [neuraloperator](https://github.com/neuraloperator/neuraloperator) |
| Physics-Informed SHM | [ScienceDirect Review 2024](https://www.sciencedirect.com/science/article/pii/S0957417424015458) |
