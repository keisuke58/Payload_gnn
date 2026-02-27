# データセット徹底調査: CFRP SHM / デボンディング検出

> 本プロジェクト（H3 フェアリング SHM）で利用・参考にできるデータセットの網羅的調査  
> **GNN に限らず** CNN / RNN / Transformer / Point Cloud / Neural Operator 等の多様なモデルに対応  
> 調査日: 2026-02-28

---

## 1. 誘導波 SHM データセット（直接関連）

### 1.1 Open Guided Waves (OGW) プラットフォーム

**URL**: https://openguidedwaves.de/downloads/

| # | 内容 | デボンディング | 形式 | リンク |
|---|------|---------------|------|--------|
| **OGW #1** | CFRP 平板、疑似欠陥（Al質量付加） | 擬似 | 波場計測、SHM計測 | openguidedwaves.de |
| **OGW #2** | 温度変動 (20–60°C) 下の誘導波伝播 | なし | 時系列波形 | [Scientific Data 2019](https://www.nature.com/articles/s41597-019-0208-1) |
| **OGW #3** | CFRP板 + ストリンガー、複数欠陥サイズ | 人工欠陥 | ナローバンド/ブロードバンド、波場、シミュレーション | openguidedwaves.de |
| **OGW #4** | CFRP板 + **オメガストリンガー、完全接着 vs 部分デボンディング** | ✅ **あり** | 3D LDV 波場、HDF5 | [Zenodo 5105861](https://doi.org/10.5281/zenodo.5105861) |
| **OGW #5** | 長期 SHM（4.5年、約640万計測） | 13種の損傷 | 時系列、温度・湿度・気圧 | [Nature Sci Data 2025](https://www.nature.com/articles/s41597-025-05300-5) |

**OGW #4 の詳細** (フェアリングに最も近い):
- 試料: CFRP板 + オメガストリンガー（スキン-ストリンガー界面）
- 状態: 完全接着 / 部分デボンディング
- 計測: 3D レーザードップラー振動計による波場
- 論文: Kudela et al., Data in Brief 42, 108078 (2022)

---

### 1.2 DINS-SHM (Deep Inverse Neural Surrogates)

**GitHub**: https://github.com/mahindrautela/DINS-SHM  
**論文**: Rautela & Gopalakrishnan, Expert Systems with Applications 167, 114189 (2021)

| 項目 | 内容 |
|------|------|
| **データ** | スペクトル有限要素の順問題シミュレーション出力 |
| **形式** | 時系列 + 2D CWT 時間-周波数表現 |
| **ノイズ** | 複数レベルのガウスノイズ |
| **対象** | 等方性・複合材ウェーブガイド |
| **用途** | 損傷検出（分類）+ 位置推定（回帰） |
| **Zenodo** | README に記載（要確認） |

---

### 1.3 SHM Open Database (Bologna 大学)

**URL**: http://shm.ing.unibo.it/

| 項目 | 内容 |
|------|------|
| **対象** | フルスケール複合材航空機翼デモンストレーター |
| **センサ** | 160 個の圧電トランスデューサ |
| **条件** | T0: 静的, T1: 組立, T2: 9万サイクル疲労後, T3: 衝撃後 |
| **形式** | 誘導波信号（ピエゾ駆動・受信） |
| **適用モデル** | GNN（センサグラフ）, 1D CNN, LSTM, アンサンブル |

---

### 1.4 4TU.ResearchData — 複合材トーションボックス

**URL**: https://data.4tu.nl/

| データセット | 内容 |
|-------------|------|
| **誘導波 + 電気機械リアクタンス** | フルスケール複合材トーションボックス、BVID 評価 |
| **CFRP 平面デラミネーション** | 準静的押し込み、DIC、超音波スキャン、AE |
| **デラミネーション onset** | 直交異方性複合材、準静的デラミネーション成長 |
| **Mode II デラミネーション** | 炭素/エポキシ、ELS 試験、AE、エネルギー解放率 |

---

### 1.5 CONCEPT Dataset (UNESP)

**GitHub**: https://github.com/shm-unesp/DATASET_PLATEUN01

- 複合材板の健全・損傷状態の Lamb 波計測
- 分析用サンプルコード付き

---

### 1.6 Mendeley — 振動ベースデラミネーション

**URL**: https://data.mendeley.com/datasets/n35zwbzhcf/1

- 積層複合材の健全・デラミネーション状態の振動データ
- 加速度計、カンチレバー条件

---

## 2. NASA / PHM Society データセット

**URL**: https://data.phmsociety.org/nasa/

### 2.1 CFRP Composites (NASA #2)

**直接ダウンロード**: https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip

| 項目 | 内容 |
|------|------|
| **実験** | Run-to-failure、引張-引張疲労 |
| **センサ** | 16 PZT（Lamb波）、複数 triaxial ひずみゲージ |
| **Ground Truth** | 定期 X 線撮影による内部損傷 |
| **構成** | 3 種の積層 |
| **提供** | Stanford SACL + NASA Ames PCoE |

### 2.2 Fatigue Crack Growth in Aluminum Lap Joint (NASA #18)

**ダウンロード**: https://phm-datasets.s3.amazonaws.com/NASA/18.+Fatigue+Crack+Growth+in+Aluminum+Lap+Joint.zip

- アルミラップ継手の疲労き裂成長
- Lamb 波信号（PZT アクチュエータ-受信対）
- 光学計測によるき裂長さ（Ground Truth）
- 2019 PHM Data Challenge で使用

---

## 3. FEM / メッシュベース データセット（GNN / FNO / CNN 等）

### 3.1 SimuStruct (Inductiva)

**概要**: FEniCS による 2D 構造 FEM シミュレーション

| 項目 | 内容 |
|------|------|
| **試料** | 鋼板（E=210 GPa, ν=0.3）、6 穴の矩形パターン |
| **荷重** | 一軸 100 MPa |
| **サンプル数** | 1,000 |
| **出力** | メッシュ、von Mises 応力（ノード単位） |
| **形式** | CSV（幾何、メッシュ、応力） |
| **適用モデル** | GNN, FNO, U-Net, CNN（グリッド化時） |
| **入手** | [Inductiva Blog](https://inductiva.ai/blog/article/simustruct-dataset), datasets@inductiva.ai |

### 3.2 Mechanical MNIST Crack Path

**GitHub**: https://github.com/saeedmhz/Mechanical-MNIST-Crack-Path  
**Dryad**: https://datadryad.org/dataset/doi:10.5061/dryad.rv15dv486

| 項目 | 内容 |
|------|------|
| **モデル** | 相場モデルによる準静的脆性き裂伝播 |
| **サンプル数** | 70,000（学習 60,000 + テスト 10,000） |
| **出力** | 変位場、損傷場、力-変位曲線 |
| **サイズ** | 拡張版 148 GB |
| **適用モデル** | U-Net, MultiRes-WNet, FNO, CNN（2D 画像/グリッド） |

### 3.3 Johns Hopkins — Phase-Field Fracture

**URL**: https://archive.data.jhu.edu/

- 機能傾斜板、円形インクルージョン
- 構成あたり 1,000 実現、31 準静的荷重ステップ
- 変位場、相場損傷、荷重-変位曲線

---

## 4. アコースティックエミッション (AE) データセット

### 4.1 TU Delft / 4TU

| データセット | 内容 |
|-------------|------|
| **CFRP 平面デラミネーション** | 準静的押し込み、DIC、超音波、AE |
| **CFRP 圧縮試験** | AE 計測、損傷モード分類 |
| **複合材接着継手** | AE、破壊メカニズム（cohesive、デラミネーション、繊維破断等） |

---

## 5. 3D 点群・異常検出（参考）

| データセット | 内容 | 用途 |
|-------------|------|------|
| **MVTec 3D-AD** | 4,000+ スキャン、10 カテゴリ | 教師なし異常検出 |
| **3D-ADAM** | 14,120 スキャン、27,346 欠陥 | 工業欠陥検出 |
| **Real3D-AD** | 1,254 アイテム、高解像度点群 | 異常検出 |
| **MiniShift** | 2,577 点群、微細欠陥 | 高解像度異常検出 |

※ これらは主に表面欠陥用。内部デボンディングとは形式が異なるが、Point Transformer / PointMAE 等の 3D 点群モデルの検証に利用可能。

---

## 6. その他リソース

### 6.1 LANL SHMTools

**URL**: https://lanl.gov/projects/national-security-education-center/engineering/software/shm-data-sets-and-software.php

- MATLAB パッケージ、110+ 関数
- ベンチマークデータセット（3 階建て建物等）
- 加速度、ひずみ、誘導波に対応

### 6.2 FDSP (Fatigue Delamination Shape Prognostics)

**GitHub**: https://github.com/fdspz/FDSP

- 疲労デラミネーション形状予測
- 数値シミュレーション支援の転移学習

### 6.3 i2a2/shm_ml_datasets

**GitHub**: https://github.com/i2a2/shm_ml_datasets

- SHM 向け ML データセット
- sensors、signals_examples 等

---

## 7. 適用モデル別マトリクス（GNN に限らず）

| データ形式 | 適用可能なモデル |
|-----------|------------------|
| **時系列波形** | 1D CNN, LSTM, Transformer, TCN, SVM, RF |
| **2D 時間-周波数 (CWT/STFT)** | 2D CNN, ViT, ResNet |
| **波場 (空間×時間)** | 2D/3D CNN, FNO, U-Net, Neural Operator |
| **メッシュ (ノード+エッジ)** | GNN (GCN/GAT/GIN/SAGE), Point Transformer, FNO |
| **点群 (x,y,z + 特徴)** | PointNet, PointNet++, Point Transformer, PointMAE |
| **グリッド画像** | CNN, U-Net, SegFormer, YOLO |
| **スパースセンサ** | GNN（センサグラフ）, 1D CNN, LSTM |

---

## 8. 本プロジェクトとの適合性マトリクス

| データセット | デボンディング | 形式 | 適用モデル例 | 推奨用途 |
|-------------|---------------|------|--------------|----------|
| **OGW #4** | ✅ | 波場 HDF5 | 2D/3D CNN, FNO, GNN（波場→グラフ変換） | **デボンディング検出** |
| **OGW #2** | なし | 時系列 | 1D CNN, LSTM, 温度補償 ML | **温度補償** |
| **OGW #1, #3** | 擬似/人工 | 波場・時系列 | CNN, LSTM, 従来 ML | パイプライン検証 |
| **NASA CFRP** | 疲労 | 時系列 (16 PZT) | 1D CNN, LSTM, GNN（センサグラフ） | 転移学習 |
| **DINS-SHM** | シミュ | 時系列 + CWT | 1D/2D CNN, LSTM（論文実装済み） | ベースライン |
| **SHM Open DB** | 疲労・衝撃 | 時系列 (160 PZT) | GNN, CNN, アンサンブル | 大規模センサ |
| **SimuStruct** | なし | メッシュ CSV | GNN, FNO, U-Net | **応力場予測** |
| **Mechanical MNIST** | き裂 | 2D グリッド | U-Net, FNO, CNN | 損傷場予測 |
| **MVTec 3D-AD 等** | 表面欠陥 | 点群 | PointNet++, 3D CNN | 異常検出参考 |
| **自作 Abaqus** | ✅ | メッシュ CSV | **GNN, Point Transformer, FNO, UV+CNN** | **本番** |

---

## 8. 推奨アクションプラン

1. **即時利用**
   - **OGW #4** (Zenodo 5105861): デボンディング検出のベースライン
   - **NASA CFRP** (PHM #2): 疲労損傷検出の転移学習
   - **SimuStruct**: GNN メッシュ処理の検証

2. **パイプライン検証**
   - **DINS-SHM**: 波形→特徴量→CNN/RNN の流れを参考
   - **OGW #1**: 平板での「波形→グラフ」変換の試行

3. **温度・環境変動**
   - **OGW #2**: 温度補償アルゴリズムの検証
   - **OGW #5**: 長期・実環境データでのロバスト性評価

4. **自作データの継続**
   - 円筒・オジャイブ形状、熱応力、スキン-コア界面は既存データにほぼなし
   - 自作 FEM が本命、上記は補助・検証用

---

## 10. ダウンロードリンク一覧

| データセット | URL |
|-------------|-----|
| OGW #4 | https://doi.org/10.5281/zenodo.5105861 |
| OGW 一覧 | https://openguidedwaves.de/downloads/ |
| NASA CFRP | https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip |
| NASA 全データ | https://data.phmsociety.org/nasa/ |
| DINS-SHM | https://github.com/mahindrautela/DINS-SHM |
| SHM Open DB | http://shm.ing.unibo.it/ |
| 4TU | https://data.4tu.nl/ |
| Mechanical MNIST | https://datadryad.org/dataset/doi:10.5061/dryad.rv15dv486 |
| SimuStruct | https://inductiva.ai/blog/article/simustruct-dataset |
