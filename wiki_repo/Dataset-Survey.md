[← Home](Home)

# データセット調査: CFRP SHM / デボンディング検出

> 本プロジェクトで利用・参考にできる外部データセットの網羅的調査  
> **GNN に限らず** CNN / RNN / Transformer / Point Cloud / Neural Operator 等に対応  
> 調査日: 2026-02-28

---

## 日本語概要

OGW、NASA CFRP、DINS-SHM、SimuStruct 等の外部データセットを調査。**必須**: OGW #4 (デボンディング)、NASA CFRP (疲労)。適用モデル別マトリクス、本プロジェクトとの適合性、ダウンロード手順を記載。用語は [用語集](Vocabulary) を参照。

---

## 必須データセット（ダウンロード状況）

| データセット | 保存先 | 状態 | 用途 |
|-------------|--------|------|------|
| **NASA CFRP** (疲労損傷) | `data/external/NASA_Composites/` | ✅ 完了 | 転移学習、損傷進展 |
| **OGW #4** (デボンディング) | `data/external/OGW_CFRP_Stringer_*.zip` | ダウンロード中 or 手動 | デボンディング検出のベースライン |

※ `data/` は .gitignore 対象。ダウンロードは `scripts/download_external_datasets.sh` を参照。

---

## 1. 誘導波 SHM データセット

### 1.1 Open Guided Waves (OGW)

**URL**: https://openguidedwaves.de/downloads/

| # | 内容 | デボンディング | 形式 | 適用モデル |
|---|------|---------------|------|------------|
| **OGW #1** | CFRP 平板、疑似欠陥（Al質量付加） | 擬似 | 波場、SHM計測 | 2D CNN, FNO, 従来 ML |
| **OGW #2** | 温度変動 (20–60°C) 下の誘導波伝播 | なし | 時系列波形 | 1D CNN, LSTM, 温度補償 |
| **OGW #3** | CFRP板 + ストリンガー、複数欠陥サイズ | 人工欠陥 | 波場、シミュレーション | 2D CNN, FNO |
| **OGW #4** | CFRP板 + **オメガストリンガー、完全接着 vs 部分デボンディング** | ✅ **あり** | 3D LDV 波場、HDF5 | 2D/3D CNN, FNO, GNN |
| **OGW #5** | 長期 SHM（4.5年、約640万計測） | 13種の損傷 | 時系列、環境 | 1D CNN, LSTM, Transformer |

**OGW #4** (フェアリングに最も近い): 試料 CFRP板 + オメガストリンガー、完全接着 / 部分デボンディング。3D LDV 波場。論文: Kudela et al., Data in Brief 42, 108078 (2022)。

### 1.2 DINS-SHM

**GitHub**: https://github.com/mahindrautela/DINS-SHM

- スペクトル有限要素シミュレーション出力
- 時系列 + 2D CWT 時間-周波数
- **適用モデル**: 1D/2D CNN, LSTM（論文実装済み）

### 1.3 SHM Open Database (Bologna 大学)

**URL**: http://shm.ing.unibo.it/

- フルスケール複合材航空機翼、160 PZT
- T0: 静的, T1: 組立, T2: 9万サイクル疲労後, T3: 衝撃後
- **適用モデル**: GNN（センサグラフ）, 1D CNN, LSTM

---

## 2. NASA / PHM Society

**URL**: https://data.phmsociety.org/nasa/

### NASA CFRP Composites (#2)

- Run-to-failure、引張-引張疲労
- 16 PZT（Lamb波）、triaxial ひずみゲージ
- Ground Truth: 定期 X 線撮影
- **適用モデル**: 1D CNN, LSTM, GNN（センサグラフ）

### NASA Fatigue Crack (#18)

- アルミラップ継手、疲労き裂成長
- Lamb 波、光学計測き裂長さ
- 2019 PHM Data Challenge

---

## 3. FEM / メッシュベース

| データセット | 内容 | 適用モデル |
|-------------|------|------------|
| **SimuStruct** | 鋼板 6 穴、1,000 サンプル、von Mises 応力 | GNN, FNO, U-Net |
| **Mechanical MNIST** | 70,000 サンプル、相場き裂、変位・損傷場 | U-Net, FNO, CNN |
| **Johns Hopkins** | 機能傾斜板、相場破壊 | FNO, CNN |

---

## 4. 適用モデル別マトリクス

| データ形式 | 適用可能なモデル |
|-----------|------------------|
| **時系列波形** | 1D CNN, LSTM, Transformer, TCN, SVM, RF |
| **2D 時間-周波数 (CWT/STFT)** | 2D CNN, ViT, ResNet |
| **波場 (空間×時間)** | 2D/3D CNN, FNO, U-Net, Neural Operator |
| **メッシュ (ノード+エッジ)** | GNN, Point Transformer, FNO |
| **点群** | PointNet++, Point Transformer, PointMAE |
| **グリッド画像** | CNN, U-Net, SegFormer, YOLO |
| **スパースセンサ** | GNN（センサグラフ）, 1D CNN, LSTM |

---

## 5. 本プロジェクトとの適合性

| データセット | デボンディング | 推奨用途 |
|-------------|---------------|----------|
| **OGW #4** | ✅ | **デボンディング検出**（必須） |
| **OGW #2** | なし | **温度補償** |
| **NASA CFRP** | 疲労 | 転移学習（必須） |
| **DINS-SHM** | シミュ | パイプライン検証 |
| **SimuStruct** | なし | 応力場予測検証 |
| **自作 Abaqus** | ✅ | **本番** |

---

## 6. ダウンロードリンク

| データセット | URL |
|-------------|-----|
| **OGW #4** | https://doi.org/10.5281/zenodo.5105861 |
| OGW 一覧 | https://openguidedwaves.de/downloads/ |
| **NASA CFRP** | https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip |
| NASA 全データ | https://data.phmsociety.org/nasa/ |
| DINS-SHM | https://github.com/mahindrautela/DINS-SHM |
| SHM Open DB | http://shm.ing.unibo.it/ |
| Mechanical MNIST | https://datadryad.org/dataset/doi:10.5061/dryad.rv15dv486 |
| SimuStruct | https://inductiva.ai/blog/article/simustruct-dataset |

---

## 7. ダウンロード手順（必須データセット）

```bash
# 保存先作成
mkdir -p data/external
cd data/external

# NASA CFRP (直接ダウンロード)
wget https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip
unzip "2.+Composites.zip" -d NASA_Composites

# OGW #4 (Zenodo - ブラウザで https://zenodo.org/records/5105861 を開き Files から取得)
# または zenodo_get 使用: pip install zenodo-get && zenodo_get 5105861
```
