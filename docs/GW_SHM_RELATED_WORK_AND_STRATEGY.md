# GW-SHM 関連研究調査 & 作戦ドキュメント

**最終更新**: 2026-06-07  
**目的**: 産業界・学術界の動向を把握し、Payload2026 の差別化戦略と WCCM2026 ネクストアクションを定める

---

## 1. 産業界サーベイ（2026-06-07 調査）

### 1.1 まとめ：「公開研究はほぼ存在しない」

| 機関 | 公開 SHM/GNN 研究 | 備考 |
|---|---|---|
| **SpaceX** | なし | Falcon/Starship フェアリング SHM は完全非公開 |
| **Rocket Lab** | なし | Electron CFRP モノコック — ML-SHM 非開示 |
| **JAXA/ISAS** | FBG センサによる CFRP/Al ライナー LH2 タンク実時間ひずみ監視（VTOL試験機）| ML/GNN なし、GW ではなく FBG |
| **DLR** | ★ SARISTU プロジェクト: 5×7 m CFRP 胴体に 584 個 PZT 埋め込み Lamb 波 SHM | 最大規模の実証、GW+PZT 手法が一致 |
| **ESA** | ★ CFRP 積層板 FBG センサの LEO 環境対応研究（再使用宇宙機対象） | フェアリング文脈で引用可 |
| **NASA Langley** | 弾性波 FEM シミュレーション（EFIT）for NDT 認証 | 直接 GNN ではない |

**→ 戦略的含意**: SpaceX・Rocket Lab・JAXA に GNN-SHM の公開先行研究がない。  
「宇宙機フェアリング向け GNN-SHM は先行研究が皆無」という**研究ギャップを正当化できる**。  
DLR/ESA を「機関的先例」として Introduction に引用し、ML 化の必然性に繋げる。

---

## 2. 学術界サーベイ — DL済み論文

### 2.1 ダウンロード済み論文一覧

| arXiv ID | タイトル | 年 | 手法 | 関連度 |
|---|---|---|---|---|
| **2605.20311** | WaveGraphNet (EPFL, Fink) | 2026-05 | GNN + Inverse-Forward 正則化 | ★★★ |
| **2606.03933** | PISAMP (PISAMP, Chinesta) | 2026-06 | 物理整合信号分解 + 楕円局在 | ★★★ |
| **2311.03765** | Honeycomb Composite Sandwich SHM (IIT Bombay) | 2023-11 | RF + ML, ハニカム複合材欠陥分類 | ★★★ |
| **2510.24614** | Semi-supervised HI from GW (aerospace CFRP) | 2025-10 | KAE 異常検知 + 半教師あり | ★★ |

### 2.2 WaveGraphNet 詳細（最重要先行研究）

**論文**: Sharma, Bharade, Fink (EPFL). arXiv:2605.20311. May 2026.

**設定**
- 構造: 500×500 mm CFRP 平板（HexPly M21）
- センサ: PZT × 12個、ピッチキャッチ、100 kHz
- タスク: 単一欠陥の **2D 座標回帰**（28 損傷位置）

**グラフ構成**
- ノード: トランスデューサ 12個（座標特徴）
- エッジ: 66 伝播パス × 双方向 = 132エッジ
- エッジ特徴: 差分 GW のスペクトル記述子（256 周波数ビン × 振幅+位相）

**アーキテクチャ（Inverse-Forward 結合）**
```
逆ブランチ: スペクトル記述子 → 4層 GAT → MaxPool → MLP → 欠陥座標 p̂
順ブランチ: 候補座標 → 幾何特徴 → 36パスのエネルギー偏差 ΔE（物理正則化器）
損失: L_total = L_coord + λ·L_fwd + μ·L_corr
```

**結果（空間ホールドアウト OGW-1 ベンチマーク）**
| 分割 | WaveGraphNet MAE | GAT baseline MAE | 改善 |
|---|---|---|---|
| Split A | **0.220±0.027** | 0.305 | ▼28% |
| Split B (難) | **0.262±0.016** | 0.383 | ▼32% |
- FPR = 0%（全ベースライン中唯一）

**限界（論文より）**
- 単一欠陥・疑似欠陥のみ
- 温度変動・多重欠陥・センサドリフト未対応
- センサ数スケール（12ノード）の小グラフのみ検証

### 2.3 PISAMP（最新 2026-06、物理整合 GW-SHM）

**論文**: Rodriguez, Chinesta 他. arXiv:2606.03933. Jun 2026.

**手法**: 単一 PZT 励起 → 複数分散 GW モードを物理拘束付き信号分解（PISAMP）で分離  
→ 波数関数 + 伝播距離を直接推定 → 楕円局在法で欠陥位置特定  
**強み**: 純データ駆動より解釈可能、計算効率が高い  
**関連**: GNN と組み合わせる余地あり（前処理として PISAMP → グラフ特徴量化）

### 2.4 Honeycomb Composite Sandwich SHM（構造一致）

**論文**: Sawant, Thalapil, Tallur, Banerjee 他 (IIT Bombay). arXiv:2311.03765.

**設定**: Al ハニカムサンドイッチ複合構造（HCSS）— **あなたの構造と直接一致**  
欠陥タイプ: コアクラッシュ(CC), 高密度コア(HDC), フィルム接着剤欠損(LFA), テフロン剥離フィルム(TRF)  
**手法**: 特徴量工学（時間/周波数ドメイン）+ Random Forest  
**精度**: 77.89%（シミュレーションデータ）  
**含意**: 同一の HCSS 構造で GNN ベースの手法と比較すれば有意な貢献になる

---

## 3. 差別化戦略（Payload2026 の強み）

### 3.1 先行研究との比較表

| 観点 | WaveGraphNet | HCSS (IIT Bombay) | **Payload2026（我々）** |
|---|---|---|---|
| 対象構造 | CFRP 平板（500×500 mm） | Al ハニカムサンドイッチ平板 | **CFRP/Al-HC フェアリング（曲面、大型）** |
| センサ | PZT 12 個、GW | PZT 4 個、GW | PZT 9 個、GW + FEM 静解析 |
| グラフ | センサグラフ（12 ノード） | なし（非グラフ） | **FEM メッシュグラフ（10,897 ノード）** |
| タスク | 欠陥座標回帰 | 欠陥種別分類 | **欠陥クラス分類（5クラス）** |
| 物理量 | GW 応答のみ | GW 応答のみ | **変位 + 応力 + 温度（FEM 由来）** |
| 応用 | CFRP 構造 SHM 一般 | 航空機 SHM | **宇宙機フェアリング（実構造）** |
| データ | 実験（28 欠陥位置） | シミュレーション | **FEM シミュレーション（100 サンプル）** |

**我々の差別化ポイント**:
1. **スケール**: センサ数グラフ（O(10)ノード）ではなくメッシュグラフ（O(10⁴)ノード）
2. **物理量リッチ**: GW 応答に加えて FEM 由来の変位・応力・温度場
3. **実構造**: H3 ロケット形状（Barrel + Ogive 曲面）
4. **産業的文脈**: 再使用宇宙機の Sim-to-Real SHM パイプライン

---

## 4. WCCM2026 作戦（6/23 ミーティングまで）

### 4.1 現在稼働中のジョブ（Stuttgart03）

| PID | モデル | ログ | 期待 F1 |
|---|---|---|---|
| 520831 | SAGE baseline (retrain) | `logs/baseline_sage_rerun.log` | ~0.78 |
| 521655 | Transolver | `logs/transolver_v1.log` | 未知 |
| 528177 | GW-GAT | `logs/gw_gat_v1.log` | 未知 |
| 528863 | GW-STGNN | `logs/gw_stgnn_v1.log` | 未知 |
| 530934 | MIFNO-GW | `logs/mifno_gw_v1.log` | 未知 |

**チェックコマンド**:
```bash
ssh stuttgart03 "tail -20 ~/Payload2026/logs/baseline_sage_rerun.log"
ssh stuttgart03 "tail -20 ~/Payload2026/logs/gw_gat_v1.log"
ssh stuttgart03 "tail -20 ~/Payload2026/logs/mifno_gw_v1.log"
```

### 4.2 発表の3本柱（確定）

```
柱1: FEM-mesh GNN for CFRP fairing defect classification
     → GCN/GAT/GIN/SAGE 比較、F1 ~0.78 実績
柱2: GW sparse-sensor classification
     → GW-GAT / GW-STGNN (訓練中)
柱3: MIFNO-GW (neural operator for GW-SHM)
     → 106 CSV サンプルから Healthy/Defect 分類
```

### 4.3 6/23 ミーティングまでのアクション

| 優先 | タスク | いつ | 備考 |
|---|---|---|---|
| 🔴 | Stuttgart03 ジョブ結果確認（ログ） | 本日〜6/8 | SAGE/GW-GAT が先に終わるはず |
| 🔴 | SAGE F1 最終値 → Table 1 更新 | 6/8〜6/10 | 論文の核心数値 |
| 🟡 | GW-GAT / STGNN 結果取得 | 6/10〜6/15 | 柱2 の数値 |
| 🟡 | MIFNO-GW 収束確認 | 6/10〜6/15 | 柱3 の実証 |
| 🟢 | Related Work セクション草稿 | 6/15〜6/20 | この文書を元に書く |
| 🟢 | スライド更新（2本目デッキ） | 6/20〜6/22 | WCCM presentation_prep |

### 4.4 論文 Related Work 執筆案（この文書から引用）

```
Section 2: Related Work

2.1 Industrial SHM for Aerospace Composites
DLR (SARISTU) demonstrated the largest embedded PZT-SHM 
demonstration on CFRP at scale. ESA funded FBG-based 
monitoring of CFRP components for reusable launchers. 
Despite growing industrial interest, no public GNN-based 
SHM work exists for rocket fairings.

2.2 Graph Neural Networks for Guided-Wave SHM
WaveGraphNet [Sharma 2026] proposed a coupled inverse-
forward GNN on CFRP plates (12-node sensor graph), 
achieving 28% lower localization error vs. GAT baseline.
IIT Bombay [Sawant 2023] classified 4 damage types in 
Al-honeycomb sandwich structures using ML on guided waves
(77.89% accuracy).
Our work differs: we operate on full FEM mesh graphs
(10,897 nodes) of an actual rocket fairing geometry with
physics-rich node features (displacement, stress, temperature).
```

---

## 5. 中長期作戦（WCCM 後）

### 5.1 データ強化（要 Abaqus 計算）

| 優先 | 内容 | 効果 |
|---|---|---|
| A | GW 動解析データで ODB 完全抽出（2〜3日計算） | MIFNO-GW の精度向上 |
| B | FEM サンプル数を 100→500 に拡大 | メッシュ GNN の汎化性向上 |
| C | 25 mm メッシュ（~40K ノード）実験 | 微細欠陥の検出限界評価 |

### 5.2 モデル改善候補

| アイデア | ソース | 難易度 |
|---|---|---|
| WaveGraphNet 式 Inverse-Forward 正則化を FEM グラフに適用 | WaveGraphNet | 中 |
| PISAMP で GW 前処理 → グラフ特徴量化 | PISAMP 2606.03933 | 中 |
| 温度補正（Domain-Adaptive GAT, PMC12656345） | 論文 | 高 |
| MIFNO にフルウェーブフィールド ODB データを接続 | 独自 | 高 |

### 5.3 論文投稿候補

| ジャーナル/会議 | IF / 格 | 締切目安 |
|---|---|---|
| **WCCM2026** | 国際会議 | 2026-07-22 登壇 |
| Composite Structures | IF ~6.3 | 年中受付 |
| NDT & E International | IF ~4.8 | 年中受付 |
| Smart Materials and Structures | IF ~4.1 | 年中受付 |
| Mechanical Systems and Signal Processing | IF ~8.4 | 年中受付 |

---

## 6. 今すぐやること（6/7〜6/8）

```bash
# 1. ジョブ生死確認
ssh stuttgart03 "ps aux | grep python | grep -v grep"

# 2. 各ログ末尾確認
ssh stuttgart03 "for f in ~/Payload2026/logs/*.log; do echo \"=== \$f ===\"; tail -5 \$f; done"

# 3. SAGE 最終 F1 取得
ssh stuttgart03 "grep 'Best F1\|test_f1\|Final' ~/Payload2026/logs/baseline_sage_rerun.log | tail -5"
```

---

*このドキュメントは Payload2026 の研究戦略の中枢。定期的に更新する。*
