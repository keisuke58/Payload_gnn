[← Home](Home) · [2-Year-Goals](2-Year-Goals)

# 詳細ロードマップ 2028 — Detailed Roadmap "World-Class Edition"

> 最終更新: 2026-03-07
> 5 Phase × 24 ヶ月の詳細タスク・マイルストーン

---

## 全体タイムライン

```
2026        2027                    2028
Q2  Q3  Q4  Q1  Q2  Q3  Q4  Q1  Q2
│───│───│───│───│───│───│───│───│
├─Phase 1──┤                          基盤強化 + データスケール
    ├────Phase 2────┤                  マルチフィジックス + 全打上げシーケンス
            ├────Phase 3────┤          Physics Foundation Model ★
                    ├────Phase 4────┤  自律損傷管理 + Digital Twin
                                ├─P5─┤ 統合 + 論文 + OSS公開
```

---

## Phase 1: 基盤強化 + データスケール（Month 1–4 / 2026 Q2–Q3）

**目的**: 現在のフェアリング専用GNNを、Foundation Model構築に必要なデータ基盤へスケールアップ

### 1.1 GW全欠陥対応

現在 GW（Guided Wave）シミュレーションでは Debonding のみ実装済。残り 5 タイプを追加。

| 欠陥タイプ | GW実装方法 | 難易度 | 期間 |
|-----------|-----------|--------|------|
| FOD (Foreign Object Debris) | コア内に剛体インクルージョン + Explicit | ★★ | 2 週間 |
| Impact (BVID) | スキン/コアの局所剛性低下 + 残留変形 | ★★ | 2 週間 |
| Delamination | 層間CZMで分離面を定義 | ★★★ | 3 週間 |
| Inner Debond | 内皮-コア界面の Tie 除去 | ★★ | 1 週間 |
| Thermal Progression | CTE不整合による界面応力集中 → CZM劣化 | ★★★ | 3 週間 |

### 1.2 音響疲労シミュレーション

| 項目 | 仕様 |
|------|------|
| 手法 | Abaqus Random Response（モーダル重畳法） |
| 入力 | SPL 147 dB OASPL, 20–2000 Hz |
| 出力 | 応力PSD → Dirlik法で等価応力振幅 → S-N曲線で疲労損傷 |
| 検証 | Epsilon User's Manual の音響環境データと比較 |

### 1.3 PI-GNN v1（Physics-Informed GNN）

損失関数にPDE残差を追加:

```
L_total = L_focal + λ₁ L_physics + λ₂ L_boundary

L_physics = ‖∇·σ + f‖²   (静的平衡)
L_boundary = ‖u - u_BC‖²  (境界条件)
```

- [Physics-Residual-Anomaly-Detection](Physics-Residual-Anomaly-Detection) の L_physics を GNN 学習に直接統合
- λ₁, λ₂ は学習初期に小さく → 後半で増加（カリキュラム学習）

### 1.4 大規模データ生成

| 手法 | サンプル数 | 速度 | 用途 |
|------|-----------|------|------|
| Abaqus FEM（frontale並列） | 2,000 | ~5 min/sample | 高忠実度ベース |
| JAX-FEM | 3,000 | ~30 sec/sample | 中忠実度補完 |
| FNO サロゲート | 5,000 | ~0.1 sec/sample | Foundation Model 事前学習 |
| **合計** | **10,000** | — | Phase 3 の学習データ |

### 1.5 マルチ構造体データ生成

Foundation Model のためにフェアリング以外の構造体も生成:

| 構造体 | FEM手法 | サンプル数 | 欠陥タイプ |
|--------|---------|-----------|-----------|
| CFRP平板 | Abaqus S4R | 5,000 | Debond, Delam, Impact |
| 円筒殻（金属） | Abaqus S4R | 3,000 | 腐食, 亀裂 |
| 円筒殻（CFRP） | Abaqus S4R + Composite | 2,000 | Debond, Delam |
| スティフナ付きパネル | Abaqus C3D8R | 3,000 | Debond, 亀裂 |
| **小計** | — | **13,000** | — |

### 1.6 Open Dataset リリース

| 公開先 | 内容 | 時期 |
|--------|------|------|
| Zenodo | FEMデータセット v1（フェアリング N=1,000） | Month 3 |
| HuggingFace | PyG形式のグラフデータ + メタデータ | Month 4 |
| GitHub | データ生成・前処理コード | Month 4 |

### Phase 1 マイルストーン

- [ ] GW全7欠陥タイプ実装・検証完了
- [ ] 音響疲労シミュレーション検証完了
- [ ] PI-GNN v1 学習・ベースライン比較
- [ ] フェアリング N=10,000 データセット完成
- [ ] マルチ構造体 N=13,000 データセット完成
- [ ] Zenodo/HuggingFace でデータ公開
- [ ] **論文1 投稿準備開始**（Composites Part B）

---

## Phase 2: マルチフィジックス + 全打上げシーケンス（Month 5–10 / 2026 Q3–2027 Q1）

**目的**: フェアリングが経験する全物理環境を再現し、Digital Twinの基盤を構築

### 2.1 打上げ全フェーズシミュレーション

```
T-0     T+60s    T+120s    T+180s    T+240s
│        │         │         │         │
リフトオフ Max-Q    超音速     フェアリング 軌道投入
│        │         │       分離│         │
▼        ▼         ▼         ▼         ▼
振動+音響 空力+音響  熱+空力   衝撃+分離  無重力+熱
│        │         │         │         │
└────────┴─────────┴─────────┴─────────┘
     全フェーズでGNN-SHMが損傷を追跡
```

| フェーズ | 時刻 | 主要荷重 | シミュレーション手法 |
|---------|------|---------|-------------------|
| リフトオフ | T-0 – T+10s | 振動 + 音響 147dB | Modal + Random Response |
| Max-Q | T+60s | 空力 30kPa + 音響 | Static + Acoustic |
| 超音速加熱 | T+120s | 空力加熱 200°C + 差圧 | Transient Thermal + Structural |
| フェアリング分離 | T+180s | 火工品衝撃 + 分離力学 | Explicit Dynamic + Contact |
| 軌道上 | T+240s~ | 熱サイクル + 微小重力 | Steady-State Thermal |

### 2.2 FSI（流体-構造連成）

| 項目 | 仕様 |
|------|------|
| CFD | OpenFOAM or SU2 (compressible RANS) |
| 連成 | 片方向: CFD → Abaqus（圧力 + 熱流束マッピング） |
| メッシュ | CFD: 構造化 100K cells, FEM: 既存 120K nodes |
| マッピング | preCICE or カスタム Python（最近傍補間） |
| 検証 | Epsilon Manual の Cp プロファイルと比較 |

### 2.3 非線形座屈解析

| 項目 | 仕様 |
|------|------|
| 手法 | Abaqus Riks法（弧長法） |
| 初期不整 | 第1座屈モード × 0.1t の不整を付与 |
| 欠陥影響 | Debonding 領域で座屈荷重がどれだけ低下するかを定量化 |
| 出力 | 荷重-変位曲線 + 座屈モード形状 → GNN特徴量に追加 |

### 2.4 損傷進展シミュレーション

| 項目 | 仕様 |
|------|------|
| 手法 | CZM疲労（Abaqus VCCT or Cohesive Fatigue） |
| 荷重 | 熱サイクル（-50°C ↔ +120°C）× 1000 cycles |
| 出力 | 剥離面積の時間発展 a(N) |
| Paris則 | da/dN = C(ΔG)^m, パラメータは文献値 |
| GNN統合 | 進展データ系列を時系列GNN（ST-GNN）で学習 |

### 2.5 合成センサデータ

FEMの変位場からPZTセンサの電圧応答を合成:

```
FEM 変位場 u(x,t)
      ↓ PZT圧電方程式
センサ電圧 V(t) = d₃₁ × ∫ ε(x,t) dA
      ↓ ADC量子化 + ノイズ付加
合成センサ信号 V̂(t)
      ↓ FFT + Hilbert
特徴量抽出 → GNN入力
```

### Phase 2 マイルストーン

- [ ] 打上げ全5フェーズの荷重条件定義完了
- [ ] FSI簡易版（OpenFOAM → Abaqus片方向）実装
- [ ] 非線形座屈解析 + 欠陥影響の定量化
- [ ] CZM疲労による損傷進展シミュレーション
- [ ] 合成センサデータ生成パイプライン
- [ ] ST-GNN（時空間GNN）プロトタイプ
- [ ] **論文1 投稿**（Composites Part B）
- [ ] **論文2 投稿準備開始**（CMAME）

---

## Phase 3: Physics Foundation Model（Month 9–16 / 2026 Q4–2027 Q3）★最重要

**目的**: 構造力学版の汎用基盤モデルを構築し、未知の構造体にゼロショットで損傷検出を実現

詳細設計: [Foundation-Model](Foundation-Model)

### 3.1 アーキテクチャ

```
┌─────────────────────────────────────────┐
│   Structural Foundation Model (SFM)     │
│                                         │
│   Input: Graph(nodes, edges, features)  │
│     ↓                                   │
│   Tokenizer: Patch-based graph tokens   │
│     ↓                                   │
│   Backbone: Graph Transformer           │
│     (12 layers, 768 dim, 12 heads)      │
│     ↓                                   │
│   Task Head:                            │
│     - Damage Detection (classification) │
│     - Stress Prediction (regression)    │
│     - Damage Localization (segmentation)│
│     - Remaining Life (regression)       │
└─────────────────────────────────────────┘
```

### 3.2 事前学習戦略

| ステージ | 手法 | データ | 目的 |
|---------|------|--------|------|
| Stage 1 | Masked Node Prediction | 全28K samples | 構造力学の汎用表現学習 |
| Stage 2 | Physics-Informed Contrastive | 同上 + PDE残差 | 物理法則の埋め込み |
| Stage 3 | Multi-Task Fine-Tuning | タスク別データ | 下流タスクへの適応 |

### 3.3 スケーリング則の検証

| モデルサイズ | パラメータ | 学習データ | 期待性能 |
|------------|-----------|-----------|---------|
| SFM-Small | 10M | 5K | ベースライン |
| SFM-Base | 50M | 15K | スケーリング検証 |
| SFM-Large | 200M | 28K | 最終モデル |

学習曲線から Power Law ($L \propto N^{-\alpha}$) の $\alpha$ を推定し、構造力学における Neural Scaling Law を初めて実証。

### 3.4 Zero-Shot 評価

学習に含まれない構造体での評価:

| テスト構造体 | ソース | 期待 F1 |
|------------|--------|---------|
| 衛星パネル（Al-HC） | 自作FEM | > 0.65 |
| 航空機翼構造（CFRP） | NASA CompDam | > 0.60 |
| 圧力容器（金属） | 公開データ | > 0.55 |

### Phase 3 マイルストーン

- [ ] Graph Transformer アーキテクチャ実装
- [ ] Stage 1 事前学習（Masked Node Prediction）完了
- [ ] Stage 2 Physics-Informed Contrastive Learning 完了
- [ ] スケーリング則の実証（3サイズ比較）
- [ ] Zero-Shot 評価（3構造体 × F1 > 0.60）
- [ ] **論文3 投稿**（Nature Computational Science）
- [ ] **NeurIPS ML4PS Workshop 投稿**
- [ ] **論文2 投稿**（CMAME）

---

## Phase 4: 自律損傷管理 + Digital Twin（Month 15–22 / 2027 Q2–2027 Q4）

**目的**: 検出→診断→予後→判断→修復→検証の完全自律ループを構築

詳細設計: [Autonomous-Damage-Management](Autonomous-Damage-Management)

### 4.1 リアルタイム推論パイプライン

```
GNN (PyTorch)
  ↓ torch.jit.trace
TorchScript
  ↓ onnx.export
ONNX Model
  ↓ TensorRT / OpenVINO
Optimized Runtime (<10ms)
  ↓ INT8量子化 + プルーニング
Edge Device (<1ms)
```

| デバイス | 推論速度 | 精度劣化 | 用途 |
|---------|---------|---------|------|
| RTX 4090 (FP32) | ~1ms | 0% | ベースライン |
| RTX 4090 (FP16) | ~0.5ms | <0.1% | 高速推論 |
| Jetson Orin (INT8) | ~5ms | <2% | 機上エッジ |
| FPGA (Xilinx) | <1ms | <3% | 超低遅延 |

### 4.2 ベイズ状態推定

| 手法 | 計算コスト | 精度 | 適用 |
|------|-----------|------|------|
| Extended Kalman Filter | 低 | 線形近似 | 初期プロトタイプ |
| Ensemble Kalman Filter | 中 | 非線形対応 | メイン手法 |
| Particle Filter | 高 | 最高精度 | 検証用 |

状態ベクトル: $\mathbf{x} = [a, \dot{a}, D, T, \sigma_{res}]$ (剥離長, 進展速度, 損傷度, 温度, 残留応力)

### 4.3 Self-Healing シミュレーション

ESA Project Cassandra (HealTech) の物理モデルを参考:

| パラメータ | 値 | ソース |
|-----------|-----|--------|
| 修復温度 | 100–140°C | CompPair HealTech |
| 修復時間 | 5–30 min | ESA 試験結果 |
| 修復率 | 70–95% | 損傷サイズ依存 |
| CZMモデル | 界面強度の部分回復 | カスタム UMAT |

### 4.4 Reinforcement Learning による修復最適化

| 要素 | 定義 |
|------|------|
| 状態 | 損傷位置・サイズ・進展速度・温度場 |
| 行動 | 加熱ゾーン選択 × 温度 × 時間 |
| 報酬 | 修復率 − エネルギーコスト − ダウンタイム |
| アルゴリズム | PPO (Proximal Policy Optimization) |

### Phase 4 マイルストーン

- [ ] ONNX/TensorRT 変換 + 推論 <10ms 達成
- [ ] Ensemble Kalman Filter 実装 + 状態推定検証
- [ ] Self-Healing CZM モデル実装
- [ ] RL による修復最適化プロトタイプ
- [ ] Edge AI（Jetson Orin）デモ
- [ ] センサ配置最適化（遺伝的アルゴリズム）
- [ ] **論文4 投稿**（AIAA Journal）

---

## Phase 5: 統合 + 論文 + OSS公開（Month 21–24 / 2028 Q1–Q2）

**目的**: 全Phaseの成果を統合し、世界に発信

### 5.1 Full System Demo

打上げシーケンス全体を Digital Twin で通しシミュレーション:

```
T-0: リフトオフ
  → GNN推論: 損傷なし ✅
  → ベイズ更新: P(damage) = 0.02

T+60s: Max-Q
  → GNN推論: 微小損傷検出 ⚠️
  → ベイズ更新: P(damage) = 0.15
  → 予後: 残存寿命 > 1000 cycles → 飛行継続 ✅

T+120s: 超音速加熱
  → GNN推論: 損傷進展 ⚠️
  → ベイズ更新: P(damage) = 0.35
  → 予後: 残存寿命 = 200 cycles → 監視継続 ⚠️

T+180s: フェアリング分離
  → 分離衝撃解析 → 損傷影響なし ✅
  → ミッション完遂 🎉
```

### 5.2 OSS公開計画

| 公開物 | プラットフォーム | 内容 |
|--------|---------------|------|
| **SFM-Base** | HuggingFace | 事前学習済み Foundation Model (50M params) |
| **StructuralBench** | HuggingFace Datasets | 28K samples, 5構造体, 7欠陥タイプ |
| **SHM-Pipeline** | GitHub | FEM → Graph → GNN → Digital Twin の全パイプライン |
| **Edge-SHM** | GitHub | Jetson/FPGA 向け推論エンジン |

### 5.3 量子GNN統合

| 構成 | 説明 |
|------|------|
| Classical GNN + VQC Head | GNN特徴抽出 → 量子分類器で最終判定 |
| Quantum Graph Attention | Attention重みを量子回路で計算 |
| Foundation Model + Quantum Layer | SFM の中間層に量子層を挿入 |

### Phase 5 マイルストーン

- [ ] Full System Demo（打上げ全シーケンス）完成
- [ ] Hardware-in-the-Loop 検証（CFRP試験片 + PZTセンサ）
- [ ] OSS公開（HuggingFace + GitHub）
- [ ] 量子GNN統合 + 性能比較
- [ ] JAXA実機搭載提案書作成
- [ ] **論文5 投稿**（Nature Communications）
- [ ] **AIAA SciTech 2028 発表**

---

## 計算資源計画

| Phase | GPU (vancouver) | CPU (frontale) | ストレージ |
|-------|----------------|---------------|-----------|
| 1 | GNN学習 + FNO | Abaqus並列 ×3台 | ~500 GB |
| 2 | ST-GNN + マルチスケール | FSI (OpenFOAM) | ~1 TB |
| 3 | Foundation Model学習 (4×4090, 数日) | — | ~200 GB |
| 4 | RL + Edge最適化 | — | ~100 GB |
| 5 | デモ + 検証 | — | ~50 GB |

---

## 関連ページ

| ページ | 内容 |
|--------|------|
| [2-Year-Goals](2-Year-Goals) | 全体目標・数値目標 |
| [Foundation-Model](Foundation-Model) | Phase 3 詳細設計 |
| [Autonomous-Damage-Management](Autonomous-Damage-Management) | Phase 4 詳細設計 |
| [Roadmap](Roadmap) | 現行フェーズ (Phase 2) の詳細 |
| [Architecture](Architecture) | 現行パイプライン |
| [Cutting-Edge-ML](Cutting-Edge-ML) | ML技術サーベイ |
