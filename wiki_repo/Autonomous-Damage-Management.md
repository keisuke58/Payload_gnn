[← Home](Home) · [2-Year-Goals](2-Year-Goals) · [Roadmap-2028](Roadmap-2028)

# Autonomous Damage Management (ADM) — Phase 4 詳細設計

> 最終更新: 2026-03-07
> **検出 → 診断 → 予後 → 判断 → 修復 → 検証 の完全自律ループ**

---

## 日本語概要

従来のSHMは「損傷の検出」で終わるが、本システムは**6段階の完全自律ループ**を閉じる。GWセンサで損傷を検出し、GNN/Foundation Modelで診断・位置特定し、Paris則で残存寿命を予測し、リスク評価で飛行可否を判断し、Self-Healing材料で修復し、再検査で修復完了を確認する。ESA Project Cassandra の HealTech やNASA JSTAR Digital Twin を参考に、世界初の宇宙機向け自律損傷管理シミュレーションシステムを構築する。

---

## 1. 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│              Autonomous Damage Management               │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ ① DETECT │───→│② DIAGNOSE│───→│③ PROGNOSE│          │
│  │ GW + GNN │    │ Type/Loc │    │ RUL Est  │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│       ↑                               │                 │
│       │                               ▼                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ ⑥ VERIFY │←───│ ⑤ HEAL   │←───│ ④ DECIDE │          │
│  │ GW Rescan│    │ Self-Heal│    │ Risk Eval│          │
│  └──────────┘    └──────────┘    └──────────┘          │
│                                                         │
│  ベイズ状態推定（全ステージを通じてオンライン更新）       │
│  Digital Twin（物理モデルとの双方向同期）                │
└─────────────────────────────────────────────────────────┘
```

---

## 2. ステージ詳細

### ① Detect — 損傷検出

| 項目 | 仕様 |
|------|------|
| センサ | PZT アレイ（pitch-catch, 50–300 kHz） |
| 信号処理 | Hilbert変換 → エンベロープ抽出 → TOF/振幅変化 |
| ML | GNN / Foundation Model でノードレベル二値分類 |
| 出力 | P(damage) per node — 確率マップ |
| 速度 | <10ms（ONNX Runtime） |

**ベースライン比較**:

| 手法 | F1 | 推論時間 | 備考 |
|------|-----|---------|------|
| 閾値法 (ToF) | ~0.40 | <1ms | 感度低い |
| CNN (2D展開) | ~0.70 | ~5ms | 曲面歪み |
| **GNN (SAGE)** | ~0.85 | ~10ms | 現行最良 |
| **Foundation Model** | ~0.90 | ~10ms | ゼロショット含む |

### ② Diagnose — 損傷診断

検出されたら、損傷のタイプ・サイズ・位置を特定:

| 出力 | 手法 | 精度目標 |
|------|------|---------|
| 欠陥タイプ | 7クラス分類（GNN Head） | Accuracy > 0.85 |
| 欠陥位置 | ノードセグメンテーション → 重心計算 | 誤差 < 20mm |
| 欠陥サイズ | セグメンテーション面積 | 誤差 < 15% |
| 深さ | 外皮/内皮/コア の3レベル分類 | Accuracy > 0.80 |

### ③ Prognose — 残存寿命予測

損傷の時間発展を予測:

**Paris則ベースのき裂進展**:
```
da/dN = C × (ΔG)^m

a: 剥離長さ [mm]
N: 荷重サイクル数
ΔG: エネルギー解放率振幅 [J/m²]
C, m: 材料定数 (CFRP/Al-HC界面)
```

| パラメータ | 値 | ソース |
|-----------|-----|--------|
| C | 1.2 × 10⁻⁸ | 文献値 (CFRP/epoxy) |
| m | 3.5 | 文献値 |
| G_Ic | 800 J/m² | Mode I 臨界値 |
| G_IIc | 1200 J/m² | Mode II 臨界値 |

**GNN ベースの予測**:
- 入力: 現在の損傷状態 + 荷重履歴 + 環境条件
- 出力: 残存荷重サイクル数 (RUL)
- 学習: Phase 2 の CZM 疲労シミュレーション結果

### ④ Decide — 飛行可否判断

リスクベースの意思決定フレームワーク:

```
┌─────────────────────────────────────┐
│ Risk Assessment Matrix              │
│                                     │
│ P(failure)    Action                │
│ ─────────    ──────                 │
│ < 10⁻⁶      Continue (Green)       │
│ 10⁻⁶–10⁻⁴   Monitor (Yellow)      │
│ 10⁻⁴–10⁻²   Repair if possible    │
│ > 10⁻²      Abort / No-Go (Red)    │
│                                     │
│ Residual Strength:                  │
│ σ_res / σ_design > SF (1.5)        │
│ → Continue                         │
│ σ_res / σ_design < SF (1.5)        │
│ → Repair or Abort                  │
└─────────────────────────────────────┘
```

| 判定基準 | 値 | 根拠 |
|---------|-----|------|
| 安全率 SF | 1.5 | JAXA-JMR-002 |
| P(failure) 閾値 | 10⁻⁴ | 航空宇宙規格 |
| 最小残存強度 | 60% σ_design | No-Growth 要件 |
| 最大許容剥離面積 | 構造依存 | FEM座屈解析で決定 |

### ⑤ Heal — 自己修復

ESA Project Cassandra / CompPair HealTech の物理モデル:

```
修復プロセス:
  損傷検出 → 加熱ゾーン選択 → 温度制御 → 修復材活性化 → 冷却

  ┌──────────────────────────────┐
  │  Self-Healing CFRP           │
  │                              │
  │  ┌─────┐  加熱   ┌─────┐   │
  │  │Crack│ ──────→ │Healed│  │
  │  └─────┘  100°C  └─────┘   │
  │           30min             │
  │                              │
  │  Healing Agent:              │
  │  - Thermoplastic resin       │
  │  - Reflow at 100-140°C       │
  │  - Recovery: 70-95%          │
  └──────────────────────────────┘
```

**CZM修復モデル**:

| パラメータ | 修復前 | 修復後 |
|-----------|--------|--------|
| 界面強度 t_n | 0 (完全剥離) | 0.7–0.95 × t_n0 |
| 界面靱性 G_Ic | 0 | 0.7–0.95 × G_Ic0 |
| CZM状態変数 D | 1.0 (破壊) | 0.05–0.30 (部分回復) |

**Reinforcement Learning による修復最適化**:

| RL要素 | 定義 |
|--------|------|
| **状態** s | [損傷位置(x,y,z), サイズ(a), 進展速度(da/dN), 温度場T(x), 残存強度σ_res] |
| **行動** a | [加熱ゾーン(i,j), 目標温度(100-140°C), 加熱時間(5-30min)] |
| **報酬** r | η_repair × w₁ − E_heat × w₂ − t_down × w₃ |
| **アルゴリズム** | PPO (Proximal Policy Optimization) |
| **環境** | CZM FEM シミュレーション（高速化版） |

### ⑥ Verify — 修復検証

修復完了後にGWセンサで再検査:

| 検証項目 | 基準 | 手法 |
|---------|------|------|
| 波動伝搬回復 | ToF誤差 < 2% vs 健全時 | Pitch-catch再計測 |
| 散乱パターン消失 | 散乱振幅 < 閾値 | Hilbert包絡線 |
| GNN再推論 | P(damage) < 0.05 | Foundation Model |
| 残存強度回復 | σ_res > SF × σ_design | FEM再計算 |

---

## 3. Digital Twin エンジン

### 3.1 リアルタイムアーキテクチャ

```
Physical Structure (Fairing)
    │
    │ PZT Sensor Data (50-300 kHz, 5-20 channels)
    │ Thermocouples (temperature field)
    │ Accelerometers (vibration)
    ▼
┌──────────────────────────┐
│ Data Acquisition Layer   │
│ ADC → FFT → Feature Ext  │
│ Sampling: 1 MHz          │
│ Update rate: 10 Hz       │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Inference Layer          │
│ GNN/SFM (ONNX Runtime)  │
│ Latency: <10ms           │
│ Output: P(damage), type  │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ State Estimation Layer   │
│ Ensemble Kalman Filter   │
│ State: [a, ȧ, D, T, σ]  │
│ Update: every sensor read│
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Decision Layer           │
│ Risk assessment          │
│ Heal/Continue/Abort      │
│ Response time: <100ms    │
└──────────────────────────┘
```

### 3.2 ベイズ状態推定

**Ensemble Kalman Filter (EnKF)**:

状態ベクトル:
```
x = [a₁, a₂, ..., aₖ,     # k 箇所の剥離長さ
     ȧ₁, ȧ₂, ..., ȧₖ,     # 進展速度
     D₁, D₂, ..., Dₖ,     # 損傷度 (0-1)
     T₁, T₂, ..., Tₘ,     # m 点の温度
     σ_res]                # 残存強度

x ∈ R^{3k + m + 1}
```

| EnKF パラメータ | 値 |
|----------------|-----|
| アンサンブルサイズ | 50 |
| プロセスノイズ Q | 対角, σ_a=0.1mm, σ_T=0.5°C |
| 観測ノイズ R | センサ精度から決定 |
| 更新頻度 | 10 Hz |

---

## 4. Edge AI 実装

### 4.1 モデル最適化パイプライン

```
PyTorch Model (FP32, 50M params)
  ↓ Knowledge Distillation
Compact Model (FP32, 5M params)
  ↓ Quantization-Aware Training
INT8 Model (5M params, 4x smaller)
  ↓ Pruning (structured)
Sparse INT8 Model (2M effective params)
  ↓ Platform-specific compilation
  ├── TensorRT (Jetson Orin)
  └── Vitis AI (Xilinx FPGA)
```

### 4.2 ターゲットデバイス

| デバイス | 演算性能 | メモリ | 消費電力 | 推論速度 |
|---------|---------|--------|---------|---------|
| Jetson Orin NX | 100 TOPS (INT8) | 16 GB | 25W | ~5ms |
| Xilinx ZCU104 | — | 4 GB | 15W | <1ms |
| Intel Movidius | 4 TOPS | 1 GB | 1.5W | ~20ms |

### 4.3 精度-速度トレードオフ

| モデル | F1 | 推論速度 | サイズ |
|--------|-----|---------|-------|
| SFM-Base (FP32, GPU) | 0.90 | 10ms | 200MB |
| SFM-Distilled (FP32) | 0.87 | 5ms | 20MB |
| SFM-Distilled (INT8) | 0.86 | 2ms | 5MB |
| SFM-Pruned (INT8) | 0.84 | <1ms | 3MB |

---

## 5. Uncertainty Quantification

### 5.1 手法

| 手法 | 計算コスト | 精度 | 実装 |
|------|-----------|------|------|
| MC Dropout | 低 (10回推論) | 中 | PyTorch native |
| Deep Ensemble | 高 (5モデル) | 高 | 5×学習 |
| Evidential DL | 低 (1回推論) | 中 | カスタム損失 |
| Conformal Prediction | 低 (1回推論) | 理論保証あり | 後処理 |

### 5.2 出力形式

```
Detection Output:
  P(damage) = 0.82 ± 0.05 (MC Dropout, 95% CI)
  Epistemic uncertainty = 0.03 (model uncertainty)
  Aleatoric uncertainty = 0.02 (data noise)

Prognosis Output:
  RUL = 1200 ± 300 cycles (Ensemble, 95% CI)
  P(failure within 100 cycles) = 0.003
```

---

## 6. 実装ロードマップ

| 月 | タスク | 成果物 |
|----|--------|--------|
| Month 15 | ONNX変換 + TensorRT最適化 | 推論 <10ms |
| Month 16 | EnKF実装 + シミュレーション検証 | 状態推定デモ |
| Month 17 | CZM Self-Healing モデル実装 | 修復シミュレーション |
| Month 18 | RL修復最適化 (PPO) | 最適修復ポリシー |
| Month 19 | Edge AI (Jetson Orin) デモ | 組込み推論 |
| Month 20 | Digital Twin統合 (全6ステージ) | End-to-End デモ |
| Month 21 | センサ配置最適化 (GA) | 最適配置 |
| Month 22 | UQ統合 + 論文執筆 | AIAA Journal 投稿 |

---

## 7. 先行事例・参考

| プロジェクト | 組織 | 本研究との関連 |
|------------|------|--------------|
| [Project Cassandra](https://www.esa.int/Enabling_Support/Space_Transportation/Future_space_transportation/Self-repairing_spacecraft_could_change_future_missions) | ESA / CompPair | HealTech 自己修復CFRP — ⑤Heal の物理モデル |
| [Self-Healing Composite](https://news.ncsu.edu/2026/01/healing-composite-lasts-centuries/) | NC State | 数百年持続する修復材料 — 材料パラメータ参考 |
| [NASA JSTAR](https://www.nasa.gov/jstar-digital-twins/) | NASA | 構造Digital Twinフレームワーク — アーキテクチャ参考 |
| [Alabama Self-Heal Aircraft](https://news.ua.edu/2025/10/flying-into-the-future-aircraft-that-detect-damage-and-self-heal/) | U. Alabama | 航空機の自己検出・自己修復 — コンセプト参考 |

---

## 8. 関連ページ

| ページ | 内容 |
|--------|------|
| [2-Year-Goals](2-Year-Goals) | 全体目標 |
| [Roadmap-2028](Roadmap-2028) | 詳細ロードマップ |
| [Foundation-Model](Foundation-Model) | Phase 3: Foundation Model（ADMの推論エンジン） |
| [Uncertainty-Quantification](Uncertainty-Quantification) | UQ手法の詳細 |
| [Sensor-Data-Acquisition](Sensor-Data-Acquisition) | PZTセンサ設計 |
| [SHM-Context](SHM-Context) | SHM物理背景 |
