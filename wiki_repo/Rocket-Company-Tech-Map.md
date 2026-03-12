# Rocket Company — Technology Map

> ロケット会社立ち上げに向けた技術領域マップ・調査状況
> 最終更新: 2026-03-13

---

## 1. ビジョン

ロケット会社を立ち上げるために、全技術領域を一通り習得する。
現在は **構造** と **軌道** を主軸に、FEM・GNN・シミュレーションの実装力を積み上げている。

---

## 2. 技術領域マップ

| # | 領域 | ステータス | 概要 |
|---|------|-----------|------|
| 1 | **構造 (Structures)** | **主軸** | CFRP/Al-HC フェアリング FEM + GNN-SHM |
| 2 | **軌道 (Orbit & GNC)** | **主軸** | 軌道投入シミュレーション、着陸誘導 |
| 3 | 燃焼 (Propulsion) | 未着手 | エンジン設計・燃焼解析 |
| 4 | イベント (Flight Events) | 未着手 | 分離・展開・アボートシーケンス |
| 5 | 地上 (Ground Systems) | 未着手 | 射場設備・打上げオペレーション |
| 6 | 経営 (Business) | 未着手 | コスト・スケジュール・ライセンス |

---

## 3. 構造 (Structures) — 主軸

### 3.1 GNN-SHM (現メインプロジェクト)

JAXA H3 ロケット CFRP/Al-Honeycomb フェアリングのデボンディング欠陥検出。

| 項目 | 状態 | 詳細 |
|------|------|------|
| FEM モデル (CZM) | 完了 | 1/12 セクター、COH3D8 接着層、C3D10 ソリッドコア → [Realistic-Fairing-FEM](Realistic-Fairing-FEM) |
| バッチ INP | 完了 (250個) | 161 Debond + 41 FOD + 38 Impact + 10 他 → [Batch-INP-Status](Batch-INP-Status) |
| GNN 学習 | 完了 (初期) | GCN / GAT / GIN / SAGE 4種比較 → [Architecture](Architecture) |
| ガイド波 FEM | 進行中 | GW 伝播シミュレーション → [Guided-Wave-Simulation](Guided-Wave-Simulation) |
| FNO サロゲート | 進行中 | GW FNO + Foundation Model → [Foundation-Model](Foundation-Model) |
| 量子コンピューティング | 計画 | VQC ハイブリッド GNN → [Quantum-Integration](Quantum-Integration) |

### 3.2 フェアリング分離ダイナミクス (NEW)

H3-8 号機事故 (2025/12) の分離異常振動 (6–7 Hz) を FEM で再現。

| 項目 | 状態 | 詳細 |
|------|------|------|
| Abaqus/Explicit モデル | INP 生成完了 | 1/4 モデル (90°×2 セクター)、117K 行 |
| 対称 BC | 修正済 | XSYMM at θ=0° & θ=180° |
| ソルバー実行 | ライセンス待ち | Job 11649 on frontale03 |
| MODEL CHANGE (ボルト/パイロ) | TODO (Phase 2) | Frangible bolt 除去による分離再現 |
| DOE (異常ケース) | TODO (Phase 3) | ボルト不発、バネ非対称 |
| GNN-SHM 統合 | 将来展望 (Phase 4) | 分離異常の自動検出 |

→ 詳細: [Fairing-Separation-Dynamics](Fairing-Separation-Dynamics)

### 3.3 将来展望

| テーマ | 関連論文 | 優先度 |
|--------|---------|--------|
| トポロジー最適化 | NatureComms_ML_TopologyOpt.pdf | 中 (将来) |
| フェアリング軽量化 | CompositeFairing_HyperSizer.pdf, BeyondGravity_ReusableFairing_MDAO.pdf | 中 (将来) |
| 音響振動 | — | 低 (将来) |
| TPS (耐熱防護) | NASA2023_TPS_StateOfIndustry.pdf, SpaceX_PICAX_HeatShield.pdf, NASA_LOFTID_Aerothermo.pdf | 低 (将来) |

---

## 4. 軌道 (Orbit & GNC) — 主軸

### 4.1 軌道投入シミュレーション (完了)

H3-22S の gravity-turn 軌道投入を RocketPy + カスタムソルバーで再現。
Google Earth KML で可視化済み。精度 90%+。

→ 詳細: [Rocket-Launch-Simulation](Rocket-Launch-Simulation)

| シミュレーション | ツール | 結果 |
|----------------|--------|------|
| H3-22S 軌道投入 | Custom gravity-turn + scipy optimize | LEO 293×365 km, e=0.005 |
| ピッチ最適化 | differential_evolution (7パラメータ) | MECO 誤差 3.5% (実機比) |
| 軌道遷移 | poliastro | GTO Δv=2.43 km/s, TLI Δv=3.11 km/s |

### 4.2 着陸誘導 (趣味枠)

H3 は再利用ではないが、SpaceX Falcon 9 の着陸制御を勉強ベースで実装。

| テーマ | ツール/手法 | 状態 |
|--------|-----------|------|
| G-FOLD (燃料最適大機動誘導) | cvxpy SOCP | 完了: 1,523 kg fuel, 0.0 m/s landing |
| RL 着陸制御 | PPO (PyTorch) | 学習中 (GPU 必要) |
| Flip & Landing | — | 論文調査済 |

### 4.3 姿勢制御

| 論文 | 手法 |
|------|------|
| DeepRL_Attitude_KeepOut_arXiv.pdf | Deep RL + Keep-out zone 回避 |
| DeepRL_SpacecraftAttitude_Glasgow.pdf | Glasgow 大 Deep RL |
| NASA_AutonomousAttitude_DeepRL.pdf | NASA 自律姿勢制御 |

---

## 5. 燃焼 (Propulsion) — 未着手

### 5.1 収集済み文献

| 論文 | 内容 |
|------|------|
| NASA_CFD_Injector_Optimization.pdf | インジェクター形状 CFD 最適化 |
| NozzleDesignOptimization_Thesis.pdf | ノズル設計最適化 |

### 5.2 やるべきこと

- LE-9 エキスパンダーブリードサイクルの熱流体解析
- インジェクター/ノズル設計パラメトリック スタディ
- 燃焼安定性解析（高周波不安定性）
- OpenFOAM or SU2 による CFD

---

## 6. イベント (Flight Events) — 未着手

### 6.1 分離イベント

フェアリング分離ダイナミクスが最初のエントリポイント（§3.2 で着手済み）。

| イベント | 関連技術 | 状態 |
|---------|---------|------|
| フェアリング分離 | Abaqus/Explicit, MODEL CHANGE | **着手済** |
| SRB 分離 | 衝撃解析、火工品 | 未着手 |
| 段間分離 | マルチボディダイナミクス | 未着手 |
| 衛星分離 | バネ放出、回転制御 | 未着手 |

### 6.2 収集済み文献

| 論文 | 内容 |
|------|------|
| FairingSep_Reliability_MultiUncertainty.pdf | 分離信頼性・不確定性 |
| Pyroshock_ExplosiveBolts_Hydrocodes.pdf | 火工品衝撃解析 |
| MechanicalShock_FEA.pdf | 機械衝撃 FEA |
| VEGA_Fairing_ModalCoupling.pdf | VEGA フェアリング モーダルカップリング |
| NASA_PropellantSloshDynamics.pdf | 推進剤スロッシング |
| NASA_SDO_PropellantSlosh.pdf | SDO スロッシング |
| Orion_PropellantSloshing_FLOW3D.pdf | Orion スロッシング CFD |

---

## 7. 地上 (Ground Systems) — 未着手

- 射場設備設計
- 推進剤ハンドリング (LH2/LOX)
- 打上げオペレーション自動化
- テレメトリ・追跡管制

---

## 8. 経営 (Business) — 未着手

- コスト構造分析 (SpaceX vs H3)
- 打上げサービス市場分析
- 宇宙活動法ライセンス取得フロー
- 射場選定 (種子島 / 大樹町 / 海外)

---

## 9. 収集済みリソース一覧

### 9.1 論文 (35本)

| カテゴリ | 本数 | ディレクトリ |
|---------|------|-------------|
| SpaceX 関連 | 8 | `papers/spacex/` |
| ロケット全般 | 16 | `papers/rocket_general/` |
| フェアリング分離 | 8 | `papers/fairing_separation/` |

### 9.2 GitHub リポジトリ (14個)

| リポジトリ | 分野 | 概要 |
|-----------|------|------|
| **RocketPy** | 軌道 | 6-DOF 飛行シミュレーション (Python) |
| **poliastro** | 軌道 | 軌道力学ライブラリ |
| **MAPLEAF** | 軌道 | 6-DOF + 制御 |
| **gfold-py** | 着陸 | G-FOLD (燃料最適大機動誘導) Python 実装 |
| **lcvx-pdg** | 着陸 | Lossless Convexification for Powered Descent |
| **rocket-recycling** | 着陸 | Falcon 9 着陸シミュレーション |
| **Landing-Starships** | 着陸 | Starship flip-landing |
| **RocketLander** | 着陸 | RL ベース着陸制御 |
| **TensorAeroSpace** | 制御 | Tensor ベース宇宙航空制御 |
| **rl-attitude-control** | 制御 | RL 姿勢制御 |
| **deep_learning_topology_opt** | 構造 | DL トポロジー最適化 |
| **EXUDYN** | 構造 | マルチボディダイナミクスエンジン |
| **pydy** | 構造 | Python マルチボディダイナミクス |
| **awesome-space** | 総合 | 宇宙関連 awesome list |

### 9.3 主要参照文献 (フェアリング分離)

| 文献 | 重要度 | 要点 |
|------|--------|------|
| **JAXA H3-8 調査報告** (2025/12/25) | 最重要 | 分離時 6–7 Hz 異常振動（正常 18 Hz）、1.5 s 持続（正常 0.1 s）、PSS 破壊 |
| **KTH thesis** | 高 | Abaqus ビーム要素除去による分離 FE モデリング手法 |
| **Kawasaki H3 Fairing** | 高 | Frangible bolt (V-notch) メカニズム詳細、CFRP/HC 構造仕様 |

---

## 10. ロードマップ

### Phase 1: 構造 + 軌道 (現在)

```
2026 Q1–Q2
├── GNN-SHM メインプロジェクト (継続)
│   ├── ガイド波 FEM バッチ生成
│   ├── FNO サロゲート + Foundation Model
│   └── 量子 VQC 試行
├── フェアリング分離ダイナミクス (NEW)
│   ├── Phase 1: プリロード + 分離解析 ← 今ここ
│   ├── Phase 2: MODEL CHANGE (ボルト/パイロ)
│   └── Phase 3: DOE (異常ケース)
└── 軌道シミュレーション (完了、拡張可能)
    ├── H3-22S 軌道投入 ✅
    ├── G-FOLD 着陸誘導 ✅
    └── RL 着陸制御 (GPU 学習中)
```

### Phase 2: 燃焼 + イベント (次)

```
2026 Q3–Q4
├── 燃焼: CFD (OpenFOAM/SU2)
│   ├── インジェクター設計
│   └── ノズル最適化
└── イベント: 分離シーケンス
    ├── SRB 分離衝撃
    └── スロッシング解析
```

### Phase 3: 地上 + 経営 (将来)

```
2027–
├── 地上設備設計
├── 打上げオペレーション
└── 事業計画・ライセンス
```
