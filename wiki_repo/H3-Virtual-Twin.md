# H3 Virtual Twin — Architecture & Roadmap

> H3 ロケット全機のバーチャルツイン（デジタルツイン）構築計画
> 一人 + OSS + AI で実現する統合シミュレーション環境

---

## 1. Virtual Twin とは

物理ロケットの全サブシステムをソフトウェア上に再現し、
**打上げ前の予測 → 飛行中のリアルタイム比較 → 飛行後の診断** を一気通貫で行う。

```
┌─────────────────────────────────────────────────────┐
│              H3 Virtual Twin                         │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Structure │  │Propulsion│  │   GNC    │           │
│  │  FEM/GNN  │  │ CEA/CFD  │  │ 6DOF/MPC │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │                 │
│  ┌────┴──────────────┴──────────────┴─────┐          │
│  │        Flight Orchestrator              │          │
│  │   (時系列イベント + 状態遷移管理)        │          │
│  └────┬──────────────┬──────────────┬─────┘          │
│       │              │              │                 │
│  ┌────┴─────┐  ┌─────┴────┐  ┌─────┴────┐           │
│  │  Aero    │  │  Thermal │  │  Events  │           │
│  │ CFD/aero │  │ heat/cool│  │ sep/abort│           │
│  └──────────┘  └──────────┘  └──────────┘           │
│                                                      │
│  ┌──────────────────────────────────────────┐        │
│  │          SHM Layer (GNN/FNO)             │        │
│  │  構造健全性のリアルタイム推定・異常検知     │        │
│  └──────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

---

## 2. サブシステム一覧 — 現状と計画

### 凡例
- **Done**: 完了・動作確認済み
- **WIP**: 進行中
- **TODO**: 未着手だが一人で実現可能
- **Hard**: 実現困難 (ハード/計算資源の壁)

| # | サブシステム | 状態 | 既存資産 | 追加作業 |
|---|-------------|------|---------|---------|
| 1 | **Trajectory (軌道)** | Done | `jaxa_h3_orbital.py`, `optimize_h3_trajectory.py` | Monte Carlo 分散解析 |
| 2 | **Orbital Transfer** | Done | `h3_orbital_transfer.py` (poliastro) | — |
| 3 | **Structure: Fairing FEM** | Done | 250+ INP, CZM debonding model | — |
| 4 | **Structure: GNN-SHM** | Done | `train_gw.py`, 4 GNN architectures | Foundation Model |
| 5 | **Structure: FNO Surrogate** | WIP | `train_fno_gw.py` | 精度向上 |
| 6 | **Fairing Separation** | WIP | `generate_fairing_separation.py` (Explicit) | Phase 2-3 |
| 7 | **Landing Guidance** | Done | G-FOLD (cvxpy SOCP) | — |
| 8 | **RL Landing Control** | WIP | PPO (PyTorch), rl-attitude-control | GPU 学習完了待ち |
| 9 | **Attitude Control** | Done | `src/vt/attitude_control.py` | MPC 高度化 |
| 10 | **Propulsion: LE-9** | Done | `src/vt/propulsion.py` | CEA 連携 |
| 11 | **Propulsion: SRB-3** | Done | `src/vt/propulsion.py` | grain shape 詳細化 |
| 12 | **Propulsion: LE-5B-3** | Done | `src/vt/propulsion.py` | — |
| 13 | **Nozzle Design** | Done | `src/vt/propulsion.py` | CFD 検証 (Phase C) |
| 14 | **External Aerodynamics** | Done | `src/vt/aerodynamics.py` | CFD テーブル差替 |
| 15 | **Aerothermal (TPS)** | Done | `src/vt/aerothermal.py` | CFD 連成 (Phase C) |
| 16 | **SRB Separation** | Done | `src/vt/orchestrator.py` (簡易) | EXUDYN 詳細化 |
| 17 | **Stage Separation** | Done | `src/vt/orchestrator.py` (簡易) | EXUDYN 詳細化 |
| 18 | **Sloshing** | TODO | OpenFOAM VOF | タンク内液体動揺 |
| 19 | **Flight Orchestrator** | Done | `src/vt/orchestrator.py` | 軌道最適化 |
| 20 | **Quantum SHM** | TODO | PennyLane prototype | VQC ハイブリッド |

---

## 3. Flight Orchestrator — 統合エンジン設計

バーチャルツインの核心。全サブシステムを時系列で統合する。

### 3.1 飛行フェーズ定義

```
T-0        Liftoff
T+0~116s   Phase 1: SRB+S1 (LE-9×2 + SRB-3×2)
T+116s     Event:   SRB-3 Separation
T+116~207s Phase 2: S1 only (LE-9×2)
T+207s     Event:   Fairing Jettison
T+207~298s Phase 3: S1 coast (above atmosphere)
T+298s     Event:   MECO (Main Engine Cut-Off)
T+298s     Event:   Stage Separation (1段/2段)
T+298~850s Phase 4: S2 burn (LE-5B-3)
T+850s     Event:   SECO (Second Engine Cut-Off)
T+850s+    Phase 5: Orbit insertion / Coast
```

### 3.2 各フェーズで呼び出すサブシステム

| Phase | Trajectory | Aero | Propulsion | Structure | Thermal | GNC |
|-------|-----------|------|-----------|-----------|---------|-----|
| P1 SRB+S1 | gravity-turn | Cd(Ma) | LE-9+SRB-3 | 振動荷重 | 空力加熱 | pitch program |
| SRB Sep | event | 衝撃波 | SRB burnout | 衝撃応答 | — | 姿勢外乱 |
| P2 S1 | gravity-turn | Cd(Ma) | LE-9 | 振動荷重 | 空力加熱 | pitch program |
| Fair Sep | event | Cd 変化 | — | **分離ダイナミクス** | — | 姿勢外乱 |
| MECO/Sep | event | — | shutdown | 衝撃応答 | — | 姿勢外乱 |
| P4 S2 | coast+burn | 無視(真空) | LE-5B-3 | 微小重力荷重 | 放射冷却 | inertial GNC |
| SECO | orbit insert | — | shutdown | — | — | 軌道投入精度 |

### 3.3 実装方針

```python
# h3_virtual_twin.py — 統合オーケストレーター構想

class H3VirtualTwin:
    """H3 ロケット全機バーチャルツイン"""

    def __init__(self, config="H3-22S"):
        self.trajectory = TrajectoryModule()      # 既存: jaxa_h3_orbital.py
        self.propulsion = PropulsionModule()       # 新規: cea + bamboo
        self.aero = AerodynamicsModule()           # 新規: OpenFOAM/SU2 lookup
        self.structure = StructureModule()          # 既存: FEM + GNN-SHM
        self.thermal = ThermalModule()              # 新規: bamboo + 1D model
        self.gnc = GNCModule()                      # 既存 + 新規: PID/MPC
        self.events = EventManager()                # 新規: 分離シーケンス

    def run_mission(self, t_end=900.0, dt=0.1):
        """打上げから軌道投入まで一気通貫シミュレーション"""
        state = self.initial_state()
        for t in np.arange(0, t_end, dt):
            # 1. イベント判定 (分離、MECO 等)
            self.events.check(t, state)
            # 2. 推進力
            thrust, mdot = self.propulsion.get_thrust(t, state)
            # 3. 空力
            drag, lift = self.aero.get_forces(state.mach, state.alpha, state.alt)
            # 4. GNC コマンド
            cmd = self.gnc.update(t, state)
            # 5. 運動方程式積分
            state = self.trajectory.step(state, thrust, drag, lift, cmd, dt)
            # 6. 構造応答 (サロゲート: FNO)
            stress = self.structure.estimate_stress(state.accel, state.q_dyn)
            # 7. 熱応答
            temp = self.thermal.estimate_temp(state.mach, state.alt)
            # 8. SHM 判定
            health = self.structure.shm_check(stress, temp)
        return state
```

---

## 4. 開発ロードマップ

### Phase A: 基盤モジュール — ✅ 完了

```
A1. LE-9 1D サイクル解析              ✅ src/vt/propulsion.py
    └ エキスパンダーブリード、推力・Isp・温度プロファイル、Bartz熱解析
A2. SRB-3 推力プロファイル             ✅ src/vt/propulsion.py
    └ 台形推力プロファイル、高度 Isp 補間
A3. LE-5B-3 特性                     ✅ src/vt/propulsion.py
    └ 再着火可能、真空エンジンモデル
A4. H3 外部空力テーブル               ✅ src/vt/aerodynamics.py
    └ Cd/Cn/Cp vs Mach×α (33点×11点)、ISA大気モデル
A5. 1D 空力加熱モデル                 ✅ src/vt/aerothermal.py
    └ Sutton-Graves + Tauber-Sutton + lumped capacitance
A6. 姿勢制御 PID                     ✅ src/vt/attitude_control.py
    └ 3軸PID + TVC ジンバル + 重力ターン + RCS
```

### Phase B: 統合オーケストレーター — ✅ 完了

```
B1. Flight Orchestrator               ✅ src/vt/orchestrator.py
    └ 7イベント管理 + テレメトリ記録 + ミッションサマリー
B2. SRB 分離イベント                   ✅ (質量投棄 + 姿勢外乱)
B3. 段間分離イベント                   ✅ (2段質量設定 + 姿勢再同期)
B4. 全段統合テスト                     ✅ H3-22S: Max-Q 46.5kPa, V_SECO=8.4km/s
```

### Phase C: 高忠実度化 + OSS 活用 (2026 Q3-Q4)

```
C1. RocketPy 6DOF 統合               [RocketPy]            2週間
    └ 現行 3DOF → RocketPy 6DOF に差替え、風・モンテカルロ
C2. JSBSim 飛行力学                   [jsbsim]              2週間
    └ NASA 由来の高忠実度飛行力学エンジン (Python binding)
C3. OpenTsiolkovsky 参考実装          [OpenTsiolkovsky]     1週間
    └ インターステラテクノロジズの 3DOF/6DOF (Rust+WASM)
C4. 確率的 SHM デジタルツイン          [UAV-Digital-Twin]    3週間
    └ ベイズ推論で GNN-SHM 結果を飛行中に逐次更新
C5. LE-9 ノズル CFD                   [OpenFOAM/SU2]       1ヶ月
    └ 3D RANS → 推力補正テーブル
C6. フェアリング分離 CFD-Structure     [Abaqus+OpenFOAM]    1ヶ月
    └ 空力荷重 + 構造応答の連成
C7. AeroVECTOR TVC 検証               [AeroVECTOR]         1週間
    └ アクティブ制御 SIL テスト
C8. Monte Carlo 落下分散              [RocketPy/MAPLEAF]    2週間
    └ 風・推力偏差・分離タイミング不確実性
C9. SHM リアルタイム統合              [GNN + FNO]           2週間
    └ 飛行中の構造診断パイプライン
```

### Phase D: ダッシュボード + 可視化 (2027 Q1)

```
D1. NASA Open MCT ダッシュボード       [openmct]            3週間 ★
    └ NASA 公式ミッション管制フレームワーク (12,830★)
    └ テレメトリ表示、タイムライン、プラグイン拡張
    └ 推進/空力/構造/熱のリアルタイムパネル
D2. CesiumJS 3D 軌道可視化            [CesiumJS]           2週間
    └ 3D 地球上に軌道表示 (FlightClub 方式)
    └ CZML で時系列アニメーション
D3. Three.js 構造 3D モデル            [Three.js]           2週間
    └ フェアリング断面の応力/温度マップ
    └ GNN-SHM 推論結果のリアルタイム表示
D4. Takram H3 FIP 参考実装            [参考: JAXA実機]      ─
    └ 速度/高度/軌道の統合表示 (計画 vs 実績)
D5. 再利用シナリオ                     [G-FOLD + RL]        2週間
    └ 1段回収のバーチャルツイン拡張
D6. ForRocket 比較検証                [ForRocket]           1週間
    └ 日本人開発 6DOF で結果クロスチェック
```

---

## 5. 技術スタック

```
   Simulation Core            ML/AI Layer            Dashboard / Viz
┌──────────────────┐    ┌──────────────────┐    ┌───────────────────┐
│ Abaqus 2024      │    │ PyTorch 2.10     │    │ NASA Open MCT ★   │
│ OpenFOAM / SU2   │    │ PyG 2.7          │    │ CesiumJS (3D地球) │
│ NASA CEA         │    │ FNO (neuralop)   │    │ Three.js (構造3D) │
│ CVXPY            │    │ GNN (GCN/GAT)    │    │ Plotly / Dash     │
│ RocketPy (6DOF)  │    │ PennyLane (QC)   │    │ Google Earth/KML  │
│ JSBSim (6DOF)    │    │ PINN             │    │ ParaView          │
│ EXUDYN           │    │ Foundation Model │    │ TensorBoard       │
│ poliastro        │    │ Bayesian SHM     │    └───────────────────┘
│ ForRocket        │    └──────────────────┘
│ OpenTsiolkovsky  │
│ AeroVECTOR (TVC) │
│ MAPLEAF (batch)  │
└──────────────────┘

   OSS リポジトリ (papers/repos/ : 37 個)
┌──────────────────────────────────────────────────────┐
│ ★ openmct (NASA, 12,830★)  : ミッション管制ダッシュボード │
│ ★ jsbsim (NASA, 1,938★)    : 飛行力学エンジン          │
│ ★ RocketPy (898★)          : 6DOF + Monte Carlo       │
│ ★ UAV-Digital-Twin (MIT)    : 確率的 SHM デジタルツイン  │
│ ★ OpenTsiolkovsky (IST)     : 日本ロケット会社 OSS      │
│ ★ ForRocket (日本)          : 6DOF シミュレータ         │
│ ★ AeroVECTOR (166★)        : TVC + アクティブ制御       │
│   simupy-flight (NASA)      : SimuPy 飛行力学          │
│   Launch-Dashboard-API       : テレメトリ API 参考       │
│   MAPLEAF, EXUDYN, gncpy, gfold-py, SU2, ...         │
└──────────────────────────────────────────────────────┘

   Compute                    Data Flow
┌────────────────┐    ┌──────────────────────────────────┐
│ frontale01-04  │    │ INP → ODB → CSV → Graph          │
│  (Abaqus, CFD) │    │        → PyG → GNN/FNO           │
│ vancouver01-02 │    │                                   │
│  (GPU×4 学習)   │    │ CEA → thrust table → ODE         │
│ fifawc (local) │    │ CFD → aero table → lookup         │
│                │    │ GNN inference → Bayesian update   │
│                │    │ All telemetry → Open MCT → web    │
└────────────────┘    └──────────────────────────────────┘
```

### 参考システム

| 名称 | 開発元 | 概要 | 参考度 |
|------|--------|------|--------|
| **H3 FIP** | Takram + JAXA | H3 実機の飛行状態表示システム。3D地球+軌道+テレメトリ | ★★★ |
| **FlightClub.io** | Community | CesiumJS ベースの打上げ軌道シミュレータ・可視化 | ★★★ |
| **NASA DT Paradigm** | NASA/USAF (AIAA 2012-1818) | デジタルツイン概念の原典論文 | ★★★ |
| **JAXA JEDI** | JAXA | LS-Flow CFD + JSS スパコンによる H3 数値シミュレーション | ★★ |
| **DLR MODELICA** | DLR/ESA | 再利用ロケットのマルチボディ統合シミュレーション | ★★ |

---

## 6. 計算量の見積もり

| タスク | 計算時間 | マシン | 備考 |
|--------|---------|--------|------|
| LE-9 CEA 解析 | ~1分 | ローカル | 化学平衡のみ |
| LE-9 bamboo 1D 熱解析 | ~10分 | ローカル | 冷却チャネル設計 |
| H3 空力 CFD (1条件) | ~2-4時間 | frontale | OpenFOAM RANS |
| H3 空力テーブル (50条件) | ~1週間 | frontale | 並列投入 |
| LE-9 ノズル CFD (3D) | ~12-24時間 | frontale | RANS, 500K cells |
| フェアリング FEM (1ケース) | ~8分 | frontale | C3D10, 2.7M DOF |
| GNN 学習 (100 epochs) | ~30分 | vancouver02 | RTX 4090 |
| FNO 学習 (200 epochs) | ~2時間 | vancouver02 | RTX 4090 |
| 全段統合シミュ (1回) | ~10秒 | ローカル | サロゲート使用時 |
| Monte Carlo (1000回) | ~3時間 | ローカル | 並列化可能 |

---

## 7. 成功指標

| レベル | 基準 | 状態 |
|--------|------|------|
| **Lv1** | 全段統合: T-0→SECO 一気通貫実行 | ✅ 達成 (Phase B) |
| **Lv2** | イベント再現: SRB/フェアリング/段間分離 | ✅ 達成 (Phase B) |
| **Lv3** | Max-Q ±20%: 46.5 kPa (実機 ~40-50 kPa) | ✅ 達成 |
| **Lv4** | 推進忠実度: LE-9 推力・Isp が公称値 ±3% | ✅ 達成 |
| **Lv5** | 6DOF 高忠実度: RocketPy/JSBSim 統合 | TODO (Phase C) |
| **Lv6** | 空力忠実度: CFD テーブル差替え ±10% | TODO (Phase C) |
| **Lv7** | SHM 統合: GNN → ベイズ推論リアルタイム | TODO (Phase C) |
| **Lv8** | Open MCT ダッシュボード動作 | TODO (Phase D) ★ |
| **Lv9** | CesiumJS 3D 軌道 + Three.js 構造可視化 | TODO (Phase D) |
| **Lv10** | 再利用拡張: 1段回収シナリオ | TODO (Phase D) |

---

## 8. 一人で作る意義

1. **全系統を理解した上での統合** — 分業では見えないサブシステム間の相互作用を把握
2. **H3-8 事故のような異常の早期検知** — SHM + 飛行データの統合判断
3. **再利用ロケット時代の必須技術** — 毎飛行の構造診断を AI で自動化
4. **ロケット会社の CTO レベルの技術俯瞰** — 全領域をシミュレーションで体験
5. **論文/特許のネタ** — 「一人で構築した H3 バーチャルツイン」は十分にユニーク

---

## 9. リスクと対策

| リスク | 対策 |
|--------|------|
| 空力 CFD の計算コスト | → 低忠実度モデル(Missile DATCOM)で開始、後から CFD 補正 |
| LE-9 の詳細仕様が非公開 | → 公開論文 + CEA で理論値を推定、公称値に合わせてキャリブレーション |
| 統合時のインターフェース設計 | → 各モジュールを独立した Python クラスに、JSON で状態受け渡し |
| モチベーション維持 | → 各 Phase で動くデモを作り、小さい成功体験を積む |
