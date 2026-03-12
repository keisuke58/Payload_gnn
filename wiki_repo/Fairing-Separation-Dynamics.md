# Fairing Separation Dynamics — Abaqus/Explicit Model

> H3 フェアリング分離ダイナミクスの FEM モデル
> 最終更新: 2026-03-12

---

## 1. 背景・動機

### 1.1 H3-8 号機事故 (2025/12)

JAXA 調査報告（2025/12/25）で判明した分離時の異常振動:

| 項目 | 正常 | F8 実測 |
|------|------|---------|
| **分離振動周波数** | ~18 Hz | **6–7 Hz** |
| **振動持続時間** | ~0.1 s | **~1.5 s** |
| **結果** | 正常分離 | PSS 破壊、衛星脱落 |

### 1.2 分離メカニズム

H3 フェアリングの分離は3段階:

1. **水平分離**: Frangible bolt (V-notch) × 72本 → 一斉破断
2. **縦分離**: Pyro-cord (火工線) による縦シーム切断
3. **開放**: バネ + ヒンジ回転で 2 半殻を展開

異常ケース（ボルト不発、バネ非対称等）で 6–7 Hz 低周波振動が発生する仮説を FEM で検証。

---

## 2. FEM モデル仕様

### 2.1 モデル概要

| 項目 | 値 |
|------|-----|
| **ソルバー** | Abaqus/Explicit |
| **対称性** | 1/4 モデル (90° × 2 セクター) |
| **セクター** | Q1: θ=0°–90°, Q2: θ=90°–180° |
| **シーム** | θ=90° (Q1–Q2 接触面) |
| **対称 BC** | XSYMM at θ=0° (Q1 端), XSYMM at θ=180° (Q2 端) |
| **DOF 概算** | ~50K (full model ~2M DOF の 1/4) |

### 2.2 形状パラメータ

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| `RADIUS` | 2,600 mm | 内スキン半径 |
| `CORE_T` | 38 mm | Al ハニカムコア厚 |
| `FACE_T` | 1.0 mm | CFRP スキン厚（各面） |
| `BARREL_Z_MAX` | 5,000 mm | バレル高さ |
| `FAIRING_DIAMETER` | 5,200 mm | 全径 |
| `ADAPTER_HEIGHT` | 100 mm | アダプターリング高 |

### 2.3 パーツ構成 (7 parts)

| Part | 要素タイプ | 材料 | 説明 |
|------|-----------|------|------|
| Q1-InnerSkin | S4R (shell) | Mat-CFRP | 内スキン (θ=0°–90°) |
| Q1-Core | C3D8R (solid) | Mat-Honeycomb | ハニカムコア |
| Q1-OuterSkin | S4R (shell) | Mat-CFRP | 外スキン |
| Q2-InnerSkin | S4R (shell) | Mat-CFRP | 内スキン (θ=90°–180°) |
| Q2-Core | C3D8R (solid) | Mat-Honeycomb | ハニカムコア |
| Q2-OuterSkin | S4R (shell) | Mat-CFRP | 外スキン |
| Adapter | S4R (shell) | Mat-Al7075 | アダプターリング |

### 2.4 材料定数

| 材料 | E / E_ij (GPa) | ν | ρ (ton/mm³) | Damping α |
|------|----------------|---|-------------|-----------|
| **CFRP** | 160/10/10 (Engineering Constants) | 0.3 | 1.6e-9 | 100 |
| **Honeycomb** | 1.0/0.01/0.01 | 0.001 | 5e-11 | 100 |
| **Al7075** | 71.7 | 0.33 | 2.81e-9 | 100 |
| **Steel** | 200 | 0.3 | 7.85e-9 | — |

### 2.5 拘束・荷重

| 種類 | 内容 |
|------|------|
| **Tie (4本)** | Q1/Q2 × Inner/Outer スキン → コア 接着 |
| **BC: Adapter** | ENCASTRE (全自由度固定) |
| **BC: Symmetry** | XSYMM at θ=0° (Q1), XSYMM at θ=180° (Q2) |
| **Gravity** | 29,430 mm/s² (3G) in -Y direction |

### 2.6 ステップ

| Step | 時間 | Mass Scaling | 内容 |
|------|------|-------------|------|
| **Step-Preload** | 5 ms | factor=0.0001 | 重力 + 差圧プリロード |
| **Step-Separation** | 200 ms | — | 分離ダイナミクス |

### 2.7 出力

- **Field**: U, V, A, RF, S, LE, STATUS (2ms 間隔 = 500 Hz)
- **History**: ALLKE, ALLSE, ALLAE, ALLIE, ETOTAL

---

## 3. Phase 2: 分離メカニズム実装 (TODO)

### 3.1 Frangible Bolt → `*MODEL CHANGE`

- 72本の V-notch ボルトをビーム要素で表現
- `*MODEL CHANGE, REMOVE` で Step-Separation 開始時に一斉除去
- `--n_stuck_bolts N` で N 本のボルト不発を再現

### 3.2 Pyro-cord

- 縦シーム (θ=90°) に沿ったビーム要素
- `--pyro_asymmetry` で非対称切断を再現

### 3.3 Opening Springs

- 下端部のスプリング要素 (`--spring_stiffness`)
- フェアリング半殻の展開駆動力

---

## 4. 使い方

### 4.1 INP 生成 + 解析 (qsub)

```bash
# 正常分離
qsub -v JOB_NAME=Sep-Normal scripts/qsub_fairing_sep.sh

# 異常分離 (ボルト3本不発)
qsub -v JOB_NAME=Sep-Abnormal,EXTRA_ARGS="--n_stuck_bolts 3" scripts/qsub_fairing_sep.sh
```

### 4.2 INP 生成のみ (ローカル)

```bash
LD_PRELOAD=/home/nishioka/libfake_x11.so \
  abaqus cae noGUI=src/generate_fairing_separation.py -- --job Sep-Test --no_run
```

---

## 5. GNN-SHM との連携 (将来展望)

分離ダイナミクスの異常検出を GNN-SHM パイプラインに統合:

1. **DOE**: 正常/異常分離パラメータの実験計画
2. **FEM バッチ**: 100+ ケースの分離シミュレーション
3. **特徴量抽出**: 加速度・応力時系列 → GNN ノード特徴量
4. **異常検出**: 6–7 Hz 振動パターンの自動分類

---

## 6. 参考文献

| 文献 | 内容 |
|------|------|
| JAXA H3-8 調査報告 (2025/12/25) | 分離異常振動の実測データ |
| KTH thesis (Fairing separation FE) | Abaqus ビーム要素除去による分離モデリング |
| Kawasaki H3 fairing development | Frangible bolt (V-notch) メカニズム |

---

## 7. ファイル一覧

| ファイル | 説明 |
|---------|------|
| `src/generate_fairing_separation.py` | メインスクリプト (Abaqus CAE) |
| `scripts/qsub_fairing_sep.sh` | PBS ジョブスクリプト |
| `abaqus_work/Sep-*.inp` | 生成された INP ファイル |
| `abaqus_work/Sep-*.odb` | 解析結果 ODB |
