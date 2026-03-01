[← Home](Home) | [FEM-Realism-Roadmap](FEM-Realism-Roadmap) | [Realistic-Fairing-FEM](Realistic-Fairing-FEM)

# CZM ソルバー試行錯誤ログ — Cohesive 接着層モデルの求解

> **期間**: 2026-03-01 〜 03-02
> **ステータス**: 未解決 (求解成功していない)
> **対象スクリプト**: `src/generate_cohesive_fairing.py`

---

## 1. 概要

5パート・サンドイッチ構造 (InnerSkin → AdhesiveInner → Core → AdhesiveOuter → OuterSkin) の CZM モデルを Abaqus Standard で求解しようとした記録。INP 生成は成功したが、**ソルバーが安定して完了しない** 状態が続いている。

### モデル構成

```
InnerSkin (S4R shell)
  ↓ Tie
AdhesiveInner (COH3D8 cohesive, 0.2mm)
  ↓ Tie
Core (C3D10 tet solid, ~37.6mm)
  ↓ Tie
AdhesiveOuter (COH3D8 cohesive, 0.2mm)
  ↓ Tie
OuterSkin (S4R shell)
  + 4 Ring Frames (S4R shell) Tie→InnerSkin
```

---

## 2. 試行一覧

| # | ジョブ名 | seed | 方程式数 | INP | サーバー | 結果 | 原因 |
|---|----------|:----:|:--------:|:---:|:-------:|------|------|
| 1 | Job-CZM-Test | 25 | — | 37MB | frontale02 | **14 FATAL ERRORS** | Core メッシュが空 (JNL に deleteMesh 残存) |
| 2 | Job-CZM-Test2 | 25 | 3.7M | 129MB | frontale02 | **変位ゼロ**, メモリ 92GB | Core ogive 未切詰め → 浮遊ノード |
| 3 | Job-CZM-Test3 | 25 | 3.7M | 124MB | frontale04 | **変位ゼロ**, メモリ 92GB | r_min 修正済みだがメモリ超過 |
| 4 | Job-CZM-Test4 | 25 | 3.7M | 130MB | frontale04 | **反復発散** | CG+PAMG が TIE/CZM の剛性コントラストで発散 |
| 5 | Job-CZM-Coarse | 50 | — | 51MB | frontale04 | **要素歪み** | 接着層 seed=50 → アスペクト比 250:1 |
| 6 | Job-CZM-Med | 40 | — | 67MB | frontale04 | **体積ゼロ** | Core tet 要素 1個が体積ゼロ |
| 7 | Job-CZM-S35 (v1) | 35 | 2.17M | 74MB | frontale04 | **数値特異性** | AdhesiveOuter ノード変位 3×10⁶ mm |
| 8 | Job-CZM-S35 (v2) | 35 | 2.17M | 74MB | frontale04 | **メモリ 92GB → フリーズ** | OgiveCap BC 追加後も直接法メモリ超過 |
| 9 | Job-CZM-S35 (OOC) | 35 | 2.17M | 74MB | marinos03 | **数値特異性**, 分解成功 | memory=20GB でアウトオブコア成功、しかし浮遊ノード |
| 10 | Job-CZM-S35v3 | 35 | 2.17M | 74MB | marinos03 | **数値特異性** | position tolerance=5.0 でも TIE サーフェス定義が不完全 |
| 11 | Job-CZM-S35v4 | 35 | 2.17M | 74MB | marinos03 | **数値特異性** | STABILIZE 追加でも改善せず |
| 12 | Job-CZM-S35v2 | 35 | 4.5M | 130MB | marinos03 | **10 FATAL ERRORS** | 面分類修正で再生成 → 要素歪み (異なる乱数パラメータ) |

---

## 3. 特定した問題と実施した修正

### 3.1 Core メッシュ空問題 (試行 1)

**症状**: INP に Core インスタンスのメッシュデータが 2 行のみ
**原因**: 以前の CAE セッションで `deleteMesh` + `STRUCTURED` リトライが失敗し、メッシュ削除状態がコミットされた
**修正**: コード修正済み (コミット 9f7f43c) で再生成 → Core メッシュ正常 (1.37M 行)

### 3.2 Core オジーブ切詰め (試行 2-3)

**症状**: 変位がすべてゼロ。残差力は非ゼロだが解が得られない
**原因**: Core の ogive が r=0 まで延びる一方、接着層は r_min=100mm で切詰め → Core 先端のノードが接着層に接続できず浮遊
**修正**: `_create_solid_revolve_part` に `r_min=ADH_R_MIN` を追加し、Core を同じ位置で切詰め

```python
p_core = _create_solid_revolve_part(
    model, 'Part-Core', ..., r_min=ADH_R_MIN)
```

### 3.3 直接疎行列ソルバーのメモリ超過 (試行 2-3, 8)

**症状**: 93GB サーバーで RSS 92GB → OOM またはフリーズ
**原因**: TIE 拘束のラグランジュ乗数が疎行列分解の fill-in を大幅に増加
**対処**: `memory="20 gb"` でアウトオブコア解法を強制 → 分解は成功するが遅い (5-10分/回)

```bash
abaqus job=... cpus=1 memory="20 gb" interactive
```

**メモリ推定値 vs 実際**:
| モデル | 方程式数 | 最小推定 | 最適推定 | 実際使用量 |
|--------|:--------:|:--------:|:--------:|:----------:|
| seed=25 | 3.7M | 6.3 GB | 97 GB | 92+ GB |
| seed=35 | 2.17M | — | ~45 GB | 92+ GB (直接法) / 4-20 GB (OOC) |

### 3.4 反復法ソルバーの発散 (試行 4)

**症状**: CG+PAMG で残差ノルム 63% → 265% (発散)
**原因**: TIE 拘束 + CZM の剛性コントラスト (K_n=10⁵ vs E_core_2=1.0) で条件数が悪い
**結論**: Standard の反復法はこのモデルには不適。直接法アウトオブコアまたは Explicit が必要

### 3.5 接着層メッシュのアスペクト比 (試行 5-6)

**症状**: seed=50 で接着層要素が負の厚さ; seed=40 で体積ゼロ要素
**原因**: 0.2mm 厚の層に対して seed=50mm → アスペクト比 250:1 → SWEEP メッシュが反転
**修正**: 接着層 seed を最大 25mm にキャップ

```python
adh_seed = min(global_seed, 25.0)
```

### 3.6 AdhesiveOuter 浮遊ノード — 数値特異性 (試行 7, 9-11) **← 未解決**

**症状**: AdhesiveOuter の特定ノードで特異剛性比 10⁹〜10¹⁵、変位 10⁴〜10⁶ mm
**原因**: `_classify_solid_faces()` がトレランスベース (0.09mm) で面を分類 → オジーブ領域でメッシュ離散化による半径ずれが 0.09mm を超える面が「どちらでもない」に分類 → TIE サーフェスに含まれないノードが完全に浮遊

**試した対策**:
| 対策 | 結果 |
|------|------|
| Nose tip BC 拡張 (全5インスタンス) | 先端は改善、他は変わらず |
| OgiveCap BC (r<105, y>5000 のノードを固定) | 42053ノードが inactive DOF 警告 → TIE slave なので無効 |
| `position tolerance=5.0` (TIE) | 変わらず — サーフェス定義にない面のノードは拾えない |
| `*Static, stabilize, factor=1e-4` | 変わらず — ゼロ剛性ノードには安定化力もゼロ |
| 面分類を nearest-match に変更して再生成 | 異なる乱数パラメータで要素歪みエラー (10 FATAL) |

---

## 4. 根本原因の分析

### TIE サーフェス定義の不完全性

```
_classify_solid_faces() の問題:

接着層 (0.2mm 厚):
  inner 面: r ≈ r_inner  (tol=0.09mm で判定)
  outer 面: r ≈ r_outer  (tol=0.09mm で判定)
  エッジ面: r は inner と outer の中間

オジーブ領域ではメッシュ離散化により:
  face_centroid_r - r_expected > 0.09mm → 分類されない → TIE に含まれない

結果: TIE サーフェスに穴が開き、そこのノードが自由度ゼロで浮遊
      → 剛性行列の対角項がゼロ → 数値特異性
```

### INP のサーフェス定義の証拠

```
Surf-AdhOuter-Inner: S4 + S2 (2 面タイプ)    ← 旧 (不完全)
Surf-AdhOuter-Inner: S4 + S5 + S2 (3 面タイプ) ← 新 (修正後)
                          ↑ この S5 が欠落していた
```

---

## 5. 次のステップ (候補)

### A. 面分類の改善 + 決定論的パラメータで再生成

面分類は nearest-match に修正済み。再生成時に**同じ欠陥パラメータ**を使用して要素歪みを回避する。

```bash
abaqus cae noGUI=generate_cohesive_fairing.py -- \
  --job Job-CZM-Test --seed 35 --param_file fixed_params.json --no_run
```

### B. Abaqus/Explicit への切り替え

直接法の行列分解が不要 → メモリ問題を完全に回避。準静的解析は smooth step + mass scaling で対応可能。

```python
*Step, name=Step-1
*Dynamic, Explicit
...
*Amplitude, name=SmoothStep, definition=SMOOTH STEP
0., 0., 1., 1.
```

### C. モデル簡略化

- 全周モデルではなく 1/6 セクタに戻す
- 接着層を 1 層のみ (Inner or Outer) にして問題を切り分け
- seed=50 + `*PREPRINT, PARCHECK=NO` で要素品質チェックを回避

### D. 全ノード制約 + ソフトスプリング

浮遊ノードに微小な接地バネ (k=0.01 N/mm) を追加して剛性行列の特異性を防ぐ。

```
*SPRING, ELSET=STABILIZE
0.01
```

---

## 6. 学んだこと

1. **COH3D8 の面分類はトレランスに敏感**: 0.2mm 厚の薄層では inner/outer 面が非常に近く、半径ベースの分類が失敗しやすい
2. **TIE のサーフェス定義は完全でなければならない**: 1ノードでも浮遊すると全体の解が破綻する
3. **直接法ソルバーのメモリ**: TIE の fill-in で推定値の 2-3 倍のメモリを消費。93GB サーバーでは 2M+ 方程式が限界
4. **アウトオブコア解法**: `memory="20 gb"` で強制可能。分解は成功するが遅い (5-10分)
5. **反復法は TIE+CZM で使えない**: 条件数が悪すぎて CG が発散
6. **STABILIZE は浮遊ノードに無効**: 剛性ゼロのノードには粘性力もゼロ

---

## 7. コード修正の要約 (未コミット)

`src/generate_cohesive_fairing.py` に対する 5 つの修正:

| # | 行 | 修正内容 |
|---|:--:|----------|
| 1 | ~675 | Core part に `r_min=ADH_R_MIN` を追加 (ogive 切詰め) |
| 2 | ~1335 | Nose tip BC を全5インスタンスに拡張 |
| 3 | ~1352 | OgiveCap BC を追加 (截頭部ノード固定) |
| 4 | ~1410 | 接着層 seed を max 25mm にキャップ |
| 5 | ~1121 | 面分類を nearest-match アルゴリズムに変更 |

---

## 関連ページ

| ページ | 内容 |
|--------|------|
| [FEM-Realism-Roadmap](FEM-Realism-Roadmap) | CZM 実装ロードマップ |
| [Realistic-Fairing-FEM](Realistic-Fairing-FEM) | 現行 FEM モデルの詳細 |
| [Node-Features](Node-Features) | 34次元ノード特徴量 (CZM で +SDEG 追加予定) |
