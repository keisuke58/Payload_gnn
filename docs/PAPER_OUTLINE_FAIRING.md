# フェアリング論文化 骨子 (Paper Outline) — 2026-10 時点

> 修士論文は別テーマ。本研究（フェアリング GNN-SHM）は RA の成果として **対外発表を1本だけ** 出す。
> 既存の `docs/PAPER_ROADMAP.md`（2026-02 時点の状況）と
> `results/composites_b_table1_draft.tex`（Table 1 下書き）を引き継ぐ。
> 数値はすべてリポジトリ内の結果ファイルから転記（出典を併記）。図・表ラベルは英語。

---

## 0. 方針: 発表は1本だけ

| | 発表する（§1） | 発表しない（§2・水素タンク） |
|---|---|---|
| 内容 | GNN による多種欠陥の検出・局在化（FEM 700 サンプル） | 実測データでの誤報制御（OGW conformal）、水素タンク拡張 |
| 出し先 | **JSASS 年会 2027**（講演、締切 1〜2月目安） | 研究室・JAXA 向けの報告資料、研究室リポに引き継ぎ |
| 理由 | 結果がそろっていて図表の下書きもある。追加作業は Table 1 確定とシード反復くらい | 修論が別テーマなので、論文化の作業は増やさない |

- ジャーナル投稿はしない前提。先生から勧められたら相談する（候補メモは `wiki_repo/Publication-Venues.md`）。
- §2 は記録として残す（後任が論文化するときの出発点）。

---

## 1. 発表する1本 — Geometry-aware GNN for multi-type defect localization on a launch-vehicle CFRP/Al-honeycomb fairing

### 構成と図表の割り当て

| 節 | 内容 | 使う図表・根拠 |
|----|------|--------------|
| 1. Introduction | H3 F8 事故 → CFRP/Al-HC 界面の SHM 需要。貢献の列挙 | `wiki_repo/F8-Accident-Analysis.md` |
| 2. Related work | ML-SHM、メッシュ上の GNN（MeshGraphNet 系）、物理損失 | `MESHGRAPHNET_VARIANTS.md`, `LITERATURE_REVIEW.md` |
| 3.1 FEM dataset | H3 Type-S 1/6 扇形、5 欠陥タイプ（debond / FOD / impact / delamination / healthy）、700 サンプル | `docs/DEFECT_MODELS_ACADEMIC.md`, `src/generate_fairing_dataset.py` |
| 3.2 Graph | 曲率対応グラフ、34 次元ノード特徴 | `src/build_graph.py` |
| 3.3 Models & losses | LGSTA（提案）＋ GCN/GIN/SAGE/GATv2/MeshGNN/Transolver。物理損失 L1（応力勾配）/L2（平衡残差）、マルチタスク（重心回帰） | `src/models.py`, `src/train.py` |
| 4.1 Main result | **Table 1**: アーキテクチャ比較 & 物理損失アブレーション | `results/composites_b_table1_draft.tex` |
| 4.2 Localization | **Fig.**: 予測ヒートマップ vs 正解 | `results/attention/fig_prediction_localization.png` |
| 4.3 OOD | **Table 2**: 欠陥半径による OOD 汎化 | 同 tex（**未完**） |
| 4.4 Calibration | **Fig.**: 信頼度校正（location head） | `results/defect_type_posterior/fig_reliability.png` |
| 5. Discussion | 限界と負の結果（下記） | — |
| 6. Conclusion | | |

### 現時点の数値（Table 1 下書きより）
- **LGSTA (NC+L1+L2)**: OptF1 **0.896**、AUC **0.999**（最良）
- 他アーキテクチャ（MT+L1+L2）: OptF1 0.821–0.834、AUC 0.998–0.999
- Transolver: OptF1 0.010、AUC 0.493（**負の結果**として記載）
- 局在化の例（GAT, 5-class）: node-level F1 0.926–0.962（4 検証サンプル）

### 正直に書くべき弱点（Discussion 用）
- **欠陥タイプ別 F1 が低い**: LGSTA でも FOD 0.367 / impact 0.344 / delamination 0.341。2値検出は強いが種別判別は弱い
- **location head の校正が悪い**: ECE 0.383 → 温度スケーリング後 0.253（`fig_reliability.png`）
- **他構造へのゼロショット転移は効かない**: 分離データで AUC 0.41（`results/cross_dataset_eval.json`）
- Table 1 は「最良検証チェックポイント（ep 58–132）」で、200 epoch 完走後の値に差し替えが必要

### 講演までの TODO
- [ ] Table 1 を 200 epoch 完走後の値で確定
- [ ] Table 2（OOD）を埋める（現状 PENDING、途中経過 ep3: ID 0.732 / OOD 0.857）
- [ ] **シード違いの反復（平均 ± 標準偏差）** — 現状は単発の値
  - 手順: `python scripts/table1_seeds.py plan --seeds 0 1 2 --execute`（元の設定を `runs/pinn_l12_*/best_model.pt` から復元して再学習）
    → `python scripts/table1_seeds.py aggregate` で `results/table1_seeds.tex`（平均 ± 標準偏差）を生成。
    同じ手順で 200 epoch 完走後の値に差し替わる
- [ ] タイプ別 F1 の改善 or 限界として位置づける判断
- [ ] Data/Code availability: Zenodo DOI `10.5281/zenodo.20495444`
- [ ] AI 支援ツール使用の開示文（学会・大学の規定を確認）

---

## 2. 発表しない（記録・引き継ぎ用） — Distribution-free false-alarm control for guided-wave SHM under temperature shift

### 構成と図表の割り当て

| 節 | 内容 | 使う図表・根拠 |
|----|------|--------------|
| 1. Introduction | 実運用では温度で波形分布がずれ、誤報が増える。「FPR を保証する」問題設定 | `TEMPERATURE_ROBUSTNESS.md` |
| 2. Data | Open Guided Waves（長期・季節変動の実測） | `results/ogw/ogw_dataset_overview.png` |
| 3. Method | ラベル不要の分布整合（z-score / CORAL / 分位マッチ）＋ split-conformal による閾値決定 | `src/domain_adapt.py`, `src/payload_da_gw.py` |
| 4.1 FPR control | **Fig.**: 名目 α に対する実測 FPR | `results/ogw/ogw_conformal.png` |
| 4.2 Long-term shift | **Fig.**: 9→30 °C で FPR 14% → 5% | `results/payload_da_longterm.png` |
| 4.3 Over time | **Fig.**: 月ごとの検出率と誤報率 | `results/ogw/fig_detection_over_time.png` |
| 4.4 Pipeline | 検出 → 特性同定 → 予後（実データ端から端） | `results/ogw/fairing_real_e2e.png`, `fig_sim2real_3stage.png` |
| 5. Discussion | 誤報制御と見逃しのトレードオフ（下記） | — |

### 現時点の数値
- 分位マッチ + conformal: 名目 α = 0.025 / 0.05 / 0.10 / 0.15 に対し、実測 FPR **0.016 / 0.050 / 0.082 / 0.093**（≦ α を満たす）
- 補正なし: 同じ α で 0.096 / 0.141 / 0.371 / 0.589（保証が崩れる）
- 長期データ（9→30 °C）: 健全 FPR 14% → 5%（設計目標 5% を達成）
  （出典: `results/ogw/ogw_conformal.json`, `results/payload_da_longterm.png`）

### 正直に書くべき弱点
- **誤報を抑えると見逃しが増える**: 長期データで recall ≈ 21%。生の実データ検出も recall 0.57 / FPR 0.51 程度
- 厳冬期に conformal 保証が崩れ、Mondrian（条件別）で回復させた経緯（`scripts/ogw_conformal_detection.py`）— 手法の適用範囲として記載
- 損傷の導入時期と季節・経年が交絡している（`defect_type_posterior/metrics.json` の confound probe）。種別判別の精度には環境要因が混ざる

### 引き継ぎ用 TODO（任意）
- [ ] 厳冬期の破綻と Mondrian 版の結果を図1枚に整理
- [ ] recall とのトレードオフ曲線（α を振った図）
- [ ] 交絡の扱い（評価分割の設計）を Methods に明記

---

## 3. スケジュール（RA 計画 P0 と対応）

| 時期 | 作業 |
|------|------|
| 2026-11〜12 | Table 1 確定・シード反復 |
| **2027-01〜02** | **JSASS 年会 2027 に申込・講演原稿** |
| 2027-04 | 講演 |

> 締切は当年の案内で要確認。
