# Hydrogen Tank SHM — 設計メモ (Design Note)

> **Status**: 設計フェーズ（未実装）。本メモは水素タンク (LH2) ラインの着手前に
> ジオメトリ・材料・荷重・欠陥・グラフ/特徴量・実行計画を固めるためのもの。
> 数値は **design target / assumption**（要 FEM・文献検証）であり確定値ではない。
> 図・表ラベルは英語（プロジェクト規約）。

## 0. 位置づけ

フェアリング SHM で確立した資産（曲率対応グラフ GNN・Guided-Wave・2/3段 SHM・
ドメイン適応・conformal 検出・温度ロバスト性）を、次の構造ターゲットである
**極低温水素タンク**へ拡張する。フェアリングとは損傷物理が異なるが、
パイプラインの大半はそのまま流用でき、追加すべき差分は限定的である
（本メモの §5–§6 で明示）。

関連: `TEMPERATURE_ROBUSTNESS.md`（動作点シフト → DA で回復）が
極低温という最も極端な動作点に対する既存フレームの直接の足場になる。

---

## 1. ジオメトリ (Geometry)

| Item | Design assumption | 備考 |
|------|-------------------|------|
| 対象 | H3 2段 LH2 タンク相当の円筒殻 + ドーム | 詳細寸法は非公開 → 代表値で設計 |
| 直径 | φ ≈ 5.2 m（フェアリングと同径帯） | H3 上段構造の代表径 |
| モデル化 | **対称セクタ 30°（1/12）**+ 周方向対称 BC | 実装: `SECTOR_DEG = 30`、胴高さ 4000 mm、壁厚 4 mm |
| 要素 | 壁: シェル（S4R 既定）／接着層: COH3D8／断熱フォーム: シェル | GW 解析は面内伝播が主 |
| 部位 | barrel（円筒部）。鏡板は**軸方向推力 pR/2 として作用**させ形状は省略。**溶接線**は扇形中央 15° に1本 | 欠陥位置は扇形内 6–24°（最大パッチ半幅 ≈4.9° の余白） |

フェアリングの `generate_fairing_dataset.py` / `generate_realistic_fairing.py` の
セクタ生成・対称 BC・メッシュ制御のロジックを雛形として再利用する。

---

## 2. 材料 (Materials) — 極低温物性

主対象は **Al-Li 合金**（H-IIA/H3 タンク系譜、例: 2219 / 2195 系）。
室温 (RT) → 液体水素温度 (LH2, −253 ℃ / 20 K) で物性がシフトする点が本質。

| Property | RT (design) | Cryo (LH2, design) | シフト傾向 |
|----------|-------------|--------------------|-----------|
| Young's modulus E | ~70–78 GPa | +5–15 % | 増加 |
| Yield strength σy | ~380–450 MPa | +15–30 % | 増加 |
| Elongation | ~10–12 % | 減少 | **脆化方向** |
| CTE α | ~23×10⁻⁶/℃ | 積分 CTE 低下（収縮量で扱う） | 低下 |
| Fracture toughness | — | 低下しうる | 要検証 |

> ⚠️ 上表は文献レンジからの **設計仮定**。実装前に材料データ（JAXA/文献）で確定する。
> 将来ライン: **CFRP 複合材クライオタンク**（微小亀裂・水素透過）では材料モデルを差し替える。

---

## 3. 荷重 (Loading)

フェアリング（熱 CTE + 静的）と異なり、タンクは以下を重畳する。

1. **内圧 (internal pressure)** — 加圧充填〜飛行の運用圧
2. **極低温熱応力 (cryogenic thermal stress)** — RT→20 K の収縮拘束による応力
3. **充填サイクル疲労 (fill-cycle fatigue)** — 加圧/減圧・熱サイクルの繰返し
4. （動的 GW 解析時）**弾性波励起** — アクチュエータ加振 50–300 kHz

静的解析（欠陥応力集中の把握）と GW 動的解析（センサ時刻歴）の2系統を、
フェアリングの静的／GW 2ブランチと同じ構成で並走させる。

**実装済みの荷重・境界（`generate_cryotank_dataset.py`）**
- 内圧: 壁内面に一様圧
- **鏡板推力**: 上端に軸方向エッジ荷重 **pR/2 [N/mm]**（閉端容器の σz = pR/2t を再現。`--open-end` で無効）
- 熱: 基準 293 K → 運用温度の一様温度場（CTE 基準温度 = 293 K）
- **底辺 = スカート取付**: 半径・軸方向を固定（既定）。一様冷却でも取付部で収縮が拘束され、熱応力・曲げが出る。軸方向のみ固定（`--bottom-bc axial`）だと自由収縮で熱応力はほぼゼロ
- 対称面: θ=0 と θ=30° の切断面（回転方向の符号に追従）

---

## 4. 欠陥モデル (Defect Models)

| Defect | 物理 | モデル化方針 | 対応フェアリング欠陥 |
|--------|------|-------------|---------------------|
| **Weld flaw** | 溶接線の気孔・融合不良・割れ | 溶接線上の剛性/連続性低下（要素弱化 or cohesive） | — (新規) |
| **Thermal-cycle microcrack** | 極低温サイクルの微小亀裂 | 局所剛性低下 + 部分接触不連続 | delam に類似 |
| **Insulation debond** | foam/MLI 断熱材の剥離 | 界面 cohesive の劣化 | **skin-core debond と同型** |
| **H-embrittlement** | 水素脆化による靭性低下 | 材料靭性パラメータ低下（感度解析） | — (UQ 側で扱う) |

**実装済みの欠陥表現（すべて局所 (θ, z) パッチ = 欠陥円の外接矩形）**

| Defect | FE での表現 | 主なパラメータ（設計仮定） |
|--------|------------|------------------------|
| Weld flaw | 壁の剛性低下（パッチ内 E × 0.5） | 溶接線 15° 上 |
| Microcrack | 壁の剛性低下（E × 0.3） | — |
| Insulation debond | 壁の外に **接着層（COH3D8, CZM）+ 断熱フォーム（シェル）** を追加し Tie で結合。パッチ内の接着層を**事前剥離**（剛性 × 1e-3）。外側は MAXS 開始 + BK エネルギー進展で**剥離の進展も表現可能** | フォーム 25 mm, E = 7 MPa, CTE 5.4×10⁻⁵/℃。接着層 0.2 mm, tn = 0.35 MPa, ts = 0.30 MPa, GIc = 0.10 N/mm, GIIc = 0.30 N/mm |

> 断熱フォームの CTE は 20 K の Al-Li の約 4 倍。冷却で接着層に界面応力が生じることが剥離の駆動力になる。
> 接着強度はエポキシではなく**フォーム側の強度で律速**（≈0.3 MPa 級）として設定（要データ確認）。
> 材料・セクション・メッシュ（SWEEP + COH3D8）・Tie は `generate_cohesive_fairing.py` と同じ方式。

---

## 5. グラフ & 特徴量 (Graph & Features) — 差分のみ

既存の静的グラフは **34次元ノード特徴**（`build_graph.py`）。
水素タンク向けには基本スキーマを維持しつつ、以下を**追加/差し替え**する。

**追加候補（差分）**
- `internal_pressure_flag` / 局所内圧応力成分（+1–3 dim）
- `cryo_property_delta` — RT 物性からの極低温シフト量（E/α のスカラー化, +1–2 dim）
- `weld_line_flag` — 溶接線近傍ノードの境界フラグ（+1 dim, 既存 boundary フラグ拡張）

**維持**: 位置・幾何(10)、変位(4)、応力(5)、ひずみ(3)、熱応力(1)、繊維配向は
Al-Li 等方材では 0 埋め or 省略（CFRP クライオタンク時に復活）。

> 設計原則: **新スキーマを作らず既存34次元に最小差分で足す**。
> こうすることで `train.py` / `models.py` / DA / conformal を改修なしで接続できる。

GW グラフ（センサ=ノード）は `build_gw_graph.py` のスキーマをそのまま流用。

---

## 6. 再利用マップ (Reuse Map)

| 既存資産 | 水素タンクでの役割 | 改修 |
|----------|-------------------|------|
| `build_graph.py` / `build_gw_graph.py` | グラフ構築 | 特徴量 +数次元のみ |
| `train.py` / `train_gw.py` | 学習 | **無改修**（`--data_dir` 差し替え） |
| `models.py`（GAT/GCN/GIN/SAGE ほか） | モデル | 無改修 |
| `domain_adapt.py` / `payload_da_gw.py` | 常温試験→極低温運用の sim2real | 無改修（X 行列渡し） |
| OGW conformal（`scripts/*ogw*`） | 漏洩リスク判定の FPR 保証 | 閾値/コスト再設定 |
| `fairing_stage2.py` | Stage-2 特性同定（亀裂サイズ/漏洩量） | ラベル定義を tank 用に |
| `TEMPERATURE_ROBUSTNESS.md` | 極低温=極端動作点の評価枠 | 動作点を 20 K に拡張 |
| `pce_driver.py` / `reliability_analysis.py` | 水素脆化・靭性の UQ | 不確かさ変数を tank 用に |

**新規に書くのは実質 FEM 生成のみ**（§7）。

---

## 7. 1サンプル FEM 実行計画 (One-Sample Plan)

プロジェクト流儀に従い、**まず1サンプルで検証**してからバッチ化する。
着手済みコード: `src/generate_cryotank_doe.py`（DOE, Abaqus 非依存・テスト済み）、
`src/generate_cryotank_dataset.py`（Abaqus CAE 雛形, `--dry-run` は非 Abaqus で解析プラン確認可）。

```bash
# 0) DOE 生成（healthy 1件, Abaqus 不要）
python src/generate_cryotank_doe.py --healthy_only --n_samples 1 --output doe_cryotank_1sample.json
# ローカルで解析プランを確認（Abaqus 不要）
python src/generate_cryotank_dataset.py --dry-run --defect doe_cryotank_1sample.json

# 1) FEM 生成（Abaqus）
abaqus cae noGUI=src/generate_cryotank_dataset.py -- --job CryoTank_Healthy_0000 --defect doe_cryotank_1sample.json
#    断熱層つきの健全ベースライン（剥離サンプルとの比較用）
abaqus cae noGUI=src/generate_cryotank_dataset.py -- --job CryoTank_HealthyIns_0000 --defect doe_cryotank_1sample.json --insulation

# 2) ODB 抽出（既存を流用）
abaqus python src/extract_odb_results.py --odb abaqus_work/Job-CryoTank-Healthy.odb

# 3) グラフ化 → 1件だけ目視検証（特徴分布・欠陥ラベル）
cd src && python build_graph.py --data_dir ../dataset_cryotank_1sample

# 4) OK なら DOE 拡張 → run_batch.py でバッチ、以降フェアリングと同フロー
```

クラスタ（PBS/Torque）実行は `scripts/dispatch_parallel.sh` を tank ジョブ名で流用。

### M1 の答え合わせ用 理論値（健全, 20 K, 内圧 0.368 MPa）

| 量 | 理論値 | 見る場所 |
|----|--------|---------|
| 周方向応力 σθ = pR/t | **239 MPa** | 胴中央 |
| 軸方向応力 σz = pR/2t | **120 MPa** | 胴中央（鏡板推力が効いていれば） |
| von Mises（膜） | 207 MPa（20 K 降伏 480 MPa → 安全率 2.3） | 胴中央 |
| 半径変位 | 内圧 +6.3 mm、冷却 −9.0 mm → **正味 −2.7 mm** | 胴中央（底辺から十分離れた位置） |
| 胴の軸方向収縮（冷却） | −13.8 mm | 上端 |
| 完全拘束時の熱応力 E·α·ΔT | 285 MPa（上限の目安） | スカート取付部付近は局所的にこの方向へ |

σθ が圧縮で出たら内圧の作用面が逆（`side1Faces` ↔ `side2Faces`）、σz ≈ 0 なら鏡板推力が効いていない。

### マイルストーン
1. **M1**: healthy 1サンプル生成 → グラフ目視 OK（本メモの検証）
2. **M2**: 各欠陥タイプ 1件ずつ生成 → 応力集中/波形変化の物理妥当性確認
3. **M3**: DOE バッチ（N≈100）→ `train.py` で検出ベースライン
4. **M4**: 常温↔極低温 DA + conformal で漏洩リスク FPR 保証
5. **M5**: Stage-2 特性同定（亀裂サイズ/漏洩量）→ go/no-go 予後

---

## 8. リスク & 未確定事項 (Open Items)

- Al-Li 極低温物性・靭性の**確定データ**（§2 は仮定）
- タンク実寸法・溶接配置（非公開 → 代表値運用の妥当性）
- GW 周波数帯とセンサ配置（フェアリング設定の転用可否）
- 内圧×極低温×疲労の**連成順序**（解析ステップ設計）
- CFRP クライオタンク（将来）への材料モデル差し替え範囲

---

## 9. 次アクション

- [x] `src/generate_cryotank_doe.py` / `src/generate_cryotank_dataset.py` 雛形作成（DOE + dry-run はテスト済み）
- [x] Abaqus モデル構築の実装（扇形シェル・極低温材料・内圧+熱・鏡板推力・スカート境界・対称面）
- [x] 欠陥の局所パッチ化、断熱材剥離の CZM 接着層モデル
- [x] `CLAUDE.md` の Research Lines に「Hydrogen Tank SHM」を追加
- [ ] 本メモのレビュー・数値の確定（材料・寸法・荷重・フォーム/接着層）
- [ ] M1: healthy 1件を生成し、上の理論値と照合 → 本メモに結果追記

参照: `docs/ARCHITECTURE.md`, `TEMPERATURE_ROBUSTNESS.md`, `docs/index.html`(公開ロードマップ)
