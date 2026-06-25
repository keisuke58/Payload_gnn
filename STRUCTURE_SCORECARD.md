# 構造SHMポートフォリオ 進捗スコアカード（超厳格・正直版）

> 作成 2026-06-20。overclaim禁止（[[feedback_shm_depth_over_breadth]]）。
> 「存在＝完了」としない。**有効に解けた／検証した／ML結果が出た**のみ加点。
> **S6方針**: 各構造で最強の in-domain モデル（oracle, S6a）を先に確立する。oracle なき LOSO 評価は比較基準を欠くため、S6b（cross-structure汎化）は S6a 確立後に行う。

## 採点ルーブリック（各部品 /100）

| 段階 | 配点 | 加点条件（厳格） |
|---|---|---|
| S1 モデル/メッシュ | 10 | datacheck通過・実部品寸法 |
| S2 健全solve収束 | 20 | **実応力が出る**（datacheck通過だけは0点） |
| S3 欠陥注入機構 | 15 | 有効な欠陥odbが出る（inp生成だけは部分点） |
| S4 データセット量産 | 20 | **実際にsolve完了**したケース数（inp数ではない） |
| S5 物理検証 | 15 | 理論一致/GCI/収束/座屈検証など定量 |
| S6a in-domain oracle | 10 | **その構造の最強モデル**確立（アーキ比較・hyperparam最適化完了・F1/AUC報告） |
| S6b cross-structure汎化 | 5 | LOSO or 少量 fine-tune 後に**他構造で**指標が出ている（S6a 確立が前提） |
| S7 end-to-end実検証 | 5 | 実データ or sim2real の検出指標（本質的弱点軸） |

---

## 🎯 優先3部品（詳細）

### ② 衛星(ペイロード)フェアリング = H3 cohesive
> **2トラックに分離**（詳細=[FAIRING_USABLE_ASSETS.md](FAIRING_USABLE_ASSETS.md)）：
> - **トラックA：GW動解析 検出＋実データe2e** … **〜82/100** 🟢（odb133・CSV100・F1 0.788＋**実34ヶ月GW縦断検証/OGW実DI 0/0.57/0.998 monotone 完了**）。温度補正FULL(amp+regress,2018-2019→2020-2022): FPR=**0.508**↓(AMP-only 0.577), Recall=0.566。スコア+2（但し補正でrecall 0.702→0.566も低下のためFPR問題は部分解消のみ）
> - **OGW sim2real DA（2026-06-22）**: FEM訓練モデルをOGW実データ(Intact/FirstImpact/SecondImpact)に適用 → **2/3 acc**（src-norm/tgt-norm両方）。**Intact のみ FP（偽陽性）** = FEM healthy 特徴と実測 healthy の分布ズレ (domain gap 定量値)。両 Impact は正しく検出。→ paper §domain-gap の根拠データ
> - S6a=**7**/10（GW in-domain oracle途上）：SAGE F1 0.788。GAT/MGN比較未完→アーキ総当り後に oracle 確定
> - S6b=**3**/5（cross-structure汎化）：OGW sim2real 2/3 acc・Intact FP=domain gap定量済 → cross-structure fine-tune評価へ
> - **トラックB：静的Max-Q応力/SCF** … **14/100** 🔴（ENCASTRE BC修正+2.0mm adhesive → job40859 H3_healthy_a3 実行中 2026-06-21）

#### トラックB詳細（静的Max-Q応力）= **14 / 100** 🔴（修正実行中）
- S1=10：9部品サンドイッチ、datacheck通過
- S2=**0**：全シナリオ求解失敗。**真因2つ**: (1) **outer skin shell BC に回転拘束(DOF4-6)なし**→剛体回転モード→zero pivot（TieがNODE-TO-SURFACEに降格→outer skinが宙ぶらり） (2) adhesive cohesive厚0.2mm→歪み要素
  - BC修正(ENCASTRE) + adhesive=2.0mm は適用済
  - **H3_healthy_a3b (job40938)**: 求解完了したが `Unable to access the new solver solution` で失敗
    - 真因: TIE `POSITION TOLERANCE=COMPUTED` がエッジノードをカバーできず未接続ノード多数発生（Node 418残差 -2532 N = 許容値の6000×）
    - **修正済(2026-06-22)**: `H3_healthy_a3.inp` を直接パッチ → 全8 TIEに `position tolerance=3.0` 追加
  - **H3_healthy_a3d (job 40955)**: `adjust=yes` → cohesive 要素の両面が master に引込まれ negative thickness (10 FATAL ERROR)
    - **真因**: adjust=yes + position tolerance=3.0 > cohesive 厚(2.0mm) → COH3D8 両面が同一 master に引込まれ裏返し
    - **修正済(2026-06-22)**: 全8 TIE を `adjust=no, position tolerance=3.0` に変更
  - **H3_healthy_a3e (job 40957)**: adjust=no 修正版 submit ✅ — zombie job40938 が QSD 占有のため license 待ち中（~09:04 JST 解放予定）
- S3=2：7シナリオparam_fileあるが**有効odbゼロ**（a3収束後にgen_and_solve_defects.sh実行予定）
- S4=0：有効データ0（a3収束確認後に7欠陥一括生成）
- S5=0 / S6a=0/10・S6b=0/5（Max-Q応力FEM：solve未完のため ML 不可。GW SAGE F1 0.788 は別トラックの値） / S7=0
- **次の一手**：H3_healthy_a3.sta が COMPLETED → `bash gen_and_solve_defects.sh` → 7欠陥ジョブ submit → S2=20, S3=15, S4増加

### ① 段間構造 = CFRP 1穴サンドイッチパネル（実物）  ……  **82 / 100** 🟢（S6実行中・暫定）
- S1=10：実段間部・曲面・L1-L20積層+FOAMCORE（GNNグラフ: **15,206ノード/sample**; ⚠旧記述「193,866節点」は誤記→実測値に修正 2026-06-23）
- S2=20：健全odb 717MB 収束済
- S3=15：node-split忠実欠陥（材料スワップでなく真の変位不連続）＋接触版backlog
- S4=**20**：忠実100ケース **100/100 solve完了**（D005-D008 "COMPLETED"確認→batch_status修正済 2026-06-21）
  - PyGデータセット: `processed_s12_czm_100_binary`(80/20) + `processed_s12_czm_100_8class`(stratified 80/20, 全7クラスval確保)
- S5=6：剥離成長スイープ図あり（open-seam線形）だが**段間応力の理論照合は未**
- S6a=**10**/10（oracle 確立済 2026-06-23）：proxy→fine-tune EXP3 = **0.931 ± 0.004** が scratch EXP1(0.914)を超え oracle 確定
  - EXP1 from-scratch czm_96_binary: **完了** — seed42 **0.9384**(ep232)/seed123 **0.8964**(ep52)/seed7 **0.9060**(ep48) → mean **0.914 ± 0.022**; val_AUC 0.9997-0.9999 → **現 oracle 候補**
  - EXP2a proxy(mixed_400): 全seed完了 — mean **0.856 ± 0.009**（scratch比 −0.058）
  - EXP3 fine-tune: **完走(2026-06-23)** — seed42 **0.9284**(ep138)/seed123 **0.9271**(ep107)/seed7 **0.9362**(ep42) → mean **0.931 ± 0.004** ✅ **EXP1 scratch(0.914)を+0.017上回る = oracle 確定**
  - EXP4 8-class → **完走(2026-06-23)** — seed42 **0.8774**(ep81)/seed123 **0.8921**(ep144)/seed7 **0.8851**(ep142) → OptF1 mean **0.885 ± 0.006**, Macro-F1 **0.328 ± 0.033**; acoustic_fatigue F1=0.000(全seed)
- S6b=**2**/5（cross-structure汎化）：
  - EXP2b zero-shot: **AUC≈0.5 (定数予測)** = 完全非転移 → LOSO fine-tuning 必須を定量化 [正直]
  - **proxy vs scratch ギャップ: −0.058**（旧推定 −0.084 より小さい）
  - EXP3完走後に LOSO 評価 → S6b更新予定
- S7=**5**（満点）：✅**児嶋TSA sim2real — Stage-0記述子を実測独立モダリティで検証済（2026-06-13）**
  - 実 TSA 22体（totsu 12 + ou 10、~13k nodes）を**FEMと同一コード経路**で通した
  - Spearman(c, measured peak stress) **+0.787**、peak/median ratio **+0.876**、totsu>ou **+0.532**
  - ref-free ランク利用 = パイプラインの唯一の使い方が実データで妥当と確認 [R]
  - 絶対閾値の非転移(FEM c≈7.29 vs 実 c≈2.0, 3.6×)は機構的に説明済(ノッチ穴が支配) [正直]
  - `GNN/ingest/figs/kojima_sim2real.png`
- **次の一手**：(a)EXP3/4完走済・S6a確定✅ (b)LOSO cross-structure fine-tune評価→S6b更新 (c)proxy→faithfulギャップ定量=論文Table1

### ③ 水素タンク = H3 LH2/LOX FEM + BAM/DLR COPV(実測)  ……  **71 / 100** 🟢
- S1=10：LH2(33761 S8R)/LOX(12395 S8R) + COPV FEM
- S2=20：両タンクCOMPLETED、de-pole後 Mises 103
- S3=13：FSW溶接線シーム亀裂 node-split 実装（金属はdelam不可→溶接線が忠実）
- S4=**10**：掃引12ケースprep / 17 odb（部分）。⚠長亀裂800mmは COD>板厚半分で **nlgeom=YES 再solve必須**（線形妥当域逸脱）
- S5=**13**：pR/t=113.8 vs 求解103一致・COD vs Folias・極特異点de-pole処理＝物理検証強い
- S6a=**3**/10（in-domain oracle着手 2026-06-23）：`h3tank_gnn_trial.py` LOO-CV 試走完了
  - SAGE binary, subsample=3k nodes, 100 epochs, n=16 graphs（healthy×1 + defect×15）
  - defect-F1=**0.857**（12/15正解）・acc=0.750・AUROC=無効（healthy 1件問題で崩壊）
  - 誤分類: `circ_200_mid` / `wc_200` / `wc_400` = 小亀裂・周方向が難しい
  - **oracle 未達理由**: healthy=1件→LOO で健全参照なし fold → 全件 defect 予測 → AUROC 崩壊
  - **oracle 確立条件**: healthy odb ×3-5（圧力/温度条件違い）+ defect 50件以上
- S6b=0/5：S6a oracle 未確立のため LOSO 評価不可
- S7=**5**（満点）：✅**BAM COPV 実測 end-to-end SHM pipeline 完了（2026-06-21）** — 25PZT×600pair，DI=1−corr，AUROC 0.92(RD)/0.94(ID) @180kHz，10×healthy floor，正しい重篤度順序（ID>RD），周波数選択性（60kHz失敗→180kHz最適），20–700bar圧力ロバスト，RAPID局所化6.6×null。全て [R]。`PAPER_DRAFT_copv_guidedwave.md` §5–7
- ✅**忠実度穴 解消(2026-06-21)**：COPV FEMを **CF+GFRP 8プライ直交異方性layup** に完全更新（T700M21 CF × TVR380M12R GFRP，CompositeShellSection）。CLT A-matrix: Ex=17.3 GPa(旧50 GPa)・Ey=60.0 GPa・ρ=1594 kg/m³。PBS job 40544(60kHz✅)→40545(120kHz R)→40546-48(H)・静的COPV_v2(±54°)=job40576 R。ply厚さ[U]
- **次の一手**：(a)PBS GW 5本完了→`extract_rx_batch.py`→`sim2real_batch.py`→dispersion再計算 (b)COPV_v2完了→Mises比較 (c)長亀裂nlgeom=YES再solve
- **S6a oracle 確立ロードマップ（2026-06-23策定）**

  **Step 1 — healthy 追加（最優先・½日・QSD 3 token）**
  ```
  lh2_healthy_p050.inp  — 0.5×  内圧（既存 inp から圧力スケール変更）
  lh2_healthy_p075.inp  — 0.75× 内圧
  lh2_healthy_p125.inp  — 1.25× 内圧
  → healthy = 4件 → LOO で各 fold に healthy が入り AUROC 計算可能
  submit: abaqus job=lh2_healthy_p0XX interactive（frontale 直列）
  ```

  **Step 2 — defect 追加（〜50件・JSCES2027向け）**
  ```
  拡張軸: 亀裂長さ +100/600/1000mm、方向 +th45（inp 例あり）、位置 +top/shoulder
  4長さ × 3方向 × 3位置 = 36ケース（現12 + 新24）+ wc 3 = 計 39 defect
  ```

  **Step 3 — 再試走（GPU 30分）**
  ```
  python h3tank_gnn_trial.py --subsample 0 --epochs 200
  oracle 認定ライン: AUROC > 0.8 → S6a = 7-8/10
  ```

---

## 📋 その他の部品（コンパクト採点）

| 部品 | 種別 | 点 | 主因 |
|---|---|---|---|
| OGW omega-stringer | 実SLDV波動・CFRP | ~55 | 実データ+Stage0検出(inter0.9999)あり・分類/end2end薄い |
| DLR MT板 (AA2024) | 実DIC・疲労き裂 | ~50 | 実データ強・GNN統合途上 |
| framework: 締結板boltskin | FEM | ~55 | Kt3.02/Hashin/GCI/3D層間まで検証◎・データ/ML無 |
| framework: 着陸脚 | FEM | ~50 | 座屈Lanczos+Riks検証◎・SHM用途未 |
| framework: 極低温/shaft/前縁/音響 | FEM | ~40 | 健全求解+配向検証済・欠陥/ML無 |
| 児嶋 TSAクーポン(22体) | 実赤外応力 | ~40 | 実測貴重・GNN検証直結だが未接続 |
| ReMAP補剛板 | 実DFOS | ~35 | データのみ |
| ドライブシャフト管(111体) | 実S-N | ~30 | 寿命モデル供給用・検出グラフでない |
| SRB-3 モーターケース | FEM(軸対称CAX) | ~30 | 2D断面のみ・3D内圧再solve要 |

---

## 全体の正直な総括
- **強い＝段間①(S7満点・S6a oracle途上)・水素タンク③(S7満点・物理検証◎)**。①③ともS7=5で実データ end-to-end 検証済。①の S7 は児嶋TSA 22体実測によるStage-0記述子のランク転移確認（Spearman+0.787）。
- **S6a（in-domain oracle）**: ①進行中（scratch 0.914 が暫定 oracle・EXP3完走待ち）/ ②A GW部分達成(SAGE F1 0.788・アーキ比較未完) / ③=**3/10 着手済**（2026-06-23 試走・defect-F1 0.857・healthy不足でoracle未確立）/ ②B=0（solve未完）。
- **S6b（cross-structure LOSO）**: ①EXP2b で zero-shot AUC≈0.5=転移失敗を定量済・fine-tune LOSO は EXP3後 / ②A OGW sim2real 2/3 acc / ③・②B=0。
- **止まってる＝フェアリング②B**（cohesive厚→再生成1コマンドで復活する既知案件）。
- **次に効く順**: ①EXP3完走→S6a oracle確定・論文Table ＞ ②A アーキ比較→S6a=10 ＞ ②B cohesive再生成（S2=20回収）＞ ③GNN初回→S6a着手。

---

## 🗓 次フェーズ計画（2026-06-21 時点）

### 今夜〜明日（自律実行）
| タスク | 担当 | 完了条件 |
|---|---|---|
| binary 転移完走 (EXP1-3, 9runs) | Vancouver GPU0 🔄 | `transfer_summary.json` 生成 |
| EXP4 8-class 起動 | binary完走後に手動 `bash run_s12_multiclass.sh 0` | — |
| COPV PBS jobs 確認 (40544-40548) | frontale確認要 | ODB5本の`Status:COMPLETED` |

### 今週中（手動判断要）
| タスク | 効果 | 条件 |
|---|---|---|
| S6確定値でスコアカード更新 | 段間①S6確定→論文数値確定 | 転移完走後 |
| BAM COPV ODB解析→sim2real dispersion更新 | ③S5改善・PAPER_DRAFT §3精緻化 | PBS GW 5本完了後 |
| 段間①残10ケースsolve | S4=18→20 | abaqus serial queue |
| フェアリングB ENCASTRE+2.0mm 収束確認 | ②S2=0→20 | H3_healthy_a3 .sta COMPLETED確認後にgen_and_solve_defects.sh |

### 中期（〜7月末）= oracle確立 → 論文形成フェーズ
| 目標 | 段階 | 内容 |
|---|---|---|
| **①段間 oracle確定** | S6a完了 | EXP3完走→seed平均F1確定→scratch vs fine-tuneギャップ=論文Table |
| **②A フェアリングGW oracle確定** | S6a完了 | GAT/MGN vs SAGE 比較→best F1→0.788超え狙い |
| **③水素タンク GNN初回** | S6a着手 | COPV検出→分類 GNN: 初回F1確認→oracle候補 |
| **3構造 LOSO 統合評価** | S6b | S6a 3構造確定後→cross-structure F1・構造非依存SHM本体論文 |
| proxy→faithful ギャップ定量 | 論文核心 | EXP1 vs EXP2b vs EXP3 F1差=Table1 |
| COPV sim2real dispersion | IWSHM候補 | PBS完了波形→実PZT F1比較 |

_更新規則：solve完了数・ML指標が動いたら S4/S6/S7 を実数で書き換える。inp生成やdatacheck通過では加点しない。_
