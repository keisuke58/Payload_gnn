# 構造SHMポートフォリオ 進捗スコアカード（超厳格・正直版）

> 作成 2026-06-20。overclaim禁止（[[feedback_shm_depth_over_breadth]]）。
> 「存在＝完了」としない。**有効に解けた／検証した／ML結果が出た**のみ加点。

## 採点ルーブリック（各部品 /100）

| 段階 | 配点 | 加点条件（厳格） |
|---|---|---|
| S1 モデル/メッシュ | 10 | datacheck通過・実部品寸法 |
| S2 健全solve収束 | 20 | **実応力が出る**（datacheck通過だけは0点） |
| S3 欠陥注入機構 | 15 | 有効な欠陥odbが出る（inp生成だけは部分点） |
| S4 データセット量産 | 20 | **実際にsolve完了**したケース数（inp数ではない） |
| S5 物理検証 | 15 | 理論一致/GCI/収束/座屈検証など定量 |
| S6 ML/GNN結果 | 15 | **その構造で**学習＋指標が出ている |
| S7 end-to-end実検証 | 5 | 実データ or sim2real の検出指標（本質的弱点軸） |

---

## 🎯 優先3部品（詳細）

### ② 衛星(ペイロード)フェアリング = H3 cohesive
> **2トラックに分離**（詳細=[FAIRING_USABLE_ASSETS.md](FAIRING_USABLE_ASSETS.md)）：
> - **トラックA：GW動解析 検出＋実データe2e** … **〜80/100** 🟢（odb133・CSV100・F1 0.788＋**実34ヶ月GW縦断検証/OGW実DI 0/0.57/0.998 monotone 完了**）。残り穴＝**温度交絡FPR(0.577)のみ**。「実検証ゼロ」はもう該当しない
> - **トラックB：静的Max-Q応力/SCF** … **14/100** 🔴（下記。旧データ全削除・2.0mmで再構築中）

#### トラックB詳細（静的Max-Q応力）= **14 / 100** 🔴
- S1=10：9部品サンドイッチ、datacheck通過
- S2=**0**：全シナリオ求解失敗。真因＝**cohesive厚0.2mm→歪み要素で解けない**（既知。datacheck通るが実solveで落ちる）
- S3=2：8シナリオparam_fileあるが**有効odbゼロ**
- S4=0：有効データ0（mq8本+t2全滅）
- S5=0 / S6=2（※別途GW無荷重検出 SAGE F1 0.788 は存在するが本データ＝Max-Q応力とは別物）/ S7=0
- **次の一手**：`--adhesive_thickness 2.0` で再生成→収束確認（**2026-06-20 実行中：job名 H3_healthy_a2**）。通れば一気にS2→20, S3,S4回収可能。

### ① 段間構造 = CFRP 1穴サンドイッチパネル（実物）  ……  **77 / 100** 🟢（S6実行中・暫定）
- S1=10：実段間部180,944要素/193,866節点・曲面・L1-L20積層+FOAMCORE
- S2=20：健全odb 717MB 収束済
- S3=15：node-split忠実欠陥（材料スワップでなく真の変位不連続）＋接触版backlog
- S4=**20**：忠実100ケース **100/100 solve完了**（D005-D008 "COMPLETED"確認→batch_status修正済 2026-06-21）
  - PyGデータセット: `processed_s12_czm_100_binary`(80/20) + `processed_s12_czm_100_8class`(stratified 80/20, 全7クラスval確保)
- S5=6：剥離成長スイープ図あり（open-seam線形）だが**段間応力の理論照合は未**
- S6=**8**：転移学習 3実験×3seed **Vancouver GPU0 実行中**（2026-06-21）
  - EXP1 from-scratch on czm_96_binary / EXP2a proxy-only / EXP2b zero-shot / EXP3 fine-tune → 完走待ち
  - EXP4 **8クラス多クラス** on czm_100_8class → binary完走後に起動
  - EXP1/seed42 epoch39: val_opt_f1=**0.887** (AUC=0.9997) 🔄
  - 確定スコアは全seed完走後に更新（暫定8）
- S7=0
- **次の一手**：(a)binary転移完走→`transfer_summary.json`→S6確定 (b)EXP4 8-class起動 (c)proxy→faithfulギャップ定量=論文Table1

### ③ 水素タンク = H3 LH2/LOX FEM + BAM/DLR COPV(実測)  ……  **71 / 100** 🟢
- S1=10：LH2(33761 S8R)/LOX(12395 S8R) + COPV FEM
- S2=20：両タンクCOMPLETED、de-pole後 Mises 103
- S3=13：FSW溶接線シーム亀裂 node-split 実装（金属はdelam不可→溶接線が忠実）
- S4=**10**：掃引12ケースprep / 17 odb（部分）。⚠長亀裂800mmは COD>板厚半分で **nlgeom=YES 再solve必須**（線形妥当域逸脱）
- S5=**13**：pR/t=113.8 vs 求解103一致・COD vs Folias・極特異点de-pole処理＝物理検証強い
- S6=0：タンクでのML/GNN未着手
- S7=**5**（満点）：✅**BAM COPV 実測 end-to-end SHM pipeline 完了（2026-06-21）** — 25PZT×600pair，DI=1−corr，AUROC 0.92(RD)/0.94(ID) @180kHz，10×healthy floor，正しい重篤度順序（ID>RD），周波数選択性（60kHz失敗→180kHz最適），20–700bar圧力ロバスト，RAPID局所化6.6×null。全て [R]。`PAPER_DRAFT_copv_guidedwave.md` §5–7
- ✅**忠実度穴 解消(2026-06-21)**：COPV FEMを **CF+GFRP 8プライ直交異方性layup** に完全更新（T700M21 CF × TVR380M12R GFRP，CompositeShellSection）。CLT A-matrix: Ex=17.3 GPa(旧50 GPa)・Ey=60.0 GPa・ρ=1594 kg/m³。PBS job 40544(60kHz✅)→40545(120kHz R)→40546-48(H)・静的COPV_v2(±54°)=job40576 R。ply厚さ[U]
- **次の一手**：(a)PBS GW 5本完了→`extract_rx_batch.py`→`sim2real_batch.py`→dispersion再計算(CLT設計値 vs 実FEM速度) (b)COPV_v2完了→Mises比較(±54°vs±15°構造応答変化) (c)長亀裂nlgeom=YES再solve (d)S6前進=GNN分類試行（COPVは検出→分類の次ステップ）

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
- **強い＝段間①(データ90%+S6実行中)・水素タンク③(S7満点・物理検証◎)**。③のS7は「実データ end-to-end 完了」でポートフォリオ唯一の満点。
- **S6(ML/GNN)** が①以外では空。③は検出→分類まで進める余地あり。
- **止まってる＝フェアリング②B**（cohesive厚→再生成1コマンドで復活する既知案件）。
- 次に効く順：**①転移完走+8-class（S6確定・論文数値）＞ ②B cohesive再生成（S2=20回収）＞ ③PBS GW完了→sim2real更新**。

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
| フェアリングB 2.0mm cohesive収束確認 | ②S2=0→20 | H3_healthy_a2 確認 |

### 中期（〜7月末）= 論文形成フェーズ
| 目標 | 論文 | 内容 |
|---|---|---|
| 3構造 LOSO 統合評価 | 構造非依存SHM本体 | 段間+フェアリングGW+水素→cross-structure F1 |
| proxy→faithful ギャップ定量 | 段間S6の核心 | EXP1 vs EXP2b vs EXP3 F1差 |
| COPV sim2real dispersion | IWSHM候補 | PBS完了波形→実PZT F1比較 |

_更新規則：solve完了数・ML指標が動いたら S4/S6/S7 を実数で書き換える。inp生成やdatacheck通過では加点しない。_
