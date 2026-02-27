[← Home](Home)

# 難しい英単語集 — Technical Vocabulary

> 最終更新: 2026-02-28  
> 本プロジェクトで頻出する専門用語・略語の日本語訳と簡易説明。

---

## 1. 機械学習・深層学習

| 英語 | 読み | 日本語 | 補足 |
|------|------|--------|------|
| **GNN** | ジーエヌエヌ | グラフニューラルネットワーク | グラフ構造データを扱う NN |
| **GCN** | ジーシーエヌ | Graph Convolutional Network | スペクトルフィルタリング |
| **GAT** | ギャット | Graph Attention Network | アテンションで隣接重み付け |
| **GIN** | ジン | Graph Isomorphism Network | WL テストと等価の表現力 |
| **GraphSAGE** | グラフセージ | Graph Sample and Aggregate | サンプリング＋集約、スケーラブル |
| **over-squashing** | オーバースクワッシュ | 情報圧縮 | GNN で遠くのノード情報が失われる |
| **over-smoothing** | オーバースムーシング | 過平滑化 | 層を重ねると表現が似通う |
| **Focal Loss** | フォーカルロス | 焦点損失 | クラス不均衡対策の損失関数 |
| **AUC** | エーユーシー | 曲線下面積 | ROC 曲線の下の面積、0–1 |
| **IoU** | アイオーユー | Intersection over Union | 検出領域の重なり率 |
| **Zero-Shot** | ゼロショット | ゼロショット | 追加学習なしで未知条件に汎化 |
| **Fine-Tune** | ファインチューン | 微調整 | 事前学習モデルをタスクに適応 |
| **DANN** | ダン | Domain Adversarial NN | ドメイン識別を欺く転移学習 |
| **FNO** | エフエヌオー | Fourier Neural Operator | フーリエ空間で PDE 解演算子を学習 |
| **PINN** | ピン | Physics-Informed Neural Network | 損失に物理方程式を組み込み |
| **SSM** | エスエスエム | State Space Model | 状態空間モデル。Mamba の基盤 |
| **Equivariant** | イクイバリアント | 等変 | 回転・並進に対して対応して変わる |
| **Invariant** | インバリアント | 不変 | 回転・並進で変わらない |

---

## 2. 構造・材料・SHM

| 英語 | 読み | 日本語 | 補足 |
|------|------|--------|------|
| **SHM** | エスエイチエム | 構造ヘルスモニタリング | Structural Health Monitoring |
| **CFRP** | シーエフアールピー | 炭素繊維強化プラスチック | Carbon Fiber Reinforced Polymer |
| **debonding** | デボンディング | 界面剥離 | スキン-コアの接着不良 |
| **delamination** | デラミネーション | 層間剥離 | 積層内の剥離 |
| **BVID** | ビビッド | 目視不能衝撃損傷 | Barely Visible Impact Damage |
| **NDI/NDT** | エヌディーアイ/エヌディーティー | 非破壊検査 | Non-Destructive Inspection/Testing |
| **PZT** | ピーゼット | 圧電素子 | 圧電トランスデューサ |
| **PWAS** | ピーワス | 圧電ウェハ | Piezoelectric Wafer Active Sensor |
| **Lamb wave** | ラム波 | ラム波 | 板を伝わる弾性波 |
| **pitch-catch** | ピッチキャッチ | 送受信 | 送信-受信対で計測 |
| **CTE** | シーティーイー | 熱膨張係数 | Coefficient of Thermal Expansion |
| **CBM** | シービーエム | 状態基準保全 | Condition-Based Maintenance |
| **PSS** | ピーエスエス | 衛星搭載構造 | Payload Support Structure |

---

## 3. FEM・シミュレーション

| 英語 | 読み | 日本語 | 補足 |
|------|------|--------|------|
| **FEM** | エフイーエム | 有限要素法 | Finite Element Method |
| **ODB** | オーディービー | 出力データベース | Abaqus の結果ファイル |
| **DOE** | ディーオーイー | 実験計画法 | Design of Experiments |
| **mesh** | メッシュ | メッシュ | 有限要素の分割 |
| **node** | ノード | 節点 | メッシュの頂点 |
| **element** | エレメント | 要素 | メッシュのセル |
| **von Mises** | フォン・ミーゼス | フォン・ミーゼス応力 | 等価応力の指標 |
| **shell element** | シェルエレメント | シェル要素 | 板・殻を表現 |
| **solid element** | ソリッドエレメント | 立体要素 | 3D 体積を表現 |
| **Tie constraint** | タイ拘束 | タイ拘束 | 面同士を剛体接合 |

---

## 4. 学会・論文

| 英語 | 読み | 日本語 | 補足 |
|------|------|--------|------|
| **IWSHM** | アイダブリューエスエイチエム | 構造ヘルスモニタリング国際ワークショップ | 隔年 Stanford 等 |
| **JSASS** | ジェイサス | 日本航空宇宙学会 | The Japan Society for Aeronautical and Space Sciences |
| **JCCM** | ジェイシーシーエム | 日本複合材料会議 | Japan Joint Conference on Composite Materials |
| **ICCM** | アイシーシーエム | 国際複合材料会議 | International Conference on Composite Materials |
| **NDT & E** | エヌディーティーアンドイー | 非破壊検査国際誌 | NDT & E International |
| **IF** | アイエフ | インパクトファクター | Impact Factor |
| **Q1** | キューわん | 第1四分位 | ジャーナルランキング上位 25% |

---

## 5. その他

| 英語 | 読み | 日本語 | 補足 |
|------|------|--------|------|
| **Sim-to-Real** | シムトゥリアル | シミュレーションから実機へ | シミュで学習したモデルを実データに適用 |
| **surrogate** | サロゲート | 代理モデル | 高コスト計算の近似モデル |
| **benchmark** | ベンチマーク | ベンチマーク | 性能比較の基準 |
| **ablation** | アブレーション | アブレーション | 要素を除去して寄与を評価 |
| **manifold** | マニフォールド | 多様体 | 曲がった空間。フェアリング表面など |
| **Euclidean** | ユークリッド | ユークリッド | 平坦な空間。直交座標 |
| **Ogive** | オジャイブ | オジャイブ | 流線形の先端形状 |
| **AFP** | エーエフピー | 自動繊維配置 | Automated Fiber Placement |
| **OoA** | オーオーエー | 脱オートクレーブ | Out of Autoclave |

---

## 6. 略語一覧 (アルファベット順)

| 略語 | 展開 | 日本語 |
|------|------|--------|
| ACCM | Asian-Australasian Conference on Composite Materials | アジア・オセアニア複合材料会議 |
| AFP | Automated Fiber Placement | 自動繊維配置 |
| AUC | Area Under the Curve | 曲線下面積 |
| BVID | Barely Visible Impact Damage | 目視不能衝撃損傷 |
| CBM | Condition-Based Maintenance | 状態基準保全 |
| CFRP | Carbon Fiber Reinforced Polymer | 炭素繊維強化プラスチック |
| CTE | Coefficient of Thermal Expansion | 熱膨張係数 |
| DANN | Domain Adversarial Neural Network | ドメイン敵対 NN |
| DOE | Design of Experiments | 実験計画法 |
| FEM | Finite Element Method | 有限要素法 |
| FNO | Fourier Neural Operator | フーリエニューラル演算子 |
| GAT | Graph Attention Network | グラフアテンションネットワーク |
| GCN | Graph Convolutional Network | グラフ畳み込みネットワーク |
| GIN | Graph Isomorphism Network | グラフ同型ネットワーク |
| GNN | Graph Neural Network | グラフニューラルネットワーク |
| ICCM | International Conference on Composite Materials | 国際複合材料会議 |
| IF | Impact Factor | インパクトファクター |
| IoU | Intersection over Union | 交差/合併比 |
| IWSHM | International Workshop on Structural Health Monitoring | SHM 国際ワークショップ |
| JCCM | Japan Joint Conference on Composite Materials | 日本複合材料会議 |
| JSASS | Japan Society for Aeronautical and Space Sciences | 日本航空宇宙学会 |
| NDI | Non-Destructive Inspection | 非破壊検査 |
| NDT | Non-Destructive Testing | 非破壊試験 |
| OGW | Open Guided Waves | オープン誘導波（データセット） |
| OoA | Out of Autoclave | 脱オートクレーブ |
| PINN | Physics-Informed Neural Network | 物理情報 NN |
| PSS | Payload Support Structure | 衛星搭載構造 |
| PZT | Lead Zirconate Titanate (piezoelectric) | 圧電素子 |
| PWAS | Piezoelectric Wafer Active Sensor | 圧電ウェハアクティブセンサ |
| SHM | Structural Health Monitoring | 構造ヘルスモニタリング |
| SSM | State Space Model | 状態空間モデル |
| UV mapping | — | UV マッピング（曲面の 2D 展開） |
