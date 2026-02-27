[← Home](Home)

# Analysis Results: H3 Type-S Fairing (Healthy Baseline)

> **Simulation**: Analytical Structural Response (Simulated)
> **Model**: H3 Type-S (Dia 5.2m, Len 10.4m)
> **Condition**: Max Q Load Case (~50 kPa Dynamic Pressure)

本ページでは、H3フェアリング（Type-S）のFEMモデルに基づく構造解析結果（変位、応力、振動モード）の可視化データを掲載します。
これらは、デボンディング欠陥検出アルゴリズム（GNN）の学習データセットのベースラインとなる「健全状態（Healthy）」の挙動です。

## 1. 変位解析 (Displacement)

最大動圧時（Max Q）の空気力および慣性力に対するフェアリングの変形挙動。
片持ち梁としての曲げ変形に加え、シェル構造特有の面外変形（Breathing Mode）が確認できます。

| 3D Displacement Field | Unfolded 2D Map (Theta-Z) |
|:---:|:---:|
| ![Displacement 3D](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/analysis/displacement_3d.png) | ![Displacement 2D](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/analysis/displacement_2d_map.png) |

*   **最大変位**: 先端部およびバレル中央部で発生。
*   **挙動**: 軸対称荷重下でも、内部補強材やフェアリング分割面の影響により局所的な変形モードが現れます。

## 2. 応力解析 (Von Mises Stress)

CFRPフェースシートに発生する Von Mises 相当応力の分布。
ペイロードアダプタとの結合部（Base, Z=0）に最大の曲げ応力が集中します。

| 3D Stress Distribution | Unfolded 2D Map (Theta-Z) |
|:---:|:---:|
| ![Stress 3D](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/analysis/stress_3d.png) | ![Stress 2D](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/analysis/stress_2d_map.png) |

*   **最大応力**: 固定端（Z=0）付近で約 **75 MPa**。
*   **安全性**: CFRP (T1000G) の引張強度は >2000 MPa であり、十分な安全率が確保されています。
*   **上部**: ノーズコーン（Ogive）部分は比較的低応力ですが、空力加熱による熱応力の影響を受ける可能性があります（本解析では機械的荷重のみ）。

## 3. 固有値解析 (Modal Analysis)

第1次曲げモード（1st Bending Mode）のモードシェイプ。
ロケット全体の制御系設計において重要なパラメータです。

![Mode Shape 1](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/analysis/mode_shape_1.png)

*   **周波数**: **63.1 Hz** (Analytical Estimate)
*   **モード**: シンプルな片持ち梁の1次曲げモード。
*   **意義**: この振動モードの変化をモニタリングすることで、大規模な構造欠陥や剛性低下を検知できる可能性があります。

---

> *Note: These visualizations are generated from the analytical physics model of the H3 fairing structure.*
