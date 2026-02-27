[← Home](Home)

# Analysis Results: H3 Fairing (Debonding Defect Case)

> **Simulation**: Defect Impact Analysis (Simulated)
> **Model**: H3 Type-S (Dia 5.2m, Len 10.4m)
> **Condition**: Max Q Load Case (~50 kPa Dynamic Pressure)

本ページでは、フェアリング構造内部に**デボンディング（剥離）欠陥**が存在する場合の構造挙動を可視化しています。
健全状態（Healthy）との差分（Residual）を解析することで、GNNモデルが学習すべき「欠陥のシグナル」を特定します。

## 1. 変位解析における欠陥シグナル (Displacement Residual)

バレル部（Z=2500mm付近）に大規模なデボンディング（剛性低下領域）を模擬した場合の変位分布比較です。

| Healthy | Defective | Residual (Signal) |
|:---:|:---:|:---:|
| ![Disp Healthy](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/defects/displacement_comparison.png) | (See Comparison) | (See Comparison) |

*   **現象**: 剥離により局所剛性が低下し、内圧・空力荷重により**局所的な膨らみ（Bulge）**が発生します。
*   **シグナル**: 全体変形（数mmオーダー）に対し、欠陥による変位増分は微小（<1mm）ですが、Residualプロットでは明確な「ホットスポット」として現れます。
*   **3D可視化**:
    ![Defect 3D](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/defects/defect_3d_residual.png)
    *   3D空間上での欠陥位置の特定が容易であることがわかります。

## 2. 応力集中と再分布 (Stress Concentration)

ノーズ部（Z=6000mm付近）のデボンディング周辺での応力変化。

| Stress Analysis Comparison (Healthy vs Defective vs Residual) |
|:---:|
| ![Stress Comparison](https://raw.githubusercontent.com/keisuke58/Payload_gnn/main/wiki_repo/images/defects/stress_comparison.png) |

*   **現象**: 欠陥エッジ周辺に応力集中が発生（Stress Concentration）。一方で、荷重伝達が遮断される剥離内部では応力が低下するケースもあります（本シミュレーションでは簡易的に応力集中を強調表示）。
*   **検出難易度**: 変位に比べて応力変化は局所的かつ急峻であるため、センサー配置密度が検出性能に大きく影響します。

## 3. GNNモデルへの示唆 (Implications for GNN)

*   **入力特徴量**: 
    *   単なる「変位」や「歪み」の絶対値だけでなく、**近傍ノードとの差分（勾配）**や、健全状態の予測値との**残差（Residual）**を学習させることが有効です。
*   **グラフ構造**:
    *   欠陥による影響は局所的であるため、Graph Conv層での近傍集約（Message Passing）により、局所的な異常パターンを検出可能です。

---

> *Note: These visualizations simulate the structural impact of debonding defects using analytical perturbation models.*
