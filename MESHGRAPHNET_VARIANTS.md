# MeshGraphNet 系・Neural Operator 系モデル候補メモ

> 対象: H3 ペイロードフェアリング CFRP/Al-Honeycomb サンドイッチ構造の固定メッシュ SHM / 欠陥局在化。  
> 最終更新: 2026-06-06。arXiv で 2026-06-06 時点の関連ページを確認。

---

## 1. このレポでの前提

- 現行の実装は `src/train.py` から `--arch` を指定して GCN / GAT / GIN / GraphSAGE 系のノード分類を回す構成。
- データは CFRP/Al-HC フェアリングをメッシュグラフ化し、ノードごとに応力・温度・曲率・法線などの特徴から欠陥/健全を分類するタスクが中心。
- 直近の最適化対象は **「同じ/近い固定メッシュ上で、欠陥領域のノード分類 F1・Recall・境界精度を上げること」**。したがって、まずは既存 `train.py` に追加しやすい graph backbone を優先し、違う形状・違うメッシュ・連続場回帰へ進む段階で Neural Operator 系を本格化する。

---

## 2. ほかに試す価値がありそうなモデル

| 候補 | 主な狙い | 固定 CFRP メッシュ分類との相性 | 実装難度 | 短期優先度 | このレポでの使いどころ |
|---|---|---:|---:|---:|---|
| **MeshGraphNet-Transformer (MGN-T)** | MGN の局所 message passing に Transformer 系の大域 processor を足して長距離依存を直接扱う | ★★★★★ | ★★★★☆ | **最優先** | センサ/欠陥/境界条件が離れているケース、フェアリング全周の相関、under-reaching 対策 |
| **Transolver 系** | Physics-aware token / slice attention で不規則メッシュの PDE 的相関を線形に近い計算量で扱う | ★★★★★ | ★★★★☆ | **最優先** | 固定メッシュ上の応力・温度・曲率・異方性特徴から、遠方の物理状態をまとめて参照 |
| **LinearNO / LANO 系** | Transolver の slice/deslice を linear attention として再整理し、軽量な operator block にする | ★★★★☆ | ★★★★☆ | 高 | MGN-T/Transolver が重い場合の軽量 attention processor 候補 |
| **PGOT / GAOT 系** | geometry-aware operator transformer。境界・曲率・幾何埋め込みを明示的に保持 | ★★★★☆ | ★★★★★ | 中-高 | 曲率・境界条件・欠陥境界が精度ボトルネックになった後の発展候補 |
| **X-MeshGraphNet** | partition + halo + multi-scale graph で大規模 mesh/point cloud を扱う | ★★★★☆ | ★★★★☆ | 高 | 10k ノードから 100k ノード級へ拡張、全フェアリング高解像度化、推論時の mesh 依存低減 |
| **PIORF / physics-informed rewiring** | over-squashing bottleneck を物理量と Ollivier-Ricci curvature で rewiring | ★★★★☆ | ★★★☆☆ | 高 | 既存 GAT/GIN/SAGE に比較的小さく足せる長距離 edge augmentation |
| **GINO** | Graph Neural Operator + FNO で任意形状/任意離散化の連続場 operator learning | ★★★☆☆ | ★★★★★ | 中 | 違う形状、違うメッシュ、連続場回帰、Abaqus 代替 surrogate へ拡張するとき |
| **Graph Neural Operator (GNO)** | integral operator を graph message passing で近似し、離散化をまたぐ PDE surrogate を学習 | ★★★☆☆ | ★★★★☆ | 中 | 固定分類より、解演算子・解像度非依存 surrogate の基礎 |
| **Equivariant GNN / EGNO** | 3D 回転・並進・ベクトル/テンソル量への物理整合性 | ★★★☆☆ | ★★★★☆ | 中 | 変位・速度・応力テンソルや時系列ダイナミクスを扱う段階で有望 |
| **Graph Mamba / SSM processor** | O(N) に近い長距離依存、Transformer より軽い global context | ★★★★☆ | ★★★☆☆ | 高 | ノード数増大時に Transolver より実装を軽くしたい場合の代替 |

### 短期方針

CFRP 固定メッシュでは、まず **MeshGraphNet-Transformer** と **Transolver 系** を優先する。理由は次の 3 点。

1. **欠陥局在は長距離依存を含む**: CFRP/Al-HC の界面剥離では、局所応力だけでなく境界条件、曲率、熱応力、センサ配置、異方性方向が離れた位置から効く。
2. **既存のノード分類 API と相性がよい**: 既存の `train.py` はノード特徴 `x`、`edge_index`、`edge_attr`、ラベル `y` を使う構成なので、global processor を backbone として差し替えやすい。
3. **operator learning より評価が早い**: GINO / GNO は本質的には連続場回帰・形状汎化で強い。固定メッシュ分類だけなら、まず MGN-T / Transolver / rewiring の ablation を行う方が短期の論文・実装成果に直結する。

---

## 3. Transolver 系のこのレポ向け整理

### 3.1 physics-aware token / slice attention の意味

Transolver は、各メッシュ点をそのまま全点 attention するのではなく、**似た物理状態の点を learnable slice に割り当て、slice から physics-aware token を作って attention する**発想である。CFRP フェアリングでは、同じ欠陥近傍・同じ温度勾配・同じ曲率帯・同じ繊維方向応答を持つノードを、幾何的に離れていても同じ物理 token としてまとめられる可能性がある。

### 3.2 CFRP 固定メッシュで試す理由

- **長距離依存**: 局所 message passing では到達に多層が必要な遠方ノードを、slice token 経由で短絡できる。
- **異方性のクラスタリング**: CFRP の繊維方向、曲率、法線、熱応力などを入力特徴に含めると、単純な距離ではなく「物理状態の近さ」で attention できる。
- **計算量**: 全ノード self-attention より軽くできるため、10k ノード級グラフの batch 学習に現実味がある。
- **解釈性**: slice assignment を可視化すれば、欠陥境界・熱応力集中・曲率帯が別 token に分かれるかを確認できる。

### 3.3 既存 `train.py` へ統合する際の確認事項

| 確認項目 | 具体的な見るポイント |
|---|---|
| 入出力 shape | 既存モデルと同じく `data.x`, `data.edge_index`, `data.edge_attr`, `data.batch` を受け、ノード logits `(num_nodes, num_classes)` を返す。 |
| batching | PyG の複数 graph batch で slice attention が graph 間に漏れないよう、`batch` ごとに attention mask または graph-wise pooling を使う。 |
| edge_attr の扱い | Transolver 本体は点 token 寄りなので、edge_attr は MGN encoder や edge bias として残す。edge を捨てると既存メッシュ物理の利点を失う。 |
| 座標/幾何特徴 | `pos` がないデータでは、node feature 内の座標・法線・曲率 index を明示し、geometry MLP に入れる。 |
| class imbalance | 現行の focal loss、boundary-aware weighting、defect-centric sampler はそのまま比較に残す。backbone だけを変える ablation にする。 |
| 評価指標 | node F1 だけでなく defect Recall、境界 IoU、connected-component 単位の検出率、校正誤差を追加する。 |
| 可視化 | slice assignment / attention heatmap を欠陥 mask・曲率・温度・応力と重ねる。 |

### 3.4 Transolver 系の注意点

2025--2026 年の後続研究では、Transolver の効果は slice attention そのものより **slice/deslice 変換と linear attention 的な構造**に由来する可能性が指摘されている。したがって実装順は、(1) slice-token Transolver、(2) slice attention を簡略化した LinearNO 風 block、(3) MGN-T global processor、の 3 系統を同条件で比較するのがよい。

---

## 4. 2026-06 時点で追加で見るべき「最新寄り」アイデア

| 新しめの候補 | 何が新しいか | このレポへの仮説 | 優先実験 |
|---|---|---|---|
| **MGN-T (2026)** | MeshGraphNet の inductive bias を保ったまま global Transformer processor を使う | 固定メッシュ分類では GAT より境界条件・遠方欠陥の取り込みが強いはず | `--arch mgn_tiny` として hidden 64/128、global heads 4、local MGN 2層 + global 2層 |
| **PGOT (2025/2026)** | physics slicing に geometry injection と geometry aliasing 対策を入れる | 曲率・境界・欠陥端での誤分類を減らせる可能性 | 欠陥境界ノードだけの F1/IoU を主要指標に追加 |
| **LinearNO / LANO (2025/2026)** | physics attention をより軽い linear attention operator にする | 10k ノード級の GPU メモリを抑えつつ Transolver 近い長距離依存を得る | Transolver block の attention 部だけ差し替え可能な実装にする |
| **PIORF rewiring (2025)** | 物理量で長距離 edge を追加し over-squashing を減らす | 既存 GAT/GIN/SAGE の強化として最も安い。欠陥境界から遠方 sensor/BC への edge が効くか検証 | 前処理で stress/temperature/fiber 類似 top-k edge を追加する ablation |
| **GAOT (2025/2026)** | geometry-aware encoder/decoder + Transformer processor | GINO より Transformer 寄りで、将来の連続場 surrogate に移行しやすい | まず wave/stress field 回帰 dataset が揃ってから |
| **X-MeshGraphNet (2024)** | partition + halo + multi-scale point graph | 全周高解像度 mesh や STL からの推論では強い | 現行 10k graph ではなく、100k node 以上のロードマップ用 |

### 最短で成果を出すための実験順

1. **Baseline 固定**: 既存 GAT / GIN / SAGE を同じ seed、同じ sampler、同じ focal loss で再計測。
2. **PIORF 風 rewiring**: 実装が軽いので、追加 edge の効果を先に見る。改善すれば「長距離 edge が効く」という仮説を確認できる。
3. **MGN-Tiny**: local MGN encoder + 低層 global attention processor + node decoder。最もレポの `MeshGraphNet` 文脈と整合。
4. **Transolver-Tiny**: node を slice token に圧縮して global context を入れる。MGN-T と同等 parameter budget で比較。
5. **LinearNO/PGOT 要素**: Transolver が効く場合だけ、軽量化・geometry injection を追加する。
6. **GINO/GNO**: 固定分類ではなく、Abaqus 代替の連続場 surrogate、メッシュ解像度変更、形状違いへ移る段階で本格実装。

---

## 5. GINO / Graph Neural Operator の位置づけ

GINO と Graph Neural Operator は、**固定メッシュ分類を少し改善するための第一候補ではない**。むしろ次の拡張で有望である。

- **違うメッシュ**: 粗密や要素分割が違っても同じ operator として推論したい。
- **違う形状**: 平板、円筒、フェアリング ogive、局所補強部など形状をまたぐ。
- **連続場回帰**: 欠陥ラベルだけでなく、変位場、応力場、温度場、ガイド波場を直接予測する。
- **operator learning**: 荷重条件・境界条件・欠陥 mask から場全体への写像を学び、Abaqus / FEM を surrogate 化する。
- **データ生成高速化**: 欠陥パラメータ sweep を高速化し、GNN 分類器の augmentation に使う。

固定メッシュ分類の短期ゴールでは、GINO/GNO は「将来の surrogate backbone」として文献・設計を押さえ、現時点の優先実装は MGN-T / Transolver / rewiring に寄せる。

---

## 6. 参考文献

- MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation for Solid Mechanics, arXiv:2601.23177, 2026. https://arxiv.org/abs/2601.23177
- Transolver: A Fast Transformer Solver for PDEs on General Geometries, arXiv:2402.02366, 2024. https://arxiv.org/abs/2402.02366
- Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention, arXiv:2511.06294, 2025/2026. https://arxiv.org/abs/2511.06294
- PGOT: A Physics-Geometry Operator Transformer for Complex PDEs, arXiv:2512.23192, 2025/2026. https://arxiv.org/abs/2512.23192
- PIORF: Physics-Informed Ollivier-Ricci Flow for Long-Range Interactions in Mesh Graph Neural Networks, arXiv:2504.04052, 2025. https://arxiv.org/abs/2504.04052
- Geometry Aware Operator Transformer as an Efficient and Accurate Neural Surrogate for PDEs on Arbitrary Domains, arXiv:2505.18781, 2025/2026. https://arxiv.org/abs/2505.18781
- X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation, arXiv:2411.17164, 2024. https://arxiv.org/abs/2411.17164
- Geometry-Informed Neural Operator for Large-Scale 3D PDEs, arXiv:2309.00583, 2023. https://arxiv.org/abs/2309.00583
- Neural Operator: Graph Kernel Network for Partial Differential Equations, arXiv:2003.03485, 2020. https://arxiv.org/abs/2003.03485
- Neural Operator: Learning Maps Between Function Spaces, arXiv:2108.08481 / JMLR 2023, revised 2024. https://arxiv.org/abs/2108.08481
- Equivariant Graph Neural Operator for Modeling 3D Dynamics, arXiv:2401.11037 / ICML 2024. https://arxiv.org/abs/2401.11037

---

## 7. BibTeX

```bibtex
@misc{iparraguirre2026meshgraphnettransformer,
  title={MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation for Solid Mechanics},
  author={Mikel M. Iparraguirre and Iciar Alfaro and David Gonzalez and Elias Cueto},
  year={2026},
  eprint={2601.23177},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{wu2024transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  year={2024},
  eprint={2402.02366},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{hu2025transolverlinear,
  title={Transolver is a Linear Transformer: Revisiting Physics-Attention through the Lens of Linear Attention},
  author={Wenjie Hu and Sidun Liu and Peng Qiao and Zhenglun Sun and Yong Dou},
  year={2025},
  eprint={2511.06294},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{zhang2025pgot,
  title={PGOT: A Physics-Geometry Operator Transformer for Complex PDEs},
  author={Zhuo Zhang and Xi Yang and Ying Miao and Xiaobin Hu and Yifu Gao and Yuan Zhao and Yong Yang and Canqun Yang and Boocheong Khoo},
  year={2025},
  eprint={2512.23192},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{yu2025piorf,
  title={PIORF: Physics-Informed Ollivier-Ricci Flow for Long-Range Interactions in Mesh Graph Neural Networks},
  author={Youn-Yeol Yu and Jeongwhan Choi and Jaehyeon Park and Kookjin Lee and Noseong Park},
  year={2025},
  eprint={2504.04052},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{wen2025gaot,
  title={Geometry Aware Operator Transformer as an Efficient and Accurate Neural Surrogate for PDEs on Arbitrary Domains},
  author={Shizheng Wen and Arsh Kumbhat and Levi Lingsch and Sepehr Mousavi and Yizhou Zhao and Praveen Chandrashekar and Siddhartha Mishra},
  year={2025},
  eprint={2505.18781},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{nabian2024xmeshgraphnet,
  title={X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation},
  author={Mohammad Amin Nabian and Chang Liu and Rishikesh Ranade and Sanjay Choudhry},
  year={2024},
  eprint={2411.17164},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{li2023gino,
  title={Geometry-Informed Neural Operator for Large-Scale 3D PDEs},
  author={Zongyi Li and Nikola Borislavov Kovachki and Chris Choy and Boyi Li and Jean Kossaifi and Shourya Prakash Otta and Mohammad Amin Nabian and Maximilian Stadler and Christian Hundt and Kamyar Azizzadenesheli and Anima Anandkumar},
  year={2023},
  eprint={2309.00583},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{li2020graphkerneloperator,
  title={Neural Operator: Graph Kernel Network for Partial Differential Equations},
  author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
  year={2020},
  eprint={2003.03485},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@article{kovachki2023neuraloperator,
  title={Neural Operator: Learning Maps Between Function Spaces},
  author={Nikola Kovachki and Zongyi Li and Burigede Liu and Kamyar Azizzadenesheli and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={89},
  pages={1--97},
  year={2023},
  note={arXiv:2108.08481}
}

@misc{xu2024egno,
  title={Equivariant Graph Neural Operator for Modeling 3D Dynamics},
  author={Minkai Xu and Jiaqi Han and Aaron Lou and Jean Kossaifi and Arvind Ramanathan and Kamyar Azizzadenesheli and Jure Leskovec and Stefano Ermon and Anima Anandkumar},
  year={2024},
  eprint={2401.11037},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
