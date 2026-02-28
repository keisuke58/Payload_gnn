# 公開データセットと統合ガイド

> FEniCS/JAX-FEM プロトタイプ、外部データの統合方針  
> 最終更新: 2026-02-28

---

## 1. 概要

本プロジェクトは **Abaqus シミュレーション** を主データソースとするが、以下を並行して検討する:

| カテゴリ | 内容 |
|----------|------|
| **代替 FEM** | FEniCS, JAX-FEM プロトタイプ（`src/fenics_fairing.py`, `src/jaxfem_fairing.py`） |
| **公開データ** | OGW, NASA CFRP, SimuStruct 等の転移学習・検証 |
| **開口部** | H3 フェアリングのアクセスドア・空調ドアを Abaqus モデルに追加 |

---

## 2. 代替 FEM プロトタイプ

### 2.1 FEniCS (dolfinx)

```bash
# インストール
pip install fenics-dolfinx  # or: conda install -c conda-forge fenics-dolfinx

# 実行（簡易平板モデル）
python src/fenics_fairing.py --output dataset_output/fenics_proto

# 2D 軸対称（実験的）
python src/fenics_fairing.py --2d
```

| モード | 説明 | 出力 |
|--------|------|------|
| デフォルト | 2D 平板（線形弾性） | nodes.csv, elements.csv |
| `--2d` | 軸対称 1D | displacement.xdmf |
| `--3d` | 1/4 円筒（pygmsh） | .msh |

**制限**: サンドイッチ・直交異方性・接触は未実装。線形弾性の検証用。

### 2.2 JAX-FEM

```bash
pip install jax jaxlib jax-fem

python src/jaxfem_fairing.py --output dataset_output/jaxfem_proto
```

| 利点 | 用途 |
|------|------|
| **微分可能** | 逆問題、感度解析、PINN 統合 |
| GPU 対応 | バッチ計算の高速化 |
| 軽量 | jax-fem が無くても JAX のみで最小プロトタイプ動作 |

**制限**: jax-fem の API は開発中。複雑なジオメトリは要検証。

### 2.3 パイプライン統合

FEniCS/JAX-FEM 出力を `build_graph.py` に渡す場合:

```bash
# nodes.csv, elements.csv が Abaqus と同じ形式なら
python src/build_graph.py dataset_output/fenics_proto fairing_graph.pt
```

CSV 形式: `node_id,x,y,z,ux,uy,uz,temp` (nodes), `elem_id,elem_type,n1,n2,n3,n4,area` (elements)

---

## 3. 公開データセット一覧

詳細は [Dataset-Survey](../wiki_repo/Dataset-Survey.md) を参照。

### 3.1 必須（ダウンロード推奨）

| データセット | URL | 用途 |
|-------------|-----|------|
| **OGW #4** | https://doi.org/10.5281/zenodo.5105861 | デボンディング検出ベースライン |
| **NASA CFRP** | https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip | 疲労損傷、転移学習 |

### 3.2 補助

| データセット | 内容 | 用途 |
|-------------|------|------|
| OGW #2 | 温度変動下の誘導波 | 温度補償 |
| DINS-SHM | スペクトル有限要素シミュ | パイプライン検証 |
| SimuStruct | 鋼板 6 穴、von Mises 応力 | GNN/FNO 検証 |
| Mechanical MNIST | 相場き裂、変位場 | U-Net/FNO |
| SHM Open DB (Bologna) | 複合材翼、160 PZT | GNN センサグラフ |

---

## 4. ダウンロードスクリプト

```bash
# scripts/download_external_datasets.sh を実行
bash scripts/download_external_datasets.sh
```

保存先: `data/external/` (.gitignore 対象)

---

## 5. 統合方針

### 5.1 転移学習

1. **NASA CFRP** で事前学習（疲労損傷パターン）
2. **OGW #4** でデボンディング特化ファインチューニング
3. **自作 Abaqus** で H3 フェアリングに適応

### 5.2 検証

- FEniCS 平板 vs Abaqus 平板: 単純ケースで解の一致を確認
- OGW #4 のデボンディング検出精度をベースラインとして報告

### 5.3 データ形式の統一

| 形式 | 用途 |
|------|------|
| nodes.csv + elements.csv | build_graph.py 入力 |
| HDF5 (波場) | OGW, FNO 入力 |
| PyTorch Geometric Data | GNN 学習 |

---

## 6. 参考文献

- Kudela et al., Data in Brief 42, 108078 (2022) — OGW #4
- JAX-FEM: https://github.com/deepmodeling/jax-fem
- FEniCSx: https://fenicsproject.org/download/
