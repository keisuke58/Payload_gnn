# NVIDIA Japan 面接対策 — GPU/CUDAエンジニア

## ポートフォリオ要点

このポートフォリオ (`cuda_fem_solver.py`) では以下を実装している:
- **7本のカスタムCUDA Cカーネル** (CuPy RawKernel)
- **共有メモリ最適化** (tiled matmul, parallel reduction)
- **CG反復ソルバー** をGPUプリミティブから自作
- **Rooflineモデル分析** で性能特性を定量評価
- **メモリ転送トラッキング** (H2D/D2H帯域幅認識)

---

## 1. CUDAプログラミングモデル

### Q: CUDAのスレッド階層を説明してください

**Thread → Warp (32スレッド) → Block (max 1024スレッド) → Grid**

- **Thread**: 最小の実行単位。各スレッドが `threadIdx.x/y/z` を持つ
- **Warp**: 32スレッドが**SIMT** (Single Instruction, Multiple Thread) で同一命令を実行。GPUの真の実行単位
- **Block**: 共有メモリを共有するスレッド群。`__syncthreads()` で同期可能。1 SM上で実行
- **Grid**: カーネル起動全体。ブロック間は独立（同期不可）

```
kernel<<<grid_dim, block_dim, shared_mem>>>(args...)
```

**ポートフォリオでの実例**:
```python
kern((grid_size(n_elem),), (BLOCK_SIZE,), (...))
# grid_size = (n_elem + 255) // 256  → 要素数に応じたグリッド
# BLOCK_SIZE = 256 → 8 warps per block
```

### Q: Warpダイバージェンスとは？どう回避する？

同一Warp内のスレッドが異なるif分岐に入ると、両方の分岐を逐次実行するため性能低下。

**回避策**:
- 条件分岐をwarp境界に合わせる（`threadIdx.x < 32` は全スレッド同じ）
- データ依存の分岐をソート等で排除
- ポートフォリオの `dot_product` カーネルではreductionループが `s = blockDim/2` から半減していくので、warp内は常に同じ分岐を取る

### Q: Occupancyとは？

SM上で同時に実行可能なwarp数 / SM最大warp数。

**制約要因**:
1. レジスタ使用量 → 多すぎるとスレッド数制限
2. 共有メモリ使用量 → 多すぎるとブロック数制限
3. ブロック内スレッド数

**最適化**: `nvcc --ptxas-options=-v` でリソース使用量を確認し、レジスタスピルを減らす

---

## 2. GPUメモリ階層

### Q: GPUメモリ階層を速度順に説明してください

| レベル | 容量 | レイテンシ | スコープ |
|--------|------|-----------|---------|
| レジスタ | ~256KB/SM | 0 cycles | スレッド |
| 共有メモリ | 48-100KB/SM | ~20 cycles | ブロック |
| L1キャッシュ | 128KB/SM | ~30 cycles | SM |
| L2キャッシュ | 6-50MB | ~200 cycles | デバイス全体 |
| グローバルメモリ (HBM) | 24-80GB | ~400 cycles | デバイス全体 |

### Q: メモリコアレッシング (Coalesced Access) とは？

Warp内の32スレッドが連続したメモリアドレスにアクセスすると、1回のメモリトランザクションに統合される。

**良い例** (ポートフォリオのaxpyカーネル):
```c
// Thread i がアドレス i にアクセス → コアレスド
y[i] = alpha * x[i] + y[i];
```

**悪い例**: `y[i * stride]` (stride > 1)、`y[random_index[i]]` (SpMVのcol_idxアクセス)

### Q: 共有メモリのバンクコンフリクトとは？

共有メモリは32バンクに分割されている。同じバンクに2スレッドが同時アクセスするとシリアライズされる。

**回避**: パディングで別バンクに分散（`__shared__ float A[TILE][TILE+1]`）

**ポートフォリオの tiled_matmul**: TILE_SIZE=16 で 16x16 共有メモリタイルを使用。FP64なので8byte幅、バンクコンフリクトは最小限。

### Q: ピン留めメモリ (Pinned Memory) とは？

ページロックされたホストメモリ。H2D/D2H転送でDMAを使えるため、通常メモリの2-3倍の転送速度。非同期転送（cudaMemcpyAsync）も可能になる。

```python
# CuPyでの使用例
pinned = cp.cuda.alloc_pinned_memory(nbytes)
```

---

## 3. GPUアーキテクチャ

### Q: NVIDIA GPUアーキテクチャの世代を説明してください

| 世代 | 代表GPU | 特徴 |
|------|---------|------|
| Pascal (2016) | V100 | HBM2, NVLink 1.0 |
| Turing (2018) | RTX 2080Ti | RT Core, Tensor Core (INT8) |
| Ampere (2020) | A100/RTX 3090 | TF32, 3rd gen Tensor Core, MIG |
| Hopper (2022) | H100 | FP8, Transformer Engine, DPX |
| Blackwell (2024) | B200 | FP4, 2nd gen Transformer Engine |
| **面接時**: "現在RTX 4090(Ada Lovelace)で開発し、A100/H100への移植を想定した設計" |

### Q: Tensor Coreとは？

行列演算専用ハードウェア。4x4の行列積和 (D = A*B + C) を1サイクルで実行。
- FP16, BF16, TF32, FP8, INT8 をサポート
- 主にDL学習・推論で使用。HPC向けにはFP64 Tensor Core (A100以降)

### Q: NVLinkとは？

GPU間の高帯域インターコネクト。
- NVLink 4.0 (H100): 双方向 900GB/s
- PCIe Gen5: 双方向 128GB/s
- **用途**: マルチGPU学習、GPUダイレクト通信

---

## 4. パフォーマンス最適化

### Q: Rooflineモデルとは？

カーネルの性能上限を可視化するモデル。

```
Attainable Performance = min(Peak FLOP/s, Peak BW * Arithmetic Intensity)
```

- **Arithmetic Intensity (AI)** = FLOP / bytes transferred
- **Ridge Point**: AI > ridge → compute-bound, AI < ridge → memory-bound

**ポートフォリオでの分析結果**:
- SpMV: AI ≈ 0.07 → **memory-bound** (典型的なFEM/CG)
- Dot product: AI = 0.125 → **memory-bound**
- CG全体: memory-bound → **メモリ帯域幅の最適化が最重要**

### Q: カーネルフュージョンとは？

複数の小さなカーネルを1つに統合して、グローバルメモリの読み書き回数を減らす。

**ポートフォリオの例**:
```c
// Separate: 2回読み + 2回書き
y = alpha * x + y;  // axpy: read x,y, write y
z = a * x + b * y;  // scale_add: read x,y, write z

// Fused scale_add: 2回読み + 1回書き（33%帯域幅削減）
z[i] = a * x[i] + b * y[i];
```

### Q: プロファイリングツールは？

- **Nsight Compute**: カーネルレベルの詳細分析（occupancy, メモリスループット, 命令スループット）
- **Nsight Systems**: タイムライン分析（CPU-GPU同期、カーネル起動オーバーヘッド）
- **nvprof** (legacy): 簡易プロファイリング

---

## 5. 数値計算 on GPU

### Q: SpMVがGPUで困難な理由は？

1. **不規則メモリアクセス**: CSRのcol_idx経由でxベクトルにランダムアクセス → キャッシュ効率悪い
2. **低Arithmetic Intensity**: AI ≈ 0.07 → 完全にmemory-bound
3. **負荷不均衡**: 行ごとに非零要素数が異なる → thread間の仕事量がばらつく

**改善策**:
- CSR5/ELL/SELL-C-σ等の高性能フォーマット
- Warp単位で行を処理（short rows → 1 warp = 1 row, long rows → 複数warp）
- cuSPARSEが最適フォーマットを自動選択

### Q: CGソルバーをGPUで実装する際の注意点は？

1. **各イテレーションは逐次的** → ループ自体は並列化不可
2. **各操作は並列**: SpMV, dot, axpy が独立カーネル → **カーネル起動オーバーヘッド**が支配的になりうる
3. **収束性**: 前処理なしCGは条件数が大きいと遅い → **Jacobi/ILU前処理**が実用的
4. **精度**: FP32は丸め誤差蓄積 → CG はFP64推奨（ポートフォリオもFP64使用）

**ポートフォリオでの実装**:
```
1 CG iteration = spmv + 2×dot + axpy + scale_add + max_abs
全て自作CUDAカーネルで構成
```

---

## 6. ポートフォリオ → NVIDIA製品への接続

### どうNVIDIA事業に貢献できるか

| ポートフォリオの技術 | NVIDIA製品/事業 |
|---------------------|----------------|
| カスタムCUDAカーネル | CUDA Toolkit, cuSPARSE, cuSOLVER 開発 |
| FEM+GPU | NVIDIA Modulus (Physics-ML), SimNet |
| Roofline分析 | Nsight Compute の性能分析チーム |
| GNN-SHM (本研究) | cuGraph, PyG on GPU, NVIDIA Omniverse |
| 数値線形代数 | AmgX (代数マルチグリッド on GPU) |
| HPC × AI | Digital Twin (自動車/航空宇宙顧客) |

### キートーキングポイント

1. **"FEMの全パイプラインをGPU化した経験がある"**
   - アセンブリ、ソルバー、後処理まで自作カーネルで実装
   - 単にライブラリを呼ぶだけでなく、内部アルゴリズムを理解している

2. **"Memory-bound問題への対処法を理解している"**
   - FEM/CFDの多くはmemory-bound → 帯域幅最適化がカギ
   - カーネルフュージョン、共有メモリ活用、コアレスドアクセス

3. **"研究でGNN+FEMの融合をやっている"**
   - NVIDIA Modulusの Physics-Informed ML と同じ方向性
   - Graph Neural Network = PyG on GPU → cuGraph と親和性高い

4. **"ドイツの大学(LUH)でのダブルディグリー + 英語力"**
   - NVIDIAはグローバルチーム → 英語でのコミュニケーション必須
   - ドイツ留学で異文化チームワーク経験

---

## 7. 想定される行動面接質問

### Q: なぜNVIDIA Japan?

"ロケットフェアリングの構造ヘルスモニタリング研究で、FEM+GNNのGPU実装をやっています。
FEMのボトルネックがGPU上のSpMVだと気づき、カスタムCUDAカーネルを書いて
Roofline分析をするうちに、GPU Computing自体に深い興味を持ちました。
NVIDIAはHPC/AIの両方でGPUを使う唯一の企業で、
自分のFEM+ML+CUDAの経験が最もレバレッジできる場所だと思います。"

### Q: 最も困難だった技術的課題は？

"FEMのペナルティ法BCを適用した大規模疎行列のCGソルバーをGPUで動かした時、
条件数が10^15のオーダーになり、FP32では発散しました。
原因はペナルティ係数が大きすぎることと、
SpMVカーネルの丸め誤差蓄積でした。
FP64に切り替え、Jacobi前処理を追加することで解決しました。
この経験から、GPUの数値精度と性能のトレードオフを深く理解しています。"

### Q: チームでの開発経験は？

"LUH（ドイツ）での共同研究で、ドイツ人・インド人・中国人のチームで
FEMコードを開発しました。GitHubでPR/レビュー、
週次ミーティングは全て英語で、
異なるバックグラウンドのメンバーと技術議論する経験を積みました。"

---

## 8. コーディング面接対策

### 典型的なCUDA面接コーディング問題

1. **並列Prefix Sum (Scan)**
   - Blellochアルゴリズム: up-sweep → down-sweep
   - 共有メモリ使用、バンクコンフリクト回避

2. **ヒストグラム (Histogram)**
   - atomicAdd でグローバルメモリ → 遅い
   - 共有メモリで部分ヒストグラム → グローバルに集約

3. **行列転置 (Transpose)**
   - Naive: 列方向読み取り = 非コアレスド
   - Tiled: 共有メモリに読み込み → 転置して書き出し

4. **SpMV CSR** (ポートフォリオで実装済み)
   - 1 thread per row → 基本版
   - 1 warp per row → 長い行に対応

### C++ コーディングスタイル

NVIDIAの面接ではC++も重要:
- Modern C++ (C++17/20): `auto`, `constexpr`, `std::variant`, CTAD
- テンプレートメタプログラミング
- RAII、ムーブセマンティクス
- STL アルゴリズム (`std::transform`, `std::reduce`)

---

## 9. 面接プロセス (NVIDIA Japan)

| ステップ | 内容 | 対策 |
|---------|------|------|
| 1. 書類選考 | 履歴書 + ポートフォリオ | GitHubリンク必須 |
| 2. OA/電話面接 | コーディング or 技術質問 | LeetCode Medium + CUDA基礎 |
| 3. 技術面接 1 | CUDAプログラミング | 上記Q&A + ホワイトボード |
| 4. 技術面接 2 | システム設計 or ドメイン | FEM+GPU設計を説明 |
| 5. 行動面接 | カルチャーフィット | 上記行動質問 |
| 6. (場合により) チーム面接 | ペアプログラミング | CUDA kernel を一緒に書く |

**想定所要時間**: 1-2ヶ月
**使用言語**: 英語 (一部日本語OK)
**リモート**: ほぼ全てオンライン (最終のみオンサイトの場合あり)
