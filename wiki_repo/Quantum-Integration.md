# 量子統合ガイド（AE/QAOA 入口）

## 目的
- UQ（確率推定）と離散設計探索に量子計算の考え方を導入し、研究の将来拡張に備える
- 現状はシミュレータ上での PoC を想定（Qiskit/Aer）

## インストール

```bash
python -m pip install --upgrade pip
python -m pip install qiskit qiskit-aer qiskit-algorithms qiskit-optimization
```

## 動作確認（最小）

```python
from qiskit import QuantumCircuit
from qiskit_aer import Aer

backend = Aer.get_backend('aer_simulator')
qc = QuantumCircuit(1, 1)
qc.h(0); qc.measure(0, 0)
res = backend.run(qc, shots=1024).result().get_counts(qc)
print(res)  # おおむね {'0': ~512, '1': ~512}
```

## PoC シナリオ

- 振幅推定（AE）
  - 目的: p = P(metric ≥ threshold) の推定
  - 構成: 粗モデル（FNO/GNN）で metric を生成 → 閾値判定オラクル → AE（Iterative / MLAE）
  - 効果: 理論上はサンプル複雑性 O(1/ε)（古典 O(1/ε^2) 比）

- QAOA（離散最適化）
  - 目的: リブ配置・ベントパターンなどの組合せ設計
  - 構成: QUBO 定式化 → QAOA で近似探索 → 古典メタヒューリスティクスと比較

## AE の最小コード例（振幅エンコード）

```python
import math
from qiskit import QuantumCircuit
from qiskit_aer import Aer

p = 0.42
theta = math.asin(p ** 0.5)
qc = QuantumCircuit(1, 1)
qc.ry(2*theta, 0)
qc.measure(0, 0)
backend = Aer.get_backend('aer_simulator')
res = backend.run(qc, shots=4096).result().get_counts(qc)
print(res)  # '1' の比率 ≈ p
```

## QAOA の入口（概念）

```python
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit_aer.primitives import Estimator
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit

qp = QuadraticProgram()
qp.binary_var('x0'); qp.binary_var('x1')
qp.minimize(constant=0, linear={'x0': -1, 'x1': -1}, quadratic={('x0','x1'): 2})
estimator = Estimator()
qaoa = QAOA(estimator=estimator, reps=1, optimizer=None)
opt = MinimumEigenOptimizer(qaoa)
result = opt.solve(qp)
print(result)
```

## パイプライン統合の考え方
- 粗モデル出力から metric（応力・変位・温度など）を得て閾値判定（合否）を作成
- AE: 判定オラクルを回路に組み込み、p を推定（シミュレータ上）
- QAOA: 小規模 QUBO を構成し、設計案の探索を実施

## 評価軸
- 計算量: 試行数・反復数（古典 vs 量子）
- 精度: ε（推定誤差）、δ（失敗確率）
- 再現性: 乱数固定、設定出力の記録
- スケーラビリティ: 変数数・回路深さ・実行時間
