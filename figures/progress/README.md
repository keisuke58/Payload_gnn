# Dataset Progress Figures — 進捗確認用

教授との進捗確認で使用する可視化図の一覧。

## ファイル一覧

| ファイル | 内容 |
|----------|------|
| `01_summary_dashboard.png` | データセット全体のサマリ（サンプル数・品質・欠陥パラメータ範囲） |
| `02_feature_distributions.png` | 特徴量分布（欠陥半径・位置・ノード数・品質） |
| `03_3d_sample*_*.png` | 代表サンプルの 3D メッシュ（変位・温度・欠陥ラベルで色分け） |
| `04_unfolded_sample*_*.png` | 展開図（円筒を平面に展開、欠陥位置が直感的） |

## 再生成

```bash
python scripts/visualize_dataset_progress.py
# 特定サンプルを指定
python scripts/visualize_dataset_progress.py --samples "1,10,15,20"
```
