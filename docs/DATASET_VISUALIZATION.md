# Dataset Visualization — 進捗確認用

メインリポジトリ内の Markdown で画像を表示（GitHub で確実に表示される）。

## サマリダッシュボード

![Dataset Summary](https://github.com/keisuke58/Payload_gnn/raw/main/wiki_repo/images/progress/01_summary_dashboard.png)

## 特徴量分布

![Feature Distributions](https://github.com/keisuke58/Payload_gnn/raw/main/wiki_repo/images/progress/02_feature_distributions.png)

## 代表サンプル 3D 可視化

**Sample 0003** — 変位 / 温度 / 欠陥

![3D displacement](https://github.com/keisuke58/Payload_gnn/raw/main/wiki_repo/images/progress/03_3d_sample0003_displacement.png)
![3D temp](https://github.com/keisuke58/Payload_gnn/raw/main/wiki_repo/images/progress/03_3d_sample0003_temp.png)
![3D defect](https://github.com/keisuke58/Payload_gnn/raw/main/wiki_repo/images/progress/03_3d_sample0003_defect.png)

## 展開図

![Unfolded displacement](https://github.com/keisuke58/Payload_gnn/raw/main/wiki_repo/images/progress/04_unfolded_sample0003_displacement.png)
![Unfolded defect](https://github.com/keisuke58/Payload_gnn/raw/main/wiki_repo/images/progress/04_unfolded_sample0003_defect.png)

## 再生成

```bash
python scripts/visualize_dataset_progress.py --samples "1,2,3,10,15,20"
```
