# GitHub リポジトリ設定ガイド — 界隈で有名になるために

> リポジトリの説明・トピック・ソーシャルプレビューを設定して、GitHub での発見性を最大化する。

---

## 1. リポジトリ説明 (Description)

**Settings → General → Description** に以下を設定:

```
GNN + FEM for debonding detection on JAXA H3 CFRP fairing. PyTorch Geometric, Abaqus. Structural Health Monitoring.
```

または日本語:
```
JAXA H3 フェアリングのデボンディング検出。GNN + FEM。PyTorch Geometric, Abaqus。
```

---

## 2. トピック (Topics)

**Settings → General → Topics** に追加 (最大 20 個):

```
gnn
graph-neural-networks
structural-health-monitoring
cfrp
aerospace
jaxa
abaqus
pytorch-geometric
deep-learning
defect-detection
debonding
composite-materials
finite-element-method
machine-learning
python
```

---

## 3. ソーシャルプレビュー画像

**Settings → General → Social preview** で画像をアップロード。

推奨サイズ: **1280 × 640 px**

内容例:
- ロゴ + 「GNN-SHM: H3 Fairing Debonding Detection」
- パイプライン図 (FEM → GNN → API)
- フェアリング 3D 可視化 + Defect Heatmap

`figures/` の既存画像を加工するか、[Canva](https://canva.com) 等で作成。

---

## 4. ウェブサイト (Website)

**Settings → General → Website** に Wiki またはドキュメント URL を設定:

```
https://github.com/keisuke58/Payload_gnn/blob/main/wiki_repo/Home.md
```

---

## 5. その他の推奨設定

| 項目 | 推奨 |
|------|------|
| **Issues** | 有効 |
| **Discussions** | 有効 (Q&A 用) |
| **Projects** | ロードマップ可視化に活用 |
| **Wiki** | wiki_repo で管理 → GitHub Wiki に自動同期 |

---

## 6. Wiki の push（wiki_repo）

`wiki_repo` は main リポジトリ内で管理。push は以下で完璧に実行:

```bash
# 変更をコミットして push（デフォルトメッセージ）
./scripts/wiki_push.sh

# カスタムメッセージ
./scripts/wiki_push.sh "docs(wiki): add Defect-Physics-Validation"

# ドライラン（実際には push しない）
./scripts/wiki_push.sh -n

# 非対話モード（CI 用）
./scripts/wiki_push.sh -y "docs(wiki): automated update"
```

スクリプトは以下を自動実行:
- リモート確認
- 画像参照の簡易検証
- `wiki_repo/` の stage → commit → push

### 自動化（GitHub Actions）

`main` に push すると、**wiki_repo の変更があれば GitHub Wiki へ自動同期**されます。

| トリガー | 動作 |
|----------|------|
| `wiki_repo/**` または `scripts/sync_github_wiki.py` が変更された push | `sync_github_wiki.py` を実行 → GitHub Wiki に反映 |

手動で同期する場合: `python scripts/sync_github_wiki.py`

---

## 7. 発見性向上のコツ

1. **README の先頭**に明確な価値提案 (Done)
2. **Badges** で技術スタックを明示 (Done)
3. **CITATION.cff** で論文引用を容易に (Done)
4. **CONTRIBUTING.md** で貢献を歓迎 (Done)
5. **定期的なコミット**でアクティブさをアピール
6. **関連プロジェクト** (OGW, PyG, Abaqus) の README で言及してもらう
7. **Wiki 更新後**は `./scripts/wiki_push.sh` で確実に push
