# Agent Instructions — Payload GNN-SHM

JAXA H3 CFRP fairing debonding detection via FEM + PyTorch Geometric GNNs.

## Start here

| Task | Command |
|------|---------|
| Light sanity check | `./scripts/reproduce_core.sh` |
| Train GNN | `cd src && python train.py --arch gat --data_dir ../dataset/processed --epochs 200` |
| H3 spec validation | `python scripts/validate_h3_specs.py` |
| Tests | `pytest tests/ -q` |
| Compare runs | `python scripts/compare_model_results.py` |

## Rules

1. **Training scripts run from `src/`** — imports use bare `from models import ...`
2. **Don't commit** `runs/`, `dataset_output*/`, `*.odb`, large `data/`
3. **Abaqus/FEM batch jobs** are heavy — don't run in light reproduce
4. **Wiki sync** — `.github/workflows/sync-wiki.yml` handles `wiki_repo/`

## Key paths

| Path | Purpose |
|------|---------|
| `src/train.py` | Main GNN training entry |
| `src/models.py` | GCN/GAT/GIN/SAGE builders |
| `src/build_graph.py` | Curvature-aware graph construction |
| `scripts/compare_model_results.py` | Cross-run metrics + CSV export |
| `wiki_repo/Home.md` | Full documentation index |

## Dependencies

```bash
pip install -r requirements-gnn.txt
pip install -r requirements-quantum.txt  # optional
pip install -r requirements-fem-alt.txt  # optional FEniCS/JAX-FEM
```

See [CLAUDE.md](CLAUDE.md) for full project instructions.
