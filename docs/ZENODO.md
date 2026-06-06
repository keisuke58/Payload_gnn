# Zenodo Archive

Software archive for citation and long-term preservation.

## DOI

| Type | DOI | Link |
|------|-----|------|
| **Software archive** | [10.5281/zenodo.20495444](https://doi.org/10.5281/zenodo.20495444) | https://zenodo.org/records/20495444 |

New GitHub releases receive a version-specific DOI under the same concept record when the [Zenodo–GitHub integration](https://zenodo.org/account/settings/github/) is enabled for `keisuke58/Payload_gnn`.

## Publish a new release on Zenodo

1. Create a GitHub release (tag + notes), e.g. `gh release create v0.2.0 …`
2. Open https://zenodo.org/account/settings/github/ and confirm the repo is ON
3. After a few minutes, check https://zenodo.org/deposit — a draft may appear
4. Review metadata (auto-filled from `.zenodo.json` + `CITATION.cff`) and click **Publish**
5. Update `CITATION.cff` `version` and add the new version DOI to `identifiers`

## Cite

```bibtex
@software{nishioka2026payload,
  author    = {Nishioka, Keisuke and Kojima, Yuta and Saito, Toshiya},
  title     = {GNN-SHM: Graph Neural Networks for H3 Rocket Fairing SHM},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20495444},
  url       = {https://github.com/keisuke58/Payload_gnn}
}
```

For the *Frontiers in Materials* perforated-CFRP paper, cite both the manuscript and this software archive.