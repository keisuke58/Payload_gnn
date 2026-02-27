# Contributing to Payload_gnn

Thank you for your interest in contributing! This document provides guidelines for contributing to the GNN-SHM project.

## Code of Conduct

By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include: Python version, PyTorch/PyG version, error traceback, steps to reproduce

### Suggesting Features

- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Check the [Wiki Roadmap](wiki_repo/Roadmap.md) and [GitHub Issues](https://github.com/keisuke58/Payload_gnn/issues) for existing plans

### Pull Requests

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Follow the code style (Python: PEP 8, type hints where helpful)
4. Add tests if applicable
5. Update the Wiki if you change behavior
6. Submit a PR with a clear description

### Areas We Welcome Contributions

- **GNN models**: Graph Mamba, Equivariant GNN, Point Transformer integration
- **Data augmentation**: Noise, temperature perturbation
- **Documentation**: Wiki improvements, English translations
- **Benchmarks**: OGW dataset integration, Sim-to-Real experiments

## Development Setup

```bash
pip install -r requirements.txt
# Optional: pre-commit for linting
pip install pre-commit && pre-commit install
```

## Questions?

Open an [Issue](https://github.com/keisuke58/Payload_gnn/issues) or refer to the [Wiki](wiki_repo/Home.md).
