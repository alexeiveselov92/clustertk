# Installation

ClusterTK supports Python 3.8 and above on Linux, macOS, and Windows.

## Basic Installation

Install ClusterTK with core functionality (no visualization dependencies):

```bash
pip install clustertk
```

This installs the minimal set of dependencies:
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- joblib >= 1.0.0

## Installation with Visualization

To use visualization features (`plot_*` methods), install with viz extras:

```bash
pip install clustertk[viz]
```

Additional dependencies:
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Installation with Extended Features

For advanced dimensionality reduction (UMAP) and clustering (HDBSCAN):

```bash
pip install clustertk[extras]
```

Additional dependencies:
- umap-learn >= 0.5.0
- hdbscan >= 0.8.0

## Full Installation

Install all optional dependencies:

```bash
pip install clustertk[all]
```

## Development Installation

For contributing to ClusterTK:

```bash
git clone https://github.com/alexeiveselov92/clustertk.git
cd clustertk
pip install -e .[dev]
```

Development dependencies include:
- pytest >= 7.0.0
- pytest-cov >= 3.0.0
- black >= 22.0.0
- flake8 >= 4.0.0
- mypy >= 0.950
- sphinx >= 4.0.0
- sphinx-rtd-theme >= 1.0.0

## Verify Installation

```python
import clustertk
print(clustertk.__version__)
```

## Requirements

### System Requirements
- **Python:** 3.8, 3.9, 3.10, or 3.11
- **Operating System:** Linux, macOS, Windows
- **Memory:** Minimum 2GB RAM (8GB+ recommended for large datasets)

### Python Dependencies

**Core (always installed):**
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- joblib >= 1.0.0

**Optional (visualization):**
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

**Optional (extended features):**
- umap-learn >= 0.5.0
- hdbscan >= 0.8.0

## Troubleshooting

### ImportError: No module named 'clustertk'

Make sure ClusterTK is installed:
```bash
pip install clustertk
```

### Visualization features not available

Install visualization dependencies:
```bash
pip install clustertk[viz]
```

### UMAP or HDBSCAN not available

Install extended features:
```bash
pip install clustertk[extras]
```

### Permission denied errors

Use `--user` flag:
```bash
pip install --user clustertk
```

Or use a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install clustertk
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade clustertk
```

To upgrade with all extras:

```bash
pip install --upgrade clustertk[all]
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [User Guide](user_guide/README.md)
- [API Reference](api_reference.md)
