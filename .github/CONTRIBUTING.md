# Contributing to ClusterTK

Thank you for considering contributing to ClusterTK! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/clustertk.git
   cd clustertk
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e .[dev,viz,extras]
   ```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=clustertk --cov-report=term-missing

# Run specific test file
pytest tests/test_clustering/test_kmeans.py -v
```

## Code Style

We use automated tools to maintain code quality:

### Format code with black:
```bash
black clustertk/ tests/
```

### Sort imports with isort:
```bash
isort clustertk/ tests/
```

### Lint with flake8:
```bash
flake8 clustertk/ tests/
```

### Type check with mypy:
```bash
mypy clustertk/ --ignore-missing-imports
```

### Run all checks at once:
```bash
# Format
black clustertk/ tests/
isort clustertk/ tests/

# Lint
flake8 clustertk/ tests/
mypy clustertk/ --ignore-missing-imports

# Test
pytest tests/ -v --cov=clustertk
```

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clear, concise code
   - Add docstrings (NumPy/Google style)
   - Include type hints
   - Follow existing code style

3. **Add tests:**
   - Write tests for new functionality
   - Ensure all tests pass
   - Aim for >80% coverage for new code

4. **Update documentation:**
   - Update relevant `.md` files in `docs/`
   - Add examples if applicable
   - Update `CHANGELOG.md` (Unreleased section)

5. **Run all checks:**
   ```bash
   black clustertk/ tests/
   isort clustertk/ tests/
   flake8 clustertk/ tests/
   pytest tests/ -v --cov=clustertk
   ```

6. **Commit and push:**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request:**
   - Go to GitHub and create a PR
   - Fill in the PR template
   - Link any related issues
   - Wait for CI checks to pass
   - Respond to review comments

## CI/CD Pipeline

Our GitHub Actions workflows automatically:

✅ **Tests** - Run on all PRs and pushes
- Test on Ubuntu, macOS, Windows
- Test Python 3.8, 3.9, 3.10, 3.11
- Generate coverage reports

✅ **Lint** - Check code style on all PRs
- Black formatting
- isort import sorting
- flake8 linting
- mypy type checking

✅ **Publish** - Auto-publish to PyPI on releases
- Triggered by creating a GitHub release
- Validates version matches tag
- Builds and publishes to PyPI

## Code Guidelines

### Docstrings
Use NumPy/Google style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief one-line description.

    Longer description if needed. Explain what the function
    does, not how it does it.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> my_function(42, "test")
    True
    """
    return True
```

### Type Hints
Always use type hints:

```python
from typing import Optional, List, Dict, Union

def process_data(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    config: Optional[Dict[str, Union[int, str]]] = None
) -> pd.DataFrame:
    ...
```

### Testing
Write comprehensive tests:

```python
def test_my_feature():
    """Test my feature with various inputs."""
    # Arrange
    input_data = ...

    # Act
    result = my_function(input_data)

    # Assert
    assert result == expected_result
    assert isinstance(result, ExpectedType)
```

## Release Process

(For maintainers only)

1. Update version in `setup.py`, `pyproject.toml`, `clustertk/__init__.py`
2. Update `CHANGELOG.md`
3. Commit: `git commit -m "Bump version to X.Y.Z"`
4. Create tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. Push: `git push && git push --tags`
6. Create GitHub Release - CI will auto-publish to PyPI

## Questions?

- **Issues**: https://github.com/alexeiveselov92/clustertk/issues
- **Discussions**: https://github.com/alexeiveselov92/clustertk/discussions
- **Email**: alexei.veselov92@gmail.com

Thank you for contributing! 🎉
