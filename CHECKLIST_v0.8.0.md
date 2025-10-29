# v0.8.0 Publication Checklist

## ✅ Development (100% Complete)

- [x] HDBSCAN algorithm implemented
- [x] Auto parameter tuning (min_cluster_size, min_samples)
- [x] Pipeline integration
- [x] compare_algorithms() integration
- [x] Test suite created (39 tests)
- [x] pytest infrastructure setup
- [x] All tests passing
- [x] Documentation written
- [x] Code committed to git
- [x] Code pushed to GitHub

## ✅ Version & Build (100% Complete)

- [x] Version updated in setup.py (0.8.0)
- [x] Version updated in pyproject.toml (0.8.0)
- [x] Version updated in __init__.py (0.8.0)
- [x] Build packages created (`python3 -m build`)
- [x] Packages verified (`twine check`)
- [x] Local installation tested
- [x] Import test successful

## ✅ Git & GitHub (100% Complete)

- [x] All changes committed
- [x] All commits pushed to main
- [x] Git tag v0.8.0 created
- [x] Git tag pushed to GitHub
- [x] Release notes created (RELEASE_NOTES_v0.8.0.md)
- [x] CLAUDE.md updated
- [x] TODO.md updated

## ⏳ PyPI Publication (Pending - Manual Action)

- [ ] **Get PyPI API token** from https://pypi.org/manage/account/token/
- [ ] **Set environment variables:**
  ```bash
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="pypi-YOUR_TOKEN_HERE"
  ```
- [ ] **Run publish script:** `./PUBLISH_NOW.sh`
  - OR manually: `python3 -m twine upload dist/clustertk-0.8.0*`
- [ ] **Wait 1-2 minutes** for PyPI processing
- [ ] **Verify on PyPI:** https://pypi.org/project/clustertk/
- [ ] **Test installation:**
  ```bash
  pip install --upgrade clustertk
  python3 -c "import clustertk; print(clustertk.__version__)"
  ```

## ⏳ GitHub Release (Pending - Manual Action)

- [ ] **Go to:** https://github.com/alexeiveselov92/clustertk/releases/new
- [ ] **Select tag:** v0.8.0
- [ ] **Set title:** v0.8.0 - HDBSCAN Algorithm & Test Suite
- [ ] **Copy description** from RELEASE_NOTES_v0.8.0.md
- [ ] **Upload files:**
  - [ ] dist/clustertk-0.8.0-py3-none-any.whl
  - [ ] dist/clustertk-0.8.0.tar.gz
- [ ] **Publish release**

## ⏳ Post-Publication (After PyPI & GitHub Release)

- [ ] Verify PyPI page shows v0.8.0
- [ ] Test fresh installation: `pip install clustertk==0.8.0`
- [ ] Test HDBSCAN import: `from clustertk.clustering import HDBSCANClustering`
- [ ] Check GitHub Release page
- [ ] Update TODO.md to mark PyPI publication as complete
- [ ] Consider: Tweet/LinkedIn post about release (optional)
- [ ] Consider: Reddit post in r/Python or r/MachineLearning (optional)

## 📋 Quick Reference

**Files ready for upload:**
```
dist/clustertk-0.8.0-py3-none-any.whl  (80 KB)
dist/clustertk-0.8.0.tar.gz            (108 KB)
```

**PyPI Token URL:**
https://pypi.org/manage/account/token/

**GitHub Release URL:**
https://github.com/alexeiveselov92/clustertk/releases/new

**Publishing Commands:**
```bash
# Set credentials
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-YOUR_TOKEN_HERE"

# Quick publish
./PUBLISH_NOW.sh

# Or manual publish
python3 -m twine upload dist/clustertk-0.8.0*
```

**Verification Commands:**
```bash
# Check version
pip install --upgrade clustertk
python3 -c "import clustertk; print(clustertk.__version__)"

# Test HDBSCAN
python3 -c "from clustertk.clustering import HDBSCANClustering; print('OK!')"

# Run full test suite
pytest

# Check coverage
pytest --cov=clustertk --cov-report=html
```

## 🎯 Success Criteria

v0.8.0 is successfully published when:

✓ PyPI shows version 0.8.0 as latest
✓ `pip install clustertk` installs 0.8.0
✓ HDBSCAN can be imported successfully
✓ GitHub Release exists with v0.8.0 tag
✓ Release notes are visible on GitHub

---

**Current Status:** Development Complete ✅ | Awaiting Manual Publication ⏳

**Last Updated:** October 29, 2025
