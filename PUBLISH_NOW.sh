#!/bin/bash
# Quick publish script for v0.8.0
# Usage: ./PUBLISH_NOW.sh

set -e

echo "==================================="
echo "ClusterTK v0.8.0 - Quick Publish"
echo "==================================="
echo ""

# Check if in correct directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: Run this script from the clustertk root directory"
    exit 1
fi

# Check if dist files exist
if [ ! -f "dist/clustertk-0.8.0-py3-none-any.whl" ]; then
    echo "❌ Error: Distribution files not found. Run: python3 -m build"
    exit 1
fi

echo "✅ Distribution files found"
echo ""

# Verify packages
echo "📦 Verifying packages..."
python3 -m twine check dist/clustertk-0.8.0* || {
    echo "❌ Package verification failed"
    exit 1
}
echo "✅ Packages verified"
echo ""

# Check for credentials
if [ -z "$TWINE_PASSWORD" ]; then
    echo "⚠️  TWINE_PASSWORD not set"
    echo ""
    echo "Set your PyPI API token:"
    echo "  export TWINE_USERNAME='__token__'"
    echo "  export TWINE_PASSWORD='pypi-YOUR_TOKEN_HERE'"
    echo ""
    echo "Or create ~/.pypirc file (see PUBLISH_v0.8.0.md)"
    echo ""
    exit 1
fi

echo "✅ Credentials found"
echo ""

# Confirm upload
echo "📤 Ready to upload to PyPI"
echo ""
echo "Files to upload:"
echo "  - clustertk-0.8.0-py3-none-any.whl (80K)"
echo "  - clustertk-0.8.0.tar.gz (108K)"
echo ""
read -p "Continue with upload? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Upload cancelled"
    exit 1
fi

# Upload to PyPI
echo ""
echo "🚀 Uploading to PyPI..."
python3 -m twine upload dist/clustertk-0.8.0* || {
    echo "❌ Upload failed"
    exit 1
}

echo ""
echo "✅ Successfully published to PyPI!"
echo ""
echo "Next steps:"
echo "  1. Visit: https://pypi.org/project/clustertk/"
echo "  2. Wait 1-2 minutes for PyPI to process"
echo "  3. Test: pip install --upgrade clustertk"
echo "  4. Verify: python3 -c 'import clustertk; print(clustertk.__version__)'"
echo "  5. Create GitHub Release: https://github.com/alexeiveselov92/clustertk/releases/new"
echo ""
echo "🎉 v0.8.0 is live!"
