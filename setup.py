"""Setup configuration for clustertk package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Core dependencies
install_requires = [
    'numpy>=1.20.0',
    'pandas>=1.3.0',
    'scikit-learn>=1.0.0',
    'scipy>=1.7.0',
    'joblib>=1.0.0',  # For pipeline serialization
]

# Optional dependencies for visualization
viz_requires = [
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
]

# Optional dependencies for extended functionality
extras_requires = [
    'umap-learn>=0.5.0',
    'hdbscan>=0.8.0',
]

# Development dependencies
dev_requires = [
    'pytest>=7.0.0',
    'pytest-cov>=3.0.0',
    'black>=22.0.0',
    'flake8>=4.0.0',
    'mypy>=0.950',
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=1.0.0',
]

setup(
    name='clustertk',
    version='0.10.0',
    author='Aleksey Veselov',
    author_email='alexei.veselov92@gmail.com',
    description='A comprehensive toolkit for cluster analysis with full pipeline support',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alexeiveselov92/clustertk',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'docs']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'viz': viz_requires,
        'extras': extras_requires,
        'dev': dev_requires,
        'all': viz_requires + extras_requires + dev_requires,
    },
    keywords='clustering, machine-learning, data-analysis, pipeline, kmeans, pca, data-science',
    project_urls={
        'Bug Reports': 'https://github.com/alexeiveselov92/clustertk/issues',
        'Source': 'https://github.com/alexeiveselov92/clustertk',
    },
)
