# Glazing

[![PyPI version](https://img.shields.io/pypi/v/glazing)](https://pypi.org/project/glazing/)
[![Python versions](https://img.shields.io/pypi/pyversions/glazing)](https://pypi.org/project/glazing/)
[![License](https://img.shields.io/pypi/l/glazing)](https://github.com/aaronstevenwhite/glazing/blob/main/LICENSE)
[![CI](https://github.com/aaronstevenwhite/glazing/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aaronstevenwhite/glazing/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17185626.svg)](https://doi.org/10.5281/zenodo.17185626)

Unified data models and interfaces for syntactic and semantic frame ontologies.

## Overview

Glazing provides a unified, type-safe interface for working with FrameNet, PropBank, VerbNet, and WordNet. It simplifies the process of downloading, converting, and searching across these datasets with a consistent API.

## Key Features

- 🚀 **One-command initialization:** Download and convert all datasets with `glazing init`
- 📦 **Type-safe data models:** Using Pydantic v2 for validation and serialization
- 🔍 **Command-line interface:** Download, convert, and search datasets from the command line
- 🔗 **Cross-dataset references:** Find connections between different linguistic resources
- 🐍 **Python 3.13+:** Python 3.13+ with full type hints
- 📊 **Efficient storage:** JSON Lines format for fast loading and streaming

## Supported Datasets

### [FrameNet 1.7](https://framenet.icsi.berkeley.edu/)
Semantic frames and frame-to-frame relations representing schematic situations.

### [PropBank 3.4](https://propbank.github.io/)
Predicate-argument structures and semantic role labels for verbs.

### [VerbNet 3.4](https://verbs.colorado.edu/verbnet/)
Hierarchical verb classes with thematic roles and syntactic patterns.

### [WordNet 3.1](https://wordnet.princeton.edu/)
Lexical database of synonyms, hypernyms, and other semantic relations.

## Quick Start

```bash
# Install the package
pip install glazing

# Initialize all datasets (downloads ~54MB)
glazing init
```

After initialization, you can immediately start exploring the data:

```python
from glazing.search import UnifiedSearch

# Automatically uses default data directory after 'glazing init'
search = UnifiedSearch()

# Search across all datasets
results = search.search("give")

for result in results[:5]:
    print(f"{result.dataset}: {result.name} - {result.description}")
```

## Documentation

Start with [Installation](installation.md) for system requirements, then follow the [Quick Start](quick-start.md) to get running in minutes. The [User Guide](user-guide/cli.md) covers detailed usage, while the [API Reference](api/index.md) documents all classes and methods. See [Contributing](contributing.md) if you'd like to help improve the project.

## Why Glazing?

Working with linguistic resources traditionally requires understanding different data formats (XML, custom databases), writing custom parsers for each resource, managing cross-references manually, and dealing with inconsistent APIs. Glazing solves these problems by providing unified data models across all resources, automatic data conversion to efficient formats, built-in cross-reference resolution, and consistent search and access patterns.

## Project Status

Glazing is actively maintained and welcomes contributions. The project follows semantic versioning and includes extensive test coverage.

## Links

- [GitHub Repository](https://github.com/aaronstevenwhite/glazing)
- [PyPI Package](https://pypi.org/project/glazing/)
- [Issue Tracker](https://github.com/aaronstevenwhite/glazing/issues)
- [Changelog](https://github.com/aaronstevenwhite/glazing/blob/main/CHANGELOG.md)

## License

This package is licensed under an MIT License. See [LICENSE](https://github.com/aaronstevenwhite/glazing/blob/main/LICENSE) file for details.

## Citation

If you use Glazing in your research, please cite:

```bibtex
@software{glazing2025,
  author = {White, Aaron Steven},
  title = {Glazing: Unified Data Models and Interfaces for Syntactic and Semantic Frame Ontologies},
  year = {2025},
  url = {https://github.com/aaronstevenwhite/glazing},
  version = {0.2.1},
  doi = {10.5281/zenodo.17185626}
}
```

## Acknowledgments

This project was funded by a [National Science Foundation](https://www.nsf.gov/) ([BCS-2040831](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2040831)) and builds upon the foundational work of the FrameNet, PropBank, VerbNet, and WordNet teams.
