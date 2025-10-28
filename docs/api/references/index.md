# glazing.references

Cross-dataset reference resolution with automatic extraction and fuzzy matching.

## Overview

The references module provides utilities for extracting and resolving cross-references between datasets. The new `CrossReferenceIndex` class provides an ergonomic API with automatic extraction, caching, and fuzzy matching support.

## Quick Start

```python
from glazing.references.index import CrossReferenceIndex

# Automatic extraction and caching
xref = CrossReferenceIndex()

# Resolve references
refs = xref.resolve("give.01", source="propbank")
print(refs["verbnet_classes"])  # ['give-13.1']

# Find data with variations or inconsistencies
refs = xref.resolve("realize.01", source="propbank", fuzzy=True)
```

## Main Classes

### CrossReferenceIndex

The primary interface for cross-reference operations:

```python
class CrossReferenceIndex(
    auto_extract: bool = True,
    cache_dir: Path | None = None,
    show_progress: bool = True
)
```

**Key Methods:**
- `resolve(entity_id, source, fuzzy=False)` - Resolve cross-references
- `find_mappings(source_id, source_dataset, target_dataset)` - Find direct mappings
- `extract_all()` - Manually trigger extraction
- `clear_cache()` - Clear cached references

## Modules

- **[Models](models.md)** - Reference data models
- **[Extractor](extractor.md)** - Lower-level extraction interface
- **[Resolver](resolver.md)** - Reference resolution logic
- **[Mapper](mapper.md)** - Mapping between dataset identifiers

## Features

- **Automatic Extraction**: References are extracted automatically on first use
- **Caching**: Extracted references are cached for fast subsequent loads
- **Fuzzy Matching**: Find data with typos, morphological variants, and spelling inconsistencies
- **Confidence Scores**: All mappings include confidence scores
- **Progress Indicators**: Visual feedback during extraction

::: glazing.references
    options:
      show_source: false
      members: false
