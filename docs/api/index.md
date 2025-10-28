# API Reference

This section contains the complete API documentation for Glazing, automatically generated from the source code.

## Package Organization

The Glazing package is organized into several submodules:

### Core Modules

- **[glazing](glazing.md)** - Main package initialization and version info
- **[glazing.base](base.md)** - Base models and shared functionality
- **[glazing.types](types.md)** - Type definitions and enums
- **[glazing.search](search.md)** - Unified search interface
- **[glazing.downloader](downloader.md)** - Dataset downloading utilities
- **[glazing.initialize](initialize.md)** - Initialization and setup functions

### Dataset Modules

- **[FrameNet](framenet/index.md)** - Semantic frames and frame elements
- **[PropBank](propbank/index.md)** - Predicate-argument structures
- **[VerbNet](verbnet/index.md)** - Verb classes and thematic roles
- **[WordNet](wordnet/index.md)** - Synsets and lexical relations

### Cross-Reference Module

- **[References](references/index.md)** - Cross-dataset reference resolution

### Utility Modules

- **[Utils](utils/index.md)** - Shared utilities and helpers
- **[CLI](cli/index.md)** - Command-line interface implementation

## Quick Navigation

### By Functionality

#### Data Loading
- [FrameNetLoader](framenet/loader.md#glazing.framenet.loader.FrameNetLoader)
- [PropBankLoader](propbank/loader.md#glazing.propbank.loader.PropBankLoader)
- [VerbNetLoader](verbnet/loader.md#glazing.verbnet.loader.VerbNetLoader)
- [WordNetLoader](wordnet/loader.md#glazing.wordnet.loader.WordNetLoader)

#### Data Conversion
- [FrameNet Converter](framenet/converter.md)
- [PropBank Converter](propbank/converter.md)
- [VerbNet Converter](verbnet/converter.md)
- [WordNet Converter](wordnet/converter.md)

#### Searching
- [UnifiedSearch](search.md#glazing.search.UnifiedSearch)
- [FrameNetSearch](framenet/search.md#glazing.framenet.search.FrameNetSearch)
- [PropBankSearch](propbank/search.md#glazing.propbank.search.PropBankSearch)
- [VerbNetSearch](verbnet/search.md#glazing.verbnet.search.VerbNetSearch)
- [WordNetSearch](wordnet/search.md#glazing.wordnet.search.WordNetSearch)

#### Cross-References
- [ReferenceExtractor](references/extractor.md#glazing.references.extractor.ReferenceExtractor)
- [ReferenceResolver](references/resolver.md#glazing.references.resolver.ReferenceResolver)

## Usage Examples

### Basic Import

```python
from glazing.search import UnifiedSearch
from glazing.framenet.loader import FrameNetLoader
from glazing.references.resolver import ReferenceResolver
```

### Loading Data

```python
from pathlib import Path
from glazing.verbnet.loader import VerbNetLoader

loader = VerbNetLoader(Path("data/verbnet.jsonl"))
verb_classes = loader.load_verb_classes()
```

### Searching

```python
from glazing.search import UnifiedSearch

search = UnifiedSearch()
results = search.search("abandon")
```

## Type Safety

All models use Pydantic v2 for validation and provide complete type hints. This ensures:

- Runtime validation of data
- IDE autocomplete support
- Static type checking with mypy
- Automatic documentation generation

## Performance Considerations

- **Streaming**: All loaders support streaming for memory-efficient processing
- **Caching**: Search indices are cached for repeated queries
- **Lazy Loading**: Data is loaded on-demand where possible
- **JSON Lines**: Efficient line-based format for large datasets

## Error Handling

All modules follow consistent error handling patterns:

```python
try:
    frames = loader.load_frames(path)
except FileNotFoundError:
    print("Data file not found")
except ValidationError as e:
    print(f"Invalid data: {e}")
```

## Version Compatibility

This documentation covers Glazing version 0.2.1. Check your installed version:

```python
import glazing
print(glazing.__version__)
```
