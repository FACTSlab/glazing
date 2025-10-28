# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2025-10-28

### Fixed

- **FrameNet lexical units now properly loaded during conversion**
  - Lexical units are now parsed from `luIndex.xml` during frame conversion
  - All frames now include their associated lexical units with complete metadata
  - Fixes critical data completeness issue where `frame.lexical_units` was always empty
  - Enables querying frames by lexical unit name via the frame index
  - Approximately 13,500 lexical units now correctly associated with their frames

## [0.2.0] - 2025-09-30

### Added

#### Symbol Parsing System
- **Symbol parsers** for all four linguistic resources (FrameNet, PropBank, VerbNet, WordNet)
- **Structured symbol extraction** for parsing and normalizing entity identifiers
- **Type-safe parsed symbol representations** using TypedDict patterns
- **Symbol parser documentation** - Complete API documentation for all symbol parser modules
- **Symbol parser caching** - LRU cache decorators on all parsing functions for better performance
- Support for parsing complex symbols like ARG1-PPT, ?Theme_i, Core[Agent]

#### Fuzzy Search and Matching
- **Fuzzy search capability** with Levenshtein distance-based matching to find data with typos, morphological variants, and spelling inconsistencies
- **Configurable similarity thresholds** for controlling match precision
- **Multi-field fuzzy matching** across names, descriptions, and identifiers
- **Search result ranking** - New ranking module for scoring search results by match type and field relevance
- **Batch search methods** - `batch_by_lemma` method in UnifiedSearch for processing multiple queries
- `--fuzzy` flag in CLI commands with `--threshold` parameter
- `search_with_fuzzy()` method in UnifiedSearch and dataset-specific search classes

#### Syntax-Based Search
- **Unified syntax patterns** for searching by syntactic structure
- **Hierarchical pattern matching** where general patterns match specific subtypes
- **Syntax parser** for converting string patterns to unified format
- **Support for wildcards** and optional elements in patterns
- New CLI command: `glazing search syntax`
- `search_by_syntax()` method in UnifiedSearch class

#### Cross-Reference Enhancements
- **Automatic cross-reference extraction** on first use with progress indicators
- **Fuzzy resolution** for cross-references with typo tolerance
- **Confidence scoring** for mapping quality (0.0 to 1.0 scale)
- **Transitive mapping support** for indirect relationships
- **Reverse lookup capabilities** for bidirectional navigation
- New CLI commands: `glazing xref resolve`, `glazing xref extract`, `glazing xref clear-cache`

#### Structured Role/Argument Search
- **Property-based role search** for VerbNet thematic roles (optional, required, etc.)
- **Argument type filtering** for PropBank arguments (ARGM-LOC, ARGM-TMP, etc.)
- **Frame element search** by core type in FrameNet
- Support for complex queries with multiple property filters

#### Docker Support
- **Dockerfile** for containerized usage without local installation
- Full CLI exposed through Docker container
- Volume support for persistent data storage
- Docker Compose configuration example
- Interactive Python session support via container

#### CLI Improvements
- `--json` output mode for all search and xref commands
- `--progress` flag for long-running operations
- `--force` flag for cache clearing and re-extraction
- Better error messages with actionable suggestions
- Support for batch operations

### Changed

#### Type System Improvements
- Expanded `ArgumentNumber` type to include all modifier patterns (M-LOC, M-TMP, etc.)
- Added "C" and "R" prefixes to `FunctionTag` for continuation/reference support
- Stricter validation for `ThematicRoleType` with proper indexed variants
- More precise TypedDict definitions for parsed symbols

#### API Refinements
- `CrossReferenceIndex` now supports fuzzy matching in `resolve()` method
- `UnifiedSearch` class (renamed from `Search` for clarity)
- Consistent `None` returns for missing values (not empty strings or -1)
- Better separation of concerns between extraction, mapping, and resolution

### Fixed

- **CacheBase abstract methods** now have default implementations instead of NotImplementedError
- **VerbNet class ID generation** now uses deterministic pattern-based generation instead of hash-based fallback
- **Backward compatibility code removed** from PropBank symbol parser - no longer checks for argnum attribute
- **Legacy MappingSource removed** - "legacy" value no longer accepted in types
- **Documentation language** - removed promotional terms from fuzzy-match.md
- **Test compatibility** - Fixed PropBank symbol parser tests to work without backward compatibility
- PropBank `ArgumentNumber` type corrected to match actual data (removed invalid values like "7", "M-ADJ")
- ARGA argument in PropBank now correctly handled with proper arg_number value
- VerbNet member `verbnet_key` validation fixed to require proper format (e.g., "give#1")
- ThematicRole validation properly handles indexed role types (Patient_i, Theme_j)
- Import paths corrected for UnifiedSearch class
- Modifier type extraction returns `None` for non-modifiers consistently
- Frame element parsing handles abbreviations correctly
- Test fixtures updated to use correct data models and validation rules

### Technical Improvements

- Full mypy strict mode compliance across all modules
- Comprehensive test coverage for new symbol parsing features
- Performance optimizations for fuzzy matching with large datasets
- Better memory management for cross-reference extraction
- Caching improvements for repeated fuzzy searches

## [0.1.1] - 2025-09-27

### Fixed

- CLI search commands now work without requiring `--data-dir` argument
- Commands use the same default path as `glazing init` (`~/.local/share/glazing/converted/`)
- Updated help text to show default directory path

### Changed

- Improved documentation clarity and conciseness

## [0.1.0] - 2025-09-23

Initial release of `glazing`, a package containing unified data models and interfaces for syntactic and semantic frame ontologies.

### Added

#### Core Features
- **Unified data models** for all four linguistic resources using Pydantic v2
- **One-command initialization** with `glazing init` to download and convert all datasets
- **JSON Lines format** for efficient storage and streaming of large datasets
- **Type-safe interfaces** with comprehensive type hints using Python 3.13+ conventions
- **Cross-reference resolution** between FrameNet, PropBank, VerbNet, and WordNet
- **Memory-efficient streaming** support for processing large datasets

#### Command-Line Interface
- `glazing init` - Initialize all datasets with a single command
- `glazing download` - Download individual or all datasets
- `glazing convert` - Convert from source formats to JSON Lines
- `glazing search` - Search across datasets with various filters
- Support for custom data directories via `--data-dir` or environment variables
- JSON output mode for programmatic use

#### Python API
- `UnifiedSearch` class for searching across all datasets
- Dataset-specific loaders: `FrameNetLoader`, `PropBankLoader`, `VerbNetLoader`, `WordNetLoader`
- `ReferenceExtractor` and `ReferenceResolver` for cross-dataset mappings
- Streaming iterators for memory-efficient processing
- Comprehensive search methods: by lemma, by query, by entity ID

#### Dataset Support
- **FrameNet 1.7**: Semantic frames, frame elements, lexical units, and frame relations
- **PropBank 3.4**: Framesets, rolesets, and semantic role labels
- **VerbNet 3.4**: Verb classes, thematic roles, syntactic frames, and GL semantics
- **WordNet 3.1**: Synsets, lemmas, lexical relations, and morphological processing

#### Data Processing
- XML to JSON Lines converters for all datasets
- Automatic validation using Pydantic models
- Support for incremental and streaming conversion
- Efficient caching for repeated operations

#### Development Tools
- Comprehensive test suite with pytest
- Code quality checks with ruff and mypy
- Pre-commit hooks for consistent code style
- CI/CD pipeline with GitHub Actions
- Support for Python 3.13

#### Documentation
- Complete API documentation with mkdocstrings
- User guides for CLI and Python API usage
- Quick start tutorials
- Contributing guidelines
- Cross-reference usage examples

### Technical Details
- **Package size**: ~5MB (code only)
- **Dataset size**: ~54MB raw downloads, ~130MB total after conversion
- **Performance**: Streaming support for datasets up to several GB
- **Compatibility**: Python 3.13+, all major operating systems

### Dependencies
- `pydantic >= 2.5.0` (data validation)
- `typing-extensions >= 4.9.0` (extended type hints)
- `python-dateutil >= 2.8.2` (date/time parsing)
- `lxml >= 5.0.0` (XML parsing)
- `click >= 8.0.0` (CLI framework)
- `requests >= 2.25.0` (dataset downloading)
- `tqdm >= 4.60.0` (progress bars)
- `rich >= 13.0.0` (CLI formatting)

[Unreleased]: https://github.com/aaronstevenwhite/glazing/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/aaronstevenwhite/glazing/releases/tag/v0.2.1
[0.2.0]: https://github.com/aaronstevenwhite/glazing/releases/tag/v0.2.0
[0.1.1]: https://github.com/aaronstevenwhite/glazing/releases/tag/v0.1.1
[0.1.0]: https://github.com/aaronstevenwhite/glazing/releases/tag/v0.1.0
