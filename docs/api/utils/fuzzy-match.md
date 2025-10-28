# glazing.utils.fuzzy_match

Fuzzy string matching utilities using Levenshtein distance.

## Overview

The fuzzy_match module provides functions for fuzzy string matching using Levenshtein distance and other similarity metrics. It includes text normalization and caching for performance. The primary use case is finding data that contains typos, morphological variants, or spelling inconsistencies in the underlying datasets.

## Functions

### normalize_text

```python
def normalize_text(text: str, preserve_case: bool = False) -> str
```

Normalize text for fuzzy matching by removing accents, replacing separators with spaces, and normalizing whitespace.

**Parameters:**
- **text** (str): Text to normalize
- **preserve_case** (bool, default=False): Whether to preserve letter case

**Returns:**
- **str**: Normalized text

**Example:**
```python
>>> normalize_text("Hello-World_123")
'hello world 123'
>>> normalize_text("café")
'cafe'
```

### levenshtein_ratio

```python
def levenshtein_ratio(s1: str, s2: str, normalize: bool = True) -> float
```

Calculate Levenshtein ratio between two strings. The ratio is computed as: `1 - (distance / max(len(s1), len(s2)))`

**Parameters:**
- **s1** (str): First string
- **s2** (str): Second string
- **normalize** (bool, default=True): Whether to normalize strings before comparison

**Returns:**
- **float**: Similarity ratio between 0.0 and 1.0

**Example:**
```python
>>> levenshtein_ratio("hello", "helo")
0.89
>>> levenshtein_ratio("cat", "dog")
0.0
```

### fuzzy_match

```python
def fuzzy_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.8,
    max_results: int | None = None
) -> list[FuzzyMatchResult]
```

Find best fuzzy matches from candidates.

**Parameters:**
- **query** (str): Query string to match
- **candidates** (list[str]): List of candidate strings
- **threshold** (float, default=0.8): Minimum similarity score (0.0 to 1.0)
- **max_results** (int | None, default=None): Maximum number of results to return

**Returns:**
- **list[FuzzyMatchResult]**: Sorted list of matches above threshold

**Example:**
```python
>>> candidates = ["instrument", "argument", "document"]
>>> results = fuzzy_match("instsrument", candidates, threshold=0.7)
>>> results[0]["match"]
'instrument'
>>> results[0]["score"]
0.91
```

### find_best_match

```python
def find_best_match(query: str, candidates: list[str]) -> str | None
```

Find the single best match from candidates. First tries exact match, then fuzzy match with threshold 0.6.

**Parameters:**
- **query** (str): Query string to match
- **candidates** (list[str]): List of candidate strings

**Returns:**
- **str | None**: Best matching candidate or None if no good match

**Example:**
```python
>>> find_best_match("give", ["give", "take", "make"])
'give'
>>> find_best_match("giv", ["give", "take", "make"])
'give'
```

## Types

### FuzzyMatchResult

```python
class FuzzyMatchResult(TypedDict):
    match: str           # The matched string
    score: float         # Similarity score (0.0 to 1.0)
    normalized_query: str  # Normalized form of the query
    normalized_match: str  # Normalized form of the match
```

## Performance Notes

- Functions use `@lru_cache` decorator for caching results
- `normalize_text` has cache size of 1024 entries
- `levenshtein_ratio` has cache size of 4096 entries
- Cache significantly improves performance for repeated comparisons

## Dependencies

Requires `python-Levenshtein>=0.20.0` for Levenshtein distance calculations.

::: glazing.utils.fuzzy_match
    options:
      show_source: false
