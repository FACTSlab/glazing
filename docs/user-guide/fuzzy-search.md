# Fuzzy Search

Fuzzy search uses Levenshtein distance to find data that contains typos, morphological variants, or spelling inconsistencies.

## When to Use Fuzzy Search

Fuzzy search is useful when the underlying datasets contain typos, morphological variants (e.g., "realise" vs "realize"), spelling inconsistencies, or partial forms. These datasets are compiled from various sources over time and may contain inconsistencies that would otherwise prevent exact matches.

## Implementation

The system calculates Levenshtein distance between strings, measuring the minimum number of single-character edits (insertions, deletions, or substitutions) needed to transform one string into another. Text is normalized by removing accents and punctuation before comparison. Similarity scores range from 0.0 (no match) to 1.0 (exact match).

## CLI Usage

### Basic Fuzzy Search

```bash
# Enable fuzzy matching with default threshold (0.8)
glazing search query "realize" --fuzzy

# Custom threshold (lower = more permissive)
glazing search query "organize" --fuzzy --threshold 0.7

# Combine with dataset filter
glazing search query "analyze" --fuzzy --dataset propbank
```

### Cross-Reference Resolution

```bash
# Fuzzy match cross-references
glazing xref resolve "realize.01" --source propbank --fuzzy

# With custom threshold
glazing xref resolve "organize.01" --source propbank --fuzzy --threshold 0.8
```

## Python API

### Basic Usage

```python
from glazing.search import UnifiedSearch

search = UnifiedSearch()

# Fuzzy search with default threshold
results = search.search_with_fuzzy("realize")

# Custom threshold
results = search.search_with_fuzzy("organize", fuzzy_threshold=0.7)

# Check match scores
for result in results[:5]:
    print(f"{result.name}: {result.score:.2f}")
```

### Cross-Reference Resolution

```python
from glazing.references.index import CrossReferenceIndex

xref = CrossReferenceIndex()

# Resolve with fuzzy matching
refs = xref.resolve("realize.01", source="propbank", fuzzy=True)

# With confidence threshold
refs = xref.resolve(
    "organize.01",
    source="propbank",
    fuzzy=True,
    confidence_threshold=0.8
)
```

### Direct Fuzzy Matching

```python
from glazing.utils.fuzzy_match import fuzzy_match, find_best_match

# Find multiple matches
candidates = ["realize", "realise", "recognition"]
results = fuzzy_match("realise", candidates, threshold=0.8)

# Find single best match
best = find_best_match("organize", ["organize", "organise", "organisation"])
```

## Threshold Selection

| Threshold | Use Case | Example Matches |
|-----------|----------|-----------------|
| 0.9-1.0 | Near-exact matches | "organize" → "organise" |
| 0.8-0.9 | Minor variations | "realize" → "realise" |
| 0.7-0.8 | Multiple variations | "color" → "colour" |
| 0.6-0.7 | Significant differences | "analyze" → "analyse" |
| Below 0.6 | Very loose matching | "recognise" → "recognize" |

## Text Normalization

Text undergoes automatic normalization before matching: accents are removed (café → cafe), case is normalized to lowercase (Give → give), punctuation is removed (give.01 → give 01), and whitespace is normalized ("give  to" → "give to").

## Examples

### Finding Data with Spelling Variants

```python
search.search_with_fuzzy("realize")  # Finds "realise" if dataset contains this variant
search.search_with_fuzzy("organize") # Finds "organise" if dataset contains this variant
search.search_with_fuzzy("analyze")  # Finds "analyse" if dataset contains this variant
```

### Partial Forms

Short forms or abbreviations in data require lower thresholds:

```python
search.search_with_fuzzy("recognise", fuzzy_threshold=0.7)  # Finds "recognize"
search.search_with_fuzzy("colour", fuzzy_threshold=0.7)     # Finds "color"
search.search_with_fuzzy("favour", fuzzy_threshold=0.8)     # Finds "favor"
```

### Morphological Variants

The system finds British and American spelling differences in the data:

```python
search.search_with_fuzzy("realise")   # Finds "realize" if present
search.search_with_fuzzy("colour")    # Finds "color" if present
search.search_with_fuzzy("analyse")   # Finds "analyze" if present
```

## Performance

Fuzzy matching is computationally more expensive than exact matching. Lower thresholds increase search time as more candidates must be evaluated. Results are cached for repeated queries. For optimal performance, attempt exact matching first and use fuzzy matching as a fallback.

## Usage Recommendations

Start with higher thresholds (0.8+) and decrease if needed. Check confidence scores in results to evaluate match quality. Combine fuzzy matching with other filters to reduce false positives. Use exact matching when possible for better performance.

## Batch Processing

```python
from glazing.search import UnifiedSearch

search = UnifiedSearch()

# Process multiple queries
queries = ["realize", "organize", "analyze"]
for query in queries:
    results = search.search_with_fuzzy(query, fuzzy_threshold=0.8)
    if results:
        print(f"{query} → {results[0].name}")
```

## Troubleshooting

- **Too Many Results**: Increase the threshold (e.g., 0.8 → 0.9), add dataset filters, or use more specific queries.
- **No Results Found**: Decrease the threshold (e.g., 0.8 → 0.6), verify text normalization is working correctly, or try partial query terms.
- **Unexpected Matches**: Review the normalization rules, adjust the threshold, and check similarity scores for match quality.
