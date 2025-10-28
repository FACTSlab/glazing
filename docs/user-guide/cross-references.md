# Cross-References

Glazing connects FrameNet, PropBank, VerbNet, and WordNet through their internal cross-references. While these connections exist in the original datasets, extracting and using them typically requires understanding each dataset's format. Glazing provides a unified interface to work with these references.

## How References Work

The four datasets reference each other in different ways. PropBank rolesets often specify their corresponding VerbNet classes. VerbNet members include WordNet sense keys. FrameNet lexical units sometimes reference WordNet synsets. These connections allow you to trace a concept across different linguistic representations.

For example, the PropBank roleset `give.01` maps to VerbNet class `give-13.1`, which contains members linked to WordNet senses like `give%2:40:00::`. This lets you connect PropBank's argument structure to VerbNet's thematic roles and WordNet's semantic hierarchy.

## Basic Usage

The simplest way to find cross-references is through the CLI:

```bash
# Find what VerbNet classes correspond to a PropBank roleset
glazing search cross-ref --source propbank --id "give.01" --target verbnet
```

In Python, use the new ergonomic CrossReferenceIndex API:

```python
from glazing.references.index import CrossReferenceIndex

# Automatic extraction on first use (cached for future runs)
xref = CrossReferenceIndex()

# Resolve references for a PropBank roleset
refs = xref.resolve("give.01", source="propbank")
print(f"VerbNet classes: {refs['verbnet_classes']}")
print(f"Confidence scores: {refs['confidence_scores']}")

# Find data with variations or inconsistencies
refs = xref.resolve("realize.01", source="propbank", fuzzy=True)
print(f"VerbNet classes: {refs['verbnet_classes']}")
```

## Working with References

The CrossReferenceIndex automatically handles extraction and caching. On first use, it scans all datasets and builds an index, which is cached for future runs. The index includes confidence scores based on the quality of mappings and fuzzy matching similarity.

When you resolve references for an item, you get back all related items across datasets with confidence scores. Not every item has cross-references to all other datasets. Some connections are direct (explicitly stated in the data) while others use fuzzy matching to find potential matches.

## Practical Examples

To find semantic equivalents across datasets, search each one and collect the results:

```python
from glazing.search import UnifiedSearch

search = UnifiedSearch()
results = search.by_lemma("give")

# Group results by dataset
by_dataset = {}
for result in results:
    by_dataset.setdefault(result.dataset, []).append(result)
```

For analyzing coverage of a concept across datasets:

```python
def check_coverage(lemma):
    search = UnifiedSearch()
    results = search.by_lemma(lemma)

    coverage = set(r.dataset for r in results)
    missing = {'propbank', 'verbnet', 'wordnet', 'framenet'} - coverage

    if missing:
        print(f"{lemma} not found in: {', '.join(missing)}")
    return coverage
```

## Additional Features

### Manual Control

If you prefer manual control over extraction:

```python
from glazing.references.index import CrossReferenceIndex

# Disable auto-extraction
xref = CrossReferenceIndex(auto_extract=False)

# Extract when ready
xref.extract_all()

# Clear cache if needed
xref.clear_cache()
```

### Fuzzy Matching

The system supports fuzzy matching for finding data with typos, morphological variants, and spelling inconsistencies:

```python
# Find data with variations
refs = xref.resolve("organize.01", source="propbank", fuzzy=True, threshold=0.8)

# The system will find variants if they exist and return their references
```

### Confidence Scores

All mappings include confidence scores based on:
- Original mapping confidence from the dataset
- Fuzzy matching similarity scores
- Mapping type (direct vs. inferred)

## Limitations

Cross-references in these datasets are incomplete and sometimes approximate. VerbNet members don't always have WordNet mappings. PropBank rolesets may lack VerbNet mappings. The quality and coverage of references varies between dataset pairs. The datasets themselves may contain typos or morphological variants, which fuzzy matching helps to address. Fuzzy matching can occasionally produce false positives at lower thresholds.
