# Quick Start

Get Glazing running in minutes. This guide assumes you have Python 3.13+ and pip installed.

## Installation and Setup

```bash
pip install glazing
glazing init  # Downloads ~54MB, creates ~130MB of data
```

The `init` command downloads all four datasets and converts them to an efficient format. This can take a few minutes but only needs to be done once.

## Command Line

Search across all datasets:

```bash
glazing search query "give"
glazing search query "give" --dataset verbnet  # Limit to one dataset
```

Find cross-references between datasets:

```bash
glazing search cross-ref --source propbank --id "give.01" --target verbnet
```

## Python API

```python
from glazing.search import UnifiedSearch

# Search all datasets
search = UnifiedSearch()
results = search.search("abandon")

for result in results[:5]:
    print(f"{result.dataset}: {result.name} - {result.description}")
```

Load specific datasets:

```python
from glazing.verbnet.loader import VerbNetLoader

loader = VerbNetLoader()
verb_classes = list(loader.classes.values())

# Find a specific class
give_class = next((vc for vc in verb_classes if vc.id == "give-13.1"), None)
if give_class:
    print(f"Members: {[m.name for m in give_class.members]}")
    print(f"Roles: {[tr.role_type for tr in give_class.themroles]}")
```

Work with WordNet synsets:

```python
from glazing.wordnet.loader import WordNetLoader

loader = WordNetLoader()
synsets = list(loader.synsets.values())

# Find synsets for "dog"
dog_synsets = [s for s in synsets if any(l.lemma == "dog" for l in s.lemmas)]
for synset in dog_synsets[:3]:
    print(f"{synset.id}: {synset.definition}")
```

Extract cross-references:

```python
from glazing.references.index import CrossReferenceIndex

# Automatic extraction and caching
xref = CrossReferenceIndex()

# Resolve references
refs = xref.resolve("give.01", source="propbank")
print(f"VerbNet classes: {refs['verbnet_classes']}")
print(f"Confidence scores: {refs['confidence_scores']}")

# Find data with variations or inconsistencies
refs = xref.resolve("realize.01", source="propbank", fuzzy=True)
print(f"VerbNet classes: {refs['verbnet_classes']}")
```

## Next Steps

- [CLI Documentation](user-guide/cli.md) for command-line options
- [Python API Guide](user-guide/python-api.md) for programming details
- [Cross-References](user-guide/cross-references.md) for connecting datasets
- [API Reference](api/index.md) for complete documentation
