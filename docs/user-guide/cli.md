# CLI Usage

The `glazing` command provides access to all functionality from the terminal.

## Quick Reference

```bash
glazing init                    # Download and set up all datasets
glazing search query "give"     # Search across datasets
glazing search entity give.01   # Look up specific entity
glazing download list           # Show available datasets
```

## Initialization

The first step is always `glazing init`, which downloads and prepares all datasets. You can specify a custom location:

```bash
glazing init --data-dir /my/data/path
```

Or use an environment variable:

```bash
export GLAZING_DATA_DIR=/my/data/path
glazing init
```

## Searching

The search command is the main way to explore the data. Search by text query:

```bash
glazing search query "abandon"
glazing search query "run" --dataset verbnet
glazing search query "give" --limit 10 --json
```

### Fuzzy Search

Use fuzzy matching to find data with typos, morphological variants, or spelling inconsistencies:

```bash
# Find data with variations
glazing search query "realize" --fuzzy
glazing search query "organize" --fuzzy --threshold 0.8

# Adjust the threshold (0.0-1.0, higher is stricter)
glazing search query "analyze" --fuzzy --threshold 0.85
```

### Syntactic Pattern Search

Search for syntactic patterns across datasets with hierarchical matching:

```bash
# Find all patterns with NP V PP structure
glazing search syntax "NP V PP"

# Find patterns with specific PP semantic roles
glazing search syntax "NP V PP.instrument"

# Use wildcards to match any following elements
glazing search syntax "NP V NP *"

# Search in specific dataset
glazing search syntax "NP V PP" --dataset verbnet
```

The syntax search supports hierarchical matching where general patterns like "NP V PP" will match more specific patterns like "NP V PP.instrument" or "NP V PP.goal" with full confidence.

### Entity Lookup

Look up specific entities by their IDs:

```bash
glazing search entity give-13.1 --dataset verbnet
glazing search entity 01772306 --dataset wordnet
glazing search entity give.01 --dataset propbank
```

## Cross-Reference Resolution

The xref commands provide cross-dataset reference resolution:

### Extract Cross-References

Build the cross-reference index (required before resolving):

```bash
# Extract all cross-references
glazing xref extract

# Extract with progress indicator
glazing xref extract --progress

# Force rebuild of the index
glazing xref extract --force

# Use custom cache directory
glazing xref extract --cache-dir /path/to/cache
```

### Resolve Cross-References

Find mappings between datasets:

```bash
# Basic resolution
glazing xref resolve "give.01" --source propbank
glazing xref resolve "give-13.1" --source verbnet

# Find data with variations or inconsistencies
glazing xref resolve "realize.01" --source propbank --fuzzy
glazing xref resolve "organize-74" --source verbnet --fuzzy --threshold 0.8

# Get JSON output
glazing xref resolve "Giving" --source framenet --json
```

### Clear Cache

Remove cached cross-reference data:

```bash
# Clear with confirmation prompt
glazing xref clear-cache

# Clear without confirmation
glazing xref clear-cache --yes

# Clear specific cache directory
glazing xref clear-cache --cache-dir /path/to/cache
```

## Downloading and Converting

If you need to work with individual datasets or update them:

```bash
glazing download dataset --dataset verbnet
glazing convert dataset --dataset verbnet --input-dir raw --output-dir converted
```

To see what's available:

```bash
glazing download list
glazing download info verbnet
```

## Output Formats

By default, output is formatted for human reading. Add `--json` for programmatic use:

```bash
glazing search query "give" --json | jq '.results[0]'
```

## Troubleshooting

If searches return no results, check that initialization completed:

```bash
ls ~/.local/share/glazing/converted/
```

For permission issues, use a different directory:

```bash
glazing init --data-dir ~/my-data
export GLAZING_DATA_DIR=~/my-data
```

Memory issues with large datasets can be avoided by processing them individually rather than using `--dataset all`.

## Environment Variables

- `GLAZING_DATA_DIR`: Where to store/find data (default: `~/.local/share/glazing`)
- `GLAZING_SKIP_INIT_CHECK`: Skip the initialization warning when importing the package
- `XDG_DATA_HOME`: Alternative base directory on Linux/macOS
