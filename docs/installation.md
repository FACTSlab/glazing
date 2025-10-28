# Installation

This guide covers the installation of Glazing and its dependencies.

## Requirements

- **Python 3.13 or higher**
- pip package manager
- ~130MB disk space for all datasets after conversion
- Internet connection for dataset downloads

## Install from PyPI

The simplest way to install Glazing is via pip:

```bash
pip install glazing
```

## Install from Source

To install the latest development version:

```bash
git clone https://github.com/aaronstevenwhite/glazing.git
cd glazing
pip install -e .
```

For development with all dependencies:

```bash
pip install -e ".[dev,docs]"
```

## Initialize Datasets

After installation, initialize the datasets:

```bash
glazing init
```

This command will:
1. Download all datasets (~54MB compressed)
2. Extract and convert them to JSON Lines format
3. Store them in `~/.local/share/glazing/` (or `%LOCALAPPDATA%\glazing\` on Windows)

### Custom Data Directory

To use a custom directory for datasets:

```bash
glazing init --data-dir /path/to/your/data
```

Or set the environment variable:

```bash
export GLAZING_DATA_DIR=/path/to/your/data
glazing init
```

## Verify Installation

Check that Glazing is properly installed:

```bash
# Check version
glazing --version

# View available commands
glazing --help

# Verify datasets are initialized
python -c "from glazing.initialize import check_initialization; print('Initialized' if check_initialization() else 'Not initialized')"
```

## Optional Dependencies

### Development Tools

For contributing to Glazing:

```bash
pip install glazing[dev]
```

This installs:
- ruff (linting and formatting)
- mypy (type checking)
- pytest (testing framework)
- pre-commit (git hooks)

### Documentation Tools

For building documentation:

```bash
pip install glazing[docs]
```

This installs:
- mkdocs (documentation generator)
- mkdocs-material (theme)
- mkdocstrings (API documentation)

## Environment Variables

Glazing respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GLAZING_DATA_DIR` | Directory for storing datasets | `~/.local/share/glazing/` |
| `XDG_DATA_HOME` | Base directory for user data (Linux/macOS) | `~/.local/share/` |
| `GLAZING_SKIP_INIT_CHECK` | Skip initialization check on import | `false` |

## Platform-Specific Notes

### macOS

On macOS, you might need to install Xcode Command Line Tools:

```bash
xcode-select --install
```

### Windows

On Windows, we recommend using:

- Windows Terminal for better CLI experience
- WSL2 for Unix-like environment

### Linux

Most Linux distributions come with Python pre-installed. Ensure you have Python 3.13+:

```bash
python3 --version
```

## Docker Installation

Glazing provides a Docker image for containerized usage, allowing you to use the full CLI without installing dependencies on your system.

### Building the Docker Image

Clone the repository and build the image:

```bash
git clone https://github.com/aaronstevenwhite/glazing.git
cd glazing
docker build -t glazing:latest .
```

### Running with Docker

The Docker container exposes the entire Glazing CLI. You can run any glazing command by passing it to the container:

```bash
# Show help
docker run --rm glazing:latest --help

# Initialize datasets (mount volume to persist data)
docker run --rm -v glazing-data:/data glazing:latest init

# Search across datasets
docker run --rm -v glazing-data:/data glazing:latest search query "give"

# Find data with variations using fuzzy matching
docker run --rm -v glazing-data:/data glazing:latest search query "realize" --fuzzy

# Extract cross-references
docker run --rm -v glazing-data:/data glazing:latest xref extract

# Resolve cross-references
docker run --rm -v glazing-data:/data glazing:latest xref resolve "give.01" --source propbank
```

### Using Local Data

To use your existing local data directory:

```bash
# Mount your local data directory
docker run --rm -v /path/to/your/data:/data glazing:latest search query "run"
```

### Interactive Shell

For an interactive Python session with Glazing:

```bash
docker run --rm -it -v glazing-data:/data --entrypoint python glazing:latest
```

Then in Python:

```python
from glazing.search import UnifiedSearch
from pathlib import Path

search = UnifiedSearch(data_dir=Path("/data"))
results = search.search("give")
```

### Docker Compose

For more complex setups, use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  glazing:
    image: glazing:latest
    volumes:
      - glazing-data:/data
    environment:
      - GLAZING_DATA_DIR=/data

volumes:
  glazing-data:
```

Then run:

```bash
# Initialize datasets
docker-compose run glazing init

# Use the CLI
docker-compose run glazing search query "transfer"
```

## Troubleshooting

### Import Warning

If you see a warning about uninitialized datasets:

```python
Warning: Glazing datasets not initialized.
Run 'glazing init' to download and convert all datasets.
```

Simply run:

```bash
glazing init
```

### Permission Errors

If you encounter permission errors:

```bash
# Use a different directory
glazing init --data-dir ~/my-glazing-data

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/share/glazing
```

### FrameNet Manual Download

FrameNet requires manual download due to licensing:

1. Visit the [FrameNet data page](https://framenet.icsi.berkeley.edu/fndrupal/framenet_data)
2. Download FrameNet 1.7
3. Convert manually:

```bash
glazing convert dataset --dataset framenet --input-dir framenet_v17/ --output-dir data/
```

### Memory Issues

If you encounter memory issues during conversion:

```bash
# Process datasets one at a time
glazing download dataset --dataset verbnet
glazing convert dataset --dataset verbnet --input-dir raw/ --output-dir data/

# Repeat for other datasets
```

## Next Steps

Once installed, proceed to the [Quick Start](quick-start.md) guide to begin using Glazing.
