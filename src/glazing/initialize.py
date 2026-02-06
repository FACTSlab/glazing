"""Initialize glazing by downloading and converting all datasets.

This module provides functionality to automatically download and convert
all linguistic datasets on first use or installation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from glazing.downloader import (
    BaseDownloader,
    FrameNetDownloader,
    PropBankDownloader,
    VerbNetDownloader,
    WordNetDownloader,
)
from glazing.framenet.converter import FrameNetConverter
from glazing.propbank.converter import PropBankConverter
from glazing.verbnet.converter import VerbNetConverter
from glazing.wordnet.converter import WordNetConverter


def get_default_data_dir() -> Path:
    """Get the default data directory for glazing.

    Returns
    -------
    Path
        Default data directory path.
    """
    # Check GLAZING_DATA_DIR first (used in Docker and for custom installations)
    glazing_data = os.environ.get("GLAZING_DATA_DIR")
    if glazing_data:
        return Path(glazing_data)

    # Use XDG_DATA_HOME if available, otherwise ~/.local/share
    xdg_data = os.environ.get("XDG_DATA_HOME")
    base_dir = Path(xdg_data) if xdg_data else Path.home() / ".local" / "share"

    return base_dir / "glazing"


def get_default_data_path(filename: str | None = None) -> Path:
    """Get the default path for a converted data file.

    Parameters
    ----------
    filename : str | None, optional
        Filename to append to the converted data directory.
        If None, returns the converted directory path.

    Returns
    -------
    Path
        Path to the data file or directory.
    """
    base = get_default_data_dir() / "converted"
    return base / filename if filename else base


def _get_dataset_config(name: str) -> tuple[BaseDownloader | None, object | None, str]:
    """Get downloader, converter, and output file for a dataset.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    tuple[BaseDownloader | None, object | None, str]
        Downloader, converter, and output file name.
    """
    if name == "verbnet":
        return VerbNetDownloader(), VerbNetConverter(), "verbnet.jsonl"
    if name == "propbank":
        return PropBankDownloader(), PropBankConverter(), "propbank.jsonl"
    if name == "wordnet":
        return WordNetDownloader(), WordNetConverter(), "wordnet.jsonl"
    if name == "framenet":
        return FrameNetDownloader(), FrameNetConverter(), "framenet.jsonl"
    return None, None, ""


def _get_display_name(name: str) -> str:
    """Get the display name for a dataset.

    Parameters
    ----------
    name : str
        Dataset name (lowercase).

    Returns
    -------
    str
        Display name with proper capitalization.
    """
    display_names = {
        "verbnet": "VerbNet",
        "propbank": "PropBank",
        "wordnet": "WordNet",
        "framenet": "FrameNet",
    }
    return display_names.get(name, name)


def _process_dataset(name: str, data_dir: Path, verbose: bool) -> bool:
    """Process a single dataset: download and convert.

    Parameters
    ----------
    name : str
        Dataset name.
    data_dir : Path
        Data directory.
    verbose : bool
        Print progress messages.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        display_name = _get_display_name(name)

        if verbose:
            click.echo(f"\n{display_name}:")
            click.echo("-" * 40)

        # Download
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        if verbose:
            click.echo(f"  Downloading {display_name}...")

        downloader, converter, output_file = _get_dataset_config(name)

        if not downloader or not converter:
            return False

        download_path = downloader.download(output_dir=raw_dir)

        if verbose:
            click.echo(f"  ✓ Downloaded to {download_path}")

        # Convert
        converted_dir = data_dir / "converted"
        converted_dir.mkdir(exist_ok=True)

        if verbose:
            click.echo(f"  Converting {display_name}...")

        output = converted_dir / output_file
        _convert_dataset(name, download_path, output, converter, verbose)

    except (ValueError, TypeError, RuntimeError, FileNotFoundError, PermissionError) as e:
        if verbose:
            click.echo(f"  ✗ Error: {e!s}", err=True)
        return False
    else:
        return True


def _convert_wordnet_supplementary(
    converter: object, source: Path, converted_dir: Path, verbose: bool
) -> None:
    sense_count = converter.convert_sense_index(source, converted_dir / "wordnet_senses.jsonl")  # type: ignore[attr-defined]
    exc_count = converter.convert_exceptions(source, converted_dir / "wordnet_exceptions.jsonl")  # type: ignore[attr-defined]
    if verbose:
        click.echo(f"  ✓ Converted {sense_count} senses, {exc_count} exceptions")


def _convert_framenet_supplementary(
    converter: object, download_path: Path, converted_dir: Path, verbose: bool
) -> None:
    semtype_count = converter.convert_semtypes_file(  # type: ignore[attr-defined]
        download_path / "semTypes.xml", converted_dir / "framenet_semtypes.jsonl"
    )
    if verbose:
        click.echo(f"  ✓ Converted {semtype_count} semantic types")
    fulltext_count = converter.convert_fulltext_directory(  # type: ignore[attr-defined]
        download_path / "fulltext", converted_dir / "framenet_fulltext.jsonl"
    )
    if verbose:
        click.echo(f"  ✓ Converted {fulltext_count} fulltext sentences")


def _convert_dataset(
    name: str, download_path: Path, output: Path, converter: object, verbose: bool
) -> None:
    """Convert a dataset from raw format to JSON Lines.

    Parameters
    ----------
    name : str
        Dataset name.
    download_path : Path
        Path where dataset was downloaded.
    output : Path
        Output file path.
    converter : object
        Converter instance.
    verbose : bool
        Print progress messages.
    """
    if name == "verbnet":
        source = download_path / "verbnet3.4"
        count = converter.convert_verbnet_directory(source, output)  # type: ignore[attr-defined]
        if verbose:
            click.echo(f"  ✓ Converted {count} files")
    elif name == "propbank":
        source = download_path / "frames"
        count = converter.convert_framesets_directory(source, output)  # type: ignore[attr-defined]
        if verbose:
            click.echo(f"  ✓ Converted {count} framesets")
    elif name == "wordnet":
        source = download_path
        stats = converter.convert_wordnet_database(source, output)  # type: ignore[attr-defined]
        if verbose:
            total = stats.get("total_synsets", 0)
            click.echo(f"  ✓ Converted {total} synsets")
        _convert_wordnet_supplementary(converter, source, output.parent, verbose)
    elif name == "framenet":
        source = download_path / "frame"
        count = converter.convert_frames_directory(source, output)  # type: ignore[attr-defined]
        if verbose:
            click.echo(f"  ✓ Converted {count} frames")
        _convert_framenet_supplementary(converter, download_path, output.parent, verbose)


def initialize_datasets(
    data_dir: Path | None = None, force: bool = False, verbose: bool = True
) -> bool:
    """Download and convert all datasets.

    Parameters
    ----------
    data_dir : Path | None
        Directory to store data. If None, uses default.
    force : bool
        Force re-download even if data exists.
    verbose : bool
        Print progress messages.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already initialized (unless force is True)
    marker_file = data_dir / ".initialized"
    if marker_file.exists() and not force:
        if verbose:
            click.echo("Datasets already initialized. Use --force to re-download.")
        return True

    if verbose:
        click.echo(f"Initializing glazing datasets in {data_dir}")
        click.echo("=" * 60)

    # Process each dataset
    datasets = ["verbnet", "propbank", "wordnet", "framenet"]
    results = [_process_dataset(name, data_dir, verbose) for name in datasets]
    success = all(results)

    # Create marker file
    if success:
        marker_file.touch()
        if verbose:
            click.echo("\n" + "=" * 60)
            click.echo("✅ All datasets successfully initialized!")
            click.echo(f"Data location: {data_dir}")
    elif verbose:
        click.echo("\n⚠️  Some datasets failed to initialize", err=True)

    return success


def check_initialization(data_dir: Path | None = None) -> bool:
    """Check if datasets are initialized.

    Parameters
    ----------
    data_dir : Path | None
        Data directory to check.

    Returns
    -------
    bool
        True if initialized, False otherwise.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    marker_file = data_dir / ".initialized"
    return marker_file.exists()


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(),
    help="Directory to store datasets (default: ~/.local/share/glazing)",
)
@click.option("--force", is_flag=True, help="Force re-download even if data exists")
@click.option("--quiet", is_flag=True, help="Suppress output messages")
def main(data_dir: str | Path | None, force: bool, quiet: bool) -> None:
    """Set up all datasets. Downloads raw data and converts to JSON Lines format."""
    # Convert to Path if provided
    if data_dir is not None:
        data_dir = Path(data_dir)

    success = initialize_datasets(data_dir=data_dir, force=force, verbose=not quiet)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
