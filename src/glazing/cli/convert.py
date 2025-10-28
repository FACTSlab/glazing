"""CLI commands for converting datasets to JSON Lines format.

This module provides commands for converting linguistic datasets from their
native formats (XML, database) to JSON Lines format for efficient processing.

Commands
--------
convert dataset
    Convert a specific dataset or all datasets to JSON Lines.
convert list
    List available datasets for conversion.
convert info
    Get information about a dataset's conversion process.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Literal, TypedDict

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from glazing.framenet.converter import FrameNetConverter
from glazing.propbank.converter import PropBankConverter
from glazing.verbnet.converter import VerbNetConverter
from glazing.wordnet.converter import WordNetConverter

console = Console()

DatasetName = Literal["verbnet", "propbank", "wordnet", "framenet", "all"]


class DatasetInfoDict(TypedDict, total=False):
    """Type for dataset information."""

    name: str
    description: str
    input_format: str
    output_files: list[str]
    converter: type[VerbNetConverter | PropBankConverter | WordNetConverter | FrameNetConverter]


DATASET_INFO: dict[str, DatasetInfoDict] = {
    "verbnet": {
        "name": "VerbNet",
        "description": "Convert VerbNet XML files to JSON Lines",
        "input_format": "XML files (*.xml)",
        "output_files": ["verbnet.jsonl"],
        "converter": VerbNetConverter,
    },
    "propbank": {
        "name": "PropBank",
        "description": "Convert PropBank XML framesets to JSON Lines",
        "input_format": "XML files (*.xml)",
        "output_files": ["propbank.jsonl"],
        "converter": PropBankConverter,
    },
    "wordnet": {
        "name": "WordNet",
        "description": "Convert WordNet database files to JSON Lines",
        "input_format": "Database files (data.*, index.*, *.exc)",
        "output_files": ["wordnet.jsonl"],
        "converter": WordNetConverter,
    },
    "framenet": {
        "name": "FrameNet",
        "description": "Convert FrameNet XML frames to JSON Lines",
        "input_format": "XML files in frame/ directory",
        "output_files": ["framenet.jsonl"],
        "converter": FrameNetConverter,
    },
}


def convert_verbnet(input_dir: Path, output_dir: Path, verbose: bool = False) -> None:
    """Convert VerbNet XML files to JSON Lines.

    Parameters
    ----------
    input_dir : Path
        Directory containing VerbNet XML files.
    output_dir : Path
        Directory to write JSON Lines files to.
    verbose : bool
        Show verbose output.
    """
    converter = VerbNetConverter()
    output_file = output_dir / "verbnet.jsonl"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting VerbNet files...", total=None)

        count = converter.convert_verbnet_directory(str(input_dir), str(output_file))

        progress.update(task, completed=True)

    if verbose:
        console.print(f"[green]✓[/green] Converted {count} verb classes")
        console.print(f"  Output: {output_file}")


def convert_propbank(input_dir: Path, output_dir: Path, verbose: bool = False) -> None:
    """Convert PropBank XML framesets to JSON Lines.

    Parameters
    ----------
    input_dir : Path
        Directory containing PropBank XML files.
    output_dir : Path
        Directory to write JSON Lines files to.
    verbose : bool
        Show verbose output.
    """
    converter = PropBankConverter()
    output_file = output_dir / "propbank.jsonl"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting PropBank files...", total=None)

        count = converter.convert_framesets_directory(str(input_dir), str(output_file))

        progress.update(task, completed=True)

    if verbose:
        console.print(f"[green]✓[/green] Converted {count} framesets")
        console.print(f"  Output: {output_file}")


def convert_wordnet(input_dir: Path, output_dir: Path, verbose: bool = False) -> None:
    """Convert WordNet database to JSON Lines.

    Parameters
    ----------
    input_dir : Path
        Directory containing WordNet database files.
    output_dir : Path
        Directory to write JSON Lines files to.
    verbose : bool
        Show verbose output.
    """
    converter = WordNetConverter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting WordNet database...", total=None)

        output_file = output_dir / "wordnet.jsonl"
        stats = converter.convert_wordnet_database(str(input_dir), str(output_file))

        progress.update(task, completed=True)

    if verbose:
        console.print("[green]✓[/green] Converted WordNet database")
        for key, count in stats.items():
            console.print(f"  {key}: {count} entries")


def convert_framenet(input_dir: Path, output_dir: Path, verbose: bool = False) -> None:
    """Convert FrameNet XML frames to JSON Lines with lexical units.

    Parameters
    ----------
    input_dir : Path
        Directory containing FrameNet XML files (should have frame/ subdirectory).
    output_dir : Path
        Directory to write JSON Lines files to.
    verbose : bool
        Show verbose output.
    """
    converter = FrameNetConverter()

    # FrameNet has frames in a subdirectory
    frames_dir = input_dir / "frame"
    if not frames_dir.exists():
        frames_dir = input_dir  # Fallback to input_dir if no frame/ subdirectory

    output_file = output_dir / "framenet.jsonl"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting FrameNet frames and lexical units...", total=None)

        count = converter.convert_frames_directory(str(frames_dir), str(output_file))

        progress.update(task, completed=True)

    if verbose:
        console.print(f"[green]✓[/green] Converted {count} frames with lexical units")
        console.print(f"  Output: {output_file}")


def _find_dataset_input_dir(base_dir: Path, dataset: str) -> Path:
    """Find the appropriate input directory for a dataset.

    Parameters
    ----------
    base_dir : Path
        Base input directory.
    dataset : str
        Dataset name.

    Returns
    -------
    Path
        The actual input directory to use.
    """
    # Define potential subdirectories for each dataset
    dataset_paths = {
        "verbnet": ["verbnet3.4", "vn3.4/verbnet3.4"],
        "propbank": ["frames", "propbank-frames/frames"],
        "wordnet": ["wn3.1"],
        "framenet": ["framenet_v17"],
    }

    if dataset in dataset_paths:
        for path_str in dataset_paths[dataset]:
            candidate = base_dir / path_str
            if candidate.exists():
                return candidate

    return base_dir


def _convert_single_dataset(dataset: str, input_dir: Path, output_dir: Path, verbose: bool) -> None:
    """Convert a single dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.
    input_dir : Path
        Input directory.
    output_dir : Path
        Output directory.
    verbose : bool
        Verbose output flag.
    """
    console.print(f"\n[bold]Converting {DATASET_INFO[dataset]['name']}...[/bold]")

    actual_input_dir = _find_dataset_input_dir(input_dir, dataset)

    if dataset == "verbnet":
        convert_verbnet(actual_input_dir, output_dir, verbose)
    elif dataset == "propbank":
        convert_propbank(actual_input_dir, output_dir, verbose)
    elif dataset == "wordnet":
        convert_wordnet(actual_input_dir, output_dir, verbose)
    elif dataset == "framenet":
        convert_framenet(actual_input_dir, output_dir, verbose)


@click.group()
def convert() -> None:
    """Convert linguistic datasets to JSON Lines format."""


@convert.command(name="dataset")
@click.option(
    "--dataset",
    type=click.Choice(["verbnet", "propbank", "wordnet", "framenet", "all"]),
    required=True,
    help="Dataset to convert.",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Input directory containing dataset files.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory for JSON Lines files.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show verbose output.",
)
def convert_dataset_cmd(
    dataset: DatasetName, input_dir: str | Path, output_dir: str | Path, verbose: bool
) -> None:
    """Convert a dataset to JSON Lines format.

    Examples
    --------
    Convert VerbNet:
        $ glazing convert dataset --dataset verbnet \\
            --input-dir vn3.4/verbnet3.4 --output-dir output/

    Convert all datasets:
        $ glazing convert dataset --dataset all --input-dir data/ --output-dir output/
    """
    try:
        # Ensure Path objects
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        if dataset == "all":
            datasets_to_convert = ["verbnet", "propbank", "wordnet", "framenet"]
        else:
            datasets_to_convert = [dataset]

        for ds in datasets_to_convert:
            _convert_single_dataset(ds, input_dir, output_dir, verbose)

        console.print("\n[bold green]✓ Conversion complete![/bold green]")

    except FileNotFoundError as e:
        console.print(f"[red]✗ Input directory not found: {e}[/red]")
        sys.exit(1)
    except PermissionError as e:
        console.print(f"[red]✗ Permission denied: {e}[/red]")
        sys.exit(1)
    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Conversion failed: {e}[/red]")
        if verbose:
            traceback.print_exc()
        sys.exit(1)


@convert.command(name="list")
def list_datasets() -> None:
    """List available datasets for conversion."""
    table = Table(title="Available Datasets for Conversion")
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Input Format", style="yellow")

    for key, info in DATASET_INFO.items():
        if key != "all":  # Skip 'all' in the listing
            table.add_row(info["name"], info["description"], info["input_format"])

    console.print(table)


@convert.command(name="info")
@click.option(
    "--dataset",
    type=click.Choice(["verbnet", "propbank", "wordnet", "framenet"]),
    required=True,
    help="Dataset to get information about.",
)
def dataset_info_cmd(dataset: str) -> None:
    """Get information about a dataset's conversion process."""
    info = DATASET_INFO[dataset]

    console.print(f"\n[bold cyan]{info['name']} Conversion Information[/bold cyan]")
    console.print(f"[white]Description:[/white] {info['description']}")
    console.print(f"[white]Input Format:[/white] {info['input_format']}")
    console.print("[white]Output Files:[/white]")
    for output_file in info["output_files"]:
        console.print(f"  • {output_file}")

    converter_class = info.get("converter")
    if converter_class:
        console.print(f"\n[white]Converter Class:[/white] {converter_class.__name__}")

    # Add dataset-specific notes
    if dataset == "verbnet":
        console.print("\n[yellow]Notes:[/yellow]")
        console.print("• Converts VerbNet 3.4 XML files")
        console.print("• Preserves verb class hierarchy")
        console.print("• Includes thematic roles and selectional restrictions")
    elif dataset == "propbank":
        console.print("\n[yellow]Notes:[/yellow]")
        console.print("• Converts PropBank framesets")
        console.print("• Preserves argument structure and examples")
        console.print("• Includes cross-references to VerbNet")
    elif dataset == "wordnet":
        console.print("\n[yellow]Notes:[/yellow]")
        console.print("• Converts WordNet 3.1 database files")
        console.print("• Creates separate files for each POS")
        console.print("• Includes synsets, indices, and exceptions")
    elif dataset == "framenet":
        console.print("\n[yellow]Notes:[/yellow]")
        console.print("• Converts FrameNet 1.7 XML frames")
        console.print("• Preserves frame elements and relations")
        console.print("• Includes lexical units and examples")
