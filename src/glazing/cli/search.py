"""CLI commands for searching across linguistic datasets.

This module provides commands for searching and querying converted datasets
in JSON Lines format.

Commands
--------
search query
    Search across datasets with a text query.
search entity
    Get details about a specific entity.
search role
    Search for semantic roles across datasets.
search cross-ref
    Find cross-references between datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from glazing.framenet.models import Frame
from glazing.initialize import get_default_data_path
from glazing.propbank.models import Frameset
from glazing.search import SearchResult, UnifiedSearch
from glazing.verbnet.models import VerbClass
from glazing.wordnet.models import Synset

console = Console()

DatasetName = Literal["all", "verbnet", "propbank", "wordnet", "framenet"]


def _display_verbnet_details(entity: VerbClass) -> None:
    """Show VerbNet class members, roles, and frames."""
    if hasattr(entity, "members"):
        console.print(f"[white]Members:[/white] {len(entity.members)}")
    if hasattr(entity, "themroles"):
        roles = ", ".join(r.type for r in entity.themroles)
        console.print(f"[white]Thematic Roles:[/white] {roles}")
    if hasattr(entity, "frames"):
        entity_frames = getattr(entity, "frames", [])
        console.print(f"[white]Frames:[/white] {len(entity_frames)}")


def _display_propbank_details(entity: Frameset) -> None:
    """Show PropBank frameset rolesets."""
    if hasattr(entity, "rolesets"):
        console.print(f"[white]Rolesets:[/white] {len(entity.rolesets)}")
        for rs in entity.rolesets[:3]:  # Show first 3
            console.print(f"  • {rs.id}: {rs.name}")


def _display_wordnet_details(entity: Synset) -> None:
    """Show WordNet synset words and definition."""
    if hasattr(entity, "words"):
        entity_words = getattr(entity, "words", [])
        words = ", ".join(getattr(w, "lemma", str(w)) for w in entity_words[:5])
        console.print(f"[white]Words:[/white] {words}")
    if hasattr(entity, "definition"):
        console.print(f"[white]Definition:[/white] {entity.definition}")


def _display_framenet_details(entity: Frame) -> None:
    """Show FrameNet frame elements."""
    if hasattr(entity, "frame_elements"):
        console.print(f"[white]Frame Elements:[/white] {len(entity.frame_elements)}")
        for fe in entity.frame_elements[:5]:  # Show first 5
            console.print(f"  • {fe.name} ({fe.core_type})")


def _display_entity_details(
    entity: VerbClass | Frameset | Synset | Frame, entity_id: str, dataset: str
) -> None:
    """Display formatted entity details based on dataset type.

    Parameters
    ----------
    entity
        Entity object to display.
    entity_id : str
        Entity identifier.
    dataset : str
        Dataset name.
    """
    console.print(f"\n[bold cyan]{dataset.upper()} Entity Details[/bold cyan]")
    console.print(f"[white]ID:[/white] {entity_id}")

    if dataset == "verbnet" and isinstance(entity, VerbClass):
        _display_verbnet_details(entity)
    elif dataset == "propbank" and isinstance(entity, Frameset):
        _display_propbank_details(entity)
    elif dataset == "wordnet" and isinstance(entity, Synset):
        _display_wordnet_details(entity)
    elif dataset == "framenet" and isinstance(entity, Frame):
        _display_framenet_details(entity)


def _load_dataset_files(search: UnifiedSearch, data_dir: Path, dataset: str) -> None:
    """Load files for a specific dataset into the search index.

    Parameters
    ----------
    search : UnifiedSearch
        Search object to load into.
    data_dir : Path
        Directory containing JSON Lines files.
    dataset : str
        Dataset name to load.
    """
    if dataset == "verbnet":
        file = data_dir / "verbnet.jsonl"
        if file.exists():
            search.load_verbnet_from_jsonl(str(file))
    elif dataset == "propbank":
        file = data_dir / "propbank.jsonl"
        if file.exists():
            search.load_propbank_from_jsonl(str(file))
    elif dataset == "wordnet":
        # Load all WordNet files
        for pos in ["noun", "verb", "adj", "adv"]:
            synset_file = data_dir / f"synsets_{pos}.jsonl"
            index_file = data_dir / f"index_{pos}.jsonl"
            if synset_file.exists() and index_file.exists():
                search.load_wordnet_from_jsonl(str(synset_file), str(index_file), pos)
    elif dataset == "framenet":
        file = data_dir / "framenet.jsonl"
        if file.exists():
            search.load_framenet_from_jsonl(str(file))


def load_search_index(data_dir: str | Path, datasets: list[str] | None = None) -> UnifiedSearch:
    """Load search index from converted JSON Lines files.

    Parameters
    ----------
    data_dir : Path
        Directory containing converted JSON Lines files.
    datasets : list[str] | None
        List of datasets to load, or None for all.

    Returns
    -------
    UnifiedSearch
        Initialized search object.
    """
    search = UnifiedSearch()
    data_dir = Path(data_dir)  # Ensure it's a Path object

    if datasets is None:
        datasets = ["verbnet", "propbank", "wordnet", "framenet"]

    # Load each dataset's JSON Lines files
    for dataset in datasets:
        _load_dataset_files(search, data_dir, dataset)

    return search


@click.group()
def search() -> None:
    """Query linguistic datasets from the command line."""


@search.command(name="query")
@click.argument("query_text")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files "
    "(default: ~/.local/share/glazing/converted).",
)
@click.option(
    "--dataset",
    type=click.Choice(["all", "verbnet", "propbank", "wordnet", "framenet"]),
    default="all",
    help="Dataset to search in.",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of results to show.",
)
@click.option(
    "--fuzzy",
    is_flag=True,
    help="Enable fuzzy matching for typo correction.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.8,
    help="Minimum similarity threshold for fuzzy matching (0.0-1.0).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON.",
)
def search_query(  # noqa: PLR0913
    query_text: str,
    data_dir: str | Path,
    dataset: DatasetName,
    limit: int,
    fuzzy: bool,
    threshold: float,
    output_json: bool,
) -> None:
    """Search across datasets with a text query.

    Examples
    --------
    Search all datasets:
        $ glazing search query "give" --data-dir output/

    Search only VerbNet:
        $ glazing search query "transfer" --data-dir output/ --dataset verbnet
    """
    try:
        # Determine which datasets to load
        datasets_to_load: list[str] | None = None if dataset == "all" else [dataset]

        # Load search index
        search_engine = load_search_index(data_dir, datasets_to_load)

        # Perform search with or without fuzzy matching
        if fuzzy:
            results = search_engine.search_with_fuzzy(query_text, threshold)
        else:
            results = search_engine.search(query_text)

        if output_json:
            # Output as JSON
            json_results = []
            for result in results[:limit]:
                json_results.append(
                    {
                        "dataset": result.dataset,
                        "id": result.id,
                        "type": result.type,
                        "name": result.name,
                        "description": result.description,
                        "score": result.score,
                    }
                )
            console.print(json.dumps(json_results, indent=2))
        else:
            # Output as formatted table
            if not results:
                console.print("[yellow]No results found.[/yellow]")
                return

            title = f"{'Fuzzy ' if fuzzy else ''}Search Results for '{query_text}'"
            table = Table(title=title)
            table.add_column("Dataset", style="cyan", no_wrap=True)
            table.add_column("Type", style="magenta")
            table.add_column("ID/Name", style="green")
            table.add_column("Description", style="white")
            table.add_column("Score", style="yellow")

            display_names = {
                "verbnet": "VerbNet",
                "propbank": "PropBank",
                "wordnet": "WordNet",
                "framenet": "FrameNet",
            }

            for result in results[:limit]:
                dataset_display = display_names.get(result.dataset.lower(), result.dataset)
                table.add_row(
                    dataset_display,
                    result.type,
                    f"{result.id}\n{result.name}" if result.name != result.id else result.id,
                    (
                        result.description[:80] + "..."
                        if len(result.description) > 80
                        else result.description
                    ),
                    f"{result.score:.2f}",
                )

            console.print(table)

            if len(results) > limit:
                console.print(f"\n[dim]Showing {limit} of {len(results)} results.[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]✗ Data files not found: {e}[/red]")
        console.print(
            "[yellow]Tip: Make sure you've converted the datasets first "
            "using 'glazing convert dataset'[/yellow]"
        )
        sys.exit(1)
    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Search failed: {e}[/red]")
        sys.exit(1)


@search.command(name="entity")
@click.argument("entity_id")
@click.option(
    "--dataset",
    type=click.Choice(["verbnet", "propbank", "wordnet", "framenet"]),
    required=True,
    help="Dataset the entity belongs to.",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files "
    "(default: ~/.local/share/glazing/converted).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON.",
)
def get_entity(
    entity_id: str,
    dataset: str,
    data_dir: str | Path,
    output_json: bool,
) -> None:
    """Get details about a specific entity.

    Examples
    --------
    Get VerbNet class details:
        $ glazing search entity "give-13.1" --dataset verbnet --data-dir output/

    Get PropBank roleset details:
        $ glazing search entity "give.01" --dataset propbank --data-dir output/
    """
    try:
        # Load search index for specific dataset
        search_engine = load_search_index(data_dir, [dataset])

        # Get entity details
        entity = search_engine.get_entity(entity_id, dataset)

        if entity is None:
            console.print(f"[yellow]Entity '{entity_id}' not found in {dataset}.[/yellow]")
            return

        if output_json:
            # Output as JSON
            console.print(entity.model_dump_json(indent=2))
        else:
            _display_entity_details(entity, entity_id, dataset)

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Failed to get entity: {e}[/red]")
        sys.exit(1)


@search.command(name="role")
@click.argument("role_name")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files "
    "(default: ~/.local/share/glazing/converted).",
)
@click.option(
    "--dataset",
    type=click.Choice(["all", "verbnet", "propbank", "framenet"]),
    default="all",
    help="Dataset to search in.",
)
def search_role(
    role_name: str,
    data_dir: str | Path,
    dataset: str,
) -> None:
    """Search for semantic roles across datasets.

    Examples
    --------
    Search for Agent role:
        $ glazing search role "Agent" --data-dir output/

    Search for ARG0 in PropBank:
        $ glazing search role "ARG0" --data-dir output/ --dataset propbank
    """
    try:
        # Determine which datasets to load
        datasets_to_load = ["verbnet", "propbank", "framenet"] if dataset == "all" else [dataset]

        # Load search index
        search_engine = load_search_index(data_dir, datasets_to_load)

        # Search for roles
        results = search_engine.search_semantic_roles(role_name)

        if not results:
            console.print(f"[yellow]No roles matching '{role_name}' found.[/yellow]")
            return

        # Group results by dataset
        by_dataset: dict[str, list[SearchResult]] = {}
        for result in results:
            if result.dataset not in by_dataset:
                by_dataset[result.dataset] = []
            by_dataset[result.dataset].append(result)

        # Display results
        tree = Tree(f"[bold]Semantic Roles matching '{role_name}'[/bold]")

        for ds, items in by_dataset.items():
            ds_branch = tree.add(f"[cyan]{ds.upper()}[/cyan] ({len(items)} matches)")
            for item in items[:5]:  # Show first 5 per dataset
                ds_branch.add(f"{item.id}: {item.description[:60]}...")
            if len(items) > 5:
                ds_branch.add(f"[dim]... and {len(items) - 5} more[/dim]")

        console.print(tree)

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Role search failed: {e}[/red]")
        sys.exit(1)


@search.command(name="cross-ref")
@click.option(
    "--source",
    type=click.Choice(["verbnet", "propbank", "wordnet", "framenet"]),
    required=True,
    help="Source dataset.",
)
@click.option(
    "--target",
    type=click.Choice(["verbnet", "propbank", "wordnet", "framenet"]),
    required=True,
    help="Target dataset.",
)
@click.option(
    "--id",
    "entity_id",
    required=True,
    help="Entity ID in source dataset.",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files "
    "(default: ~/.local/share/glazing/converted).",
)
def find_cross_ref(
    source: str,
    target: str,
    entity_id: str,
    data_dir: str | Path,
) -> None:
    """Find cross-references between datasets.

    Examples
    --------
    Find PropBank references for VerbNet class:
        $ glazing search cross-ref --source verbnet --target propbank \
            --id "give-13.1" --data-dir output/

    Find VerbNet references for PropBank roleset:
        $ glazing search cross-ref --source propbank --target verbnet \
            --id "give.01" --data-dir output/
    """
    try:
        # Load search index
        search_engine = load_search_index(data_dir, [source, target])

        # Find cross-references
        references = search_engine.find_cross_references(entity_id, source, target)

        if not references:
            console.print(
                f"[yellow]No {target} references found for {source} entity '{entity_id}'.[/yellow]"
            )
            return

        # Display results
        table = Table(title=f"Cross-References: {source.upper()} → {target.upper()}")
        table.add_column("Source", style="cyan")
        table.add_column("Target", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Confidence", style="yellow")

        for ref in references:
            table.add_row(
                f"{source}: {entity_id}",
                f"{target}: {ref['target_id']}",
                str(ref.get("mapping_type", "direct")),
                f"{ref.get('confidence', 1.0):.2f}",
            )

        console.print(table)

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Cross-reference search failed: {e}[/red]")
        sys.exit(1)


@search.command(name="fuzzy")
@click.argument("query_text")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.8,
    help="Minimum similarity threshold (0.0-1.0).",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of results to show.",
)
def search_fuzzy(
    query_text: str,
    data_dir: str | Path,
    threshold: float,
    limit: int,
) -> None:
    """Search with fuzzy matching and typo correction.

    Examples
    --------
    Search with typo correction:
        $ glazing search fuzzy "instsrument" --threshold 0.7
    """
    try:
        search_engine = load_search_index(data_dir)
        results = search_engine.search_with_fuzzy(query_text, threshold)

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        table = Table(title=f"Fuzzy Search Results for '{query_text}'")
        table.add_column("Dataset", style="cyan", no_wrap=True)
        table.add_column("ID", style="green")
        table.add_column("Name", style="white")
        table.add_column("Score", style="yellow")

        for result in results[:limit]:
            table.add_row(
                result.dataset.upper(),
                result.id,
                result.name,
                f"{result.score:.3f}",
            )

        console.print(table)

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Fuzzy search failed: {e}[/red]")
        sys.exit(1)


@search.command(name="roles")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files.",
)
@click.option("--optional", is_flag=True, help="Find optional roles.")
@click.option("--indexed", is_flag=True, help="Find indexed roles (_I, _J).")
@click.option("--verb-specific", is_flag=True, help="Find verb-specific roles.")
@click.option("--dataset", default="verbnet", help="Dataset to search (default: verbnet).")
def search_roles(
    data_dir: str | Path,
    optional: bool,
    indexed: bool,
    verb_specific: bool,
    dataset: str,
) -> None:
    """Search for semantic roles with specific properties.

    Examples
    --------
    Find optional roles:
        $ glazing search roles --optional

    Find indexed roles:
        $ glazing search roles --indexed
    """
    try:
        search_engine = load_search_index(data_dir, [dataset])

        if dataset == "verbnet":
            classes = search_engine.search_verbnet_roles(
                optional=optional if optional else None,
                indexed=indexed if indexed else None,
                verb_specific=verb_specific if verb_specific else None,
            )

            if not classes:
                console.print("[yellow]No matching classes found.[/yellow]")
                return

            table = Table(title="VerbNet Classes with Matching Roles")
            table.add_column("Class ID", style="cyan")
            table.add_column("Members", style="green")
            table.add_column("Roles", style="white")

            for cls in classes[:20]:
                role_str = ", ".join(r.type for r in cls.themroles[:5])
                if len(cls.themroles) > 5:
                    role_str += f" (+{len(cls.themroles) - 5} more)"
                table.add_row(
                    cls.id,
                    str(len(cls.members)),
                    role_str,
                )

            console.print(table)

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Role search failed: {e}[/red]")
        sys.exit(1)


@search.command(name="args")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files.",
)
@click.option(
    "--type",
    "arg_type",
    type=click.Choice(["core", "modifier"]),
    help="Argument type.",
)
@click.option(
    "--prefix",
    type=click.Choice(["C", "R"]),
    help="Continuation or reference prefix.",
)
@click.option("--modifier", help="Modifier type (e.g., LOC, TMP).")
@click.option("--number", type=int, help="Argument number (0-7).")
@click.option("--dataset", default="propbank", help="Dataset to search (default: propbank).")
def search_args(  # noqa: PLR0913
    data_dir: str | Path,
    arg_type: str | None,
    prefix: str | None,
    modifier: str | None,
    number: int | None,
    dataset: str,
) -> None:
    """Search for arguments with specific properties.

    Examples
    --------
    Find core arguments:
        $ glazing search args --type core

    Find location modifiers:
        $ glazing search args --modifier LOC

    Find continuation arguments:
        $ glazing search args --prefix C
    """
    try:
        search_engine = load_search_index(data_dir, [dataset])

        if dataset == "propbank":
            rolesets = search_engine.search_propbank_args(
                arg_type=arg_type,
                prefix=prefix,
                modifier=modifier,
                arg_number=str(number) if number is not None else None,
            )

            if not rolesets:
                console.print("[yellow]No matching rolesets found.[/yellow]")
                return

            table = Table(title="PropBank Rolesets with Matching Arguments")
            table.add_column("Roleset ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Arguments", style="white")

            for roleset in rolesets[:20]:
                arg_str = ", ".join(a.n for a in roleset.roles[:5])
                if len(roleset.roles) > 5:
                    arg_str += f" (+{len(roleset.roles) - 5} more)"
                table.add_row(
                    roleset.id,
                    roleset.name,
                    arg_str,
                )

            console.print(table)

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Argument search failed: {e}[/red]")
        sys.exit(1)


@search.command(name="relations")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files.",
)
@click.option(
    "--type",
    "relation_type",
    help="Relation type (e.g., hypernym, hyponym, antonym).",
    required=True,
)
@click.option("--dataset", default="wordnet", help="Dataset to search (default: wordnet).")
def search_relations(
    data_dir: str | Path,
    relation_type: str,
    dataset: str,
) -> None:
    """Search for synsets with specific relations.

    Examples
    --------
    Find hypernyms:
        $ glazing search relations --type hypernym

    Find antonyms:
        $ glazing search relations --type antonym
    """
    try:
        search_engine = load_search_index(data_dir, [dataset])

        if dataset == "wordnet":
            synsets = search_engine.search_wordnet_relations(relation_type)

            if not synsets:
                console.print("[yellow]No matching synsets found.[/yellow]")
                return

            table = Table(title=f"WordNet Synsets with {relation_type} Relations")
            table.add_column("Synset ID", style="cyan")
            table.add_column("Words", style="green")
            table.add_column("Definition", style="white", no_wrap=False)

            for synset in synsets[:20]:
                synset_id = f"{synset.offset:08d}{synset.ss_type}"
                words = ", ".join(w.lemma for w in synset.words[:3])
                if len(synset.words) > 3:
                    words += f" (+{len(synset.words) - 3})"
                definition = (
                    synset.gloss[:80] + "..."
                    if synset.gloss and len(synset.gloss) > 80
                    else synset.gloss or ""
                )
                table.add_row(synset_id, words, definition)

            console.print(table)

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Relation search failed: {e}[/red]")
        sys.exit(1)


@search.command(name="syntax")
@click.argument("pattern")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files.",
)
@click.option(
    "--dataset",
    type=click.Choice(["all", "verbnet", "propbank", "framenet"]),
    default="all",
    help="Dataset to search in.",
)
@click.option(
    "--limit",
    type=int,
    default=20,
    help="Maximum number of results to show.",
)
def search_syntax(
    pattern: str,
    data_dir: str | Path,
    dataset: str,
    limit: int,
) -> None:
    """Search for syntactic patterns with morphological features.

    Supports hierarchical matching and morphological features. General patterns
    match specific ones (e.g., "NP V PP" matches "NP V PP.instrument").

    Examples
    --------
    Find all patterns with NP V PP:
        $ glazing search syntax "NP V PP"

    Find patterns with specific PP type:
        $ glazing search syntax "NP V PP.instrument"

    Find patterns with specific preposition:
        $ glazing search syntax "NP V PP[with]"

    Find patterns with morphological features:
        $ glazing search syntax "NP V[ING] NP"

    Find patterns with wildcards:
        $ glazing search syntax "NP V NP *"
    """
    try:
        # Determine which datasets to load (skip wordnet for syntax search)
        datasets_to_load = ["verbnet", "propbank", "framenet"] if dataset == "all" else [dataset]

        # Load search index
        search_engine = load_search_index(data_dir, datasets_to_load)

        # Search by syntax
        results = search_engine.search_by_syntax(pattern)

        if not results:
            console.print(f"[yellow]No syntactic patterns matching '{pattern}' found.[/yellow]")
            return

        # Display results
        table = Table(title=f"Syntactic Patterns matching '{pattern}'")
        table.add_column("Dataset", style="cyan")
        table.add_column("Entity", style="green")
        table.add_column("Pattern", style="white")
        table.add_column("Confidence", style="yellow")

        for result in results[:limit]:
            table.add_row(
                result.dataset.upper(),
                result.id,
                result.description[:60] + "..."
                if len(result.description) > 60
                else result.description,
                f"{result.score:.2f}",
            )

        console.print(table)

        if len(results) > limit:
            console.print(f"\n[dim]Showing {limit} of {len(results)} results.[/dim]")

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Syntax search failed: {e}[/red]")
        sys.exit(1)


@search.command(name="elements")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=get_default_data_path,
    help="Directory containing converted JSON Lines files.",
)
@click.option(
    "--core-type",
    type=click.Choice(["Core", "Non-Core", "Extra-Thematic"]),
    help="Core type of frame elements.",
)
@click.option("--dataset", default="framenet", help="Dataset to search (default: framenet).")
def search_elements(
    data_dir: str | Path,
    core_type: str | None,
    dataset: str,
) -> None:
    """Search for frame elements with specific properties.

    Examples
    --------
    Find core elements:
        $ glazing search elements --core-type Core

    Find non-core elements:
        $ glazing search elements --core-type Non-Core
    """
    try:
        search_engine = load_search_index(data_dir, [dataset])

        if dataset == "framenet":
            frames = search_engine.search_framenet_elements(core_type=core_type)

            if not frames:
                console.print("[yellow]No matching frames found.[/yellow]")
                return

            table = Table(title=f"FrameNet Frames with {core_type or 'Matching'} Elements")
            table.add_column("Frame", style="cyan")
            table.add_column("Elements", style="green")
            table.add_column("Definition", style="white", no_wrap=False)

            for frame in frames[:20]:
                elem_str = ", ".join(fe.name for fe in frame.frame_elements[:5])
                if len(frame.frame_elements) > 5:
                    elem_str += f" (+{len(frame.frame_elements) - 5} more)"
                if frame.definition and len(frame.definition.plain_text) > 60:
                    definition = frame.definition.plain_text[:60] + "..."
                elif frame.definition:
                    definition = frame.definition.plain_text
                else:
                    definition = ""
                table.add_row(frame.name, elem_str, definition)

            console.print(table)

    except (ValueError, TypeError, RuntimeError) as e:
        console.print(f"[red]✗ Element search failed: {e}[/red]")
        sys.exit(1)
