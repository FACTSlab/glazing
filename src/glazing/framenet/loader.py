"""FrameNet JSON Lines data loader.

This module provides functionality to load FrameNet data from JSON Lines files
with validation, lazy loading for large datasets, and frame index building.

Classes
-------
FrameNetLoader
    Load FrameNet data from JSON Lines format with automatic loading.
FrameIndex
    In-memory index for fast frame lookups.

Functions
---------
load_frames
    Load Frame models from JSON Lines file.
load_lexical_units
    Load LexicalUnit models from JSON Lines file.
build_frame_index
    Build searchable index from frames data.

Examples
--------
>>> from glazing.framenet.loader import FrameNetLoader
>>> # Data loads automatically on initialization
>>> loader = FrameNetLoader()
>>> frames = loader.frames  # Access loaded frames via property
>>>
>>> # Or disable autoload for manual control
>>> loader = FrameNetLoader(autoload=False)
>>> frames = loader.load()  # Load manually when needed
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from glazing.framenet.models import Frame, LexicalUnit, SemanticType, Sentence
from glazing.framenet.types import FrameID
from glazing.initialize import get_default_data_path


class FrameIndex:
    """In-memory index for fast FrameNet frame lookups.

    Provides efficient lookup methods for frames by various criteria
    including name, ID, frame elements, and lexical units.

    Parameters
    ----------
    frames : list[Frame]
        List of Frame models to index.

    Attributes
    ----------
    _by_id : dict[FrameID, Frame]
        Index frames by ID.
    _by_name : dict[str, Frame]
        Index frames by name (case-insensitive).
    _by_fe_name : dict[str, list[Frame]]
        Index frames by frame element names.
    _by_lu_name : dict[str, list[Frame]]
        Index frames by lexical unit names.
    _by_definition : dict[str, list[Frame]]
        Index frames by words in definition.

    Methods
    -------
    get_frame_by_id(frame_id)
        Get frame by ID.
    get_frame_by_name(name)
        Get frame by name (case-insensitive).
    find_frames_with_fe(fe_name)
        Find frames containing a specific frame element.
    find_frames_with_lu(lu_name)
        Find frames containing a specific lexical unit.
    search_definitions(query)
        Search frames by definition text.
    get_all_frame_names()
        Get sorted list of all frame names.
    get_statistics()
        Get index statistics.
    """

    def __init__(self, frames: list[Frame] | None = None) -> None:
        """Initialize frame index.

        Parameters
        ----------
        frames : list[Frame] | None, default=None
            Frames to index. If None, creates empty index.
        """
        self._by_id: dict[FrameID, Frame] = {}
        self._by_name: dict[str, Frame] = {}
        self._by_fe_name: dict[str, list[Frame]] = defaultdict(list)
        self._by_lu_name: dict[str, list[Frame]] = defaultdict(list)
        self._by_definition: dict[str, list[Frame]] = defaultdict(list)

        if frames:
            self.add_frames(frames)

    def add_frames(self, frames: list[Frame]) -> None:
        """Add frames to the index.

        Parameters
        ----------
        frames : list[Frame]
            Frames to add to index.
        """
        for frame in frames:
            self.add_frame(frame)

    def add_frame(self, frame: Frame) -> None:
        """Add a single frame to the index.

        Parameters
        ----------
        frame : Frame
            Frame to add to index.
        """
        # Index by ID and name
        self._by_id[frame.id] = frame
        self._by_name[frame.name.lower()] = frame

        # Index by frame elements
        for fe in frame.frame_elements:
            self._by_fe_name[fe.name.lower()].append(frame)

        # Index by lexical units
        for lu in frame.lexical_units:
            lu_lemma = lu.name.split(".")[0].lower()  # Extract lemma from "word.pos"
            self._by_lu_name[lu_lemma].append(frame)

        # Index by definition words
        if frame.definition and frame.definition.plain_text:
            words = frame.definition.plain_text.lower().split()
            for word in words:
                # Clean word of punctuation
                clean_word = "".join(c for c in word if c.isalnum())
                if clean_word and len(clean_word) > 2:  # Skip short words
                    self._by_definition[clean_word].append(frame)

    def get_frame_by_id(self, frame_id: FrameID) -> Frame | None:
        """Get frame by ID.

        Parameters
        ----------
        frame_id : FrameID
            Frame ID to look up.

        Returns
        -------
        Frame | None
            Frame if found, None otherwise.
        """
        return self._by_id.get(frame_id)

    def get_frame_by_name(self, name: str) -> Frame | None:
        """Get frame by name (case-insensitive).

        Parameters
        ----------
        name : str
            Frame name to look up.

        Returns
        -------
        Frame | None
            Frame if found, None otherwise.
        """
        return self._by_name.get(name.lower())

    def find_frames_with_fe(self, fe_name: str) -> list[Frame]:
        """Find frames containing a specific frame element.

        Parameters
        ----------
        fe_name : str
            Frame element name to search for.

        Returns
        -------
        list[Frame]
            Frames containing the frame element.
        """
        return self._by_fe_name.get(fe_name.lower(), [])

    def find_frames_with_lu(self, lu_name: str) -> list[Frame]:
        """Find frames containing a specific lexical unit.

        Parameters
        ----------
        lu_name : str
            Lexical unit name (lemma) to search for.

        Returns
        -------
        list[Frame]
            Frames containing the lexical unit.
        """
        return self._by_lu_name.get(lu_name.lower(), [])

    def search_definitions(self, query: str) -> list[Frame]:
        """Search frames by definition text.

        Parameters
        ----------
        query : str
            Search query (single word or phrase).

        Returns
        -------
        list[Frame]
            Frames with definitions containing the query.
        """
        if not query:
            return []

        query = query.lower().strip()

        if " " not in query:
            # Single word search
            return self._by_definition.get(query, [])
        # Multi-word search - find frames containing all words
        words = query.split()
        if not words:
            return []

        # Start with frame IDs containing the first word
        result_frame_ids = {frame.id for frame in self._by_definition.get(words[0], [])}

        # Intersect with frame IDs containing each additional word
        for word in words[1:]:
            word_frame_ids = {frame.id for frame in self._by_definition.get(word, [])}
            result_frame_ids &= word_frame_ids

        # Convert back to Frame objects
        return [self._by_id[frame_id] for frame_id in result_frame_ids if frame_id in self._by_id]

    def get_all_frame_names(self) -> list[str]:
        """Get sorted list of all frame names.

        Returns
        -------
        list[str]
            Sorted frame names.
        """
        return sorted(frame.name for frame in self._by_id.values())

    def get_statistics(self) -> dict[str, int]:
        """Get index statistics.

        Returns
        -------
        dict[str, int]
            Dictionary with index statistics.
        """
        total_fes = sum(len(frame.frame_elements) for frame in self._by_id.values())
        total_lus = sum(len(frame.lexical_units) for frame in self._by_id.values())

        return {
            "total_frames": len(self._by_id),
            "total_frame_elements": total_fes,
            "total_lexical_units": total_lus,
            "unique_fe_names": len(self._by_fe_name),
            "unique_lu_lemmas": len(self._by_lu_name),
            "indexed_definition_words": len(self._by_definition),
        }


class FrameNetLoader:
    """Load FrameNet data from JSON Lines format with automatic loading.

    Handles loading of Frame and LexicalUnit models from JSON Lines files
    with validation, lazy loading, and index building capabilities.
    By default, data is loaded automatically on initialization.

    Parameters
    ----------
    data_path : Path | str | None, optional
        Path to FrameNet JSON Lines file. If None, uses default path.
    lazy : bool, default=False
        Whether to use lazy loading.
    autoload : bool, default=True
        Whether to automatically load data on initialization.
        Only applies when lazy=False.
    cache_size : int, default=1000
        Number of items to cache when using lazy loading.

    Attributes
    ----------
    frames : list[Frame]
        Property that returns loaded frames, loading them if needed.

    Methods
    -------
    load()
        Load all frames from the data file.
    load_frames(filepath, skip_errors)
        Load Frame models from JSON Lines file.
    load_lexical_units(filepath, skip_errors)
        Load LexicalUnit models from JSON Lines file.
    load_semantic_types(filepath, skip_errors)
        Load SemanticType models from JSON Lines file.
    build_frame_index(frames)
        Build searchable index from frames data.
    load_and_index_frames(filepath, skip_errors)
        Load frames and build index in one step.
    validate_frame_data(filepath)
        Validate frame data file without loading into memory.

    Examples
    --------
    >>> # Automatic loading (default)
    >>> loader = FrameNetLoader()
    >>> frames = loader.frames  # Already loaded
    >>>
    >>> # Manual loading
    >>> loader = FrameNetLoader(autoload=False)
    >>> frames = loader.load()
    """

    def __init__(
        self,
        data_path: Path | str | None = None,
        lazy: bool = False,
        autoload: bool = True,
        cache_size: int = 1000,
    ) -> None:
        """Initialize FrameNet loader.

        Parameters
        ----------
        data_path : Path | str | None, optional
            Path to FrameNet JSON Lines file.
            If None, uses default path from environment.
        lazy : bool, default=False
            Whether to use lazy loading.
        autoload : bool, default=True
            Whether to automatically load data on initialization.
            Only applies when lazy=False.
        cache_size : int, default=1000
            Number of items to cache when using lazy loading.
        """
        if data_path is None:
            data_path = get_default_data_path("framenet.jsonl")
        self.data_path = Path(data_path)
        self.lazy = lazy
        self.cache_size = cache_size
        self._frames: list[Frame] | None = None
        self._index: FrameIndex | None = None

        # Autoload data if requested and not lazy loading
        if autoload and not lazy:
            self.load()

    def load(self) -> list[Frame]:
        """Load all frames from the data file.

        Returns
        -------
        list[Frame]
            List of loaded Frame models.
        """
        if self._frames is None:
            self._frames = self.load_frames(self.data_path)
        return self._frames

    @property
    def frames(self) -> list[Frame]:
        """Get loaded frames.

        Returns
        -------
        list[Frame]
            List of loaded Frame models. Loads automatically if not yet loaded.
        """
        if self._frames is None:
            self.load()
        return self._frames if self._frames is not None else []

    def load_frames(
        self, filepath: Path | str | None = None, skip_errors: bool = False
    ) -> list[Frame]:
        """Load Frame models from JSON Lines file.

        Parameters
        ----------
        filepath : Path | str | None, optional
            Path to JSON Lines file containing Frame data.
            If None, uses the default data path from initialization.
        skip_errors : bool, default=False
            If True, skip invalid lines rather than raising errors.

        Returns
        -------
        list[Frame]
            List of loaded Frame models.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If skip_errors=False and a line fails validation.
        """
        filepath = self.data_path if filepath is None else Path(filepath)
        if not filepath.exists():
            msg = f"FrameNet data file not found: {filepath}"
            raise FileNotFoundError(msg)

        frames = []
        for frame in Frame.from_json_lines_file(filepath, skip_errors=skip_errors):
            frames.append(frame)

        return frames

    def load_lexical_units(
        self, filepath: Path | str, skip_errors: bool = False
    ) -> list[LexicalUnit]:
        """Load LexicalUnit models from JSON Lines file.

        Parameters
        ----------
        filepath : Path | str
            Path to JSON Lines file containing LexicalUnit data.
        skip_errors : bool, default=False
            If True, skip invalid lines rather than raising errors.

        Returns
        -------
        list[LexicalUnit]
            List of loaded LexicalUnit models.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If skip_errors=False and a line fails validation.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"FrameNet LU data file not found: {filepath}"
            raise FileNotFoundError(msg)

        lexical_units = []
        for lu in LexicalUnit.from_json_lines_file(filepath, skip_errors=skip_errors):
            lexical_units.append(lu)

        return lexical_units

    def load_semantic_types(
        self, filepath: Path | str | None = None, skip_errors: bool = False
    ) -> list[SemanticType]:
        """Load SemanticType models from JSON Lines file.

        Parameters
        ----------
        filepath : Path | str | None, optional
            Path to JSON Lines file containing SemanticType data.
            If None, looks for ``framenet_semtypes.jsonl`` alongside the primary data file.
        skip_errors : bool, default=False
            If True, skip invalid lines rather than raising errors.

        Returns
        -------
        list[SemanticType]
            List of loaded SemanticType models.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If skip_errors=False and a line fails validation.
        """
        if filepath is None:
            filepath = self.data_path.parent / "framenet_semtypes.jsonl"
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"FrameNet semantic types file not found: {filepath}"
            raise FileNotFoundError(msg)

        sem_types = []
        for sem_type in SemanticType.from_json_lines_file(filepath, skip_errors=skip_errors):
            sem_types.append(sem_type)

        return sem_types

    def load_fulltext(
        self, filepath: Path | str | None = None, skip_errors: bool = False
    ) -> list[Sentence]:
        """Load Sentence models from fulltext JSON Lines file.

        Parameters
        ----------
        filepath : Path | str | None, optional
            Path to JSON Lines file containing Sentence data.
            If None, looks for ``framenet_fulltext.jsonl`` alongside the primary data file.
        skip_errors : bool, default=False
            If True, skip invalid lines rather than raising errors.

        Returns
        -------
        list[Sentence]
            List of loaded Sentence models.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        """
        if filepath is None:
            filepath = self.data_path.parent / "framenet_fulltext.jsonl"
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"FrameNet fulltext file not found: {filepath}"
            raise FileNotFoundError(msg)

        sentences = []
        for sentence in Sentence.from_json_lines_file(filepath, skip_errors=skip_errors):
            sentences.append(sentence)

        return sentences

    def build_frame_index(self, frames: list[Frame]) -> FrameIndex:
        """Build searchable index from frames data.

        Parameters
        ----------
        frames : list[Frame]
            Frames to index.

        Returns
        -------
        FrameIndex
            Searchable frame index.
        """
        return FrameIndex(frames)

    def load_and_index_frames(self, filepath: Path | str, skip_errors: bool = False) -> FrameIndex:
        """Load frames and build index in one step.

        Parameters
        ----------
        filepath : Path | str
            Path to JSON Lines file containing Frame data.
        skip_errors : bool, default=False
            If True, skip invalid lines rather than raising errors.

        Returns
        -------
        FrameIndex
            Loaded and indexed frames.
        """
        frames = self.load_frames(filepath, skip_errors)
        return self.build_frame_index(frames)

    def validate_frame_data(self, filepath: Path | str) -> dict[str, str | int | float | list[str]]:
        """Validate frame data file without loading into memory.

        Parameters
        ----------
        filepath : Path | str
            Path to JSON Lines file to validate.

        Returns
        -------
        dict[str, str | int | float | list[str]]
            Validation results including error counts and statistics.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"FrameNet data file not found: {filepath}"
            raise FileNotFoundError(msg)

        total_lines = 0
        valid_lines = 0
        errors = []

        # Use generator to avoid loading all data into memory
        try:
            for _frame in Frame.from_json_lines_file(filepath, skip_errors=True):
                total_lines += 1
                valid_lines += 1
        except (ValueError, TypeError, AttributeError) as e:
            errors.append(str(e))

        # Count total lines including invalid ones
        with filepath.open("r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f if line.strip())

        error_count = total_lines - valid_lines

        return {
            "filepath": str(filepath),
            "total_lines": total_lines,
            "valid_lines": valid_lines,
            "error_count": error_count,
            "validation_errors": errors,
            "success_rate": valid_lines / total_lines if total_lines > 0 else 0.0,
        }


# Convenience functions


def load_frames(filepath: Path | str, skip_errors: bool = False) -> list[Frame]:
    """Load Frame models from JSON Lines file.

    Parameters
    ----------
    filepath : Path | str
        Path to JSON Lines file.
    skip_errors : bool, default=False
        Whether to skip invalid lines.

    Returns
    -------
    list[Frame]
        Loaded Frame models.
    """
    loader = FrameNetLoader(autoload=False)
    return loader.load_frames(filepath, skip_errors)


def load_lexical_units(filepath: Path | str, skip_errors: bool = False) -> list[LexicalUnit]:
    """Load LexicalUnit models from JSON Lines file.

    Parameters
    ----------
    filepath : Path | str
        Path to JSON Lines file.
    skip_errors : bool, default=False
        Whether to skip invalid lines.

    Returns
    -------
    list[LexicalUnit]
        Loaded LexicalUnit models.
    """
    loader = FrameNetLoader(autoload=False)
    return loader.load_lexical_units(filepath, skip_errors)


def build_frame_index(frames: list[Frame]) -> FrameIndex:
    """Build searchable index from frames data.

    Parameters
    ----------
    frames : list[Frame]
        Frames to index.

    Returns
    -------
    FrameIndex
        Searchable frame index.
    """
    return FrameIndex(frames)


def load_and_index_frames(filepath: Path | str, skip_errors: bool = False) -> FrameIndex:
    """Load frames and build index in one step.

    Parameters
    ----------
    filepath : Path | str
        Path to JSON Lines file.
    skip_errors : bool, default=False
        Whether to skip invalid lines.

    Returns
    -------
    FrameIndex
        Loaded and indexed frames.
    """
    loader = FrameNetLoader(autoload=False)
    return loader.load_and_index_frames(filepath, skip_errors)
