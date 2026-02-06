"""WordNet database loader with index building and caching.

This module provides functionality to load WordNet data from JSON Lines files,
build efficient indices for fast lookups, and construct relation graphs for
traversal operations.

Classes
-------
WordNetLoader
    Loads and indexes WordNet database from JSON Lines format with automatic loading.

Functions
---------
load_wordnet
    Load a complete WordNet database from JSON Lines files.

Examples
--------
>>> from glazing.wordnet.loader import WordNetLoader
>>> # Data loads automatically on initialization
>>> loader = WordNetLoader()
>>> synset = loader.get_synset("00001740")
>>> senses = loader.get_senses_by_lemma("dog", pos="n")
>>>
>>> # Or disable autoload for manual control
>>> loader = WordNetLoader(autoload=False)
>>> loader.load()  # Load manually when needed
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from glazing.initialize import get_default_data_path
from glazing.utils.cache import LRUCache
from glazing.wordnet.models import (
    ExceptionEntry,
    Sense,
    Synset,
)
from glazing.wordnet.types import (
    SenseKey,
    SynsetOffset,
    WordNetPOS,
)


class WordNetLoader:
    """Load and index WordNet database from JSON Lines format with automatic loading.

    This class provides efficient loading and indexing of WordNet data,
    including synsets, senses, and morphological exceptions. It builds
    multiple indices for fast lookups and supports lazy loading of
    large datasets. By default, data is loaded automatically on initialization.

    Parameters
    ----------
    data_path : Path | str | None, optional
        Path to the WordNet JSONL file (e.g., wordnet.jsonl).
        If None, uses default path from environment.
    lazy : bool, default=False
        If True, load synsets on demand rather than all at once.
    autoload : bool, default=True
        Whether to automatically load data on initialization.
        Only applies when lazy=False.
    cache_size : int, default=1000
        Number of synsets to cache when using lazy loading.

    Attributes
    ----------
    synsets : dict[SynsetOffset, Synset]
        All loaded synsets indexed by offset.
    lemma_index : dict[str, dict[WordNetPOS, list[SynsetOffset]]]
        Index from lemmas to synset offsets by POS.
    sense_index : dict[SenseKey, Sense]
        Index from sense keys to sense objects.
    exceptions : dict[WordNetPOS, dict[str, list[str]]]
        Morphological exceptions by POS.

    Methods
    -------
    load()
        Load all WordNet data from JSON Lines files.
    get_synset(offset)
        Get a synset by its offset.
    get_senses_by_lemma(lemma, pos)
        Get all senses for a lemma and optional POS.
    get_sense_by_key(sense_key)
        Get a sense by its unique sense key.

    Examples
    --------
    >>> # Automatic loading (default)
    >>> loader = WordNetLoader()
    >>> dog_synsets = loader.get_synsets_by_lemma("dog", "n")
    >>> for synset in dog_synsets:
    ...     print(f"{synset.offset}: {synset.gloss}")

    >>> # Manual loading
    >>> loader = WordNetLoader(autoload=False)
    >>> loader.load()
    >>> synsets = loader.synsets  # Now accessible
    """

    def __init__(
        self,
        data_path: Path | str | None = None,
        lazy: bool = False,
        autoload: bool = True,
        cache_size: int = 1000,
    ) -> None:
        """Initialize WordNet loader.

        Parameters
        ----------
        data_path : Path | str | None, optional
            Path to the WordNet JSONL file (e.g., wordnet.jsonl).
            If None, uses default path from environment.
        lazy : bool, default=False
            If True, load synsets on demand.
        autoload : bool, default=True
            Whether to automatically load data on initialization.
            Only applies when lazy=False.
        cache_size : int, default=1000
            Size of LRU cache for lazy loading.
        """
        if data_path is None:
            data_path = get_default_data_path("wordnet.jsonl")
        self.data_path = Path(data_path)
        self.lazy = lazy
        self.cache_size = cache_size

        # Core data structures
        self.synsets: dict[SynsetOffset, Synset] = {}
        self.lemma_index: dict[str, dict[WordNetPOS, list[SynsetOffset]]] = defaultdict(dict)
        self.sense_index: dict[SenseKey, Sense] = {}
        self.exceptions: dict[WordNetPOS, dict[str, list[str]]] = {}

        # Relation indices for efficient traversal
        self.hypernym_index: dict[SynsetOffset, list[SynsetOffset]] = defaultdict(list)
        self.hyponym_index: dict[SynsetOffset, list[SynsetOffset]] = defaultdict(list)
        self.meronym_index: dict[SynsetOffset, list[SynsetOffset]] = defaultdict(list)
        self.holonym_index: dict[SynsetOffset, list[SynsetOffset]] = defaultdict(list)

        # File index for lazy loading (offset -> byte position in file)
        self._synset_file_index: dict[SynsetOffset, int] = {}

        # Cache for lazy loading
        if lazy:
            self._cache: LRUCache[Synset] | None = LRUCache(cache_size)
        else:
            self._cache = None

        # Track loaded state
        self._loaded = False

        # Autoload data if requested and not lazy loading
        if autoload and not lazy:
            self.load()

    def load(self) -> None:
        """Load all WordNet data from JSON Lines files.

        This method loads synsets from the primary JSONL file, builds
        lemma and relation indices from loaded data, and optionally loads
        supplementary sense and exception data.

        Raises
        ------
        FileNotFoundError
            If the primary JSONL file doesn't exist.
        ValidationError
            If JSON data doesn't match expected schema.
        """
        if self._loaded:
            return

        # Load synsets from single JSONL file
        if self.lazy:
            self._build_file_index()
        else:
            self._load_all_synsets()

        # Build lemma index from loaded synsets
        if not self.lazy:
            self._build_lemma_index()

        # Load supplementary data if available
        self._load_sense_index()
        self._load_exceptions()

        # Build relation indices
        if not self.lazy:
            self._build_relation_indices()

        self._loaded = True

    def _load_all_synsets(self) -> None:
        """Load all synsets from single JSONL file."""
        if not self.data_path.exists():
            return

        with self.data_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    synset = Synset.model_validate(data)
                    self.synsets[synset.offset] = synset
                except (json.JSONDecodeError, ValidationError):
                    continue

    def _build_file_index(self) -> None:
        """Build byte-offset index for lazy loading from single JSONL file."""
        if not self.data_path.exists():
            return

        with self.data_path.open(encoding="utf-8") as f:
            while True:
                byte_pos = f.tell()
                line = f.readline()
                if not line:
                    break

                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    offset = data.get("offset")
                    if offset:
                        self._synset_file_index[offset] = byte_pos
                except json.JSONDecodeError:
                    pass

    def _load_synset_lazy(self, offset: SynsetOffset) -> Synset | None:
        """Load a single synset on demand.

        Parameters
        ----------
        offset : SynsetOffset
            The synset offset to load.

        Returns
        -------
        Synset | None
            The loaded synset or None if not found.
        """
        if offset in self.synsets:
            return self.synsets[offset]

        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(offset)
            if cached is not None:
                return cached

        # Load from file using byte offset
        byte_pos = self._synset_file_index.get(offset)
        if byte_pos is None:
            return None

        try:
            with self.data_path.open(encoding="utf-8") as f:
                f.seek(byte_pos)
                line = f.readline()
                data = json.loads(line)
                synset = Synset.model_validate(data)

                # Cache it
                if self._cache is not None:
                    self._cache.put(offset, synset)

                return synset
        except (json.JSONDecodeError, ValidationError):
            return None

    def _build_lemma_index(self) -> None:
        """Build lemma→synset index from loaded synset data."""
        for synset in self.synsets.values():
            pos = synset.ss_type
            for word in synset.words:
                lemma = word.lemma.lower()
                if pos not in self.lemma_index[lemma]:
                    self.lemma_index[lemma][pos] = []
                if synset.offset not in self.lemma_index[lemma][pos]:
                    self.lemma_index[lemma][pos].append(synset.offset)

    def _load_sense_index(self) -> None:
        """Load sense index from supplementary JSONL file."""
        sense_file = self.data_path.parent / "wordnet_senses.jsonl"
        if not sense_file.exists():
            return

        with sense_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    sense = Sense.model_validate(data)
                    self.sense_index[sense.sense_key] = sense
                except (json.JSONDecodeError, ValidationError):
                    continue

    def _load_exceptions(self) -> None:
        """Load morphological exceptions from supplementary JSONL file."""
        exc_file = self.data_path.parent / "wordnet_exceptions.jsonl"
        if not exc_file.exists():
            return

        with exc_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    entry = ExceptionEntry.model_validate(data)
                    # Determine POS from the base form by looking up in synsets
                    # Store under all POS for simplicity (exceptions file doesn't have POS)
                    pos = data.get("pos")
                    if pos and pos in ("n", "v", "a", "r", "s"):
                        if pos not in self.exceptions:
                            self.exceptions[pos] = {}
                        self.exceptions[pos][entry.inflected_form] = entry.base_forms
                except (json.JSONDecodeError, ValidationError):
                    continue

    def _build_relation_indices(self) -> None:
        """Build relation indices for efficient traversal."""
        for synset in self.synsets.values():
            for pointer in synset.pointers:
                # Hypernym/hyponym relations
                if pointer.symbol == "@":
                    if pointer.offset not in self.hypernym_index[synset.offset]:
                        self.hypernym_index[synset.offset].append(pointer.offset)
                    if synset.offset not in self.hyponym_index[pointer.offset]:
                        self.hyponym_index[pointer.offset].append(synset.offset)
                elif pointer.symbol == "~":
                    if pointer.offset not in self.hyponym_index[synset.offset]:
                        self.hyponym_index[synset.offset].append(pointer.offset)
                    if synset.offset not in self.hypernym_index[pointer.offset]:
                        self.hypernym_index[pointer.offset].append(synset.offset)

                # Meronym/holonym relations
                elif pointer.symbol in ("%m", "%s", "%p"):
                    if pointer.offset not in self.meronym_index[synset.offset]:
                        self.meronym_index[synset.offset].append(pointer.offset)
                    if synset.offset not in self.holonym_index[pointer.offset]:
                        self.holonym_index[pointer.offset].append(synset.offset)
                elif pointer.symbol in ("#m", "#s", "#p"):
                    if pointer.offset not in self.holonym_index[synset.offset]:
                        self.holonym_index[synset.offset].append(pointer.offset)
                    if synset.offset not in self.meronym_index[pointer.offset]:
                        self.meronym_index[pointer.offset].append(synset.offset)

    def get_synset(self, offset: SynsetOffset) -> Synset | None:
        """Get a synset by its offset.

        Parameters
        ----------
        offset : SynsetOffset
            The 8-digit synset offset.

        Returns
        -------
        Synset | None
            The synset or None if not found.

        Examples
        --------
        >>> synset = loader.get_synset("02084442")
        >>> print(synset.gloss)
        """
        if self.lazy:
            return self._load_synset_lazy(offset)
        return self.synsets.get(offset)

    def get_synsets_by_lemma(self, lemma: str, pos: WordNetPOS | None = None) -> list[Synset]:
        """Get all synsets containing a lemma.

        Parameters
        ----------
        lemma : str
            The word lemma to search for.
        pos : WordNetPOS | None, default=None
            Part of speech filter. If None, returns all POS.

        Returns
        -------
        list[Synset]
            List of synsets containing the lemma.

        Examples
        --------
        >>> synsets = loader.get_synsets_by_lemma("run", "v")
        >>> for synset in synsets:
        ...     print(synset.gloss)
        """
        synsets: list[Synset] = []
        lemma_lower = lemma.lower()

        if lemma_lower not in self.lemma_index:
            return synsets

        # Get POS tags to search
        pos_tags: list[WordNetPOS]
        if pos:
            pos_tags = [pos] if pos in self.lemma_index[lemma_lower] else []
        else:
            pos_tags = list(self.lemma_index[lemma_lower].keys())

        # Collect synsets from offset lists
        for pos_tag in pos_tags:
            for offset in self.lemma_index[lemma_lower].get(pos_tag, []):
                synset = self.get_synset(offset)
                if synset:
                    synsets.append(synset)

        return synsets

    def get_sense_by_key(self, sense_key: SenseKey) -> Sense | None:
        """Get a sense by its unique sense key.

        Parameters
        ----------
        sense_key : SenseKey
            The unique sense key.

        Returns
        -------
        Sense | None
            The sense or None if not found.

        Examples
        --------
        >>> sense = loader.get_sense_by_key("dog%1:05:00::")
        >>> print(sense.synset_offset)
        """
        return self.sense_index.get(sense_key)

    def get_senses_by_lemma(self, lemma: str, pos: WordNetPOS | None = None) -> list[Sense]:
        """Get all senses for a lemma.

        Parameters
        ----------
        lemma : str
            The word lemma to search for.
        pos : WordNetPOS | None, default=None
            Part of speech filter.

        Returns
        -------
        list[Sense]
            List of senses for the lemma, sorted by sense number.

        Examples
        --------
        >>> senses = loader.get_senses_by_lemma("run", "v")
        >>> for sense in senses:
        ...     print(f"{sense.sense_key}: {sense.sense_number}")
        """
        senses = []

        for sense in self.sense_index.values():
            if sense.lemma == lemma and (pos is None or sense.ss_type == pos):
                senses.append(sense)

        # Sort by sense number (frequency order)
        senses.sort(key=lambda s: s.sense_number)

        return senses

    def get_hypernyms(self, synset: Synset) -> list[Synset]:
        """Get direct hypernyms of a synset.

        Parameters
        ----------
        synset : Synset
            The synset to get hypernyms for.

        Returns
        -------
        list[Synset]
            List of hypernym synsets.
        """
        hypernyms = []
        for offset in self.hypernym_index.get(synset.offset, []):
            hypernym = self.get_synset(offset)
            if hypernym:
                hypernyms.append(hypernym)
        return hypernyms

    def get_hyponyms(self, synset: Synset) -> list[Synset]:
        """Get direct hyponyms of a synset.

        Parameters
        ----------
        synset : Synset
            The synset to get hyponyms for.

        Returns
        -------
        list[Synset]
            List of hyponym synsets.
        """
        hyponyms = []
        for offset in self.hyponym_index.get(synset.offset, []):
            hyponym = self.get_synset(offset)
            if hyponym:
                hyponyms.append(hyponym)
        return hyponyms

    def get_meronyms(self, synset: Synset) -> list[Synset]:
        """Get all meronyms (parts) of a synset.

        Parameters
        ----------
        synset : Synset
            The synset to get meronyms for.

        Returns
        -------
        list[Synset]
            List of meronym synsets.
        """
        meronyms = []
        for offset in self.meronym_index.get(synset.offset, []):
            meronym = self.get_synset(offset)
            if meronym:
                meronyms.append(meronym)
        return meronyms

    def get_holonyms(self, synset: Synset) -> list[Synset]:
        """Get all holonyms (wholes) of a synset.

        Parameters
        ----------
        synset : Synset
            The synset to get holonyms for.

        Returns
        -------
        list[Synset]
            List of holonym synsets.
        """
        holonyms = []
        for offset in self.holonym_index.get(synset.offset, []):
            holonym = self.get_synset(offset)
            if holonym:
                holonyms.append(holonym)
        return holonyms

    def get_exceptions(self, pos: WordNetPOS) -> dict[str, list[str]]:
        """Get morphological exceptions for a POS.

        Parameters
        ----------
        pos : WordNetPOS
            The part of speech.

        Returns
        -------
        dict[str, list[str]]
            Mapping from inflected forms to base forms.
        """
        return self.exceptions.get(pos, {})


def load_wordnet(
    data_path: Path | str, lazy: bool = False, cache_size: int = 1000
) -> WordNetLoader:
    """Load a WordNet database from JSON Lines files.

    Parameters
    ----------
    data_path : Path | str
        Path to the WordNet JSONL file (e.g., wordnet.jsonl).
    lazy : bool, default=False
        If True, load synsets on demand.
    cache_size : int, default=1000
        Size of LRU cache for lazy loading.

    Returns
    -------
    WordNetLoader
        Loaded WordNet database.

    Examples
    --------
    >>> wn = load_wordnet("data/wordnet.jsonl")
    >>> dog = wn.get_synsets_by_lemma("dog", "n")[0]
    >>> print(dog.gloss)
    """
    loader = WordNetLoader(data_path, lazy=lazy, cache_size=cache_size, autoload=False)
    loader.load()
    return loader
