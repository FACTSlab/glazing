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
from typing import cast

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
    SynsetKey,
    SynsetOffset,
    WordNetPOS,
    make_synset_key,
    normalize_pos,
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
    synsets : dict[SynsetKey, Synset]
        All loaded synsets indexed by canonical key (offset plus normalized
        POS, e.g. ``"00001740n"``). Offsets alone collide across parts of
        speech and so cannot key this mapping.
    lemma_index : dict[str, dict[WordNetPOS, list[SynsetKey]]]
        Index from lemmas to synset keys by POS. Adjective satellites appear
        under both ``"s"`` and ``"a"``.
    sense_index : dict[SenseKey, Sense]
        Index from sense keys to sense objects.
    exceptions : dict[WordNetPOS, dict[str, list[str]]]
        Morphological exceptions by POS.

    Methods
    -------
    load()
        Load all WordNet data from JSON Lines files.
    get_synset(offset, pos)
        Get a synset by its offset or canonical key.
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

        # Core data structures, keyed by canonical synset key (offset + POS)
        self.synsets: dict[SynsetKey, Synset] = {}
        self.lemma_index: dict[str, dict[WordNetPOS, list[SynsetKey]]] = defaultdict(dict)
        self.sense_index: dict[SenseKey, Sense] = {}
        self.exceptions: dict[WordNetPOS, dict[str, list[str]]] = {}

        # Relation indices for efficient traversal
        self.hypernym_index: dict[SynsetKey, list[SynsetKey]] = defaultdict(list)
        self.hyponym_index: dict[SynsetKey, list[SynsetKey]] = defaultdict(list)
        self.meronym_index: dict[SynsetKey, list[SynsetKey]] = defaultdict(list)
        self.holonym_index: dict[SynsetKey, list[SynsetKey]] = defaultdict(list)

        # Bare offset -> keys sharing it, for resolving unqualified lookups
        self._keys_by_offset: dict[SynsetOffset, list[SynsetKey]] = defaultdict(list)

        # File index for lazy loading (key -> byte position in file)
        self._synset_file_index: dict[SynsetKey, int] = {}

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
                    self.synsets[synset.key] = synset
                    self._keys_by_offset[synset.offset].append(synset.key)
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
                    ss_type = data.get("ss_type")
                    if offset and ss_type:
                        key = make_synset_key(offset, ss_type)
                        self._synset_file_index[key] = byte_pos
                        self._keys_by_offset[offset].append(key)
                except json.JSONDecodeError:
                    pass

    def _load_synset_lazy(self, key: SynsetKey) -> Synset | None:
        """Load a single synset on demand.

        Parameters
        ----------
        key : SynsetKey
            The canonical synset key to load.

        Returns
        -------
        Synset | None
            The loaded synset or None if not found.
        """
        if key in self.synsets:
            return self.synsets[key]

        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # Load from file using byte offset
        byte_pos = self._synset_file_index.get(key)
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
                    self._cache.put(key, synset)

                return synset
        except (json.JSONDecodeError, ValidationError):
            return None

    def _build_lemma_index(self) -> None:
        """Build lemma→synset index from loaded synset data.

        Adjective satellites are indexed under both ``"s"`` and the normalized
        ``"a"``, so a query for adjectives returns heads and satellites together
        while a query for ``"s"`` still isolates satellites.
        """
        for synset in self.synsets.values():
            positions: set[WordNetPOS] = {synset.ss_type, normalize_pos(synset.ss_type)}
            for word in synset.words:
                lemma = word.lemma.lower()
                for pos in positions:
                    if pos not in self.lemma_index[lemma]:
                        self.lemma_index[lemma][pos] = []
                    if synset.key not in self.lemma_index[lemma][pos]:
                        self.lemma_index[lemma][pos].append(synset.key)

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
        """Build relation indices for efficient traversal.

        Both sides are keyed canonically, so a pointer resolves to the synset it
        actually names rather than to whichever synset happens to share its
        offset in another part of speech.
        """
        for synset in self.synsets.values():
            source = synset.key
            for pointer in synset.pointers:
                target = pointer.target_key

                # Hypernym/hyponym relations
                if pointer.symbol == "@":
                    if target not in self.hypernym_index[source]:
                        self.hypernym_index[source].append(target)
                    if source not in self.hyponym_index[target]:
                        self.hyponym_index[target].append(source)
                elif pointer.symbol == "~":
                    if target not in self.hyponym_index[source]:
                        self.hyponym_index[source].append(target)
                    if source not in self.hypernym_index[target]:
                        self.hypernym_index[target].append(source)

                # Meronym/holonym relations
                elif pointer.symbol in ("%m", "%s", "%p"):
                    if target not in self.meronym_index[source]:
                        self.meronym_index[source].append(target)
                    if source not in self.holonym_index[target]:
                        self.holonym_index[target].append(source)
                elif pointer.symbol in ("#m", "#s", "#p"):
                    if target not in self.holonym_index[source]:
                        self.holonym_index[source].append(target)
                    if source not in self.meronym_index[target]:
                        self.meronym_index[target].append(source)

    def get_synset(
        self, offset: SynsetOffset | SynsetKey, pos: WordNetPOS | None = None
    ) -> Synset | None:
        """Get a synset by its offset or canonical key.

        Parameters
        ----------
        offset : SynsetOffset | SynsetKey
            An 8-digit synset offset, or a 9-character canonical key with the
            part of speech appended (e.g., ``"00001740n"``).
        pos : WordNetPOS | None, default=None
            Part of speech qualifying `offset`. Ignored when `offset` is
            already a canonical key.

        Returns
        -------
        Synset | None
            The synset, or None if not found or if a bare offset is ambiguous.

        Notes
        -----
        Offsets are byte offsets into a per-POS data file, so 330 of them name
        more than one synset in WordNet 3.1. An unqualified lookup on such an
        offset returns None rather than choosing arbitrarily; pass `pos` or a
        canonical key to disambiguate.

        Examples
        --------
        >>> synset = loader.get_synset("02084442")
        >>> print(synset.gloss)
        >>> loader.get_synset("00001740", "v")  # qualified
        >>> loader.get_synset("00001740v")  # equivalent
        """
        key = self._resolve_key(offset, pos)
        if key is None:
            return None
        if self.lazy:
            return self._load_synset_lazy(key)
        return self.synsets.get(key)

    def _resolve_key(
        self, offset: SynsetOffset | SynsetKey, pos: WordNetPOS | None = None
    ) -> SynsetKey | None:
        """Resolve an offset, key, or offset plus POS to a canonical key.

        Parameters
        ----------
        offset : SynsetOffset | SynsetKey
            Bare 8-digit offset or 9-character canonical key.
        pos : WordNetPOS | None, default=None
            Part of speech qualifying a bare offset.

        Returns
        -------
        SynsetKey | None
            The canonical key, or None if a bare offset is ambiguous.
        """
        if len(offset) == 9:
            return make_synset_key(offset[:8], cast(WordNetPOS, offset[8]))
        if pos is not None:
            return make_synset_key(offset, pos)
        keys = self._keys_by_offset.get(offset)
        if keys is None or len(keys) != 1:
            return None
        return keys[0]

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

        # Collect synsets by key. Satellites are indexed under both "s" and
        # "a", so track seen keys to avoid returning them twice.
        seen: set[SynsetKey] = set()
        for pos_tag in pos_tags:
            for key in self.lemma_index[lemma_lower].get(pos_tag, []):
                if key in seen:
                    continue
                seen.add(key)
                synset = self.get_synset(key)
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
        for key in self.hypernym_index.get(synset.key, []):
            hypernym = self.get_synset(key)
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
        for key in self.hyponym_index.get(synset.key, []):
            hyponym = self.get_synset(key)
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
        for key in self.meronym_index.get(synset.key, []):
            meronym = self.get_synset(key)
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
        for key in self.holonym_index.get(synset.key, []):
            holonym = self.get_synset(key)
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
