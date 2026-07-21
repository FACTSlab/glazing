"""WordNet search functionality.

This module provides search capabilities for WordNet data,
including synset searches by lemma, offset, sense key, and pattern-based
searches with domain-specific filtering.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

from glazing.syntax.models import UnifiedSyntaxPattern
from glazing.syntax.parser import SyntaxParser
from glazing.wordnet.models import Sense, Synset
from glazing.wordnet.symbol_parser import filter_by_relation_type
from glazing.wordnet.types import (
    LexFileName,
    SenseKey,
    SynsetKey,
    SynsetOffset,
    VerbFrameNumber,
    WordNetPOS,
    make_synset_key,
    normalize_pos,
)


class WordNetSearch:
    """Search interface for WordNet data.

    Provides methods for finding synsets by various criteria
    including lemma, offset, sense key, pattern matching, and
    domain-specific searches.

    Parameters
    ----------
    synsets : list[Synset] | None
        Initial synsets to index. If None, creates empty search.
    senses : list[Sense] | None
        Initial senses to index. If None, creates empty search.

    Attributes
    ----------
    _synsets : dict[SynsetKey, Synset]
        Mapping from canonical synset key to synset object.
    _synsets_by_lemma : dict[str, dict[WordNetPOS, set[SynsetKey]]]
        Mapping from lemma and POS to synset keys. Adjective satellites appear
        under both ``"s"`` and ``"a"``.
    _senses : dict[SenseKey, Sense]
        Mapping from sense key to sense object.
    _synsets_by_domain : dict[LexFileName, set[SynsetKey]]
        Mapping from lexical file name to synset keys.

    Methods
    -------
    add_synset(synset)
        Add a synset to the search index.
    add_sense(sense)
        Add a sense to the search index.
    by_offset(offset, pos)
        Find synset by offset or canonical key, optionally filtered by POS.
    by_lemma(lemma, pos)
        Find synsets containing a lemma.
    by_sense_key(sense_key)
        Find synset by sense key.
    by_pattern(pattern, pos, case_sensitive)
        Find synsets matching a pattern.
    by_domain(domain)
        Find synsets in a specific domain.
    by_gloss_pattern(pattern, pos, case_sensitive)
        Find synsets with glosses matching a pattern.
    get_lemma_senses(lemma, pos)
        Get all senses for a lemma.

    Examples
    --------
    >>> search = WordNetSearch()
    >>> search.add_synset(dog_synset)
    >>> synsets = search.by_lemma("dog", "n")
    >>> synset = search.by_offset("02084442", "n")
    """

    def __init__(
        self, synsets: list[Synset] | None = None, senses: list[Sense] | None = None
    ) -> None:
        """Initialize WordNet search with optional initial data."""
        self._synsets: dict[SynsetKey, Synset] = {}
        self._synsets_by_lemma: dict[str, dict[WordNetPOS, set[SynsetKey]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._senses: dict[SenseKey, Sense] = {}
        self._synsets_by_domain: dict[LexFileName, set[SynsetKey]] = defaultdict(set)
        self._synsets_by_pos: dict[WordNetPOS, set[SynsetKey]] = defaultdict(set)

        if synsets:
            for synset in synsets:
                self.add_synset(synset)

        if senses:
            for sense in senses:
                self.add_sense(sense)

    def add_synset(self, synset: Synset) -> None:
        """Add a synset to the search index.

        Parameters
        ----------
        synset : Synset
            Synset to add to index.

        Raises
        ------
        ValueError
            If a synset with the same canonical key already exists.
        """
        if synset.key in self._synsets:
            msg = f"Synset with key {synset.key} already exists"
            raise ValueError(msg)

        self._synsets[synset.key] = synset

        # Satellites index under both "s" and "a" so that either query finds
        # them, while ss_type on the record still reports "s".
        positions: set[WordNetPOS] = {synset.ss_type, normalize_pos(synset.ss_type)}
        for pos in positions:
            self._synsets_by_pos[pos].add(synset.key)
            for word in synset.words:
                self._synsets_by_lemma[word.lemma][pos].add(synset.key)

        # Index by domain (lexical file)
        self._synsets_by_domain[synset.lex_filename].add(synset.key)

    def add_sense(self, sense: Sense) -> None:
        """Add a sense to the search index.

        Parameters
        ----------
        sense : Sense
            Sense to add to index.

        Raises
        ------
        ValueError
            If sense with same key already exists.
        """
        if sense.sense_key in self._senses:
            msg = f"Sense with key {sense.sense_key} already exists"
            raise ValueError(msg)

        self._senses[sense.sense_key] = sense

    def by_offset(
        self, offset: SynsetOffset | SynsetKey, pos: WordNetPOS | None = None
    ) -> Synset | None:
        """Find synset by offset or canonical key, optionally filtered by POS.

        Parameters
        ----------
        offset : SynsetOffset | SynsetKey
            Synset offset (8-digit string) or canonical key (9 characters).
        pos : WordNetPOS | None
            Part of speech to filter by. Passing ``"a"`` matches adjective
            heads and satellites alike; ``"s"`` matches satellites only.

        Returns
        -------
        Synset | None
            Synset if found, None otherwise.

        Notes
        -----
        An unqualified offset shared by synsets in more than one part of speech
        is ambiguous and yields None; pass `pos` or a canonical key instead.
        """
        if len(offset) == 9:
            key = make_synset_key(offset[:8], cast(WordNetPOS, offset[8]))
        elif pos is not None:
            key = make_synset_key(offset, pos)
        else:
            candidates = [k for k in self._synsets if k[:8] == offset]
            if len(candidates) != 1:
                return None
            key = candidates[0]

        synset = self._synsets.get(key)
        # "s" is not encoded in the key, so filter satellites explicitly.
        if synset and pos == "s" and synset.ss_type != "s":
            return None
        return synset

    def by_lemma(self, lemma: str, pos: WordNetPOS | None = None) -> list[Synset]:
        """Find synsets containing a lemma.

        Parameters
        ----------
        lemma : str
            Lemma to search for (lowercase, underscores for spaces).
        pos : WordNetPOS | None
            Part of speech to filter by.

        Returns
        -------
        list[Synset]
            Synsets containing the lemma.
        """
        # Normalize lemma
        lemma = lemma.lower().replace(" ", "_")

        if pos is not None:
            # Search specific POS
            keys = self._synsets_by_lemma.get(lemma, {}).get(pos, set())
        else:
            # Search all POS. Satellites are indexed under both "s" and "a",
            # so take the union rather than concatenating per-POS results.
            keys = set()
            for pos_keys in self._synsets_by_lemma.get(lemma, {}).values():
                keys |= pos_keys

        synsets = [self._synsets[key] for key in keys]
        return sorted(synsets, key=lambda s: s.offset)

    def by_sense_key(self, sense_key: SenseKey) -> Synset | None:
        """Find synset by sense key.

        Parameters
        ----------
        sense_key : SenseKey
            Sense key to look up.

        Returns
        -------
        Synset | None
            Synset containing the sense, None if not found.
        """
        sense = self._senses.get(sense_key)
        if sense:
            return self._synsets.get(make_synset_key(sense.synset_offset, sense.ss_type))
        return None

    def by_pattern(
        self, pattern: str, pos: WordNetPOS | None = None, case_sensitive: bool = False
    ) -> list[Synset]:
        """Find synsets matching a pattern.

        Parameters
        ----------
        pattern : str
            Regular expression pattern to match against lemmas.
        pos : WordNetPOS | None
            Part of speech to filter by.
        case_sensitive : bool
            Whether search should be case-sensitive.

        Returns
        -------
        list[Synset]
            Synsets with lemmas matching the pattern.

        Raises
        ------
        re.error
            If pattern is invalid regular expression.
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        matching_keys = set()

        # Determine which synsets to search
        if pos is not None:
            synsets_to_search = [self._synsets[key] for key in self._synsets_by_pos[pos]]
        else:
            synsets_to_search = list(self._synsets.values())

        for synset in synsets_to_search:
            for word in synset.words:
                if regex.search(word.lemma):
                    matching_keys.add(synset.key)
                    break

        synsets = [self._synsets[key] for key in matching_keys]
        return sorted(synsets, key=lambda s: s.offset)

    def by_domain(self, domain: LexFileName) -> list[Synset]:
        """Find synsets in a specific domain.

        Parameters
        ----------
        domain : LexFileName
            Lexical file name (domain).

        Returns
        -------
        list[Synset]
            Synsets in the specified domain.
        """
        keys = self._synsets_by_domain.get(domain, set())
        synsets = [self._synsets[key] for key in keys]
        return sorted(synsets, key=lambda s: s.offset)

    def by_gloss_pattern(
        self, pattern: str, pos: WordNetPOS | None = None, case_sensitive: bool = False
    ) -> list[Synset]:
        """Find synsets with glosses matching a pattern.

        Parameters
        ----------
        pattern : str
            Regular expression pattern to match against glosses.
        pos : WordNetPOS | None
            Part of speech to filter by.
        case_sensitive : bool
            Whether search should be case-sensitive.

        Returns
        -------
        list[Synset]
            Synsets with glosses matching the pattern.

        Raises
        ------
        re.error
            If pattern is invalid regular expression.
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        matching_synsets = []

        # Determine which synsets to search
        if pos is not None:
            synsets_to_search = [self._synsets[key] for key in self._synsets_by_pos[pos]]
        else:
            synsets_to_search = list(self._synsets.values())

        for synset in synsets_to_search:
            if regex.search(synset.gloss):
                matching_synsets.append(synset)

        return sorted(matching_synsets, key=lambda s: s.offset)

    def get_lemma_senses(self, lemma: str, pos: WordNetPOS | None = None) -> list[Sense]:
        """Get all senses for a lemma.

        Parameters
        ----------
        lemma : str
            Lemma to get senses for.
        pos : WordNetPOS | None
            Part of speech to filter by.

        Returns
        -------
        list[Sense]
            Senses for the lemma, ordered by sense number.
        """
        # Normalize lemma
        lemma = lemma.lower().replace(" ", "_")

        matching_senses = []
        for sense in self._senses.values():
            if sense.lemma == lemma and (pos is None or sense.ss_type == pos):
                matching_senses.append(sense)

        return sorted(matching_senses, key=lambda s: s.sense_number)

    def get_all_lemmas(self, pos: WordNetPOS | None = None) -> list[str]:
        """Get all unique lemmas.

        Parameters
        ----------
        pos : WordNetPOS | None
            Part of speech to filter by.

        Returns
        -------
        list[str]
            Sorted list of unique lemmas.
        """
        if pos is not None:
            lemmas = set()
            for lemma, pos_dict in self._synsets_by_lemma.items():
                if pos in pos_dict:
                    lemmas.add(lemma)
            return sorted(lemmas)
        return sorted(self._synsets_by_lemma.keys())

    def get_all_domains(self) -> list[LexFileName]:
        """Get all lexical file names (domains).

        Returns
        -------
        list[LexFileName]
            Sorted list of domains.
        """
        return sorted(self._synsets_by_domain.keys())

    def get_all_synsets(self) -> list[Synset]:
        """Get all synsets in the search index.

        Returns
        -------
        list[Synset]
            All synsets sorted by offset.
        """
        return sorted(self._synsets.values(), key=lambda s: s.offset)

    def by_relation_type(self, relation_type: str) -> list[Synset]:
        """Find synsets with specific relation type.

        Parameters
        ----------
        relation_type : str
            Relation type (e.g., "hypernym", "hyponym", "antonym").

        Returns
        -------
        list[Synset]
            Synsets with the specified relation type.
        """
        matching_synsets = []
        for synset in self._synsets.values():
            filtered_ptrs = filter_by_relation_type(synset.pointers, relation_type)
            if filtered_ptrs:
                matching_synsets.append(synset)

        return sorted(matching_synsets, key=lambda s: s.offset)

    def by_syntax(self, pattern: str) -> list[Synset]:
        """Find synsets with verbs matching a syntactic pattern.

        Parameters
        ----------
        pattern : str
            Syntactic pattern (e.g., "NP V", "NP V NP", "NP V PP").

        Returns
        -------
        list[Synset]
            Synsets containing verbs with matching syntactic frames.
        """
        parser = SyntaxParser()
        parsed_pattern = parser.parse(pattern)

        # Get frame numbers that match this pattern
        matching_frame_numbers = self._get_frame_numbers_for_pattern(parsed_pattern)

        if not matching_frame_numbers:
            return []

        matching_synsets = []
        for synset in self._synsets.values():
            if synset.ss_type == "v" and synset.frames:  # Only verb synsets with frames
                for verb_frame in synset.frames:
                    if verb_frame.frame_number in matching_frame_numbers:
                        matching_synsets.append(synset)
                        break

        return sorted(matching_synsets, key=lambda s: s.offset)

    def _get_frame_numbers_for_pattern(
        self, parsed_pattern: UnifiedSyntaxPattern
    ) -> set[VerbFrameNumber]:
        """Map syntax pattern to WordNet verb frame numbers."""
        # Complete WordNet verb frame to syntax pattern mapping (35 frames)
        verb_frame_patterns = {
            # Basic intransitive patterns (1-7)
            1: "NP V",  # Something ----s
            2: "NP V PP",  # Somebody ----s PP
            3: "NP V ADV",  # Somebody ----s Adverb
            4: "NP V",  # Something is ----ing PP
            5: "NP V ADJ",  # Something ----s Adjective/Noun
            6: "NP V ADJ",  # Something ----s Adjective/Noun
            7: "NP V NP",  # Somebody ----s somebody
            # Basic transitive patterns (8-12)
            8: "NP V NP",  # Somebody ----s something
            9: "NP V NP PP",  # Somebody ----s somebody PP
            10: "NP V NP NP",  # Something ----s somebody something
            11: "NP V NP PP",  # Something ----s something to somebody
            12: "NP V NP",  # Something ----s something
            # Reflexive and reciprocal patterns (13-16)
            13: "NP V NP",  # Somebody ----s himself
            14: "NP V NP",  # Somebody ----s somebody
            15: "NP V NP",  # Something ----s something
            16: "NP V PP",  # Somebody ----s PP
            # Movement and change of state (17-24)
            17: "NP V PP",  # Somebody ----s from something
            18: "NP V PP",  # Somebody ----s on something
            19: "NP V PP",  # Somebody ----s with something
            20: "NP V PP",  # Somebody ----s of something
            21: "NP V NP PP",  # Somebody ----s something on something
            22: "NP V NP PP",  # Somebody ----s something with something
            23: "NP V NP PP",  # Somebody ----s something from something
            24: "NP V NP PP",  # Somebody ----s something to something
            # Sentential complement patterns (25-29)
            25: "NP V S",  # Somebody ----s that CLAUSE
            26: "NP V NP S",  # Somebody ----s somebody that CLAUSE
            27: "NP V TO VP",  # Somebody ----s to INFINITIVE
            28: "NP V NP TO VP",  # Somebody ----s somebody to INFINITIVE
            29: "NP V NP VP[ING]",  # Somebody ----s somebody into V-ing something
            # Complex locative and resultative patterns (30-35)
            30: "NP V PP",  # Somebody ----s PP
            31: "NP V NP PP",  # Somebody ----s something PP
            32: "NP V PP PP",  # Somebody ----s PP PP
            33: "NP V NP AP",  # Somebody ----s something Adjective/Noun
            34: "NP V NP AP",  # Somebody ----s somebody Adjective/Noun
            35: "NP V AP",  # Something ----s Adjective/Noun
        }

        pattern_str = self._pattern_to_string(parsed_pattern)

        matching_frames: set[VerbFrameNumber] = set()
        for frame_num, frame_pattern in verb_frame_patterns.items():
            if self._patterns_match(pattern_str, frame_pattern, parsed_pattern):
                matching_frames.add(frame_num)  # type: ignore[arg-type]

        return matching_frames

    def _pattern_to_string(self, parsed_pattern: UnifiedSyntaxPattern) -> str:
        """Convert parsed pattern back to string for comparison."""
        elements: list[str] = []
        for element in parsed_pattern.elements:
            if hasattr(element, "constituent"):
                elements.append(element.constituent)
            else:
                elements.append(str(element))
        return " ".join(elements)

    def _patterns_match(
        self, search_pattern: str, frame_pattern: str, parsed_pattern: UnifiedSyntaxPattern
    ) -> bool:
        """Check if search pattern matches frame pattern with hierarchical matching."""
        parser = SyntaxParser()
        try:
            parsed_frame = parser.parse(frame_pattern)
        except (ValueError, AttributeError):
            # If parsing fails, fall back to simple string comparison
            return search_pattern == frame_pattern

        # Use hierarchical matching from syntax module
        if len(parsed_pattern.elements) != len(parsed_frame.elements):
            return False

        for search_elem, frame_elem in zip(
            parsed_pattern.elements, parsed_frame.elements, strict=False
        ):
            matches, _ = search_elem.matches_hierarchically(frame_elem)
            if not matches:
                return False

        return True

    def get_synset_by_id(self, synset_id: str) -> Synset | None:
        """Get a synset by its ID string.

        Parameters
        ----------
        synset_id : str
            Synset ID in format "offsetpos" (e.g., "01234567n").

        Returns
        -------
        Synset | None
            The synset if found, None otherwise.
        """
        if len(synset_id) == 9 and synset_id[:-1].isdigit() and synset_id[-1] in "nvasr":
            offset = synset_id[:-1]
            pos = cast(WordNetPOS, synset_id[-1])
            return self.by_offset(offset, pos)
        return None

    def get_statistics(self) -> dict[str, int]:
        """Get search index statistics.

        Returns
        -------
        dict[str, int]
            Statistics about indexed data.
        """
        total_words = sum(len(s.words) for s in self._synsets.values())

        total_lemmas = len(self._synsets_by_lemma)

        # Count synsets by POS from the records themselves; _synsets_by_pos
        # indexes satellites twice and would double-count adjectives.
        pos_counts = Counter(s.ss_type for s in self._synsets.values())

        return {
            "synset_count": len(self._synsets),
            "sense_count": len(self._senses),
            "unique_lemmas": total_lemmas,
            "total_words": total_words,
            "domain_count": len(self._synsets_by_domain),
            **{f"{pos}_synsets": count for pos, count in pos_counts.items()},
        }

    @classmethod
    def from_jsonl_files(
        cls, synsets_path: Path | str | None = None, senses_path: Path | str | None = None
    ) -> WordNetSearch:
        """Load search index from JSON Lines files.

        Parameters
        ----------
        synsets_path : Path | str | None
            Path to JSON Lines file containing synsets.
        senses_path : Path | str | None
            Path to JSON Lines file containing senses.

        Returns
        -------
        WordNetSearch
            Search index populated with data from files.

        Raises
        ------
        FileNotFoundError
            If specified file does not exist.
        ValueError
            If file contains invalid data.
        """
        synsets = []
        senses = []

        if synsets_path:
            synsets_path = Path(synsets_path)
            if not synsets_path.exists():
                msg = f"Synsets file not found: {synsets_path}"
                raise FileNotFoundError(msg)

            with synsets_path.open(encoding="utf-8") as f:
                for line_raw in f:
                    line = line_raw.strip()
                    if line:
                        synset = Synset.model_validate_json(line)
                        synsets.append(synset)

        if senses_path:
            senses_path = Path(senses_path)
            if not senses_path.exists():
                msg = f"Senses file not found: {senses_path}"
                raise FileNotFoundError(msg)

            with senses_path.open(encoding="utf-8") as f:
                for line_raw in f:
                    line = line_raw.strip()
                    if line:
                        sense = Sense.model_validate_json(line)
                        senses.append(sense)

        return cls(synsets=synsets, senses=senses)
