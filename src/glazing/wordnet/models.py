"""WordNet data models.

This module implements WordNet 3.1 data models including synsets, words,
senses, and relations using Pydantic v2 for validation and type safety.

Classes
-------
Synset
    WordNet synset (set of cognitive synonyms).
Word
    Word/lemma in a synset.
Pointer
    Relation/pointer to another synset or word.
VerbFrame
    Syntactic frame for a verb.
Sense
    Word sense (word-meaning pair).
IndexEntry
    Entry in WordNet index file.
ExceptionEntry
    Morphological exception mapping.
WordNetCrossRef
    Cross-reference to WordNet from other resources.

Examples
--------
>>> from glazing.wordnet.models import Synset, Word
>>> synset = Synset(
...     offset="00001740",
...     lex_filenum=5,
...     lex_filename="noun.animal",
...     ss_type="n",
...     words=[Word(lemma="dog", lex_id=0)],
...     pointers=[],
...     gloss="a domesticated carnivorous mammal"
... )
"""

from __future__ import annotations

import re

from pydantic import Field, field_validator

from glazing.base import GlazingBaseModel
from glazing.types import LEMMA_PATTERN
from glazing.wordnet.types import (
    LexFileName,
    LexID,
    PointerSymbol,
    SenseKey,
    SenseNumber,
    SynsetOffset,
    TagCount,
    VerbFrameNumber,
    WordNetPOS,
)


class Word(GlazingBaseModel):
    """A word/lemma in a synset.

    Attributes
    ----------
    lemma : str
        Word form (lowercase, underscores for spaces).
    lex_id : LexID
        Distinguishes same word in synset (0-15).
    sense_number : int | None, default=None
        Frequency-based sense ordering from index.sense.
    tag_count : int, default=0
        Semantic concordance tag count.

    Examples
    --------
    >>> word = Word(lemma="dog", lex_id=0)
    >>> word.lemma
    'dog'
    >>> word.lex_id
    0
    """

    lemma: str = Field(description="Word form (lowercase, underscores for spaces)")
    lex_id: LexID = Field(description="Lexical ID distinguishing same word in synset")
    sense_number: int | None = Field(default=None, description="Frequency-based sense ordering")
    tag_count: int = Field(default=0, ge=0, description="Semantic concordance tag count")

    @field_validator("lemma")
    @classmethod
    def validate_lemma(cls, v: str) -> str:
        """Validate lemma format.

        Parameters
        ----------
        v : str
            The lemma to validate.

        Returns
        -------
        str
            The validated lemma.

        Raises
        ------
        ValueError
            If lemma format is invalid.
        """
        if not re.match(LEMMA_PATTERN, v):
            msg = f"Invalid lemma format: {v!r}"
            raise ValueError(msg)
        return v


class Pointer(GlazingBaseModel):
    """A relation/pointer to another synset or word.

    Attributes
    ----------
    symbol : PointerSymbol
        Relation type symbol.
    offset : SynsetOffset
        Target synset offset.
    pos : WordNetPOS
        Target part of speech.
    source : int
        Source word number (0 = entire synset).
    target : int
        Target word number (0 = entire synset).

    Methods
    -------
    is_lexical()
        Check if this is a lexical (word-to-word) relation.
    is_semantic()
        Check if this is a semantic (synset-to-synset) relation.

    Examples
    --------
    >>> pointer = Pointer(
    ...     symbol="@",
    ...     offset="00002084",
    ...     pos="n",
    ...     source=0,
    ...     target=0
    ... )
    >>> pointer.is_semantic()
    True
    """

    symbol: PointerSymbol = Field(description="Relation type symbol")
    offset: SynsetOffset = Field(description="Target synset offset")
    pos: WordNetPOS = Field(description="Target part of speech")
    source: int = Field(ge=0, description="Source word number (0 = entire synset)")
    target: int = Field(ge=0, description="Target word number (0 = entire synset)")

    def is_lexical(self) -> bool:
        """Check if this is a lexical (word-to-word) relation.

        Returns
        -------
        bool
            True if either source or target is non-zero.
        """
        return self.source != 0 or self.target != 0

    def is_semantic(self) -> bool:
        """Check if this is a semantic (synset-to-synset) relation.

        Returns
        -------
        bool
            True if both source and target are zero.
        """
        return self.source == 0 and self.target == 0


class VerbFrame(GlazingBaseModel):
    """Syntactic frame for a verb.

    Attributes
    ----------
    frame_number : VerbFrameNumber
        Frame number (1-35).
    word_indices : list[int]
        Word indices (0 = all words, or specific indices).
    template : str | None, default=None
        Natural language frame template (e.g., "Something ----s").
    example_sentence : str | None, default=None
        Example sentence with %s placeholder for verb.

    Examples
    --------
    >>> frame = VerbFrame(frame_number=8, word_indices=[0])
    >>> frame.frame_number
    8
    """

    frame_number: VerbFrameNumber = Field(description="Frame number (1-35)")
    word_indices: list[int] = Field(
        default_factory=list, description="Word indices (0 = all words)"
    )
    template: str | None = Field(default=None, description="Natural language frame template")
    example_sentence: str | None = Field(
        default=None, description="Example sentence with %s placeholder"
    )

    @field_validator("word_indices")
    @classmethod
    def validate_word_indices(cls, v: list[int]) -> list[int]:
        """Validate word indices.

        Parameters
        ----------
        v : list[int]
            The word indices to validate.

        Returns
        -------
        list[int]
            The validated indices.

        Raises
        ------
        ValueError
            If any index is negative.
        """
        for idx in v:
            if idx < 0:
                msg = f"Word index cannot be negative: {idx}"
                raise ValueError(msg)
        return v


class Synset(GlazingBaseModel):
    """A WordNet synset (set of cognitive synonyms).

    Attributes
    ----------
    offset : SynsetOffset
        8-digit identifier.
    lex_filenum : int
        Lexical file number (0-44).
    lex_filename : LexFileName
        Validated lexical file name.
    ss_type : WordNetPOS
        Synset type (n, v, a, r, s).
    words : list[Word]
        Words in this synset.
    pointers : list[Pointer]
        Relations to other synsets.
    frames : list[VerbFrame] | None, default=None
        Verb frames (verbs only).
    gloss : str
        Definition and examples.

    Methods
    -------
    get_lemmas()
        Get all lemmas in the synset.
    get_hypernyms()
        Get hypernym pointers.
    get_hyponyms()
        Get hyponym pointers.

    Examples
    --------
    >>> synset = Synset(
    ...     offset="00001740",
    ...     lex_filenum=5,
    ...     lex_filename="noun.animal",
    ...     ss_type="n",
    ...     words=[Word(lemma="dog", lex_id=0)],
    ...     pointers=[],
    ...     gloss="a domesticated carnivorous mammal"
    ... )
    >>> synset.get_lemmas()
    ['dog']
    """

    offset: SynsetOffset = Field(description="8-digit synset identifier")
    lex_filenum: int = Field(ge=0, le=44, description="Lexical file number (0-44)")
    lex_filename: LexFileName = Field(description="Lexical file name")
    ss_type: WordNetPOS = Field(description="Synset type")
    words: list[Word] = Field(description="Words in this synset")
    pointers: list[Pointer] = Field(default_factory=list, description="Relations")
    frames: list[VerbFrame] | None = Field(None, description="Verb frames (verbs only)")
    gloss: str = Field(description="Definition and examples")

    def get_lemmas(self) -> list[str]:
        """Get all lemmas in the synset.

        Returns
        -------
        list[str]
            List of lemma strings.
        """
        return [word.lemma for word in self.words]

    def get_hypernyms(self) -> list[Pointer]:
        """Get hypernym pointers.

        Returns
        -------
        list[Pointer]
            Pointers with '@' symbol.
        """
        return [p for p in self.pointers if p.symbol == "@"]

    def get_hyponyms(self) -> list[Pointer]:
        """Get hyponym pointers.

        Returns
        -------
        list[Pointer]
            Pointers with '~' symbol.
        """
        return [p for p in self.pointers if p.symbol == "~"]

    def get_pointers_by_symbol(self, symbol: PointerSymbol) -> list[Pointer]:
        """Get pointers by relation symbol.

        Parameters
        ----------
        symbol : PointerSymbol
            Relation symbol to filter by.

        Returns
        -------
        list[Pointer]
            Pointers with the specified symbol.

        Examples
        --------
        >>> synset = Synset(...)
        >>> antonyms = synset.get_pointers_by_symbol("!")
        """
        return [p for p in self.pointers if p.symbol == symbol]

    def has_relation(self, symbol: PointerSymbol) -> bool:
        """Check if synset has a specific relation type.

        Parameters
        ----------
        symbol : PointerSymbol
            Relation symbol to check for.

        Returns
        -------
        bool
            True if synset has at least one pointer with this symbol.

        Examples
        --------
        >>> synset = Synset(...)
        >>> has_hypernyms = synset.has_relation("@")
        """
        return any(p.symbol == symbol for p in self.pointers)

    def get_semantic_pointers(self) -> list[Pointer]:
        """Get semantic (synset-to-synset) pointers only.

        Returns
        -------
        list[Pointer]
            Pointers where source=0 and target=0.
        """
        return [p for p in self.pointers if p.is_semantic()]

    def get_lexical_pointers(self) -> list[Pointer]:
        """Get lexical (word-to-word) pointers only.

        Returns
        -------
        list[Pointer]
            Pointers where source!=0 or target!=0.
        """
        return [p for p in self.pointers if p.is_lexical()]


class Sense(GlazingBaseModel):
    """A word sense (word-meaning pair).

    Attributes
    ----------
    sense_key : SenseKey
        Unique sense identifier.
    lemma : str
        Word form.
    ss_type : WordNetPOS
        Synset type.
    lex_filenum : int
        Lexical file number.
    lex_id : LexID
        Lexical ID.
    head_word : str | None, default=None
        For adjective satellites.
    head_id : int | None, default=None
        Head word lex_id.
    synset_offset : SynsetOffset
        Synset containing this sense.
    sense_number : SenseNumber
        Frequency-based ordering.
    tag_count : TagCount
        Semantic concordance count.

    Methods
    -------
    parse_sense_key()
        Parse sense key into components.

    Examples
    --------
    >>> sense = Sense(
    ...     sense_key="dog%1:05:00::",
    ...     lemma="dog",
    ...     ss_type="n",
    ...     lex_filenum=5,
    ...     lex_id=0,
    ...     synset_offset="00001740",
    ...     sense_number=1,
    ...     tag_count=15
    ... )
    >>> components = sense.parse_sense_key()
    >>> components['lemma']
    'dog'
    """

    sense_key: SenseKey = Field(description="Unique sense identifier")
    lemma: str = Field(description="Word form")
    ss_type: WordNetPOS = Field(description="Synset type")
    lex_filenum: int = Field(ge=0, le=44, description="Lexical file number")
    lex_id: LexID = Field(description="Lexical ID")
    head_word: str | None = Field(None, description="For adjective satellites")
    head_id: int | None = Field(None, description="Head word lex_id")
    synset_offset: SynsetOffset = Field(description="Synset containing this sense")
    sense_number: SenseNumber = Field(description="Frequency-based ordering")
    tag_count: TagCount = Field(description="Semantic concordance count")

    def parse_sense_key(self) -> dict[str, str | int | None]:
        """Parse sense key into components.

        Returns
        -------
        dict[str, str | int | None]
            Dictionary with components: lemma, ss_type, lex_filenum, lex_id,
            head_word, head_id.

        Examples
        --------
        >>> sense = Sense(sense_key="dog%1:05:00::", ...)
        >>> components = sense.parse_sense_key()
        >>> components['ss_type']
        1
        """
        parts = self.sense_key.split("%")
        lemma = parts[0]
        rest = parts[1].split(":")
        return {
            "lemma": lemma,
            "ss_type": int(rest[0]),
            "lex_filenum": int(rest[1]),
            "lex_id": int(rest[2]),
            "head_word": rest[3] if rest[3] else None,
            "head_id": int(rest[4]) if rest[4] else None,
        }


class IndexEntry(GlazingBaseModel):
    """An entry in a WordNet index file.

    Attributes
    ----------
    lemma : str
        Word form.
    pos : WordNetPOS
        Part of speech.
    synset_cnt : int
        Number of synsets.
    p_cnt : int
        Number of pointer types.
    ptr_symbols : list[PointerSymbol]
        Pointer symbols for this word.
    sense_cnt : int
        Same as synset_cnt.
    tagsense_cnt : int
        Number of senses in semantic concordances.
    synset_offsets : list[SynsetOffset]
        Synsets containing this word.

    Examples
    --------
    >>> entry = IndexEntry(
    ...     lemma="dog",
    ...     pos="n",
    ...     synset_cnt=7,
    ...     p_cnt=4,
    ...     ptr_symbols=["!", "@", "~", "#m"],
    ...     sense_cnt=7,
    ...     tagsense_cnt=6,
    ...     synset_offsets=["00001740", "00002084"]
    ... )
    """

    lemma: str = Field(description="Word form")
    pos: WordNetPOS = Field(description="Part of speech")
    synset_cnt: int = Field(ge=0, description="Number of synsets")
    p_cnt: int = Field(ge=0, description="Number of pointer types")
    ptr_symbols: list[PointerSymbol] = Field(description="Pointer symbols for this word")
    sense_cnt: int = Field(ge=0, description="Same as synset_cnt")
    tagsense_cnt: int = Field(ge=0, description="Semantic concordance senses")
    synset_offsets: list[SynsetOffset] = Field(description="Synsets with this word")


class ExceptionEntry(GlazingBaseModel):
    """Morphological exception mapping.

    Attributes
    ----------
    inflected_form : str
        Inflected/irregular form.
    base_forms : list[str]
        Base/lemma forms.

    Examples
    --------
    >>> entry = ExceptionEntry(
    ...     inflected_form="geese",
    ...     base_forms=["goose"]
    ... )
    """

    inflected_form: str = Field(description="Inflected/irregular form")
    base_forms: list[str] = Field(description="Base/lemma forms")
    pos: WordNetPOS | None = Field(default=None, description="Part of speech")

    @field_validator("inflected_form", "base_forms")
    @classmethod
    def validate_forms(cls, v: str | list[str]) -> str | list[str]:
        """Validate word forms.

        Parameters
        ----------
        v : str | list[str]
            The value to validate.

        Returns
        -------
        str | list[str]
            The validated value.

        Raises
        ------
        ValueError
            If word form is invalid.
        """
        if isinstance(v, str):
            cleaned = v.replace("_", "").replace("-", "").replace("'", "").replace(".", "")
            if not v or not cleaned.isalpha():
                msg = f"Invalid word form: {v}"
                raise ValueError(msg)
        elif isinstance(v, list):
            for form in v:
                cleaned = form.replace("_", "").replace("-", "").replace("'", "").replace(".", "")
                if not form or not cleaned.isalpha():
                    msg = f"Invalid word form: {form}"
                    raise ValueError(msg)
        return v


class WordNetCrossRef(GlazingBaseModel):
    """Cross-reference to WordNet from other resources.

    Attributes
    ----------
    sense_key : SenseKey | None, default=None
        Preferred: stable across versions.
    synset_offset : SynsetOffset | None, default=None
        Alternative: version-specific.
    lemma : str
        Word lemma.
    pos : WordNetPOS
        Part of speech.
    sense_number : SenseNumber | None, default=None
        Sense number for ordering.

    Methods
    -------
    to_percentage_notation()
        Convert to VerbNet percentage notation.
    from_percentage_notation(notation)
        Parse VerbNet percentage notation.
    is_valid_reference()
        Check if reference has valid identifiers.
    get_primary_identifier()
        Get primary identifier (sense_key preferred).

    Examples
    --------
    >>> ref = WordNetCrossRef(
    ...     sense_key="give%2:40:00::",
    ...     lemma="give",
    ...     pos="v"
    ... )
    >>> notation = ref.to_percentage_notation()
    >>> notation
    'give%2:40:00'
    >>> ref.is_valid_reference()
    True
    """

    sense_key: SenseKey | None = Field(None, description="Stable sense identifier")
    synset_offset: SynsetOffset | None = Field(None, description="Version-specific offset")
    lemma: str = Field(description="Word lemma")
    pos: WordNetPOS = Field(description="Part of speech")
    sense_number: SenseNumber | None = Field(None, description="Sense ordering")

    def to_percentage_notation(self) -> str:
        """Convert to VerbNet percentage notation.

        Returns
        -------
        str
            Percentage notation (e.g., "give%2:40:00").

        Examples
        --------
        >>> ref = WordNetCrossRef(sense_key="give%2:40:00::", lemma="give", pos="v")
        >>> ref.to_percentage_notation()
        'give%2:40:00'
        """
        if self.sense_key:
            # Extract components from sense key (format: lemma%ss_type:lex_filenum:lex_id::)
            parts = self.sense_key.split("%")
            if len(parts) >= 2:
                sense_part = parts[1].split(":")
                if len(sense_part) >= 3:
                    return f"{self.lemma}%{sense_part[0]}:{sense_part[1]}:{sense_part[2]}"
        return ""

    def is_valid_reference(self) -> bool:
        """Check if reference has valid identifiers.

        Returns
        -------
        bool
            True if has sense_key or synset_offset.

        Examples
        --------
        >>> ref = WordNetCrossRef(sense_key="give%2:40:00::", lemma="give", pos="v")
        >>> ref.is_valid_reference()
        True
        """
        return self.sense_key is not None or self.synset_offset is not None

    def get_primary_identifier(self) -> str | None:
        """Get primary identifier (sense_key preferred).

        Returns
        -------
        str | None
            Sense key if available, otherwise synset offset.

        Examples
        --------
        >>> ref = WordNetCrossRef(sense_key="give%2:40:00::", lemma="give", pos="v")
        >>> ref.get_primary_identifier()
        'give%2:40:00::'
        """
        return self.sense_key or self.synset_offset

    @classmethod
    def from_percentage_notation(cls, notation: str) -> WordNetCrossRef:
        """Parse VerbNet percentage notation.

        Parameters
        ----------
        notation : str
            Percentage notation (e.g., "give%2:40:00").

        Returns
        -------
        WordNetCrossRef
            Cross-reference object.

        Raises
        ------
        ValueError
            If notation format is invalid.

        Examples
        --------
        >>> ref = WordNetCrossRef.from_percentage_notation("give%2:40:00")
        >>> ref.lemma
        'give'
        >>> ref.pos
        'v'
        """
        match = re.match(r"^([a-z_-]+)%([1-5]):([0-9]{2}):([0-9]{2})$", notation)
        if not match:
            msg = f"Invalid percentage notation: {notation}"
            raise ValueError(msg)

        lemma = match.group(1)
        ss_type = int(match.group(2))
        lex_filenum = match.group(3)
        lex_id = match.group(4)

        # Map ss_type to POS
        pos_map: dict[int, WordNetPOS] = {1: "n", 2: "v", 3: "a", 4: "r", 5: "s"}
        pos = pos_map[ss_type]

        # Construct partial sense key
        sense_key = f"{lemma}%{ss_type}:{lex_filenum}:{lex_id}::"

        return cls(sense_key=sense_key, synset_offset=None, lemma=lemma, pos=pos, sense_number=None)
