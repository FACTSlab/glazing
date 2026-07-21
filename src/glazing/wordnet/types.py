"""WordNet-specific type definitions.

Defines type aliases and literal types specific to WordNet 3.1 data structures
including synsets, senses, relations, and morphological processing.

Constants
---------
WordNetPOS : type[Literal]
    WordNet part-of-speech tags (n, v, a, r, s).
PointerSymbol : type[Literal]
    All WordNet relation symbols (40+ types).
LexFileName : type[Literal]
    Lexical file names for 45 semantic categories.
VerbFrameNumber : type[Literal]
    Valid verb frame numbers (1-35).
AdjPosition : type[Literal]
    Adjective positions (attributive, predicative, postnominal).
SynsetID : type[Annotated[str, Field]]
    Full synset identifier with POS (e.g., "00001740-n").
SynsetOffset : type[Annotated[str, Field]]
    8-digit synset identifier with validation.
SynsetKey : type[Annotated[str, Field]]
    Canonical synset key: offset plus normalized POS (e.g., "00001740n").
SenseKey : type[Annotated[str, Field]]
    WordNet sense key with format validation.
LemmaKey : type[Annotated[str, Field]]
    Lemma key with format validation (e.g., "dog#n#1").
LexID : type[Annotated[int, Field]]
    Lexical ID (0-15) for distinguishing words in synsets.
SenseNumber : type[Annotated[int, Field]]
    Sense number for frequency-based ordering.
TagCount : type[Annotated[int, Field]]
    Semantic concordance tag count.
SYNSET_ID_PATTERN : str
    Regex pattern for synset ID with POS.
WORDNET_OFFSET_PATTERN : str
    Regex pattern for 8-digit synset offsets.
WORDNET_SENSE_KEY_PATTERN : str
    Regex pattern for sense key validation.
LEMMA_KEY_PATTERN : str
    Regex pattern for lemma key validation.
PERCENTAGE_NOTATION_PATTERN : str
    Regex pattern for VerbNet percentage notation.
WORDNET_SYNSET_KEY_PATTERN : str
    Regex pattern for canonical synset keys.

Functions
---------
normalize_pos
    Normalize a WordNet POS to its data-file category ("s" becomes "a").
make_synset_key
    Build the canonical key identifying a synset across the database.

Examples
--------
>>> from glazing.wordnet.types import WordNetPOS, SynsetOffset, make_synset_key
>>> pos: WordNetPOS = "v"
>>> offset: SynsetOffset = "00001740"
>>> make_synset_key(offset, pos)
'00001740v'
"""

from typing import Annotated, Literal

from pydantic import Field

# WordNet part-of-speech categories
type WordNetPOS = Literal[
    "n",  # noun
    "v",  # verb
    "a",  # adjective
    "r",  # adverb
    "s",  # satellite adjective
]

# All WordNet pointer symbols (semantic and lexical relations)
type PointerSymbol = Literal[
    "!",  # Antonym
    "@",  # Hypernym (is-a)
    "@i",  # Instance Hypernym
    "~",  # Hyponym
    "~i",  # Instance Hyponym
    "#m",  # Member holonym
    "#s",  # Substance holonym
    "#p",  # Part holonym
    "%m",  # Member meronym
    "%s",  # Substance meronym
    "%p",  # Part meronym
    "=",  # Attribute
    "+",  # Derivationally related form
    ";c",  # Domain of synset - TOPIC
    "-c",  # Member of this domain - TOPIC
    ";r",  # Domain of synset - REGION
    "-r",  # Member of this domain - REGION
    ";u",  # Domain of synset - USAGE
    "-u",  # Member of this domain - USAGE
    "*",  # Entailment (verbs)
    ">",  # Cause (verbs)
    "^",  # Also see
    "$",  # Verb Group
    "&",  # Similar to (adjectives)
    "<",  # Participle of verb (adjectives)
    "\\",  # Pertainym (pertains to noun) (adjectives)
]

# WordNet lexical file names (45 categories)
type LexFileName = Literal[
    # Noun files
    "noun.Tops",
    "noun.act",
    "noun.animal",
    "noun.artifact",
    "noun.attribute",
    "noun.body",
    "noun.cognition",
    "noun.communication",
    "noun.event",
    "noun.feeling",
    "noun.food",
    "noun.group",
    "noun.location",
    "noun.motive",
    "noun.object",
    "noun.person",
    "noun.phenomenon",
    "noun.plant",
    "noun.possession",
    "noun.process",
    "noun.quantity",
    "noun.relation",
    "noun.shape",
    "noun.state",
    "noun.substance",
    "noun.time",
    # Verb files
    "verb.body",
    "verb.change",
    "verb.cognition",
    "verb.communication",
    "verb.competition",
    "verb.consumption",
    "verb.contact",
    "verb.creation",
    "verb.emotion",
    "verb.motion",
    "verb.perception",
    "verb.possession",
    "verb.social",
    "verb.stative",
    "verb.weather",
    # Adjective files
    "adj.all",
    "adj.pert",
    "adj.ppl",
    # Adverb files
    "adv.all",
]

# Valid verb frame numbers (1-35)
type VerbFrameNumber = Literal[
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
]

# Adjective position markers
type AdjPosition = Literal[
    "a",  # attributive (prenominal)
    "p",  # predicative
    "ip",  # immediately postnominal
]

# Regex patterns for WordNet identifier formats

# WordNet synset offset (8-digit zero-padded)
WORDNET_OFFSET_PATTERN = r"^[0-9]{8}$"

# WordNet sense key format
WORDNET_SENSE_KEY_PATTERN = r"^[a-z0-9_.-]+%[1-5]:[0-9]{2}:[0-9]{2}:[a-z0-9_.-]*:[0-9]*$"

# VerbNet percentage notation (WordNet reference format)
PERCENTAGE_NOTATION_PATTERN = r"^[a-z_-]+%[1-5]:[0-9]{2}:[0-9]{2}$"

# WordNet synset ID (offset with POS)
SYNSET_ID_PATTERN = r"^[0-9]{8}-?[nvasr]$"

# WordNet lemma key (lemma#pos#sense)
LEMMA_KEY_PATTERN = r"^[a-z0-9_.-]+#[nvasr]#[0-9]+$"

# Canonical synset key (offset with normalized POS; satellites key as "a")
WORDNET_SYNSET_KEY_PATTERN = r"^[0-9]{8}[nvar]$"

# Validated string types with constraints

# Full synset identifier with POS (e.g., "00001740-n" or "00001740n")
type SynsetID = Annotated[
    str,
    Field(
        pattern=SYNSET_ID_PATTERN,
        description="Synset ID with POS (e.g., '00001740-n', '00001740n')",
    ),
]

# 8-digit synset identifier
type SynsetOffset = Annotated[
    str,
    Field(
        pattern=WORDNET_OFFSET_PATTERN,
        description="8-digit zero-padded synset offset (e.g., '00001740')",
    ),
]

# Canonical synset key: offset plus normalized POS (e.g., "00001740n")
type SynsetKey = Annotated[
    str,
    Field(
        pattern=WORDNET_SYNSET_KEY_PATTERN,
        description="Synset key: offset plus normalized POS (e.g., '00001740n')",
    ),
]

# WordNet sense key with full format validation
type SenseKey = Annotated[
    str,
    Field(
        pattern=WORDNET_SENSE_KEY_PATTERN,
        description="WordNet sense key (e.g., 'abandon%2:40:01::')",
    ),
]

# WordNet lemma key (lemma#pos#sense)
type LemmaKey = Annotated[
    str,
    Field(
        pattern=LEMMA_KEY_PATTERN,
        description="Lemma key (e.g., 'dog#n#1', 'give#v#2')",
    ),
]

# Lexical ID for distinguishing words in same synset
type LexID = Annotated[
    int,
    Field(
        ge=0,
        le=15,
        description="Lexical ID (0-15) for word disambiguation in synset",
    ),
]

# Sense number for frequency-based ordering
type SenseNumber = Annotated[
    int,
    Field(
        ge=1,
        description="1-based sense number for frequency ordering",
    ),
]

# Semantic concordance tag count
type TagCount = Annotated[
    int,
    Field(
        ge=0,
        description="Number of times sense appears in semantic concordances",
    ),
]

# Raw lemma string
type Lemma = str

# Synset offset string
type Offset = str


def normalize_pos(pos: WordNetPOS) -> WordNetPOS:
    """Normalize a WordNet POS to its data-file category.

    Adjective satellites (``"s"``) live in ``data.adj`` alongside adjective
    heads, and WordNet pointers targeting a satellite always carry ``"a"`` in
    their POS field. Normalizing ``"s"`` to ``"a"`` therefore lets a synset
    record and any pointer referencing it agree on a single key.

    Parameters
    ----------
    pos : WordNetPOS
        Part of speech to normalize.

    Returns
    -------
    WordNetPOS
        ``"a"`` when `pos` is ``"s"``, otherwise `pos` unchanged.

    Examples
    --------
    >>> normalize_pos("s")
    'a'
    >>> normalize_pos("v")
    'v'
    """
    return "a" if pos == "s" else pos


def make_synset_key(offset: SynsetOffset, pos: WordNetPOS) -> SynsetKey:
    """Build the canonical key identifying a synset.

    WordNet offsets are byte offsets into a per-POS data file, so they are
    unique only within a file - 330 offsets in WordNet 3.1 occur in more than
    one ``data.*`` file. Pairing the offset with its normalized POS yields a
    key that is unique across the whole database.

    Parameters
    ----------
    offset : SynsetOffset
        8-digit zero-padded synset offset.
    pos : WordNetPOS
        Part of speech; ``"s"`` is normalized to ``"a"``.

    Returns
    -------
    SynsetKey
        Offset concatenated with the normalized POS.

    Examples
    --------
    >>> make_synset_key("00001740", "n")
    '00001740n'
    >>> make_synset_key("00014377", "s")
    '00014377a'
    """
    return f"{offset}{normalize_pos(pos)}"
