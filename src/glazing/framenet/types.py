"""FrameNet-specific type definitions.

This module defines all type aliases and literal types specific to FrameNet.

Constants
---------
CoreType : type[Literal]
    Frame element core types (Core, Peripheral, etc.).
FrameNetPOS : type[Literal]
    Part-of-speech tags used in FrameNet.
AnnotationStatus : type[Literal]
    Annotation completion status values.
FrameRelationType : type[Literal]
    Frame-to-frame relation types.
LayerType : type[Literal]
    Annotation layer types.
GrammaticalFunction : type[Literal]
    Grammatical function labels.
PhraseType : type[Literal]
    Phrase type labels.
FrameRelationSubType : type[Literal]
    Frame relation subtypes (FE-level mappings).
InstantiationType : type[Literal]
    Null instantiation types.
MarkupType : type[Literal]
    Text markup types in definitions.
DefinitionPrefix : type[Literal]
    Definition source prefixes.
CorpusName : type[Literal]
    Corpus types.
FrameID : type[int]
    Frame identifier.
SemTypeID : type[int]
    Semantic type identifier.
LexicalUnitID : type[int]
    Lexical unit identifier.
SentenceID : type[int]
    Sentence identifier.
AnnotationSetID : type[int]
    Annotation set identifier.
LabelID : type[int]
    Label identifier.
CorpusID : type[int]
    Corpus identifier.
DocumentID : type[int]
    Document identifier.
ParagraphID : type[int]
    Paragraph identifier.
FrameName : type[str]
    Frame name (validated).
FEName : type[str]
    Frame element name (validated).
FEAbbrev : type[str]
    FE abbreviation (validated).
LexicalUnitName : type[str]
    LU name (lemma.pos format).
Username : type[str]
    Creator/annotator username.
FRAME_NAME_PATTERN : str
    Frame name validation pattern.
FE_NAME_PATTERN : str
    Frame element name validation pattern.
FE_ABBREV_PATTERN : str
    FE abbreviation validation pattern.
LU_NAME_PATTERN : str
    Lexical unit name validation pattern.
USERNAME_PATTERN : str
    Username validation pattern.
LEXEME_NAME_PATTERN : str
    Lexeme name validation pattern.
MAX_FRAME_ELEMENTS : int
    Maximum FEs per frame.
MAX_ANNOTATION_LAYERS : int
    Maximum annotation layers.
MAX_LEXICAL_UNITS : int
    Maximum LUs per frame.

Functions
---------
is_valid_frame_name
    Validate FrameNet frame name format.
is_valid_fe_name
    Validate frame element name format.
is_valid_fe_abbrev
    Validate FE abbreviation format.
is_valid_lu_name
    Validate lexical unit name format.
is_valid_username
    Validate FrameNet username format.
is_valid_hex_color
    Validate 6-digit hex color code.

Examples
--------
>>> from glazing.framenet.types import CoreType, FrameNetPOS
>>> core: CoreType = "Core"
>>> pos: FrameNetPOS = "V"
>>>
>>> from glazing.framenet.types import is_valid_frame_name
>>> is_valid_frame_name("Abandonment")
True
>>> is_valid_frame_name("123-invalid!")
False
"""

import re
from typing import Literal

from glazing.types import HEX_COLOR_PATTERN

# Core frame element types
type CoreType = Literal[
    "Core",  # Central to frame meaning
    "Core-Unexpressed",  # Core but typically unexpressed
    "Peripheral",  # Optional, modifies core meaning
    "Extra-Thematic",  # Outside core frame semantics
]

# FrameNet part-of-speech tags
type FrameNetPOS = Literal[
    "A",  # Adjective
    "ADV",  # Adverb
    "ART",  # Article
    "AVP",  # Adverb phrase (German)
    "C",  # Conjunction
    "IDIO",  # Idiom
    "INTJ",  # Interjection
    "N",  # Noun
    "NUM",  # Number
    "PREP",  # Preposition
    "PRON",  # Pronoun
    "SCON",  # Subordinating conjunction
    "V",  # Verb
]

# Annotation status (verified in actual FrameNet v1.7 data)
type AnnotationStatus = Literal[
    "Add_Annotation",  # Needs additional annotation
    "AUTO_APP",  # Automatically approved
    "AUTO_EDITED",  # Automatically edited
    "BTDT",  # Been there, done that
    "Created",  # Newly created
    "Finished_Initial",  # Initial annotation finished
    "Finished_X-Gov",  # Finished with extra-governmental
    "FN1_Sent",  # FrameNet 1 sentence
    "In_Use",  # Currently in use
    "Insufficient_Attestations",  # Not enough examples
    "MANUAL",  # Manually annotated
    "Needs_SCs",  # Needs subcorpus
    "New",  # New annotation
    "Pre-Marked",  # Pre-marked for annotation
    "Rules_Defined",  # Rules defined
    "UNANN",  # Unannotated
]

# Frame-to-frame relation types
type FrameRelationType = Literal[
    "Inherits from",  # Child inherits from parent
    "Is Inherited by",  # Parent inherited by child
    "Perspective on",  # Alternative perspective
    "Is Perspectivized in",  # Has perspective
    "Uses",  # Uses another frame
    "Is Used by",  # Used by another frame
    "Subframe of",  # Part of larger frame
    "Has Subframe(s)",  # Contains subframes
    "Precedes",  # Temporal precedence
    "Is Preceded by",  # Temporally preceded
    "Is Inchoative of",  # Beginning of state
    "Is Causative of",  # Causes another frame
    "See also",  # Related frame
]

# Annotation layer types (verified in actual FrameNet v1.7 data)
type LayerType = Literal[
    # Core layers
    "FE",  # Frame element
    "GF",  # Grammatical function
    "PT",  # Phrase type
    "Target",  # Target word(s)
    # POS layers
    "Verb",  # Verb annotation
    "Noun",  # Noun annotation
    "Adj",  # Adjective annotation
    "Adv",  # Adverb annotation
    "Prep",  # Preposition annotation
    "Art",  # Article annotation
    "Scon",  # Subordinating conjunction annotation
    "Other",  # Other POS
    # Corpus layers
    "Sent",  # Sentence
    "NER",  # Named entity recognition
    "WSL",  # Word sense layer
    "BNC",  # British National Corpus
    "PENN",  # Penn Treebank tags
    # Constructional layers
    "CE",  # Constructional element
    "CEE",  # Constructional element extended
    "CstrPT",  # Constructional phrase type
    "GovX",  # Government/external argument
]

# Grammatical functions (verified in actual FrameNet v1.7 data)
type GrammaticalFunction = Literal[
    "Dep",  # Dependent
    "Ext",  # External argument
    "Obj",  # Object
    "Head",  # Head
    "Gen",  # Genitive
    "Quant",  # Quantifier
    "Appositive",  # Apposition
]

# Phrase types (verified in actual FrameNet v1.7 data)
type PhraseType = Literal[
    # Basic phrase types
    "NP",  # Noun phrase
    "AVP",  # Adverb phrase
    "AJP",  # Adjective phrase
    "APos",  # Adjectival position
    "PPadjP",  # PP functioning as adjective phrase
    "PPinterrog",  # Interrogative PP
    # Verb phrase variants
    "VPfin",  # Finite VP
    "VPing",  # -ing VP
    "VPto",  # to-infinitive VP
    "VPbrst",  # Bare stem VP
    "VPed",  # Past participle VP
    "VPtorel",  # to-infinitive relative VP
    # Sentence types
    "Sfin",  # Finite sentence
    "Sing",  # -ing sentence
    "Sto",  # to-infinitive sentence
    "Sbrst",  # Bare stem sentence
    "Swhether",  # whether-clause
    "Sinterrog",  # Interrogative sentence
    "Srel",  # Relative sentence
    "Sabs",  # Absolute sentence
    "Sforto",  # for-to sentence
    "Sub",  # Subordinate clause
    "Sun",  # Unknown sentence type
    # Simple types
    "Poss",  # Possessive
    "N",  # Noun
    "A",  # Adjective
    "Num",  # Number
    "Obj",  # Object
    "QUO",  # Quotation
    # Person markers
    "2nd",  # Second person
    "3rd",  # Third person
    # Null instantiation types
    "CNI",  # Constructional null instantiation
    "DNI",  # Definite null instantiation
    "INI",  # Indefinite null instantiation
    "INC",  # Incorporated
    "DEN",  # Definite entity null
    # Special
    "--",  # Unspecified
    "unknown",  # Unknown type
]

# Frame relation subtypes (FE-level mappings)
type FrameRelationSubType = Literal[
    "Mapping",  # Direct mapping
    "Inheritance",  # Inherited relation
    "Equivalence",  # Equivalent FEs
]

# Null instantiation types
type InstantiationType = Literal[
    "INI",  # Indefinite null instantiation
    "DNI",  # Definite null instantiation
    "CNI",  # Constructional null instantiation
    "INC",  # Incorporated (found in data)
]

# Text markup types in definitions
type MarkupType = Literal[
    "def-root",  # Definition root
    "fex",  # Frame element example
    "fen",  # Frame element name
    "t",  # Target
    "ex",  # Example
    "m",  # Mention
    "gov",  # Governor
    "x",  # Cross-reference
]

# Definition source prefixes
type DefinitionPrefix = Literal[
    "COD",  # Concise Oxford Dictionary
    "FN",  # FrameNet definition
]

# Corpus types (verified in actual FrameNet v1.7 data)
type CorpusName = Literal[
    "ANC",  # American National Corpus
    "KBEval",  # Knowledge Base Evaluation
    "LUCorpus-v0.3",  # Lexical Unit Corpus
    "Miscellaneous",  # Various sources
    "NTI",  # Nuclear Threat Initiative
    "PropBank",  # PropBank corpus
    "WikiTexts",  # Wikipedia texts
]

# Type aliases for IDs and references

# Numeric IDs
type FrameID = int  # Frame identifier
type SemTypeID = int  # Semantic type identifier
type LexicalUnitID = int  # Lexical unit identifier
type SentenceID = int  # Sentence identifier
type AnnotationSetID = int  # Annotation set identifier
type LabelID = int  # Label identifier
type CorpusID = int  # Corpus identifier
type DocumentID = int  # Document identifier
type ParagraphID = int  # Paragraph identifier

# String IDs
type FrameName = str  # Frame name (validated)
type FrameElementName = str  # Frame element name (validated)
type FEName = str  # Frame element name (validated)
type FEAbbrev = str  # FE abbreviation (validated)
type LexicalUnitName = str  # LU name (lemma.pos format)
type Username = str  # Creator/annotator username

# Validation helper functions


def is_valid_frame_name(name: str) -> bool:
    """Check if a string is a valid FrameNet frame name.

    Parameters
    ----------
    name : str
        The name to validate.

    Returns
    -------
    bool
        True if the name matches the frame name pattern.
    """
    return bool(re.match(FRAME_NAME_PATTERN, name))


def is_valid_fe_name(name: str) -> bool:
    """Check if a string is a valid frame element name.

    Parameters
    ----------
    name : str
        The name to validate.

    Returns
    -------
    bool
        True if the name matches the FE name pattern.
    """
    return bool(re.match(FE_NAME_PATTERN, name))


def is_valid_fe_abbrev(abbrev: str) -> bool:
    """Check if a string is a valid FE abbreviation.

    Parameters
    ----------
    abbrev : str
        The abbreviation to validate.

    Returns
    -------
    bool
        True if the abbreviation matches the pattern.
    """
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_-]*$", abbrev))


def is_valid_lu_name(name: str) -> bool:
    """Check if a string is a valid lexical unit name.

    FrameNet LU names follow the pattern: lemma.pos where lemma can be
    any string including multi-word expressions, proper nouns, acronyms,
    and special characters. Examples: "abandon.v", "April.n", "a bit.n",
    "(can't) help.v", "American [N and S Am].n"

    Parameters
    ----------
    name : str
        The LU name to validate (should be lemma.pos format).

    Returns
    -------
    bool
        True if the name matches the LU name pattern (has at least one char and .pos).
    """
    # Very permissive: just require something.pos format
    return bool(re.match(LU_NAME_PATTERN, name))


def is_valid_username(username: str) -> bool:
    """Check if a string is a valid FrameNet username.

    Parameters
    ----------
    username : str
        The username to validate.

    Returns
    -------
    bool
        True if the username matches the pattern.
    """
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9]*$", username))


def is_valid_hex_color(color: str) -> bool:
    """Check if a string is a valid 6-digit hex color.

    Parameters
    ----------
    color : str
        The color code to validate.

    Returns
    -------
    bool
        True if the color is a valid 6-digit hex code.
    """
    return bool(re.match(HEX_COLOR_PATTERN, color))


# FrameNet-specific pattern constants
FRAME_NAME_PATTERN = r"^[A-Za-z0-9_\-]+$"  # Allow hyphens in frame names
FE_NAME_PATTERN = r"^[A-Za-z0-9_\-\.\'\s]+$"  # Allow hyphens, periods, apostrophes, spaces
FE_ABBREV_PATTERN = r"^[A-Za-z0-9_\-\.\'\s/]+$"  # Allow slashes like H/C
# LU names: proper nouns, multi-word expressions, special chars (parentheses, brackets, etc.)
# Examples: "abandon.v", "April.n", "(can't) help.v", "a bit.n"
LU_NAME_PATTERN = r"^.+\.[a-z]+$"  # Anything followed by .pos (very permissive)
USERNAME_PATTERN = r"^[A-Za-z][A-Za-z0-9]*$"
# Lexeme names: spaces, hyphens, apostrophes, slashes, parentheses, brackets, digits
# Examples: "abandon", "a bit", "Boxing Day", "Scud-B missile", "(can't"
LEXEME_NAME_PATTERN = r"^.+$"  # Very permissive - any non-empty string

# Counts and limits
MAX_FRAME_ELEMENTS = 100  # Maximum FEs per frame
MAX_ANNOTATION_LAYERS = 50  # Maximum annotation layers
MAX_LEXICAL_UNITS = 1000  # Maximum LUs per frame
