"""Shared type definitions used across all linguistic resources.

Defines type aliases and literal types used throughout the glazing package
for cross-dataset functionality.

Constants
---------
DatasetType : type[Literal]
    Primary dataset types (FrameNet, PropBank, VerbNet, WordNet).
ResourceType : type[Literal]
    Extended resource types including additional datasets.
MappingSource : type[Literal]
    Mapping source provenance values.
LogicType : type[Literal]
    Logical operators for combining restrictions (or, and).
MappingConfidenceScore : type[Annotated[float, Field]]
    Confidence score for mappings (0.0 to 1.0).
VersionString : type[Annotated[str, Field]]
    Version string following semantic versioning.
MappingType : type[Literal]
    Mapping type classifications.
AlignmentType : type[Literal]
    Alignment types for cross-dataset alignments.
ConflictType : type[Literal]
    Conflict types in mappings.
ValidationStatus : type[Literal]
    Validation status for mappings.
OperationType : type[Literal]
    Common dataset operations.
FRAME_ID_PATTERN : str
    FrameNet frame ID pattern (numeric).
VERBNET_CLASS_PATTERN : str
    VerbNet class ID pattern.
VERBNET_KEY_PATTERN : str
    VerbNet key pattern.
LEMMA_PATTERN : str
    Word lemma pattern.
HEX_COLOR_PATTERN : str
    6-digit hex color pattern.

Examples
--------
>>> from glazing.types import DatasetType, MappingSource
>>> dataset: DatasetType = "framenet"
>>> source: MappingSource = "manual"
"""

from typing import Annotated, Literal

from pydantic import Field

# Use Python 3.13+ type statement for all aliases

# Primary dataset types
type DatasetType = Literal["framenet", "propbank", "verbnet", "wordnet"]

# Extended resource types including additional datasets
type ResourceType = Literal[
    "verbnet",
    "framenet",
    "wordnet",
    "propbank",
    "AMR",
    "UMR",
    "Flickr",
    "THYME",
    "Spatial",
    "VerbNet",  # Variant capitalization found in some files
    "FrameNet",  # Variant capitalization found in some files
    "WordNet",  # Variant capitalization found in some files
    "PropBank",  # Variant capitalization found in some files
    "Framenet",  # Variant capitalization found in some PropBank files
]

# Mapping source provenance
type MappingSource = Literal[
    "manual",  # Manually created mapping
    "automatic",  # Automatically generated
    "manual+strict-conv",  # Manual with strict conversion
    "manualchecks",  # Manual with additional checks
    "auto",  # Short for automatic
    "gold",  # Gold standard annotation
    "silver",  # Silver standard (less reliable)
    "inherited",  # Inherited from parent class/frame
]

# Logical operators for combining restrictions
type LogicType = Literal["or", "and"]

# Confidence score for mappings (0.0 to 1.0)
type MappingConfidenceScore = Annotated[
    float, Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
]

# Version string following semantic versioning
type VersionString = Annotated[
    str,
    Field(
        pattern=r"^\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?$",
        description="Semantic version string (e.g., '1.0.0', '2.1.0-alpha')",
    ),
]

# Mapping type classifications
type MappingType = Literal[
    "direct",  # Direct one-to-one mapping
    "inherited",  # Inherited from parent structure
    "inferred",  # Inferred through analysis
    "partial",  # Partial/incomplete mapping
    "transitive",  # Through intermediate resource
    "manual",  # Manually specified
    "automatic",  # Automatically generated
    "hybrid",  # Combination of methods
]

# Alignment types for cross-dataset alignments
type AlignmentType = Literal[
    "exact",  # Exact match
    "equivalent",  # Semantically equivalent
    "subsumes",  # Source subsumes target
    "subsumed_by",  # Source subsumed by target
    "overlaps",  # Partial overlap
    "related",  # Related but not equivalent
    "contradicts",  # Contradictory mappings
]

# Conflict types in mappings
type ConflictType = Literal[
    "ambiguous",  # Multiple equally valid mappings
    "contradictory",  # Mutually exclusive mappings
    "version_mismatch",  # Different dataset versions
    "inheritance",  # Conflict in inheritance chain
]

# Validation status for mappings
type ValidationStatus = Literal[
    "validated",  # Fully validated
    "unvalidated",  # Not yet validated
    "disputed",  # Under dispute
    "deprecated",  # No longer recommended
]

# Common dataset operations
type OperationType = Literal[
    "search",  # Search operation
    "load",  # Data loading
    "convert",  # Format conversion
    "validate",  # Data validation
    "index",  # Index building
    "cache",  # Cache operation
]

# Regex patterns for common identifier formats
# These are used for validation across modules

# Frame/Class/Roleset ID patterns
FRAME_ID_PATTERN = r"^\d+$"  # FrameNet frame ID (numeric)
VERBNET_CLASS_PATTERN = r"^[a-z_]+-[0-9]+(?:\.[0-9]+)*(?:-[0-9]+)*$"  # e.g., "give-13.1-1"

# Sense key patterns
VERBNET_KEY_PATTERN = r"^[a-z_-]+#\d+$"  # e.g., "give#2"

# Name validation patterns
LEMMA_PATTERN = (
    r"^[a-zA-Z0-9][a-zA-Z0-9_\'\-\.\/]*$"  # Word lemmas (incl. proper nouns, abbreviations)
)

# Color validation for FrameNet
HEX_COLOR_PATTERN = r"^#?[0-9A-Fa-f]{6}$"  # 6-digit hex color with optional # prefix


# Shared error types
class DataNotLoadedError(Exception):
    """Raised when attempting to access data that hasn't been loaded."""


class InvalidReferenceError(Exception):
    """Raised when a cross-reference cannot be resolved."""


class MappingConflictError(Exception):
    """Raised when conflicting mappings are detected."""


class ValidationError(Exception):
    """Raised when data fails validation against schema."""


# Type guards for runtime checking
def is_dataset_type(value: str) -> bool:
    """Check if a string is a valid DatasetType.

    Parameters
    ----------
    value : str
        The string to check.

    Returns
    -------
    bool
        True if the value is a valid DatasetType.
    """
    return value in {"framenet", "propbank", "verbnet", "wordnet"}


def is_resource_type(value: str) -> bool:
    """Check if a string is a valid ResourceType.

    Parameters
    ----------
    value : str
        The string to check.

    Returns
    -------
    bool
        True if the value is a valid ResourceType.
    """
    return value in {
        "verbnet",
        "framenet",
        "wordnet",
        "propbank",
        "VerbNet",
        "FrameNet",
        "WordNet",
        "PropBank",
        "Framenet",
        "AMR",
        "UMR",
        "Flickr",
        "THYME",
        "Spatial",
    }


def is_valid_confidence(value: float) -> bool:
    """Check if a float is a valid confidence score.

    Parameters
    ----------
    value : float
        The value to check.

    Returns
    -------
    bool
        True if the value is between 0.0 and 1.0 inclusive.
    """
    return 0.0 <= value <= 1.0
