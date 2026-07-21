"""Unified data models and interfaces for syntactic and semantic frame ontologies.

This package provides type-safe data models and utilities for working with
four major linguistic resources. All models use Pydantic v2 for validation
and support JSON Lines serialization.

Modules
-------
framenet
    Models and utilities for FrameNet semantic frames.
propbank
    Models and utilities for PropBank rolesets.
verbnet
    Models and utilities for VerbNet verb classes.
wordnet
    Models and utilities for WordNet synsets and relations.
references
    Cross-reference resolution between datasets.
utils
    Shared utilities and helper functions.

Examples
--------
>>> from glazing import FrameNet, PropBank, VerbNet, WordNet
>>> fn = FrameNet.load("data/framenet.jsonl")
>>> frames = fn.get_frames_by_lemma("give")
"""

import os
import sys
import warnings

from glazing.__version__ import __version__, __version_info__
from glazing.initialize import check_initialization, get_default_data_dir, is_data_current


def _check_initialization() -> None:
    """Check if datasets are initialized and prompt user if not."""
    # Skip check in certain environments
    if any(
        [
            os.environ.get("GLAZING_SKIP_INIT_CHECK"),
            "sphinx" in sys.modules,  # Building docs
            "pytest" in sys.modules,  # Running tests
            os.environ.get("CI"),  # CI environment
        ]
    ):
        return

    data_dir = get_default_data_dir()
    if not check_initialization():
        warnings.warn(
            f"\nGlazing datasets not initialized.\n"
            f"Run 'glazing init' to download and convert all datasets.\n"
            f"Data will be stored in: {data_dir}",
            UserWarning,
            stacklevel=2,
        )
    elif not is_data_current():
        warnings.warn(
            f"\nGlazing datasets were converted by an older version and are out of date.\n"
            f"Run 'glazing init' to re-convert them from the downloaded source files.\n"
            f"Data location: {data_dir}",
            UserWarning,
            stacklevel=2,
        )


# Check initialization on import (can be disabled via env var)
_check_initialization()


__all__ = [
    "__version__",
    "__version_info__",
]
