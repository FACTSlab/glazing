"""Special case registry for handling edge cases in linguistic data.

This module provides a centralized registry for handling exceptional cases
in the data that don't fit standard patterns. It keeps edge case logic
separate from core validation code.

Classes
-------
SpecialCaseRegistry
    Central registry for all special case handling.
"""

import re
from typing import ClassVar


class SpecialCaseRegistry:
    """Registry for handling edge cases in linguistic datasets.

    This class maintains a registry of known edge cases and provides
    methods to check and handle them. It keeps all special case logic
    centralized and out of the main validation code.
    """

    # PropBank special cases
    PROPBANK_XML_FIXES: ClassVar[dict[str, list[dict[str, str]]]] = {
        "check.xml": [
            {
                "pattern": ">in</rel>",
                "replacement": ">in</arg>",
                "description": "Mismatched closing tag",
            }
        ],
        "AMR-UMR-91-rolesets.xml": [
            {
                "pattern": "      </example>\n      </example>",
                "replacement": "      </example>",
                "description": "Duplicate </example> closing tag in reference-illustration.91",
            }
        ],
    }

    PROPBANK_ROLESET_EXCEPTIONS: ClassVar[dict[str, str]] = {
        # Special suffixes that aren't numeric or LV
        r".*\.yy$": "allow_yy_suffix",  # Found in point.yy
        # Numeric-only predicates
        r"^\d+\.\d+$": "numeric_predicate",  # Found in 300.01, 1500.01
    }

    PROPBANK_ARG_EXCEPTIONS: ClassVar[dict[str, str]] = {
        # Special argument references that don't follow standard patterns
        "M-LOC": "modifier_location",  # Special modifier-location reference
    }

    # FrameNet special cases
    FRAMENET_ABBREV_EXCEPTIONS: ClassVar[dict[str, str]] = {
        "H/C": "hot_cold_abbreviation",  # Hot/Cold abbreviation with slash
    }

    @classmethod
    def fix_propbank_xml(cls, xml_content: str, filename: str) -> str:
        """Apply XML fixes for known PropBank issues.

        Parameters
        ----------
        xml_content : str
            The XML content to fix.
        filename : str
            The filename to check for specific fixes.

        Returns
        -------
        str
            The fixed XML content.
        """
        if filename in cls.PROPBANK_XML_FIXES:
            for fix in cls.PROPBANK_XML_FIXES[filename]:
                xml_content = xml_content.replace(fix["pattern"], fix["replacement"])
        return xml_content

    @classmethod
    def is_valid_roleset_exception(cls, roleset_id: str) -> bool:
        """Check if a roleset ID matches a known exception pattern.

        Parameters
        ----------
        roleset_id : str
            The roleset ID to check.

        Returns
        -------
        bool
            True if this is a known exception.
        """
        return any(re.match(pattern, roleset_id) for pattern in cls.PROPBANK_ROLESET_EXCEPTIONS)

    @classmethod
    def is_valid_arg_exception(cls, arg_ref: str) -> bool:
        """Check if an argument reference is a known exception.

        Parameters
        ----------
        arg_ref : str
            The argument reference to check.

        Returns
        -------
        bool
            True if this is a known exception.
        """
        return arg_ref in cls.PROPBANK_ARG_EXCEPTIONS

    @classmethod
    def is_valid_abbrev_exception(cls, abbrev: str) -> bool:
        """Check if a FrameNet abbreviation is a known exception.

        Parameters
        ----------
        abbrev : str
            The abbreviation to check.

        Returns
        -------
        bool
            True if this is a known exception.
        """
        return abbrev in cls.FRAMENET_ABBREV_EXCEPTIONS
