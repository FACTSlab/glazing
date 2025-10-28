"""Tests for FrameNet type definitions.

This module tests all FrameNet-specific type definitions including
literal types, type aliases, and validation functions.
"""

import re
from typing import get_args

from glazing.framenet.types import (
    FE_ABBREV_PATTERN,
    FE_NAME_PATTERN,
    FRAME_NAME_PATTERN,
    HEX_COLOR_PATTERN,
    LEXEME_NAME_PATTERN,
    LU_NAME_PATTERN,
    MAX_ANNOTATION_LAYERS,
    MAX_FRAME_ELEMENTS,
    MAX_LEXICAL_UNITS,
    USERNAME_PATTERN,
    AnnotationStatus,
    CoreType,
    CorpusName,
    DefinitionPrefix,
    FrameNetPOS,
    FrameRelationSubType,
    FrameRelationType,
    GrammaticalFunction,
    InstantiationType,
    LayerType,
    MarkupType,
    PhraseType,
    is_valid_fe_abbrev,
    is_valid_fe_name,
    is_valid_frame_name,
    is_valid_hex_color,
    is_valid_lu_name,
    is_valid_username,
)


def get_type_args(type_alias):
    """Get arguments from a type alias created with Python 3.13+ type statement."""
    if hasattr(type_alias, "__value__"):
        # Python 3.13+ type statement
        return get_args(type_alias.__value__)
    # Regular type alias
    return get_args(type_alias)


class TestCoreTypes:
    """Test core FrameNet type literals."""

    def test_core_type_values(self):
        """Test that CoreType has all expected values."""
        expected = {"Core", "Core-Unexpressed", "Peripheral", "Extra-Thematic"}
        actual = set(get_type_args(CoreType))
        assert actual == expected

    def test_framenet_pos_values(self):
        """Test that FrameNetPOS has all expected values."""
        expected = {
            "A",
            "ADV",
            "ART",
            "AVP",
            "C",
            "IDIO",
            "INTJ",
            "N",
            "NUM",
            "PREP",
            "PRON",
            "SCON",
            "V",
        }
        actual = set(get_type_args(FrameNetPOS))
        assert actual == expected

    def test_annotation_status_values(self):
        """Test that AnnotationStatus has values from actual data."""
        # These are verified from actual FrameNet v1.7 data
        expected_subset = {
            "UNANN",
            "MANUAL",
            "FN1_Sent",
            "Rules_Defined",
            "Finished_Initial",
            "Insufficient_Attestations",
            "Created",
        }
        actual = set(get_type_args(AnnotationStatus))
        assert expected_subset.issubset(actual)
        assert len(actual) >= 10  # Should have at least 10 status values


class TestRelationTypes:
    """Test frame relation type literals."""

    def test_frame_relation_type_values(self):
        """Test that FrameRelationType has all expected values."""
        expected = {
            "Inherits from",
            "Is Inherited by",
            "Perspective on",
            "Is Perspectivized in",
            "Uses",
            "Is Used by",
            "Subframe of",
            "Has Subframe(s)",
            "Precedes",
            "Is Preceded by",
            "Is Inchoative of",
            "Is Causative of",
            "See also",
        }
        actual = set(get_type_args(FrameRelationType))
        assert actual == expected

    def test_frame_relation_subtype_values(self):
        """Test that FrameRelationSubType has all expected values."""
        expected = {"Mapping", "Inheritance", "Equivalence"}
        actual = set(get_type_args(FrameRelationSubType))
        assert actual == expected


class TestAnnotationTypes:
    """Test annotation-related type literals."""

    def test_layer_type_count(self):
        """Test that LayerType has expected number of values."""
        layer_types = get_type_args(LayerType)
        # We have 21 verified layer types from actual data
        assert len(layer_types) == 21

    def test_layer_type_core_values(self):
        """Test that LayerType contains core annotation layers."""
        layer_types = set(get_type_args(LayerType))
        core_layers = {"FE", "GF", "PT", "Target", "Verb", "Noun", "Adj", "Adv"}
        assert core_layers.issubset(layer_types)

    def test_grammatical_function_values(self):
        """Test that GrammaticalFunction has values from actual data."""
        expected = {
            "Dep",
            "Ext",
            "Obj",
            "Head",
            "Gen",
            "Quant",
            "Appositive",  # Actual value in data
        }
        actual = set(get_type_args(GrammaticalFunction))
        assert actual == expected

    def test_phrase_type_count(self):
        """Test that PhraseType has expected number of values."""
        phrase_types = get_type_args(PhraseType)
        # We have 38 core types (100+ parameterized PP types exist but not included)
        assert len(phrase_types) == 38

    def test_phrase_type_core_values(self):
        """Test that PhraseType contains core phrase types."""
        phrase_types = set(get_type_args(PhraseType))
        core_types = {"NP", "AVP", "AJP", "Sfin", "CNI", "DNI", "INI"}
        assert core_types.issubset(phrase_types)


class TestSpecialTypes:
    """Test special FrameNet type literals."""

    def test_instantiation_type_values(self):
        """Test that InstantiationType has all expected values."""
        expected = {"INI", "DNI", "CNI", "INC"}
        actual = set(get_type_args(InstantiationType))
        assert actual == expected

    def test_markup_type_values(self):
        """Test that MarkupType has all expected values."""
        expected = {"def-root", "fex", "fen", "t", "ex", "m", "gov", "x"}
        actual = set(get_type_args(MarkupType))
        assert actual == expected

    def test_definition_prefix_values(self):
        """Test that DefinitionPrefix has expected values."""
        expected = {"COD", "FN"}
        actual = set(get_type_args(DefinitionPrefix))
        assert actual == expected

    def test_corpus_name_values(self):
        """Test that CorpusName has expected corpus names."""
        corpus_names = set(get_type_args(CorpusName))
        # From actual data
        expected_subset = {"ANC", "PropBank", "KBEval", "LUCorpus-v0.3", "Miscellaneous", "NTI"}
        assert expected_subset.issubset(corpus_names)


class TestValidationFunctions:
    """Test validation helper functions."""

    def test_valid_frame_names(self):
        """Test frame name validation with valid names."""
        valid_names = [
            "Abandonment",
            "Activity_finish",
            "Body_parts",
            "Change_of_leadership",
            "A",
            "ABC123",
            "Test_Frame_123",
        ]
        for name in valid_names:
            assert is_valid_frame_name(name), f"Should accept valid frame name: {name}"

    def test_invalid_frame_names(self):
        """Test frame name validation with invalid names."""
        # Based on actual patterns, lowercase/hyphens are now allowed
        invalid_names = [
            "Frame@Name",  # @ not allowed
            "Frame!Name",  # ! not allowed
            "Frame#Name",  # # not allowed
            "Frame$Name",  # $ not allowed
            "",  # Empty not allowed
        ]
        for name in invalid_names:
            assert not is_valid_frame_name(name), f"Should reject invalid frame name: {name}"

    def test_valid_fe_names(self):
        """Test FE name validation with valid names."""
        valid_names = ["Agent", "Theme", "Body_part", "Co_participant", "A", "FE123"]
        for name in valid_names:
            assert is_valid_fe_name(name), f"Should accept valid FE name: {name}"

    def test_invalid_fe_names(self):
        """Test FE name validation with invalid names."""
        # Based on actual patterns, lowercase/hyphens/spaces are now allowed
        invalid_names = ["FE@Name", "FE!Name", "FE#Name", "FE$Name", ""]
        for name in invalid_names:
            assert not is_valid_fe_name(name), f"Should reject invalid FE name: {name}"

    def test_valid_fe_abbrevs(self):
        """Test FE abbreviation validation with valid abbreviations."""
        valid_abbrevs = ["Agt", "Thm", "BodP", "Co-Part", "A1", "FE_123"]
        for abbrev in valid_abbrevs:
            assert is_valid_fe_abbrev(abbrev), f"Should accept valid FE abbrev: {abbrev}"

    def test_invalid_fe_abbrevs(self):
        """Test FE abbreviation validation with invalid abbreviations."""
        invalid_abbrevs = ["123Agt", "_Agt", "", " Agt", "Agt "]
        for abbrev in invalid_abbrevs:
            assert not is_valid_fe_abbrev(abbrev), f"Should reject invalid FE abbrev: {abbrev}"

    def test_valid_lu_names(self):
        """Test LU name validation with valid names."""
        valid_names = [
            "abandon.v",
            "give_up.v",
            "look.n",
            "beautiful.a",
            "quickly.adv",
            "test-word.prep",
            "it's.pron",
        ]
        for name in valid_names:
            assert is_valid_lu_name(name), f"Should accept valid LU name: {name}"

    def test_invalid_lu_names(self):
        """Test LU name validation with invalid names."""
        # Real FrameNet data is very permissive (proper nouns, spaces, special chars, etc.)
        # Only reject truly invalid patterns: no .pos suffix, empty, or just a dot
        invalid_names = ["abandon", "abandon.", ".v", "", "test."]
        for name in invalid_names:
            assert not is_valid_lu_name(name), f"Should reject invalid LU name: {name}"

    def test_valid_usernames(self):
        """Test username validation with valid usernames."""
        valid_usernames = ["KmG", "CHg", "JKR", "User123", "A", "TestUser"]
        for username in valid_usernames:
            assert is_valid_username(username), f"Should accept valid username: {username}"

    def test_invalid_usernames(self):
        """Test username validation with invalid usernames."""
        invalid_usernames = ["123User", "_User", "User-Name", "User Name", "", "user@email"]
        for username in invalid_usernames:
            assert not is_valid_username(username), f"Should reject invalid username: {username}"

    def test_valid_hex_colors(self):
        """Test hex color validation with valid colors."""
        valid_colors = ["FFFFFF", "000000", "FF0000", "00FF00", "0000FF", "ABC123"]
        for color in valid_colors:
            assert is_valid_hex_color(color), f"Should accept valid hex color: {color}"

    def test_invalid_hex_colors(self):
        """Test hex color validation with invalid colors."""
        invalid_colors = ["FFF", "GGGGGG", "FF00FF0", "", "FF 00 FF"]
        for color in invalid_colors:
            assert not is_valid_hex_color(color), f"Should reject invalid hex color: {color}"


class TestPatternConstants:
    """Test regex pattern constants."""

    def test_frame_name_pattern(self):
        """Test FRAME_NAME_PATTERN regex."""
        pattern = re.compile(FRAME_NAME_PATTERN)
        assert pattern.match("Abandonment")
        assert pattern.match("Activity_finish")
        assert pattern.match("abandonment")
        assert pattern.match("Activity-finish")
        assert not pattern.match("Frame@Name")
        assert not pattern.match("Frame!Name")

    def test_fe_name_pattern(self):
        """Test FE_NAME_PATTERN regex."""
        pattern = re.compile(FE_NAME_PATTERN)
        assert pattern.match("Agent")
        assert pattern.match("Body_part")
        assert pattern.match("agent")
        assert pattern.match("Body part")
        assert pattern.match("Person's")
        assert not pattern.match("FE@Name")

    def test_fe_abbrev_pattern(self):
        """Test FE_ABBREV_PATTERN regex."""
        pattern = re.compile(FE_ABBREV_PATTERN)
        assert pattern.match("Agt")
        assert pattern.match("Co-Part")
        assert pattern.match("H/C")
        assert not pattern.match("Agt@Bad")

    def test_lu_name_pattern(self):
        """Test LU_NAME_PATTERN regex."""
        pattern = re.compile(LU_NAME_PATTERN, re.IGNORECASE)
        assert pattern.match("abandon.v")
        assert pattern.match("give_up.v")
        assert not pattern.match("abandon")

    def test_username_pattern(self):
        """Test USERNAME_PATTERN regex."""
        pattern = re.compile(USERNAME_PATTERN)
        assert pattern.match("KmG")
        assert pattern.match("User123")
        assert not pattern.match("123User")

    def test_hex_color_pattern(self):
        """Test HEX_COLOR_PATTERN regex."""
        pattern = re.compile(HEX_COLOR_PATTERN)
        assert pattern.match("FFFFFF")
        assert pattern.match("ABC123")
        assert pattern.match("ffffff")  # Lowercase is now allowed
        assert pattern.match("#FFFFFF")  # Hash prefix is now allowed
        assert not pattern.match("FFF")  # Too short
        assert not pattern.match("GGGGGG")  # Invalid hex chars

    def test_lexeme_name_pattern(self):
        """Test LEXEME_NAME_PATTERN regex."""
        # Lexeme names are very permissive in real FrameNet data (spaces, digits, symbols)
        # Examples from real data: "a bit", "Boxing Day", "Scud-B missile", "(can't"
        pattern = re.compile(LEXEME_NAME_PATTERN)
        assert pattern.match("abandon")
        assert pattern.match("give")
        assert pattern.match("it's")
        assert pattern.match("123word")  # Real FrameNet has numbers
        assert pattern.match("a bit")  # Real FrameNet multi-word expressions
        assert not pattern.match("")  # Only empty string is invalid


class TestConstants:
    """Test numeric constants and limits."""

    def test_max_values(self):
        """Test that maximum values are defined and reasonable."""
        assert MAX_FRAME_ELEMENTS == 100
        assert MAX_ANNOTATION_LAYERS == 50
        assert MAX_LEXICAL_UNITS == 1000

    def test_max_values_types(self):
        """Test that maximum values are integers."""
        assert isinstance(MAX_FRAME_ELEMENTS, int)
        assert isinstance(MAX_ANNOTATION_LAYERS, int)
        assert isinstance(MAX_LEXICAL_UNITS, int)


class TestTypeCoverage:
    """Test completeness of type definitions."""

    def test_no_duplicate_values_in_literals(self):
        """Test that literal types don't have duplicate values."""
        literal_types = [
            CoreType,
            FrameNetPOS,
            AnnotationStatus,
            FrameRelationType,
            LayerType,
            GrammaticalFunction,
            PhraseType,
            InstantiationType,
            MarkupType,
            DefinitionPrefix,
        ]

        for literal_type in literal_types:
            values = get_type_args(literal_type)
            assert len(values) == len(set(values)), f"Duplicate values in {literal_type.__name__}"

    def test_layer_type_completeness(self):
        """Test that LayerType covers all major categories."""
        layer_types = set(get_type_args(LayerType))

        # Core layers
        assert "FE" in layer_types
        assert "GF" in layer_types
        assert "PT" in layer_types
        assert "Target" in layer_types

        # POS layers
        pos_layers = {"Verb", "Noun", "Adj", "Adv", "Prep", "Other"}
        assert pos_layers.issubset(layer_types)

        # Corpus layers
        corpus_layers = {"NER", "WSL", "BNC", "PENN"}
        assert corpus_layers.issubset(layer_types)

    def test_phrase_type_completeness(self):
        """Test that PhraseType covers all major categories."""
        phrase_types = set(get_type_args(PhraseType))

        # Basic phrase types (note: PP is parameterized, not a simple type)
        basic_types = {"NP", "AVP", "AJP"}
        assert basic_types.issubset(phrase_types)

        # Verb phrase variants
        vp_variants = {"VPfin", "VPing", "VPto", "VPbrst"}
        assert vp_variants.issubset(phrase_types)

        # Sentence types
        sentence_types = {"Sfin", "Sing", "Sto", "Sbrst", "Swhether", "Sinterrog", "Srel"}
        assert sentence_types.issubset(phrase_types)

        # Null instantiation types
        null_types = {"CNI", "DNI", "INI"}
        assert null_types.issubset(phrase_types)
