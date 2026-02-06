"""Tests for shared type definitions.

This module tests the type definitions, validators, and patterns defined
in glazing.types to ensure they work correctly and maintain consistency.
"""

import re

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from glazing.types import (
    # Regex patterns
    FRAME_ID_PATTERN,
    HEX_COLOR_PATTERN,
    LEMMA_PATTERN,
    VERBNET_CLASS_PATTERN,
    VERBNET_KEY_PATTERN,
    AlignmentType,
    ConflictType,
    # Exceptions
    DataNotLoadedError,
    # Type literals
    DatasetType,
    InvalidReferenceError,
    LogicType,
    # Annotated types
    MappingConfidenceScore,
    MappingConflictError,
    MappingSource,
    MappingType,
    OperationType,
    ValidationError,
    ValidationStatus,
    VersionString,
    # Type guards
    is_dataset_type,
    is_resource_type,
    is_valid_confidence,
)


class TestTypeLiterals:
    """Test literal type definitions."""

    def test_dataset_type_values(self):
        """Test DatasetType guard accepts correct values."""
        # Test valid values
        assert is_dataset_type("framenet")
        assert is_dataset_type("propbank")
        assert is_dataset_type("verbnet")
        assert is_dataset_type("wordnet")

        # Test invalid values
        assert not is_dataset_type("Unknown")
        assert not is_dataset_type("AMR")
        assert not is_dataset_type("FrameNet")  # Capitalized versions are invalid
        assert not is_dataset_type("PropBank")
        assert not is_dataset_type("VerbNet")
        assert not is_dataset_type("WordNet")

    def test_resource_type_values(self):
        """Test ResourceType guard accepts correct values."""
        # Test all valid resource types
        for resource in [
            "verbnet",
            "framenet",
            "wordnet",
            "propbank",
            "VerbNet",  # Also accept capitalized variants
            "FrameNet",
            "WordNet",
            "PropBank",
            "AMR",
            "UMR",
            "Flickr",
            "THYME",
            "Spatial",
        ]:
            assert is_resource_type(resource)

        # Test invalid values
        assert not is_resource_type("Unknown")
        assert not is_resource_type("Other")

    def test_mapping_source_values(self):
        """Test MappingSource values in Pydantic model."""

        class TestModel(BaseModel):
            source: MappingSource

        # Test all valid sources
        for src in [
            "manual",
            "automatic",
            "manual+strict-conv",
            "manualchecks",
            "auto",
            "gold",
            "silver",
            "inherited",
        ]:
            model = TestModel(source=src)
            assert model.source == src

        # Test invalid source
        with pytest.raises(PydanticValidationError):
            TestModel(source="invalid")

    def test_logic_type_values(self):
        """Test LogicType values in Pydantic model."""

        class TestModel(BaseModel):
            logic: LogicType

        # Test valid values
        assert TestModel(logic="or").logic == "or"
        assert TestModel(logic="and").logic == "and"

        # Test invalid value
        with pytest.raises(PydanticValidationError):
            TestModel(logic="xor")

    def test_mapping_type_values(self):
        """Test MappingType values in Pydantic model."""

        class TestModel(BaseModel):
            mapping: MappingType

        # Test all valid types
        for mt in [
            "direct",
            "inherited",
            "inferred",
            "partial",
            "transitive",
            "manual",
            "automatic",
            "hybrid",
        ]:
            model = TestModel(mapping=mt)
            assert model.mapping == mt

        # Test invalid type
        with pytest.raises(PydanticValidationError):
            TestModel(mapping="invalid")

    def test_alignment_type_values(self):
        """Test AlignmentType values in Pydantic model."""

        class TestModel(BaseModel):
            alignment: AlignmentType

        # Test all valid types
        for at in [
            "exact",
            "equivalent",
            "subsumes",
            "subsumed_by",
            "overlaps",
            "related",
            "contradicts",
        ]:
            model = TestModel(alignment=at)
            assert model.alignment == at

        # Test invalid type
        with pytest.raises(PydanticValidationError):
            TestModel(alignment="invalid")

    def test_conflict_type_values(self):
        """Test ConflictType values in Pydantic model."""

        class TestModel(BaseModel):
            conflict: ConflictType

        # Test all valid types
        for ct in ["ambiguous", "contradictory", "version_mismatch", "inheritance"]:
            model = TestModel(conflict=ct)
            assert model.conflict == ct

        # Test invalid type
        with pytest.raises(PydanticValidationError):
            TestModel(conflict="invalid")

    def test_validation_status_values(self):
        """Test ValidationStatus values in Pydantic model."""

        class TestModel(BaseModel):
            status: ValidationStatus

        # Test all valid statuses
        for vs in ["validated", "unvalidated", "disputed", "deprecated"]:
            model = TestModel(status=vs)
            assert model.status == vs

        # Test invalid status
        with pytest.raises(PydanticValidationError):
            TestModel(status="invalid")

    def test_operation_type_values(self):
        """Test OperationType values in Pydantic model."""

        class TestModel(BaseModel):
            operation: OperationType

        # Test all valid operations
        for op in ["search", "load", "convert", "validate", "index", "cache"]:
            model = TestModel(operation=op)
            assert model.operation == op

        # Test invalid operation
        with pytest.raises(PydanticValidationError):
            TestModel(operation="invalid")


class TestAnnotatedTypes:
    """Test annotated types with constraints."""

    def test_mapping_confidence_score(self):
        """Test MappingConfidenceScore validation."""

        # Create a test model using the annotated type
        class TestModel(BaseModel):
            confidence: MappingConfidenceScore

        # Valid scores
        assert TestModel(confidence=0.0).confidence == 0.0
        assert TestModel(confidence=0.5).confidence == 0.5
        assert TestModel(confidence=1.0).confidence == 1.0

        # Invalid scores
        with pytest.raises(PydanticValidationError):
            TestModel(confidence=-0.1)
        with pytest.raises(PydanticValidationError):
            TestModel(confidence=1.1)

    def test_version_string(self):
        """Test VersionString validation."""

        class TestModel(BaseModel):
            version: VersionString

        # Valid versions
        assert TestModel(version="1.0.0").version == "1.0.0"
        assert TestModel(version="2.1").version == "2.1"
        assert TestModel(version="1.0.0-alpha").version == "1.0.0-alpha"
        assert TestModel(version="3.2.1-beta2").version == "3.2.1-beta2"

        # Invalid versions
        with pytest.raises(PydanticValidationError):
            TestModel(version="1")  # Missing minor version
        with pytest.raises(PydanticValidationError):
            TestModel(version="a.b.c")  # Non-numeric
        with pytest.raises(PydanticValidationError):
            TestModel(version="1.0.0.0")  # Too many parts


class TestRegexPatterns:
    """Test regex patterns for identifier validation."""

    def test_frame_id_pattern(self):
        """Test FrameNet frame ID pattern."""
        pattern = re.compile(FRAME_ID_PATTERN)

        # Valid IDs
        assert pattern.match("1")
        assert pattern.match("123")
        assert pattern.match("9999")

        # Invalid IDs
        assert not pattern.match("abc")
        assert not pattern.match("1.0")
        assert not pattern.match("-1")
        assert not pattern.match("1a")

    def test_verbnet_class_pattern(self):
        """Test VerbNet class ID pattern."""
        pattern = re.compile(VERBNET_CLASS_PATTERN)

        # Valid class IDs
        assert pattern.match("give-13.1")
        assert pattern.match("give-13.1-1")
        assert pattern.match("leave-51.2")
        assert pattern.match("transfer-11.1-1-2")
        assert pattern.match("be_located_at-47.3")  # Fixed: double underscore

        # Invalid class IDs
        assert not pattern.match("give")
        assert not pattern.match("13.1")
        assert not pattern.match("Give-13.1")  # Capital letter
        assert not pattern.match("give-")

    def test_verbnet_key_pattern(self):
        """Test VerbNet member key pattern."""
        pattern = re.compile(VERBNET_KEY_PATTERN)

        # Valid keys
        assert pattern.match("give#2")
        assert pattern.match("abandon#1")
        assert pattern.match("run_up#3")
        assert pattern.match("be-located-at#1")

        # Invalid keys
        assert not pattern.match("give")
        assert not pattern.match("give#")
        assert not pattern.match("Give#2")  # Capital letter
        assert not pattern.match("give#two")

    def test_lemma_pattern(self):
        """Test lemma pattern."""
        pattern = re.compile(LEMMA_PATTERN)

        # Valid lemmas
        assert pattern.match("give")
        assert pattern.match("abandon")
        assert pattern.match("run_up")
        assert pattern.match("don't")
        assert pattern.match("mother-in-law")
        assert pattern.match("Give")  # Capital letter allowed
        assert pattern.match("123run")  # Digit start allowed
        assert pattern.match("Dr.")  # Dots allowed

        # Invalid lemmas
        assert not pattern.match("")  # Empty string
        assert not pattern.match(" give")  # Leading space

    def test_hex_color_pattern(self):
        """Test hex color pattern."""
        pattern = re.compile(HEX_COLOR_PATTERN)

        # Valid colors
        assert pattern.match("FF0000")
        assert pattern.match("00FF00")
        assert pattern.match("0000FF")
        assert pattern.match("ABCDEF")
        assert pattern.match("123456")

        # Lowercase and # prefix are now allowed
        assert pattern.match("ff0000")  # Lowercase is now allowed
        assert pattern.match("#FF0000")  # Hash prefix is now allowed

        # Invalid colors
        assert not pattern.match("FF00")  # Too short
        assert not pattern.match("FF00000")  # Too long
        assert not pattern.match("GGGGGG")  # Invalid hex


class TestTypeGuards:
    """Test type guard functions."""

    def test_is_dataset_type(self):
        """Test is_dataset_type guard."""
        # Valid dataset types
        assert is_dataset_type("framenet")
        assert is_dataset_type("propbank")
        assert is_dataset_type("verbnet")
        assert is_dataset_type("wordnet")

        # Invalid dataset types
        assert not is_dataset_type("AMR")
        assert not is_dataset_type("FrameNet")  # Capitalized versions are invalid
        assert not is_dataset_type("PropBank")
        assert not is_dataset_type("VerbNet")
        assert not is_dataset_type("WordNet")
        assert not is_dataset_type("Unknown")
        assert not is_dataset_type("")

    def test_is_resource_type(self):
        """Test is_resource_type guard."""
        # Valid resource types - both lowercase and capitalized versions
        assert is_resource_type("framenet")
        assert is_resource_type("propbank")
        assert is_resource_type("verbnet")
        assert is_resource_type("wordnet")
        assert is_resource_type("FrameNet")
        assert is_resource_type("PropBank")
        assert is_resource_type("VerbNet")
        assert is_resource_type("WordNet")
        assert is_resource_type("Framenet")  # PropBank variant
        assert is_resource_type("AMR")
        assert is_resource_type("UMR")
        assert is_resource_type("Flickr")
        assert is_resource_type("THYME")
        assert is_resource_type("Spatial")

        # Invalid resource types
        assert not is_resource_type("Unknown")
        assert not is_resource_type("")

    def test_is_valid_confidence(self):
        """Test is_valid_confidence guard."""
        # Valid confidence scores
        assert is_valid_confidence(0.0)
        assert is_valid_confidence(0.5)
        assert is_valid_confidence(1.0)
        assert is_valid_confidence(0.95)
        assert is_valid_confidence(0.001)

        # Invalid confidence scores
        assert not is_valid_confidence(-0.1)
        assert not is_valid_confidence(1.1)
        assert not is_valid_confidence(2.0)
        assert not is_valid_confidence(-1.0)


class TestExceptions:
    """Test custom exception classes."""

    def test_data_not_loaded_error(self):
        """Test DataNotLoadedError exception."""
        with pytest.raises(DataNotLoadedError):
            raise DataNotLoadedError("Data not loaded")

    def test_invalid_reference_error(self):
        """Test InvalidReferenceError exception."""
        with pytest.raises(InvalidReferenceError):
            raise InvalidReferenceError("Invalid reference")

    def test_mapping_conflict_error(self):
        """Test MappingConflictError exception."""
        with pytest.raises(MappingConflictError):
            raise MappingConflictError("Mapping conflict")

    def test_validation_error(self):
        """Test ValidationError exception."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed")


class TestTypeIntegration:
    """Test type integration with Pydantic models."""

    def test_types_with_pydantic_model(self):
        """Test that types work correctly in Pydantic models."""

        class TestMapping(BaseModel):
            source_dataset: DatasetType
            target_dataset: DatasetType
            mapping_type: MappingType
            confidence: MappingConfidenceScore
            source: MappingSource
            status: ValidationStatus

        # Valid model
        mapping = TestMapping(
            source_dataset="framenet",
            target_dataset="verbnet",
            mapping_type="direct",
            confidence=0.95,
            source="manual",
            status="validated",
        )

        assert mapping.source_dataset == "framenet"
        assert mapping.confidence == 0.95

        # Invalid dataset type
        with pytest.raises(PydanticValidationError):
            TestMapping(
                source_dataset="Unknown",  # Invalid
                target_dataset="verbnet",
                mapping_type="direct",
                confidence=0.95,
                source="manual",
                status="validated",
            )

        # Invalid confidence
        with pytest.raises(PydanticValidationError):
            TestMapping(
                source_dataset="framenet",
                target_dataset="verbnet",
                mapping_type="direct",
                confidence=1.5,  # Invalid
                source="manual",
                status="validated",
            )
