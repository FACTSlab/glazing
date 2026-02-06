"""Tests for the base models and utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from glazing.base import (
    ConflictResolution,
    CrossReferenceBase,
    GlazingBaseModel,
    MappingBase,
    validate_confidence_score,
    validate_fe_name,
    validate_frame_id,
    validate_frame_name,
    validate_hex_color,
    validate_lemma,
    validate_percentage_notation,
    validate_propbank_roleset,
    validate_verbnet_class,
    validate_verbnet_key,
    validate_wordnet_offset,
    validate_wordnet_sense_key,
)


class TestGlazingBaseModel:
    """Test the GlazingBaseModel class."""

    def test_basic_model(self):
        """Test basic model creation and validation."""

        class TestModel(GlazingBaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        assert model.name == "test"
        assert model.value == 42

    def test_to_jsonl(self):
        """Test JSON Lines export."""

        class TestModel(GlazingBaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        jsonl = model.to_jsonl()

        # Parse back to verify
        data = json.loads(jsonl)
        assert data["name"] == "test"
        assert data["value"] == 42

    def test_from_jsonl(self):
        """Test JSON Lines import."""

        class TestModel(GlazingBaseModel):
            name: str
            value: int

        jsonl = '{"name": "test", "value": 42}'
        model = TestModel.from_jsonl(jsonl)

        assert model.name == "test"
        assert model.value == 42

    def test_jsonl_round_trip(self):
        """Test JSON Lines export and import round trip."""

        class TestModel(GlazingBaseModel):
            name: str
            value: int
            optional: str | None = None

        original = TestModel(name="test", value=42, optional="extra")
        jsonl = original.to_jsonl()
        loaded = TestModel.from_jsonl(jsonl)

        assert loaded.name == original.name
        assert loaded.value == original.value
        assert loaded.optional == original.optional

    def test_json_lines_file(self, tmp_path):
        """Test reading and writing JSON Lines files."""

        class TestModel(GlazingBaseModel):
            name: str
            value: int

        # Write to file
        model = TestModel(name="test", value=42)
        file_path = tmp_path / "test.jsonl"
        model.to_json_lines_file(file_path)

        # Read from file
        models = list(TestModel.from_json_lines_file(file_path))
        assert len(models) == 1
        assert models[0].name == "test"
        assert models[0].value == 42

    def test_validate_many(self):
        """Test batch validation."""

        class TestModel(GlazingBaseModel):
            name: str
            value: int

        items = [
            {"name": "test1", "value": 1},
            {"name": "test2", "value": 2},
            {"name": "test3"},  # Missing value
            {"name": "test4", "value": 4},
        ]

        results = TestModel.validate_many(items)

        assert len(results) == 4
        assert results[0][1] is None  # No error
        assert results[1][1] is None  # No error
        assert results[2][0] is None  # Error - missing value
        assert isinstance(results[2][1], ValidationError)
        assert results[3][1] is None  # No error

    def test_populate_by_name(self):
        """Test field alias support."""

        class TestModel(GlazingBaseModel):
            my_field: str

        # Should accept both field name and alias
        model1 = TestModel(my_field="value")
        assert model1.my_field == "value"

        # Test with dict
        model2 = TestModel.model_validate({"my_field": "value"})
        assert model2.my_field == "value"


class TestCrossReferenceBase:
    """Test the CrossReferenceBase class."""

    def test_basic_cross_reference(self):
        """Test basic cross-reference creation."""
        ref = CrossReferenceBase(
            source_dataset="framenet",
            source_id="frame_123",
            target_dataset="propbank",
            target_id="give.01",
            mapping_type="direct",
            confidence=0.95,
        )

        assert ref.source_dataset == "framenet"
        assert ref.source_id == "frame_123"
        assert ref.target_dataset == "propbank"
        assert ref.target_id == "give.01"
        assert ref.confidence == 0.95

    def test_multiple_target_ids(self):
        """Test cross-reference with multiple targets."""
        ref = CrossReferenceBase(
            source_dataset="verbnet",
            source_id="give-13.1",
            target_dataset="propbank",
            target_id=["give.01", "give.02"],
            mapping_type="direct",
        )

        assert isinstance(ref.target_id, list)
        assert len(ref.target_id) == 2
        assert "give.01" in ref.target_id

    def test_empty_id_validation(self):
        """Test that empty IDs are rejected."""
        with pytest.raises(ValidationError):
            CrossReferenceBase(
                source_dataset="framenet",
                source_id="",  # Empty ID
                target_dataset="propbank",
                target_id="give.01",
            )

        with pytest.raises(ValidationError):
            CrossReferenceBase(
                source_dataset="framenet",
                source_id="frame_123",
                target_dataset="propbank",
                target_id=[],  # Empty list
            )

    def test_same_dataset_validation(self):
        """Test that same source and target datasets are rejected for direct mappings."""
        # Should fail for direct mapping
        with pytest.raises(ValidationError):
            CrossReferenceBase(
                source_dataset="framenet",
                source_id="frame_123",
                target_dataset="framenet",
                target_id="frame_456",
                mapping_type="direct",
            )

        # Should succeed for inherited mapping
        ref = CrossReferenceBase(
            source_dataset="framenet",
            source_id="frame_123",
            target_dataset="framenet",
            target_id="frame_456",
            mapping_type="inherited",
        )
        assert ref.source_dataset == ref.target_dataset

    def test_confidence_methods(self):
        """Test confidence score methods."""
        # With confidence
        ref1 = CrossReferenceBase(
            source_dataset="framenet",
            source_id="frame_123",
            target_dataset="propbank",
            target_id="give.01",
            confidence=0.85,
        )
        assert ref1.get_confidence_score() == 0.85
        assert ref1.is_high_confidence(threshold=0.8)
        assert not ref1.is_high_confidence(threshold=0.9)

        # Without confidence
        ref2 = CrossReferenceBase(
            source_dataset="framenet",
            source_id="frame_123",
            target_dataset="propbank",
            target_id="give.01",
        )
        assert ref2.get_confidence_score() == 0.5  # Default
        assert not ref2.is_high_confidence(threshold=0.8)


class TestMappingBase:
    """Test the MappingBase class."""

    def test_basic_mapping(self):
        """Test basic mapping creation."""
        mapping = MappingBase(created_by="test_user", version="1.0.0")

        assert mapping.created_by == "test_user"
        assert mapping.version == "1.0.0"
        assert mapping.validation_status == "unvalidated"
        assert isinstance(mapping.created_date, datetime)

    def test_modification_validation(self):
        """Test that modification fields must be consistent."""
        # Should fail - modified_date without modified_by
        with pytest.raises(ValidationError):
            MappingBase(created_by="test_user", version="1.0.0", modified_date=datetime.now(UTC))

        # Should fail - modified_by without modified_date
        with pytest.raises(ValidationError):
            MappingBase(created_by="test_user", version="1.0.0", modified_by="modifier")

        # Should succeed - both provided
        mapping = MappingBase(
            created_by="test_user",
            version="1.0.0",
            modified_date=datetime.now(UTC),
            modified_by="modifier",
        )
        assert mapping.modified_by == "modifier"

    def test_mark_validated(self):
        """Test marking a mapping as validated."""
        mapping = MappingBase(created_by="test_user", version="1.0.0")

        assert mapping.validation_status == "unvalidated"

        mapping.mark_validated("manual_review", "validator_user")

        assert mapping.validation_status == "validated"
        assert mapping.validation_method == "manual_review"
        assert mapping.modified_by == "validator_user"
        assert mapping.modified_date is not None


class TestValidators:
    """Test the validator functions."""

    def test_validate_frame_id(self):
        """Test FrameNet frame ID validation."""
        assert validate_frame_id(123) == "123"
        assert validate_frame_id("456") == "456"

        with pytest.raises(ValueError):
            validate_frame_id("abc")  # Not numeric

    def test_validate_frame_name(self):
        """Test FrameNet frame name validation."""
        assert validate_frame_name("Abandonment") == "Abandonment"
        assert validate_frame_name("Activity_finish") == "Activity_finish"
        assert validate_frame_name("abandonment") == "abandonment"
        assert validate_frame_name("Activity-finish") == "Activity-finish"

        with pytest.raises(ValueError):
            validate_frame_name("Activity finish!")

    def test_validate_fe_name(self):
        """Test frame element name validation."""
        assert validate_fe_name("Agent") == "Agent"
        assert validate_fe_name("Body_part") == "Body_part"
        assert validate_fe_name("agent") == "agent"
        assert validate_fe_name("Body part") == "Body part"
        assert validate_fe_name("Person's") == "Person's"
        assert validate_fe_name("H.C.") == "H.C."

        with pytest.raises(ValueError):
            validate_fe_name("Agent@123")

    def test_validate_verbnet_class(self):
        """Test VerbNet class ID validation."""
        assert validate_verbnet_class("give-13.1") == "give-13.1"
        assert validate_verbnet_class("give-13.1-1") == "give-13.1-1"
        assert validate_verbnet_class("spray-9.7-2-1") == "spray-9.7-2-1"

        with pytest.raises(ValueError):
            validate_verbnet_class("Give-13.1")  # Uppercase
        with pytest.raises(ValueError):
            validate_verbnet_class("give_13.1")  # Underscore instead of hyphen

    def test_validate_verbnet_key(self):
        """Test VerbNet member key validation."""
        assert validate_verbnet_key("give#2") == "give#2"
        assert validate_verbnet_key("spray_paint#1") == "spray_paint#1"

        with pytest.raises(ValueError):
            validate_verbnet_key("give2")  # Missing #
        with pytest.raises(ValueError):
            validate_verbnet_key("give#")  # Missing number

    def test_validate_propbank_roleset(self):
        """Test PropBank roleset ID validation."""
        assert validate_propbank_roleset("give.01") == "give.01"
        assert validate_propbank_roleset("be-located-at.91") == "be-located-at.91"

        with pytest.raises(ValueError):
            validate_propbank_roleset("give01")  # Missing dot
        with pytest.raises(ValueError):
            validate_propbank_roleset("give.1a")  # Non-numeric sense

    def test_validate_wordnet_offset(self):
        """Test WordNet synset offset validation."""
        assert validate_wordnet_offset("00001740") == "00001740"
        assert validate_wordnet_offset("12345678") == "12345678"

        with pytest.raises(ValueError):
            validate_wordnet_offset("1740")  # Too short
        with pytest.raises(ValueError):
            validate_wordnet_offset("0000174X")  # Non-numeric

    def test_validate_wordnet_sense_key(self):
        """Test WordNet sense key validation."""
        assert validate_wordnet_sense_key("abandon%2:40:01::") == "abandon%2:40:01::"
        assert validate_wordnet_sense_key("dog%1:05:00::") == "dog%1:05:00::"

        with pytest.raises(ValueError):
            validate_wordnet_sense_key("abandon:2:40:01::")  # Missing %
        with pytest.raises(ValueError):
            validate_wordnet_sense_key("abandon%6:40:01::")  # Invalid POS (6)

    def test_validate_percentage_notation(self):
        """Test VerbNet's WordNet percentage notation."""
        assert validate_percentage_notation("give%2:40:00") == "give%2:40:00"
        assert validate_percentage_notation("abandon%2:40:01") == "abandon%2:40:01"

        with pytest.raises(ValueError):
            validate_percentage_notation("give%2:40:00::")  # Too many parts
        with pytest.raises(ValueError):
            validate_percentage_notation("give%2:40")  # Too few parts

    def test_validate_lemma(self):
        """Test lemma validation."""
        assert validate_lemma("abandon") == "abandon"
        assert validate_lemma("spray_paint") == "spray_paint"
        assert validate_lemma("don't") == "don't"
        assert validate_lemma("Abandon") == "Abandon"  # Uppercase allowed
        assert validate_lemma("123abandon") == "123abandon"  # Digit start allowed
        assert validate_lemma("Dr.") == "Dr."  # Dots allowed

        with pytest.raises(ValueError):
            validate_lemma("")  # Empty
        with pytest.raises(ValueError):
            validate_lemma(" abandon")  # Leading space

    def test_validate_hex_color(self):
        """Test hex color validation."""
        assert validate_hex_color("FF0000") == "FF0000"
        assert validate_hex_color("00FF00") == "00FF00"
        assert validate_hex_color("0000FF") == "0000FF"
        assert validate_hex_color("ff0000") == "ff0000"
        assert validate_hex_color("#FF0000") == "#FF0000"

        with pytest.raises(ValueError):
            validate_hex_color("FF00")
        with pytest.raises(ValueError):
            validate_hex_color("GG0000")

    def test_validate_confidence_score(self):
        """Test confidence score validation."""
        assert validate_confidence_score(0.0) == 0.0
        assert validate_confidence_score(0.5) == 0.5
        assert validate_confidence_score(1.0) == 1.0

        with pytest.raises(ValueError):
            validate_confidence_score(-0.1)  # Below 0
        with pytest.raises(ValueError):
            validate_confidence_score(1.1)  # Above 1


class TestConflictResolution:
    """Test the ConflictResolution class."""

    def test_basic_conflict_resolution(self):
        """Test basic conflict resolution creation."""
        ref1 = CrossReferenceBase(
            source_dataset="framenet",
            source_id="frame_123",
            target_dataset="propbank",
            target_id="give.01",
            confidence=0.8,
        )

        ref2 = CrossReferenceBase(
            source_dataset="framenet",
            source_id="frame_123",
            target_dataset="propbank",
            target_id="transfer.01",
            confidence=0.7,
        )

        resolution = ConflictResolution(
            conflict_type="ambiguous",
            resolution_strategy="highest_confidence",
            selected_mapping=ref1,
            rejected_mappings=[ref2],
            resolution_confidence=0.9,
        )

        assert resolution.conflict_type == "ambiguous"
        assert resolution.selected_mapping == ref1
        assert len(resolution.rejected_mappings) == 1
        assert resolution.rejected_mappings[0] == ref2

    def test_resolution_validation(self):
        """Test that resolution must have mappings."""
        with pytest.raises(ValidationError):
            ConflictResolution(
                conflict_type="ambiguous",
                resolution_strategy="manual",
                resolution_confidence=0.9,
                # No selected or rejected mappings
            )
