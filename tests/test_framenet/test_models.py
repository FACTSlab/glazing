"""Tests for FrameNet core models.

This module contains comprehensive tests for the FrameNet data models including
validation, markup parsing, and constraint checking.
"""

from datetime import UTC, datetime

import pytest

from glazing.framenet.models import (
    AnnotatedText,
    AnnotationLayer,
    AnnotationSet,
    FEGroupRealization,
    FERealization,
    FERelation,
    Frame,
    FrameElement,
    FrameIndexEntry,
    FrameRelation,
    Label,
    Lexeme,
    LexicalUnit,
    SemanticType,
    SemTypeRef,
    Sentence,
    SentenceCount,
    TextAnnotation,
    ValenceAnnotationPattern,
    ValencePattern,
    ValenceRealizationPattern,
    ValenceUnit,
)


class TestTextAnnotation:
    """Test cases for TextAnnotation model."""

    def test_valid_annotation(self):
        """Test creating a valid text annotation."""
        annotation = TextAnnotation(
            start=0,
            end=5,
            type="fex",
            name="Agent",
            text="Agent",
        )
        assert annotation.start == 0
        assert annotation.end == 5
        assert annotation.type == "fex"
        assert annotation.name == "Agent"
        assert annotation.text == "Agent"

    def test_annotation_length(self):
        """Test annotation length calculation."""
        annotation = TextAnnotation(start=10, end=15, type="fex", name="Theme", text="theme")
        assert annotation.get_length() == 5

    def test_annotation_overlap(self):
        """Test annotation overlap detection."""
        ann1 = TextAnnotation(start=0, end=5, type="fex", name="Agent", text="Agent")
        ann2 = TextAnnotation(start=3, end=8, type="fex", name="Theme", text="Theme")
        ann3 = TextAnnotation(start=10, end=15, type="fex", name="Goal", text="Goal")

        assert ann1.overlaps_with(ann2)
        assert ann2.overlaps_with(ann1)
        assert not ann1.overlaps_with(ann3)
        assert not ann3.overlaps_with(ann1)

    def test_invalid_positions(self):
        """Test validation of invalid positions."""
        with pytest.raises(ValueError, match=r"End position.*must be at or after start position"):
            TextAnnotation(start=10, end=5, type="fex", name="Agent", text="Agent")

    def test_negative_positions(self):
        """Test validation of negative positions."""
        with pytest.raises(ValueError):
            TextAnnotation(start=-1, end=5, type="fex", name="Agent", text="Agent")

    def test_fex_requires_name(self):
        """Test that fex annotations require a name."""
        with pytest.raises(ValueError, match="requires a name"):
            TextAnnotation(start=0, end=5, type="fex", text="Agent")

    def test_fen_requires_name(self):
        """Test that fen annotations require a name."""
        with pytest.raises(ValueError, match="requires a name"):
            TextAnnotation(start=0, end=5, type="fen", text="Agent")

    def test_target_no_name_required(self):
        """Test that target annotations don't require a name."""
        annotation = TextAnnotation(start=0, end=5, type="t", text="abandon")
        assert annotation.name is None
        assert annotation.type == "t"

    def test_invalid_fe_name_format(self):
        """Test validation of FE name format."""
        with pytest.raises(ValueError, match="Invalid FE name format"):
            TextAnnotation(start=0, end=5, type="fex", name="Agent@Bad", text="Agent")

        with pytest.raises(ValueError, match="Invalid FE name format"):
            TextAnnotation(start=0, end=5, type="fex", name="Agent!Bad", text="Agent")


class TestAnnotatedText:
    """Test cases for AnnotatedText model."""

    def test_empty_text(self):
        """Test parsing empty text."""
        annotated = AnnotatedText.parse("")
        assert annotated.raw_text == ""
        assert annotated.plain_text == ""
        assert len(annotated.annotations) == 0

    def test_text_without_markup(self):
        """Test parsing text without markup."""
        text = "This is plain text without markup"
        annotated = AnnotatedText.parse(text)
        assert annotated.raw_text == text
        assert annotated.plain_text == text
        assert len(annotated.annotations) == 0

    def test_single_fex_annotation(self):
        """Test parsing single fex annotation."""
        text = "An <fex>Agent</fex> performs an action"
        annotated = AnnotatedText.parse(text)

        assert annotated.raw_text == text
        assert annotated.plain_text == "An Agent performs an action"
        assert len(annotated.annotations) == 1

        annotation = annotated.annotations[0]
        assert annotation.type == "fex"
        assert annotation.name == "Agent"  # Content is used as name when not explicit
        assert annotation.text == "Agent"
        assert annotation.start == 3  # Position of "Agent" in plain text
        assert annotation.end == 8

    def test_fex_with_name_attribute(self):
        """Test parsing fex with explicit name attribute."""
        text = "The <fex name='Actor'>person</fex> walks"
        annotated = AnnotatedText.parse(text)

        assert annotated.plain_text == "The person walks"
        assert len(annotated.annotations) == 1

        annotation = annotated.annotations[0]
        assert annotation.type == "fex"
        assert annotation.name == "Actor"
        assert annotation.text == "person"
        assert annotation.start == 4
        assert annotation.end == 10

    def test_multiple_annotations(self):
        """Test parsing multiple annotations."""
        text = (
            "The <fex>Agent</fex> gives the <fex name='Theme'>book</fex> "
            "to the <fex name='Recipient'>child</fex>"
        )
        annotated = AnnotatedText.parse(text)

        expected_plain = "The Agent gives the book to the child"
        assert annotated.plain_text == expected_plain
        assert len(annotated.annotations) == 3

        # Check Agent annotation
        agent = annotated.annotations[0]
        assert agent.type == "fex"
        assert agent.name == "Agent"  # Content used as name
        assert agent.text == "Agent"
        assert agent.start == 4
        assert agent.end == 9

        # Check Theme annotation
        theme = annotated.annotations[1]
        assert theme.type == "fex"
        assert theme.name == "Theme"
        assert theme.text == "book"
        assert theme.start == 20
        assert theme.end == 24

        # Check Recipient annotation
        recipient = annotated.annotations[2]
        assert recipient.type == "fex"
        assert recipient.name == "Recipient"
        assert recipient.text == "child"
        assert recipient.start == 32
        assert recipient.end == 37

    def test_mixed_annotation_types(self):
        """Test parsing different annotation types."""
        text = "The <fex>Agent</fex> <t>abandons</t> the <fen>Theme</fen>"
        annotated = AnnotatedText.parse(text)

        assert annotated.plain_text == "The Agent abandons the Theme"
        assert len(annotated.annotations) == 3

        types = [ann.type for ann in annotated.annotations]
        assert "fex" in types
        assert "t" in types
        assert "fen" in types

    def test_get_annotations_by_type(self):
        """Test filtering annotations by type."""
        text = "The <fex>Agent</fex> <t>abandons</t> the <fex>Theme</fex>"
        annotated = AnnotatedText.parse(text)

        fex_annotations = annotated.get_annotations_by_type("fex")
        assert len(fex_annotations) == 2

        target_annotations = annotated.get_annotations_by_type("t")
        assert len(target_annotations) == 1
        assert target_annotations[0].text == "abandons"

    def test_get_fe_references(self):
        """Test getting FE references."""
        text = "The <fex>Agent</fex> does something to the <fen>Theme</fen>"
        annotated = AnnotatedText.parse(text)

        fe_refs = annotated.get_fe_references()
        assert len(fe_refs) == 2

        types = [ann.type for ann in fe_refs]
        assert "fex" in types
        assert "fen" in types

    def test_get_targets(self):
        """Test getting target annotations."""
        text = "To <t>abandon</t> means to <t>leave</t> behind"
        annotated = AnnotatedText.parse(text)

        targets = annotated.get_targets()
        assert len(targets) == 2
        assert targets[0].text == "abandon"
        assert targets[1].text == "leave"

    def test_ref_id_parsing(self):
        """Test parsing ref_id attributes."""
        text = "See <fex ref_id='123' name='Example'>text</fex>"
        annotated = AnnotatedText.parse(text)

        assert len(annotated.annotations) == 1
        annotation = annotated.annotations[0]
        assert annotation.ref_id == 123
        assert annotation.name == "Example"


class TestFrameElement:
    """Test cases for FrameElement model."""

    def test_valid_frame_element(self):
        """Test creating a valid frame element."""
        definition = AnnotatedText.parse("The entity that performs an action")

        fe = FrameElement(
            id=123,
            name="Agent",
            abbrev="Agt",
            definition=definition,
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
        )

        assert fe.id == 123
        assert fe.name == "Agent"
        assert fe.abbrev == "Agt"
        assert fe.core_type == "Core"
        assert fe.is_core()

    def test_invalid_fe_name_format(self):
        """Test validation of FE name format."""
        definition = AnnotatedText.parse("Invalid name")

        with pytest.raises(ValueError, match="Invalid frame element name format"):
            FrameElement(
                id=123,
                name="agent@bad",
                abbrev="agt",
                definition=definition,
                core_type="Core",
                bg_color="FF0000",
                fg_color="FFFFFF",
            )

    def test_invalid_color_format(self):
        """Test validation of hex color format."""
        definition = AnnotatedText.parse("Valid definition")

        with pytest.raises(ValueError, match="Invalid hex color format"):
            FrameElement(
                id=123,
                name="Agent",
                abbrev="Agt",
                definition=definition,
                core_type="Core",
                bg_color="GGGGGG",  # Invalid hex
                fg_color="FFFFFF",
            )

    def test_fe_dependencies(self):
        """Test FE dependency constraints."""
        definition = AnnotatedText.parse("A dependent FE")

        fe = FrameElement(
            id=123,
            name="Dependent",
            abbrev="Dep",
            definition=definition,
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
            requires_fe=["Agent", "Theme"],
            excludes_fe=["Goal"],
        )

        assert fe.has_dependencies()
        assert not fe.conflicts_with("Agent")
        assert fe.conflicts_with("Goal")

    def test_self_dependency_validation(self):
        """Test that FE cannot depend on itself."""
        definition = AnnotatedText.parse("Self-dependent FE")

        with pytest.raises(ValueError, match="FE cannot require itself"):
            FrameElement(
                id=123,
                name="Agent",
                abbrev="Agt",
                definition=definition,
                core_type="Core",
                bg_color="FF0000",
                fg_color="FFFFFF",
                requires_fe=["Agent"],
            )

        with pytest.raises(ValueError, match="FE cannot exclude itself"):
            FrameElement(
                id=123,
                name="Agent",
                abbrev="Agt",
                definition=definition,
                core_type="Core",
                bg_color="FF0000",
                fg_color="FFFFFF",
                excludes_fe=["Agent"],
            )

    def test_contradictory_dependencies(self):
        """Test that FE cannot both require and exclude the same FE."""
        definition = AnnotatedText.parse("Contradictory dependencies")

        with pytest.raises(ValueError, match="cannot both require and exclude"):
            FrameElement(
                id=123,
                name="Agent",
                abbrev="Agt",
                definition=definition,
                core_type="Core",
                bg_color="FF0000",
                fg_color="FFFFFF",
                requires_fe=["Theme"],
                excludes_fe=["Theme"],
            )

    def test_peripheral_fe(self):
        """Test peripheral frame element."""
        definition = AnnotatedText.parse("A peripheral element")

        fe = FrameElement(
            id=123,
            name="Manner",
            abbrev="Man",
            definition=definition,
            core_type="Peripheral",
            bg_color="00FF00",
            fg_color="000000",
        )

        assert not fe.is_core()

    def test_username_validation(self):
        """Test username format validation."""
        definition = AnnotatedText.parse("Created by user")

        fe = FrameElement(
            id=123,
            name="Agent",
            abbrev="Agt",
            definition=definition,
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
            created_by="JKR",
        )

        assert fe.created_by == "JKR"

        with pytest.raises(ValueError, match="Invalid username format"):
            FrameElement(
                id=123,
                name="Agent",
                abbrev="Agt",
                definition=definition,
                core_type="Core",
                bg_color="FF0000",
                fg_color="FFFFFF",
                created_by="123invalid",  # Cannot start with number
            )


class TestFrame:
    """Test cases for Frame model."""

    def test_valid_frame(self):
        """Test creating a valid frame."""
        definition = AnnotatedText.parse("An <fex>Agent</fex> leaves behind a <fex>Theme</fex>")

        agent_fe = FrameElement(
            id=1,
            name="Agent",
            abbrev="Agt",
            definition=AnnotatedText.parse("The leaver"),
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
        )

        theme_fe = FrameElement(
            id=2,
            name="Theme",
            abbrev="Thm",
            definition=AnnotatedText.parse("What is left behind"),
            core_type="Core",
            bg_color="00FF00",
            fg_color="000000",
        )

        frame = Frame(
            id=2031,
            name="Abandonment",
            definition=definition,
            frame_elements=[agent_fe, theme_fe],
        )

        assert frame.id == 2031
        assert frame.name == "Abandonment"
        assert len(frame.frame_elements) == 2

    def test_get_fe_by_name(self):
        """Test getting FE by name."""
        definition = AnnotatedText.parse("A frame definition")

        agent_fe = FrameElement(
            id=1,
            name="Agent",
            abbrev="Agt",
            definition=AnnotatedText.parse("The agent"),
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
        )

        frame = Frame(
            id=2031,
            name="Test_Frame",
            definition=definition,
            frame_elements=[agent_fe],
        )

        found_fe = frame.get_fe_by_name("Agent")
        assert found_fe is not None
        assert found_fe.name == "Agent"

        not_found = frame.get_fe_by_name("NonExistent")
        assert not_found is None

    def test_core_vs_peripheral_elements(self):
        """Test separation of core and peripheral elements."""
        definition = AnnotatedText.parse("Frame with mixed FE types")

        core_fe = FrameElement(
            id=1,
            name="Agent",
            abbrev="Agt",
            definition=AnnotatedText.parse("Core element"),
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
        )

        peripheral_fe = FrameElement(
            id=2,
            name="Manner",
            abbrev="Man",
            definition=AnnotatedText.parse("Peripheral element"),
            core_type="Peripheral",
            bg_color="00FF00",
            fg_color="000000",
        )

        frame = Frame(
            id=2031,
            name="Test_Frame",
            definition=definition,
            frame_elements=[core_fe, peripheral_fe],
        )

        core_elements = frame.get_core_elements()
        assert len(core_elements) == 1
        assert core_elements[0].name == "Agent"

        peripheral_elements = frame.get_peripheral_elements()
        assert len(peripheral_elements) == 1
        assert peripheral_elements[0].name == "Manner"

    def test_duplicate_fe_names(self):
        """Test validation against duplicate FE names."""
        definition = AnnotatedText.parse("Frame with duplicate FEs")

        fe1 = FrameElement(
            id=1,
            name="Agent",
            abbrev="Agt1",
            definition=AnnotatedText.parse("First agent"),
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
        )

        fe2 = FrameElement(
            id=2,
            name="Agent",  # Duplicate name
            abbrev="Agt2",
            definition=AnnotatedText.parse("Second agent"),
            core_type="Core",
            bg_color="00FF00",
            fg_color="000000",
        )

        with pytest.raises(ValueError, match="Duplicate frame element names"):
            Frame(
                id=2031,
                name="Test_Frame",
                definition=definition,
                frame_elements=[fe1, fe2],
            )

    def test_unknown_fe_dependencies(self):
        """Test validation of FE dependency references."""
        definition = AnnotatedText.parse("Frame with invalid dependencies")

        fe = FrameElement(
            id=1,
            name="Agent",
            abbrev="Agt",
            definition=AnnotatedText.parse("Agent with bad dependency"),
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
            requires_fe=["NonExistent"],  # FE doesn't exist in frame
        )

        with pytest.raises(ValueError, match="requires unknown FE"):
            Frame(
                id=2031,
                name="Test_Frame",
                definition=definition,
                frame_elements=[fe],
            )

    def test_validate_fe_constraints(self):
        """Test FE constraint validation method."""
        definition = AnnotatedText.parse("Frame for constraint testing")

        agent_fe = FrameElement(
            id=1,
            name="Agent",
            abbrev="Agt",
            definition=AnnotatedText.parse("The agent"),
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
            requires_fe=["Theme"],
        )

        theme_fe = FrameElement(
            id=2,
            name="Theme",
            abbrev="Thm",
            definition=AnnotatedText.parse("The theme"),
            core_type="Core",
            bg_color="00FF00",
            fg_color="000000",
            excludes_fe=["Goal"],
        )

        goal_fe = FrameElement(
            id=3,
            name="Goal",
            abbrev="Gol",
            definition=AnnotatedText.parse("The goal"),
            core_type="Peripheral",
            bg_color="0000FF",
            fg_color="FFFFFF",
        )

        frame = Frame(
            id=2031,
            name="Test_Frame",
            definition=definition,
            frame_elements=[agent_fe, theme_fe, goal_fe],
        )

        # Valid FE set
        result = frame.validate_fe_constraints(["Agent", "Theme"])
        assert len(result["errors"]) == 0

        # Missing required FE
        result = frame.validate_fe_constraints(["Agent"])
        assert len(result["errors"]) == 1
        assert "requires missing FEs" in result["errors"][0]

        # Conflicting FEs
        result = frame.validate_fe_constraints(["Theme", "Goal"])
        assert len(result["errors"]) == 1
        assert "conflicts with present FEs" in result["errors"][0]

        # Unknown FE
        result = frame.validate_fe_constraints(["Unknown"])
        assert len(result["errors"]) == 1
        assert "Unknown frame element" in result["errors"][0]

    def test_invalid_frame_name_format(self):
        """Test validation of frame name format."""
        definition = AnnotatedText.parse("Frame with invalid name")

        with pytest.raises(ValueError, match="Invalid frame name format"):
            Frame(
                id=2031,
                name="abandonment@bad",
                definition=definition,
                frame_elements=[],
            )


class TestFERelation:
    """Test cases for FERelation model."""

    def test_valid_fe_relation(self):
        """Test creating a valid FE relation."""
        fe_rel = FERelation(
            sub_fe_name="Giver",
            super_fe_name="Agent",
            relation_type="Inheritance",
            alignment_confidence=0.95,
            semantic_similarity=0.92,
        )

        assert fe_rel.sub_fe_name == "Giver"
        assert fe_rel.super_fe_name == "Agent"
        assert fe_rel.is_inheritance()
        assert not fe_rel.is_equivalence()

    def test_combined_score_calculation(self):
        """Test combined confidence score calculation."""
        fe_rel = FERelation(
            sub_fe_name="Giver",
            super_fe_name="Agent",
            alignment_confidence=0.9,
            semantic_similarity=0.8,
            syntactic_similarity=0.85,
        )

        expected_score = (0.9 + 0.8 + 0.85) / 3
        assert abs(fe_rel.get_combined_score() - expected_score) < 0.001

    def test_missing_fe_validation(self):
        """Test validation requires at least one FE reference."""
        with pytest.raises(ValueError, match="sub_fe_id or sub_fe_name must be provided"):
            FERelation(
                super_fe_name="Agent",
                relation_type="Inheritance",
            )

        with pytest.raises(ValueError, match="super_fe_id or super_fe_name must be provided"):
            FERelation(
                sub_fe_name="Giver",
                relation_type="Inheritance",
            )


class TestFrameRelation:
    """Test cases for FrameRelation model."""

    def test_valid_frame_relation(self):
        """Test creating a valid frame relation."""
        fe_rel = FERelation(
            sub_fe_name="Giver",
            super_fe_name="Agent",
            relation_type="Inheritance",
        )

        frame_rel = FrameRelation(
            type="Inherits from",
            sub_frame_name="Giving",
            super_frame_name="Transfer",
            fe_relations=[fe_rel],
        )

        assert frame_rel.type == "Inherits from"
        assert frame_rel.is_inheritance()
        assert len(frame_rel.fe_relations) == 1

    def test_get_fe_mapping(self):
        """Test getting FE mapping by name."""
        fe_rel1 = FERelation(sub_fe_name="Giver", super_fe_name="Agent")
        fe_rel2 = FERelation(sub_fe_name="Recipient", super_fe_name="Goal")

        frame_rel = FrameRelation(
            type="Inherits from",
            sub_frame_name="Giving",
            super_frame_name="Transfer",
            fe_relations=[fe_rel1, fe_rel2],
        )

        mapping = frame_rel.get_fe_mapping("Giver")
        assert mapping is not None
        assert mapping.super_fe_name == "Agent"

        no_mapping = frame_rel.get_fe_mapping("NonExistent")
        assert no_mapping is None


class TestSemanticType:
    """Test cases for SemanticType model."""

    def test_root_semantic_type(self):
        """Test root semantic type creation."""
        sem_type = SemanticType(
            id=1,
            name="Sentient",
            abbrev="sent",
            definition="Capable of perception and feeling",
        )

        assert sem_type.is_root_type()
        assert sem_type.get_depth() == 0

    def test_semantic_type_with_parent(self):
        """Test semantic type with parent."""
        sem_type = SemanticType(
            id=2,
            name="Human",
            abbrev="hum",
            definition="A human being",
            super_type_id=1,
            super_type_name="Sentient",
        )

        assert not sem_type.is_root_type()
        assert sem_type.get_depth() == 1

    def test_incomplete_parent_info(self):
        """Test validation of parent type information."""
        with pytest.raises(ValueError, match="super_type_name required"):
            SemanticType(
                id=2,
                name="Human",
                abbrev="hum",
                definition="A human being",
                super_type_id=1,  # Missing super_type_name
            )

        with pytest.raises(ValueError, match="super_type_id required"):
            SemanticType(
                id=2,
                name="Human",
                abbrev="hum",
                definition="A human being",
                super_type_name="Sentient",  # Missing super_type_id
            )


class TestFrameIndexEntry:
    """Test cases for FrameIndexEntry model."""

    def test_valid_frame_index_entry(self):
        """Test creating a valid frame index entry."""
        entry = FrameIndexEntry(
            id=2031,
            name="Abandonment",
            modified_date=datetime.now(UTC),
        )

        assert entry.id == 2031
        assert entry.name == "Abandonment"
        assert isinstance(entry.modified_date, datetime)

    def test_invalid_frame_name_in_index(self):
        """Test validation of frame name in index entry."""
        with pytest.raises(ValueError, match="Invalid frame name format"):
            FrameIndexEntry(
                id=2031,
                name="abandonment@bad",
                modified_date=datetime.now(UTC),
            )


class TestIntegration:
    """Integration tests for model interactions."""

    def test_frame_with_annotated_definition(self):
        """Test frame with properly parsed definition."""
        definition_text = "An <fex>Agent</fex> leaves behind a <fex name='Theme'>possession</fex>"
        definition = AnnotatedText.parse(definition_text)

        agent_fe = FrameElement(
            id=1,
            name="Agent",
            abbrev="Agt",
            definition=AnnotatedText.parse("The entity that abandons"),
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
        )

        theme_fe = FrameElement(
            id=2,
            name="Theme",
            abbrev="Thm",
            definition=AnnotatedText.parse("What is abandoned"),
            core_type="Core",
            bg_color="00FF00",
            fg_color="000000",
        )

        frame = Frame(
            id=2031,
            name="Abandonment",
            definition=definition,
            frame_elements=[agent_fe, theme_fe],
        )

        # Verify definition parsing
        assert "Agent leaves behind a possession" in frame.definition.plain_text
        assert len(frame.definition.annotations) == 2

        # Verify FE references in definition match actual FEs
        fe_refs = frame.definition.get_fe_references()
        fe_names_in_def = {ann.name or ann.text for ann in fe_refs}
        actual_fe_names = {fe.name for fe in frame.frame_elements}

        # Should have some overlap (though not necessarily complete)
        assert len(fe_names_in_def & actual_fe_names) > 0

    def test_json_serialization_roundtrip(self):
        """Test that models can be serialized and deserialized."""
        definition = AnnotatedText.parse("An <fex>Agent</fex> performs an action")

        agent_fe = FrameElement(
            id=1,
            name="Agent",
            abbrev="Agt",
            definition=AnnotatedText.parse("The performer"),
            core_type="Core",
            bg_color="FF0000",
            fg_color="FFFFFF",
            requires_fe=["Theme"],
        )

        theme_fe = FrameElement(
            id=2,
            name="Theme",
            abbrev="Thm",
            definition=AnnotatedText.parse("The affected entity"),
            core_type="Core",
            bg_color="00FF00",
            fg_color="000000",
        )

        original_frame = Frame(
            id=2031,
            name="Action",
            definition=definition,
            frame_elements=[agent_fe, theme_fe],
            created_by="TestUser",
            created_date=datetime.now(UTC),
        )

        # Serialize to JSON Lines
        jsonl = original_frame.to_jsonl()
        assert isinstance(jsonl, str)

        # Deserialize back
        restored_frame = Frame.from_jsonl(jsonl)

        # Verify all data is preserved
        assert restored_frame.id == original_frame.id
        assert restored_frame.name == original_frame.name
        assert len(restored_frame.frame_elements) == 2
        assert restored_frame.frame_elements[0].requires_fe == ["Theme"]

        # Verify the annotated definition is preserved
        assert len(restored_frame.definition.annotations) == 1
        assert restored_frame.definition.annotations[0].type == "fex"


class TestSemTypeRef:
    """Test cases for SemTypeRef model."""

    def test_valid_semtype_ref(self):
        """Test creating a valid semantic type reference."""
        ref = SemTypeRef(name="Sentient", id=123)
        assert ref.name == "Sentient"
        assert ref.id == 123
        assert ref.is_valid_name()

    def test_semtype_with_spaces(self):
        """Test semantic type with spaces (allowed)."""
        ref = SemTypeRef(name="Body of water", id=456)
        assert ref.name == "Body of water"
        assert not ref.is_valid_name()  # Not standard format but allowed

    def test_empty_name_validation(self):
        """Test validation of empty name."""
        with pytest.raises(ValueError):
            SemTypeRef(name="", id=123)


class TestSentenceCount:
    """Test cases for SentenceCount model."""

    def test_valid_sentence_count(self):
        """Test creating valid sentence count."""
        count = SentenceCount(annotated=50, total=100)
        assert count.annotated == 50
        assert count.total == 100
        assert count.get_annotation_rate() == 0.5
        assert count.has_annotations()

    def test_no_annotations(self):
        """Test sentence count with no annotations."""
        count = SentenceCount(annotated=0, total=100)
        assert count.get_annotation_rate() == 0.0
        assert not count.has_annotations()

    def test_no_total_sentences(self):
        """Test annotation rate with no total sentences."""
        count = SentenceCount(annotated=0, total=0)
        assert count.get_annotation_rate() == 0.0

    def test_annotated_exceeds_total(self):
        """Test validation when annotated exceeds total."""
        with pytest.raises(ValueError, match="cannot exceed total"):
            SentenceCount(annotated=150, total=100)

    def test_negative_counts(self):
        """Test validation of negative counts."""
        with pytest.raises(ValueError):
            SentenceCount(annotated=-1, total=100)


class TestLexeme:
    """Test cases for Lexeme model."""

    def test_valid_lexeme(self):
        """Test creating a valid lexeme."""
        lexeme = Lexeme(name="abandon", pos="V", headword=True, order=1)
        assert lexeme.name == "abandon"
        assert lexeme.pos == "V"
        assert lexeme.is_headword()
        assert lexeme.order == 1

    def test_non_headword_lexeme(self):
        """Test non-headword lexeme."""
        lexeme = Lexeme(name="up", pos="PREP", headword=False, order=2)
        assert not lexeme.is_headword()

    def test_invalid_lexeme_name(self):
        """Test validation of lexeme name."""
        # Lexeme names are very permissive in real FrameNet data (spaces, special chars, etc.)
        # Only empty strings should be rejected
        with pytest.raises(ValueError, match="Invalid lexeme name format"):
            Lexeme(name="", pos="V")

    def test_break_before_alias(self):
        """Test breakBefore field alias."""
        lexeme = Lexeme(name="word", pos="N", break_before=True)
        assert lexeme.break_before


class TestLabel:
    """Test cases for Label model."""

    def test_valid_label(self):
        """Test creating a valid label."""
        label = Label(name="Agent", start=0, end=5)
        assert label.name == "Agent"
        assert label.start == 0
        assert label.end == 5
        assert label.get_span_length() == 5
        assert not label.is_null_instantiation()

    def test_null_instantiation_label(self):
        """Test null instantiation label."""
        label = Label(name="Theme", start=0, end=1, is_instantiated_null=True)
        assert label.is_null_instantiation()

    def test_label_overlap(self):
        """Test label overlap detection."""
        label1 = Label(name="Agent", start=0, end=5)
        label2 = Label(name="Theme", start=3, end=8)
        label3 = Label(name="Goal", start=10, end=15)

        assert label1.overlaps_with(label2)
        assert not label1.overlaps_with(label3)

    def test_invalid_label_positions(self):
        """Test validation of invalid positions."""
        with pytest.raises(ValueError, match=r"End position.*must be at or after start position"):
            Label(name="Agent", start=5, end=4)

    def test_label_with_fe_id(self):
        """Test label with frame element ID."""
        label = Label(name="Agent", start=0, end=5, fe_id=123)
        assert label.fe_id == 123


class TestAnnotationLayer:
    """Test cases for AnnotationLayer model."""

    def test_valid_annotation_layer(self):
        """Test creating a valid annotation layer."""
        labels = [Label(name="Agent", start=0, end=5), Label(name="Theme", start=10, end=15)]
        layer = AnnotationLayer(name="FE", rank=1, labels=labels)

        assert layer.name == "FE"
        assert layer.rank == 1
        assert layer.get_label_count() == 2

    def test_get_labels_by_name(self):
        """Test getting labels by name."""
        labels = [
            Label(name="Agent", start=0, end=5),
            Label(name="Agent", start=20, end=25),
            Label(name="Theme", start=10, end=15),
        ]
        layer = AnnotationLayer(name="FE", labels=labels)

        agent_labels = layer.get_labels_by_name("Agent")
        assert len(agent_labels) == 2

    def test_overlapping_labels_detection(self):
        """Test detection of overlapping labels."""
        overlapping_labels = [
            Label(name="Agent", start=0, end=5),
            Label(name="Theme", start=3, end=8),  # Overlaps with Agent
        ]
        layer = AnnotationLayer(name="FE", labels=overlapping_labels)
        assert layer.has_overlapping_labels()

    def test_non_overlapping_labels(self):
        """Test non-overlapping labels."""
        non_overlapping_labels = [
            Label(name="Agent", start=0, end=5),
            Label(name="Theme", start=10, end=15),
        ]
        layer = AnnotationLayer(name="FE", labels=non_overlapping_labels)
        assert not layer.has_overlapping_labels()


class TestAnnotationSet:
    """Test cases for AnnotationSet model."""

    def test_valid_annotation_set(self):
        """Test creating a valid annotation set."""
        layer = AnnotationLayer(name="FE", labels=[])
        anno_set = AnnotationSet(
            id=123, status="MANUAL", sentence_id=456, layers=[layer], created_by="TestUser"
        )

        assert anno_set.id == 123
        assert anno_set.status == "MANUAL"
        assert anno_set.sentence_id == 456
        assert len(anno_set.layers) == 1

    def test_get_layer_by_name(self):
        """Test getting layer by name."""
        fe_layer = AnnotationLayer(name="FE", labels=[])
        gf_layer = AnnotationLayer(name="GF", labels=[])

        anno_set = AnnotationSet(
            id=123, status="MANUAL", sentence_id=456, layers=[fe_layer, gf_layer]
        )

        found_layer = anno_set.get_layer_by_name("FE")
        assert found_layer is not None
        assert found_layer.name == "FE"

        not_found = anno_set.get_layer_by_name("NONEXISTENT")
        assert not_found is None

    def test_get_fe_layer(self):
        """Test getting FE layer specifically."""
        fe_layer = AnnotationLayer(name="FE", labels=[])
        anno_set = AnnotationSet(id=123, status="MANUAL", sentence_id=456, layers=[fe_layer])

        fe = anno_set.get_fe_layer()
        assert fe is not None
        assert fe.name == "FE"

    def test_has_layer(self):
        """Test checking if layer exists."""
        layer = AnnotationLayer(name="Target", labels=[])
        anno_set = AnnotationSet(id=123, status="MANUAL", sentence_id=456, layers=[layer])

        assert anno_set.has_layer("Target")
        assert not anno_set.has_layer("NONEXISTENT")

    def test_invalid_username(self):
        """Test validation of creator username."""
        with pytest.raises(ValueError, match="Invalid username format"):
            AnnotationSet(id=123, status="MANUAL", sentence_id=456, created_by="123invalid")


class TestSentence:
    """Test cases for Sentence model."""

    def test_valid_sentence(self):
        """Test creating a valid sentence."""
        sentence = Sentence(
            id=123,
            text="John abandoned the car.",
            paragraph_no=1,
            sentence_no=2,
            doc_id=456,
            corpus_id=789,
        )

        assert sentence.id == 123
        assert sentence.text == "John abandoned the car."
        assert sentence.paragraph_no == 1
        assert sentence.sentence_no == 2
        assert not sentence.has_annotations()
        assert sentence.get_annotation_count() == 0

    def test_sentence_with_annotations(self):
        """Test sentence with annotation sets."""
        anno_set = AnnotationSet(id=100, status="MANUAL", sentence_id=123, layers=[])

        sentence = Sentence(id=123, text="Test sentence.", annotation_sets=[anno_set])

        assert sentence.has_annotations()
        assert sentence.get_annotation_count() == 1

        found_set = sentence.get_annotation_set_by_id(100)
        assert found_set is not None
        assert found_set.id == 100

        not_found = sentence.get_annotation_set_by_id(999)
        assert not_found is None

    def test_empty_sentence_text(self):
        """Test validation of empty sentence text."""
        with pytest.raises(ValueError):
            Sentence(id=123, text="")


class TestValenceUnit:
    """Test cases for ValenceUnit model."""

    def test_valid_valence_unit(self):
        """Test creating a valid valence unit."""
        unit = ValenceUnit(gf="Ext", pt="NP", fe="Agent")
        assert unit.gf == "Ext"
        assert unit.pt == "NP"
        assert unit.fe == "Agent"
        assert unit.has_grammatical_function()
        assert not unit.is_null_instantiation()

    def test_empty_grammatical_function(self):
        """Test valence unit with empty GF."""
        unit = ValenceUnit(gf="", pt="NP", fe="Agent")
        assert not unit.has_grammatical_function()

    def test_null_instantiation_unit(self):
        """Test null instantiation valence unit."""
        unit = ValenceUnit(gf="", pt="CNI", fe="Agent")
        assert unit.is_null_instantiation()

    def test_special_gf_values(self):
        """Test special GF values."""
        unit = ValenceUnit(gf="CNI", pt="--", fe="Theme")
        assert unit.gf == "CNI"


class TestValenceRealizationPattern:
    """Test cases for ValenceRealizationPattern model."""

    def test_valid_realization_pattern(self):
        """Test creating a valid realization pattern."""
        units = [ValenceUnit(gf="Ext", pt="NP", fe="Agent")]
        pattern = ValenceRealizationPattern(valence_units=units, anno_set_ids=[1, 2, 3], total=3)

        assert len(pattern.valence_units) == 1
        assert pattern.total == 3
        assert "Ext:NP:Agent" in pattern.get_pattern_signature()
        assert not pattern.has_null_instantiation()

    def test_pattern_with_null_instantiation(self):
        """Test pattern with null instantiation."""
        units = [
            ValenceUnit(gf="Ext", pt="NP", fe="Agent"),
            ValenceUnit(gf="", pt="CNI", fe="Theme"),
        ]
        pattern = ValenceRealizationPattern(valence_units=units, anno_set_ids=[], total=1)

        assert pattern.has_null_instantiation()

    def test_pattern_signature(self):
        """Test pattern signature generation."""
        units = [
            ValenceUnit(gf="Ext", pt="NP", fe="Agent"),
            ValenceUnit(gf="Obj", pt="NP", fe="Theme"),
        ]
        pattern = ValenceRealizationPattern(valence_units=units, anno_set_ids=[1], total=1)

        signature = pattern.get_pattern_signature()
        assert signature == "Ext:NP:Agent|Obj:NP:Theme"


class TestFERealization:
    """Test cases for FERealization model."""

    def test_valid_fe_realization(self):
        """Test creating a valid FE realization."""
        pattern = ValenceRealizationPattern(
            valence_units=[ValenceUnit(gf="Ext", pt="NP", fe="Agent")], anno_set_ids=[1, 2], total=2
        )

        fe_real = FERealization(fe_name="Agent", total=10, patterns=[pattern])

        assert fe_real.fe_name == "Agent"
        assert fe_real.total == 10
        assert fe_real.has_patterns()
        assert fe_real.get_pattern_count() == 1

    def test_most_frequent_pattern(self):
        """Test getting most frequent pattern."""
        pattern1 = ValenceRealizationPattern(
            valence_units=[ValenceUnit(gf="Ext", pt="NP", fe="Agent")], anno_set_ids=[1], total=1
        )
        pattern2 = ValenceRealizationPattern(
            valence_units=[ValenceUnit(gf="", pt="CNI", fe="Agent")],
            anno_set_ids=[2, 3, 4],
            total=3,
        )

        fe_real = FERealization(fe_name="Agent", total=4, patterns=[pattern1, pattern2])

        most_frequent = fe_real.get_most_frequent_pattern()
        assert most_frequent is not None
        assert most_frequent.total == 3

    def test_fe_realization_without_patterns(self):
        """Test FE realization without patterns."""
        fe_real = FERealization(fe_name="Agent", total=5, patterns=[])
        assert not fe_real.has_patterns()
        assert fe_real.get_most_frequent_pattern() is None

    def test_invalid_fe_name(self):
        """Test validation of FE name."""
        with pytest.raises(ValueError, match="Invalid FE name format"):
            FERealization(fe_name="agent@bad", total=5)


class TestFEGroupRealization:
    """Test cases for FEGroupRealization model."""

    def test_valid_fe_group(self):
        """Test creating a valid FE group realization."""
        group = FEGroupRealization(
            fe_names=["Agent", "Theme"], grammatical_function="Ext", phrase_type="NP"
        )

        assert group.get_fe_count() == 2
        assert group.contains_fe("Agent")
        assert group.contains_fe("Theme")
        assert not group.contains_fe("Goal")

    def test_single_fe_group(self):
        """Test group with single FE."""
        group = FEGroupRealization(fe_names=["Agent"], grammatical_function="Ext", phrase_type="NP")

        assert group.get_fe_count() == 1
        assert group.contains_fe("Agent")

    def test_invalid_fe_name_in_group(self):
        """Test validation of FE names in group."""
        with pytest.raises(ValueError, match="Invalid FE name in group"):
            FEGroupRealization(
                fe_names=["Agent", "invalid@name"],
                grammatical_function="Ext",
                phrase_type="NP",
            )

    def test_empty_fe_group(self):
        """Test validation of empty FE group."""
        with pytest.raises(ValueError):
            FEGroupRealization(fe_names=[], grammatical_function="Ext", phrase_type="NP")


class TestValenceAnnotationPattern:
    """Test cases for ValenceAnnotationPattern model."""

    def test_valid_valence_annotation_pattern(self):
        """Test creating a valid valence annotation pattern."""
        group = FEGroupRealization(fe_names=["Agent"], grammatical_function="Ext", phrase_type="NP")

        pattern = ValenceAnnotationPattern(anno_sets=[100, 101, 102], pattern=[group])

        assert pattern.get_annotation_count() == 3
        assert len(pattern.get_fe_groups()) == 1

    def test_empty_annotation_pattern(self):
        """Test pattern with no annotation sets."""
        group = FEGroupRealization(fe_names=["Agent"], grammatical_function="Ext", phrase_type="NP")

        pattern = ValenceAnnotationPattern(anno_sets=[], pattern=[group])

        assert pattern.get_annotation_count() == 0


class TestValencePattern:
    """Test cases for ValencePattern model."""

    def test_valid_valence_pattern(self):
        """Test creating a valid valence pattern."""
        fe_real = FERealization(fe_name="Agent", total=5, patterns=[])
        anno_pattern = ValenceAnnotationPattern(anno_sets=[1, 2], pattern=[])

        valence = ValencePattern(
            total_annotated=100, fe_realizations=[fe_real], patterns=[anno_pattern]
        )

        assert valence.total_annotated == 100
        assert valence.has_fe_realizations()
        assert len(valence.patterns) == 1

    def test_get_fe_realization(self):
        """Test getting FE realization by name."""
        agent_real = FERealization(fe_name="Agent", total=10, patterns=[])
        theme_real = FERealization(fe_name="Theme", total=8, patterns=[])

        valence = ValencePattern(total_annotated=50, fe_realizations=[agent_real, theme_real])

        found = valence.get_fe_realization("Agent")
        assert found is not None
        assert found.fe_name == "Agent"

        not_found = valence.get_fe_realization("NonExistent")
        assert not_found is None

    def test_most_frequent_fe(self):
        """Test getting most frequent FE."""
        agent_real = FERealization(fe_name="Agent", total=10, patterns=[])
        theme_real = FERealization(fe_name="Theme", total=15, patterns=[])

        valence = ValencePattern(total_annotated=25, fe_realizations=[agent_real, theme_real])

        most_frequent = valence.get_most_frequent_fe()
        assert most_frequent is not None
        assert most_frequent.fe_name == "Theme"
        assert most_frequent.total == 15

    def test_valence_without_realizations(self):
        """Test valence pattern without FE realizations."""
        valence = ValencePattern(total_annotated=0, fe_realizations=[])

        assert not valence.has_fe_realizations()
        assert valence.get_most_frequent_fe() is None


class TestLexicalUnit:
    """Test cases for LexicalUnit model."""

    def test_valid_lexical_unit(self):
        """Test creating a valid lexical unit."""
        sentence_count = SentenceCount(annotated=50, total=100)
        lexeme = Lexeme(name="abandon", pos="V", headword=True)

        lu = LexicalUnit(
            id=1234,
            name="abandon.v",
            pos="V",
            definition="To leave behind permanently",
            frame_id=2031,
            frame_name="Abandonment",
            sentence_count=sentence_count,
            lexemes=[lexeme],
        )

        assert lu.id == 1234
        assert lu.name == "abandon.v"
        assert lu.pos == "V"
        assert lu.frame_id == 2031
        assert lu.get_annotation_rate() == 0.5
        assert not lu.is_multi_word()

    def test_multi_word_lexical_unit(self):
        """Test multi-word lexical unit."""
        sentence_count = SentenceCount(annotated=10, total=20)
        lexeme1 = Lexeme(name="give", pos="V", headword=True, order=1)
        lexeme2 = Lexeme(name="up", pos="PREP", headword=False, order=2)

        lu = LexicalUnit(
            id=5678,
            name="give_up.v",
            pos="V",
            definition="To surrender or quit",
            frame_id=2031,
            frame_name="Abandonment",
            sentence_count=sentence_count,
            lexemes=[lexeme1, lexeme2],
        )

        assert lu.is_multi_word()
        assert len(lu.lexemes) == 2

        headword = lu.get_headword_lexeme()
        assert headword is not None
        assert headword.name == "give"

    def test_lu_with_valence_patterns(self):
        """Test LU with valence patterns."""
        sentence_count = SentenceCount(annotated=30, total=50)
        lexeme = Lexeme(name="abandon", pos="V", headword=True)

        fe_real = FERealization(fe_name="Agent", total=25, patterns=[])
        valence = ValencePattern(total_annotated=30, fe_realizations=[fe_real])

        lu = LexicalUnit(
            id=1234,
            name="abandon.v",
            pos="V",
            definition="To leave behind permanently",
            frame_id=2031,
            frame_name="Abandonment",
            sentence_count=sentence_count,
            lexemes=[lexeme],
            valence_patterns=[valence],
        )

        assert lu.has_valence_patterns()

        most_frequent = lu.get_most_frequent_valence()
        assert most_frequent is not None
        assert most_frequent.total_annotated == 30

    def test_lu_with_semantic_types(self):
        """Test LU with semantic type references."""
        sentence_count = SentenceCount(annotated=10, total=20)
        lexeme = Lexeme(name="abandon", pos="V", headword=True)
        semtype = SemTypeRef(name="Sentient", id=123)

        lu = LexicalUnit(
            id=1234,
            name="abandon.v",
            pos="V",
            definition="To leave behind permanently",
            frame_id=2031,
            frame_name="Abandonment",
            sentence_count=sentence_count,
            lexemes=[lexeme],
            semtypes=[semtype],
        )

        assert len(lu.semtypes) == 1
        assert lu.semtypes[0].name == "Sentient"

    def test_lu_with_annotation_sets(self):
        """Test LU with annotation sets."""
        sentence_count = SentenceCount(annotated=5, total=10)
        lexeme = Lexeme(name="abandon", pos="V", headword=True)

        anno_set = AnnotationSet(id=100, status="MANUAL", sentence_id=200, layers=[])

        lu = LexicalUnit(
            id=1234,
            name="abandon.v",
            pos="V",
            definition="To leave behind permanently",
            frame_id=2031,
            frame_name="Abandonment",
            sentence_count=sentence_count,
            lexemes=[lexeme],
            annotation_sets=[anno_set],
        )

        found_set = lu.get_annotation_set_by_id(100)
        assert found_set is not None
        assert found_set.id == 100

        not_found = lu.get_annotation_set_by_id(999)
        assert not_found is None

    def test_lu_validation_no_lexemes(self):
        """Test validation when LU has no lexemes."""
        sentence_count = SentenceCount(annotated=0, total=0)

        with pytest.raises(ValueError, match="must have at least one lexeme"):
            LexicalUnit(
                id=1234,
                name="abandon.v",
                pos="V",
                definition="Test definition",
                frame_id=2031,
                frame_name="Abandonment",
                sentence_count=sentence_count,
                lexemes=[],  # No lexemes
            )

    def test_lu_validation_no_headword(self):
        """Test validation when no lexeme is marked as headword."""
        sentence_count = SentenceCount(annotated=0, total=0)
        lexeme = Lexeme(name="abandon", pos="V", headword=False)  # Not headword

        with pytest.raises(ValueError, match="exactly one headword lexeme"):
            LexicalUnit(
                id=1234,
                name="abandon.v",
                pos="V",
                definition="Test definition",
                frame_id=2031,
                frame_name="Abandonment",
                sentence_count=sentence_count,
                lexemes=[lexeme],
            )

    def test_lu_validation_multiple_headwords(self):
        """Test validation when multiple lexemes are marked as headword."""
        sentence_count = SentenceCount(annotated=0, total=0)
        lexeme1 = Lexeme(name="give", pos="V", headword=True)  # Headword
        lexeme2 = Lexeme(name="up", pos="PREP", headword=True)  # Also headword - invalid

        with pytest.raises(ValueError, match="exactly one headword lexeme"):
            LexicalUnit(
                id=1234,
                name="give_up.v",
                pos="V",
                definition="Test definition",
                frame_id=2031,
                frame_name="Abandonment",
                sentence_count=sentence_count,
                lexemes=[lexeme1, lexeme2],
            )

    def test_lu_validation_total_annotated_consistency(self):
        """Test validation of total_annotated vs sentence count."""
        sentence_count = SentenceCount(annotated=50, total=100)
        lexeme = Lexeme(name="abandon", pos="V", headword=True)

        with pytest.raises(ValueError, match="cannot exceed sentence count total"):
            LexicalUnit(
                id=1234,
                name="abandon.v",
                pos="V",
                definition="Test definition",
                frame_id=2031,
                frame_name="Abandonment",
                sentence_count=sentence_count,
                lexemes=[lexeme],
                total_annotated=150,  # Exceeds sentence count total
            )

    def test_invalid_lu_name_format(self):
        """Test validation of LU name format."""
        sentence_count = SentenceCount(annotated=0, total=0)
        lexeme = Lexeme(name="abandon", pos="V", headword=True)

        with pytest.raises(ValueError, match="Invalid lexical unit name format"):
            LexicalUnit(
                id=1234,
                name="abandon",  # Missing .pos suffix
                pos="V",
                definition="Test definition",
                frame_id=2031,
                frame_name="Abandonment",
                sentence_count=sentence_count,
                lexemes=[lexeme],
            )

    def test_invalid_frame_name_format(self):
        """Test validation of frame name format."""
        sentence_count = SentenceCount(annotated=0, total=0)
        lexeme = Lexeme(name="abandon", pos="V", headword=True)

        with pytest.raises(ValueError, match="Invalid frame name format"):
            LexicalUnit(
                id=1234,
                name="abandon.v",
                pos="V",
                definition="Test definition",
                frame_id=2031,
                frame_name="abandonment@bad",
                sentence_count=sentence_count,
                lexemes=[lexeme],
            )
