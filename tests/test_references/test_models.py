"""Tests for cross-reference models.

Tests all models in glazing.references.models including confidence scoring,
transitive mapping, and conflict resolution.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from glazing.references.models import (
    CrossReference,
    FEAlignment,
    FEInheritanceChain,
    FERelation,
    FrameNetLURef,
    LexLink,
    MappingConfidence,
    MappingConflict,
    MappingIndex,
    MappingMetadata,
    MultiMapping,
    PropBankCrossRefs,
    PropBankRolesetRef,
    RoleLink,
    RoleMappingTable,
    TransitiveMapping,
    UnifiedLemma,
    UnifiedRoleMapping,
    VerbNetCrossRefs,
    VerbNetFrameNetMapping,
    VerbNetMemberRef,
)


class TestMappingConfidence:
    """Test confidence scoring model."""

    def test_valid_confidence(self):
        """Test valid confidence score."""
        conf = MappingConfidence(
            score=0.85, method="manual_annotation", factors={"lexical": 0.9, "syntactic": 0.8}
        )
        assert conf.score == 0.85
        assert conf.method == "manual_annotation"
        assert len(conf.factors) == 2

    def test_invalid_confidence_too_high(self):
        """Test confidence score above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            MappingConfidence(score=1.5, method="test")
        assert "between 0 and 1" in str(exc_info.value)

    def test_invalid_confidence_negative(self):
        """Test negative confidence score."""
        with pytest.raises(ValidationError) as exc_info:
            MappingConfidence(score=-0.1, method="test")
        assert "between 0 and 1" in str(exc_info.value)

    def test_boundary_confidence_scores(self):
        """Test boundary values 0.0 and 1.0."""
        conf_zero = MappingConfidence(score=0.0, method="test")
        assert conf_zero.score == 0.0

        conf_one = MappingConfidence(score=1.0, method="test")
        assert conf_one.score == 1.0


class TestMappingMetadata:
    """Test mapping metadata model."""

    def test_full_metadata(self):
        """Test metadata with all fields."""
        now = datetime.now(UTC)
        meta = MappingMetadata(
            created_date=now,
            created_by="manual",
            modified_date=now,
            modified_by="reviewer",
            version="1.7",
            validation_status="validated",
            validation_method="expert_review",
            notes="High quality mapping",
        )
        assert meta.created_date == now
        assert meta.validation_status == "validated"

    def test_minimal_metadata(self):
        """Test metadata with required fields only."""
        meta = MappingMetadata(
            created_date=datetime.now(UTC),
            created_by="auto",
            version="1.0",
            validation_status="unvalidated",
        )
        assert meta.modified_date is None
        assert meta.validation_method is None


class TestCrossReference:
    """Test cross-reference base model."""

    def test_single_target_mapping(self):
        """Test mapping to single target."""
        ref = CrossReference(
            source_dataset="verbnet",
            source_id="give-13.1",
            source_version="3.4",
            target_dataset="framenet",
            target_id="Giving",
            mapping_type="direct",
            confidence=MappingConfidence(score=0.95, method="manual"),
            metadata=MappingMetadata(
                created_date=datetime.now(UTC),
                created_by="expert",
                version="1.7",
                validation_status="validated",
            ),
        )
        assert ref.source_dataset == "verbnet"
        assert ref.target_id == "Giving"
        assert ref.confidence.score == 0.95

    def test_multiple_target_mapping(self):
        """Test mapping to multiple targets."""
        ref = CrossReference(
            source_dataset="propbank",
            source_id="give.01",
            source_version="3.1",
            target_dataset="verbnet",
            target_id=["give-13.1", "give-13.1-1"],
            mapping_type="automatic",
            confidence=None,
            metadata=MappingMetadata(
                created_date=datetime.now(UTC),
                created_by="system",
                version="3.4",
                validation_status="unvalidated",
            ),
        )
        assert isinstance(ref.target_id, list)
        assert len(ref.target_id) == 2

    def test_inherited_mapping(self):
        """Test inherited mapping from parent class."""
        ref = CrossReference(
            source_dataset="verbnet",
            source_id="give-13.1-1",
            source_version="3.4",
            target_dataset="framenet",
            target_id="Giving",
            mapping_type="inferred",
            confidence=MappingConfidence(score=0.85, method="inheritance"),
            metadata=MappingMetadata(
                created_date=datetime.now(UTC),
                created_by="system",
                version="1.7",
                validation_status="validated",
            ),
            inherited_from="give-13.1",
        )
        assert ref.inherited_from == "give-13.1"
        assert ref.mapping_type == "inferred"


class TestMultiMapping:
    """Test one-to-many mapping model."""

    def test_get_best_mapping(self):
        """Test finding best mapping by confidence."""
        mappings = [
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.95, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="expert",
                    version="1.7",
                    validation_status="validated",
                ),
            ),
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Transfer",
                mapping_type="automatic",
                confidence=MappingConfidence(score=0.7, method="auto"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="system",
                    version="1.7",
                    validation_status="unvalidated",
                ),
            ),
        ]

        multi = MultiMapping(
            source_dataset="verbnet", source_id="give-13.1", source_version="3.4", mappings=mappings
        )

        best = multi.get_best_mapping("framenet")
        assert best is not None
        assert best.target_id == "Giving"
        assert best.confidence.score == 0.95

    def test_no_matching_target(self):
        """Test when no mapping to target dataset exists."""
        mappings = [
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=None,
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="expert",
                    version="1.7",
                    validation_status="validated",
                ),
            )
        ]

        multi = MultiMapping(
            source_dataset="verbnet", source_id="give-13.1", source_version="3.4", mappings=mappings
        )

        best = multi.get_best_mapping("propbank")
        assert best is None


class TestTransitiveMapping:
    """Test transitive mapping through intermediate resources."""

    def test_calculate_confidence(self):
        """Test confidence propagation through chain."""
        path = [
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="wordnet",
                target_id="give%2:40:00",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.9, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="expert",
                    version="3.1",
                    validation_status="validated",
                ),
            ),
            CrossReference(
                source_dataset="wordnet",
                source_id="give%2:40:00",
                source_version="3.1",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="automatic",
                confidence=MappingConfidence(score=0.8, method="similarity"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="system",
                    version="1.7",
                    validation_status="unvalidated",
                ),
            ),
        ]

        trans = TransitiveMapping(
            source_dataset="verbnet",
            source_id="give-13.1",
            target_dataset="framenet",
            target_id="Giving",
            path=path,
            combined_confidence=0.72,  # 0.9 * 0.8
        )

        calculated = trans.calculate_confidence()
        assert calculated == pytest.approx(0.72, rel=1e-3)

    def test_empty_path_confidence(self):
        """Test confidence calculation with empty path."""
        trans = TransitiveMapping(
            source_dataset="verbnet",
            source_id="give-13.1",
            target_dataset="framenet",
            target_id="Giving",
            path=[],
            combined_confidence=0.0,
        )

        assert trans.calculate_confidence() == 0.0


class TestVerbNetCrossRefs:
    """Test VerbNet cross-reference model."""

    def test_primary_framenet_mapping(self):
        """Test getting highest confidence FrameNet mapping."""
        mappings = [
            VerbNetFrameNetMapping(
                frame_name="Giving",
                confidence=MappingConfidence(score=0.95, method="manual"),
                mapping_source="gold",
            ),
            VerbNetFrameNetMapping(
                frame_name="Transfer",
                confidence=MappingConfidence(score=0.7, method="auto"),
                mapping_source="automatic",
            ),
        ]

        refs = VerbNetCrossRefs(
            verbnet_key="give#2", class_id="give-13.1", lemma="give", fn_mappings=mappings
        )

        primary = refs.get_primary_framenet_mapping()
        assert primary is not None
        assert primary.frame_name == "Giving"
        assert primary.confidence.score == 0.95

    def test_conflicting_mappings_detection(self):
        """Test detection of conflicting high-confidence mappings."""
        mappings = [
            VerbNetFrameNetMapping(
                frame_name="Giving",
                confidence=MappingConfidence(score=0.85, method="manual"),
                mapping_source="gold",
            ),
            VerbNetFrameNetMapping(
                frame_name="Transfer",
                confidence=MappingConfidence(score=0.80, method="manual"),
                mapping_source="gold",
            ),
        ]

        refs = VerbNetCrossRefs(
            verbnet_key="give#2", class_id="give-13.1", lemma="give", fn_mappings=mappings
        )

        assert refs.has_conflicting_mappings() is True

    def test_no_conflicts_single_mapping(self):
        """Test no conflicts with single mapping."""
        refs = VerbNetCrossRefs(
            verbnet_key="give#2",
            class_id="give-13.1",
            lemma="give",
            fn_mappings=[
                VerbNetFrameNetMapping(
                    frame_name="Giving",
                    confidence=MappingConfidence(score=0.95, method="manual"),
                    mapping_source="gold",
                )
            ],
        )

        assert refs.has_conflicting_mappings() is False


class TestPropBankCrossRefs:
    """Test PropBank cross-reference model."""

    def test_get_verbnet_classes(self):
        """Test extracting VerbNet classes from rolelinks and lexlinks."""
        refs = PropBankCrossRefs(
            roleset_id="give.01",
            rolelinks=[
                RoleLink(class_name="give-13.1", resource="verbnet", version="3.4", role="Agent")
            ],
            lexlinks=[
                LexLink(
                    class_name="give-13.1-1",
                    confidence=0.85,
                    resource="verbnet",
                    version="3.4",
                    src="automatic",
                )
            ],
        )

        vn_classes = refs.get_verbnet_classes()
        assert len(vn_classes) == 2
        assert ("give-13.1", None) in vn_classes
        assert ("give-13.1-1", 0.85) in vn_classes


class TestUnifiedRoleMapping:
    """Test unified role mapping across datasets."""

    def test_alignment_score_calculation(self):
        """Test calculating overall alignment score."""
        mapping = UnifiedRoleMapping(
            concept="agent",
            verbnet_roles=[("give-13.1", "Agent")],
            framenet_fes=[("Giving", "Donor")],
            propbank_args=[("give.01", "ARG0")],
            wordnet_restrictions=["animate", "volitional"],
            confidence_matrix={
                "verbnet:give-13.1:Agent": {
                    "framenet:Giving:Donor": 0.95,
                    "propbank:give.01:ARG0": 0.98,
                },
                "framenet:Giving:Donor": {"propbank:give.01:ARG0": 0.92},
            },
        )

        score = mapping.get_alignment_score()
        expected = (0.95 + 0.98 + 0.92) / 3
        assert score == pytest.approx(expected, rel=1e-3)

    def test_empty_confidence_matrix(self):
        """Test alignment score with empty confidence matrix."""
        mapping = UnifiedRoleMapping(
            concept="agent",
            verbnet_roles=[],
            framenet_fes=[],
            propbank_args=[],
            wordnet_restrictions=[],
        )

        assert mapping.get_alignment_score() == 0.0


class TestUnifiedLemma:
    """Test unified lemma model."""

    def test_valid_lemma(self):
        """Test valid lemma format."""
        lemma = UnifiedLemma(
            lemma="give",
            pos="v",
            framenet_lus=[
                FrameNetLURef(lu_id=1, frame_name="Giving", definition="transfer possession")
            ],
            propbank_rolesets=[PropBankRolesetRef(roleset_id="give.01", name="transfer")],
            verbnet_members=[VerbNetMemberRef(verbnet_key="give#2", class_id="give-13.1")],
            wordnet_senses=[],
        )
        assert lemma.lemma == "give"
        assert lemma.pos == "v"

    def test_invalid_lemma_format(self):
        """Test invalid lemma format."""
        with pytest.raises(ValidationError) as exc_info:
            UnifiedLemma(
                lemma=" give",  # Leading space invalid
                pos="v",
                framenet_lus=[],
                propbank_rolesets=[],
                verbnet_members=[],
                wordnet_senses=[],
            )
        assert "Invalid lemma format" in str(exc_info.value)


class TestRoleMappingTable:
    """Test role mapping table model."""

    def test_is_agentive(self):
        """Test agentive role detection."""
        # Test with verbnet Agent
        mapping1 = RoleMappingTable(verbnet_role="Agent", framenet_fe="Donor", propbank_arg="ARG0")
        assert mapping1.is_agentive() is True

        # Test with propbank ARG0
        mapping2 = RoleMappingTable(verbnet_role="Theme", propbank_arg="ARG0")
        assert mapping2.is_agentive() is True

        # Test with framenet Agent-containing FE
        mapping3 = RoleMappingTable(verbnet_role="Theme", framenet_fe="Agent_of_change")
        assert mapping3.is_agentive() is True

        # Test non-agentive
        mapping4 = RoleMappingTable(verbnet_role="Theme", framenet_fe="Gift", propbank_arg="ARG1")
        assert mapping4.is_agentive() is False


class TestFEAlignment:
    """Test FE alignment model."""

    def test_combined_score_calculation(self):
        """Test combined score with different alignment types."""
        base_confidence = MappingConfidence(score=0.9, method="manual", factors={})

        # Direct alignment
        direct = FEAlignment(
            source_frame="Giving",
            source_fe="Donor",
            target_dataset="verbnet",
            target_role="Agent",
            alignment_type="direct",
            confidence=base_confidence,
        )
        assert direct.get_combined_score() == 0.9

        # Inherited alignment
        inherited = FEAlignment(
            source_frame="Giving",
            source_fe="Donor",
            target_dataset="verbnet",
            target_role="Agent",
            alignment_type="inherited",
            confidence=base_confidence,
        )
        assert inherited.get_combined_score() == pytest.approx(0.81, rel=1e-3)

        # Inferred alignment
        inferred = FEAlignment(
            source_frame="Giving",
            source_fe="Donor",
            target_dataset="verbnet",
            target_role="Agent",
            alignment_type="inferred",
            confidence=base_confidence,
        )
        assert inferred.get_combined_score() == pytest.approx(0.72, rel=1e-3)

        # Partial alignment
        partial = FEAlignment(
            source_frame="Giving",
            source_fe="Donor",
            target_dataset="verbnet",
            target_role="Agent",
            alignment_type="partial",
            confidence=base_confidence,
        )
        assert partial.get_combined_score() == pytest.approx(0.63, rel=1e-3)


class TestFERelation:
    """Test FE relation model."""

    def test_fe_name_validation(self):
        """Test FE name format validation."""
        # Valid FE names using aliases
        rel = FERelation(subFEName="Agent", supFEName="Donor", relation_type="Inheritance")
        assert rel.sub_fe_name == "Agent"
        assert rel.super_fe_name == "Donor"

        # Invalid FE name
        with pytest.raises(ValidationError) as exc_info:
            FERelation(subFEName="agent")  # Lowercase invalid
        assert "Invalid FE name format" in str(exc_info.value)

    def test_relation_type_checks(self):
        """Test relation type checking methods."""
        inheritance_rel = FERelation(
            subFEName="Agent", supFEName="Donor", relation_type="Inheritance"
        )
        assert inheritance_rel.is_inheritance() is True
        assert inheritance_rel.is_equivalence() is False

        equivalence_rel = FERelation(
            subFEName="Theme", supFEName="Theme", relation_type="Equivalence"
        )
        assert equivalence_rel.is_inheritance() is False
        assert equivalence_rel.is_equivalence() is True


class TestFEInheritanceChain:
    """Test FE inheritance chain model."""

    def test_inheritance_depth(self):
        """Test calculating inheritance depth."""
        chain = FEInheritanceChain(
            fe_name="Donor",
            frame_chain=["Commerce_goods-transfer", "Transfer", "Giving"],
            inheritance_path=[
                FERelation(subFEName="Seller", supFEName="Transferor", relation_type="Inheritance"),
                FERelation(subFEName="Transferor", supFEName="Donor", relation_type="Inheritance"),
            ],
        )

        assert chain.get_inheritance_depth() == 2


class TestMappingConflict:
    """Test mapping conflict resolution."""

    def test_resolve_by_confidence(self):
        """Test resolving conflict by highest confidence."""
        mappings = [
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.95, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="expert",
                    version="1.7",
                    validation_status="validated",
                ),
            ),
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Transfer",
                mapping_type="automatic",
                confidence=MappingConfidence(score=0.7, method="auto"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="system",
                    version="1.7",
                    validation_status="unvalidated",
                ),
            ),
        ]

        conflict = MappingConflict(
            conflict_type="ambiguous",
            source_dataset="verbnet",
            source_id="give-13.1",
            conflicting_mappings=mappings,
        )

        resolved = conflict.resolve_by_confidence()
        assert resolved is not None
        assert resolved.target_id == "Giving"

    def test_resolve_by_source(self):
        """Test resolving conflict by preferred source."""
        mappings = [
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=None,
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="automatic",
                    version="1.7",
                    validation_status="unvalidated",
                ),
            ),
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Transfer",
                mapping_type="manual",
                confidence=None,
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="1.7",
                    validation_status="validated",
                ),
            ),
        ]

        conflict = MappingConflict(
            conflict_type="ambiguous",
            source_dataset="verbnet",
            source_id="give-13.1",
            conflicting_mappings=mappings,
        )

        resolved = conflict.resolve_by_source("manual")
        assert resolved is not None
        assert resolved.target_id == "Transfer"


class TestMappingIndex:
    """Test bidirectional mapping index."""

    def test_add_mapping(self):
        """Test adding mapping to bidirectional index."""
        index = MappingIndex()

        mapping = CrossReference(
            source_dataset="verbnet",
            source_id="give-13.1",
            source_version="3.4",
            target_dataset="framenet",
            target_id="Giving",
            mapping_type="direct",
            confidence=None,
            metadata=MappingMetadata(
                created_date=datetime.now(UTC),
                created_by="manual",
                version="1.7",
                validation_status="validated",
            ),
        )

        index.add_mapping(mapping)

        # Check forward index
        forward_key = "verbnet:give-13.1"
        assert forward_key in index.forward_index
        assert len(index.forward_index[forward_key]) == 1
        assert index.forward_index[forward_key][0] == mapping

        # Check reverse index
        reverse_key = "framenet:Giving"
        assert reverse_key in index.reverse_index
        assert len(index.reverse_index[reverse_key]) == 1
        assert index.reverse_index[reverse_key][0] == mapping

    def test_add_mapping_with_multiple_targets(self):
        """Test adding mapping with multiple targets."""
        index = MappingIndex()

        mapping = CrossReference(
            source_dataset="propbank",
            source_id="give.01",
            source_version="3.1",
            target_dataset="verbnet",
            target_id=["give-13.1", "give-13.1-1"],
            mapping_type="direct",
            confidence=None,
            metadata=MappingMetadata(
                created_date=datetime.now(UTC),
                created_by="manual",
                version="3.4",
                validation_status="validated",
            ),
        )

        index.add_mapping(mapping)

        # Check both targets in reverse index
        assert "verbnet:give-13.1" in index.reverse_index
        assert "verbnet:give-13.1-1" in index.reverse_index
        assert len(index.reverse_index["verbnet:give-13.1"]) == 1
        assert len(index.reverse_index["verbnet:give-13.1-1"]) == 1

    def test_find_transitive_mappings_simple_path(self):
        """Test finding basic A→B→C transitive mapping."""
        index = MappingIndex()

        # Create A → B mapping
        mapping1 = CrossReference(
            source_dataset="verbnet",
            source_id="give-13.1",
            source_version="3.4",
            target_dataset="wordnet",
            target_id="give%2:40:00",
            mapping_type="direct",
            confidence=MappingConfidence(score=0.9, method="manual"),
            metadata=MappingMetadata(
                created_date=datetime.now(UTC),
                created_by="manual",
                version="3.1",
                validation_status="validated",
            ),
        )

        # Create B → C mapping
        mapping2 = CrossReference(
            source_dataset="wordnet",
            source_id="give%2:40:00",
            source_version="3.1",
            target_dataset="framenet",
            target_id="Giving",
            mapping_type="direct",
            confidence=MappingConfidence(score=0.8, method="automatic"),
            metadata=MappingMetadata(
                created_date=datetime.now(UTC),
                created_by="system",
                version="1.7",
                validation_status="validated",
            ),
        )

        index.add_mapping(mapping1)
        index.add_mapping(mapping2)

        # Find transitive path from VerbNet to FrameNet
        results = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=2)

        assert len(results) == 1
        assert results[0].source_dataset == "verbnet"
        assert results[0].source_id == "give-13.1"
        assert results[0].target_dataset == "framenet"
        assert results[0].target_id == "Giving"
        assert len(results[0].path) == 2
        assert results[0].combined_confidence == pytest.approx(0.72, rel=1e-3)  # 0.9 * 0.8

    def test_find_transitive_mappings_multiple_paths(self):
        """Test finding multiple paths to same target."""
        index = MappingIndex()

        # Path 1: verbnet → wordnet → framenet (high confidence)
        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="wordnet",
                target_id="give%2:40:00",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.95, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.1",
                    validation_status="validated",
                ),
            )
        )
        index.add_mapping(
            CrossReference(
                source_dataset="wordnet",
                source_id="give%2:40:00",
                source_version="3.1",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.9, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="1.7",
                    validation_status="validated",
                ),
            )
        )

        # Path 2: verbnet → propbank → framenet (lower confidence)
        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="propbank",
                target_id="give.01",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.8, method="automatic"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="system",
                    version="3.1",
                    validation_status="unvalidated",
                ),
            )
        )
        index.add_mapping(
            CrossReference(
                source_dataset="propbank",
                source_id="give.01",
                source_version="3.1",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.7, method="automatic"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="system",
                    version="1.7",
                    validation_status="unvalidated",
                ),
            )
        )

        results = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=2)

        assert len(results) == 2
        # Should be sorted by confidence (high to low)
        assert results[0].combined_confidence == pytest.approx(0.855, rel=1e-3)  # 0.95 * 0.9
        assert results[1].combined_confidence == pytest.approx(0.56, rel=1e-3)  # 0.8 * 0.7

    def test_find_transitive_mappings_max_hops(self):
        """Test respecting max_hops limit."""
        index = MappingIndex()

        # Create a long chain: A → B → C → D
        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="wordnet",
                target_id="give%2:40:00",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.9, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.1",
                    validation_status="validated",
                ),
            )
        )
        index.add_mapping(
            CrossReference(
                source_dataset="wordnet",
                source_id="give%2:40:00",
                source_version="3.1",
                target_dataset="propbank",
                target_id="give.01",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.8, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.1",
                    validation_status="validated",
                ),
            )
        )
        index.add_mapping(
            CrossReference(
                source_dataset="propbank",
                source_id="give.01",
                source_version="3.1",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.85, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="1.7",
                    validation_status="validated",
                ),
            )
        )

        # With max_hops=2, should not find the 3-hop path
        results = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=2)
        assert len(results) == 0

        # With max_hops=3, should find the 3-hop path
        results = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=3)
        assert len(results) == 1
        assert len(results[0].path) == 3

    def test_find_transitive_mappings_no_path(self):
        """Test when no path exists."""
        index = MappingIndex()

        # Add isolated mappings
        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="wordnet",
                target_id="give%2:40:00",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.9, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.1",
                    validation_status="validated",
                ),
            )
        )

        # No path from verbnet to framenet
        results = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=3)
        assert len(results) == 0

    def test_find_transitive_mappings_cycle_prevention(self):
        """Test that cycles are properly handled."""
        index = MappingIndex()

        # Create a cycle: A → B → C → A
        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="wordnet",
                target_id="give%2:40:00",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.9, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.1",
                    validation_status="validated",
                ),
            )
        )
        index.add_mapping(
            CrossReference(
                source_dataset="wordnet",
                source_id="give%2:40:00",
                source_version="3.1",
                target_dataset="propbank",
                target_id="give.01",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.8, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.1",
                    validation_status="validated",
                ),
            )
        )
        index.add_mapping(
            CrossReference(
                source_dataset="propbank",
                source_id="give.01",
                source_version="3.1",
                target_dataset="verbnet",
                target_id="give-13.1",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.85, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.4",
                    validation_status="validated",
                ),
            )
        )

        # Also add target
        index.add_mapping(
            CrossReference(
                source_dataset="propbank",
                source_id="give.01",
                source_version="3.1",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.75, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="1.7",
                    validation_status="validated",
                ),
            )
        )

        # Should find path without infinite loop
        results = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=5)
        assert len(results) > 0
        # Should find the shortest path
        assert results[0].target_id == "Giving"

    def test_find_transitive_mappings_confidence_calculation(self):
        """Test confidence propagation with missing scores."""
        index = MappingIndex()

        # Mapping with confidence
        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="wordnet",
                target_id="give%2:40:00",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.9, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.1",
                    validation_status="validated",
                ),
            )
        )

        # Mapping without confidence (should use 0.5 default)
        index.add_mapping(
            CrossReference(
                source_dataset="wordnet",
                source_id="give%2:40:00",
                source_version="3.1",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=None,
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="1.7",
                    validation_status="validated",
                ),
            )
        )

        results = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=2)

        assert len(results) == 1
        assert results[0].combined_confidence == pytest.approx(0.45, rel=1e-3)  # 0.9 * 0.5

    def test_find_transitive_mappings_cache(self):
        """Test caching behavior."""
        index = MappingIndex()

        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="wordnet",
                target_id="give%2:40:00",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.9, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.1",
                    validation_status="validated",
                ),
            )
        )

        # First call should compute and cache
        results1 = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=2)

        # Second call should return cached results
        results2 = index.find_transitive_mappings("verbnet:give-13.1", "framenet", max_hops=2)

        # Should be the same object (cached)
        assert results1 is results2

        # Cache key should be in transitive_cache (includes max_hops)
        cache_key = ("verbnet:give-13.1", "framenet", 2)
        assert cache_key in index.transitive_cache

    def test_find_transitive_mappings_multiple_targets(self):
        """Test with mapping that has multiple target IDs."""
        index = MappingIndex()

        # Mapping with multiple targets
        index.add_mapping(
            CrossReference(
                source_dataset="propbank",
                source_id="give.01",
                source_version="3.1",
                target_dataset="verbnet",
                target_id=["give-13.1", "give-13.1-1"],
                mapping_type="direct",
                confidence=MappingConfidence(score=0.9, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="3.4",
                    validation_status="validated",
                ),
            )
        )

        # Add paths from both verbnet variants to framenet
        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Giving",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.85, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="1.7",
                    validation_status="validated",
                ),
            )
        )
        index.add_mapping(
            CrossReference(
                source_dataset="verbnet",
                source_id="give-13.1-1",
                source_version="3.4",
                target_dataset="framenet",
                target_id="Transfer",
                mapping_type="direct",
                confidence=MappingConfidence(score=0.8, method="manual"),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by="manual",
                    version="1.7",
                    validation_status="validated",
                ),
            )
        )

        results = index.find_transitive_mappings("propbank:give.01", "framenet", max_hops=2)

        # Should find paths through both verbnet variants
        assert len(results) == 2
        target_ids = {r.target_id for r in results}
        assert "Giving" in target_ids
        assert "Transfer" in target_ids


class TestLexLink:
    """Test lexical link model."""

    def test_confidence_validation(self):
        """Test confidence score validation for LexLink."""
        # Valid confidence
        link = LexLink(
            class_name="give-13.1",
            confidence=0.85,
            resource="VerbNet",
            version="3.4",
            src="automatic",
        )
        assert link.confidence == 0.85

        # Invalid confidence
        with pytest.raises(ValidationError) as exc_info:
            LexLink(
                class_name="give-13.1",
                confidence=1.5,
                resource="VerbNet",
                version="3.4",
                src="automatic",
            )
        assert "between 0 and 1" in str(exc_info.value)
