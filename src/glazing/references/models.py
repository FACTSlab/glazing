"""Cross-reference models for mapping between linguistic datasets.

This module provides models for managing cross-references between FrameNet,
PropBank, VerbNet, and WordNet. It includes confidence scoring, transitive
mapping resolution, and unified representations across datasets.

Classes
-------
MappingMetadata
    Metadata for cross-dataset mappings.
MappingConfidence
    Confidence scoring for mappings.
CrossReference
    Base mapping between entities in different datasets.
MultiMapping
    One-to-many mapping with ranked alternatives.
TransitiveMapping
    Indirect mapping through intermediate resources.
VerbNetFrameNetMapping
    VerbNet to FrameNet mapping with confidence.
VerbNetFrameNetRoleMapping
    Role-level mapping between VerbNet and FrameNet.
VerbNetCrossRefs
    VerbNet member cross-references.
PropBankCrossRefs
    PropBank roleset cross-references with confidence.
PropBankRoleMapping
    PropBank role to other dataset role mapping.
UnifiedRoleMapping
    Complete role mapping across all datasets.
UnifiedLemma
    A lemma with all its representations across datasets.
ConceptAlignment
    Alignment of semantic concepts across datasets.
RoleMappingTable
    Maps roles across different datasets.
FEAlignment
    Cross-dataset FE alignment with full metadata.
FEInheritanceChain
    Tracks FE inheritance through frame hierarchy.
MappingConflict
    Represents a conflict in cross-dataset mappings.
MappingIndex
    Bidirectional index for fast mapping lookups.

Notes
-----
All confidence scores are normalized to 0.0-1.0 range.
Transitive mappings propagate confidence through the chain.
"""

import re
from collections import deque
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from glazing.propbank.models import LexLink, RoleLink
from glazing.types import LEMMA_PATTERN, DatasetType, MappingSource
from glazing.wordnet.models import Sense, WordNetCrossRef
from glazing.wordnet.types import SynsetOffset

# Type aliases for cross-references
type ValidationStatus = Literal["validated", "unvalidated", "disputed", "deprecated"]
type MappingType = Literal["direct", "inferred", "transitive", "manual", "automatic", "hybrid"]
type AlignmentType = Literal["direct", "inherited", "inferred", "partial"]
type ConflictType = Literal["ambiguous", "contradictory", "version_mismatch", "inheritance"]


class MappingMetadata(BaseModel):
    """Metadata for cross-dataset mappings.

    Attributes
    ----------
    created_date : datetime
        When the mapping was created.
    created_by : str
        Person or system that created mapping.
    modified_date : datetime | None
        When the mapping was last modified.
    modified_by : str | None
        Person or system that last modified mapping.
    version : str
        Dataset version this mapping was created for.
    validation_status : ValidationStatus
        Current validation status of the mapping.
    validation_method : str | None
        How the mapping was validated.
    notes : str | None
        Additional notes about the mapping.
    """

    created_date: datetime
    created_by: str
    modified_date: datetime | None = None
    modified_by: str | None = None
    version: str
    validation_status: ValidationStatus
    validation_method: str | None = None
    notes: str | None = None


class MappingConfidence(BaseModel):
    """Confidence scoring for mappings.

    Attributes
    ----------
    score : float
        Confidence score between 0.0 and 1.0.
    method : str
        Method used to calculate confidence.
    factors : dict[str, float]
        Component scores contributing to overall confidence.

    Raises
    ------
    ValueError
        If score is not between 0.0 and 1.0.
    """

    score: float
    method: str
    factors: dict[str, float] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate confidence score is in valid range.

        Parameters
        ----------
        v : float
            Score to validate.

        Returns
        -------
        float
            Validated score.

        Raises
        ------
        ValueError
            If score is not between 0.0 and 1.0.
        """
        if not 0.0 <= v <= 1.0:
            msg = f"Confidence score must be between 0 and 1: {v}"
            raise ValueError(msg)
        return v


class CrossReference(BaseModel):
    """Cross-dataset reference with full metadata.

    Attributes
    ----------
    source_dataset : DatasetType
        Source dataset name.
    source_id : str
        Identifier in source dataset.
    source_version : str
        Version of source dataset.
    target_dataset : DatasetType
        Target dataset name.
    target_id : str | list[str]
        Identifier(s) in target dataset.
    mapping_type : MappingType
        Type of mapping relationship.
    confidence : MappingConfidence | None
        Confidence scoring for the mapping.
    metadata : MappingMetadata
        Metadata about the mapping.
    inherited_from : str | None
        For mappings inherited from parent classes.
    """

    source_dataset: DatasetType
    source_id: str
    source_version: str
    target_dataset: DatasetType
    target_id: str | list[str]
    mapping_type: MappingType
    confidence: MappingConfidence | None = None
    metadata: MappingMetadata
    inherited_from: str | None = None


class MultiMapping(BaseModel):
    """One-to-many mapping with ranked alternatives.

    Attributes
    ----------
    source_dataset : DatasetType
        Source dataset name.
    source_id : str
        Identifier in source dataset.
    source_version : str
        Version of source dataset.
    mappings : list[CrossReference]
        Ordered list of mappings by confidence.
    primary_mapping : CrossReference | None
        Highest confidence or manually selected mapping.
    conflict_resolution : str | None
        How conflicts were resolved.
    """

    source_dataset: DatasetType
    source_id: str
    source_version: str
    mappings: list[CrossReference]
    primary_mapping: CrossReference | None = None
    conflict_resolution: str | None = None

    def get_best_mapping(self, target_dataset: DatasetType) -> CrossReference | None:
        """Get highest confidence mapping to target dataset.

        Parameters
        ----------
        target_dataset : DatasetType
            Target dataset to find mapping for.

        Returns
        -------
        CrossReference | None
            Highest confidence mapping or None if not found.
        """
        candidates = [m for m in self.mappings if m.target_dataset == target_dataset]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.confidence.score if m.confidence else 0.0)


class TransitiveMapping(BaseModel):
    """Indirect mapping through intermediate resource.

    Attributes
    ----------
    source_dataset : DatasetType
        Source dataset name.
    source_id : str
        Identifier in source dataset.
    target_dataset : DatasetType
        Target dataset name.
    target_id : str
        Identifier in target dataset.
    path : list[CrossReference]
        Chain of mappings from source to target.
    combined_confidence : float
        Propagated confidence through chain.
    """

    source_dataset: DatasetType
    source_id: str
    target_dataset: DatasetType
    target_id: str
    path: list[CrossReference]
    combined_confidence: float

    def calculate_confidence(self) -> float:
        """Calculate combined confidence through chain.

        Returns
        -------
        float
            Combined confidence score.
        """
        if not self.path:
            return 0.0
        scores = [m.confidence.score for m in self.path if m.confidence]
        if not scores:
            return 0.0
        # Multiply confidences (assuming independence)
        result = 1.0
        for score in scores:
            result *= score
        return result


class VerbNetFrameNetMapping(BaseModel):
    """VerbNet to FrameNet mapping with confidence.

    Attributes
    ----------
    frame_name : str
        FrameNet frame name.
    confidence : MappingConfidence | None
        Confidence score for the mapping.
    mapping_source : MappingSource
        Source of the mapping.
    role_mappings : list[VerbNetFrameNetRoleMapping]
        Role-level mappings between VerbNet and FrameNet.
    """

    frame_name: str
    confidence: MappingConfidence | None = None
    mapping_source: MappingSource
    role_mappings: list["VerbNetFrameNetRoleMapping"] = Field(default_factory=list)


class VerbNetFrameNetRoleMapping(BaseModel):
    """Role-level mapping between VerbNet and FrameNet.

    Attributes
    ----------
    vn_role : str
        VerbNet thematic role.
    fn_fe : str
        FrameNet frame element.
    confidence : float | None
        Confidence score for the role mapping.
    notes : str | None
        Additional notes about the mapping.
    """

    vn_role: str
    fn_fe: str
    confidence: float | None = None
    notes: str | None = None


class VerbNetCrossRefs(BaseModel):
    """VerbNet member cross-references.

    Attributes
    ----------
    verbnet_key : str
        Unique identifier for VerbNet member.
    class_id : str
        VerbNet class this belongs to.
    lemma : str
        Base form of the verb.
    fn_mappings : list[VerbNetFrameNetMapping]
        Multi-way FrameNet mappings with confidence.
    pb_groupings : list[str]
        PropBank senses.
    wn_mappings : list[WordNetCrossRef]
        WordNet mappings with metadata.
    inherited_mappings : list[CrossReference]
        Inherited mappings from parent classes.
    """

    verbnet_key: str
    class_id: str
    lemma: str
    fn_mappings: list[VerbNetFrameNetMapping] = Field(default_factory=list)
    pb_groupings: list[str] = Field(default_factory=list)
    wn_mappings: list[WordNetCrossRef] = Field(default_factory=list)
    inherited_mappings: list[CrossReference] = Field(default_factory=list)

    def get_primary_framenet_mapping(self) -> VerbNetFrameNetMapping | None:
        """Get highest confidence FrameNet mapping.

        Returns
        -------
        VerbNetFrameNetMapping | None
            Highest confidence mapping or None if not found.
        """
        if not self.fn_mappings:
            return None
        return max(self.fn_mappings, key=lambda m: m.confidence.score if m.confidence else 0.0)

    def has_conflicting_mappings(self) -> bool:
        """Check if there are conflicting FrameNet mappings.

        Returns
        -------
        bool
            True if multiple high-confidence mappings exist.
        """
        if len(self.fn_mappings) <= 1:
            return False
        # Check if multiple mappings have high confidence
        high_conf = [m for m in self.fn_mappings if m.confidence and m.confidence.score > 0.7]
        return len(high_conf) > 1


class PropBankCrossRefs(BaseModel):
    """PropBank roleset cross-references with confidence.

    Attributes
    ----------
    roleset_id : str
        PropBank roleset identifier.
    rolelinks : list[RoleLink]
        Direct role-level mappings.
    lexlinks : list[LexLink]
        Lexical links with confidence scores.
    wn_mappings : list[WordNetCrossRef]
        WordNet mappings if available.
    """

    roleset_id: str
    rolelinks: list[RoleLink]
    lexlinks: list[LexLink]
    wn_mappings: list[WordNetCrossRef] = Field(default_factory=list)

    def get_verbnet_classes(self) -> list[tuple[str, float | None]]:
        """Get VerbNet classes with optional confidence.

        Returns
        -------
        list[tuple[str, float | None]]
            VerbNet class names with confidence scores.
        """
        vn_classes: list[tuple[str, float | None]] = []
        # From rolelinks (no confidence)
        for rl in self.rolelinks:
            if rl.resource == "verbnet":
                vn_classes.append((rl.class_name, None))
        # From lexlinks (with confidence)
        for ll in self.lexlinks:
            if ll.resource == "verbnet":
                vn_classes.append((ll.class_name, ll.confidence))
        return vn_classes

    def get_wordnet_senses(self) -> list[WordNetCrossRef]:
        """Get WordNet senses for this roleset.

        Returns
        -------
        list[WordNetCrossRef]
            WordNet cross-references.
        """
        return self.wn_mappings


class PropBankRoleMapping(BaseModel):
    """PropBank role to other dataset role mapping.

    Attributes
    ----------
    pb_arg : str
        PropBank argument (e.g., "ARG0").
    target_dataset : DatasetType
        Target dataset name.
    target_role : str
        Target role/FE name.
    confidence : MappingConfidence | None
        Confidence scoring for the mapping.
    mapping_source : MappingSource
        Source of the mapping.
    """

    pb_arg: str
    target_dataset: DatasetType
    target_role: str
    confidence: MappingConfidence | None = None
    mapping_source: MappingSource


class UnifiedRoleMapping(BaseModel):
    """Complete role mapping across all datasets.

    Attributes
    ----------
    concept : str
        Semantic concept (e.g., "agent", "patient").
    verbnet_roles : list[tuple[str, str]]
        VerbNet (class_id, role) pairs.
    framenet_fes : list[tuple[str, str]]
        FrameNet (frame, fe) pairs.
    propbank_args : list[tuple[str, str]]
        PropBank (roleset, arg) pairs.
    wordnet_restrictions : list[str]
        Inferred semantic restrictions from WordNet.
    confidence_matrix : dict[str, dict[str, float]]
        Pairwise confidence scores between mappings.
    """

    concept: str
    verbnet_roles: list[tuple[str, str]]
    framenet_fes: list[tuple[str, str]]
    propbank_args: list[tuple[str, str]]
    wordnet_restrictions: list[str]
    confidence_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)

    def get_alignment_score(self) -> float:
        """Calculate overall alignment score.

        Returns
        -------
        float
            Average confidence across all pairwise mappings.
        """
        if not self.confidence_matrix:
            return 0.0
        scores: list[float] = []
        for source in self.confidence_matrix.values():
            scores.extend(source.values())
        return sum(scores) / len(scores) if scores else 0.0


class FrameNetLURef(BaseModel):
    """Reference to a FrameNet lexical unit.

    Attributes
    ----------
    lu_id : int
        Lexical unit ID.
    frame_name : str
        Frame that contains this LU.
    definition : str
        LU definition.
    """

    lu_id: int
    frame_name: str
    definition: str


class PropBankRolesetRef(BaseModel):
    """Reference to a PropBank roleset.

    Attributes
    ----------
    roleset_id : str
        Roleset identifier.
    name : str | None
        Descriptive name.
    """

    roleset_id: str
    name: str | None = None


class VerbNetMemberRef(BaseModel):
    """Reference to a VerbNet member.

    Attributes
    ----------
    verbnet_key : str
        Unique member identifier.
    class_id : str
        VerbNet class ID.
    """

    verbnet_key: str
    class_id: str


class UnifiedLemma(BaseModel):
    """A lemma with all its representations across datasets.

    Attributes
    ----------
    lemma : str
        Base lemma form.
    pos : Literal[...]
        Unified part of speech.
    framenet_lus : list[FrameNetLURef]
        FrameNet lexical units.
    propbank_rolesets : list[PropBankRolesetRef]
        PropBank rolesets.
    verbnet_members : list[VerbNetMemberRef]
        VerbNet members with class membership.
    wordnet_senses : list[Sense]
        WordNet senses.
    """

    lemma: str
    pos: Literal["n", "v", "a", "r", "s"]  # WordNet POS tags only
    framenet_lus: list[FrameNetLURef]
    propbank_rolesets: list[PropBankRolesetRef]
    verbnet_members: list[VerbNetMemberRef]
    wordnet_senses: list[Sense]

    @field_validator("lemma")
    @classmethod
    def validate_lemma(cls, v: str) -> str:
        """Validate lemma format.

        Parameters
        ----------
        v : str
            Lemma to validate.

        Returns
        -------
        str
            Validated lemma.

        Raises
        ------
        ValueError
            If lemma format is invalid.
        """
        if not re.match(LEMMA_PATTERN, v):
            msg = f"Invalid lemma format: {v!r}"
            raise ValueError(msg)
        return v


class ConceptAlignment(BaseModel):
    """Alignment of semantic concepts across datasets.

    Attributes
    ----------
    concept_name : str
        Name of the semantic concept.
    concept_type : str
        Type of concept (e.g., "frame", "event").
    framenet_frames : list[str]
        Related FrameNet frames.
    propbank_rolesets : list[str]
        Related PropBank rolesets.
    verbnet_classes : list[str]
        Related VerbNet classes.
    wordnet_synsets : list[SynsetOffset]
        Related WordNet synsets.
    confidence : float | None
        Overall alignment confidence.
    alignment_method : str
        Method used for alignment.
    alignment_criteria : list[str]
        Criteria used for alignment.
    """

    concept_name: str
    concept_type: str
    framenet_frames: list[str]
    propbank_rolesets: list[str]
    verbnet_classes: list[str]
    wordnet_synsets: list[SynsetOffset]
    confidence: float | None = None
    alignment_method: str
    alignment_criteria: list[str]


class RoleMappingTable(BaseModel):
    """Maps roles across different datasets.

    Attributes
    ----------
    verbnet_role : str
        VerbNet thematic role.
    framenet_fe : str | None
        Corresponding FrameNet FE.
    propbank_arg : str | None
        Corresponding PropBank argument.
    wordnet_semantic_role : str | None
        Inferred WordNet semantic role.
    mapping_notes : str
        Notes about the mapping.
    """

    verbnet_role: str
    framenet_fe: str | None = None
    propbank_arg: str | None = None
    wordnet_semantic_role: str | None = None
    mapping_notes: str = ""

    def is_agentive(self) -> bool:
        """Check if this represents an agentive role.

        Returns
        -------
        bool
            True if role is agentive.
        """
        agentive_roles = {"Agent", "ARG0", "Actor", "Causer"}
        return bool(
            self.verbnet_role in agentive_roles
            or self.propbank_arg == "ARG0"
            or (self.framenet_fe and "Agent" in self.framenet_fe)
        )


class FEAlignment(BaseModel):
    """Cross-dataset FE alignment with full metadata.

    Attributes
    ----------
    source_frame : str
        FrameNet frame name.
    source_fe : str
        FrameNet FE name.
    target_dataset : DatasetType
        Target dataset.
    target_role : str
        Target role/arg name.
    alignment_type : AlignmentType
        Type of alignment.
    confidence : MappingConfidence
        Confidence scoring.
    evidence : list[str]
        Supporting evidence for alignment.
    """

    source_frame: str
    source_fe: str
    target_dataset: DatasetType
    target_role: str
    alignment_type: AlignmentType
    confidence: MappingConfidence
    evidence: list[str] = Field(default_factory=list)

    def get_combined_score(self) -> float:
        """Get combined alignment score.

        Returns
        -------
        float
            Adjusted confidence score based on alignment type.
        """
        base_score = self.confidence.score
        if self.alignment_type == "inherited":
            base_score *= 0.9
        elif self.alignment_type == "inferred":
            base_score *= 0.8
        elif self.alignment_type == "partial":
            base_score *= 0.7
        return base_score


class FERelation(BaseModel):
    """FE mapping between related frames with alignment metadata.

    Attributes
    ----------
    sub_fe_id : int | None
        Sub-frame FE ID.
    sub_fe_name : str | None
        Sub-frame FE name.
    super_fe_id : int | None
        Super-frame FE ID.
    super_fe_name : str | None
        Super-frame FE name.
    relation_type : str | None
        Type of relation.
    alignment_confidence : float | None
        Alignment confidence score.
    semantic_similarity : float | None
        Semantic similarity score.
    syntactic_similarity : float | None
        Syntactic similarity score.
    mapping_notes : str | None
        Notes about the mapping.
    """

    sub_fe_id: int | None = Field(None, alias="subID")
    sub_fe_name: str | None = Field(None, alias="subFEName")
    super_fe_id: int | None = Field(None, alias="supID")
    super_fe_name: str | None = Field(None, alias="supFEName")
    relation_type: str | None = None
    alignment_confidence: float | None = None
    semantic_similarity: float | None = None
    syntactic_similarity: float | None = None
    mapping_notes: str | None = None

    @field_validator("sub_fe_name", "super_fe_name")
    @classmethod
    def validate_fe_names(cls, v: str | None) -> str | None:
        """Validate FE name format.

        Parameters
        ----------
        v : str | None
            FE name to validate.

        Returns
        -------
        str | None
            Validated FE name.

        Raises
        ------
        ValueError
            If FE name format is invalid.
        """
        if v and not re.match(r"^[A-Z][A-Za-z0-9_]*$", v):
            msg = f"Invalid FE name format: {v}"
            raise ValueError(msg)
        return v

    def is_inheritance(self) -> bool:
        """Check if this is an inheritance relation.

        Returns
        -------
        bool
            True if relation type is inheritance.
        """
        return self.relation_type == "Inheritance"

    def is_equivalence(self) -> bool:
        """Check if FEs are equivalent.

        Returns
        -------
        bool
            True if relation type is equivalence.
        """
        return self.relation_type == "Equivalence"


class FEInheritanceChain(BaseModel):
    """Tracks FE inheritance through frame hierarchy.

    Attributes
    ----------
    fe_name : str
        Frame element name.
    frame_chain : list[str]
        Frames from child to parent.
    inheritance_path : list[FERelation]
        Chain of FE relations.
    final_mapping : FEAlignment | None
        Final cross-dataset mapping.
    """

    fe_name: str
    frame_chain: list[str]
    inheritance_path: list[FERelation]
    final_mapping: FEAlignment | None = None

    def get_inheritance_depth(self) -> int:
        """Get depth of inheritance chain.

        Returns
        -------
        int
            Number of inheritance steps.
        """
        return len(self.frame_chain) - 1


class MappingConflict(BaseModel):
    """Represents a conflict in cross-dataset mappings.

    Attributes
    ----------
    conflict_type : ConflictType
        Type of conflict.
    source_dataset : DatasetType
        Source dataset.
    source_id : str
        Source identifier.
    conflicting_mappings : list[CrossReference]
        Conflicting mapping alternatives.
    resolution_strategy : str | None
        Strategy used to resolve conflict.
    resolved_mapping : CrossReference | None
        Resolved mapping after conflict resolution.
    """

    conflict_type: ConflictType
    source_dataset: DatasetType
    source_id: str
    conflicting_mappings: list[CrossReference]
    resolution_strategy: str | None = None
    resolved_mapping: CrossReference | None = None

    def resolve_by_confidence(self) -> CrossReference | None:
        """Resolve conflict by selecting highest confidence mapping.

        Returns
        -------
        CrossReference | None
            Highest confidence mapping or None.
        """
        if not self.conflicting_mappings:
            return None
        return max(
            self.conflicting_mappings, key=lambda m: m.confidence.score if m.confidence else 0.0
        )

    def resolve_by_source(self, preferred_source: MappingSource) -> CrossReference | None:
        """Resolve by preferring specific mapping source.

        Parameters
        ----------
        preferred_source : MappingSource
            Preferred source for resolution.

        Returns
        -------
        CrossReference | None
            Mapping from preferred source or None.
        """
        for mapping in self.conflicting_mappings:
            if mapping.metadata.created_by == preferred_source:
                return mapping
        return None


class MappingIndex(BaseModel):
    """Bidirectional index for fast mapping lookups.

    Attributes
    ----------
    forward_index : dict[str, list[CrossReference]]
        Source to target mappings.
    reverse_index : dict[str, list[CrossReference]]
        Target to source mappings.
    transitive_cache : dict[tuple[str, str], list[TransitiveMapping]]
        Cached transitive mappings.
    """

    forward_index: dict[str, list[CrossReference]] = Field(default_factory=dict)
    reverse_index: dict[str, list[CrossReference]] = Field(default_factory=dict)
    transitive_cache: dict[tuple[str, str, int], list[TransitiveMapping]] = Field(
        default_factory=dict
    )

    def add_mapping(self, mapping: CrossReference) -> None:
        """Add mapping to bidirectional index.

        Parameters
        ----------
        mapping : CrossReference
            Mapping to add to index.
        """
        # Forward index
        key = f"{mapping.source_dataset}:{mapping.source_id}"
        if key not in self.forward_index:
            self.forward_index[key] = []
        self.forward_index[key].append(mapping)

        # Reverse index
        targets = mapping.target_id if isinstance(mapping.target_id, list) else [mapping.target_id]
        for target in targets:
            key = f"{mapping.target_dataset}:{target}"
            if key not in self.reverse_index:
                self.reverse_index[key] = []
            self.reverse_index[key].append(mapping)

    def find_transitive_mappings(
        self, source: str, target_dataset: DatasetType, max_hops: int = 3
    ) -> list[TransitiveMapping]:
        """Find indirect mappings through intermediate resources.

        Parameters
        ----------
        source : str
            Source identifier.
        target_dataset : DatasetType
            Target dataset.
        max_hops : int
            Maximum number of intermediate steps.

        Returns
        -------
        list[TransitiveMapping]
            Found transitive mappings.
        """
        cache_key = (source, target_dataset, max_hops)
        if cache_key in self.transitive_cache:
            return self.transitive_cache[cache_key]

        mappings = self._search_transitive_mappings(source, target_dataset, max_hops)
        mappings.sort(key=lambda m: m.combined_confidence, reverse=True)

        self.transitive_cache[cache_key] = mappings
        return mappings

    def _search_transitive_mappings(
        self, source: str, target_dataset: DatasetType, max_hops: int
    ) -> list[TransitiveMapping]:
        """Perform breadth-first search for transitive mappings.

        Parameters
        ----------
        source : str
            Source identifier.
        target_dataset : DatasetType
            Target dataset.
        max_hops : int
            Maximum number of intermediate steps.

        Returns
        -------
        list[TransitiveMapping]
            Found transitive mappings.
        """
        mappings: list[TransitiveMapping] = []
        visited: set[str] = set()
        queue: deque[tuple[str, list[CrossReference], int]] = deque()

        queue.append((source, [], 0))
        visited.add(source)

        while queue:
            current_node, path, hops = queue.popleft()
            direct_mappings = self.forward_index.get(current_node, [])

            for mapping in direct_mappings:
                new_path = [*path, mapping]
                targets = self._get_target_list(mapping.target_id)

                for target in targets:
                    target_key = f"{mapping.target_dataset}:{target}"

                    if mapping.target_dataset == target_dataset:
                        transitive = self._create_transitive_mapping(
                            source, target, target_dataset, new_path
                        )
                        mappings.append(transitive)
                    elif self._should_continue_search(target_key, visited, hops, max_hops):
                        visited.add(target_key)
                        queue.append((target_key, new_path, hops + 1))

        return mappings

    def _get_target_list(self, target_id: str | list[str]) -> list[str]:
        """Get normalized list of target IDs.

        Parameters
        ----------
        target_id : str | list[str]
            Target ID(s).

        Returns
        -------
        list[str]
            List of target IDs.
        """
        return target_id if isinstance(target_id, list) else [target_id]

    def _should_continue_search(
        self, target_key: str, visited: set[str], hops: int, max_hops: int
    ) -> bool:
        """Check if search should continue to this target.

        Parameters
        ----------
        target_key : str
            Target key to check.
        visited : set[str]
            Set of visited nodes.
        hops : int
            Current hop count.
        max_hops : int
            Maximum allowed hops.

        Returns
        -------
        bool
            True if search should continue.
        """
        return target_key not in visited and hops + 1 < max_hops

    def _create_transitive_mapping(
        self, source: str, target: str, target_dataset: DatasetType, path: list[CrossReference]
    ) -> TransitiveMapping:
        """Create a transitive mapping from search results.

        Parameters
        ----------
        source : str
            Original source identifier.
        target : str
            Target identifier.
        target_dataset : DatasetType
            Target dataset.
        path : list[CrossReference]
            Path of references leading to target.

        Returns
        -------
        TransitiveMapping
            Created transitive mapping.
        """
        source_dataset, source_id = self._parse_source_identifier(source, path)
        combined_confidence = self._calculate_combined_confidence(path)

        return TransitiveMapping(
            source_dataset=source_dataset,
            source_id=source_id,
            target_dataset=target_dataset,
            target_id=target,
            path=path,
            combined_confidence=combined_confidence,
        )

    def _parse_source_identifier(
        self, source: str, path: list[CrossReference]
    ) -> tuple[DatasetType, str]:
        """Parse source identifier to extract dataset and ID.

        Parameters
        ----------
        source : str
            Source identifier.
        path : list[CrossReference]
            Reference path for fallback dataset.

        Returns
        -------
        tuple[DatasetType, str]
            Source dataset and ID.
        """
        source_parts = source.split(":")
        if len(source_parts) == 2:
            source_dataset_str, source_id = source_parts
            if source_dataset_str in ["framenet", "propbank", "verbnet", "wordnet"]:
                return source_dataset_str, source_id  # type: ignore[return-value]

        # Fallback to path or default
        fallback_dataset = path[0].source_dataset if path else "framenet"
        source_id = source_parts[1] if len(source_parts) == 2 else source
        return fallback_dataset, source_id

    def _calculate_combined_confidence(self, path: list[CrossReference]) -> float:
        """Calculate combined confidence for a path.

        Parameters
        ----------
        path : list[CrossReference]
            Path of cross-references.

        Returns
        -------
        float
            Combined confidence score.
        """
        combined_confidence = 1.0
        for ref in path:
            if ref.confidence:
                combined_confidence *= ref.confidence.score
            else:
                combined_confidence *= 0.5
        return combined_confidence
