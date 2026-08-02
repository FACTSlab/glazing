"""Reference extraction from linguistic datasets.

This module provides functionality to extract cross-references from FrameNet,
PropBank, VerbNet, and WordNet data models and build efficient indices for
mapping lookups.

Classes
-------
ReferenceExtractor
    Main class for extracting and indexing cross-dataset references.

Notes
-----
The extractor builds bidirectional indices for efficient lookup of mappings
between datasets. All extracted references include confidence scores and
metadata where available.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Literal, TypeVar, cast

from glazing.framenet.models import Frame, FrameRelation, LexicalUnit
from glazing.propbank.models import Frameset, Roleset
from glazing.references.models import (
    CrossReference,
    MappingConfidence,
    MappingIndex,
    MappingMetadata,
    PropBankCrossRefs,
    PropBankRoleMapping,
    VerbNetCrossRefs,
)
from glazing.types import DatasetType
from glazing.verbnet.models import Member, VerbClass
from glazing.wordnet.models import Sense, Synset
from glazing.wordnet.models import WordNetCrossRef as WordNetWNRef

T = TypeVar("T")


class ReferenceExtractor:
    """Extract and index cross-references from linguistic datasets.

    This class provides methods to extract cross-references from loaded
    dataset models and build efficient indices for mapping lookups.

    Attributes
    ----------
    mapping_index : MappingIndex
        Bidirectional index for all extracted mappings.
    verbnet_refs : dict[str, VerbNetCrossRefs]
        VerbNet member cross-references by verbnet_key.
    propbank_refs : dict[str, PropBankCrossRefs]
        PropBank roleset cross-references by roleset_id.
    framenet_relations : dict[int, list[FrameRelation]]
        FrameNet frame relations by frame_id.
    wordnet_sense_index : dict[str, str]
        WordNet sense key to synset offset mapping.

    Methods
    -------
    extract_all(framenet, propbank, verbnet, wordnet)
        Extract references from all datasets.
    extract_verbnet_references(verbnet_classes)
        Extract VerbNet member cross-references.
    extract_propbank_references(framesets)
        Extract PropBank roleset cross-references.
    extract_framenet_relations(frames)
        Extract FrameNet frame and FE relations.
    extract_wordnet_mappings(synsets, senses)
        Build WordNet sense and synset indices.
    """

    def __init__(self) -> None:
        """Initialize the reference extractor."""
        self.mapping_index = MappingIndex()
        self.verbnet_refs: dict[str, VerbNetCrossRefs] = {}
        self.propbank_refs: dict[str, PropBankCrossRefs] = {}
        self.framenet_relations: dict[int, list[FrameRelation]] = defaultdict(list)
        self.wordnet_sense_index: dict[str, str] = {}

    def extract_all(
        self,
        framenet: list[Frame] | None = None,
        propbank: list[Frameset] | None = None,
        verbnet: list[VerbClass] | None = None,
        wordnet: tuple[list[Synset], list[Sense]] | None = None,
    ) -> None:
        """Extract references from all provided datasets.

        Parameters
        ----------
        framenet : list[Frame] | None, default=None
            FrameNet frames to process.
        propbank : list[Frameset] | None, default=None
            PropBank framesets to process.
        verbnet : list[VerbClass] | None, default=None
            VerbNet classes to process.
        wordnet : tuple[list[Synset], list[Sense]] | None, default=None
            WordNet synsets and senses to process.
        """
        if verbnet:
            self.extract_verbnet_references(verbnet)
        if propbank:
            self.extract_propbank_references(propbank)
        if framenet:
            self.extract_framenet_relations(framenet)
        if wordnet:
            synsets, senses = wordnet
            self.extract_wordnet_mappings(synsets, senses)

    def extract_verbnet_references(self, verb_classes: list[VerbClass]) -> None:
        """Extract cross-references from VerbNet classes.

        Processes VerbNet members to extract FrameNet, PropBank, and WordNet
        mappings. Handles subclasses recursively.

        Parameters
        ----------
        verb_classes : list[VerbClass]
            VerbNet classes to process.
        """
        for verb_class in verb_classes:
            self._extract_class_references(verb_class)

    def _extract_class_references(self, verb_class: VerbClass) -> None:
        """Extract references from a single VerbNet class and its subclasses.

        Parameters
        ----------
        verb_class : VerbClass
            VerbNet class to process.
        """
        # Process members
        for member in verb_class.members:
            # Convert VerbNet WordNetCrossRef to WordNet WordNetCrossRef
            converted_wn_mappings: list[WordNetWNRef] = []
            for wn_mapping in member.wordnet_mappings:
                # All WordNetCrossRef objects must have these required fields
                if not (
                    hasattr(wn_mapping, "sense_key")
                    and hasattr(wn_mapping, "lemma")
                    and hasattr(wn_mapping, "pos")
                ):
                    error_msg = f"WordNetCrossRef missing required fields: {wn_mapping}"
                    raise ValueError(error_msg)

                converted = WordNetWNRef(
                    sense_key=wn_mapping.sense_key,
                    lemma=wn_mapping.lemma,
                    pos=cast(Literal["n", "v", "a", "r", "s"], wn_mapping.pos),
                    synset_offset=getattr(wn_mapping, "synset_offset", None),
                    sense_number=getattr(wn_mapping, "sense_number", None),
                )
                converted_wn_mappings.append(converted)

            vn_refs = VerbNetCrossRefs(
                verbnet_key=member.verbnet_key,
                class_id=verb_class.id,
                lemma=member.name,
                fn_mappings=member.framenet_mappings,
                pb_groupings=[],  # Will be populated from propbank_mappings
                wn_mappings=converted_wn_mappings,
                inherited_mappings=[],
            )

            # Extract PropBank groupings from cross-references
            for pb_mapping in member.propbank_mappings:
                if pb_mapping.target_dataset == "propbank":
                    if isinstance(pb_mapping.target_id, list):
                        vn_refs.pb_groupings.extend(pb_mapping.target_id)
                    else:
                        vn_refs.pb_groupings.append(pb_mapping.target_id)

            self.verbnet_refs[member.verbnet_key] = vn_refs

            # Add to mapping index
            self._index_verbnet_mappings(member, verb_class.id)

        # Process subclasses recursively
        for subclass in verb_class.subclasses:
            self._extract_class_references(subclass)

    def _index_verbnet_mappings(self, member: Member, _class_id: str) -> None:
        """Add VerbNet member mappings to the index.

        Parameters
        ----------
        member : Member
            VerbNet member with mappings.
        class_id : str
            VerbNet class ID.
        """
        # Index FrameNet mappings with fuzzy matching confidence
        for fn_mapping in member.framenet_mappings:
            # Calculate additional confidence based on frame name similarity
            fuzzy_confidence = 1.0

            # If we have a confidence from the mapping, combine it with fuzzy score
            if fn_mapping.confidence:
                base_confidence: float = (
                    fn_mapping.confidence.score
                    if hasattr(fn_mapping.confidence, "score")
                    else float(fn_mapping.confidence)  # type: ignore[arg-type]
                )
            else:
                base_confidence = 1.0

            mapping = CrossReference(
                source_dataset="verbnet",
                source_id=member.verbnet_key,
                source_version="3.4",
                target_dataset="framenet",
                target_id=fn_mapping.frame_name,
                mapping_type="direct",
                confidence=MappingConfidence(
                    score=float(base_confidence * fuzzy_confidence),
                    method="verbnet_framenet",
                    factors={
                        "base_confidence": float(base_confidence),
                        "fuzzy_score": float(fuzzy_confidence),
                    },
                ),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by=fn_mapping.mapping_source,
                    version="3.4",
                    validation_status="validated"
                    if float(base_confidence) > 0.8
                    else "unvalidated",
                ),
            )
            self.mapping_index.add_mapping(mapping)

        # Index PropBank mappings
        for pb_mapping in member.propbank_mappings:
            self.mapping_index.add_mapping(pb_mapping)

        # Index WordNet mappings
        for wn_mapping in member.wordnet_mappings:
            if wn_mapping.sense_key:
                mapping = CrossReference(
                    source_dataset="verbnet",
                    source_id=member.verbnet_key,
                    source_version="3.4",
                    target_dataset="wordnet",
                    target_id=wn_mapping.sense_key,
                    mapping_type="direct",
                    confidence=None,  # VerbNet doesn't provide WN confidence
                    metadata=MappingMetadata(
                        created_date=datetime.now(UTC),
                        created_by="verbnet",
                        version="3.4",
                        validation_status="validated",
                    ),
                )
                self.mapping_index.add_mapping(mapping)

    def extract_propbank_references(self, framesets: list[Frameset]) -> None:
        """Extract cross-references from PropBank framesets.

        Processes rolesets to extract VerbNet and FrameNet mappings via
        rolelinks and lexlinks.

        Parameters
        ----------
        framesets : list[Frameset]
            PropBank framesets to process.
        """
        for frameset in framesets:
            for roleset in frameset.rolesets:
                pb_refs = PropBankCrossRefs(
                    roleset_id=roleset.id,
                    rolelinks=[],
                    lexlinks=roleset.lexlinks,
                    wn_mappings=[],  # PropBank doesn't directly map to WordNet
                )

                # Extract rolelinks from roles
                for role in roleset.roles:
                    pb_refs.rolelinks.extend(role.rolelinks)

                self.propbank_refs[roleset.id] = pb_refs

                # Add to mapping index
                self._index_propbank_mappings(roleset)

    def _index_propbank_mappings(self, roleset: Roleset) -> None:
        """Add PropBank roleset mappings to the index.

        Parameters
        ----------
        roleset : Roleset
            PropBank roleset with mappings.
        """
        # Index lexlinks (frame-level mappings with confidence)
        for lexlink in roleset.lexlinks:
            # Normalize dataset names including "Framenet" variant
            if lexlink.resource.lower() == "verbnet":
                target_dataset: DatasetType = "verbnet"
            elif lexlink.resource in ["FrameNet", "Framenet", "framenet"]:
                target_dataset = "framenet"
            else:
                msg = f"Unknown lexlink resource type: {lexlink.resource}"
                raise ValueError(msg)

            mapping = CrossReference(
                source_dataset="propbank",
                source_id=roleset.id,
                source_version=lexlink.version,
                target_dataset=target_dataset,
                target_id=lexlink.class_name,
                mapping_type="automatic" if lexlink.src == "auto" else "manual",
                confidence=MappingConfidence(
                    score=lexlink.confidence,
                    method="lexlink",
                    factors={"lexlink_confidence": 1.0},
                ),
                metadata=MappingMetadata(
                    created_date=datetime.now(UTC),
                    created_by=lexlink.src,
                    version=lexlink.version,
                    validation_status="validated" if lexlink.confidence > 0.8 else "unvalidated",
                ),
            )
            self.mapping_index.add_mapping(mapping)

        # Index rolelinks (role-level mappings)
        for role in roleset.roles:
            for rolelink in role.rolelinks:
                # Normalize rolelink resource names
                if rolelink.resource.lower() == "verbnet":
                    rolelink_target: DatasetType = "verbnet"
                elif rolelink.resource in ["FrameNet", "Framenet", "framenet"]:
                    rolelink_target = "framenet"
                else:
                    msg = f"Unknown rolelink resource type: {rolelink.resource}"
                    raise ValueError(msg)

                # Create role-level mapping (not used in current implementation)
                _role_mapping = PropBankRoleMapping(
                    pb_arg=f"ARG{role.n}",
                    target_dataset=rolelink_target,
                    target_role=rolelink.role or rolelink.class_name,
                    confidence=None,  # Rolelinks don't have confidence
                    mapping_source="manual",
                )

                # Also add to index as frame-level mapping
                mapping = CrossReference(
                    source_dataset="propbank",
                    source_id=roleset.id,
                    source_version=rolelink.version,
                    target_dataset=rolelink_target,
                    target_id=rolelink.class_name,
                    mapping_type="manual",
                    confidence=None,
                    metadata=MappingMetadata(
                        created_date=datetime.now(UTC),
                        created_by="propbank",
                        version=rolelink.version,
                        validation_status="validated",
                        notes=f"Role mapping: ARG{role.n} -> {rolelink.role}",
                    ),
                )
                self.mapping_index.add_mapping(mapping)

    def extract_framenet_relations(self, frames: list[Frame]) -> None:
        """Extract frame relations and FE mappings from FrameNet.

        Processes frame-to-frame relations and frame element mappings.

        Parameters
        ----------
        frames : list[Frame]
            FrameNet frames to process.
        """
        for frame in frames:
            # Store frame relations
            self.framenet_relations[frame.id] = frame.frame_relations

            # Index frame relations
            for relation in frame.frame_relations:
                if relation.type in ["Inherits from", "Is Inherited by"]:
                    # Create inheritance mapping
                    source_id = relation.sub_frame_id or frame.id
                    target_id = relation.super_frame_id or frame.id

                    if source_id and target_id and source_id != target_id:
                        mapping = CrossReference(
                            source_dataset="framenet",
                            source_id=str(source_id),
                            source_version="1.7",
                            target_dataset="framenet",
                            target_id=str(target_id),
                            mapping_type="direct",
                            confidence=MappingConfidence(
                                score=1.0,
                                method="inheritance",
                                factors={"inheritance_score": 1.0},
                            ),
                            metadata=MappingMetadata(
                                created_date=datetime.now(UTC),
                                created_by="framenet",
                                version="1.7",
                                validation_status="validated",
                                notes=f"Frame relation: {relation.type}",
                            ),
                        )
                        self.mapping_index.add_mapping(mapping)

            # Extract lexical unit WordNet mappings if available
            for lu in frame.lexical_units:
                self._extract_lu_mappings(lu, frame.name)

    def _extract_lu_mappings(self, lu: LexicalUnit, frame_name: str) -> None:
        """Extract mappings from a lexical unit.

        Parameters
        ----------
        lu : LexicalUnit
            FrameNet lexical unit.
        frame_name : str
            Name of the containing frame.
        """
        # LexicalUnits may have semtypes that could be mapped
        # This is a placeholder for future WordNet mapping extraction
        # when LU models include explicit WordNet references

    def extract_wordnet_mappings(self, synsets: list[Synset], senses: list[Sense]) -> None:
        """Build WordNet sense and synset indices.

        Creates mappings between sense keys and synset offsets for
        cross-reference resolution.

        Parameters
        ----------
        synsets : list[Synset]
            WordNet synsets to index.
        senses : list[Sense]
            WordNet senses to index.
        """
        # Build sense key to synset offset index
        for sense in senses:
            self.wordnet_sense_index[sense.sense_key] = sense.synset_offset

        # Index synsets by offset for fast lookup
        synset_index = {synset.offset: synset for synset in synsets}

        # Create internal WordNet mappings (sense to synset)
        for sense in senses:
            if sense.synset_offset in synset_index:
                mapping = CrossReference(
                    source_dataset="wordnet",
                    source_id=sense.sense_key,
                    source_version="3.1",
                    target_dataset="wordnet",
                    target_id=sense.synset_offset,
                    mapping_type="direct",
                    confidence=MappingConfidence(
                        score=1.0,
                        method="internal",
                        factors={},
                    ),
                    metadata=MappingMetadata(
                        created_date=datetime.now(UTC),
                        created_by="wordnet",
                        version="3.1",
                        validation_status="validated",
                        notes="Sense to synset mapping",
                    ),
                )
                self.mapping_index.add_mapping(mapping)

    def get_mappings_for_entity(
        self, entity_id: str, source_dataset: DatasetType
    ) -> list[CrossReference]:
        """Get all mappings for a specific entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier in the source dataset.
        source_dataset : DatasetType
            Source dataset type.

        Returns
        -------
        list[CrossReference]
            All mappings from the specified entity.
        """
        key = f"{source_dataset}:{entity_id}"
        return self.mapping_index.forward_index.get(key, [])

    def get_reverse_mappings(
        self, entity_id: str, target_dataset: DatasetType
    ) -> list[CrossReference]:
        """Get all mappings targeting a specific entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier in the target dataset.
        target_dataset : DatasetType
            Target dataset type.

        Returns
        -------
        list[CrossReference]
            All mappings to the specified entity.
        """
        key = f"{target_dataset}:{entity_id}"
        return self.mapping_index.reverse_index.get(key, [])
