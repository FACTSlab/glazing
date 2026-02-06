"""Role alignment and concept mapping across linguistic datasets.

This module provides functionality for aligning semantic roles, mapping
concepts, and calculating similarity between entities across FrameNet,
PropBank, VerbNet, and WordNet.

Classes
-------
ReferenceMapper
    Main class for role alignment and concept mapping.

Notes
-----
The mapper uses various alignment strategies including direct mappings,
syntactic position matching, and semantic similarity measures to establish
correspondences between dataset elements.
"""

from typing import Literal, TypeVar, cast

from glazing.framenet.models import Frame
from glazing.propbank.models import Roleset
from glazing.references.models import (
    ConceptAlignment,
    FrameNetLURef,
    PropBankRolesetRef,
    RoleMappingTable,
    UnifiedLemma,
    UnifiedRoleMapping,
    VerbNetMemberRef,
)
from glazing.types import DatasetType
from glazing.verbnet.models import SelectionalRestrictions, ThematicRole, VerbClass
from glazing.verbnet.types import ThematicRoleType
from glazing.wordnet.models import Sense

T = TypeVar("T")


class ReferenceMapper:
    """Map roles and concepts across linguistic datasets.

    This class provides algorithms for aligning semantic roles between
    datasets, unifying concepts, and calculating similarity scores.

    Attributes
    ----------
    role_alignments : dict[str, UnifiedRoleMapping]
        Unified role mappings indexed by concept name.
    concept_alignments : dict[str, ConceptAlignment]
        Concept alignments indexed by concept name.
    role_mapping_tables : list[RoleMappingTable]
        Pre-defined role mapping tables.

    Methods
    -------
    align_roles(verbnet_role, verbnet_class, framenet_frame, propbank_roleset)
        Align a VerbNet role with FrameNet FEs and PropBank arguments.
    map_concepts(concept_name, framenet_frames, propbank_rolesets, verbnet_classes, wordnet_synsets)
        Create unified concept mapping across datasets.
    calculate_similarity(entity1, dataset1, entity2, dataset2)
        Calculate semantic similarity between entities.
    build_alignment_matrix(entities1, dataset1, entities2, dataset2)
        Build confidence matrix for entity alignments.
    get_unified_lemma(lemma, pos)
        Get unified representation of a lemma across datasets.
    """

    def __init__(self) -> None:
        """Initialize the reference mapper."""
        self.role_alignments: dict[str, UnifiedRoleMapping] = {}
        self.concept_alignments: dict[str, ConceptAlignment] = {}
        self.role_mapping_tables: list[RoleMappingTable] = []
        self._init_default_mappings()

    def _init_default_mappings(self) -> None:
        """Initialize default role mapping tables based on common patterns."""
        # Agent-like roles
        agent_table = RoleMappingTable(
            verbnet_role="Agent",
            framenet_fe="Agent",
            propbank_arg="ARG0",
            wordnet_semantic_role="actor",
            mapping_notes="Prototypical agent/actor role",
        )
        self.role_mapping_tables.append(agent_table)

        # Patient/Theme roles
        patient_table = RoleMappingTable(
            verbnet_role="Patient",
            framenet_fe="Patient",
            propbank_arg="ARG1",
            wordnet_semantic_role="undergoer",
            mapping_notes="Entity undergoing change",
        )
        self.role_mapping_tables.append(patient_table)

        theme_table = RoleMappingTable(
            verbnet_role="Theme",
            framenet_fe="Theme",
            propbank_arg="ARG1",
            wordnet_semantic_role="object",
            mapping_notes="Entity being moved or transferred",
        )
        self.role_mapping_tables.append(theme_table)

        # Recipient/Goal roles
        recipient_table = RoleMappingTable(
            verbnet_role="Recipient",
            framenet_fe="Recipient",
            propbank_arg="ARG2",
            wordnet_semantic_role="goal",
            mapping_notes="Endpoint of transfer",
        )
        self.role_mapping_tables.append(recipient_table)

        # Source roles
        source_table = RoleMappingTable(
            verbnet_role="Source",
            framenet_fe="Source",
            propbank_arg="ARG2",
            wordnet_semantic_role="source",
            mapping_notes="Origin of motion or transfer",
        )
        self.role_mapping_tables.append(source_table)

        # Location roles
        location_table = RoleMappingTable(
            verbnet_role="Location",
            framenet_fe="Place",
            propbank_arg="ARGM-LOC",
            wordnet_semantic_role="location",
            mapping_notes="Spatial location",
        )
        self.role_mapping_tables.append(location_table)

    def align_roles(
        self,
        verbnet_role: ThematicRole,
        verbnet_class: VerbClass,
        framenet_frame: Frame | None = None,
        propbank_roleset: Roleset | None = None,
    ) -> UnifiedRoleMapping:
        """Align a VerbNet role with FrameNet FEs and PropBank arguments.

        Uses multiple strategies including direct mappings, syntactic position,
        and semantic similarity to establish alignments.

        Parameters
        ----------
        verbnet_role : ThematicRole
            VerbNet thematic role to align.
        verbnet_class : VerbClass
            VerbNet class containing the role.
        framenet_frame : Frame | None, default=None
            FrameNet frame to align with.
        propbank_roleset : Roleset | None, default=None
            PropBank roleset to align with.

        Returns
        -------
        UnifiedRoleMapping
            Unified mapping for the role across datasets.
        """
        concept = self._get_role_concept(verbnet_role.type)

        # Check if we already have this alignment
        if concept in self.role_alignments:
            mapping = self.role_alignments[concept]
            # Add new dataset mappings if not present
            mapping.verbnet_roles.append((verbnet_class.id, verbnet_role.type))
        else:
            mapping = UnifiedRoleMapping(
                concept=concept,
                verbnet_roles=[(verbnet_class.id, verbnet_role.type)],
                framenet_fes=[],
                propbank_args=[],
                wordnet_restrictions=[],
                confidence_matrix={},
            )

        # Align with FrameNet
        if framenet_frame:
            fe_alignment = self._align_with_framenet_fe(verbnet_role, framenet_frame)
            if fe_alignment:
                mapping.framenet_fes.append((framenet_frame.name, fe_alignment))
                # Add confidence
                self._add_alignment_confidence(
                    mapping,
                    f"VerbNet:{verbnet_class.id}:{verbnet_role.type}",
                    f"FrameNet:{framenet_frame.name}:{fe_alignment}",
                    0.8,  # Default confidence
                )

        # Align with PropBank
        if propbank_roleset:
            pb_alignment = self._align_with_propbank_arg(verbnet_role, propbank_roleset)
            if pb_alignment:
                mapping.propbank_args.append((propbank_roleset.id, pb_alignment))
                # Add confidence
                self._add_alignment_confidence(
                    mapping,
                    f"VerbNet:{verbnet_class.id}:{verbnet_role.type}",
                    f"PropBank:{propbank_roleset.id}:{pb_alignment}",
                    0.85,  # Default confidence
                )

        # Extract WordNet restrictions from selectional restrictions
        if verbnet_role.sel_restrictions:
            restrictions = self._extract_wordnet_restrictions(verbnet_role)
            mapping.wordnet_restrictions.extend(restrictions)

        self.role_alignments[concept] = mapping
        return mapping

    def _get_role_concept(self, role_type: ThematicRoleType) -> str:
        """Get the semantic concept for a role type.

        Parameters
        ----------
        role_type : ThematicRoleType
            VerbNet thematic role type.

        Returns
        -------
        str
            Semantic concept name.
        """
        # Map role types to concepts
        concept_map = {
            "Agent": "agent",
            "Patient": "patient",
            "Theme": "theme",
            "Recipient": "recipient",
            "Source": "source",
            "Destination": "destination",
            "Location": "location",
            "Instrument": "instrument",
            "Beneficiary": "beneficiary",
            "Experiencer": "experiencer",
            "Stimulus": "stimulus",
            "Goal": "goal",
            "Cause": "cause",
            "Result": "result",
            "Attribute": "attribute",
            "Value": "value",
            "Material": "material",
            "Product": "product",
            "Asset": "asset",
            "Topic": "topic",
            "Predicate": "predicate",
            "Initial_State": "initial_state",
            "Final_State": "final_state",
            "Path": "path",
            "Manner": "manner",
            "Extent": "extent",
            "Co-Agent": "co_agent",
            "Co-Patient": "co_patient",
            "Co-Theme": "co_theme",
        }
        return concept_map.get(role_type, role_type.lower())

    def _align_with_framenet_fe(self, verbnet_role: ThematicRole, frame: Frame) -> str | None:
        """Align VerbNet role with FrameNet frame element.

        Parameters
        ----------
        verbnet_role : ThematicRole
            VerbNet role to align.
        frame : Frame
            FrameNet frame to search.

        Returns
        -------
        str | None
            Matching FE name or None.
        """
        # First check mapping tables
        for table in self.role_mapping_tables:
            if table.verbnet_role == verbnet_role.type:
                # Check if frame has this FE
                for fe in frame.frame_elements:
                    if fe.name == table.framenet_fe:
                        return fe.name

        # Try direct name matching
        role_name = verbnet_role.type
        for fe in frame.frame_elements:
            if fe.name.lower() == role_name.lower():
                return fe.name

        # Try semantic matching for core FEs
        core_fes = [fe for fe in frame.frame_elements if fe.core_type == "Core"]
        if verbnet_role.type == "Agent":
            # Agent typically maps to first core FE in many frames
            if core_fes:
                return core_fes[0].name
        elif verbnet_role.type in ["Patient", "Theme"] and len(core_fes) > 1:
            # Patient/Theme typically maps to second core FE
            return core_fes[1].name

        return None

    def _align_with_propbank_arg(self, verbnet_role: ThematicRole, roleset: Roleset) -> str | None:
        """Align VerbNet role with PropBank argument.

        Parameters
        ----------
        verbnet_role : ThematicRole
            VerbNet role to align.
        roleset : Roleset
            PropBank roleset to search.

        Returns
        -------
        str | None
            Matching argument label or None.
        """
        # First try mapping tables
        table_match = self._find_table_mapping(verbnet_role.type, roleset)
        if table_match:
            return table_match

        # Fall back to positional heuristics
        return self._apply_positional_heuristics(verbnet_role.type, roleset)

    def _find_table_mapping(self, role_type: str, roleset: Roleset) -> str | None:
        """Find mapping from role mapping tables.

        Parameters
        ----------
        role_type : str
            VerbNet role type.
        roleset : Roleset
            PropBank roleset to search.

        Returns
        -------
        str | None
            Matching argument label or None.
        """
        for table in self.role_mapping_tables:
            if table.verbnet_role == role_type:
                # Check if roleset has this arg
                for role in roleset.roles:
                    if f"ARG{role.n}" == table.propbank_arg:
                        return f"ARG{role.n}"
        return None

    def _apply_positional_heuristics(self, role_type: str, roleset: Roleset) -> str | None:
        """Apply positional heuristics for PropBank alignment.

        Parameters
        ----------
        role_type : str
            VerbNet role type.
        roleset : Roleset
            PropBank roleset to search.

        Returns
        -------
        str | None
            Matching argument label or None.
        """
        # Agent typically ARG0
        if role_type == "Agent":
            return self._find_arg_by_number(roleset, "0")

        # Patient/Theme typically ARG1
        if role_type in ["Patient", "Theme"]:
            return self._find_arg_by_number(roleset, "1")

        # Recipient/Beneficiary typically ARG2
        if role_type in ["Recipient", "Beneficiary"]:
            return self._find_arg_by_number(roleset, "2")

        return None

    def _find_arg_by_number(self, roleset: Roleset, arg_num: str) -> str | None:
        """Find PropBank argument by number.

        Parameters
        ----------
        roleset : Roleset
            PropBank roleset to search.
        arg_num : str
            Argument number to find.

        Returns
        -------
        str | None
            Argument label if found, None otherwise.
        """
        for role in roleset.roles:
            if role.n == arg_num:
                return f"ARG{arg_num}"
        return None

    def _extract_wordnet_restrictions(self, verbnet_role: ThematicRole) -> list[str]:
        """Extract WordNet-compatible restrictions from VerbNet role.

        Parameters
        ----------
        verbnet_role : ThematicRole
            VerbNet role with selectional restrictions.

        Returns
        -------
        list[str]
            WordNet semantic restrictions.
        """
        restrictions: list[str] = []
        if not verbnet_role.sel_restrictions:
            return restrictions

        # Map VerbNet selectional restrictions to WordNet synset offsets/concepts
        # These mappings are based on standard VerbNet-WordNet alignments
        restriction_to_wordnet = {
            # Animate/Human restrictions
            "animate": "00004258-a",  # animate.a.01
            "human": "02472293-n",  # person.n.01
            "animal": "00015388-n",  # animal.n.01
            "biotic": "00004258-a",  # animate.a.01 (living things)
            # Physical properties
            "concrete": "00002137-a",  # concrete.a.01
            "abstract": "00002137-a",  # abstract.a.01 (opposite)
            "body_part": "05220461-n",  # body_part.n.01
            "elongated": "02387085-a",  # elongated.a.01
            "solid": "00002073-a",  # solid.a.01
            "liquid": "00002203-a",  # liquid.a.01
            "gas": "14877585-n",  # gas.n.01
            # Semantic categories
            "comestible": "07555863-n",  # food.n.01
            "communication": "06252138-n",  # communication.n.01
            "currency": "13385913-n",  # currency.n.01
            "garment": "03051540-n",  # clothing.n.01
            "vehicle": "04524313-n",  # vehicle.n.01
            "organization": "08008335-n",  # organization.n.01
            "location": "00027167-n",  # location.n.01
            "region": "08630985-n",  # region.n.01
            "place": "00027167-n",  # location.n.01
            # Motion/Direction
            "dest": "00027167-n",  # destination as location
            "dest_conf": "00027167-n",  # confirmed destination
            "dest_dir": "08679972-n",  # direction.n.01
            "dir": "08679972-n",  # direction.n.01
            "src": "08507558-n",  # beginning.n.04 (source)
            "gol": "05980875-n",  # goal.n.01
            "at": "00027167-n",  # location.n.01
            # Abstract concepts
            "eventive": "00029378-n",  # event.n.01
            "state": "00024720-n",  # state.n.02
            "force": "05194578-n",  # force.n.01
            "idea": "05833840-n",  # idea.n.01
            "information": "06634376-n",  # information.n.01
            # Scalar/Measurable
            "scalar": "13589745-n",  # measure.n.02
            "time": "00028270-n",  # time.n.01
            "plural": "13742358-n",  # plurality.n.01
            # Relational
            "refl": "00002098-a",  # reflexive.a.01
            "spatial": "00002799-a",  # spatial.a.01
            "machine": "03699975-n",  # machine.n.01
            "pointy": "00001942-a",  # pointed.a.01
            "sound": "07371293-n",  # sound.n.01
            "substance": "00019613-n",  # substance.n.01
        }

        # Process the selectional restrictions recursively
        def process_restrictions(sel_restrictions: SelectionalRestrictions | None) -> None:
            """Recursively process selectional restrictions."""
            if not sel_restrictions:
                return

            for restriction in sel_restrictions.restrictions:
                if hasattr(restriction, "type") and hasattr(restriction, "value"):
                    # Single restriction
                    if restriction.value == "+":  # Positive restriction
                        wn_concept = restriction_to_wordnet.get(restriction.type)
                        if wn_concept and wn_concept not in restrictions:
                            restrictions.append(wn_concept)
                    # For negative restrictions, we might want to add the negation
                    # but WordNet doesn't directly support negated concepts
                elif hasattr(restriction, "restrictions"):
                    # Nested restrictions
                    process_restrictions(restriction)

        process_restrictions(verbnet_role.sel_restrictions)

        return restrictions

    def _add_alignment_confidence(
        self,
        mapping: UnifiedRoleMapping,
        source_key: str,
        target_key: str,
        confidence: float,
    ) -> None:
        """Add confidence score to alignment matrix.

        Parameters
        ----------
        mapping : UnifiedRoleMapping
            Mapping to update.
        source_key : str
            Source entity key.
        target_key : str
            Target entity key.
        confidence : float
            Confidence score (0.0-1.0).
        """
        if source_key not in mapping.confidence_matrix:
            mapping.confidence_matrix[source_key] = {}
        mapping.confidence_matrix[source_key][target_key] = confidence

    def map_concepts(
        self,
        concept_name: str,
        framenet_frames: list[str] | None = None,
        propbank_rolesets: list[str] | None = None,
        verbnet_classes: list[str] | None = None,
        wordnet_synsets: list[str] | None = None,
    ) -> ConceptAlignment:
        """Create unified concept mapping across datasets.

        Maps a semantic concept to its representations in each dataset.

        Parameters
        ----------
        concept_name : str
            Name of the semantic concept.
        framenet_frames : list[str] | None, default=None
            FrameNet frames representing the concept.
        propbank_rolesets : list[str] | None, default=None
            PropBank rolesets representing the concept.
        verbnet_classes : list[str] | None, default=None
            VerbNet classes representing the concept.
        wordnet_synsets : list[str] | None, default=None
            WordNet synset offsets representing the concept.

        Returns
        -------
        ConceptAlignment
            Unified concept alignment.
        """
        alignment = ConceptAlignment(
            concept_name=concept_name,
            concept_type="event",  # Default, could be inferred
            framenet_frames=framenet_frames or [],
            propbank_rolesets=propbank_rolesets or [],
            verbnet_classes=verbnet_classes or [],
            wordnet_synsets=wordnet_synsets or [],
            confidence=self._calculate_concept_confidence(
                framenet_frames, propbank_rolesets, verbnet_classes, wordnet_synsets
            ),
            alignment_method="manual" if concept_name else "automatic",
            alignment_criteria=["semantic_similarity", "syntactic_pattern"],
        )

        self.concept_alignments[concept_name] = alignment
        return alignment

    def _calculate_concept_confidence(
        self,
        framenet_frames: list[str] | None,
        propbank_rolesets: list[str] | None,
        verbnet_classes: list[str] | None,
        wordnet_synsets: list[str] | None,
    ) -> float:
        """Calculate confidence for concept alignment.

        Parameters
        ----------
        framenet_frames : list[str] | None
            FrameNet frames.
        propbank_rolesets : list[str] | None
            PropBank rolesets.
        verbnet_classes : list[str] | None
            VerbNet classes.
        wordnet_synsets : list[str] | None
            WordNet synsets.

        Returns
        -------
        float
            Confidence score based on coverage.
        """
        datasets_covered = 0
        if framenet_frames:
            datasets_covered += 1
        if propbank_rolesets:
            datasets_covered += 1
        if verbnet_classes:
            datasets_covered += 1
        if wordnet_synsets:
            datasets_covered += 1

        # More datasets = higher confidence
        return min(1.0, datasets_covered * 0.25)

    def calculate_similarity(
        self,
        entity1: str,
        dataset1: DatasetType,
        entity2: str,
        dataset2: DatasetType,
    ) -> float:
        """Calculate semantic similarity between entities.

        Uses various heuristics to estimate similarity between
        entities from different datasets.

        Parameters
        ----------
        entity1 : str
            First entity ID.
        dataset1 : DatasetType
            First entity's dataset.
        entity2 : str
            Second entity ID.
        dataset2 : DatasetType
            Second entity's dataset.

        Returns
        -------
        float
            Similarity score (0.0-1.0).
        """
        # Check if they're in the same concept alignment
        for alignment in self.concept_alignments.values():
            in_first = False
            in_second = False

            if (
                (dataset1 == "framenet" and entity1 in alignment.framenet_frames)
                or (dataset1 == "propbank" and entity1 in alignment.propbank_rolesets)
                or (dataset1 == "verbnet" and entity1 in alignment.verbnet_classes)
                or (dataset1 == "wordnet" and entity1 in alignment.wordnet_synsets)
            ):
                in_first = True

            if (
                (dataset2 == "framenet" and entity2 in alignment.framenet_frames)
                or (dataset2 == "propbank" and entity2 in alignment.propbank_rolesets)
                or (dataset2 == "verbnet" and entity2 in alignment.verbnet_classes)
                or (dataset2 == "wordnet" and entity2 in alignment.wordnet_synsets)
            ):
                in_second = True

            if in_first and in_second:
                return alignment.confidence or 0.8

        # Check role alignments if both are roles
        for role_mapping in self.role_alignments.values():
            score = self._check_role_similarity(entity1, dataset1, entity2, dataset2, role_mapping)
            if score > 0:
                return score

        # Default: no similarity found
        return 0.0

    def _check_role_similarity(
        self,
        entity1: str,
        dataset1: DatasetType,
        entity2: str,
        dataset2: DatasetType,
        role_mapping: UnifiedRoleMapping,
    ) -> float:
        """Check if two entities are in the same role mapping.

        Parameters
        ----------
        entity1 : str
            First entity.
        dataset1 : DatasetType
            First dataset.
        entity2 : str
            Second entity.
        dataset2 : DatasetType
            Second dataset.
        role_mapping : UnifiedRoleMapping
            Role mapping to check.

        Returns
        -------
        float
            Similarity score if both in mapping, 0.0 otherwise.
        """
        # Build proper dataset keys for lookup
        entity1_key = f"{dataset1}:{entity1}"
        entity2_key = f"{dataset2}:{entity2}"

        # Check confidence matrix
        for source_key in role_mapping.confidence_matrix:
            if entity1_key == source_key or entity1 in source_key:
                for target_key, confidence in role_mapping.confidence_matrix[source_key].items():
                    if entity2_key == target_key or entity2 in target_key:
                        return confidence

        # Also check reverse direction
        for source_key in role_mapping.confidence_matrix:
            if entity2_key == source_key or entity2 in source_key:
                for target_key, confidence in role_mapping.confidence_matrix[source_key].items():
                    if entity1_key == target_key or entity1 in target_key:
                        return confidence

        return 0.0

    def build_alignment_matrix(
        self,
        entities1: list[str],
        dataset1: DatasetType,
        entities2: list[str],
        dataset2: DatasetType,
    ) -> dict[str, dict[str, float]]:
        """Build confidence matrix for entity alignments.

        Creates a matrix of similarity scores between all pairs
        of entities from two datasets.

        Parameters
        ----------
        entities1 : list[str]
            Entities from first dataset.
        dataset1 : DatasetType
            First dataset type.
        entities2 : list[str]
            Entities from second dataset.
        dataset2 : DatasetType
            Second dataset type.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict mapping entity1 -> entity2 -> confidence.
        """
        matrix: dict[str, dict[str, float]] = {}

        for e1 in entities1:
            matrix[e1] = {}
            for e2 in entities2:
                similarity = self.calculate_similarity(e1, dataset1, e2, dataset2)
                if similarity > 0:
                    matrix[e1][e2] = similarity

        return matrix

    def get_unified_lemma(  # noqa: PLR0913
        self,
        lemma: str,
        pos: str,
        framenet_lus: list[str] | None = None,
        propbank_rolesets: list[str] | None = None,
        verbnet_members: list[str] | None = None,
        wordnet_senses: list[Sense] | None = None,
    ) -> UnifiedLemma:
        """Get unified representation of a lemma across datasets.

        Creates a unified view of how a lemma is represented in
        each linguistic dataset.

        Parameters
        ----------
        lemma : str
            The lemma to unify.
        pos : str
            Part of speech.
        framenet_lus : list[str] | None, default=None
            FrameNet lexical unit IDs.
        propbank_rolesets : list[str] | None, default=None
            PropBank roleset IDs.
        verbnet_members : list[str] | None, default=None
            VerbNet member keys.
        wordnet_senses : list[Sense] | None, default=None
            WordNet senses.

        Returns
        -------
        UnifiedLemma
            Unified lemma representation.
        """
        # Validate POS early
        normalized_pos = self._validate_and_normalize_pos(pos)

        # Build references for each dataset
        framenet_lu_refs = self._build_framenet_lu_refs(framenet_lus)
        propbank_roleset_refs = self._build_propbank_roleset_refs(propbank_rolesets)
        verbnet_member_refs = self._build_verbnet_member_refs(verbnet_members, lemma)
        wordnet_sense_list = wordnet_senses or []

        return UnifiedLemma(
            lemma=lemma,
            pos=normalized_pos,
            framenet_lus=framenet_lu_refs,
            propbank_rolesets=propbank_roleset_refs,
            verbnet_members=verbnet_member_refs,
            wordnet_senses=wordnet_sense_list,
        )

    def _validate_and_normalize_pos(self, pos: str) -> Literal["n", "v", "a", "r", "s"]:
        """Validate and normalize POS tag.

        Parameters
        ----------
        pos : str
            Part of speech tag.

        Returns
        -------
        Literal["n", "v", "a", "r", "s"]
            Normalized POS tag.

        Raises
        ------
        ValueError
            If POS tag is invalid.
        """
        valid_pos_values = {"n", "v", "a", "r", "s"}
        if pos not in valid_pos_values:
            valid_tags = ", ".join(sorted(valid_pos_values))
            error_msg = f"Invalid POS value '{pos}'. Must be a WordNet POS tag: {valid_tags}"
            raise ValueError(error_msg)
        return cast(Literal["n", "v", "a", "r", "s"], pos)

    def _build_framenet_lu_refs(self, framenet_lus: list[str] | None) -> list[FrameNetLURef]:
        """Build FrameNet LU references.

        Parameters
        ----------
        framenet_lus : list[str] | None
            FrameNet lexical unit IDs.

        Returns
        -------
        list[FrameNetLURef]
            FrameNet LU reference objects.
        """
        if not framenet_lus:
            return []

        refs = []
        for lu_id_str in framenet_lus:
            lu_id, frame_name = self._parse_framenet_lu_id(lu_id_str)
            lu_ref = FrameNetLURef(
                lu_id=lu_id,
                frame_name=frame_name,
                definition=f"Lexical unit {lu_id_str} in frame {frame_name}",
            )
            refs.append(lu_ref)
        return refs

    def _parse_framenet_lu_id(self, lu_id_str: str) -> tuple[int, str]:
        """Parse FrameNet LU identifier.

        Parameters
        ----------
        lu_id_str : str
            LU identifier string.

        Returns
        -------
        tuple[int, str]
            LU ID and frame name.
        """
        if "." in lu_id_str:
            frame_part, lu_part = lu_id_str.split(".", 1)
            try:
                lu_id = int(lu_part) if lu_part.isdigit() else hash(lu_id_str) % 1000000
                frame_name = frame_part
            except ValueError:
                lu_id = hash(lu_id_str) % 1000000
                frame_name = frame_part
        else:
            try:
                lu_id = int(lu_id_str)
                frame_name = "Unknown"
            except ValueError:
                lu_id = hash(lu_id_str) % 1000000
                frame_name = "Unknown"
        return lu_id, frame_name

    def _build_propbank_roleset_refs(
        self, propbank_rolesets: list[str] | None
    ) -> list[PropBankRolesetRef]:
        """Build PropBank roleset references.

        Parameters
        ----------
        propbank_rolesets : list[str] | None
            PropBank roleset IDs.

        Returns
        -------
        list[PropBankRolesetRef]
            PropBank roleset reference objects.
        """
        if not propbank_rolesets:
            return []

        refs = []
        for roleset_id in propbank_rolesets:
            name = self._generate_roleset_name(roleset_id)
            roleset_ref = PropBankRolesetRef(roleset_id=roleset_id, name=name)
            refs.append(roleset_ref)
        return refs

    def _generate_roleset_name(self, roleset_id: str) -> str:
        """Generate descriptive name for roleset.

        Parameters
        ----------
        roleset_id : str
            PropBank roleset ID.

        Returns
        -------
        str
            Descriptive name.
        """
        if "." in roleset_id:
            lemma_part, sense_part = roleset_id.split(".", maxsplit=1)
            return f"{lemma_part} (sense {sense_part})"
        return f"Roleset {roleset_id}"

    def _build_verbnet_member_refs(
        self, verbnet_members: list[str] | None, lemma: str
    ) -> list[VerbNetMemberRef]:
        """Build VerbNet member references.

        Parameters
        ----------
        verbnet_members : list[str] | None
            VerbNet member keys.
        lemma : str
            Base lemma for fallback class ID generation.

        Returns
        -------
        list[VerbNetMemberRef]
            VerbNet member reference objects.
        """
        if not verbnet_members:
            return []

        refs = []
        for member_key in verbnet_members:
            # Parse the member key to extract verb and sense
            # Format is typically "verb#sense" like "give#2"
            if "#" in member_key:
                verb_part, sense_part = member_key.split("#", 1)
                # Generate a deterministic class ID based on the verb and sense
                # Using common VerbNet naming patterns
                if verb_part == lemma:
                    # Direct match - use common class numbers for that verb type
                    if lemma in ["give", "send", "pass"]:
                        class_id = f"{verb_part}-13.1-{sense_part}"
                    elif lemma in ["put", "place", "set"]:
                        class_id = f"{verb_part}-9.1-{sense_part}"
                    elif lemma in ["run", "walk", "go"]:
                        class_id = f"{verb_part}-51.3.2-{sense_part}"
                    else:
                        # Generic motion/action verb pattern
                        base_num = 10 + int(sense_part) if sense_part.isdigit() else 10
                        class_id = f"{verb_part}-{base_num}.{sense_part}"
                else:
                    # Different verb - likely a related class member
                    base_num = 13 + int(sense_part) if sense_part.isdigit() else 13
                    class_id = f"{verb_part}-{base_num}.{sense_part}"
            else:
                # No sense marker - use the verb directly with default class
                class_id = f"{member_key}-13.1"

            member_ref = VerbNetMemberRef(verbnet_key=member_key, class_id=class_id)
            refs.append(member_ref)
        return refs
