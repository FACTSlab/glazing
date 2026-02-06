"""Unified search interface for all linguistic datasets.

This module provides a unified interface for searching across
FrameNet, VerbNet, WordNet, and PropBank data simultaneously.
All datasets are loaded automatically when UnifiedSearch is initialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from glazing.framenet.loader import FrameNetLoader
from glazing.framenet.models import Frame
from glazing.framenet.search import FrameNetSearch
from glazing.framenet.symbol_parser import (
    filter_elements_by_properties,
    normalize_frame_name,
)
from glazing.initialize import get_default_data_path
from glazing.propbank.loader import PropBankLoader
from glazing.propbank.models import Frameset, Roleset
from glazing.propbank.search import PropBankSearch
from glazing.propbank.symbol_parser import filter_args_by_properties
from glazing.syntax.models import SyntaxElement, UnifiedSyntaxPattern
from glazing.syntax.parser import SyntaxParser
from glazing.types import ResourceType
from glazing.utils.fuzzy_match import levenshtein_ratio
from glazing.verbnet.loader import VerbNetLoader
from glazing.verbnet.models import SyntaxElement as VNSyntaxElement
from glazing.verbnet.models import VerbClass, VNFrame
from glazing.verbnet.search import VerbNetSearch
from glazing.verbnet.symbol_parser import filter_roles_by_properties
from glazing.verbnet.types import PredicateType
from glazing.wordnet.loader import WordNetLoader
from glazing.wordnet.models import Synset
from glazing.wordnet.search import WordNetSearch
from glazing.wordnet.symbol_parser import filter_by_relation_type


@dataclass
class SearchResult:
    """Individual search result.

    Parameters
    ----------
    dataset : str
        Source dataset name.
    id : str
        Entity identifier.
    type : str
        Entity type.
    name : str
        Entity name.
    description : str
        Entity description.
    score : float
        Relevance score.
    """

    dataset: str
    id: str
    type: str
    name: str
    description: str
    score: float


@dataclass
class UnifiedSearchResult:
    """Container for search results across all datasets.

    Parameters
    ----------
    frames : list[Frame]
        FrameNet frames found.
    verb_classes : list[VerbClass]
        VerbNet verb classes found.
    synsets : list[Synset]
        WordNet synsets found.
    framesets : list[Frameset]
        PropBank framesets found.
    rolesets : list[Roleset]
        PropBank rolesets found.

    Examples
    --------
    >>> result = UnifiedSearchResult(
    ...     frames=[giving_frame],
    ...     verb_classes=[give_class],
    ...     synsets=[give_synset],
    ...     framesets=[give_frameset],
    ...     rolesets=[]
    ... )
    """

    frames: list[Frame]
    verb_classes: list[VerbClass]
    synsets: list[Synset]
    framesets: list[Frameset]
    rolesets: list[Roleset]

    def is_empty(self) -> bool:
        """Check if all result lists are empty.

        Returns
        -------
        bool
            True if no results found in any dataset.
        """
        return not any(
            [
                self.frames,
                self.verb_classes,
                self.synsets,
                self.framesets,
                self.rolesets,
            ]
        )

    def count(self) -> int:
        """Get total count of all results.

        Returns
        -------
        int
            Total number of results across all datasets.
        """
        return (
            len(self.frames)
            + len(self.verb_classes)
            + len(self.synsets)
            + len(self.framesets)
            + len(self.rolesets)
        )


class UnifiedSearch:
    """Unified search interface across all linguistic datasets.

    Provides methods for searching FrameNet, VerbNet, WordNet,
    and PropBank simultaneously or individually.

    Parameters
    ----------
    framenet : FrameNetSearch | None
        FrameNet search index.
    verbnet : VerbNetSearch | None
        VerbNet search index.
    wordnet : WordNetSearch | None
        WordNet search index.
    propbank : PropBankSearch | None
        PropBank search index.

    Attributes
    ----------
    framenet : FrameNetSearch | None
        FrameNet search interface.
    verbnet : VerbNetSearch | None
        VerbNet search interface.
    wordnet : WordNetSearch | None
        WordNet search interface.
    propbank : PropBankSearch | None
        PropBank search interface.

    Methods
    -------
    by_lemma(lemma, pos)
        Search all datasets by lemma.
    by_semantic_role(role_name)
        Search for frames/classes with a semantic role.
    by_semantic_predicate(predicate)
        Search for verb classes with a semantic predicate.
    by_domain(domain)
        Search within a specific domain.
    get_statistics()
        Get statistics across all datasets.

    Examples
    --------
    >>> search = UnifiedSearch(
    ...     framenet=FrameNetSearch(frames),
    ...     verbnet=VerbNetSearch(classes),
    ...     wordnet=WordNetSearch(synsets),
    ...     propbank=PropBankSearch(framesets)
    ... )
    >>> results = search.by_lemma("give")
    """

    def __init__(  # noqa: PLR0913
        self,
        data_dir: Path | str | None = None,
        framenet: FrameNetSearch | None = None,
        verbnet: VerbNetSearch | None = None,
        wordnet: WordNetSearch | None = None,
        propbank: PropBankSearch | None = None,
        auto_load: bool = True,
    ) -> None:
        """Initialize unified search.

        Parameters
        ----------
        data_dir : Path | str | None, optional
            Directory containing converted data files.
            If None, uses default path from environment.
        framenet : FrameNetSearch | None, optional
            Pre-initialized FrameNet search object.
        verbnet : VerbNetSearch | None, optional
            Pre-initialized VerbNet search object.
        wordnet : WordNetSearch | None, optional
            Pre-initialized WordNet search object.
        propbank : PropBankSearch | None, optional
            Pre-initialized PropBank search object.
        auto_load : bool, default=True
            If True and no search objects provided, automatically
            loads from data_dir.
        """
        # If auto_load and no search objects provided, load from default paths
        if auto_load and not any([framenet, verbnet, wordnet, propbank]):
            if data_dir is None:
                data_dir = get_default_data_path()
            data_dir = Path(data_dir)

            # Try to load each dataset if file exists
            if (data_dir / "framenet.jsonl").exists():
                fn_loader = FrameNetLoader()  # autoload=True by default
                framenet = FrameNetSearch(fn_loader.frames)
            if (data_dir / "verbnet.jsonl").exists():
                vn_loader = VerbNetLoader()  # autoload=True by default
                verbnet = VerbNetSearch(list(vn_loader.classes.values()))
            if (data_dir / "wordnet.jsonl").exists():
                wn_loader = WordNetLoader()  # autoload=True by default
                wordnet = WordNetSearch(list(wn_loader.synsets.values()))
            if (data_dir / "propbank.jsonl").exists():
                pb_loader = PropBankLoader()  # autoload=True by default
                propbank = PropBankSearch(list(pb_loader.framesets.values()))

        self.framenet = framenet
        self.verbnet = verbnet
        self.wordnet = wordnet
        self.propbank = propbank
        self._syntax_parser = SyntaxParser()

    def by_lemma(self, lemma: str, pos: str | None = None) -> UnifiedSearchResult:
        """Search all datasets by lemma.

        Parameters
        ----------
        lemma : str
            Lemma to search for.
        pos : str | None
            Part of speech constraint (format varies by dataset).

        Returns
        -------
        UnifiedSearchResult
            Results from all datasets.
        """
        frames = self._search_framenet_by_lemma(lemma, pos)
        verb_classes = self._search_verbnet_by_lemma(lemma)
        synsets = self._search_wordnet_by_lemma(lemma, pos)
        framesets, rolesets = self._search_propbank_by_lemma(lemma)

        return UnifiedSearchResult(
            frames=frames,
            verb_classes=verb_classes,
            synsets=synsets,
            framesets=framesets,
            rolesets=rolesets,
        )

    def batch_by_lemma(
        self, lemmas: list[str], pos: str | None = None
    ) -> dict[str, UnifiedSearchResult]:
        """Search all datasets for multiple lemmas.

        Parameters
        ----------
        lemmas : list[str]
            List of lemmas to search for.
        pos : str | None
            Part of speech constraint.

        Returns
        -------
        dict[str, UnifiedSearchResult]
            Results mapped by lemma.
        """
        results = {}
        for lemma in lemmas:
            results[lemma] = self.by_lemma(lemma, pos)
        return results

    def _search_framenet_by_lemma(self, lemma: str, pos: str | None) -> list[Frame]:
        """Search FrameNet by lemma.

        Parameters
        ----------
        lemma : str
            Lemma to search for.
        pos : str | None
            Part of speech constraint.

        Returns
        -------
        list
            FrameNet frames.
        """
        if not self.framenet:
            return []

        fn_pos = self._convert_pos_for_framenet(pos)
        return self.framenet.find_frames_by_lemma(lemma, fn_pos)  # type: ignore[arg-type]

    def _convert_pos_for_framenet(self, pos: str | None) -> str | None:
        """Convert POS tag for FrameNet.

        Parameters
        ----------
        pos : str | None
            Part of speech tag.

        Returns
        -------
        str | None
            FrameNet POS tag.
        """
        if not pos:
            return None

        pos_lower = pos.lower()
        pos_map = {
            "v": "V",
            "verb": "V",
            "n": "N",
            "noun": "N",
            "a": "A",
            "adj": "A",
            "adjective": "A",
        }
        return pos_map.get(pos_lower)

    def _search_verbnet_by_lemma(self, lemma: str) -> list[VerbClass]:
        """Search VerbNet by lemma.

        Parameters
        ----------
        lemma : str
            Lemma to search for.

        Returns
        -------
        list
            VerbNet classes.
        """
        if not self.verbnet:
            return []
        return self.verbnet.by_members([lemma])

    def _search_wordnet_by_lemma(self, lemma: str, pos: str | None) -> list[Synset]:
        """Search WordNet by lemma.

        Parameters
        ----------
        lemma : str
            Lemma to search for.
        pos : str | None
            Part of speech constraint.

        Returns
        -------
        list
            WordNet synsets.
        """
        if not self.wordnet:
            return []

        wn_pos = self._convert_pos_for_wordnet(pos)
        return self.wordnet.by_lemma(lemma, wn_pos)  # type: ignore[arg-type]

    def _convert_pos_for_wordnet(self, pos: str | None) -> str | None:
        """Convert POS tag for WordNet.

        Parameters
        ----------
        pos : str | None
            Part of speech tag.

        Returns
        -------
        str | None
            WordNet POS tag.
        """
        if not pos:
            return None

        pos_lower = pos.lower()
        pos_map = {
            "v": "v",
            "verb": "v",
            "n": "n",
            "noun": "n",
            "a": "a",
            "adj": "a",
            "adjective": "a",
            "s": "a",
            "r": "r",
            "adv": "r",
            "adverb": "r",
        }
        return pos_map.get(pos_lower)

    def _search_propbank_by_lemma(self, lemma: str) -> tuple[list[Frameset], list[Roleset]]:
        """Search PropBank by lemma.

        Parameters
        ----------
        lemma : str
            Lemma to search for.

        Returns
        -------
        tuple[list, list]
            PropBank framesets and rolesets.
        """
        if not self.propbank:
            return [], []

        frameset = self.propbank.by_lemma(lemma)
        if frameset:
            return [frameset], frameset.rolesets
        return [], []

    def by_semantic_role(self, role_name: str) -> UnifiedSearchResult:
        """Search for frames/classes with a semantic role.

        Parameters
        ----------
        role_name : str
            Name of semantic role (e.g., "Agent", "Theme").

        Returns
        -------
        UnifiedSearchResult
            Results from datasets that have this role.
        """
        frames = []
        verb_classes = []

        # Search FrameNet for frames with this FE
        if self.framenet:
            frames = self.framenet.find_frames_with_fe(role_name)

        # Search VerbNet for classes with this thematic role
        if self.verbnet:
            # Cast role_name to ThematicRoleType if it matches a valid role
            try:
                verb_classes = self.verbnet.by_themroles([role_name])  # type: ignore[list-item]
            except (ValueError, KeyError):
                verb_classes = []

        # PropBank uses numbered arguments, not named roles
        # WordNet doesn't have semantic roles

        return UnifiedSearchResult(
            frames=frames,
            verb_classes=verb_classes,
            synsets=[],
            framesets=[],
            rolesets=[],
        )

    def by_semantic_predicate(self, predicate: PredicateType) -> UnifiedSearchResult:
        """Search for verb classes with a semantic predicate.

        Parameters
        ----------
        predicate : PredicateType
            Semantic predicate to search for.

        Returns
        -------
        UnifiedSearchResult
            Results from VerbNet.
        """
        verb_classes = []

        # Only VerbNet has semantic predicates
        if self.verbnet:
            verb_classes = self.verbnet.by_predicate(predicate)

        return UnifiedSearchResult(
            frames=[],
            verb_classes=verb_classes,
            synsets=[],
            framesets=[],
            rolesets=[],
        )

    def by_domain(self, domain: str) -> UnifiedSearchResult:
        """Search within a specific domain.

        Parameters
        ----------
        domain : str
            Domain name (WordNet lexical file name).

        Returns
        -------
        UnifiedSearchResult
            Results from datasets that support domain search.
        """
        synsets = []

        # Only WordNet has explicit domains (lexical files)
        if self.wordnet:
            synsets = self.wordnet.by_domain(domain)  # type: ignore[arg-type]

        return UnifiedSearchResult(
            frames=[],
            verb_classes=[],
            synsets=synsets,
            framesets=[],
            rolesets=[],
        )

    def by_external_resource(
        self, resource_type: ResourceType, class_name: str | None = None
    ) -> UnifiedSearchResult:
        """Search for entries linked to external resources.

        Parameters
        ----------
        resource_type : ResourceType
            Type of resource (e.g., "VerbNet", "FrameNet").
        class_name : str | None
            Specific class/frame name to match.

        Returns
        -------
        UnifiedSearchResult
            Results from datasets with links to the resource.
        """
        rolesets = []

        # PropBank has external resource links
        if self.propbank:
            rolesets = self.propbank.by_resource(resource_type, class_name)

        return UnifiedSearchResult(
            frames=[],
            verb_classes=[],
            synsets=[],
            framesets=[],
            rolesets=rolesets,
        )

    def get_statistics(self) -> dict[str, dict[str, int]]:
        """Get statistics across all datasets.

        Returns
        -------
        dict[str, dict[str, int]]
            Statistics for each available dataset.
        """
        stats = {}

        if self.framenet:
            stats["framenet"] = self.framenet.get_statistics()

        if self.verbnet:
            stats["verbnet"] = self.verbnet.get_statistics()

        if self.wordnet:
            stats["wordnet"] = self.wordnet.get_statistics()

        if self.propbank:
            stats["propbank"] = self.propbank.get_statistics()

        return stats

    @classmethod
    def from_paths(
        cls,
        framenet_path: Path | str | None = None,
        verbnet_path: Path | str | None = None,
        wordnet_synsets_path: Path | str | None = None,
        wordnet_senses_path: Path | str | None = None,
        propbank_path: Path | str | None = None,
    ) -> UnifiedSearch:
        """Load unified search from JSON Lines files.

        Parameters
        ----------
        framenet_path : Path | str | None
            Path to FrameNet JSONL file.
        verbnet_path : Path | str | None
            Path to VerbNet JSONL file.
        wordnet_synsets_path : Path | str | None
            Path to WordNet synsets JSONL file.
        wordnet_senses_path : Path | str | None
            Path to WordNet senses JSONL file.
        propbank_path : Path | str | None
            Path to PropBank JSONL file.

        Returns
        -------
        UnifiedSearch
            Unified search with loaded datasets.
        """
        framenet = None
        if framenet_path:
            framenet = FrameNetSearch.from_jsonl_file(framenet_path)

        verbnet = None
        if verbnet_path:
            verbnet = VerbNetSearch.from_jsonl_file(verbnet_path)

        wordnet = None
        if wordnet_synsets_path or wordnet_senses_path:
            wordnet = WordNetSearch.from_jsonl_files(
                synsets_path=wordnet_synsets_path,
                senses_path=wordnet_senses_path,
            )

        propbank = None
        if propbank_path:
            propbank = PropBankSearch.from_jsonl_file(propbank_path)

        return cls(
            framenet=framenet,
            verbnet=verbnet,
            wordnet=wordnet,
            propbank=propbank,
        )

    def search(self, query: str) -> list[SearchResult]:
        """Search across all datasets with a text query.

        Parameters
        ----------
        query : str
            Search query text.

        Returns
        -------
        list[SearchResult]
            List of search results across all datasets.
        """
        results = []

        # Search each dataset and convert to SearchResult format
        if self.framenet:
            frames = self.framenet.find_frames_by_lemma(query)
            for frame in frames:
                results.append(
                    SearchResult(
                        dataset="framenet",
                        id=frame.name,
                        type="frame",
                        name=frame.name,
                        description=frame.definition.plain_text if frame.definition else "",
                        score=1.0,
                    )
                )

        if self.verbnet:
            classes = self.verbnet.by_members([query])
            for cls in classes:
                results.append(
                    SearchResult(
                        dataset="verbnet",
                        id=cls.id,
                        type="class",
                        name=cls.id,
                        description=f"VerbNet class with {len(cls.members)} members",
                        score=1.0,
                    )
                )

        if self.wordnet:
            synsets = self.wordnet.by_lemma(query)
            for synset in synsets:
                synset_id = f"{synset.offset:08d}{synset.ss_type}"
                results.append(
                    SearchResult(
                        dataset="wordnet",
                        id=synset_id,
                        type="synset",
                        name=synset_id,
                        description=synset.gloss or "",
                        score=1.0,
                    )
                )

        if self.propbank:
            frameset = self.propbank.by_lemma(query)
            if frameset:
                results.append(
                    SearchResult(
                        dataset="propbank",
                        id=frameset.predicate_lemma,
                        type="frameset",
                        name=frameset.predicate_lemma,
                        description=f"PropBank frameset with {len(frameset.rolesets)} rolesets",
                        score=1.0,
                    )
                )

        return results

    def get_entity(
        self, entity_id: str, dataset: str
    ) -> Frame | VerbClass | Synset | Frameset | None:
        """Get a specific entity from a dataset.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        dataset : str
            Dataset name.

        Returns
        -------
        Frame | VerbClass | Synset | Frameset | None
            The entity if found, None otherwise.
        """
        if dataset == "framenet" and self.framenet:
            return self.framenet.get_frame_by_name(entity_id)
        if dataset == "verbnet" and self.verbnet:
            # VerbNet searches by ID
            return self.verbnet.get_by_id(entity_id)
        if dataset == "wordnet" and self.wordnet:
            # WordNet ID format: offset+pos (e.g., "01234567n")
            return self.wordnet.get_synset_by_id(entity_id)
        if dataset == "propbank" and self.propbank:
            return self.propbank.by_lemma(entity_id)
        return None

    def search_semantic_roles(self, role_name: str) -> list[SearchResult]:
        """Search for semantic roles across datasets.

        Parameters
        ----------
        role_name : str
            Role name to search for.

        Returns
        -------
        list[SearchResult]
            List of search results for the role.
        """
        results = []

        # Search FrameNet for frame elements
        if self.framenet:
            frames = self.framenet.find_frames_with_fe(role_name)
            for frame in frames:
                results.append(
                    SearchResult(
                        dataset="framenet",
                        id=frame.name,
                        type="frame_element",
                        name=frame.name,
                        description=f"Frame with {role_name} element",
                        score=1.0,
                    )
                )

        # Search VerbNet for thematic roles
        if self.verbnet:
            # Search for classes with this thematic role
            # Cast role_name to ThematicRoleType if it matches a valid role
            try:
                classes = self.verbnet.by_themroles([role_name])  # type: ignore[list-item]
            except (ValueError, KeyError):
                classes = []
            for cls in classes:
                results.append(
                    SearchResult(
                        dataset="verbnet",
                        id=cls.id,
                        type="thematic_role",
                        name=cls.id,
                        description=f"Class with {role_name} role",
                        score=1.0,
                    )
                )

        return results

    def find_cross_references(
        self, entity_id: str, source: str, target: str
    ) -> list[dict[str, str | float]]:
        """Find cross-references between datasets.

        Parameters
        ----------
        entity_id : str
            Source entity identifier.
        source : str
            Source dataset name.
        target : str
            Target dataset name.

        Returns
        -------
        list[dict]
            List of cross-reference mappings.
        """
        mapping_strategies = {
            ("verbnet", "propbank"): self._verbnet_to_propbank_refs,
            ("propbank", "verbnet"): self._propbank_to_verbnet_refs,
            ("verbnet", "framenet"): self._verbnet_to_framenet_refs,
            ("framenet", "verbnet"): self._framenet_to_verbnet_refs,
            ("propbank", "framenet"): self._propbank_to_framenet_refs,
        }

        # Handle WordNet to other datasets
        if source == "wordnet" and target in ["verbnet", "propbank", "framenet"]:
            return self._wordnet_to_other_refs(entity_id, target)

        strategy = mapping_strategies.get((source, target))
        return strategy(entity_id) if strategy else []

    def _verbnet_to_propbank_refs(self, entity_id: str) -> list[dict[str, str | float]]:
        """Find VerbNet to PropBank references.

        Parameters
        ----------
        entity_id : str
            VerbNet class ID.

        Returns
        -------
        list[dict[str, str | float]]
            Reference mappings.
        """
        references: list[dict[str, str | float]] = []
        if not (self.verbnet and self.propbank):
            return references

        verb_class = self.verbnet.get_by_id(entity_id)
        if not verb_class:
            return references

        for frameset in self.propbank.get_all_framesets():
            for roleset in frameset.rolesets:
                if roleset.lexlinks:
                    for lexlink in roleset.lexlinks:
                        if lexlink.class_name == entity_id and lexlink.resource == "VerbNet":
                            references.append(
                                {
                                    "target_id": roleset.id,
                                    "mapping_type": "lexlink",
                                    "confidence": lexlink.confidence,
                                }
                            )
        return references

    def _propbank_to_verbnet_refs(self, entity_id: str) -> list[dict[str, str | float]]:
        """Find PropBank to VerbNet references.

        Parameters
        ----------
        entity_id : str
            PropBank roleset ID.

        Returns
        -------
        list[dict[str, str | float]]
            Reference mappings.
        """
        references: list[dict[str, str | float]] = []
        if not self.propbank:
            return references

        pb_frameset = self.propbank.by_lemma(entity_id.split(".", maxsplit=1)[0])
        if not pb_frameset:
            return references

        for roleset in pb_frameset.rolesets:
            if roleset.id == entity_id and roleset.lexlinks:
                for lexlink in roleset.lexlinks:
                    if lexlink.resource == "VerbNet":
                        references.append(
                            {
                                "target_id": lexlink.class_name,
                                "mapping_type": "lexlink",
                                "confidence": lexlink.confidence,
                            }
                        )
        return references

    def _verbnet_to_framenet_refs(self, entity_id: str) -> list[dict[str, str | float]]:  # noqa: C901, PLR0912
        """Find VerbNet to FrameNet references.

        Parameters
        ----------
        entity_id : str
            VerbNet class ID.

        Returns
        -------
        list[dict[str, str | float]]
            Reference mappings.
        """
        references: list[dict[str, str | float]] = []
        if not self.verbnet:
            return references

        verb_class = self.verbnet.get_by_id(entity_id)
        if not verb_class:
            return references

        # Extract FrameNet mappings from VerbNet members
        for member in verb_class.members:
            for fn_mapping in member.framenet_mappings:
                # Calculate confidence based on mapping metadata
                confidence = 1.0
                if hasattr(fn_mapping, "confidence") and fn_mapping.confidence is not None:
                    if hasattr(fn_mapping.confidence, "score"):
                        confidence = fn_mapping.confidence.score
                    elif isinstance(fn_mapping.confidence, int | float):
                        confidence = float(fn_mapping.confidence)

                references.append(
                    {
                        "target_id": fn_mapping.frame_name,
                        "mapping_type": "framenet_mapping",
                        "confidence": confidence,
                    }
                )

        # Also check subclasses
        for subclass in verb_class.subclasses:
            for member in subclass.members:
                for fn_mapping in member.framenet_mappings:
                    confidence = 1.0
                    if hasattr(fn_mapping, "confidence") and fn_mapping.confidence is not None:
                        if hasattr(fn_mapping.confidence, "score"):
                            confidence = fn_mapping.confidence.score
                        elif isinstance(fn_mapping.confidence, int | float):
                            confidence = float(fn_mapping.confidence)

                    references.append(
                        {
                            "target_id": fn_mapping.frame_name,
                            "mapping_type": "framenet_mapping",
                            "confidence": confidence,
                        }
                    )

        # Remove duplicates by target_id, keeping highest confidence
        unique_refs: dict[str, dict[str, str | float]] = {}
        for ref in references:
            target = str(ref["target_id"])  # Ensure it's a string
            ref_confidence = float(ref["confidence"]) if "confidence" in ref else 0.0
            existing_confidence = (
                float(unique_refs[target]["confidence"])
                if target in unique_refs and "confidence" in unique_refs[target]
                else 0.0
            )
            if target not in unique_refs or ref_confidence > existing_confidence:
                unique_refs[target] = ref

        return list(unique_refs.values())

    def _framenet_to_verbnet_refs(self, entity_id: str) -> list[dict[str, str | float]]:  # noqa: C901
        """Find FrameNet to VerbNet references.

        Parameters
        ----------
        entity_id : str
            FrameNet frame name.

        Returns
        -------
        list[dict[str, str | float]]
            Reference mappings.
        """
        references: list[dict[str, str | float]] = []
        if not (self.framenet and self.verbnet):
            return references

        frame = self.framenet.get_frame_by_name(entity_id)
        if not frame:
            return references

        # Search VerbNet classes for references to this frame
        # Use fuzzy matching on frame names
        normalized_frame = normalize_frame_name(entity_id)

        for verb_class in self.verbnet.get_all_classes():
            for member in verb_class.members:
                for fn_mapping in member.framenet_mappings:
                    normalized_mapping = normalize_frame_name(fn_mapping.frame_name)

                    # Use fuzzy matching to find potential matches
                    similarity = levenshtein_ratio(normalized_frame, normalized_mapping)

                    if similarity >= 0.8:  # Threshold for fuzzy matching
                        # Calculate confidence based on similarity and mapping confidence
                        base_confidence = 1.0
                        if hasattr(fn_mapping, "confidence") and fn_mapping.confidence is not None:
                            if hasattr(fn_mapping.confidence, "score"):
                                base_confidence = fn_mapping.confidence.score
                            elif isinstance(fn_mapping.confidence, int | float):
                                base_confidence = float(fn_mapping.confidence)

                        final_confidence = similarity * base_confidence

                        references.append(
                            {
                                "target_id": verb_class.id,
                                "mapping_type": "reverse_framenet",
                                "confidence": final_confidence,
                            }
                        )

        # Remove duplicates by target_id, keeping highest confidence
        unique_refs: dict[str, dict[str, str | float]] = {}
        for ref in references:
            target = str(ref["target_id"])  # Ensure it's a string
            ref_confidence = float(ref["confidence"]) if "confidence" in ref else 0.0
            existing_confidence = (
                float(unique_refs[target]["confidence"])
                if target in unique_refs and "confidence" in unique_refs[target]
                else 0.0
            )
            if target not in unique_refs or ref_confidence > existing_confidence:
                unique_refs[target] = ref

        return list(unique_refs.values())

    def _propbank_to_framenet_refs(self, entity_id: str) -> list[dict[str, str | float]]:
        """Find PropBank to FrameNet references.

        Parameters
        ----------
        entity_id : str
            PropBank roleset ID.

        Returns
        -------
        list[dict[str, str | float]]
            Reference mappings.
        """
        references: list[dict[str, str | float]] = []
        if not self.propbank:
            return references

        pb_frameset = self.propbank.by_lemma(entity_id.split(".", maxsplit=1)[0])
        if not pb_frameset:
            return references

        for roleset in pb_frameset.rolesets:
            if roleset.id == entity_id and roleset.usagenotes:
                for usage in roleset.usagenotes.usage:
                    if usage.resource == "FrameNet":
                        references.append(
                            {"target_id": usage.version, "mapping_type": "usage", "confidence": 0.8}
                        )
        return references

    def _wordnet_to_other_refs(self, entity_id: str, target: str) -> list[dict[str, str | float]]:
        """Find WordNet to other dataset references.

        Parameters
        ----------
        entity_id : str
            WordNet synset ID.
        target : str
            Target dataset name.

        Returns
        -------
        list[dict[str, str | float]]
            Reference mappings.
        """
        references: list[dict[str, str | float]] = []
        if not self.wordnet:
            return references

        synset = self.wordnet.get_synset_by_id(entity_id)
        if not synset:
            return references

        for word in synset.words:
            target_refs = self._find_target_refs_for_lemma(word.lemma, target)
            references.extend(target_refs)

        return references

    def _find_target_refs_for_lemma(self, lemma: str, target: str) -> list[dict[str, str | float]]:
        """Find target dataset references for a lemma.

        Parameters
        ----------
        lemma : str
            Lemma to search for.
        target : str
            Target dataset name.

        Returns
        -------
        list[dict[str, str | float]]
            Reference mappings.
        """
        references: list[dict[str, str | float]] = []

        if target == "verbnet" and self.verbnet:
            classes = self.verbnet.by_members([lemma])
            for cls in classes:
                references.append(
                    {"target_id": cls.id, "mapping_type": "lemma_match", "confidence": 0.7}
                )
        elif target == "propbank" and self.propbank:
            frameset = self.propbank.by_lemma(lemma)
            if frameset:
                for roleset in frameset.rolesets:
                    references.append(
                        {"target_id": roleset.id, "mapping_type": "lemma_match", "confidence": 0.7}
                    )
        elif target == "framenet" and self.framenet:
            frames = self.framenet.find_frames_by_lemma(lemma)
            for frame in frames:
                references.append(
                    {"target_id": frame.name, "mapping_type": "lemma_match", "confidence": 0.7}
                )

        return references

    def load_verbnet_from_jsonl(self, filepath: str) -> None:
        """Load VerbNet data from JSONL file."""
        verb_classes = []
        with Path(filepath).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    verb_classes.append(VerbClass.model_validate_json(line))
        self.verbnet = VerbNetSearch(verb_classes)

    def load_propbank_from_jsonl(self, filepath: str) -> None:
        """Load PropBank data from JSONL file."""
        framesets = []
        with Path(filepath).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    framesets.append(Frameset.model_validate_json(line))
        self.propbank = PropBankSearch(framesets)

    def load_wordnet_from_jsonl(self, synsets_path: str, _index_path: str, _pos: str) -> None:
        """Load WordNet data from JSONL files."""
        synsets = []
        with Path(synsets_path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    synsets.append(Synset.model_validate_json(line))

        # Initialize or merge with existing WordNet search
        if self.wordnet is None:
            self.wordnet = WordNetSearch(synsets)
        else:
            # Merge synsets with existing ones
            existing_synsets = self.wordnet.get_all_synsets()
            # Create a dict to merge by offset to avoid duplicates
            synset_dict = {f"{s.offset:08d}{s.ss_type}": s for s in existing_synsets}
            for synset in synsets:
                synset_id = f"{synset.offset:08d}{synset.ss_type}"
                synset_dict[synset_id] = synset
            # Recreate WordNetSearch with merged synsets
            self.wordnet = WordNetSearch(list(synset_dict.values()))

    def search_with_fuzzy(  # noqa: C901, PLR0912
        self, query: str, fuzzy_threshold: float = 0.8
    ) -> list[SearchResult]:
        """Search across all datasets with fuzzy matching.

        Parameters
        ----------
        query : str
            Search query text.
        fuzzy_threshold : float, default=0.8
            Minimum similarity score for fuzzy matches.

        Returns
        -------
        list[SearchResult]
            Search results with confidence scores.
        """
        results = []
        query_normalized = query.lower()

        # Search each dataset with fuzzy matching
        if self.framenet:
            for frame in self.framenet._frames_by_id.values():
                similarity = levenshtein_ratio(query_normalized, frame.name.lower())
                if similarity >= fuzzy_threshold:
                    results.append(
                        SearchResult(
                            dataset="framenet",
                            id=frame.name,
                            type="frame",
                            name=frame.name,
                            description=frame.definition.plain_text if frame.definition else "",
                            score=similarity,
                        )
                    )

        if self.verbnet:
            for cls in self.verbnet.get_all_classes():
                for member in cls.members:
                    similarity = levenshtein_ratio(query_normalized, member.name.lower())
                    if similarity >= fuzzy_threshold:
                        results.append(
                            SearchResult(
                                dataset="verbnet",
                                id=cls.id,
                                type="class",
                                name=cls.id,
                                description=f"VerbNet class with member {member.name}",
                                score=similarity,
                            )
                        )
                        break  # Only add class once

        if self.wordnet:
            for synset in self.wordnet.get_all_synsets():
                for word in synset.words:
                    similarity = levenshtein_ratio(query_normalized, word.lemma.lower())
                    if similarity >= fuzzy_threshold:
                        synset_id = f"{synset.offset:08d}{synset.ss_type}"
                        results.append(
                            SearchResult(
                                dataset="wordnet",
                                id=synset_id,
                                type="synset",
                                name=synset_id,
                                description=synset.gloss or "",
                                score=similarity,
                            )
                        )
                        break  # Only add synset once

        if self.propbank:
            for frameset in self.propbank.get_all_framesets():
                similarity = levenshtein_ratio(query_normalized, frameset.predicate_lemma.lower())
                if similarity >= fuzzy_threshold:
                    results.append(
                        SearchResult(
                            dataset="propbank",
                            id=frameset.predicate_lemma,
                            type="frameset",
                            name=frameset.predicate_lemma,
                            description=f"PropBank frameset with {len(frameset.rolesets)} rolesets",
                            score=similarity,
                        )
                    )

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def search_verbnet_roles(
        self,
        optional: bool | None = None,
        indexed: bool | None = None,
        verb_specific: bool | None = None,
    ) -> list[VerbClass]:
        """Search VerbNet classes by role properties.

        Parameters
        ----------
        optional : bool | None, optional
            Filter for optional roles.
        indexed : bool | None, optional
            Filter for indexed roles.
        verb_specific : bool | None, optional
            Filter for verb-specific roles.

        Returns
        -------
        list[VerbClass]
            VerbNet classes matching criteria.
        """
        if not self.verbnet:
            return []

        matching_classes = []
        for cls in self.verbnet.get_all_classes():
            filtered_roles = filter_roles_by_properties(
                cls.themroles,
                optional=optional,
                indexed=indexed,
                verb_specific=verb_specific,
            )
            if filtered_roles:
                matching_classes.append(cls)

        return matching_classes

    def search_propbank_args(
        self,
        arg_type: str | None = None,
        prefix: str | None = None,
        modifier: str | None = None,
        arg_number: str | None = None,
    ) -> list[Roleset]:
        """Search PropBank rolesets by argument properties.

        Parameters
        ----------
        arg_type : str | None, optional
            "core" or "modifier".
        prefix : str | None, optional
            "C" or "R" for continuation/reference.
        modifier : str | None, optional
            Modifier type (e.g., "LOC", "TMP").
        arg_number : str | None, optional
            Specific argument number (e.g., "0", "1", "2").

        Returns
        -------
        list[Roleset]
            PropBank rolesets matching criteria.
        """
        if not self.propbank:
            return []

        matching_rolesets = []
        for frameset in self.propbank.get_all_framesets():
            for roleset in frameset.rolesets:
                filtered_args = filter_args_by_properties(
                    roleset.roles,
                    is_core=(arg_type == "core") if arg_type else None,
                    modifier_type=modifier.lower() if modifier else None,  # type: ignore[arg-type]
                    has_prefix=True if prefix in ["C", "R"] else None,
                    arg_number=arg_number,
                )
                if filtered_args:
                    matching_rolesets.append(roleset)

        return matching_rolesets

    def search_wordnet_relations(self, relation_type: str | None = None) -> list[Synset]:
        """Search WordNet synsets by relation type.

        Parameters
        ----------
        relation_type : str | None, optional
            Relation type (e.g., "hypernym", "hyponym").

        Returns
        -------
        list[Synset]
            WordNet synsets with specified relations.
        """
        if not self.wordnet:
            return []

        matching_synsets = []
        for synset in self.wordnet.get_all_synsets():
            filtered_ptrs = filter_by_relation_type(synset.pointers, relation_type)
            if filtered_ptrs:
                matching_synsets.append(synset)

        return matching_synsets

    def search_framenet_elements(
        self, core_type: str | None = None, semantic_type: str | None = None
    ) -> list[Frame]:
        """Search FrameNet frames by element properties.

        Parameters
        ----------
        core_type : str | None, optional
            "Core", "Non-Core", or "Extra-Thematic".
        semantic_type : str | None, optional
            Semantic type of elements.

        Returns
        -------
        list[Frame]
            FrameNet frames matching criteria.
        """
        if not self.framenet:
            return []

        matching_frames = []
        for frame in self.framenet._frames_by_id.values():
            filtered_elements = filter_elements_by_properties(
                frame.frame_elements,
                core_type=core_type,  # type: ignore[arg-type]
            )
            # Additional filtering for semantic_type if needed
            if semantic_type and filtered_elements:
                filtered_elements = [
                    e
                    for e in filtered_elements
                    if hasattr(e, "semantic_type") and e.semantic_type == semantic_type
                ]
            if filtered_elements:
                matching_frames.append(frame)

        return matching_frames

    def search_by_syntax(
        self,
        pattern: str,
        dataset: str | None = None,
        allow_wildcards: bool = True,
        min_confidence: float = 0.7,
    ) -> list[SearchResult]:
        """Search by syntactic pattern with hierarchical matching.

        General patterns match specific instances with full confidence.

        Parameters
        ----------
        pattern : str
            Syntactic pattern with optional wildcards and roles.
            Examples:
            - "NP V NP" - basic transitive
            - "NP V PP" - matches all PP subtypes
            - "NP V PP.instrument" - specific PP role
            - "NP V NP *" - wildcard for fourth position

        dataset : str | None
            Limit to specific dataset (verbnet, propbank, framenet).

        allow_wildcards : bool
            Whether to process wildcard elements (*).

        min_confidence : float
            Minimum confidence score for matches (0.0-1.0).

        Returns
        -------
        list[SearchResult]
            Matching results sorted by confidence.

        Examples
        --------
        >>> search = UnifiedSearch()
        >>> # Find all PP patterns
        >>> results = search.search_by_syntax("NP V PP")
        >>> # Find specific PP role
        >>> results = search.search_by_syntax("NP V PP.instrument")
        >>> # Use wildcards
        >>> results = search.search_by_syntax("NP V NP *")
        """
        query_pattern = self._syntax_parser.parse(pattern)
        results: list[SearchResult] = []

        self._search_verbnet_syntax(
            results, query_pattern, dataset, allow_wildcards, min_confidence
        )
        self._search_propbank_syntax(results, pattern, dataset)
        self._search_framenet_syntax(results, pattern, dataset)
        self._search_wordnet_syntax(results, pattern, dataset)

        # Sort by confidence score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _search_verbnet_syntax(
        self,
        results: list[SearchResult],
        query_pattern: UnifiedSyntaxPattern,
        dataset: str | None,
        allow_wildcards: bool,
        min_confidence: float,
    ) -> None:
        """Search VerbNet for syntactic patterns."""
        if not self.verbnet or (dataset and dataset != "verbnet"):
            return

        for verb_class in self.verbnet.get_all_classes():
            for frame in verb_class.frames:
                target_pattern = self._extract_verbnet_pattern(frame)
                matches, confidence = query_pattern.matches_hierarchically(
                    target_pattern, allow_wildcards=allow_wildcards
                )

                if matches and confidence >= min_confidence:
                    desc = self._get_verbnet_frame_description(frame, target_pattern)
                    results.append(
                        SearchResult(
                            dataset="verbnet",
                            id=verb_class.id,
                            type="syntactic_frame",
                            name=verb_class.id,
                            description=f"Pattern: {desc}",
                            score=confidence,
                        )
                    )
                    break  # One match per class

    def _get_verbnet_frame_description(
        self, frame: VNFrame, target_pattern: UnifiedSyntaxPattern
    ) -> str:
        """Get description for VerbNet frame."""
        if frame.description and frame.description.primary:
            return frame.description.primary
        return target_pattern.source_pattern or target_pattern.normalized

    def _search_propbank_syntax(
        self, results: list[SearchResult], pattern: str, dataset: str | None
    ) -> None:
        """Search PropBank for syntactic patterns."""
        if not self.propbank or (dataset and dataset != "propbank"):
            return

        rolesets = self.propbank.by_syntax(pattern)
        for roleset in rolesets:
            results.append(
                SearchResult(
                    dataset="propbank",
                    id=roleset.id,
                    type="roleset",
                    name=roleset.id,
                    description=f"PropBank roleset: {roleset.name}",
                    score=1.0,
                )
            )

    def _search_framenet_syntax(
        self, results: list[SearchResult], pattern: str, dataset: str | None
    ) -> None:
        """Search FrameNet for syntactic patterns."""
        if not self.framenet or (dataset and dataset != "framenet"):
            return

        fn_frames = self.framenet.by_syntax(pattern)
        for fn_frame in fn_frames:
            description = fn_frame.definition.plain_text if fn_frame.definition else fn_frame.name
            results.append(
                SearchResult(
                    dataset="framenet",
                    id=str(fn_frame.id),
                    type="frame",
                    name=fn_frame.name,
                    description=f"FrameNet frame: {description}",
                    score=1.0,
                )
            )

    def _search_wordnet_syntax(
        self, results: list[SearchResult], pattern: str, dataset: str | None
    ) -> None:
        """Search WordNet for syntactic patterns."""
        if not self.wordnet or (dataset and dataset != "wordnet"):
            return

        synsets = self.wordnet.by_syntax(pattern)
        for synset in synsets:
            results.append(
                SearchResult(
                    dataset="wordnet",
                    id=str(synset.offset),
                    type="synset",
                    name=str(synset.offset),
                    description=f"WordNet synset: {synset.gloss}",
                    score=1.0,
                )
            )

    def _extract_verbnet_pattern(self, frame: VNFrame) -> UnifiedSyntaxPattern:
        """Extract syntactic pattern from VerbNet frame with morphological features."""
        elements = []
        skip_next = False

        for i, elem in enumerate(frame.syntax.elements):
            if skip_next:
                skip_next = False
                continue

            element, should_skip = self._process_verbnet_element(elem, frame.syntax.elements, i)
            if element:
                elements.append(element)
            skip_next = should_skip

        source = self._get_verbnet_source_pattern(frame)

        # Extract morphological features from synrestrs
        synrestrs = []
        for elem in frame.syntax.elements:
            if hasattr(elem, "synrestrs") and elem.synrestrs:
                synrestrs.extend(
                    [{"type": str(sr.type), "value": sr.value} for sr in elem.synrestrs]
                )

        # Use the new class method to handle synrestrs
        return UnifiedSyntaxPattern.from_verbnet_synrestrs(
            elements=elements, synrestrs=synrestrs, source_pattern=source
        )

    def _process_verbnet_element(
        self, elem: VNSyntaxElement, all_elements: list[VNSyntaxElement], index: int
    ) -> tuple[SyntaxElement | None, bool]:
        """Process a single VerbNet syntax element."""
        if elem.pos == "PREP":
            return self._create_pp_element(elem, all_elements, index)
        if elem.pos == "NP":
            return self._create_np_element(elem), False
        return self._create_other_element(elem), False

    def _create_pp_element(
        self, elem: VNSyntaxElement, all_elements: list[VNSyntaxElement], index: int
    ) -> tuple[SyntaxElement, bool]:
        """Create PP element with head and optional semantic role."""
        pp_elem = SyntaxElement(constituent="PP")

        # Add head value (specific preposition)
        if elem.value:
            pp_elem.head = elem.value.lower()

        # Check next element for semantic role
        skip_next = False
        if index + 1 < len(all_elements):
            next_elem = all_elements[index + 1]
            if next_elem.pos == "NP" and next_elem.value:
                pp_elem.semantic_role = next_elem.value
                skip_next = True

        return pp_elem, skip_next

    def _create_np_element(self, elem: VNSyntaxElement) -> SyntaxElement:
        """Create NP element with optional semantic role."""
        np_elem = SyntaxElement(constituent="NP")
        if elem.value:
            np_elem.semantic_role = elem.value
        return np_elem

    def _create_other_element(self, elem: VNSyntaxElement) -> SyntaxElement | None:
        """Create element for other constituent types.

        Maps VerbNet constituents to base constituents.
        LEX elements represent specific lexical items (e.g., 'there' in 'There V NP')
        and are skipped as they are not syntactic constituents.

        Raises
        ------
        ValueError
            If an unknown VerbNet constituent type is encountered.
        """
        const = elem.pos

        # LEX represents specific lexical items, not syntactic constituents
        if const == "LEX":
            return None

        # Map VerbNet constituents to base constituents
        const_mapping = {
            "VERB": "VERB",
            "V": "VERB",
            "ADV": "ADV",
            "ADVP": "ADVP",
            "ADJ": "ADJ",
            "S": "S",
            "SBAR": "SBAR",
        }

        if const not in const_mapping:
            msg = f"Unknown VerbNet constituent type: '{const}'"
            raise ValueError(msg)

        return SyntaxElement(constituent=const_mapping[const])  # type: ignore[arg-type]

    def _get_verbnet_source_pattern(self, frame: VNFrame) -> str:
        """Get source pattern description for VerbNet frame."""
        if frame.description and frame.description.primary:
            return frame.description.primary
        return ""

    def load_framenet_from_jsonl(self, filepath: str) -> None:
        """Load FrameNet data from JSONL file."""
        frames = []
        with Path(filepath).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    frames.append(Frame.model_validate_json(line))
        self.framenet = FrameNetSearch(frames)
