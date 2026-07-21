"""WordNet relation traversal functionality.

This module provides relation traversal capabilities for WordNet,
including hypernym/hyponym chains, meronym/holonym navigation, entailment and
causation relations, and similarity measure calculations.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from glazing.wordnet.models import Synset
from glazing.wordnet.types import SynsetKey, normalize_pos

if TYPE_CHECKING:
    from glazing.wordnet.models import Pointer


class WordNetRelationTraverser:
    """Traverser for WordNet semantic and lexical relations.

    Provides methods for navigating WordNet's relation graph, including
    hypernym/hyponym traversal, meronym/holonym relations, and calculating
    semantic similarity measures.

    Parameters
    ----------
    synsets : dict[SynsetKey, Synset]
        Mapping from canonical synset key to synset object.

    Attributes
    ----------
    _synsets : dict[SynsetKey, Synset]
        Internal synset storage.

    Methods
    -------
    get_hypernyms(synset, direct_only)
        Get hypernyms (is-a relations) of a synset.
    get_hyponyms(synset, direct_only)
        Get hyponyms (inverse of hypernym) of a synset.
    get_hypernym_paths(synset, max_depth)
        Get all paths to root hypernyms.
    get_common_hypernyms(synset1, synset2)
        Find common hypernyms of two synsets.
    get_meronyms(synset, meronym_type)
        Get meronyms (part-of relations) of a synset.
    get_holonyms(synset, holonym_type)
        Get holonyms (has-part relations) of a synset.
    get_entailments(synset)
        Get entailments (verb relations) of a synset.
    get_causes(synset)
        Get causes (verb relations) of a synset.
    get_similar_to(synset)
        Get similar adjectives for an adjective synset.
    get_also_see(synset)
        Get also-see relations for a synset.
    get_antonyms(synset, lemma)
        Get antonyms for a synset or specific lemma.
    get_derivations(synset, lemma)
        Get derivationally related forms.
    calculate_path_similarity(synset1, synset2)
        Calculate path-based similarity between synsets.
    calculate_depth(synset)
        Calculate depth of synset in hypernym hierarchy.

    Examples
    --------
    >>> traverser = WordNetRelationTraverser(synsets)
    >>> hypernyms = traverser.get_hypernyms(dog_synset)
    >>> paths = traverser.get_hypernym_paths(dog_synset, max_depth=5)
    >>> similarity = traverser.calculate_path_similarity(dog_synset, cat_synset)
    """

    def __init__(self, synsets: dict[SynsetKey, Synset]) -> None:
        """Initialize relation traverser with synset data."""
        self._synsets = synsets

    def get_hypernyms(self, synset: Synset, direct_only: bool = True) -> list[Synset]:
        """Get hypernyms (is-a relations) of a synset.

        Parameters
        ----------
        synset : Synset
            Synset to get hypernyms for.
        direct_only : bool
            If True, return only direct hypernyms.
            If False, return all hypernyms up to root.

        Returns
        -------
        list[Synset]
            Hypernym synsets.
        """
        if direct_only:
            hypernyms = []
            for pointer in synset.pointers:
                if pointer.symbol == "@" and pointer.is_semantic():
                    hypernym = self._synsets.get(pointer.target_key)
                    if hypernym:
                        hypernyms.append(hypernym)
            return hypernyms
        # Get all hypernyms recursively
        all_hypernym_keys = set()
        queue = deque([synset])
        visited = {synset.key}

        while queue:
            current = queue.popleft()
            for pointer in current.pointers:
                if (
                    pointer.symbol == "@"
                    and pointer.is_semantic()
                    and pointer.target_key not in visited
                    and (hypernym := self._synsets.get(pointer.target_key))
                ):
                    all_hypernym_keys.add(hypernym.key)
                    queue.append(hypernym)
                    visited.add(pointer.target_key)

        hypernyms = [self._synsets[key] for key in all_hypernym_keys]
        return sorted(hypernyms, key=lambda s: s.key)

    def get_hyponyms(self, synset: Synset, direct_only: bool = True) -> list[Synset]:
        """Get hyponyms (inverse of hypernym) of a synset.

        Parameters
        ----------
        synset : Synset
            Synset to get hyponyms for.
        direct_only : bool
            If True, return only direct hyponyms.
            If False, return all hyponyms recursively.

        Returns
        -------
        list[Synset]
            Hyponym synsets.
        """
        if direct_only:
            hyponyms = []
            for pointer in synset.pointers:
                if pointer.symbol == "~" and pointer.is_semantic():
                    hyponym = self._synsets.get(pointer.target_key)
                    if hyponym:
                        hyponyms.append(hyponym)
            return hyponyms
        # Get all hyponyms recursively
        all_hyponym_keys = set()
        queue = deque([synset])
        visited = {synset.key}

        while queue:
            current = queue.popleft()
            for pointer in current.pointers:
                if (
                    pointer.symbol == "~"
                    and pointer.is_semantic()
                    and pointer.target_key not in visited
                    and (hyponym := self._synsets.get(pointer.target_key))
                ):
                    all_hyponym_keys.add(hyponym.key)
                    queue.append(hyponym)
                    visited.add(pointer.target_key)

        hyponyms = [self._synsets[key] for key in all_hyponym_keys]
        return sorted(hyponyms, key=lambda s: s.key)

    def get_hypernym_paths(self, synset: Synset, max_depth: int = 10) -> list[list[Synset]]:
        """Get all paths to root hypernyms.

        Parameters
        ----------
        synset : Synset
            Starting synset.
        max_depth : int
            Maximum depth to traverse.

        Returns
        -------
        list[list[Synset]]
            List of paths, each path is a list of synsets from start to root.
        """
        paths = []

        def traverse(current: Synset, path: list[Synset], depth: int) -> None:
            """Recursively traverse hypernym paths."""
            if depth >= max_depth:
                paths.append(path)
                return

            hypernyms = self.get_hypernyms(current, direct_only=True)
            if not hypernyms:
                # Reached a root
                paths.append(path)
            else:
                for hypernym in hypernyms:
                    # Avoid cycles
                    if hypernym.key not in {s.key for s in path}:
                        traverse(hypernym, [*path, hypernym], depth + 1)

        traverse(synset, [synset], 0)
        return paths

    def get_common_hypernyms(self, synset1: Synset, synset2: Synset) -> list[Synset]:
        """Find common hypernyms of two synsets.

        Parameters
        ----------
        synset1 : Synset
            First synset.
        synset2 : Synset
            Second synset.

        Returns
        -------
        list[Synset]
            Common hypernym synsets.
        """
        hypernyms1 = self.get_hypernyms(synset1, direct_only=False)
        hypernyms1_keys = {h.key for h in hypernyms1}
        hypernyms1_keys.add(synset1.key)  # Include the synset itself

        hypernyms2 = self.get_hypernyms(synset2, direct_only=False)
        hypernyms2_keys = {h.key for h in hypernyms2}
        hypernyms2_keys.add(synset2.key)  # Include the synset itself

        common_keys = hypernyms1_keys & hypernyms2_keys
        common = [self._synsets[key] for key in common_keys if key in self._synsets]
        return sorted(common, key=lambda s: s.key)

    def get_meronyms(self, synset: Synset, meronym_type: str | None = None) -> list[Synset]:
        """Get meronyms (part-of relations) of a synset.

        Parameters
        ----------
        synset : Synset
            Synset to get meronyms for.
        meronym_type : str | None
            Type of meronym: "member", "substance", "part", or None for all.

        Returns
        -------
        list[Synset]
            Meronym synsets.
        """
        meronyms = []

        # Map meronym types to pointer symbols
        symbol_map = {"member": "%m", "substance": "%s", "part": "%p"}

        symbols = [symbol_map.get(meronym_type, "")] if meronym_type else ["%m", "%s", "%p"]

        for pointer in synset.pointers:
            if pointer.symbol in symbols and pointer.is_semantic():
                meronym = self._synsets.get(pointer.target_key)
                if meronym:
                    meronyms.append(meronym)

        return meronyms

    def get_holonyms(self, synset: Synset, holonym_type: str | None = None) -> list[Synset]:
        """Get holonyms (has-part relations) of a synset.

        Parameters
        ----------
        synset : Synset
            Synset to get holonyms for.
        holonym_type : str | None
            Type of holonym: "member", "substance", "part", or None for all.

        Returns
        -------
        list[Synset]
            Holonym synsets.
        """
        holonyms = []

        # Map holonym types to pointer symbols
        symbol_map = {"member": "#m", "substance": "#s", "part": "#p"}

        symbols = [symbol_map.get(holonym_type, "")] if holonym_type else ["#m", "#s", "#p"]

        for pointer in synset.pointers:
            if pointer.symbol in symbols and pointer.is_semantic():
                holonym = self._synsets.get(pointer.target_key)
                if holonym:
                    holonyms.append(holonym)

        return holonyms

    def get_entailments(self, synset: Synset) -> list[Synset]:
        """Get entailments (verb relations) of a synset.

        Parameters
        ----------
        synset : Synset
            Verb synset to get entailments for.

        Returns
        -------
        list[Synset]
            Entailed synsets.
        """
        entailments: list[Synset] = []

        if synset.ss_type != "v":
            return entailments

        for pointer in synset.pointers:
            if pointer.symbol == "*" and pointer.is_semantic():
                entailment = self._synsets.get(pointer.target_key)
                if entailment:
                    entailments.append(entailment)

        return entailments

    def get_causes(self, synset: Synset) -> list[Synset]:
        """Get causes (verb relations) of a synset.

        Parameters
        ----------
        synset : Synset
            Verb synset to get causes for.

        Returns
        -------
        list[Synset]
            Caused synsets.
        """
        causes: list[Synset] = []

        if synset.ss_type != "v":
            return causes

        for pointer in synset.pointers:
            if pointer.symbol == ">" and pointer.is_semantic():
                cause = self._synsets.get(pointer.target_key)
                if cause:
                    causes.append(cause)

        return causes

    def get_similar_to(self, synset: Synset) -> list[Synset]:
        """Get similar adjectives for an adjective synset.

        Parameters
        ----------
        synset : Synset
            Adjective synset.

        Returns
        -------
        list[Synset]
            Similar adjective synsets.
        """
        similar: list[Synset] = []

        if synset.ss_type not in ["a", "s"]:
            return similar

        for pointer in synset.pointers:
            if pointer.symbol == "&" and pointer.is_semantic():
                sim = self._synsets.get(pointer.target_key)
                if sim:
                    similar.append(sim)

        return similar

    def get_also_see(self, synset: Synset) -> list[Synset]:
        """Get also-see relations for a synset.

        Parameters
        ----------
        synset : Synset
            Synset to get also-see relations for.

        Returns
        -------
        list[Synset]
            Related synsets.
        """
        also_see = []

        for pointer in synset.pointers:
            if pointer.symbol == "^" and pointer.is_semantic():
                related = self._synsets.get(pointer.target_key)
                if related:
                    also_see.append(related)

        return also_see

    def get_antonyms(self, synset: Synset, lemma: str | None = None) -> list[tuple[Synset, str]]:
        """Get antonyms for a synset or specific lemma.

        Parameters
        ----------
        synset : Synset
            Synset to get antonyms for.
        lemma : str | None
            Specific lemma to get antonyms for.

        Returns
        -------
        list[tuple[Synset, str]]
            List of (antonym synset, antonym lemma) pairs.
        """
        antonyms = []

        for pointer in synset.pointers:
            if pointer.symbol == "!":
                antonym_pairs = self._extract_antonym_pairs(synset, pointer, lemma)
                antonyms.extend(antonym_pairs)

        return antonyms

    def _extract_antonym_pairs(
        self, synset: Synset, pointer: Pointer, lemma: str | None
    ) -> list[tuple[Synset, str]]:
        """Extract antonym pairs from a pointer.

        Parameters
        ----------
        synset : Synset
            Source synset.
        pointer : Pointer
            Antonym pointer.
        lemma : str | None
            Specific lemma filter.

        Returns
        -------
        list[tuple[Synset, str]]
            Antonym pairs.
        """
        ant_synset = self._synsets.get(pointer.target_key)
        if not ant_synset:
            return []

        if lemma:
            return self._get_lemma_specific_antonyms(synset, pointer, ant_synset, lemma)
        return self._get_all_antonyms(pointer, ant_synset)

    def _get_lemma_specific_antonyms(
        self, synset: Synset, pointer: Pointer, ant_synset: Synset, lemma: str
    ) -> list[tuple[Synset, str]]:
        """Get antonyms for a specific lemma.

        Parameters
        ----------
        synset : Synset
            Source synset.
        pointer : Pointer
            Antonym pointer.
        ant_synset : Synset
            Antonym synset.
        lemma : str
            Target lemma.

        Returns
        -------
        list[tuple[Synset, str]]
            Antonym pairs for the specific lemma.
        """
        word_idx = self._find_word_index(synset, lemma)
        if word_idx and pointer.source == word_idx and self._is_valid_target(pointer, ant_synset):
            ant_lemma = ant_synset.words[pointer.target - 1].lemma
            return [(ant_synset, ant_lemma)]
        return []

    def _get_all_antonyms(self, pointer: Pointer, ant_synset: Synset) -> list[tuple[Synset, str]]:
        """Get all antonyms from a synset.

        Parameters
        ----------
        pointer : Pointer
            Antonym pointer.
        ant_synset : Synset
            Antonym synset.

        Returns
        -------
        list[tuple[Synset, str]]
            All antonym pairs.
        """
        if self._is_valid_target(pointer, ant_synset):
            # Specific word antonym
            ant_lemma = ant_synset.words[pointer.target - 1].lemma
            return [(ant_synset, ant_lemma)]

        # Synset-level antonym
        return [(ant_synset, word.lemma) for word in ant_synset.words]

    def _find_word_index(self, synset: Synset, lemma: str) -> int | None:
        """Find the index of a word in a synset.

        Parameters
        ----------
        synset : Synset
            Synset to search.
        lemma : str
            Lemma to find.

        Returns
        -------
        int | None
            Word index (1-based) or None.
        """
        for i, word in enumerate(synset.words, 1):
            if word.lemma == lemma:
                return i
        return None

    def _is_valid_target(self, pointer: Pointer, target_synset: Synset) -> bool:
        """Check if pointer target is valid.

        Parameters
        ----------
        pointer : Pointer
            Pointer to check.
        target_synset : Synset
            Target synset.

        Returns
        -------
        bool
            True if target is valid.
        """
        return pointer.target > 0 and pointer.target <= len(target_synset.words)

    def get_derivations(self, synset: Synset, lemma: str | None = None) -> list[tuple[Synset, str]]:
        """Get derivationally related forms.

        Parameters
        ----------
        synset : Synset
            Synset to get derivations for.
        lemma : str | None
            Specific lemma to get derivations for.

        Returns
        -------
        list[tuple[Synset, str]]
            List of (related synset, related lemma) pairs.
        """
        derivations = []

        for pointer in synset.pointers:
            if pointer.symbol == "+":
                derivation_pairs = self._extract_derivation_pairs(synset, pointer, lemma)
                derivations.extend(derivation_pairs)

        return derivations

    def _extract_derivation_pairs(
        self, synset: Synset, pointer: Pointer, lemma: str | None
    ) -> list[tuple[Synset, str]]:
        """Extract derivation pairs from a pointer.

        Parameters
        ----------
        synset : Synset
            Source synset.
        pointer : Pointer
            Derivation pointer.
        lemma : str | None
            Specific lemma filter.

        Returns
        -------
        list[tuple[Synset, str]]
            Derivation pairs.
        """
        der_synset = self._synsets.get(pointer.target_key)
        if not der_synset:
            return []

        if lemma:
            return self._get_lemma_specific_derivations(synset, pointer, der_synset, lemma)
        return self._get_all_derivations(pointer, der_synset)

    def _get_lemma_specific_derivations(
        self, synset: Synset, pointer: Pointer, der_synset: Synset, lemma: str
    ) -> list[tuple[Synset, str]]:
        """Get derivations for a specific lemma.

        Parameters
        ----------
        synset : Synset
            Source synset.
        pointer : Pointer
            Derivation pointer.
        der_synset : Synset
            Derivation synset.
        lemma : str
            Target lemma.

        Returns
        -------
        list[tuple[Synset, str]]
            Derivation pairs for the specific lemma.
        """
        word_idx = self._find_word_index(synset, lemma)
        if word_idx and pointer.source == word_idx and self._is_valid_target(pointer, der_synset):
            der_lemma = der_synset.words[pointer.target - 1].lemma
            return [(der_synset, der_lemma)]
        return []

    def _get_all_derivations(
        self, pointer: Pointer, der_synset: Synset
    ) -> list[tuple[Synset, str]]:
        """Get all derivations from a synset.

        Parameters
        ----------
        pointer : Pointer
            Derivation pointer.
        der_synset : Synset
            Derivation synset.

        Returns
        -------
        list[tuple[Synset, str]]
            All derivation pairs.
        """
        if self._is_valid_target(pointer, der_synset):
            # Specific word derivation
            der_lemma = der_synset.words[pointer.target - 1].lemma
            return [(der_synset, der_lemma)]

        # Synset-level derivation
        return [(der_synset, word.lemma) for word in der_synset.words]

    def calculate_path_similarity(self, synset1: Synset, synset2: Synset) -> float:
        """Calculate path-based similarity between synsets.

        Parameters
        ----------
        synset1 : Synset
            First synset.
        synset2 : Synset
            Second synset.

        Returns
        -------
        float
            Similarity score between 0 and 1.
            Returns 0 if synsets are not connected.
        """
        # Must be same POS. Adjective heads and satellites count as one POS:
        # they share a hierarchy through similar-to pointers.
        if normalize_pos(synset1.ss_type) != normalize_pos(synset2.ss_type):
            return 0.0

        # Same synset has similarity 1
        if synset1.key == synset2.key:
            return 1.0

        # Find shortest path through common hypernyms
        common = self.get_common_hypernyms(synset1, synset2)
        if not common:
            return 0.0

        # Calculate shortest path
        min_distance = float("inf")

        for common_synset in common:
            # Distance from synset1 to common
            dist1 = self._calculate_min_distance(synset1, common_synset)
            # Distance from synset2 to common
            dist2 = self._calculate_min_distance(synset2, common_synset)

            if dist1 >= 0 and dist2 >= 0:
                total_dist = dist1 + dist2
                min_distance = min(min_distance, total_dist)

        if min_distance == float("inf"):
            return 0.0

        # Convert distance to similarity (1 / (distance + 1))
        return 1.0 / (min_distance + 1.0)

    def _calculate_min_distance(self, start: Synset, target: Synset) -> int:
        """Calculate minimum distance between synsets.

        Parameters
        ----------
        start : Synset
            Starting synset.
        target : Synset
            Target synset.

        Returns
        -------
        int
            Minimum distance, or -1 if not connected.
        """
        if start.key == target.key:
            return 0

        # BFS to find shortest path
        queue = deque([(start, 0)])
        visited = {start.key}

        while queue:
            current, distance = queue.popleft()

            # Check hypernyms
            for hypernym in self.get_hypernyms(current, direct_only=True):
                if hypernym.key == target.key:
                    return distance + 1

                if hypernym.key not in visited:
                    visited.add(hypernym.key)
                    queue.append((hypernym, distance + 1))

        return -1

    def calculate_depth(self, synset: Synset) -> int:
        """Calculate depth of synset in hypernym hierarchy.

        Parameters
        ----------
        synset : Synset
            Synset to calculate depth for.

        Returns
        -------
        int
            Maximum depth from root (0 for root synsets).
        """
        paths = self.get_hypernym_paths(synset)
        if not paths:
            return 0

        return max(len(path) - 1 for path in paths)

    def get_verb_groups(self, synset: Synset) -> list[Synset]:
        """Get verb group members for a verb synset.

        Parameters
        ----------
        synset : Synset
            Verb synset.

        Returns
        -------
        list[Synset]
            Related verb synsets in the same group.
        """
        groups: list[Synset] = []

        if synset.ss_type != "v":
            return groups

        for pointer in synset.pointers:
            if pointer.symbol == "$" and pointer.is_semantic():
                group = self._synsets.get(pointer.target_key)
                if group:
                    groups.append(group)

        return groups

    def get_all_relations(self, synset: Synset) -> dict[str, list[Synset]]:
        """Get all relations for a synset.

        Parameters
        ----------
        synset : Synset
            Synset to get relations for.

        Returns
        -------
        dict[str, list[Synset]]
            Dictionary mapping relation names to related synsets.
        """
        relations: dict[str, list[Synset]] = {}

        # Add general hierarchical relations
        self._add_hierarchical_relations(synset, relations)

        # Add part-whole relations
        self._add_meronymy_relations(synset, relations)

        # Add POS-specific relations
        self._add_pos_specific_relations(synset, relations)

        # Add general relations
        self._add_general_relations(synset, relations)

        return relations

    def _add_hierarchical_relations(
        self, synset: Synset, relations: dict[str, list[Synset]]
    ) -> None:
        """Add hypernym and hyponym relations.

        Parameters
        ----------
        synset : Synset
            Source synset.
        relations : dict[str, list[Synset]]
            Relations dictionary to update.
        """
        hypernyms = self.get_hypernyms(synset, direct_only=True)
        if hypernyms:
            relations["hypernyms"] = hypernyms

        hyponyms = self.get_hyponyms(synset, direct_only=True)
        if hyponyms:
            relations["hyponyms"] = hyponyms

    def _add_meronymy_relations(self, synset: Synset, relations: dict[str, list[Synset]]) -> None:
        """Add meronym and holonym relations.

        Parameters
        ----------
        synset : Synset
            Source synset.
        relations : dict[str, list[Synset]]
            Relations dictionary to update.
        """
        meronyms = self.get_meronyms(synset)
        if meronyms:
            relations["meronyms"] = meronyms

        holonyms = self.get_holonyms(synset)
        if holonyms:
            relations["holonyms"] = holonyms

    def _add_pos_specific_relations(
        self, synset: Synset, relations: dict[str, list[Synset]]
    ) -> None:
        """Add part-of-speech specific relations.

        Parameters
        ----------
        synset : Synset
            Source synset.
        relations : dict[str, list[Synset]]
            Relations dictionary to update.
        """
        if synset.ss_type == "v":
            self._add_verb_relations(synset, relations)
        elif synset.ss_type in ["a", "s"]:
            self._add_adjective_relations(synset, relations)

    def _add_verb_relations(self, synset: Synset, relations: dict[str, list[Synset]]) -> None:
        """Add verb-specific relations.

        Parameters
        ----------
        synset : Synset
            Source synset.
        relations : dict[str, list[Synset]]
            Relations dictionary to update.
        """
        entailments = self.get_entailments(synset)
        if entailments:
            relations["entailments"] = entailments

        causes = self.get_causes(synset)
        if causes:
            relations["causes"] = causes

        groups = self.get_verb_groups(synset)
        if groups:
            relations["verb_groups"] = groups

    def _add_adjective_relations(self, synset: Synset, relations: dict[str, list[Synset]]) -> None:
        """Add adjective-specific relations.

        Parameters
        ----------
        synset : Synset
            Source synset.
        relations : dict[str, list[Synset]]
            Relations dictionary to update.
        """
        similar = self.get_similar_to(synset)
        if similar:
            relations["similar_to"] = similar

    def _add_general_relations(self, synset: Synset, relations: dict[str, list[Synset]]) -> None:
        """Add general relations.

        Parameters
        ----------
        synset : Synset
            Source synset.
        relations : dict[str, list[Synset]]
            Relations dictionary to update.
        """
        also_see = self.get_also_see(synset)
        if also_see:
            relations["also_see"] = also_see
