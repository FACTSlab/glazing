"""FrameNet XML to JSON Lines converter.

This module provides conversion from FrameNet XML format to JSON Lines format
using the glazing FrameNet models. Supports both full frame files and lu files.

Classes
-------
FrameNetConverter
    Convert FrameNet XML files to JSON Lines format.

Functions
---------
convert_frame_file
    Convert a single frame XML file to Frame model.
convert_lu_file
    Convert a single lexical unit XML file to LexicalUnit model.
convert_frames_directory
    Convert all frame files in a directory to JSON Lines.

Examples
--------
>>> from pathlib import Path
>>> from glazing.framenet.converter import FrameNetConverter
>>> converter = FrameNetConverter()
>>> frame = converter.convert_frame_file("frame/Abandonment.xml")
>>> print(frame.name)
'Abandonment'

>>> # Convert entire directory
>>> converter.convert_frames_directory(
...     input_dir="framenet_v17/frame",
...     output_file="frames.jsonl"
... )
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

from lxml import etree

from glazing.framenet.models import (
    AnnotatedText,
    AnnotationLayer,
    AnnotationSet,
    FERealization,
    FERelation,
    Frame,
    FrameElement,
    FrameRelation,
    Label,
    Lexeme,
    LexicalUnit,
    SemanticType,
    SemTypeRef,
    Sentence,
    SentenceCount,
    ValenceAnnotationPattern,
    ValencePattern,
    ValenceRealizationPattern,
    ValenceUnit,
)
from glazing.framenet.types import (
    AnnotationStatus,
    LayerType,
)
from glazing.utils.xml_parser import (
    parse_attributes,
    parse_with_schema,
)

# Map from frRelation.xml relation type names to (sub_type, super_type) pairs.
# sub_type is the relation from the sub-frame's perspective;
# super_type is the relation from the super-frame's perspective (None if one-directional).
FRAME_RELATION_TYPE_MAP: dict[str, tuple[str, str | None]] = {
    "Inheritance": ("Inherits from", "Is Inherited by"),
    "Using": ("Uses", "Is Used by"),
    "Subframe": ("Subframe of", "Has Subframe(s)"),
    "Precedes": ("Precedes", "Is Preceded by"),
    "Perspective_on": ("Perspective on", "Is Perspectivized in"),
    "Causative_of": ("Is Causative of", None),
    "Inchoative_of": ("Is Inchoative of", None),
    "See_also": ("See also", "See also"),
}


class FrameNetConverter:
    """Convert FrameNet XML files to JSON Lines format.

    Parameters
    ----------
    namespace : str, default="http://framenet.icsi.berkeley.edu"
        FrameNet XML namespace URI.
    validate_schema : bool, default=False
        Whether to validate against DTD/XSD.

    Attributes
    ----------
    namespace : str
        FrameNet XML namespace.
    ns : dict[str, str]
        Namespace mapping for XPath.

    Methods
    -------
    convert_frame_file(filepath)
        Convert a frame XML file to Frame model.
    convert_lu_file(filepath)
        Convert a lexical unit XML file to LexicalUnit model.
    convert_frames_directory(input_dir, output_file)
        Convert all frames in a directory to JSON Lines.
    convert_frame_relations_file(filepath)
        Convert frRelation.xml to frame relation mappings.
    convert_semtypes_file(filepath, output_file)
        Convert semTypes.xml to JSON Lines.
    convert_fulltext_file(filepath)
        Convert a fulltext XML file to Sentence models.
    convert_fulltext_directory(input_dir, output_file)
        Convert all fulltext files in a directory to JSON Lines.
    """

    def __init__(
        self,
        namespace: str = "http://framenet.icsi.berkeley.edu",
        validate_schema: bool = False,
    ) -> None:
        """Initialize the converter.

        Parameters
        ----------
        namespace : str
            FrameNet XML namespace URI.
        validate_schema : bool
            Whether to validate XML against schema.
        """
        self.namespace = namespace
        self.ns = {"fn": namespace} if namespace else {}
        self.validate_schema = validate_schema
        self._ns_prefix = f"{{{namespace}}}" if namespace else ""

    def _tag(self, local_name: str) -> str:
        """Build a namespace-qualified tag name.

        Parameters
        ----------
        local_name : str
            The local element name.

        Returns
        -------
        str
            Namespace-qualified tag name.
        """
        return f"{self._ns_prefix}{local_name}"

    def _parse_definition(self, element: etree._Element | None) -> AnnotatedText:
        """Parse a definition element with embedded markup.

        Parameters
        ----------
        element : etree._Element | None
            The definition element.

        Returns
        -------
        AnnotatedText
            Parsed definition with annotations.
        """
        if element is None or element.text is None:
            return AnnotatedText(raw_text="", plain_text="", annotations=[])

        # FrameNet definitions have HTML entities that need decoding
        raw_text = element.text
        decoded_text = html.unescape(raw_text)

        # Parse the decoded text as AnnotatedText
        return AnnotatedText.parse(decoded_text)

    def _parse_datetime(self, date_str: str | None) -> datetime | None:
        """Parse FrameNet datetime string.

        Parameters
        ----------
        date_str : str | None
            Date string like "03/05/2008 03:50:35 PST Wed".

        Returns
        -------
        datetime | None
            Parsed datetime or None.
        """
        if not date_str:
            return None

        # FrameNet uses multiple datetime formats
        # Primary format: "MM/DD/YYYY HH:MM:SS TZ Day"
        # Secondary formats for variations in the data
        formats = [
            "%m/%d/%Y %H:%M:%S",  # Without timezone/day
            "%d/%m/%Y %H:%M:%S",  # European date format
            "%Y-%m-%d %H:%M:%S",  # ISO-like format
            "%m/%d/%Y",  # Date only
            "%d/%m/%Y",  # European date only
            "%Y-%m-%d",  # ISO date only
        ]

        # Try to parse with timezone and day suffix first
        parts = date_str.split()

        # Try primary format (first two parts: date and time)
        if len(parts) >= 2:
            date_time = " ".join(parts[:2])
            for fmt in formats[:3]:  # Try datetime formats first
                try:
                    return datetime.strptime(date_time, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue

        # Try with just the first part (date only)
        if len(parts) >= 1:
            for fmt in formats[3:]:  # Try date-only formats
                try:
                    return datetime.strptime(parts[0], fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue

        # Try the full string with all formats
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=UTC)
            except ValueError:
                continue

        # If all parsing attempts fail, raise an error with details
        error_msg = (
            f"Unable to parse datetime string '{date_str}'. "
            f"Expected format like 'MM/DD/YYYY HH:MM:SS TZ Day' or similar. "
            f"Tried formats: {', '.join(formats)}"
        )
        raise ValueError(error_msg)

    def _validate_fe_attributes(self, attrs: dict[str, str | int | float | bool]) -> None:
        """Validate required frame element attributes.

        Parameters
        ----------
        attrs : dict[str, str | int]
            Attributes to validate.

        Raises
        ------
        ValueError
            If required attributes are missing.
        """
        required = ["ID", "name", "coreType", "bgColor", "fgColor"]
        for attr in required:
            if attr not in attrs:
                msg = f"Missing required '{attr}' attribute in FrameElement"
                raise ValueError(msg)

    def _parse_fe_constraints(self, fe_elem: etree._Element) -> tuple[list[str], list[str]]:
        """Parse frame element constraints.

        Parameters
        ----------
        fe_elem : etree._Element
            The FE element.

        Returns
        -------
        tuple[list[str], list[str]]
            Requires and excludes lists.
        """
        requires_fe = []
        excludes_fe = []

        requires_elem = fe_elem.find(
            f"{{{self.namespace}}}requiresFE" if self.namespace else "requiresFE"
        )
        if requires_elem is not None:
            requires_fe = [req.get("name", "") for req in requires_elem]

        excludes_elem = fe_elem.find(
            f"{{{self.namespace}}}excludesFE" if self.namespace else "excludesFE"
        )
        if excludes_elem is not None:
            excludes_fe = [exc.get("name", "") for exc in excludes_elem]

        return requires_fe, excludes_fe

    def _parse_fe_semtypes(self, fe_elem: etree._Element) -> list[int]:
        """Parse frame element semantic types.

        Parameters
        ----------
        fe_elem : etree._Element
            The FE element.

        Returns
        -------
        list[int]
            Semantic type IDs.
        """
        semtype_refs = []
        semtypes_elem = fe_elem.find(
            f"{{{self.namespace}}}semTypes" if self.namespace else "semTypes"
        )
        if semtypes_elem is not None:
            for semtype in semtypes_elem:
                sem_id = semtype.get("ID")
                if sem_id:
                    semtype_refs.append(int(sem_id))
        return semtype_refs

    def _parse_frame_element(self, fe_elem: etree._Element) -> FrameElement:
        """Parse a frame element from XML.

        Parameters
        ----------
        fe_elem : etree._Element
            The FE element.

        Returns
        -------
        FrameElement
            Parsed frame element.
        """
        attrs = parse_attributes(fe_elem, {"ID": int})
        self._validate_fe_attributes(attrs)

        # Parse definition
        def_elem = fe_elem.find(
            f"{{{self.namespace}}}definition" if self.namespace else "definition"
        )
        definition = self._parse_definition(def_elem)

        # Parse constraints and semantic types
        requires_fe, excludes_fe = self._parse_fe_constraints(fe_elem)
        semtype_refs = self._parse_fe_semtypes(fe_elem)

        # Get abbreviation - no fallback generation
        abbrev = str(attrs.get("abbrev", ""))

        return FrameElement(
            id=int(attrs["ID"]),
            name=str(attrs["name"]),
            abbrev=abbrev if abbrev else str(attrs["name"]),  # Use name if abbrev empty
            definition=definition,
            core_type=str(attrs["coreType"]),  # type: ignore[arg-type]
            bg_color=str(attrs["bgColor"]),
            fg_color=str(attrs["fgColor"]),
            requires_fe=requires_fe,
            excludes_fe=excludes_fe,
            semtype_refs=semtype_refs,
            created_by=str(attrs.get("cBy")) if attrs.get("cBy") else None,
            created_date=self._parse_datetime(
                str(attrs.get("cDate")) if attrs.get("cDate") else None
            ),
        )

    def convert_frame_file(self, filepath: Path | str) -> Frame:
        """Convert a frame XML file to Frame model.

        Parameters
        ----------
        filepath : Path | str
            Path to frame XML file.

        Returns
        -------
        Frame
            Parsed Frame model instance.

        Examples
        --------
        >>> converter = FrameNetConverter()
        >>> frame = converter.convert_frame_file("frame/Abandonment.xml")
        >>> print(f"Frame: {frame.name} (ID: {frame.id})")
        'Frame: Abandonment (ID: 2031)'
        """
        filepath = Path(filepath)

        # Parse XML
        if self.validate_schema:
            root = parse_with_schema(filepath)
        else:
            tree = etree.parse(str(filepath))
            root = tree.getroot()

        # Extract frame attributes
        attrs = parse_attributes(root, {"ID": int})

        # Parse definition
        def_elem = root.find(f"{{{self.namespace}}}definition" if self.namespace else "definition")
        definition = self._parse_definition(def_elem)

        # Parse frame elements
        frame_elements = []
        fe_tag = f"{{{self.namespace}}}FE" if self.namespace else "FE"
        for fe_elem in root.findall(fe_tag):
            frame_elements.append(self._parse_frame_element(fe_elem))

        return Frame(
            id=int(attrs.get("ID", 0)),
            name=str(attrs.get("name", "")),
            definition=definition,
            frame_elements=frame_elements,
            created_by=str(attrs.get("cBy")) if attrs.get("cBy") else None,
            created_date=self._parse_datetime(
                str(attrs.get("cDate")) if attrs.get("cDate") else None
            ),
            modified_date=self._parse_datetime(
                str(attrs.get("mDate")) if attrs.get("mDate") else None
            ),
        )

    def _parse_lu_from_index(self, lu_elem: etree._Element) -> LexicalUnit:
        """Parse a lexical unit from luIndex.xml element.

        Parameters
        ----------
        lu_elem : etree._Element
            The LU element from luIndex.xml.

        Returns
        -------
        LexicalUnit
            Parsed lexical unit.
        """
        attrs = parse_attributes(
            lu_elem,
            {"ID": int, "frameID": int, "numAnnotInstances": int, "hasAnnotation": bool},
        )

        # Extract required attributes
        lu_id = int(attrs["ID"])
        lu_name = str(attrs["name"])
        frame_id = int(attrs["frameID"])
        frame_name = str(attrs["frameName"])
        status = str(attrs.get("status", "Unknown"))
        has_annotation = bool(attrs.get("hasAnnotation", False))
        num_annotated = int(attrs.get("numAnnotInstances", 0))

        # Extract POS from name (e.g., "abandon.v" -> "V")
        parts = lu_name.split(".")
        if len(parts) >= 2:
            pos_lower = parts[-1]
            # Map lowercase POS to uppercase
            pos_map = {"v": "V", "n": "N", "a": "A", "adv": "ADV", "prep": "PREP", "num": "NUM"}
            pos = pos_map.get(pos_lower, pos_lower.upper())
        else:
            # Default to V if no POS specified
            pos = "V"

        # Create lexemes from the name
        # Extract lemma part (everything before the last dot)
        lemma = ".".join(parts[:-1]) if len(parts) >= 2 else lu_name

        # Split multi-word LUs: try underscore first, then space
        # Examples: "give_up.v" -> ["give", "up"], "a bit.n" -> ["a", "bit"]
        if "_" in lemma:
            word_parts = lemma.split("_")
        elif " " in lemma:
            word_parts = lemma.split(" ")
        else:
            word_parts = [lemma]

        lexemes = []
        for i, word in enumerate(word_parts):
            if not word:  # Skip empty strings from splitting
                continue
            # First word is typically the headword
            is_headword = i == 0
            # Keep the word as-is: real FrameNet data has parentheses, brackets, etc.
            lexemes.append(
                Lexeme(
                    name=word,
                    pos=pos,  # type: ignore[arg-type]
                    headword=is_headword,
                    order=i + 1,
                )
            )

        # Create sentence count from annotation data
        sentence_count = SentenceCount(
            annotated=num_annotated if has_annotation else 0,
            total=num_annotated,  # For luIndex, we only have annotated count
        )

        # Create minimal definition (luIndex doesn't have full definitions)
        definition = f"Lexical unit '{lu_name}' in frame '{frame_name}'."

        # Note: Using field names, not aliases, since we're constructing programmatically
        # The model uses aliases for deserialization from JSON
        lu_dict = {
            "id": lu_id,
            "name": lu_name,
            "pos": pos,
            "definition": definition,
            "status": status if status != "Unknown" else None,  # annotation_status alias
            "totalAnnotated": num_annotated if has_annotation else None,  # total_annotated
            "has_annotated_examples": has_annotation,
            "frame_id": frame_id,
            "frame_name": frame_name,
            "sentence_count": sentence_count,
            "lexemes": lexemes,
        }
        return LexicalUnit(**lu_dict)  # type: ignore[arg-type]

    def convert_lu_index_file(self, filepath: Path | str) -> list[LexicalUnit]:
        """Convert luIndex.xml to a list of LexicalUnit models.

        Parameters
        ----------
        filepath : Path | str
            Path to luIndex.xml file.

        Returns
        -------
        list[LexicalUnit]
            List of parsed LexicalUnit models.

        Examples
        --------
        >>> converter = FrameNetConverter()
        >>> lus = converter.convert_lu_index_file("framenet_v17/luIndex.xml")
        >>> print(f"Loaded {len(lus)} lexical units")
        'Loaded 13575 lexical units'
        """
        filepath = Path(filepath)

        # Parse XML
        if self.validate_schema:
            root = parse_with_schema(filepath)
        else:
            tree = etree.parse(str(filepath))
            root = tree.getroot()

        # Parse all LU elements
        lexical_units = []
        lu_tag = f"{{{self.namespace}}}lu" if self.namespace else "lu"
        for lu_elem in root.findall(lu_tag):
            try:
                lu = self._parse_lu_from_index(lu_elem)
                lexical_units.append(lu)
            except (ValueError, KeyError, TypeError) as e:
                # Skip invalid LUs but continue processing
                lu_name = lu_elem.get("name", "unknown")
                # Log error but don't fail entire conversion
                print(f"Warning: Failed to parse LU '{lu_name}': {e}")
                continue

        return lexical_units

    def convert_frame_relations_file(self, filepath: Path | str) -> dict[int, list[FrameRelation]]:
        """Convert frRelation.xml to frame relation mappings.

        Parses the frame relation types and individual frame relations,
        creating FrameRelation objects grouped by frame ID.

        Parameters
        ----------
        filepath : Path | str
            Path to frRelation.xml file.

        Returns
        -------
        dict[int, list[FrameRelation]]
            Dictionary mapping frame IDs to their FrameRelation objects.

        Examples
        --------
        >>> converter = FrameNetConverter()
        >>> relations = converter.convert_frame_relations_file("frRelation.xml")
        >>> print(f"Found relations for {len(relations)} frames")
        """
        filepath = Path(filepath)

        tree = etree.parse(str(filepath))
        root = tree.getroot()

        relations_by_frame: dict[int, list[FrameRelation]] = {}

        for rel_type_elem in root.findall(self._tag("frameRelationType")):
            type_name = rel_type_elem.get("name", "")

            if type_name not in FRAME_RELATION_TYPE_MAP:
                continue

            sub_type, super_type = FRAME_RELATION_TYPE_MAP[type_name]

            for fr_elem in rel_type_elem.findall(self._tag("frameRelation")):
                sub_frame_id = int(fr_elem.get("subID", "0"))
                sup_frame_id = int(fr_elem.get("supID", "0"))
                sub_frame_name = fr_elem.get("subFrameName", "")
                super_frame_name = fr_elem.get("superFrameName", "")
                relation_id = int(fr_elem.get("ID", "0"))

                # Parse FE relations
                fe_relations: list[FERelation] = []
                for fe_rel_elem in fr_elem.findall(self._tag("FERelation")):
                    try:
                        fe_rel = FERelation(  # type: ignore[call-arg]
                            sub_fe_id=int(fe_rel_elem.get("subID", "0")),
                            sub_fe_name=fe_rel_elem.get("subFEName"),
                            super_fe_id=int(fe_rel_elem.get("supID", "0")),
                            super_fe_name=fe_rel_elem.get("superFEName"),
                        )
                        fe_relations.append(fe_rel)
                    except (ValueError, TypeError):
                        continue

                # Create FrameRelation for the sub-frame's perspective
                try:
                    sub_relation = FrameRelation(
                        id=relation_id,
                        type=sub_type,  # type: ignore[arg-type]
                        sub_frame_id=sub_frame_id,
                        sub_frame_name=sub_frame_name,
                        super_frame_id=sup_frame_id,
                        super_frame_name=super_frame_name,
                        fe_relations=fe_relations,
                    )
                    relations_by_frame.setdefault(sub_frame_id, []).append(sub_relation)
                except (ValueError, TypeError):
                    pass

                # Create FrameRelation for the super-frame's perspective (if applicable)
                if super_type is not None:
                    try:
                        super_relation = FrameRelation(
                            id=relation_id,
                            type=super_type,  # type: ignore[arg-type]
                            sub_frame_id=sub_frame_id,
                            sub_frame_name=sub_frame_name,
                            super_frame_id=sup_frame_id,
                            super_frame_name=super_frame_name,
                            fe_relations=fe_relations,
                        )
                        relations_by_frame.setdefault(sup_frame_id, []).append(super_relation)
                    except (ValueError, TypeError):
                        pass

        return relations_by_frame

    def convert_lu_file(
        self, filepath: Path | str
    ) -> tuple[list[ValencePattern], list[SemTypeRef], list[AnnotationSet]]:
        """Convert an individual lu/*.xml file to extract valence patterns and semtypes.

        Parses valence patterns (FE realizations and their syntactic patterns),
        semantic type references, and annotation sets from a lexical unit file.

        Parameters
        ----------
        filepath : Path | str
            Path to individual lu XML file (e.g., lu/lu10.xml).

        Returns
        -------
        tuple[list[ValencePattern], list[SemTypeRef], list[AnnotationSet]]
            Tuple of (valence_patterns, semtypes, annotation_sets).

        Examples
        --------
        >>> converter = FrameNetConverter()
        >>> patterns, semtypes, annosets = converter.convert_lu_file("lu/lu10.xml")
        >>> print(f"Found {len(patterns)} valence patterns")
        """
        filepath = Path(filepath)

        tree = etree.parse(str(filepath))
        root = tree.getroot()

        # Parse semantic types (direct children of root)
        semtypes: list[SemTypeRef] = []
        for semtype_elem in root.findall(self._tag("semType")):
            st_name = semtype_elem.get("name")
            st_id = semtype_elem.get("ID")
            if st_name and st_id:
                try:
                    semtypes.append(SemTypeRef(name=st_name, id=int(st_id)))
                except (ValueError, TypeError):
                    continue

        # Parse valence patterns from <valences> element
        valence_patterns: list[ValencePattern] = []
        valences_elem = root.find(self._tag("valences"))
        if valences_elem is not None:
            # Parse FE realizations
            fe_realizations: list[FERealization] = []
            for fe_real_elem in valences_elem.findall(self._tag("FERealization")):
                fe_real_total = int(fe_real_elem.get("total", "0"))

                # Get FE name from child <FE> element
                fe_child = fe_real_elem.find(self._tag("FE"))
                fe_name = fe_child.get("name", "") if fe_child is not None else ""

                if not fe_name:
                    continue

                # Parse patterns within this FE realization
                patterns: list[ValenceRealizationPattern] = []
                for pattern_elem in fe_real_elem.findall(self._tag("pattern")):
                    pattern_total = int(pattern_elem.get("total", "0"))

                    # Parse valence units
                    valence_units: list[ValenceUnit] = []
                    for vu_elem in pattern_elem.findall(self._tag("valenceUnit")):
                        try:
                            vu = ValenceUnit(
                                GF=vu_elem.get("GF", ""),
                                PT=vu_elem.get("PT", ""),
                                FE=vu_elem.get("FE", ""),
                            )
                            valence_units.append(vu)
                        except (ValueError, TypeError):
                            continue

                    # Parse annotation set IDs
                    anno_set_ids: list[int] = []
                    for anno_elem in pattern_elem.findall(self._tag("annoSet")):
                        anno_id = anno_elem.get("ID")
                        if anno_id:
                            anno_set_ids.append(int(anno_id))

                    if valence_units and pattern_total > 0:
                        try:
                            patterns.append(
                                ValenceRealizationPattern(
                                    valence_units=valence_units,
                                    anno_set_ids=anno_set_ids,
                                    total=pattern_total,
                                )
                            )
                        except (ValueError, TypeError):
                            continue

                try:
                    fe_realizations.append(
                        FERealization(
                            fe_name=fe_name,
                            total=fe_real_total,
                            patterns=patterns,
                        )
                    )
                except (ValueError, TypeError):
                    continue

            # Build a single ValencePattern if we have FE realizations
            if fe_realizations:
                # Compute total annotated from the root <valences> or LU attributes
                total_annotated = int(root.get("totalAnnotated", "0"))

                # Parse FEGroupRealization / ValenceAnnotationPattern entries
                valence_anno_patterns: list[ValenceAnnotationPattern] = []
                # These come from <FEGroupRealization> elements in the valences section
                # (not all LU files have these)

                valence_patterns.append(
                    ValencePattern(
                        total_annotated=total_annotated,
                        fe_realizations=fe_realizations,
                        patterns=valence_anno_patterns,
                    )
                )

        # Parse annotation sets (from <subCorpus> sections)
        annotation_sets: list[AnnotationSet] = []
        # Annotation sets in lu files are nested inside subCorpus > sentence > annotationSet
        # We collect them but don't return full sentences here
        for subcorpus_elem in root.findall(self._tag("subCorpus")):
            for sentence_elem in subcorpus_elem.findall(self._tag("sentence")):
                sent_id = int(sentence_elem.get("ID", "0"))
                for annoset_elem in sentence_elem.findall(self._tag("annotationSet")):
                    try:
                        annoset = self._parse_annotation_set(annoset_elem, sent_id)
                        if annoset is not None:
                            annotation_sets.append(annoset)
                    except (ValueError, TypeError):
                        continue

        return valence_patterns, semtypes, annotation_sets

    def convert_semtypes_file(self, filepath: Path | str, output_file: Path | str) -> int:
        """Convert semTypes.xml to JSON Lines format.

        Parses the semantic type hierarchy and writes each type as a JSON line.

        Parameters
        ----------
        filepath : Path | str
            Path to semTypes.xml file.
        output_file : Path | str
            Output JSON Lines file path.

        Returns
        -------
        int
            Number of semantic types converted.

        Examples
        --------
        >>> converter = FrameNetConverter()
        >>> count = converter.convert_semtypes_file("semTypes.xml", "semtypes.jsonl")
        >>> print(f"Converted {count} semantic types")
        """
        filepath = Path(filepath)
        output_file = Path(output_file)

        tree = etree.parse(str(filepath))
        root = tree.getroot()

        # semTypes.xml uses the FrameNet namespace
        semtype_tag = self._tag("semType")
        definition_tag = self._tag("definition")
        supertype_tag = self._tag("superType")

        semantic_types: list[SemanticType] = []

        for st_elem in root.findall(semtype_tag):
            st_id = st_elem.get("ID")
            st_name = st_elem.get("name", "")
            st_abbrev = st_elem.get("abbrev", "")

            if not st_id or not st_name:
                continue

            # Parse definition
            def_elem = st_elem.find(definition_tag)
            definition_text = ""
            if def_elem is not None and def_elem.text:
                definition_text = def_elem.text.strip()
            if not definition_text:
                definition_text = f"Semantic type: {st_name}"

            # Parse super type
            super_type_id = None
            super_type_name = None
            sup_elem = st_elem.find(supertype_tag)
            if sup_elem is not None:
                sup_id = sup_elem.get("supID")
                sup_name = sup_elem.get("superTypeName")
                if sup_id:
                    super_type_id = int(sup_id)
                    super_type_name = sup_name

            try:
                sem_type = SemanticType(
                    id=int(st_id),
                    name=st_name,
                    abbrev=st_abbrev if st_abbrev else st_name,
                    definition=definition_text,
                    super_type_id=super_type_id,
                    super_type_name=super_type_name,
                    root_type_id=None,
                    root_type_name=None,
                )
                semantic_types.append(sem_type)
            except (ValueError, TypeError) as e:
                print(f"Warning: Failed to parse semantic type '{st_name}': {e}")
                continue

        # Write to output file
        count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for sem_type in semantic_types:
                json_line = sem_type.model_dump_json(exclude_none=True)
                f.write(json_line + "\n")
                count += 1

        return count

    def _parse_annotation_set(
        self, annoset_elem: etree._Element, sentence_id: int
    ) -> AnnotationSet | None:
        """Parse an annotation set element.

        Parameters
        ----------
        annoset_elem : etree._Element
            The annotationSet XML element.
        sentence_id : int
            ID of the containing sentence.

        Returns
        -------
        AnnotationSet | None
            Parsed annotation set, or None if invalid.
        """
        anno_id = annoset_elem.get("ID")
        status = annoset_elem.get("status", "")

        if not anno_id:
            return None

        # Validate status against allowed values
        valid_statuses = get_args(AnnotationStatus.__value__)
        if status not in valid_statuses:
            return None

        # Parse created_by and created_date
        cby = annoset_elem.get("cBy")
        cdate_str = annoset_elem.get("cDate")
        cdate = self._parse_datetime(cdate_str) if cdate_str else None

        # Parse layers
        layers: list[AnnotationLayer] = []
        valid_layer_types = get_args(LayerType.__value__)

        for layer_elem in annoset_elem.findall(self._tag("layer")):
            layer_name = layer_elem.get("name", "")
            layer_rank = int(layer_elem.get("rank", "1"))

            if layer_name not in valid_layer_types:
                continue

            # Parse labels
            labels: list[Label] = []
            for label_elem in layer_elem.findall(self._tag("label")):
                label_name = label_elem.get("name", "")
                if not label_name:
                    continue

                start_str = label_elem.get("start")
                end_str = label_elem.get("end")
                itype = label_elem.get("itype")
                label_id_str = label_elem.get("ID")
                fe_id_str = label_elem.get("feID")

                # Handle null instantiation labels (no start/end attributes)
                if itype and (start_str is None or end_str is None):
                    # Null instantiation: set start=0, end=0
                    start_val = 0
                    end_val = 0
                    is_null = True
                elif start_str is not None and end_str is not None:
                    start_val = int(start_str)
                    end_val = int(end_str)
                    is_null = bool(itype)
                else:
                    # Labels without start/end and without itype - skip
                    continue

                # Validate positions
                if start_val < 0 or end_val < start_val:
                    if is_null:
                        start_val = 0
                        end_val = 0
                    else:
                        continue

                try:
                    label = Label(
                        id=int(label_id_str) if label_id_str else None,
                        name=label_name,
                        start=start_val,
                        end=end_val,
                        fe_id=int(fe_id_str) if fe_id_str else None,
                        itype=is_null,
                    )
                    labels.append(label)
                except (ValueError, TypeError):
                    continue

            try:
                layers.append(
                    AnnotationLayer(
                        name=layer_name,  # type: ignore[arg-type]
                        rank=layer_rank,
                        labels=labels,
                    )
                )
            except (ValueError, TypeError):
                continue

        try:
            return AnnotationSet(
                id=int(anno_id),
                status=status,  # type: ignore[arg-type]
                sentence_id=sentence_id,
                layers=layers,
                cBy=cby,
                cDate=cdate,
            )
        except (ValueError, TypeError):
            return None

    def convert_fulltext_file(self, filepath: Path | str) -> list[Sentence]:
        """Convert a fulltext/*.xml file to Sentence models.

        Parses annotated corpus sentences with their annotation sets,
        layers, and labels.

        Parameters
        ----------
        filepath : Path | str
            Path to fulltext XML file.

        Returns
        -------
        list[Sentence]
            List of parsed Sentence models.

        Examples
        --------
        >>> converter = FrameNetConverter()
        >>> sentences = converter.convert_fulltext_file("fulltext/ANC__110CYL067.xml")
        >>> print(f"Found {len(sentences)} sentences")
        """
        filepath = Path(filepath)

        tree = etree.parse(str(filepath))
        root = tree.getroot()

        sentences: list[Sentence] = []

        for sent_elem in root.findall(self._tag("sentence")):
            sent_id_str = sent_elem.get("ID")
            if not sent_id_str:
                continue
            sent_id = int(sent_id_str)

            # Get sentence text
            text_elem = sent_elem.find(self._tag("text"))
            if text_elem is None or not text_elem.text:
                continue
            text = text_elem.text

            # Get sentence metadata
            parag_no_str = sent_elem.get("paragNo")
            sent_no_str = sent_elem.get("sentNo")
            corp_id_str = sent_elem.get("corpID")
            doc_id_str = sent_elem.get("docID")
            apos_str = sent_elem.get("aPos")

            # Parse annotation sets
            annotation_sets: list[AnnotationSet] = []
            for annoset_elem in sent_elem.findall(self._tag("annotationSet")):
                try:
                    annoset = self._parse_annotation_set(annoset_elem, sent_id)
                    if annoset is not None:
                        annotation_sets.append(annoset)
                except (ValueError, TypeError):
                    continue

            try:
                sentence = Sentence(
                    id=sent_id,
                    text=text,
                    paragNo=int(parag_no_str) if parag_no_str else None,
                    sentNo=int(sent_no_str) if sent_no_str else None,
                    corpID=int(corp_id_str) if corp_id_str else None,
                    docID=int(doc_id_str) if doc_id_str else None,
                    apos=int(apos_str) if apos_str else None,
                    annotation_sets=annotation_sets,
                )
                sentences.append(sentence)
            except (ValueError, TypeError) as e:
                print(f"Warning: Failed to parse sentence {sent_id}: {e}")
                continue

        return sentences

    def convert_fulltext_directory(
        self,
        input_dir: Path | str,
        output_file: Path | str,
        pattern: str = "*.xml",
    ) -> int:
        """Convert all fulltext files in a directory to JSON Lines.

        Parameters
        ----------
        input_dir : Path | str
            Directory containing fulltext XML files.
        output_file : Path | str
            Output JSON Lines file path.
        pattern : str, default="*.xml"
            File pattern to match.

        Returns
        -------
        int
            Number of sentences converted.

        Examples
        --------
        >>> converter = FrameNetConverter()
        >>> count = converter.convert_fulltext_directory(
        ...     "framenet_v17/fulltext",
        ...     "fulltext.jsonl"
        ... )
        >>> print(f"Converted {count} sentences")
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)

        count = 0
        errors: list[tuple[Path, Exception]] = []

        with output_file.open("w", encoding="utf-8") as f:
            for xml_file in sorted(input_dir.glob(pattern)):
                try:
                    sentences = self.convert_fulltext_file(xml_file)
                    for sentence in sentences:
                        json_line = sentence.model_dump_json(exclude_none=True)
                        f.write(json_line + "\n")
                        count += 1
                except (etree.XMLSyntaxError, ValueError, TypeError) as e:
                    errors.append((xml_file, e))

        if errors:
            error_details = "\n".join(f"  - {file}: {error}" for file, error in errors)
            total_files = len(list(input_dir.glob(pattern)))
            error_msg = (
                f"Failed to convert {len(errors)} out of {total_files} files:\n{error_details}"
            )
            raise RuntimeError(error_msg)

        return count

    def convert_frames_directory(
        self,
        input_dir: Path | str,
        output_file: Path | str,
        pattern: str = "*.xml",
    ) -> int:
        """Convert all frame files in a directory to JSON Lines with lexical units.

        This method parses frame XML files and associates them with lexical units
        from luIndex.xml (expected to be in the parent directory of input_dir).
        It also loads frame relations from frRelation.xml and enriches LUs with
        valence patterns and semantic types from individual lu/*.xml files.

        Parameters
        ----------
        input_dir : Path | str
            Directory containing frame XML files.
        output_file : Path | str
            Output JSON Lines file path.
        pattern : str, default="*.xml"
            File pattern to match.

        Returns
        -------
        int
            Number of frames converted.

        Examples
        --------
        >>> converter = FrameNetConverter()
        >>> count = converter.convert_frames_directory(
        ...     "framenet_v17/frame",
        ...     "frames.jsonl"
        ... )
        >>> print(f"Converted {count} frames")
        'Converted 1221 frames'
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)

        # First, parse all frames
        frames: list[Frame] = []
        errors: list[tuple[Path, Exception]] = []

        for xml_file in sorted(input_dir.glob(pattern)):
            try:
                frame = self.convert_frame_file(xml_file)
                frames.append(frame)
            except (etree.XMLSyntaxError, ValueError, TypeError) as e:
                errors.append((xml_file, e))

        # Load lexical units from luIndex.xml (in parent directory)
        parent_dir = input_dir.parent if input_dir.name == "frame" else input_dir
        lu_index_path = parent_dir / "luIndex.xml"

        lexical_units: list[LexicalUnit] = []
        if lu_index_path.exists():
            try:
                lexical_units = self.convert_lu_index_file(lu_index_path)
            except (etree.XMLSyntaxError, ValueError, TypeError) as e:
                print(f"Warning: Failed to load lexical units from {lu_index_path}: {e}")

        # Associate LUs with frames by frame_id
        lu_by_frame: dict[int, list[LexicalUnit]] = {}
        for lu in lexical_units:
            if lu.frame_id not in lu_by_frame:
                lu_by_frame[lu.frame_id] = []
            lu_by_frame[lu.frame_id].append(lu)

        # Update frames with their lexical units
        for frame in frames:
            frame.lexical_units = lu_by_frame.get(frame.id, [])

        # Load frame relations from frRelation.xml
        fr_relation_path = parent_dir / "frRelation.xml"
        if fr_relation_path.exists():
            try:
                relations_by_frame = self.convert_frame_relations_file(fr_relation_path)
                for frame in frames:
                    frame.frame_relations = relations_by_frame.get(frame.id, [])
            except (etree.XMLSyntaxError, ValueError, TypeError) as e:
                print(f"Warning: Failed to load frame relations from {fr_relation_path}: {e}")

        # Enrich LUs with valence patterns and semtypes from individual lu/*.xml files
        lu_dir = parent_dir / "lu"
        if lu_dir.is_dir():
            for frame in frames:
                for lu in frame.lexical_units:
                    lu_file = lu_dir / f"lu{lu.id}.xml"
                    if lu_file.exists():
                        try:
                            valence_patterns, semtypes, _annotation_sets = self.convert_lu_file(
                                lu_file
                            )
                            if valence_patterns:
                                lu.valence_patterns = valence_patterns
                            if semtypes:
                                lu.semtypes = semtypes
                        except (etree.XMLSyntaxError, ValueError, TypeError) as e:
                            print(f"Warning: Failed to parse LU file {lu_file}: {e}")
                            continue

        # Write frames with LUs to output file
        count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for frame in frames:
                json_line = frame.model_dump_json(exclude_none=True)
                f.write(json_line + "\n")
                count += 1

        # If there were any errors, raise an exception with details
        if errors:
            error_details = "\n".join(f"  - {file}: {error}" for file, error in errors)
            total_files = count + len(errors)
            error_msg = (
                f"Failed to convert {len(errors)} out of {total_files} files:\n{error_details}"
            )
            raise RuntimeError(error_msg)

        return count


# Convenience functions
def convert_frame_file(filepath: Path | str) -> Frame:
    """Convert a single frame XML file to Frame model.

    Parameters
    ----------
    filepath : Path | str
        Path to frame XML file.

    Returns
    -------
    Frame
        Parsed Frame model.
    """
    converter = FrameNetConverter()
    return converter.convert_frame_file(filepath)


def convert_frames_directory(
    input_dir: Path | str,
    output_file: Path | str,
    pattern: str = "*.xml",
) -> int:
    """Convert all frames in a directory to JSON Lines.

    Parameters
    ----------
    input_dir : Path | str
        Directory with frame XML files.
    output_file : Path | str
        Output JSON Lines file.
    pattern : str
        File pattern to match.

    Returns
    -------
    int
        Number of frames converted.
    """
    converter = FrameNetConverter()
    return converter.convert_frames_directory(input_dir, output_file, pattern)
