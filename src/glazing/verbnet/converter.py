"""VerbNet XML to JSON Lines converter.

This module provides conversion from VerbNet XML format to JSON Lines format
using the glazing VerbNet models. Handles verb class hierarchy with role
inheritance, selectional restrictions with complex logic, and cross-references.

Classes
-------
VerbNetConverter
    Convert VerbNet XML files to JSON Lines format.

Functions
---------
convert_verbnet_file
    Convert a single VerbNet XML file to VerbClass model.
convert_verbnet_directory
    Convert all VerbNet XML files in a directory to JSON Lines.
parse_member_cross_references
    Parse cross-references from member attributes.
parse_selectional_restrictions
    Parse selectional restrictions with nested logic.

Examples
--------
>>> from pathlib import Path
>>> from glazing.verbnet.converter import VerbNetConverter
>>> converter = VerbNetConverter()
>>> verb_class = converter.convert_verbnet_file("verbnet/give-13.1.xml")
>>> print(verb_class.id)
'give-13.1'

>>> # Convert entire directory
>>> converter.convert_verbnet_directory(
...     input_dir="verbnet_v34",
...     output_file="verbnet.jsonl"
... )
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from lxml import etree

from glazing.references.models import (
    CrossReference,
    MappingMetadata,
    VerbNetFrameNetMapping,
)
from glazing.types import LogicType
from glazing.utils.xml_parser import parse_attributes
from glazing.verbnet.models import (
    Example,
    FrameDescription,
    Member,
    Predicate,
    PredicateArgument,
    SelectionalRestriction,
    SelectionalRestrictions,
    Semantics,
    SyntacticRestriction,
    Syntax,
    SyntaxElement,
    ThematicRole,
    VerbClass,
    VNFrame,
    WordNetCrossRef,
)
from glazing.verbnet.types import (
    DESCRIPTION_NUMBER_PATTERN,
    VERBNET_CLASS_PATTERN,
    ArgumentType,
    PredicateType,
    RestrictionValue,
    SelectionalRestrictionType,
    SyntacticPOS,
    SyntacticRestrictionType,
    ThematicRoleType,
    VerbClassID,
)


class VerbNetConverter:
    """Convert VerbNet XML files to JSON Lines format.

    Handles VerbNet XML parsing with proper inheritance resolution,
    cross-reference extraction, and complex selectional restrictions.

    Methods
    -------
    convert_verbnet_file(filepath)
        Convert a single VerbNet XML file to VerbClass model.
    convert_verbnet_directory(input_dir, output_file)
        Convert all VerbNet XML files to JSON Lines.
    parse_verb_class(element, parent_id)
        Parse a VNCLASS element into VerbClass model.
    parse_members(element)
        Parse MEMBERS element into list of Member models.
    parse_themroles(element)
        Parse THEMROLES element into list of ThematicRole models.
    parse_frames(element)
        Parse FRAMES element into list of VNFrame models.
    """

    def convert_verbnet_file(self, filepath: Path | str) -> VerbClass:
        """Convert a single VerbNet XML file to VerbClass model.

        Parameters
        ----------
        filepath : Path | str
            Path to VerbNet XML file.

        Returns
        -------
        VerbClass
            Parsed VerbClass model with all subclasses.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If XML parsing fails or structure is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"VerbNet file not found: {filepath}"
            raise FileNotFoundError(msg)

        try:
            tree = etree.parse(str(filepath))
            root = tree.getroot()

            if root.tag != "VNCLASS":
                msg = f"Expected VNCLASS root element, got {root.tag}"
                raise ValueError(msg)

            return self.parse_verb_class(root)

        except etree.XMLSyntaxError as e:
            msg = f"XML parsing failed for {filepath}: {e}"
            raise ValueError(msg) from e

    def convert_verbnet_directory(self, input_dir: Path | str, output_file: Path | str) -> int:
        """Convert all VerbNet XML files in a directory to JSON Lines.

        Parameters
        ----------
        input_dir : Path | str
            Directory containing VerbNet XML files.
        output_file : Path | str
            Output JSON Lines file path.

        Returns
        -------
        int
            Number of files processed.

        Raises
        ------
        FileNotFoundError
            If the input directory does not exist.
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)

        if not input_dir.exists():
            msg = f"Input directory not found: {input_dir}"
            raise FileNotFoundError(msg)

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        xml_files = list(input_dir.glob("*.xml"))
        if not xml_files:
            msg = f"No XML files found in {input_dir}"
            raise FileNotFoundError(msg)

        count = 0
        with output_file.open("w", encoding="utf-8") as f:
            for xml_file in xml_files:
                try:
                    verb_class = self.convert_verbnet_file(xml_file)
                    json_line = verb_class.model_dump_json()
                    f.write(f"{json_line}\n")
                    count += 1
                except (ValueError, FileNotFoundError) as e:
                    # Log error but continue processing
                    print(f"Error processing {xml_file}: {e}")
                    continue

        return count

    def parse_verb_class(
        self, element: etree._Element, parent_id: VerbClassID | None = None
    ) -> VerbClass:
        """Parse a VNCLASS element into VerbClass model.

        Parameters
        ----------
        element : etree._Element
            VNCLASS XML element.
        parent_id : VerbClassID | None, default=None
            Parent class ID for subclasses.

        Returns
        -------
        VerbClass
            Parsed VerbClass model.

        Raises
        ------
        ValueError
            If required attributes are missing or invalid.
        """
        attrs = parse_attributes(element)
        class_id = str(attrs.get("ID", "")).strip()

        if not class_id:
            msg = "VNCLASS element missing ID attribute"
            raise ValueError(msg)

        # Validate class ID format
        if not re.match(VERBNET_CLASS_PATTERN, class_id):
            msg = f"Invalid VerbNet class ID format: {class_id}"
            raise ValueError(msg)

        # Parse components
        members = self._parse_members(element)
        themroles = self._parse_themroles(element)
        frames = self._parse_frames(element)
        subclasses = self._parse_subclasses(element, class_id)

        return VerbClass(
            id=class_id,
            members=members,
            themroles=themroles,
            frames=frames,
            subclasses=subclasses,
            parent_class=parent_id,
        )

    def _parse_members(self, element: etree._Element) -> list[Member]:
        """Parse MEMBERS element into list of Member models.

        Parameters
        ----------
        element : etree._Element
            VNCLASS or SUBCLASS element.

        Returns
        -------
        list[Member]
            List of parsed Member models.
        """
        members: list[Member] = []

        members_elem = element.find("MEMBERS")
        if members_elem is None:
            return members

        for member_elem in members_elem.findall("MEMBER"):
            attrs = parse_attributes(member_elem)

            name = str(attrs.get("name", "")).strip()
            verbnet_key = str(attrs.get("verbnet_key", "")).strip()

            if not name or not verbnet_key:
                continue  # Skip incomplete members

            # Parse cross-references from attributes
            features = {}
            wn_senses = str(attrs.get("wn", "")).strip()
            grouping = str(attrs.get("grouping", "")).strip()
            fn_mapping = str(attrs.get("fn_mapping", "")).strip()
            features_str = str(attrs.get("features", "")).strip()

            # Parse features if present
            if features_str:
                # Features are typically space-separated key=value pairs
                for feature in features_str.split():
                    if "=" in feature:
                        key, value = feature.split("=", 1)
                        features[key] = value

            # Parse WordNet mappings
            wordnet_mappings = []
            if wn_senses:
                for sense in wn_senses.split():
                    if "%" in sense:
                        try:
                            wn_ref = WordNetCrossRef.from_percentage_notation(sense)
                            wordnet_mappings.append(wn_ref)
                        except ValueError:
                            # Skip invalid percentage notation
                            continue

            # Parse FrameNet mappings from fn_mapping attribute
            framenet_mappings: list[VerbNetFrameNetMapping] = []
            if fn_mapping and fn_mapping != "None":
                framenet_mappings.append(
                    VerbNetFrameNetMapping(
                        frame_name=fn_mapping,
                        confidence=None,
                        mapping_source="manual",
                        role_mappings=[],
                    )
                )

            # Parse PropBank mappings from grouping attribute
            propbank_mappings: list[CrossReference] = []
            if grouping and grouping != "None":
                mapping_metadata = MappingMetadata(
                    created_date=datetime.now(tz=UTC),
                    created_by="verbnet_xml",
                    version="3.4",
                    validation_status="unvalidated",
                )
                for roleset_id in grouping.split():
                    if roleset_id.strip():
                        propbank_mappings.append(
                            CrossReference(
                                source_dataset="verbnet",
                                source_id=verbnet_key,
                                source_version="3.4",
                                target_dataset="propbank",
                                target_id=roleset_id.strip(),
                                mapping_type="direct",
                                metadata=mapping_metadata,
                            )
                        )

            # Create member model
            member = Member(
                name=name,
                verbnet_key=verbnet_key,
                wordnet_mappings=wordnet_mappings,
                framenet_mappings=framenet_mappings,
                propbank_mappings=propbank_mappings,
                features=features,
            )

            members.append(member)

        return members

    def _parse_themroles(self, element: etree._Element) -> list[ThematicRole]:
        """Parse THEMROLES element into list of ThematicRole models.

        Parameters
        ----------
        element : etree._Element
            VNCLASS or SUBCLASS element.

        Returns
        -------
        list[ThematicRole]
            List of parsed ThematicRole models.
        """
        roles: list[ThematicRole] = []

        themroles_elem = element.find("THEMROLES")
        if themroles_elem is None:
            return roles

        for role_elem in themroles_elem.findall("THEMROLE"):
            attrs = parse_attributes(role_elem)
            role_type = str(attrs.get("type", "")).strip()

            if not role_type:
                continue

            # Parse selectional restrictions
            sel_restrictions = None
            selrestrs_elem = role_elem.find("SELRESTRS")
            if selrestrs_elem is not None:
                sel_restrictions = self._parse_selectional_restrictions(selrestrs_elem)

            role = ThematicRole(
                type=cast(ThematicRoleType, role_type),
                sel_restrictions=sel_restrictions,
            )
            roles.append(role)

        return roles

    def _parse_selectional_restrictions(self, element: etree._Element) -> SelectionalRestrictions:
        """Parse SELRESTRS element into SelectionalRestrictions model.

        Parameters
        ----------
        element : etree._Element
            SELRESTRS XML element.

        Returns
        -------
        SelectionalRestrictions
            Parsed selectional restrictions.
        """
        attrs = parse_attributes(element)
        logic = attrs.get("logic")

        # Validate logic type
        logic_value: LogicType | None = None
        if isinstance(logic, str) and logic in ("or", "and"):
            logic_value = cast(LogicType, logic)

        restrictions: list[SelectionalRestriction | SelectionalRestrictions] = []

        # Parse individual SELRESTR elements
        for selrestr_elem in element.findall("SELRESTR"):
            restr_attrs = parse_attributes(selrestr_elem)
            value = str(restr_attrs.get("Value", "")).strip()
            type_str = str(restr_attrs.get("type", "")).strip()

            if value in ("+", "-") and type_str:
                restriction = SelectionalRestriction(
                    value=cast(RestrictionValue, value),
                    type=cast(SelectionalRestrictionType, type_str),
                )
                restrictions.append(restriction)

        # Parse nested SELRESTRS elements (for complex logic)
        for nested_elem in element.findall("SELRESTRS"):
            nested_restrictions = self._parse_selectional_restrictions(nested_elem)
            restrictions.append(nested_restrictions)

        return SelectionalRestrictions(
            logic=logic_value,
            restrictions=restrictions,
        )

    def _parse_frames(self, element: etree._Element) -> list[VNFrame]:
        """Parse FRAMES element into list of VNFrame models.

        Parameters
        ----------
        element : etree._Element
            VNCLASS or SUBCLASS element.

        Returns
        -------
        list[VNFrame]
            List of parsed VNFrame models.
        """
        frames: list[VNFrame] = []

        frames_elem = element.find("FRAMES")
        if frames_elem is None:
            return frames

        for frame_elem in frames_elem.findall("FRAME"):
            # Parse frame description
            description_elem = frame_elem.find("DESCRIPTION")
            description = None
            if description_elem is not None:
                description = self._parse_frame_description(description_elem)

            # Parse examples
            examples = []
            for example_elem in frame_elem.findall("EXAMPLES/EXAMPLE"):
                example_text = example_elem.text
                if example_text:
                    examples.append(Example(text=example_text.strip()))

            # Parse syntax
            syntax = None
            syntax_elem = frame_elem.find("SYNTAX")
            if syntax_elem is not None:
                syntax = self._parse_syntax(syntax_elem)

            # Parse semantics
            semantics = None
            semantics_elem = frame_elem.find("SEMANTICS")
            if semantics_elem is not None:
                semantics = self._parse_semantics(semantics_elem)

            if description and syntax and semantics:
                frame = VNFrame(
                    description=description,
                    examples=examples,
                    syntax=syntax,
                    semantics=semantics,
                )
                frames.append(frame)

        return frames

    def _parse_frame_description(self, element: etree._Element) -> FrameDescription:
        """Parse DESCRIPTION element into FrameDescription model.

        Parameters
        ----------
        element : etree._Element
            DESCRIPTION XML element.

        Returns
        -------
        FrameDescription
            Parsed frame description.
        """
        attrs = parse_attributes(element)

        description_number = str(attrs.get("descriptionNumber", "0.0")).strip()
        primary = str(attrs.get("primary", "")).strip()
        secondary = str(attrs.get("secondary", "")).strip()
        xtag = str(attrs.get("xtag", "")).strip()

        # Validate description number format
        if not re.match(DESCRIPTION_NUMBER_PATTERN, description_number):
            description_number = "0.0"

        # Parse primary pattern into elements
        primary_elements = primary.split() if primary else []

        # Parse secondary patterns (semicolon-separated)
        secondary_patterns = []
        if secondary:
            if ";" in secondary:
                secondary_patterns = [p.strip() for p in secondary.split(";")]
            else:
                secondary_patterns = [secondary.strip()]

        return FrameDescription(
            description_number=description_number,
            primary=primary,
            secondary=secondary,
            xtag=xtag,
            primary_elements=primary_elements,
            secondary_patterns=secondary_patterns,
        )

    def _parse_syntax(self, element: etree._Element) -> Syntax:
        """Parse SYNTAX element into Syntax model.

        Parameters
        ----------
        element : etree._Element
            SYNTAX XML element.

        Returns
        -------
        Syntax
            Parsed syntax structure.
        """
        elements = []

        for child in element:
            if child.tag in ("NP", "VERB", "PREP", "ADV", "ADJ", "LEX", "ADVP", "S", "SBAR"):
                syntax_elem = self._parse_syntax_element(child)
                if syntax_elem:
                    elements.append(syntax_elem)

        return Syntax(elements=elements)

    def _parse_syntax_element(self, element: etree._Element) -> SyntaxElement | None:
        """Parse a syntax element (NP, VERB, PREP, etc.).

        Parameters
        ----------
        element : etree._Element
            Syntax element.

        Returns
        -------
        SyntaxElement | None
            Parsed syntax element or None if invalid.
        """
        tag = element.tag
        attrs = parse_attributes(element)

        # Get value (role name or preposition values)
        value = str(attrs.get("value", "")).strip()
        if not value and element.text:
            value = element.text.strip()

        # Parse syntactic restrictions
        synrestrs = []
        synrestrs_elem = element.find("SYNRESTRS")
        if synrestrs_elem is not None:
            for synrestr_elem in synrestrs_elem.findall("SYNRESTR"):
                restr_attrs = parse_attributes(synrestr_elem)
                restr_value = str(restr_attrs.get("Value", "")).strip()
                restr_type = str(restr_attrs.get("type", "")).strip()

                if restr_value in ("+", "-") and restr_type:
                    synrestr = SyntacticRestriction(
                        value=cast(RestrictionValue, restr_value),
                        type=cast(SyntacticRestrictionType, restr_type),
                    )
                    synrestrs.append(synrestr)

        # Parse selectional restrictions (for PREP elements)
        selrestrs = []
        selrestrs_elem = element.find("SELRESTRS")
        if selrestrs_elem is not None:
            for selrestr_elem in selrestrs_elem.findall("SELRESTR"):
                restr_attrs = parse_attributes(selrestr_elem)
                restr_value = str(restr_attrs.get("Value", "")).strip()
                restr_type = str(restr_attrs.get("type", "")).strip()

                if restr_value in ("+", "-") and restr_type:
                    selrestr = SelectionalRestriction(
                        value=cast(RestrictionValue, restr_value),
                        type=cast(SelectionalRestrictionType, restr_type),
                    )
                    selrestrs.append(selrestr)

        return SyntaxElement(
            pos=cast(SyntacticPOS, tag),
            value=value if value else None,
            synrestrs=synrestrs,
            selrestrs=selrestrs,
        )

    def _parse_semantics(self, element: etree._Element) -> Semantics:
        """Parse SEMANTICS element into Semantics model.

        Parameters
        ----------
        element : etree._Element
            SEMANTICS XML element.

        Returns
        -------
        Semantics
            Parsed semantic representation.
        """
        predicates = []

        for pred_elem in element.findall("PRED"):
            predicate = self._parse_predicate(pred_elem)
            if predicate:
                predicates.append(predicate)

        return Semantics(predicates=predicates)

    def _parse_predicate(self, element: etree._Element) -> Predicate | None:
        """Parse PRED element into Predicate model.

        Parameters
        ----------
        element : etree._Element
            PRED XML element.

        Returns
        -------
        Predicate | None
            Parsed predicate or None if invalid.
        """
        attrs = parse_attributes(element)
        value = str(attrs.get("value", "")).strip()
        bool_attr = str(attrs.get("bool", "")).strip()

        if not value:
            return None

        # Parse negation
        negated = bool_attr == "!"

        # Parse arguments
        args = []
        for arg_elem in element.findall("ARGS/ARG"):
            arg = self._parse_predicate_argument(arg_elem)
            if arg:
                args.append(arg)

        return Predicate(
            value=cast(PredicateType, value),
            args=args,
            negated=negated,
        )

    def _parse_predicate_argument(self, element: etree._Element) -> PredicateArgument | None:
        """Parse ARG element into PredicateArgument model.

        Parameters
        ----------
        element : etree._Element
            ARG XML element.

        Returns
        -------
        PredicateArgument | None
            Parsed predicate argument or None if invalid.
        """
        attrs = parse_attributes(element)
        arg_type = str(attrs.get("type", "")).strip()
        value = str(attrs.get("value", "")).strip()

        if not arg_type or not value:
            return None

        # Validate argument type
        valid_types = ("ThemRole", "Event", "VerbSpecific", "PredSpecific", "Constant")
        if arg_type not in valid_types:
            return None

        return PredicateArgument(type=cast(ArgumentType, arg_type), value=value)

    def _parse_subclasses(self, element: etree._Element, parent_id: VerbClassID) -> list[VerbClass]:
        """Parse SUBCLASSES element into list of VerbClass models.

        Parameters
        ----------
        element : etree._Element
            VNCLASS or SUBCLASS element.
        parent_id : VerbClassID
            Parent class ID.

        Returns
        -------
        list[VerbClass]
            List of parsed subclass models.
        """
        subclasses: list[VerbClass] = []

        subclasses_elem = element.find("SUBCLASSES")
        if subclasses_elem is None:
            return subclasses

        for subclass_elem in subclasses_elem.findall("VNSUBCLASS"):
            subclass = self.parse_verb_class(subclass_elem, parent_id)
            subclasses.append(subclass)

        return subclasses


def convert_verbnet_file(filepath: Path | str) -> VerbClass:
    """Convert a single VerbNet XML file to VerbClass model.

    Parameters
    ----------
    filepath : Path | str
        Path to VerbNet XML file.

    Returns
    -------
    VerbClass
        Parsed VerbClass model.
    """
    converter = VerbNetConverter()
    return converter.convert_verbnet_file(filepath)


def convert_verbnet_directory(input_dir: Path | str, output_file: Path | str) -> int:
    """Convert all VerbNet XML files in a directory to JSON Lines.

    Parameters
    ----------
    input_dir : Path | str
        Directory containing VerbNet XML files.
    output_file : Path | str
        Output JSON Lines file path.

    Returns
    -------
    int
        Number of files processed.
    """
    converter = VerbNetConverter()
    return converter.convert_verbnet_directory(input_dir, output_file)
