"""PropBank XML to JSON Lines converter.

This module provides conversion from PropBank XML format to JSON Lines format
using the glazing PropBank models.

Classes
-------
PropBankConverter
    Convert PropBank XML files to JSON Lines format.

Functions
---------
convert_frameset_file
    Convert a single frameset XML file to Frameset model.
convert_framesets_directory
    Convert all frameset files in a directory to JSON Lines.

Examples
--------
>>> from pathlib import Path
>>> from glazing.propbank.converter import PropBankConverter
>>> converter = PropBankConverter()
>>> frameset = converter.convert_frameset_file("frames/abandon.xml")
>>> print(frameset.predicate_lemma)
'abandon'
"""

from __future__ import annotations

import contextlib
import tempfile
from io import BytesIO
from pathlib import Path

from lxml import etree
from pydantic import ValidationError

from glazing.propbank.models import (
    Alias,
    Aliases,
    Arg,
    ArgAlias,
    Example,
    Frameset,
    LexLink,
    PropBankAnnotation,
    Rel,
    Role,
    RoleLink,
    Roleset,
    Usage,
    UsageNotes,
)
from glazing.utils.special_cases import SpecialCaseRegistry
from glazing.utils.xml_parser import parse_attributes, parse_with_schema


class PropBankConverter:
    """Convert PropBank XML files to JSON Lines format.

    Parameters
    ----------
    validate_schema : bool, default=False
        Whether to validate against DTD.

    Attributes
    ----------
    validate_schema : bool
        Whether to validate XML against schema.

    Methods
    -------
    convert_frameset_file(filepath)
        Convert a frameset XML file to Frameset model.
    convert_framesets_directory(input_dir, output_file)
        Convert all framesets in a directory to JSON Lines.
    """

    def __init__(self, validate_schema: bool = False) -> None:
        """Initialize the converter.

        Parameters
        ----------
        validate_schema : bool
            Whether to validate XML against DTD.
        """
        self.validate_schema = validate_schema

    def _parse_rolelink(self, rolelink_elem: etree._Element) -> RoleLink:
        """Parse a rolelink element.

        Parameters
        ----------
        rolelink_elem : etree._Element
            The rolelink element.

        Returns
        -------
        RoleLink
            Parsed rolelink.
        """
        attrs = parse_attributes(rolelink_elem)

        return RoleLink(
            class_name=str(attrs.get("class", "")),
            resource=str(attrs.get("resource", "VerbNet")),  # type: ignore[arg-type]
            version=str(attrs.get("version", "")),
            role=rolelink_elem.text or None,
        )

    def _parse_role(self, role_elem: etree._Element) -> Role:
        """Parse a role element.

        Parameters
        ----------
        role_elem : etree._Element
            The role element.

        Returns
        -------
        Role
            Parsed role.
        """
        attrs = parse_attributes(role_elem)

        # Parse rolelinks
        rolelinks = []
        rolelinks_elem = role_elem.find("rolelinks")
        if rolelinks_elem is not None:
            for rolelink in rolelinks_elem.findall("rolelink"):
                rolelinks.append(self._parse_rolelink(rolelink))

        return Role(
            n=str(attrs.get("n", "0")),  # type: ignore[arg-type]
            f=str(attrs.get("f", "")),  # type: ignore[arg-type]
            descr=str(attrs.get("descr", "")),
            rolelinks=rolelinks,
        )

    def _parse_alias(self, alias_elem: etree._Element) -> Alias | ArgAlias:
        """Parse an alias element.

        Parameters
        ----------
        alias_elem : etree._Element
            The alias element.

        Returns
        -------
        Alias | ArgAlias
            Parsed alias.
        """
        attrs = parse_attributes(alias_elem)

        # Check if it's an argalias (has 'arg' attribute)
        if "arg" in attrs:
            return ArgAlias(
                text=alias_elem.text or "",
                pos=str(attrs.get("pos", "v")),  # type: ignore[arg-type]
                arg=str(attrs["arg"]),
            )
        return Alias(text=alias_elem.text or "", pos=str(attrs.get("pos", "v")))  # type: ignore[arg-type]

    def _parse_aliases(self, aliases_elem: etree._Element | None) -> Aliases | None:
        """Parse aliases element.

        Parameters
        ----------
        aliases_elem : etree._Element | None
            The aliases element.

        Returns
        -------
        Aliases | None
            Parsed aliases or None.
        """
        if aliases_elem is None:
            return None

        regular_aliases = []
        arg_aliases = []

        for alias in aliases_elem.findall("alias"):
            parsed = self._parse_alias(alias)
            if isinstance(parsed, ArgAlias):
                arg_aliases.append(parsed)
            else:
                regular_aliases.append(parsed)

        for argalias in aliases_elem.findall("argalias"):
            parsed = self._parse_alias(argalias)
            if isinstance(parsed, ArgAlias):
                arg_aliases.append(parsed)

        return Aliases(alias=regular_aliases, argalias=arg_aliases)

    def _parse_usage(self, usage_elem: etree._Element) -> Usage:
        """Parse a usage element.

        Parameters
        ----------
        usage_elem : etree._Element
            The usage element.

        Returns
        -------
        Usage
            Parsed usage.
        """
        attrs = parse_attributes(usage_elem)

        return Usage(
            resource=str(attrs.get("resource", "PropBank")),  # type: ignore[arg-type]
            version=str(attrs.get("version", "")),
            inuse=str(attrs.get("inuse", "+")),  # type: ignore[arg-type]
        )

    def _parse_lexlink(self, lexlink_elem: etree._Element) -> LexLink:
        """Parse a lexlink element.

        Parameters
        ----------
        lexlink_elem : etree._Element
            The lexlink element.

        Returns
        -------
        LexLink
            Parsed lexlink.
        """
        attrs = parse_attributes(lexlink_elem, {"confidence": float})

        return LexLink(
            class_name=str(attrs.get("class", "")),
            confidence=float(attrs.get("confidence", 0.0)),
            resource=str(attrs.get("resource", "VerbNet")),  # type: ignore[arg-type]
            version=str(attrs.get("version", "")),
            src=str(attrs.get("src", "manual")),  # type: ignore[arg-type]
        )

    def _parse_propbank_annotation(self, propbank_elem: etree._Element) -> PropBankAnnotation:
        """Parse PropBank annotation from example.

        Parameters
        ----------
        propbank_elem : etree._Element
            The propbank element.

        Returns
        -------
        PropBankAnnotation
            Parsed annotation.
        """
        args = []
        rels = []

        # Parse rel elements
        for rel_elem in propbank_elem.findall("rel"):
            attrs = parse_attributes(rel_elem)
            rels.append(Rel(relloc=str(attrs.get("relloc", "0")), text=rel_elem.text or ""))

        # Parse arg elements
        for arg in propbank_elem.findall("arg"):
            # Don't auto-convert start/end to int since they can be "?"
            attrs = parse_attributes(arg)
            # Get type - if missing, raise error
            if "type" not in attrs:
                raise ValueError("Missing required 'type' attribute in arg element")
            if "start" not in attrs:
                raise ValueError("Missing required 'start' attribute in arg element")
            if "end" not in attrs:
                raise ValueError("Missing required 'end' attribute in arg element")

            # Handle start/end which can be int or "?"
            start = attrs.get("start", 0)
            end = attrs.get("end", 0)

            # Convert to int if possible, otherwise keep as string (for "?")
            with contextlib.suppress(ValueError, TypeError):
                start = int(start)

            with contextlib.suppress(ValueError, TypeError):
                end = int(end)

            try:
                args.append(
                    Arg(
                        type=str(attrs["type"]),  # type: ignore[arg-type]
                        start=start,  # type: ignore[arg-type]
                        end=end,  # type: ignore[arg-type]
                        text=arg.text,
                    )
                )
            except ValidationError:
                # Skip args with non-standard types (e.g., AMR annotations)
                continue

        # PropBankAnnotation expects a single Rel, not a list
        # Handle missing rel element (some annotations don't have it)
        rel_result: Rel | None
        if not rels:
            rel_result = None
        elif len(rels) > 1:
            rel_count = len(rels)
            error_msg = (
                f"Multiple 'rel' elements found ({rel_count}). "
                "PropBankAnnotation expects exactly one rel element."
            )
            raise ValueError(error_msg)
        else:
            rel_result = rels[0]

        return PropBankAnnotation(args=args, rel=rel_result)

    def _parse_example(self, example_elem: etree._Element) -> Example:
        """Parse an example element.

        Parameters
        ----------
        example_elem : etree._Element
            The example element.

        Returns
        -------
        Example
            Parsed example.
        """
        attrs = parse_attributes(example_elem)

        # Get text
        text_elem = example_elem.find("text")
        text = text_elem.text or "" if text_elem is not None else ""

        # Parse PropBank annotation
        propbank_elem = example_elem.find("propbank")
        propbank = (
            self._parse_propbank_annotation(propbank_elem) if propbank_elem is not None else None
        )

        return Example(
            name=str(attrs.get("name", "")),
            src=str(attrs.get("src", "")),
            text=text,
            propbank=propbank,
        )

    def _parse_roleset(self, roleset_elem: etree._Element) -> Roleset:
        """Parse a roleset element.

        Parameters
        ----------
        roleset_elem : etree._Element
            The roleset element.

        Returns
        -------
        Roleset
            Parsed roleset.
        """
        attrs = parse_attributes(roleset_elem)

        # Parse aliases
        aliases = self._parse_aliases(roleset_elem.find("aliases"))

        # Parse roles
        roles = []
        roles_elem = roleset_elem.find("roles")
        if roles_elem is not None:
            for role in roles_elem.findall("role"):
                roles.append(self._parse_role(role))

        # Parse usage notes
        usagenotes = None
        usagenotes_elem = roleset_elem.find("usagenotes")
        if usagenotes_elem is not None:
            usage_list = []
            for usage in usagenotes_elem.findall("usage"):
                usage_list.append(self._parse_usage(usage))
            usagenotes = UsageNotes(usage=usage_list)

        # Parse lexlinks
        lexlinks = []
        lexlinks_elem = roleset_elem.find("lexlinks")
        if lexlinks_elem is not None:
            for lexlink in lexlinks_elem.findall("lexlink"):
                lexlinks.append(self._parse_lexlink(lexlink))

        # Parse examples
        examples = []
        for example in roleset_elem.findall("example"):
            examples.append(self._parse_example(example))

        # Parse notes
        notes = []
        for note in roleset_elem.findall("note"):
            if note.text:
                notes.append(note.text)

        return Roleset(
            id=str(attrs.get("id", "")),
            name=str(attrs.get("name")) if attrs.get("name") else None,
            aliases=aliases,
            roles=roles,
            usagenotes=usagenotes,
            lexlinks=lexlinks,
            examples=examples,
            notes=notes,
        )

    def _fix_xml_errors(self, xml_content: str, filepath: Path) -> str:
        """Pre-process XML to fix common syntax errors.

        Parameters
        ----------
        xml_content : str
            Raw XML content.
        filepath : Path
            Path to the XML file (for specific fixes).

        Returns
        -------
        str
            Fixed XML content.
        """
        # Use the special case registry for XML fixes
        return SpecialCaseRegistry.fix_propbank_xml(xml_content, filepath.name)

    def convert_frameset_file(self, filepath: Path | str) -> Frameset:
        """Convert a frameset XML file to Frameset model.

        Parameters
        ----------
        filepath : Path | str
            Path to frameset XML file.

        Returns
        -------
        Frameset
            Parsed Frameset model instance.

        Examples
        --------
        >>> converter = PropBankConverter()
        >>> frameset = converter.convert_frameset_file("frames/abandon.xml")
        >>> print(f"Predicate: {frameset.predicate_lemma}")
        'Predicate: abandon'
        """
        filepath = Path(filepath)

        # Read and pre-process XML content
        xml_content = filepath.read_text(encoding="utf-8")

        # Fix known XML errors
        xml_content = self._fix_xml_errors(xml_content, filepath)

        # Parse XML
        if self.validate_schema:
            # For schema validation, we need to write the fixed content to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xml", encoding="utf-8", delete=False
            ) as f:
                f.write(xml_content)
                temp_path = f.name
            try:
                root = parse_with_schema(temp_path, schema_type="dtd")
            finally:
                Path(temp_path).unlink(missing_ok=True)
        else:
            tree = etree.parse(BytesIO(xml_content.encode("utf-8")))
            root = tree.getroot()

        # Get predicate element
        predicate_elem = root.find("predicate")
        if predicate_elem is None:
            error_msg = f"No predicate element found in {filepath}"
            raise ValueError(error_msg)

        # Get predicate lemma
        predicate_lemma = predicate_elem.get("lemma", "")

        # Parse rolesets
        rolesets = []
        for roleset in predicate_elem.findall("roleset"):
            rolesets.append(self._parse_roleset(roleset))

        # Parse notes
        notes = []
        for note in root.findall("note"):
            if note.text:
                notes.append(note.text)

        return Frameset(predicate_lemma=predicate_lemma, rolesets=rolesets, notes=notes)

    def convert_combined_frameset_file(self, filepath: Path | str) -> list[Frameset]:
        """Convert a combined frameset XML file with multiple predicates.

        Handles files like AMR-UMR-91-rolesets.xml where a single <frameset>
        root contains multiple <predicate> children.

        Parameters
        ----------
        filepath : Path | str
            Path to combined frameset XML file.

        Returns
        -------
        list[Frameset]
            List of parsed Frameset model instances, one per predicate.
        """
        filepath = Path(filepath)
        xml_content = filepath.read_text(encoding="utf-8")
        xml_content = self._fix_xml_errors(xml_content, filepath)

        tree = etree.parse(BytesIO(xml_content.encode("utf-8")))
        root = tree.getroot()

        framesets: list[Frameset] = []
        for predicate_elem in root.findall("predicate"):
            predicate_lemma = predicate_elem.get("lemma", "")

            rolesets = []
            for roleset in predicate_elem.findall("roleset"):
                try:
                    rolesets.append(self._parse_roleset(roleset))
                except (ValidationError, ValueError, TypeError):
                    # Skip rolesets with non-standard values (e.g., AMR-specific types)
                    continue

            notes = []
            for note in predicate_elem.findall("note"):
                if note.text:
                    notes.append(note.text)

            framesets.append(
                Frameset(predicate_lemma=predicate_lemma, rolesets=rolesets, notes=notes)
            )

        return framesets

    def convert_framesets_directory(
        self,
        input_dir: Path | str,
        output_file: Path | str,
        pattern: str = "*.xml",
    ) -> int:
        """Convert all frameset files in a directory to JSON Lines.

        Also processes combined frameset files (e.g., AMR-UMR-91-rolesets.xml)
        found in the parent directory.

        Parameters
        ----------
        input_dir : Path | str
            Directory containing frameset XML files.
        output_file : Path | str
            Output JSON Lines file path.
        pattern : str, default="*.xml"
            File pattern to match.

        Returns
        -------
        int
            Number of framesets converted.

        Examples
        --------
        >>> converter = PropBankConverter()
        >>> count = converter.convert_framesets_directory(
        ...     "propbank-frames/frames",
        ...     "framesets.jsonl"
        ... )
        >>> print(f"Converted {count} framesets")
        'Converted 5559 framesets'
        """
        input_dir = Path(input_dir)
        output_file = Path(output_file)

        count = 0
        errors: list[tuple[Path, Exception]] = []

        with output_file.open("w", encoding="utf-8") as f:
            # Convert individual frameset files
            for xml_file in sorted(input_dir.glob(pattern)):
                try:
                    frameset = self.convert_frameset_file(xml_file)
                    json_line = frameset.model_dump_json(exclude_none=True)
                    f.write(json_line + "\n")
                    count += 1
                except (etree.XMLSyntaxError, ValueError, TypeError) as e:
                    errors.append((xml_file, e))

            # Also process combined frameset files in parent directory
            amr_file = input_dir.parent / "AMR-UMR-91-rolesets.xml"
            if amr_file.exists():
                try:
                    amr_framesets = self.convert_combined_frameset_file(amr_file)
                    for frameset in amr_framesets:
                        json_line = frameset.model_dump_json(exclude_none=True)
                        f.write(json_line + "\n")
                        count += 1
                except (etree.XMLSyntaxError, ValueError, TypeError) as e:
                    errors.append((amr_file, e))

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
def convert_frameset_file(filepath: Path | str) -> Frameset:
    """Convert a single frameset XML file to Frameset model.

    Parameters
    ----------
    filepath : Path | str
        Path to frameset XML file.

    Returns
    -------
    Frameset
        Parsed Frameset model.
    """
    converter = PropBankConverter()
    return converter.convert_frameset_file(filepath)


def convert_framesets_directory(
    input_dir: Path | str,
    output_file: Path | str,
    pattern: str = "*.xml",
) -> int:
    """Convert all framesets in a directory to JSON Lines.

    Parameters
    ----------
    input_dir : Path | str
        Directory with frameset XML files.
    output_file : Path | str
        Output JSON Lines file.
    pattern : str
        File pattern to match.

    Returns
    -------
    int
        Number of framesets converted.
    """
    converter = PropBankConverter()
    return converter.convert_framesets_directory(input_dir, output_file, pattern)
