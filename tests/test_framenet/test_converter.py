"""Tests for FrameNet XML to JSON Lines converter.

Tests the FrameNetConverter class using real FrameNet XML data.
"""

import json
from pathlib import Path

import pytest

from glazing.framenet.converter import FrameNetConverter, convert_frame_file


class TestFrameNetConverter:
    """Test FrameNetConverter class."""

    def test_convert_frame_file(self, framenet_frame_xml):
        """Test converting a frame XML file."""
        converter = FrameNetConverter()
        frame = converter.convert_frame_file(framenet_frame_xml)

        # Check frame attributes
        assert frame.id == 2031
        assert frame.name == "Abandonment"
        assert "Agent" in frame.definition.raw_text
        assert "Theme" in frame.definition.raw_text

        # Check frame elements
        assert len(frame.frame_elements) == 5
        fe_names = [fe.name for fe in frame.frame_elements]
        assert "Agent" in fe_names
        assert "Theme" in fe_names
        assert "Place" in fe_names
        assert "Time" in fe_names
        assert "Manner" in fe_names

        # Check core elements
        core_elements = frame.get_core_elements()
        assert len(core_elements) == 2
        core_names = [fe.name for fe in core_elements]
        assert "Agent" in core_names
        assert "Theme" in core_names

    def test_parse_frame_element(self, framenet_frame_xml):
        """Test parsing individual frame elements."""
        converter = FrameNetConverter()
        frame = converter.convert_frame_file(framenet_frame_xml)

        # Check Agent FE
        agent = frame.get_fe_by_name("Agent")
        assert agent is not None
        assert agent.id == 12338
        assert agent.abbrev == "Age"
        assert agent.core_type == "Core"
        assert agent.bg_color == "FF0000"
        assert agent.fg_color == "FFFFFF"
        assert "Agent" in agent.definition.raw_text

        # Check Theme FE
        theme = frame.get_fe_by_name("Theme")
        assert theme is not None
        assert theme.id == 12339
        assert theme.abbrev == "The"
        assert theme.core_type == "Core"
        assert theme.bg_color == "0000FF"
        assert theme.fg_color == "FFFFFF"

    def test_parse_definition_with_markup(self, framenet_frame_xml):
        """Test parsing definition with embedded markup."""
        converter = FrameNetConverter()
        frame = converter.convert_frame_file(framenet_frame_xml)

        # Check that definition was parsed
        assert frame.definition.plain_text
        assert frame.definition.annotations

        # Check for FE references in definition
        fe_refs = frame.definition.get_fe_references()
        assert len(fe_refs) > 0

        # Check that FE names are found
        fe_names = [ref.name for ref in fe_refs if ref.name]
        assert "Agent" in fe_names or any("Agent" in (ref.text or "") for ref in fe_refs)
        assert "Theme" in fe_names or any("Theme" in (ref.text or "") for ref in fe_refs)

    def test_convert_real_abandonment_frame(self):
        """Test converting the real Abandonment.xml frame file if available."""
        frame_file = Path("framenet_v17/frame/Abandonment.xml")
        if not frame_file.exists():
            pytest.skip("Real FrameNet data not available")

        converter = FrameNetConverter()
        frame = converter.convert_frame_file(frame_file)

        # Verify against known Abandonment frame structure
        assert frame.id == 2031
        assert frame.name == "Abandonment"
        assert len(frame.frame_elements) == 12  # Real frame has 12 FEs

        # Check some known FEs
        fe_names = [fe.name for fe in frame.frame_elements]
        assert "Agent" in fe_names
        assert "Theme" in fe_names
        assert "Place" in fe_names
        assert "Time" in fe_names
        assert "Manner" in fe_names
        assert "Duration" in fe_names
        assert "Explanation" in fe_names
        assert "Depictive" in fe_names
        assert "Degree" in fe_names
        assert "Means" in fe_names
        assert "Purpose" in fe_names
        assert "Event_description" in fe_names

    def test_json_serialization(self, framenet_frame_xml):
        """Test that converted frame can be serialized to JSON."""
        converter = FrameNetConverter()
        frame = converter.convert_frame_file(framenet_frame_xml)

        # Serialize to JSON
        json_str = frame.model_dump_json(exclude_none=True)
        assert json_str

        # Parse back and verify
        data = json.loads(json_str)
        assert data["id"] == 2031
        assert data["name"] == "Abandonment"
        assert len(data["frame_elements"]) == 5

    def test_convert_frames_directory(self, tmp_path, framenet_frame_xml):
        """Test converting a directory of frame files."""
        # Create test directory with frame files
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        # Use the fixture's frame file content
        test_frame_content = framenet_frame_xml.read_text()

        # Create a couple test files
        for i in range(2):
            frame_file = frames_dir / f"Frame{i}.xml"
            frame_file.write_text(test_frame_content)

        # Convert directory
        output_file = tmp_path / "frames.jsonl"
        converter = FrameNetConverter()
        count = converter.convert_frames_directory(frames_dir, output_file)

        assert count == 2
        assert output_file.exists()

        # Verify output
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            data = json.loads(line)
            assert data["id"] == 2031
            assert data["name"] == "Abandonment"

    def test_convenience_function(self, framenet_frame_xml):
        """Test the convenience function."""
        frame = convert_frame_file(framenet_frame_xml)

        assert frame.id == 2031
        assert frame.name == "Abandonment"
        assert len(frame.frame_elements) == 5


class TestLexicalUnitParsing:
    """Test lexical unit parsing from luIndex.xml."""

    def test_parse_lu_from_index_basic(self, tmp_path):
        """Test parsing basic LU from luIndex element."""
        lu_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <luIndex xmlns="http://framenet.icsi.berkeley.edu">
            <lu numAnnotInstances="11" hasAnnotation="true" frameID="2031"
                frameName="Abandonment" status="Finished_Initial"
                name="abandon.v" ID="12345"/>
        </luIndex>"""

        lu_file = tmp_path / "luIndex.xml"
        lu_file.write_text(lu_xml)

        converter = FrameNetConverter()
        lus = converter.convert_lu_index_file(lu_file)

        assert len(lus) == 1
        lu = lus[0]
        assert lu.id == 12345
        assert lu.name == "abandon.v"
        assert lu.pos == "V"
        assert lu.frame_id == 2031
        assert lu.frame_name == "Abandonment"
        assert lu.annotation_status == "Finished_Initial"
        assert lu.has_annotated_examples is True
        assert lu.sentence_count.total == 11
        assert lu.sentence_count.annotated == 11
        assert len(lu.lexemes) == 1
        assert lu.lexemes[0].name == "abandon"
        assert lu.lexemes[0].pos == "V"
        assert lu.lexemes[0].headword is True

    def test_parse_multi_word_lu(self, tmp_path):
        """Test parsing multi-word LU like 'give_up.v'."""
        lu_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <luIndex xmlns="http://framenet.icsi.berkeley.edu">
            <lu numAnnotInstances="5" hasAnnotation="true" frameID="100"
                frameName="Surrender" status="Created"
                name="give_up.v" ID="999"/>
        </luIndex>"""

        lu_file = tmp_path / "luIndex.xml"
        lu_file.write_text(lu_xml)

        converter = FrameNetConverter()
        lus = converter.convert_lu_index_file(lu_file)

        assert len(lus) == 1
        lu = lus[0]
        assert lu.name == "give_up.v"
        assert len(lu.lexemes) == 2
        assert lu.lexemes[0].name == "give"
        assert lu.lexemes[0].headword is True
        assert lu.lexemes[0].order == 1
        assert lu.lexemes[1].name == "up"
        assert lu.lexemes[1].headword is False
        assert lu.lexemes[1].order == 2

    def test_parse_lu_different_pos(self, tmp_path):
        """Test parsing LUs with different parts of speech."""
        lu_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <luIndex xmlns="http://framenet.icsi.berkeley.edu">
            <lu numAnnotInstances="0" hasAnnotation="false" frameID="1"
                frameName="Test" status="Created" name="test.n" ID="1"/>
            <lu numAnnotInstances="0" hasAnnotation="false" frameID="1"
                frameName="Test" status="Created" name="test.a" ID="2"/>
            <lu numAnnotInstances="0" hasAnnotation="false" frameID="1"
                frameName="Test" status="Created" name="quickly.adv" ID="3"/>
        </luIndex>"""

        lu_file = tmp_path / "luIndex.xml"
        lu_file.write_text(lu_xml)

        converter = FrameNetConverter()
        lus = converter.convert_lu_index_file(lu_file)

        assert len(lus) == 3
        assert lus[0].pos == "N"
        assert lus[1].pos == "A"
        assert lus[2].pos == "ADV"

    def test_convert_frames_with_lus(self, tmp_path):
        """Test converting frames directory with LU association."""
        # Create frame directory structure
        frames_dir = tmp_path / "frame"
        frames_dir.mkdir()

        # Create a test frame XML
        frame_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <frame cBy="Test" cDate="01/01/2000 00:00:00 PST Mon" name="TestFrame" ID="100"
               xmlns="http://framenet.icsi.berkeley.edu">
            <definition>A test frame.</definition>
            <FE name="Agent" ID="1" abbrev="Agt" coreType="Core" bgColor="FF0000" fgColor="FFFFFF">
                <definition>The agent.</definition>
            </FE>
        </frame>"""

        frame_file = frames_dir / "TestFrame.xml"
        frame_file.write_text(frame_xml)

        # Create luIndex.xml in parent directory
        lu_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <luIndex xmlns="http://framenet.icsi.berkeley.edu">
            <lu numAnnotInstances="5" hasAnnotation="true" frameID="100"
                frameName="TestFrame" status="Finished_Initial"
                name="test.v" ID="1000"/>
            <lu numAnnotInstances="3" hasAnnotation="true" frameID="100"
                frameName="TestFrame" status="Created"
                name="examine.v" ID="1001"/>
            <lu numAnnotInstances="0" hasAnnotation="false" frameID="999"
                frameName="OtherFrame" status="Created"
                name="other.v" ID="2000"/>
        </luIndex>"""

        lu_index_file = tmp_path / "luIndex.xml"
        lu_index_file.write_text(lu_xml)

        # Convert
        output_file = tmp_path / "output.jsonl"
        converter = FrameNetConverter()
        count = converter.convert_frames_directory(frames_dir, output_file)

        assert count == 1

        # Verify output
        with output_file.open("r") as f:
            data = json.loads(f.readline())

        assert data["id"] == 100
        assert data["name"] == "TestFrame"
        assert "lexical_units" in data
        assert len(data["lexical_units"]) == 2

        lu_names = [lu["name"] for lu in data["lexical_units"]]
        assert "test.v" in lu_names
        assert "examine.v" in lu_names
        assert "other.v" not in lu_names  # Different frame

    def test_lu_index_missing_no_crash(self, tmp_path):
        """Test that conversion works even if luIndex.xml is missing."""
        frames_dir = tmp_path / "frame"
        frames_dir.mkdir()

        frame_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <frame cBy="Test" cDate="01/01/2000 00:00:00 PST Mon" name="TestFrame" ID="100"
               xmlns="http://framenet.icsi.berkeley.edu">
            <definition>A test frame.</definition>
        </frame>"""

        frame_file = frames_dir / "TestFrame.xml"
        frame_file.write_text(frame_xml)

        # No luIndex.xml created

        output_file = tmp_path / "output.jsonl"
        converter = FrameNetConverter()
        count = converter.convert_frames_directory(frames_dir, output_file)

        assert count == 1

        # Frame should have empty lexical_units list
        with output_file.open("r") as f:
            data = json.loads(f.readline())

        assert data["id"] == 100
        assert "lexical_units" in data
        assert len(data["lexical_units"]) == 0


class TestFEConstraints:
    """Test FE Requires/Excludes constraint parsing (issue #7)."""

    # Frame XML in the FrameNet default namespace, mirroring the real
    # Personal_relationship frame: requiresFE/excludesFE are self-closing
    # siblings of <FE> whose target name lives in the ``name`` attribute.
    FRAME_XML = """<?xml version="1.0" encoding="UTF-8"?>
    <frame cBy="JKR" cDate="04/30/2001 12:55:11 PDT Mon" name="Personal_relationship" ID="102"
           xmlns="http://framenet.icsi.berkeley.edu">
        <definition>A frame about relationships.</definition>
        <FE bgColor="FF0000" fgColor="FFFFFF" coreType="Core" abbrev="Partner_1"
            name="Partner_1" ID="486">
            <definition>The first partner.</definition>
            <requiresFE name="Partner_2" ID="488"/>
            <excludesFE name="Partners" ID="489"/>
        </FE>
        <FE bgColor="FF0000" fgColor="FFFFFF" coreType="Core" abbrev="Partner_2"
            name="Partner_2" ID="488">
            <definition>The second partner.</definition>
            <requiresFE name="Partner_1" ID="487"/>
            <excludesFE name="Partners" ID="489"/>
        </FE>
        <FE bgColor="800080" fgColor="FFFFFF" coreType="Core" abbrev="Partners"
            name="Partners" ID="489">
            <definition>Both partners together.</definition>
            <excludesFE name="Partner_1" ID="490"/>
            <excludesFE name="Partner_2" ID="491"/>
        </FE>
        <FE bgColor="0000FF" fgColor="FFFFFF" coreType="Peripheral" abbrev="Time"
            name="Time" ID="492">
            <definition>When the relationship holds.</definition>
        </FE>
    </frame>"""

    @pytest.fixture
    def frame_file(self, tmp_path):
        """Write the constraint-bearing frame XML to a temp file."""
        path = tmp_path / "Personal_relationship.xml"
        path.write_text(self.FRAME_XML, encoding="utf-8")
        return path

    def test_requires_and_excludes_populated(self, frame_file):
        """requiresFE/excludesFE names are captured from the ``name`` attribute."""
        converter = FrameNetConverter()
        frame = converter.convert_frame_file(frame_file)
        by_name = {fe.name: fe for fe in frame.frame_elements}

        assert by_name["Partner_1"].requires_fe == ["Partner_2"]
        assert by_name["Partner_1"].excludes_fe == ["Partners"]
        assert by_name["Partner_2"].requires_fe == ["Partner_1"]
        assert by_name["Partner_2"].excludes_fe == ["Partners"]

    def test_multiple_excludes_on_one_fe(self, frame_file):
        """An FE with several <excludesFE> siblings keeps all of them."""
        converter = FrameNetConverter()
        frame = converter.convert_frame_file(frame_file)
        partners = next(fe for fe in frame.frame_elements if fe.name == "Partners")

        assert partners.excludes_fe == ["Partner_1", "Partner_2"]
        assert partners.requires_fe == []

    def test_fe_without_constraints_is_empty(self, frame_file):
        """An FE with no constraint elements yields empty lists, not None."""
        converter = FrameNetConverter()
        frame = converter.convert_frame_file(frame_file)
        time_fe = next(fe for fe in frame.frame_elements if fe.name == "Time")

        assert time_fe.requires_fe == []
        assert time_fe.excludes_fe == []

    def test_constraints_survive_json_round_trip(self, tmp_path, frame_file):
        """Constraints are present in the serialized JSONL output."""
        converter = FrameNetConverter()
        output_file = tmp_path / "frames.jsonl"
        converter.convert_frames_directory(frame_file.parent, output_file)

        with output_file.open("r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        by_name = {fe["name"]: fe for fe in data["frame_elements"]}
        assert by_name["Partner_1"]["requires_fe"] == ["Partner_2"]
        assert by_name["Partner_1"]["excludes_fe"] == ["Partners"]
        assert by_name["Partners"]["excludes_fe"] == ["Partner_1", "Partner_2"]

    def test_real_personal_relationship_frame(self):
        """Cross-check against the real FN 1.7 frame file if available."""
        frame_path = Path("framenet_v17/frame/Personal_relationship.xml")
        if not frame_path.exists():
            pytest.skip("Real FrameNet data not available")

        converter = FrameNetConverter()
        frame = converter.convert_frame_file(frame_path)
        by_name = {fe.name: fe for fe in frame.frame_elements}

        # At least one FE must carry a non-empty constraint list.
        assert any(fe.requires_fe for fe in frame.frame_elements)
        assert any(fe.excludes_fe for fe in frame.frame_elements)
        assert by_name["Partner_1"].requires_fe == ["Partner_2"]
        assert "Partners" in by_name["Partner_1"].excludes_fe
