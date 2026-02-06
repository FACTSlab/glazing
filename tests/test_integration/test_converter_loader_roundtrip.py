"""Converter-to-loader round-trip integration tests.

Tests that data survives the full pipeline: raw format → converter → JSONL → loader.
Covers all four resources (WordNet, FrameNet, VerbNet, PropBank) plus contract
and field completeness checks.
"""

import json

import pytest

from glazing.framenet.converter import FrameNetConverter
from glazing.framenet.loader import FrameNetLoader
from glazing.propbank.converter import PropBankConverter
from glazing.propbank.loader import PropBankLoader
from glazing.verbnet.converter import VerbNetConverter
from glazing.verbnet.loader import VerbNetLoader
from glazing.wordnet.converter import WordNetConverter
from glazing.wordnet.loader import WordNetLoader

# ── WordNet ────────────────────────────────────────────────────────────────


WN_LICENSE_HEADER = """\
  1 This software and database is being provided to you, the LICENSEE, by
  2 Princeton University under the following license.
  3
  4
  5
"""


class TestWordNetRoundTrip:
    """WordNet converter → JSONL → loader pipeline."""

    @pytest.fixture
    def wordnet_data(self, tmp_path):
        """Create a minimal WordNet database and run the full conversion pipeline."""
        wn_dir = tmp_path / "wn"
        wn_dir.mkdir()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # data.verb (2 synsets, second has 2 verb frames)
        (wn_dir / "data.verb").write_text(
            WN_LICENSE_HEADER
            + "00001740 29 v 01 breathe 0 002 $ 00001740 v 0000 @ 00002084 v 0000 01 + 02 00 | draw air into and expel out of the lungs\n"
            + "00002084 29 v 02 respire 0 breathe 1 001 @ 00001740 v 0000 02 + 01 00 + 02 01 | undergo respiration\n",
            encoding="utf-8",
        )
        # data.noun (1 synset)
        (wn_dir / "data.noun").write_text(
            WN_LICENSE_HEADER
            + "00002325 03 n 01 entity 0 001 ~ 00002684 n 0000 | something having concrete existence\n",
            encoding="utf-8",
        )
        # data.adj / data.adv (empty but present)
        for name in ("data.adj", "data.adv"):
            (wn_dir / name).write_text(WN_LICENSE_HEADER, encoding="utf-8")

        # index.sense  (sense_key synset_offset sense_number tag_count)
        (wn_dir / "index.sense").write_text(
            "breathe%2:29:00:: 00001740 1 25\n"
            "respire%2:29:00:: 00002084 1 3\n"
            "breathe%2:29:01:: 00002084 2 0\n"
            "entity%1:03:00:: 00002325 1 11\n",
            encoding="utf-8",
        )

        # verb.Framestext
        (wn_dir / "verb.Framestext").write_text(
            "1 Something ----s\n2 Somebody ----s\n",
            encoding="utf-8",
        )

        # sents.vrb
        (wn_dir / "sents.vrb").write_text(
            "1 The children %s to the playground\n2 The banks %s the check\n",
            encoding="utf-8",
        )

        # cntlist
        (wn_dir / "cntlist").write_text(
            "25 breathe%2:29:00:: 1\n3 respire%2:29:00:: 1\n11 entity%1:03:00:: 1\n",
            encoding="utf-8",
        )

        # verb.exc
        (wn_dir / "verb.exc").write_text(
            "breathed breathe\nrespired respire\n",
            encoding="utf-8",
        )
        # Create empty exception files for remaining POS categories
        for name in ("noun.exc", "adj.exc", "adv.exc"):
            (wn_dir / name).write_text("", encoding="utf-8")

        # Run conversions
        converter = WordNetConverter()
        stats = converter.convert_wordnet_database(wn_dir, output_dir / "wordnet.jsonl")
        sense_count = converter.convert_sense_index(wn_dir, output_dir / "wordnet_senses.jsonl")
        exc_count = converter.convert_exceptions(wn_dir, output_dir / "wordnet_exceptions.jsonl")

        loader = WordNetLoader(data_path=output_dir / "wordnet.jsonl")

        return {
            "stats": stats,
            "sense_count": sense_count,
            "exc_count": exc_count,
            "loader": loader,
            "output_dir": output_dir,
        }

    def test_synset_count_preserved(self, wordnet_data):
        """Converter counts match loader counts."""
        stats = wordnet_data["stats"]
        wn = wordnet_data["loader"]

        assert stats["synsets_verb"] == 2
        assert stats["synsets_noun"] == 1
        assert stats["total_synsets"] == 3
        assert len(wn.synsets) == 3

    def test_word_enrichment(self, wordnet_data):
        """Words have tag_count and sense_number from cntlist/index.sense."""
        wn = wordnet_data["loader"]

        # breathe in synset 00001740 should have tag_count=25, sense_number=1
        synset = wn.synsets["00001740"]
        breathe_word = synset.words[0]
        assert breathe_word.lemma == "breathe"
        assert breathe_word.tag_count == 25
        assert breathe_word.sense_number == 1

        # entity in synset 00002325 should have tag_count=11
        entity_synset = wn.synsets["00002325"]
        entity_word = entity_synset.words[0]
        assert entity_word.lemma == "entity"
        assert entity_word.tag_count == 11

    def test_verb_frame_templates(self, wordnet_data):
        """VerbFrames have template and example_sentence from verb.Framestext/sents.vrb."""
        wn = wordnet_data["loader"]
        synset = wn.synsets["00001740"]

        assert synset.frames is not None
        assert len(synset.frames) == 1

        frame = synset.frames[0]
        assert frame.frame_number == 2
        assert frame.template == "Somebody ----s"
        assert frame.example_sentence == "The banks %s the check"

    def test_pointers_preserved(self, wordnet_data):
        """Pointer relations survive the round-trip."""
        wn = wordnet_data["loader"]
        synset = wn.synsets["00001740"]

        assert len(synset.pointers) == 2
        symbols = {p.symbol for p in synset.pointers}
        assert "$" in symbols
        assert "@" in symbols

    def test_lemma_index_builds(self, wordnet_data):
        """Lemma index enables word lookups after loading."""
        wn = wordnet_data["loader"]

        assert "breathe" in wn.lemma_index
        assert "v" in wn.lemma_index["breathe"]
        assert "entity" in wn.lemma_index
        assert "n" in wn.lemma_index["entity"]

    def test_sense_index_loads(self, wordnet_data):
        """Sense index populated from supplementary wordnet_senses.jsonl."""
        wn = wordnet_data["loader"]

        assert len(wn.sense_index) == 4
        assert "breathe%2:29:00::" in wn.sense_index
        sense = wn.sense_index["breathe%2:29:00::"]
        assert sense.synset_offset == "00001740"
        assert sense.tag_count == 25

    def test_exceptions_load(self, wordnet_data):
        """Morphological exceptions loaded from supplementary wordnet_exceptions.jsonl."""
        wn = wordnet_data["loader"]

        assert "v" in wn.exceptions
        assert "breathed" in wn.exceptions["v"]
        assert wn.exceptions["v"]["breathed"] == ["breathe"]


# ── FrameNet ──────────────────────────────────────────────────────────────


class TestFrameNetRoundTrip:
    """FrameNet converter → JSONL → loader pipeline."""

    @pytest.fixture
    def framenet_data(self, tmp_path):
        """Create a minimal FrameNet dataset and run the full conversion pipeline."""
        fn_root = tmp_path / "framenet"
        frames_dir = fn_root / "frame"
        lu_dir = fn_root / "lu"
        fulltext_dir = fn_root / "fulltext"
        output_dir = tmp_path / "output"

        frames_dir.mkdir(parents=True)
        lu_dir.mkdir()
        fulltext_dir.mkdir()
        output_dir.mkdir()

        # Frame XML
        (frames_dir / "Giving.xml").write_text(
            """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<frame name="Giving" ID="139"
       xmlns="http://framenet.icsi.berkeley.edu"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <definition>&lt;def-root&gt;A Donor transfers a Theme to a Recipient.&lt;/def-root&gt;</definition>
    <FE bgColor="FF0000" fgColor="FFFFFF" coreType="Core" abbrev="Donor" name="Donor" ID="277">
        <definition>&lt;def-root&gt;The person that gives.&lt;/def-root&gt;</definition>
    </FE>
    <FE bgColor="0000FF" fgColor="FFFFFF" coreType="Core" abbrev="Theme" name="Theme" ID="278">
        <definition>&lt;def-root&gt;The object given.&lt;/def-root&gt;</definition>
    </FE>
    <FE bgColor="00FF00" fgColor="000000" coreType="Core" abbrev="Rec" name="Recipient" ID="279">
        <definition>&lt;def-root&gt;The person receiving.&lt;/def-root&gt;</definition>
    </FE>
    <frameRelation type="Inherits from">
        <relatedFrame ID="230">Transferring</relatedFrame>
    </frameRelation>
    <frameRelation type="Is Inherited by"/>
    <lexUnit status="FN1_Sent" POS="V" name="give.v" ID="614" lemmaID="304">
        <definition>COD: freely transfer the possession of</definition>
        <sentenceCount annotated="20" total="100"/>
        <lexeme order="1" headword="false" breakBefore="false" POS="V" name="give"/>
    </lexUnit>
    <lexUnit status="FN1_Sent" POS="V" name="donate.v" ID="615" lemmaID="305">
        <definition>COD: give to a good cause</definition>
        <sentenceCount annotated="10" total="50"/>
        <lexeme order="1" headword="false" breakBefore="false" POS="V" name="donate"/>
    </lexUnit>
</frame>""",
            encoding="utf-8",
        )

        # luIndex.xml
        (fn_root / "luIndex.xml").write_text(
            """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<luIndex xmlns="http://framenet.icsi.berkeley.edu">
    <lu ID="614" name="give.v" frameID="139" frameName="Giving"
        status="FN1_Sent" hasAnnotation="true" numAnnotInstances="20"/>
    <lu ID="615" name="donate.v" frameID="139" frameName="Giving"
        status="FN1_Sent" hasAnnotation="true" numAnnotInstances="10"/>
</luIndex>""",
            encoding="utf-8",
        )

        # frRelation.xml
        (fn_root / "frRelation.xml").write_text(
            """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<frameRelations xmlns="http://framenet.icsi.berkeley.edu"
                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <frameRelationType name="Inheritance" ID="1"
                       superFrameName="Parent" subFrameName="Child">
        <frameRelation ID="1" subID="139" subFrameName="Giving"
                       supID="230" superFrameName="Transferring">
            <FERelation ID="1" subID="277" subFEName="Donor"
                        supID="500" superFEName="Sender"/>
        </frameRelation>
    </frameRelationType>
</frameRelations>""",
            encoding="utf-8",
        )

        # semTypes.xml
        (fn_root / "semTypes.xml").write_text(
            """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<semTypes xmlns="http://framenet.icsi.berkeley.edu"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <semType name="Physical_entity" ID="68" abbrev="PhysObj">
        <definition>A type for physical entities</definition>
        <superType supID="70" superTypeName="Ontological_type"/>
    </semType>
</semTypes>""",
            encoding="utf-8",
        )

        # Fulltext XML
        (fulltext_dir / "TestDoc.xml").write_text(
            """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<fullTextAnnotation xmlns="http://framenet.icsi.berkeley.edu"
                    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <header>
        <corpus description="Test corpus" name="TestCorpus" ID="1">
            <document description="Test doc" name="TestDoc" ID="1"/>
        </corpus>
    </header>
    <sentence corpID="1" docID="1" sentNo="1" paragNo="1" aPos="0" ID="100">
        <text>He gave her a book.</text>
        <annotationSet ID="200" status="MANUAL" frameName="Giving"
                       frameID="139" luName="give.v" luID="614">
            <layer rank="1" name="Target">
                <label name="Target" start="3" end="6"/>
            </layer>
            <layer rank="1" name="FE">
                <label name="Donor" start="0" end="1"/>
                <label name="Recipient" start="8" end="10"/>
                <label name="Theme" start="12" end="17"/>
            </layer>
        </annotationSet>
    </sentence>
</fullTextAnnotation>""",
            encoding="utf-8",
        )

        # Run conversions
        converter = FrameNetConverter()
        frame_count = converter.convert_frames_directory(frames_dir, output_dir / "framenet.jsonl")
        semtype_count = converter.convert_semtypes_file(
            fn_root / "semTypes.xml", output_dir / "framenet_semtypes.jsonl"
        )
        fulltext_count = converter.convert_fulltext_directory(
            fulltext_dir, output_dir / "framenet_fulltext.jsonl"
        )

        loader = FrameNetLoader(data_path=output_dir / "framenet.jsonl")

        return {
            "loader": loader,
            "frame_count": frame_count,
            "semtype_count": semtype_count,
            "fulltext_count": fulltext_count,
        }

    def test_frame_count_preserved(self, framenet_data):
        """Converter reports 1 frame; loader reads 1 frame."""
        assert framenet_data["frame_count"] == 1
        frames = framenet_data["loader"].frames
        assert len(frames) == 1
        assert frames[0].id == 139
        assert frames[0].name == "Giving"

    def test_frame_elements_preserved(self, framenet_data):
        """FE names and core types survive the round trip."""
        frame = framenet_data["loader"].frames[0]
        fe_names = {fe.name for fe in frame.frame_elements}
        assert fe_names == {"Donor", "Theme", "Recipient"}
        for fe in frame.frame_elements:
            assert fe.core_type == "Core"

    def test_lexical_units_preserved(self, framenet_data):
        """LU names and POS survive the round trip."""
        frame = framenet_data["loader"].frames[0]
        lu_names = {lu.name for lu in frame.lexical_units}
        assert "give.v" in lu_names
        assert "donate.v" in lu_names
        for lu in frame.lexical_units:
            assert lu.pos == "V"

    def test_frame_relations_populated(self, framenet_data):
        """Frame relations from frRelation.xml are attached to the frame."""
        frame = framenet_data["loader"].frames[0]
        inherits = [r for r in frame.frame_relations if r.type == "Inherits from"]
        assert len(inherits) == 1
        rel = inherits[0]
        assert rel.sub_frame_id == 139
        assert rel.super_frame_id == 230
        assert rel.super_frame_name == "Transferring"
        assert len(rel.fe_relations) == 1
        assert rel.fe_relations[0].sub_fe_name == "Donor"
        assert rel.fe_relations[0].super_fe_name == "Sender"

    def test_semantic_types_load(self, framenet_data):
        """Semantic types loaded from supplementary framenet_semtypes.jsonl."""
        sem_types = framenet_data["loader"].load_semantic_types()
        assert len(sem_types) == 1
        assert sem_types[0].id == 68
        assert sem_types[0].name == "Physical_entity"

    def test_fulltext_loads(self, framenet_data):
        """Fulltext sentences loaded from supplementary framenet_fulltext.jsonl."""
        sentences = framenet_data["loader"].load_fulltext()
        assert len(sentences) == 1
        assert sentences[0].id == 100
        assert sentences[0].text == "He gave her a book."
        assert len(sentences[0].annotation_sets) == 1


# ── VerbNet ───────────────────────────────────────────────────────────────


VERBNET_XML = """\
<!DOCTYPE VNCLASS SYSTEM "vn_class-3.dtd">
<VNCLASS ID="give-13.1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:noNamespaceSchemaLocation="vn_schema-3.xsd">
  <MEMBERS>
    <MEMBER fn_mapping="Giving" grouping="give.01 give.02" name="give"
            verbnet_key="give#2" wn="give%2:40:00 give%2:40:01" features=""/>
    <MEMBER fn_mapping="None" grouping="deal.04" name="deal"
            verbnet_key="deal#2" wn="deal%2:40:01" features=""/>
    <MEMBER fn_mapping="None" grouping="" name="loan"
            verbnet_key="loan#1" wn="loan%2:40:00" features=""/>
  </MEMBERS>
  <THEMROLES>
    <THEMROLE type="Agent">
      <SELRESTRS logic="or">
        <SELRESTR Value="+" type="animate"/>
        <SELRESTR Value="+" type="organization"/>
      </SELRESTRS>
    </THEMROLE>
    <THEMROLE type="Theme">
      <SELRESTRS/>
    </THEMROLE>
    <THEMROLE type="Recipient">
      <SELRESTRS>
        <SELRESTR Value="+" type="animate"/>
      </SELRESTRS>
    </THEMROLE>
  </THEMROLES>
  <FRAMES>
    <FRAME>
      <DESCRIPTION descriptionNumber="0.2" primary="NP V NP PP.recipient"
                   secondary="NP-PP; Recipient-PP" xtag=""/>
      <EXAMPLES>
        <EXAMPLE>They lent a bicycle to me.</EXAMPLE>
      </EXAMPLES>
      <SYNTAX>
        <NP value="Agent"><SYNRESTRS/></NP>
        <VERB/>
        <NP value="Theme"><SYNRESTRS/></NP>
        <PREP value="to"><SELRESTRS/></PREP>
        <NP value="Recipient"><SYNRESTRS/></NP>
      </SYNTAX>
      <SEMANTICS>
        <PRED value="transfer">
          <ARGS>
            <ARG type="Event" value="E"/>
            <ARG type="ThemRole" value="Agent"/>
          </ARGS>
        </PRED>
      </SEMANTICS>
    </FRAME>
  </FRAMES>
  <SUBCLASSES>
    <VNSUBCLASS ID="give-13.1-1">
      <MEMBERS>
        <MEMBER fn_mapping="Commerce_sell" grouping="sell.01" name="sell"
                verbnet_key="sell#1" wn="sell%2:40:00" features=""/>
      </MEMBERS>
      <THEMROLES/>
      <FRAMES/>
    </VNSUBCLASS>
  </SUBCLASSES>
</VNCLASS>"""


class TestVerbNetRoundTrip:
    """VerbNet converter → JSONL → loader pipeline."""

    @pytest.fixture
    def verbnet_data(self, tmp_path):
        """Create VerbNet XML and run the conversion pipeline."""
        vn_dir = tmp_path / "verbnet"
        vn_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (vn_dir / "give-13.1.xml").write_text(VERBNET_XML, encoding="utf-8")

        converter = VerbNetConverter()
        count = converter.convert_verbnet_directory(vn_dir, output_dir / "verbnet.jsonl")

        loader = VerbNetLoader(data_path=output_dir / "verbnet.jsonl")

        return {"count": count, "loader": loader}

    def test_class_count_preserved(self, verbnet_data):
        """Converter count matches loader count."""
        assert verbnet_data["count"] == 1
        assert len(verbnet_data["loader"].classes) == 1
        assert "give-13.1" in verbnet_data["loader"].classes

    def test_members_preserved(self, verbnet_data):
        """Member names and keys survive the round trip."""
        vc = verbnet_data["loader"].classes["give-13.1"]
        member_names = {m.name for m in vc.members}
        assert member_names == {"give", "deal", "loan"}

        give = next(m for m in vc.members if m.name == "give")
        assert give.verbnet_key == "give#2"

    def test_member_framenet_mappings(self, verbnet_data):
        """fn_mapping attribute parsed into framenet_mappings."""
        vc = verbnet_data["loader"].classes["give-13.1"]

        give = next(m for m in vc.members if m.name == "give")
        assert len(give.framenet_mappings) == 1
        assert give.framenet_mappings[0].frame_name == "Giving"

        # "None" fn_mapping should result in no mappings
        deal = next(m for m in vc.members if m.name == "deal")
        assert len(deal.framenet_mappings) == 0

    def test_member_propbank_mappings(self, verbnet_data):
        """grouping attribute parsed into propbank_mappings."""
        vc = verbnet_data["loader"].classes["give-13.1"]

        give = next(m for m in vc.members if m.name == "give")
        pb_ids = {xr.target_id for xr in give.propbank_mappings}
        assert "give.01" in pb_ids
        assert "give.02" in pb_ids

        # empty grouping → no propbank mappings
        loan = next(m for m in vc.members if m.name == "loan")
        assert len(loan.propbank_mappings) == 0

    def test_themroles_preserved(self, verbnet_data):
        """Thematic roles survive the round trip."""
        vc = verbnet_data["loader"].classes["give-13.1"]
        role_types = {r.type for r in vc.themroles}
        assert role_types == {"Agent", "Theme", "Recipient"}

    def test_subclass_hierarchy(self, verbnet_data):
        """Subclass members accessible through the class hierarchy."""
        vc = verbnet_data["loader"].classes["give-13.1"]
        assert len(vc.subclasses) == 1

        sub = vc.subclasses[0]
        assert sub.id == "give-13.1-1"
        assert len(sub.members) == 1
        assert sub.members[0].name == "sell"

    def test_member_index_builds(self, verbnet_data):
        """Member index allows looking up class by verbnet key."""
        vn = verbnet_data["loader"]
        assert "give#2" in vn.member_index
        assert vn.member_index["give#2"] == "give-13.1"


# ── PropBank ──────────────────────────────────────────────────────────────


PROPBANK_XML = """\
<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE frameset PUBLIC "-//PB//PropBank Frame v3.4 Transitional//EN"
  "http://propbank.org/specification/dtds/v3.4/frameset.dtd">
<frameset>
  <predicate lemma="abandon">
    <roleset id="abandon.01" name="leave behind">
      <aliases>
        <alias pos="v">abandon</alias>
        <alias pos="n">abandonment</alias>
      </aliases>
      <roles>
        <role descr="abandoner" f="PPT" n="0">
          <rolelinks>
            <rolelink class="leave-51.2" resource="VerbNet" version="verbnet3.3">theme</rolelink>
          </rolelinks>
        </role>
        <role descr="entity left behind" f="DIR" n="1"/>
      </roles>
      <usagenotes>
        <usage resource="PropBank" version="3.4" inuse="+"/>
      </usagenotes>
      <lexlinks>
        <lexlink class="Abandonment" confidence="0.8" resource="FrameNet"
                 src="manual" version="1.7"/>
        <lexlink class="leave-51.2" confidence="1.0" resource="VerbNet"
                 src="manual" version="verbnet3.4"/>
      </lexlinks>
      <example name="typical transitive" src="">
        <text>John abandoned the project.</text>
        <propbank>
          <rel relloc="1">abandoned</rel>
          <arg type="ARG0" start="0" end="0">John</arg>
          <arg type="ARG1" start="2" end="3">the project</arg>
        </propbank>
      </example>
    </roleset>
  </predicate>
</frameset>"""


class TestPropBankRoundTrip:
    """PropBank converter → JSONL → loader pipeline."""

    @pytest.fixture
    def propbank_data(self, tmp_path):
        """Create PropBank XML and run the conversion pipeline."""
        pb_dir = tmp_path / "frames"
        pb_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        (pb_dir / "abandon.xml").write_text(PROPBANK_XML, encoding="utf-8")

        converter = PropBankConverter()
        count = converter.convert_framesets_directory(pb_dir, output_dir / "propbank.jsonl")

        loader = PropBankLoader(data_path=output_dir / "propbank.jsonl")

        return {"count": count, "loader": loader}

    def test_frameset_count_preserved(self, propbank_data):
        """Converter count matches loader count."""
        assert propbank_data["count"] == 1
        assert len(propbank_data["loader"].framesets) == 1
        assert "abandon" in propbank_data["loader"].framesets

    def test_roles_preserved(self, propbank_data):
        """Roles survive the round trip with correct attributes."""
        fs = propbank_data["loader"].framesets["abandon"]
        rs = fs.rolesets[0]
        assert rs.id == "abandon.01"
        assert rs.name == "leave behind"

        role_numbers = {r.n for r in rs.roles}
        assert "0" in role_numbers
        assert "1" in role_numbers

        role0 = next(r for r in rs.roles if r.n == "0")
        assert role0.descr == "abandoner"

    def test_lexlinks_preserved(self, propbank_data):
        """Lexical links survive the round trip."""
        rs = propbank_data["loader"].framesets["abandon"].rolesets[0]

        assert len(rs.lexlinks) == 2
        resources = {ll.resource for ll in rs.lexlinks}
        assert "FrameNet" in resources
        assert "VerbNet" in resources

        fn_link = next(ll for ll in rs.lexlinks if ll.resource == "FrameNet")
        assert fn_link.class_name == "Abandonment"
        assert fn_link.confidence == pytest.approx(0.8)

    def test_examples_preserved(self, propbank_data):
        """Example annotations survive the round trip."""
        rs = propbank_data["loader"].framesets["abandon"].rolesets[0]
        assert len(rs.examples) == 1

        ex = rs.examples[0]
        assert ex.name == "typical transitive"
        assert "abandoned" in ex.text

    def test_roleset_index_builds(self, propbank_data):
        """Roleset index allows looking up frameset by roleset ID."""
        pb = propbank_data["loader"]
        assert "abandon.01" in pb.roleset_index
        assert pb.roleset_index["abandon.01"] == "abandon"


# ── Contract Tests ────────────────────────────────────────────────────────


class TestConverterLoaderContracts:
    """Contract tests: all converters produce valid JSONL, all loaders can read it."""

    def test_all_converters_produce_valid_jsonl(self, tmp_path):
        """Every line in every converter output is valid JSON."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        wn_dir = tmp_path / "wn"
        wn_dir.mkdir()

        # Minimal WordNet data
        (wn_dir / "data.verb").write_text(
            WN_LICENSE_HEADER
            + "00001740 29 v 01 breathe 0 001 @ 00001740 v 0000 01 + 02 00 | breathe\n",
            encoding="utf-8",
        )
        for name in ("data.noun", "data.adj", "data.adv"):
            (wn_dir / name).write_text(WN_LICENSE_HEADER, encoding="utf-8")
        (wn_dir / "index.sense").write_text("breathe%2:29:00:: 00001740 1 0\n", encoding="utf-8")
        (wn_dir / "verb.Framestext").write_text("", encoding="utf-8")
        (wn_dir / "sents.vrb").write_text("", encoding="utf-8")
        (wn_dir / "cntlist").write_text("", encoding="utf-8")
        for name in ("verb.exc", "noun.exc", "adj.exc", "adv.exc"):
            (wn_dir / name).write_text("", encoding="utf-8")

        converter = WordNetConverter()
        converter.convert_wordnet_database(wn_dir, output_dir / "wordnet.jsonl")

        # Verify every line is valid JSON
        with (output_dir / "wordnet.jsonl").open() as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                assert isinstance(obj, dict), f"Line {i} is not a JSON object"

    def test_supplementary_files_optional(self, tmp_path):
        """Loaders work without supplementary files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        wn_dir = tmp_path / "wn"
        wn_dir.mkdir()

        (wn_dir / "data.verb").write_text(
            WN_LICENSE_HEADER
            + "00001740 29 v 01 breathe 0 001 @ 00001740 v 0000 01 + 02 00 | breathe\n",
            encoding="utf-8",
        )
        for name in ("data.noun", "data.adj", "data.adv"):
            (wn_dir / name).write_text(WN_LICENSE_HEADER, encoding="utf-8")
        (wn_dir / "index.sense").write_text("breathe%2:29:00:: 00001740 1 0\n", encoding="utf-8")
        (wn_dir / "verb.Framestext").write_text("", encoding="utf-8")
        (wn_dir / "sents.vrb").write_text("", encoding="utf-8")
        (wn_dir / "cntlist").write_text("", encoding="utf-8")
        for name in ("verb.exc", "noun.exc", "adj.exc", "adv.exc"):
            (wn_dir / name).write_text("", encoding="utf-8")

        converter = WordNetConverter()
        converter.convert_wordnet_database(wn_dir, output_dir / "wordnet.jsonl")

        # Load without supplementary files
        wn = WordNetLoader(data_path=output_dir / "wordnet.jsonl")
        assert len(wn.synsets) == 1
        # sense_index and exceptions should be empty but loader shouldn't crash
        assert len(wn.sense_index) == 0
        assert len(wn.exceptions) == 0


# ── Field Completeness Tests ──────────────────────────────────────────────


class TestFieldCompleteness:
    """Verify that key model fields are populated after conversion."""

    def test_wordnet_field_completeness(self, tmp_path):
        """WordNet synsets have all expected fields."""
        wn_dir = tmp_path / "wn"
        wn_dir.mkdir()
        output = tmp_path / "wordnet.jsonl"

        (wn_dir / "data.verb").write_text(
            WN_LICENSE_HEADER
            + "00001740 29 v 01 breathe 0 001 @ 00002084 v 0000 01 + 02 00 | draw air\n",
            encoding="utf-8",
        )
        for name in ("data.noun", "data.adj", "data.adv"):
            (wn_dir / name).write_text(WN_LICENSE_HEADER, encoding="utf-8")
        (wn_dir / "index.sense").write_text("breathe%2:29:00:: 00001740 1 5\n", encoding="utf-8")
        (wn_dir / "verb.Framestext").write_text("2 Somebody ----s\n", encoding="utf-8")
        (wn_dir / "sents.vrb").write_text("2 The banks %s\n", encoding="utf-8")
        (wn_dir / "cntlist").write_text("5 breathe%2:29:00:: 1\n", encoding="utf-8")
        for name in ("verb.exc", "noun.exc", "adj.exc", "adv.exc"):
            (wn_dir / name).write_text("", encoding="utf-8")

        converter = WordNetConverter()
        converter.convert_wordnet_database(wn_dir, output)

        with output.open() as f:
            obj = json.loads(f.readline())

        # Core fields
        assert "offset" in obj
        assert "lex_filenum" in obj
        assert "ss_type" in obj
        assert "words" in obj
        assert len(obj["words"]) > 0
        assert "pointers" in obj
        assert "gloss" in obj

        # Enriched word fields
        word = obj["words"][0]
        assert word["lemma"] == "breathe"
        assert word["tag_count"] == 5
        assert word["sense_number"] == 1

        # Verb frame fields
        assert "frames" in obj
        assert len(obj["frames"]) == 1
        frame = obj["frames"][0]
        assert "frame_number" in frame
        assert frame["template"] == "Somebody ----s"
        assert frame["example_sentence"] == "The banks %s"

    def test_verbnet_field_completeness(self, tmp_path):
        """VerbNet members have cross-resource mapping fields populated."""
        vn_dir = tmp_path / "vn"
        vn_dir.mkdir()
        output = tmp_path / "verbnet.jsonl"

        (vn_dir / "give-13.1.xml").write_text(VERBNET_XML, encoding="utf-8")

        converter = VerbNetConverter()
        converter.convert_verbnet_directory(vn_dir, output)

        with output.open() as f:
            obj = json.loads(f.readline())

        # Check members
        give = next(m for m in obj["members"] if m["name"] == "give")
        assert len(give["framenet_mappings"]) == 1
        assert len(give["propbank_mappings"]) == 2
        assert len(give["wordnet_mappings"]) >= 1

        # Check themroles
        assert len(obj["themroles"]) == 3

        # Check frames
        assert len(obj["frames"]) == 1

    def test_propbank_field_completeness(self, tmp_path):
        """PropBank rolesets have all expected fields."""
        pb_dir = tmp_path / "pb"
        pb_dir.mkdir()
        output = tmp_path / "propbank.jsonl"

        (pb_dir / "abandon.xml").write_text(PROPBANK_XML, encoding="utf-8")

        converter = PropBankConverter()
        converter.convert_framesets_directory(pb_dir, output)

        with output.open() as f:
            obj = json.loads(f.readline())

        assert "predicate_lemma" in obj
        assert obj["predicate_lemma"] == "abandon"
        assert "rolesets" in obj

        rs = obj["rolesets"][0]
        assert rs["id"] == "abandon.01"
        assert len(rs["roles"]) == 2
        assert len(rs["lexlinks"]) == 2
        assert len(rs["examples"]) == 1
