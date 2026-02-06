"""Tests for WordNet data models.

This module tests the WordNet models to ensure they work correctly with
WordNet 3.1 data structures and validation.
"""

import pytest
from pydantic import ValidationError

from glazing.wordnet.models import (
    ExceptionEntry,
    IndexEntry,
    Pointer,
    Sense,
    Synset,
    VerbFrame,
    Word,
    WordNetCrossRef,
)


class TestWord:
    """Test Word model."""

    def test_word_creation(self):
        """Test basic Word creation."""
        word = Word(lemma="dog", lex_id=0)
        assert word.lemma == "dog"
        assert word.lex_id == 0

    def test_word_lemma_validation(self):
        """Test lemma format validation."""
        # Valid lemmas
        Word(lemma="dog", lex_id=0)
        Word(lemma="run_up", lex_id=1)
        Word(lemma="mother-in-law", lex_id=0)
        Word(lemma="Dog", lex_id=0)  # Uppercase allowed (proper nouns)
        Word(lemma="Dr.", lex_id=0)  # Dots allowed (abbreviations)
        Word(lemma="123dog", lex_id=0)  # Digit start allowed

        # Invalid lemmas
        with pytest.raises(ValidationError):
            Word(lemma="", lex_id=0)  # Empty
        with pytest.raises(ValidationError):
            Word(lemma=" dog", lex_id=0)  # Leading space

    def test_word_lex_id_validation(self):
        """Test lex_id range validation."""
        # Valid lex_ids
        Word(lemma="dog", lex_id=0)
        Word(lemma="dog", lex_id=15)

        # Invalid lex_ids
        with pytest.raises(ValidationError):
            Word(lemma="dog", lex_id=-1)
        with pytest.raises(ValidationError):
            Word(lemma="dog", lex_id=16)


class TestPointer:
    """Test Pointer model."""

    def test_pointer_creation(self):
        """Test basic Pointer creation."""
        pointer = Pointer(symbol="@", offset="00002084", pos="n", source=0, target=0)
        assert pointer.symbol == "@"
        assert pointer.offset == "00002084"
        assert pointer.pos == "n"

    def test_pointer_is_semantic(self):
        """Test semantic relation detection."""
        semantic = Pointer(symbol="@", offset="00002084", pos="n", source=0, target=0)
        assert semantic.is_semantic()
        assert not semantic.is_lexical()

    def test_pointer_is_lexical(self):
        """Test lexical relation detection."""
        lexical = Pointer(symbol="+", offset="00002084", pos="n", source=1, target=2)
        assert lexical.is_lexical()
        assert not lexical.is_semantic()

    def test_pointer_validation(self):
        """Test pointer field validation."""
        # Valid pointer
        Pointer(symbol="@", offset="00001740", pos="n", source=0, target=0)

        # Invalid offset
        with pytest.raises(ValidationError):
            Pointer(symbol="@", offset="1740", pos="n", source=0, target=0)

        # Invalid POS
        with pytest.raises(ValidationError):
            Pointer(symbol="@", offset="00001740", pos="x", source=0, target=0)

        # Negative indices
        with pytest.raises(ValidationError):
            Pointer(symbol="@", offset="00001740", pos="n", source=-1, target=0)


class TestVerbFrame:
    """Test VerbFrame model."""

    def test_verb_frame_creation(self):
        """Test basic VerbFrame creation."""
        frame = VerbFrame(frame_number=8, word_indices=[0])
        assert frame.frame_number == 8
        assert frame.word_indices == [0]

    def test_verb_frame_number_validation(self):
        """Test frame number validation."""
        # Valid frame numbers
        VerbFrame(frame_number=1)
        VerbFrame(frame_number=35)

        # Invalid frame numbers
        with pytest.raises(ValidationError):
            VerbFrame(frame_number=0)
        with pytest.raises(ValidationError):
            VerbFrame(frame_number=36)

    def test_word_indices_validation(self):
        """Test word indices validation."""
        # Valid indices
        VerbFrame(frame_number=8, word_indices=[0, 1, 2])

        # Invalid indices (negative)
        with pytest.raises(ValidationError):
            VerbFrame(frame_number=8, word_indices=[-1])


class TestSynset:
    """Test Synset model."""

    def test_synset_creation(self):
        """Test basic Synset creation."""
        words = [Word(lemma="dog", lex_id=0)]
        synset = Synset(
            offset="00001740",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=words,
            gloss="a domesticated carnivorous mammal",
        )
        assert synset.offset == "00001740"
        assert synset.ss_type == "n"
        assert len(synset.words) == 1

    def test_synset_get_lemmas(self):
        """Test get_lemmas method."""
        words = [Word(lemma="dog", lex_id=0), Word(lemma="domestic_dog", lex_id=1)]
        synset = Synset(
            offset="00001740",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=words,
            gloss="a domesticated carnivorous mammal",
        )
        lemmas = synset.get_lemmas()
        assert lemmas == ["dog", "domestic_dog"]

    def test_synset_get_hypernyms(self):
        """Test get_hypernyms method."""
        pointer = Pointer(symbol="@", offset="00002084", pos="n", source=0, target=0)
        synset = Synset(
            offset="00001740",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=[Word(lemma="dog", lex_id=0)],
            pointers=[pointer],
            gloss="test",
        )
        hypernyms = synset.get_hypernyms()
        assert len(hypernyms) == 1
        assert hypernyms[0].symbol == "@"

    def test_synset_get_hyponyms(self):
        """Test get_hyponyms method."""
        pointer = Pointer(symbol="~", offset="00001850", pos="n", source=0, target=0)
        synset = Synset(
            offset="00002084",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=[Word(lemma="carnivore", lex_id=0)],
            pointers=[pointer],
            gloss="test",
        )
        hyponyms = synset.get_hyponyms()
        assert len(hyponyms) == 1
        assert hyponyms[0].symbol == "~"

    def test_synset_get_pointers_by_symbol(self):
        """Test get_pointers_by_symbol method."""
        pointers = [
            Pointer(symbol="@", offset="00002084", pos="n", source=0, target=0),
            Pointer(symbol="!", offset="00003000", pos="n", source=1, target=1),
            Pointer(symbol="@", offset="00004000", pos="n", source=0, target=0),
        ]
        synset = Synset(
            offset="00001740",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=[Word(lemma="dog", lex_id=0)],
            pointers=pointers,
            gloss="test",
        )
        hypernyms = synset.get_pointers_by_symbol("@")
        assert len(hypernyms) == 2
        antonyms = synset.get_pointers_by_symbol("!")
        assert len(antonyms) == 1

    def test_synset_has_relation(self):
        """Test has_relation method."""
        pointer = Pointer(symbol="@", offset="00002084", pos="n", source=0, target=0)
        synset = Synset(
            offset="00001740",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=[Word(lemma="dog", lex_id=0)],
            pointers=[pointer],
            gloss="test",
        )
        assert synset.has_relation("@")
        assert not synset.has_relation("!")

    def test_synset_get_semantic_pointers(self):
        """Test get_semantic_pointers method."""
        pointers = [
            Pointer(symbol="@", offset="00002084", pos="n", source=0, target=0),  # Semantic
            Pointer(symbol="!", offset="00003000", pos="n", source=1, target=1),  # Lexical
        ]
        synset = Synset(
            offset="00001740",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=[Word(lemma="dog", lex_id=0)],
            pointers=pointers,
            gloss="test",
        )
        semantic = synset.get_semantic_pointers()
        assert len(semantic) == 1
        assert semantic[0].symbol == "@"

    def test_synset_get_lexical_pointers(self):
        """Test get_lexical_pointers method."""
        pointers = [
            Pointer(symbol="@", offset="00002084", pos="n", source=0, target=0),  # Semantic
            Pointer(symbol="!", offset="00003000", pos="n", source=1, target=1),  # Lexical
        ]
        synset = Synset(
            offset="00001740",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=[Word(lemma="dog", lex_id=0)],
            pointers=pointers,
            gloss="test",
        )
        lexical = synset.get_lexical_pointers()
        assert len(lexical) == 1
        assert lexical[0].symbol == "!"

    def test_synset_validation(self):
        """Test synset field validation."""
        words = [Word(lemma="dog", lex_id=0)]

        # Valid synset
        Synset(
            offset="00001740",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=words,
            gloss="test",
        )

        # Invalid lex_filenum
        with pytest.raises(ValidationError):
            Synset(
                offset="00001740",
                lex_filenum=45,  # > 44
                lex_filename="noun.animal",
                ss_type="n",
                words=words,
                gloss="test",
            )

        # Invalid offset
        with pytest.raises(ValidationError):
            Synset(
                offset="1740",  # Too short
                lex_filenum=5,
                lex_filename="noun.animal",
                ss_type="n",
                words=words,
                gloss="test",
            )


class TestSense:
    """Test Sense model."""

    def test_sense_creation(self):
        """Test basic Sense creation."""
        sense = Sense(
            sense_key="dog%1:05:00::",
            lemma="dog",
            ss_type="n",
            lex_filenum=5,
            lex_id=0,
            synset_offset="00001740",
            sense_number=1,
            tag_count=15,
        )
        assert sense.sense_key == "dog%1:05:00::"
        assert sense.lemma == "dog"

    def test_sense_parse_sense_key(self):
        """Test sense key parsing."""
        sense = Sense(
            sense_key="dog%1:05:00::",
            lemma="dog",
            ss_type="n",
            lex_filenum=5,
            lex_id=0,
            synset_offset="00001740",
            sense_number=1,
            tag_count=15,
        )
        components = sense.parse_sense_key()
        assert components["lemma"] == "dog"
        assert components["ss_type"] == 1
        assert components["lex_filenum"] == 5
        assert components["lex_id"] == 0

    def test_sense_with_head_word(self):
        """Test sense with head word (adjective satellite)."""
        sense = Sense(
            sense_key="better%5:00:00:good:01",
            lemma="better",
            ss_type="s",
            lex_filenum=0,
            lex_id=0,
            head_word="good",
            head_id=1,
            synset_offset="00001234",
            sense_number=1,
            tag_count=5,
        )
        components = sense.parse_sense_key()
        assert components["head_word"] == "good"
        assert components["head_id"] == 1

    def test_sense_validation(self):
        """Test sense field validation."""
        # Valid sense
        Sense(
            sense_key="dog%1:05:00::",
            lemma="dog",
            ss_type="n",
            lex_filenum=5,
            lex_id=0,
            synset_offset="00001740",
            sense_number=1,
            tag_count=15,
        )

        # Invalid sense key
        with pytest.raises(ValidationError):
            Sense(
                sense_key="dog%6:05:00::",  # Invalid POS
                lemma="dog",
                ss_type="n",
                lex_filenum=5,
                lex_id=0,
                synset_offset="00001740",
                sense_number=1,
                tag_count=15,
            )

        # Invalid sense number
        with pytest.raises(ValidationError):
            Sense(
                sense_key="dog%1:05:00::",
                lemma="dog",
                ss_type="n",
                lex_filenum=5,
                lex_id=0,
                synset_offset="00001740",
                sense_number=0,  # Must be >= 1
                tag_count=15,
            )


class TestIndexEntry:
    """Test IndexEntry model."""

    def test_index_entry_creation(self):
        """Test basic IndexEntry creation."""
        entry = IndexEntry(
            lemma="dog",
            pos="n",
            synset_cnt=7,
            p_cnt=4,
            ptr_symbols=["!", "@", "~", "#m"],
            sense_cnt=7,
            tagsense_cnt=6,
            synset_offsets=["00001740", "00002084"],
        )
        assert entry.lemma == "dog"
        assert entry.synset_cnt == 7
        assert len(entry.synset_offsets) == 2

    def test_index_entry_validation(self):
        """Test index entry field validation."""
        # Valid entry
        IndexEntry(
            lemma="dog",
            pos="n",
            synset_cnt=7,
            p_cnt=4,
            ptr_symbols=["@"],
            sense_cnt=7,
            tagsense_cnt=6,
            synset_offsets=["00001740"],
        )

        # Invalid synset count
        with pytest.raises(ValidationError):
            IndexEntry(
                lemma="dog",
                pos="n",
                synset_cnt=-1,  # Negative
                p_cnt=4,
                ptr_symbols=["@"],
                sense_cnt=7,
                tagsense_cnt=6,
                synset_offsets=["00001740"],
            )


class TestExceptionEntry:
    """Test ExceptionEntry model."""

    def test_exception_entry_creation(self):
        """Test basic ExceptionEntry creation."""
        entry = ExceptionEntry(inflected_form="geese", base_forms=["goose"])
        assert entry.inflected_form == "geese"
        assert entry.base_forms == ["goose"]

    def test_exception_entry_multiple_bases(self):
        """Test exception entry with multiple base forms."""
        entry = ExceptionEntry(inflected_form="better", base_forms=["good", "well"])
        assert len(entry.base_forms) == 2

    def test_exception_entry_validation(self):
        """Test exception entry form validation."""
        # Valid forms
        ExceptionEntry(inflected_form="geese", base_forms=["goose"])
        ExceptionEntry(inflected_form="mother-in-law", base_forms=["mother-in-law"])

        # Invalid forms
        with pytest.raises(ValidationError):
            ExceptionEntry(inflected_form="", base_forms=["goose"])
        with pytest.raises(ValidationError):
            ExceptionEntry(inflected_form="geese", base_forms=[""])


class TestWordNetCrossRef:
    """Test WordNetCrossRef model."""

    def test_wordnet_cross_ref_creation(self):
        """Test basic WordNetCrossRef creation."""
        ref = WordNetCrossRef(sense_key="give%2:40:00::", lemma="give", pos="v", sense_number=1)
        assert ref.sense_key == "give%2:40:00::"
        assert ref.lemma == "give"
        assert ref.pos == "v"

    def test_to_percentage_notation(self):
        """Test conversion to percentage notation."""
        ref = WordNetCrossRef(sense_key="give%2:40:00::", lemma="give", pos="v")
        notation = ref.to_percentage_notation()
        assert notation == "give%2:40:00"

    def test_to_percentage_notation_no_sense_key(self):
        """Test percentage notation with no sense key."""
        ref = WordNetCrossRef(synset_offset="00001740", lemma="give", pos="v")
        notation = ref.to_percentage_notation()
        assert notation == ""

    def test_from_percentage_notation(self):
        """Test parsing from percentage notation."""
        ref = WordNetCrossRef.from_percentage_notation("give%2:40:00")
        assert ref.lemma == "give"
        assert ref.pos == "v"
        assert ref.sense_key == "give%2:40:00::"

    def test_from_percentage_notation_all_pos(self):
        """Test percentage notation for all POS types."""
        # Test all POS mappings
        test_cases = [
            ("dog%1:05:00", "dog", "n"),
            ("run%2:38:00", "run", "v"),
            ("good%3:00:01", "good", "a"),
            ("quickly%4:02:00", "quickly", "r"),
            ("better%5:00:00", "better", "s"),
        ]

        for notation, expected_lemma, expected_pos in test_cases:
            ref = WordNetCrossRef.from_percentage_notation(notation)
            assert ref.lemma == expected_lemma
            assert ref.pos == expected_pos

    def test_from_percentage_notation_invalid(self):
        """Test invalid percentage notation."""
        with pytest.raises(ValueError):
            WordNetCrossRef.from_percentage_notation("invalid")
        with pytest.raises(ValueError):
            WordNetCrossRef.from_percentage_notation("give%6:40:00")  # Invalid POS

    def test_is_valid_reference(self):
        """Test is_valid_reference method."""
        # With sense key
        ref1 = WordNetCrossRef(sense_key="give%2:40:00::", lemma="give", pos="v")
        assert ref1.is_valid_reference()

        # With synset offset
        ref2 = WordNetCrossRef(synset_offset="00001740", lemma="give", pos="v")
        assert ref2.is_valid_reference()

        # Without identifiers
        ref3 = WordNetCrossRef(lemma="give", pos="v")
        assert not ref3.is_valid_reference()

    def test_get_primary_identifier(self):
        """Test get_primary_identifier method."""
        # Sense key preferred
        ref1 = WordNetCrossRef(
            sense_key="give%2:40:00::", synset_offset="00001740", lemma="give", pos="v"
        )
        assert ref1.get_primary_identifier() == "give%2:40:00::"

        # Fall back to synset offset
        ref2 = WordNetCrossRef(synset_offset="00001740", lemma="give", pos="v")
        assert ref2.get_primary_identifier() == "00001740"

        # No identifiers
        ref3 = WordNetCrossRef(lemma="give", pos="v")
        assert ref3.get_primary_identifier() is None

    def test_wordnet_cross_ref_validation(self):
        """Test cross reference field validation."""
        # Valid ref
        WordNetCrossRef(lemma="give", pos="v")

        # Invalid POS
        with pytest.raises(ValidationError):
            WordNetCrossRef(lemma="give", pos="x")

        # Invalid sense key
        with pytest.raises(ValidationError):
            WordNetCrossRef(sense_key="invalid", lemma="give", pos="v")
