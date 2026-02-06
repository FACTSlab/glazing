"""Tests for WordNet database file converter.

Tests the WordNet database file parsing functionality including
data files, index files, sense index, and exception files.
"""

import pytest

from glazing.wordnet.converter import (
    WordNetConverter,
    convert_wordnet_database,
    parse_data_file,
    parse_exception_file,
    parse_index_file,
    parse_sense_index,
)


class TestWordNetConverter:
    """Test WordNet database file converter functionality."""

    @pytest.fixture
    def converter(self):
        """Create WordNet converter instance."""
        return WordNetConverter()

    @pytest.fixture
    def sample_data_file_content(self):
        """Sample WordNet data file content."""
        return """  Copyright notice and license text here
  More license text
00001740 29 v 01 breathe 0 005 $ 00001740 v 0000 @ 00002084 v 0000 ~ 00001740 v 0000 + 00002760 v 0000 ^ 00001740 v 0000 | take in and expel air through lungs
00002084 29 v 02 respire 0 breathe 1 003 $ 00001740 v 0000 @ 00002325 v 0000 ~ 00002760 v 0000 02 + 01 00 + 02 01 | undergo the biomedical and metabolic processes of respiration by taking up oxygen and producing carbon monoxide
"""

    @pytest.fixture
    def sample_index_file_content(self):
        """Sample WordNet index file content."""
        return """  Copyright notice and license text here
  More license text
abandon v 5 4 @ ~ $ + 5 5 02232813 02232523 02080923 00614907 00615748
breathe v 3 2 @ ~ 3 2 00001740 00002084 00002325
"""

    @pytest.fixture
    def sample_sense_index_content(self):
        """Sample WordNet sense index content."""
        return """  Copyright notice and license text here
  More license text
abandon%2:40:01:: 02232813 2 5
breathe%2:29:00:: 00001740 1 2
"""

    @pytest.fixture
    def sample_exception_content(self):
        """Sample exception file content."""
        return """  Copyright notice and license text here
  More license text
axes axe axis
mice mouse
women woman
"""

    @pytest.fixture
    def temp_data_file(self, sample_data_file_content, tmp_path):
        """Create temporary data file."""
        data_file = tmp_path / "data.verb"
        data_file.write_text(sample_data_file_content, encoding="utf-8")
        return data_file

    @pytest.fixture
    def temp_index_file(self, sample_index_file_content, tmp_path):
        """Create temporary index file."""
        index_file = tmp_path / "index.verb"
        index_file.write_text(sample_index_file_content, encoding="utf-8")
        return index_file

    @pytest.fixture
    def temp_sense_file(self, sample_sense_index_content, tmp_path):
        """Create temporary sense index file."""
        sense_file = tmp_path / "index.sense"
        sense_file.write_text(sample_sense_index_content, encoding="utf-8")
        return sense_file

    @pytest.fixture
    def temp_exception_file(self, sample_exception_content, tmp_path):
        """Create temporary exception file."""
        exc_file = tmp_path / "verb.exc"
        exc_file.write_text(sample_exception_content, encoding="utf-8")
        return exc_file

    def test_parse_data_file_basic(self, converter, temp_data_file):
        """Test basic data file parsing."""
        synsets = converter.parse_data_file(temp_data_file, "v")

        assert len(synsets) == 2

        # Test first synset
        synset1 = synsets[0]
        assert synset1.offset == "00001740"
        assert synset1.lex_filenum == 29
        assert synset1.ss_type == "v"
        assert len(synset1.words) == 1
        assert synset1.words[0].lemma == "breathe"
        assert synset1.words[0].lex_id == 0
        assert len(synset1.pointers) == 5
        assert synset1.gloss == "take in and expel air through lungs"

    def test_parse_data_file_with_verb_frames(self, converter, temp_data_file):
        """Test data file parsing with verb frames."""
        synsets = converter.parse_data_file(temp_data_file, "v")

        # Second synset should have verb frames
        synset2 = synsets[1]
        assert synset2.frames is not None
        assert len(synset2.frames) == 2

        frame1 = synset2.frames[0]
        assert frame1.frame_number == 1
        assert frame1.word_indices == [0]

        frame2 = synset2.frames[1]
        assert frame2.frame_number == 2
        assert frame2.word_indices == [1]

    def test_parse_data_file_pointers(self, converter, temp_data_file):
        """Test pointer parsing from data file."""
        synsets = converter.parse_data_file(temp_data_file, "v")

        synset1 = synsets[0]
        pointers = synset1.pointers

        # Check specific pointers
        verb_group = next((p for p in pointers if p.symbol == "$"), None)
        assert verb_group is not None
        assert verb_group.offset == "00001740"
        assert verb_group.pos == "v"
        assert verb_group.source == 0
        assert verb_group.target == 0

    def test_parse_data_file_nonexistent(self, converter):
        """Test parsing non-existent data file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="WordNet data file not found"):
            converter.parse_data_file("nonexistent.data", "n")

    def test_parse_index_file_basic(self, converter, temp_index_file):
        """Test basic index file parsing."""
        entries = converter.parse_index_file(temp_index_file, "v")

        assert len(entries) == 2

        # Test first entry
        entry1 = entries[0]
        assert entry1.lemma == "abandon"
        assert entry1.pos == "v"
        assert entry1.synset_cnt == 5
        assert entry1.p_cnt == 4
        assert entry1.sense_cnt == 5
        assert entry1.tagsense_cnt == 5
        assert len(entry1.synset_offsets) == 5
        assert "02232813" in entry1.synset_offsets

    def test_parse_index_file_pointers(self, converter, temp_index_file):
        """Test pointer symbol parsing from index file."""
        entries = converter.parse_index_file(temp_index_file, "v")

        entry1 = entries[0]
        assert len(entry1.ptr_symbols) == 4
        assert "@" in entry1.ptr_symbols
        assert "~" in entry1.ptr_symbols
        assert "$" in entry1.ptr_symbols
        assert "+" in entry1.ptr_symbols

    def test_parse_index_file_wrong_pos(self, converter, temp_index_file):
        """Test parsing index file with wrong POS filters correctly."""
        entries = converter.parse_index_file(temp_index_file, "n")

        # Should be empty since all entries are verbs
        assert len(entries) == 0

    def test_parse_sense_index_basic(self, converter, temp_sense_file):
        """Test basic sense index parsing."""
        senses = converter.parse_sense_index(temp_sense_file)

        assert len(senses) == 2

        # Test first sense
        sense1 = senses[0]
        assert sense1.sense_key == "abandon%2:40:01::"
        assert sense1.lemma == "abandon"
        assert sense1.ss_type == "v"  # ss_type 2 maps to verb
        assert sense1.lex_filenum == 40
        assert sense1.lex_id == 1
        assert sense1.synset_offset == "02232813"
        assert sense1.sense_number == 2
        assert sense1.tag_count == 5

    def test_parse_sense_index_pos_mapping(self, converter, temp_sense_file):
        """Test POS mapping from ss_type in sense index."""
        senses = converter.parse_sense_index(temp_sense_file)

        # Both senses should map to verb (ss_type 2)
        for sense in senses:
            assert sense.ss_type == "v"

    def test_parse_exception_file_basic(self, converter, temp_exception_file):
        """Test basic exception file parsing."""
        entries = converter.parse_exception_file(temp_exception_file)

        assert len(entries) == 3

        # Test first entry
        entry1 = entries[0]
        assert entry1.inflected_form == "axes"
        assert len(entry1.base_forms) == 2
        assert "axe" in entry1.base_forms
        assert "axis" in entry1.base_forms

        # Test second entry
        entry2 = entries[1]
        assert entry2.inflected_form == "mice"
        assert entry2.base_forms == ["mouse"]

    def test_convert_wordnet_database_basic(self, converter, tmp_path):
        """Test complete WordNet database conversion."""
        # Create minimal WordNet database
        wordnet_dir = tmp_path / "wordnet"
        wordnet_dir.mkdir()

        # Create data files
        (wordnet_dir / "data.noun").write_text(
            "00001740 03 n 01 entity 0 000 | something existing\n", encoding="utf-8"
        )
        (wordnet_dir / "data.verb").write_text(
            "00001740 29 v 01 breathe 0 000 | take air\n", encoding="utf-8"
        )

        # Create index files
        (wordnet_dir / "index.noun").write_text("entity n 1 0 1 1 00001740\n", encoding="utf-8")
        (wordnet_dir / "index.verb").write_text("breathe v 1 0 1 1 00001740\n", encoding="utf-8")

        # Create sense index
        (wordnet_dir / "index.sense").write_text(
            "entity%1:03:00:: 00001740 1 0\n", encoding="utf-8"
        )

        # Create exception file
        (wordnet_dir / "noun.exc").write_text("mice mouse\n", encoding="utf-8")

        output_file = tmp_path / "output" / "synsets.jsonl"

        counts = converter.convert_wordnet_database(wordnet_dir, output_file)

        # Check counts
        assert counts["synsets_noun"] == 1
        assert counts["synsets_verb"] == 1
        assert counts["total_synsets"] == 2

        # Check output file exists
        assert output_file.exists()

    def test_convert_wordnet_database_nonexistent_dir(self, converter, tmp_path):
        """Test conversion of non-existent directory raises FileNotFoundError."""
        output_file = tmp_path / "output" / "synsets.jsonl"

        with pytest.raises(FileNotFoundError, match="WordNet directory not found"):
            converter.convert_wordnet_database("nonexistent", output_file)

    def test_parse_data_line_malformed(self, converter):
        """Test parsing malformed data line returns None."""
        malformed_line = "invalid line format"
        result = converter._parse_data_line(malformed_line)
        assert result is None

    def test_parse_index_line_malformed(self, converter):
        """Test parsing malformed index line returns None."""
        malformed_line = "invalid line format"
        result = converter._parse_index_line(malformed_line, "v")
        assert result is None

    def test_parse_sense_line_malformed(self, converter):
        """Test parsing malformed sense line returns None."""
        malformed_line = "invalid line format"
        result = converter._parse_sense_line(malformed_line)
        assert result is None

    def test_parse_exception_line_malformed(self, converter):
        """Test parsing malformed exception line returns None."""
        malformed_line = "invalid"  # Only one word
        result = converter._parse_exception_line(malformed_line)
        assert result is None

    def test_is_valid_word_form(self, converter):
        """Test word form validation."""
        assert converter._is_valid_word_form("valid")
        assert converter._is_valid_word_form("with_underscore")
        assert converter._is_valid_word_form("with-hyphen")
        assert converter._is_valid_word_form("with'apostrophe")

        assert not converter._is_valid_word_form("")
        assert not converter._is_valid_word_form("123invalid")
        assert not converter._is_valid_word_form("invalid@")

    def test_lex_file_names_mapping(self, converter):
        """Test lexical file name mapping."""
        assert converter.LEX_FILE_NAMES[0] == "adj.all"
        assert converter.LEX_FILE_NAMES[4] == "noun.Tops"
        assert converter.LEX_FILE_NAMES[29] == "noun.time"
        assert converter.LEX_FILE_NAMES[30] == "verb.body"
        assert converter.LEX_FILE_NAMES[44] == "verb.weather"

    def test_parse_data_file_empty_lines(self, converter, tmp_path):
        """Test data file parsing handles empty lines and comments."""
        content = """  Copyright notice
  More copyright text

00001740 03 n 01 entity 0 000 | something existing

"""
        data_file = tmp_path / "data.noun"
        data_file.write_text(content, encoding="utf-8")

        synsets = converter.parse_data_file(data_file, "n")
        assert len(synsets) == 1

    def test_parse_complex_gloss(self, converter, tmp_path):
        """Test parsing synset with complex gloss containing special characters."""
        content = "00001740 03 n 01 entity 0 000 | that which is perceived or known or inferred to have its own distinct existence (living or nonliving)"
        data_file = tmp_path / "data.noun"
        data_file.write_text(content, encoding="utf-8")

        synsets = converter.parse_data_file(data_file, "n")
        assert len(synsets) == 1
        assert "perceived" in synsets[0].gloss
        assert "nonliving" in synsets[0].gloss


class TestWordNetConverterFunctions:
    """Test WordNet converter module functions."""

    def test_parse_data_file_function(self, tmp_path):
        """Test parse_data_file function."""
        content = "00001740 03 n 01 entity 0 000 | something existing\n"
        data_file = tmp_path / "data.noun"
        data_file.write_text(content, encoding="utf-8")

        synsets = parse_data_file(data_file, "n")

        assert len(synsets) == 1
        assert synsets[0].offset == "00001740"

    def test_parse_index_file_function(self, tmp_path):
        """Test parse_index_file function."""
        content = "entity n 1 0 1 1 00001740\n"
        index_file = tmp_path / "index.noun"
        index_file.write_text(content, encoding="utf-8")

        entries = parse_index_file(index_file, "n")

        assert len(entries) == 1
        assert entries[0].lemma == "entity"

    def test_parse_sense_index_function(self, tmp_path):
        """Test parse_sense_index function."""
        content = "entity%1:03:00:: 00001740 1 0\n"
        sense_file = tmp_path / "index.sense"
        sense_file.write_text(content, encoding="utf-8")

        senses = parse_sense_index(sense_file)

        assert len(senses) == 1
        assert senses[0].lemma == "entity"

    def test_parse_exception_file_function(self, tmp_path):
        """Test parse_exception_file function."""
        content = "mice mouse\n"
        exc_file = tmp_path / "noun.exc"
        exc_file.write_text(content, encoding="utf-8")

        entries = parse_exception_file(exc_file)

        assert len(entries) == 1
        assert entries[0].inflected_form == "mice"

    def test_convert_wordnet_database_function(self, tmp_path):
        """Test convert_wordnet_database function."""
        wordnet_dir = tmp_path / "wordnet"
        wordnet_dir.mkdir()

        (wordnet_dir / "data.noun").write_text(
            "00001740 03 n 01 entity 0 000 | something existing\n", encoding="utf-8"
        )

        output_file = tmp_path / "output" / "synsets.jsonl"

        counts = convert_wordnet_database(wordnet_dir, output_file)

        assert counts["synsets_noun"] == 1
        assert counts["total_synsets"] == 1
        assert output_file.exists()
