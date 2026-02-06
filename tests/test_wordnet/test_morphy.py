"""Tests for WordNet morphy module."""

import json
import tempfile
from pathlib import Path

import pytest

from glazing.wordnet.loader import WordNetLoader
from glazing.wordnet.morphy import Morphy, morphy


def _write_wordnet_files(
    data_path: Path, synsets: list[dict], exceptions: list[dict] | None = None
) -> Path:
    """Helper to write synsets and exceptions in the single-file format.

    Returns the path to the primary wordnet.jsonl file.
    """
    wordnet_file = data_path / "wordnet.jsonl"
    with open(wordnet_file, "w") as f:
        for synset in synsets:
            f.write(json.dumps(synset) + "\n")

    if exceptions:
        with open(data_path / "wordnet_exceptions.jsonl", "w") as f:
            for exc in exceptions:
                f.write(json.dumps(exc) + "\n")

    return wordnet_file


class TestMorphy:
    """Test WordNet morphological processing."""

    @pytest.fixture
    def temp_data_with_lemmas(self):
        """Create test data with lemmas for morphy testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # All synsets in a single file
            all_synsets = [
                # Noun synsets
                {
                    "offset": "02084442",
                    "lex_filenum": 5,
                    "lex_filename": "noun.animal",
                    "ss_type": "n",
                    "words": [
                        {"lemma": "dog", "lex_id": 0},
                        {"lemma": "domestic_dog", "lex_id": 1},
                    ],
                    "pointers": [],
                    "gloss": "a member of the genus Canis",
                },
                {
                    "offset": "09917593",
                    "lex_filenum": 15,
                    "lex_filename": "noun.person",
                    "ss_type": "n",
                    "words": [{"lemma": "child", "lex_id": 0}, {"lemma": "kid", "lex_id": 1}],
                    "pointers": [],
                    "gloss": "a young person",
                },
                {
                    "offset": "02866578",
                    "lex_filenum": 6,
                    "lex_filename": "noun.artifact",
                    "ss_type": "n",
                    "words": [{"lemma": "box", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "a container",
                },
                {
                    "offset": "01930374",
                    "lex_filenum": 5,
                    "lex_filename": "noun.animal",
                    "ss_type": "n",
                    "words": [{"lemma": "fly", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "two-winged insects",
                },
                # Verb synsets
                {
                    "offset": "01926311",
                    "lex_filenum": 38,
                    "lex_filename": "verb.motion",
                    "ss_type": "v",
                    "words": [{"lemma": "run", "lex_id": 0}],
                    "pointers": [],
                    "frames": [],
                    "gloss": "move fast",
                },
                {
                    "offset": "01835496",
                    "lex_filenum": 38,
                    "lex_filename": "verb.motion",
                    "ss_type": "v",
                    "words": [{"lemma": "fly", "lex_id": 0}],
                    "pointers": [],
                    "frames": [],
                    "gloss": "travel through the air",
                },
                {
                    "offset": "00010435",
                    "lex_filenum": 42,
                    "lex_filename": "verb.stative",
                    "ss_type": "v",
                    "words": [{"lemma": "be", "lex_id": 0}],
                    "pointers": [],
                    "frames": [],
                    "gloss": "have the quality of being",
                },
                {
                    "offset": "00654625",
                    "lex_filenum": 30,
                    "lex_filename": "verb.cognition",
                    "ss_type": "v",
                    "words": [{"lemma": "watch", "lex_id": 0}],
                    "pointers": [],
                    "frames": [],
                    "gloss": "look attentively",
                },
                # Adjective synsets
                {
                    "offset": "00001740",
                    "lex_filenum": 0,
                    "lex_filename": "adj.all",
                    "ss_type": "a",
                    "words": [{"lemma": "big", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "above average in size",
                },
                {
                    "offset": "00001741",
                    "lex_filenum": 0,
                    "lex_filename": "adj.all",
                    "ss_type": "a",
                    "words": [{"lemma": "nice", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "pleasant or pleasing",
                },
                {
                    "offset": "00001742",
                    "lex_filenum": 0,
                    "lex_filename": "adj.all",
                    "ss_type": "a",
                    "words": [{"lemma": "good", "lex_id": 0}, {"lemma": "well", "lex_id": 1}],
                    "pointers": [],
                    "gloss": "having desirable qualities",
                },
            ]

            # All exceptions in a single file with pos field
            all_exceptions = [
                # Noun exceptions
                {"inflected_form": "children", "base_forms": ["child"], "pos": "n"},
                {"inflected_form": "geese", "base_forms": ["goose"], "pos": "n"},
                {"inflected_form": "men", "base_forms": ["man"], "pos": "n"},
                {"inflected_form": "women", "base_forms": ["woman"], "pos": "n"},
                {"inflected_form": "teeth", "base_forms": ["tooth"], "pos": "n"},
                {"inflected_form": "feet", "base_forms": ["foot"], "pos": "n"},
                {"inflected_form": "mice", "base_forms": ["mouse"], "pos": "n"},
                # Verb exceptions
                {"inflected_form": "ran", "base_forms": ["run"], "pos": "v"},
                {"inflected_form": "went", "base_forms": ["go"], "pos": "v"},
                {"inflected_form": "was", "base_forms": ["be"], "pos": "v"},
                {"inflected_form": "were", "base_forms": ["be"], "pos": "v"},
                {"inflected_form": "been", "base_forms": ["be"], "pos": "v"},
                {"inflected_form": "flew", "base_forms": ["fly"], "pos": "v"},
                {"inflected_form": "flown", "base_forms": ["fly"], "pos": "v"},
                # Adjective exceptions
                {"inflected_form": "better", "base_forms": ["good", "well"], "pos": "a"},
                {"inflected_form": "best", "base_forms": ["good", "well"], "pos": "a"},
                {"inflected_form": "worse", "base_forms": ["bad"], "pos": "a"},
                {"inflected_form": "worst", "base_forms": ["bad"], "pos": "a"},
            ]

            wordnet_file = _write_wordnet_files(data_path, all_synsets, all_exceptions)
            yield wordnet_file

    @pytest.fixture
    def loader_with_data(self, temp_data_with_lemmas):
        """Create a loaded WordNet loader."""
        loader = WordNetLoader(temp_data_with_lemmas)
        loader.load()
        return loader

    def test_morphy_initialization(self, loader_with_data):
        """Test Morphy initialization."""
        processor = Morphy(loader_with_data)

        assert processor.loader is loader_with_data
        assert "n" in processor.suffix_rules
        assert "v" in processor.suffix_rules
        assert "a" in processor.suffix_rules
        assert "r" in processor.suffix_rules

    def test_check_exceptions_noun(self, loader_with_data):
        """Test checking noun exceptions."""
        processor = Morphy(loader_with_data)

        # Test irregular plurals
        assert processor.check_exceptions("children", "n") == ["child"]
        assert processor.check_exceptions("geese", "n") == ["goose"]
        assert processor.check_exceptions("men", "n") == ["man"]
        assert processor.check_exceptions("women", "n") == ["woman"]
        assert processor.check_exceptions("teeth", "n") == ["tooth"]
        assert processor.check_exceptions("feet", "n") == ["foot"]
        assert processor.check_exceptions("mice", "n") == ["mouse"]

        # Test non-exception
        assert processor.check_exceptions("dogs", "n") == []

    def test_check_exceptions_verb(self, loader_with_data):
        """Test checking verb exceptions."""
        processor = Morphy(loader_with_data)

        # Test irregular verbs
        assert processor.check_exceptions("ran", "v") == ["run"]
        assert processor.check_exceptions("went", "v") == ["go"]
        assert processor.check_exceptions("was", "v") == ["be"]
        assert processor.check_exceptions("were", "v") == ["be"]
        assert processor.check_exceptions("been", "v") == ["be"]
        assert processor.check_exceptions("flew", "v") == ["fly"]
        assert processor.check_exceptions("flown", "v") == ["fly"]

        # Test non-exception
        assert processor.check_exceptions("running", "v") == []

    def test_check_exceptions_adj(self, loader_with_data):
        """Test checking adjective exceptions."""
        processor = Morphy(loader_with_data)

        # Test irregular comparatives/superlatives
        assert processor.check_exceptions("better", "a") == ["good", "well"]
        assert processor.check_exceptions("best", "a") == ["good", "well"]
        assert processor.check_exceptions("worse", "a") == ["bad"]
        assert processor.check_exceptions("worst", "a") == ["bad"]

        # Test non-exception
        assert processor.check_exceptions("bigger", "a") == []

    def test_apply_rules_noun(self, loader_with_data):
        """Test applying noun morphological rules."""
        processor = Morphy(loader_with_data)

        # Test plural rules
        assert "dog" in processor.apply_rules("dogs", "n")
        assert "box" in processor.apply_rules("boxes", "n")
        assert "fly" in processor.apply_rules("flies", "n")

        # Test -ses rule
        candidates = processor.apply_rules("glasses", "n")
        assert "glass" in candidates

        # Test -ches rule
        candidates = processor.apply_rules("churches", "n")
        assert "church" in candidates

    def test_apply_rules_verb(self, loader_with_data):
        """Test applying verb morphological rules."""
        processor = Morphy(loader_with_data)

        # Test -s rule
        assert "run" in processor.apply_rules("runs", "v")

        # Test -ing rules
        candidates = processor.apply_rules("running", "v")
        assert "runn" in candidates  # -ing -> ""
        assert "run" in candidates  # doubled consonant removed
        assert "runne" in candidates  # -ing -> e

        # Test -ed rules
        candidates = processor.apply_rules("watched", "v")
        assert "watch" in candidates  # -ed -> ""
        assert "watche" in candidates  # -ed -> e

        # Test -ies rule
        assert "fly" in processor.apply_rules("flies", "v")

    def test_apply_rules_adj(self, loader_with_data):
        """Test applying adjective morphological rules."""
        processor = Morphy(loader_with_data)

        # Test comparative rules
        candidates = processor.apply_rules("bigger", "a")
        assert "bigg" in candidates  # -er -> ""
        assert "big" in candidates  # doubled consonant removed
        assert "bigge" in candidates  # -er -> e

        candidates = processor.apply_rules("biggest", "a")
        assert "bigg" in candidates  # -est -> ""
        assert "big" in candidates  # doubled consonant removed
        assert "bigge" in candidates  # -est -> e

        # Test -er/-est with e
        assert "nice" in processor.apply_rules("nicer", "a")
        assert "nice" in processor.apply_rules("nicest", "a")

    def test_morphy_noun(self, loader_with_data):
        """Test morphy for nouns."""
        processor = Morphy(loader_with_data)

        # Test regular plurals
        assert processor.morphy("dogs", "n") == ["dog"]
        assert processor.morphy("boxes", "n") == ["box"]
        assert processor.morphy("flies", "n") == ["fly"]

        # Test irregular plurals
        assert processor.morphy("children", "n") == ["child"]

        # Test base form
        assert processor.morphy("dog", "n") == ["dog"]

        # Test non-existent
        assert processor.morphy("nonexistent", "n") == []

    def test_morphy_verb(self, loader_with_data):
        """Test morphy for verbs."""
        processor = Morphy(loader_with_data)

        # Test regular forms
        assert processor.morphy("runs", "v") == ["run"]
        # Now "running" should correctly resolve to "run"
        assert processor.morphy("running", "v") == ["run"]
        assert processor.morphy("flies", "v") == ["fly"]
        assert processor.morphy("watches", "v") == ["watch"]

        # Test irregular forms
        assert processor.morphy("ran", "v") == ["run"]
        assert processor.morphy("flew", "v") == ["fly"]
        assert processor.morphy("was", "v") == ["be"]

        # Test base form
        assert processor.morphy("run", "v") == ["run"]

    def test_morphy_adj(self, loader_with_data):
        """Test morphy for adjectives."""
        processor = Morphy(loader_with_data)

        # Test regular forms
        assert processor.morphy("bigger", "a") == ["big"]
        assert processor.morphy("biggest", "a") == ["big"]
        assert processor.morphy("nicer", "a") == ["nice"]
        assert processor.morphy("nicest", "a") == ["nice"]

        # Test irregular forms
        lemmas = processor.morphy("better", "a")
        assert "good" in lemmas
        assert "well" in lemmas

        # Test base form
        assert processor.morphy("big", "a") == ["big"]

    def test_morphy_all_pos(self, loader_with_data):
        """Test morphy without specifying POS."""
        processor = Morphy(loader_with_data)

        # Word that exists as both noun and verb
        lemmas = processor.morphy("flies")
        assert "fly" in lemmas

        # Should find both noun and verb forms
        lemmas = processor.morphy("fly")
        assert "fly" in lemmas

    def test_morphy_lowercase(self, loader_with_data):
        """Test that morphy handles case properly."""
        processor = Morphy(loader_with_data)

        # Should lowercase input
        assert processor.morphy("DOGS", "n") == ["dog"]
        assert processor.morphy("Running", "v") == ["run"]
        assert processor.morphy("BETTER", "a") == ["good", "well"]

    def test_get_base_forms(self, loader_with_data):
        """Test getting all candidate base forms."""
        processor = Morphy(loader_with_data)

        # Should return all candidates, not just those in WordNet
        forms = processor.get_base_forms("running", "v")
        assert "running" in forms  # Original
        assert "run" in forms  # From rules
        assert "runn" in forms  # From rules (not in WordNet)
        assert "runne" in forms  # From rules (not in WordNet)

    def test_morphy_function(self, loader_with_data):
        """Test the convenience morphy function."""
        # Should work with loader
        lemmas = morphy("dogs", "n", loader_with_data)
        assert lemmas == ["dog"]

        # Should raise error without loader
        with pytest.raises(ValueError):
            morphy("dogs", "n", None)

    def test_ves_to_f_rule(self, loader_with_data):
        """Test the ves -> f/fe transformation for nouns."""
        processor = Morphy(loader_with_data)

        # Test ves -> f rule
        candidates = processor.apply_rules("knives", "n")
        assert "knife" in candidates  # ves -> fe
        assert "knif" in candidates  # ves -> f

        candidates = processor.apply_rules("wives", "n")
        assert "wife" in candidates  # ves -> fe
        assert "wif" in candidates  # ves -> f

    def test_period_removal(self, loader_with_data):
        """Test removal of periods from abbreviations."""
        processor = Morphy(loader_with_data)

        # If "dog." is passed, it should also try "dog"
        # Since "dog" exists in WordNet, it should be found
        lemmas = processor.morphy("dog.", "n")
        assert "dog" in lemmas

    def test_ful_suffix_handling(self):
        """Test special handling of nouns ending with 'ful'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            synsets = [
                {
                    "offset": "02883344",
                    "lex_filenum": 6,
                    "lex_filename": "noun.artifact",
                    "ss_type": "n",
                    "words": [{"lemma": "box", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "a container",
                },
                {
                    "offset": "13767879",
                    "lex_filenum": 23,
                    "lex_filename": "noun.quantity",
                    "ss_type": "n",
                    "words": [{"lemma": "boxful", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "the quantity contained in a box",
                },
            ]

            wordnet_file = _write_wordnet_files(data_path, synsets)
            loader = WordNetLoader(wordnet_file)
            loader.load()
            processor = Morphy(loader)

            # Test "boxesful" -> "boxful"
            lemmas = processor.morphy("boxesful", "n")
            assert "boxful" in lemmas

    def test_collocation_simple(self):
        """Test simple multi-word expressions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            synsets = [
                {
                    "offset": "09780632",
                    "lex_filenum": 15,
                    "lex_filename": "noun.person",
                    "ss_type": "n",
                    "words": [{"lemma": "attorney", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "a lawyer",
                },
                {
                    "offset": "10260706",
                    "lex_filenum": 15,
                    "lex_filename": "noun.person",
                    "ss_type": "n",
                    "words": [{"lemma": "general", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "a general officer",
                },
                {
                    "offset": "09781263",
                    "lex_filenum": 15,
                    "lex_filename": "noun.person",
                    "ss_type": "n",
                    "words": [{"lemma": "attorney_general", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "the chief law officer",
                },
            ]

            wordnet_file = _write_wordnet_files(data_path, synsets)
            loader = WordNetLoader(wordnet_file)
            loader.load()
            processor = Morphy(loader)

            # Test "attorneys general" -> "attorney general"
            lemmas = processor.morphy("attorneys general", "n")
            assert "attorney general" in lemmas

    def test_hyphenated_words(self):
        """Test hyphenated multi-word expressions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            synsets = [
                {
                    "offset": "10639637",
                    "lex_filenum": 15,
                    "lex_filename": "noun.person",
                    "ss_type": "n",
                    "words": [{"lemma": "son", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "a male offspring",
                },
                {
                    "offset": "10105733",
                    "lex_filenum": 15,
                    "lex_filename": "noun.person",
                    "ss_type": "n",
                    "words": [{"lemma": "son_in_law", "lex_id": 0}],
                    "pointers": [],
                    "gloss": "the husband of your daughter",
                },
            ]

            wordnet_file = _write_wordnet_files(data_path, synsets)
            loader = WordNetLoader(wordnet_file)
            loader.load()
            processor = Morphy(loader)

            # Test hyphenated form "sons-in-law" -> "son_in_law"
            # WordNet stores with underscores, but we handle hyphens
            lemmas = processor.morphy("sons-in-law", "n")
            # The result should be the base form with spaces (displayed form)
            assert "son in law" in lemmas or "son-in-law" in lemmas
