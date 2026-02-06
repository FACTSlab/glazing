"""Tests for WordNet loader module."""

import json
import tempfile
from pathlib import Path

import pytest

from glazing.wordnet.loader import WordNetLoader, load_wordnet


class TestWordNetLoader:
    """Test WordNet loader functionality."""

    @pytest.fixture
    def temp_data_file(self):
        """Create temporary directory with test data in single-file format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)

            # All synsets go into a single wordnet.jsonl file
            synsets_data = [
                {
                    "offset": "00001740",
                    "lex_filenum": 3,
                    "lex_filename": "noun.Tops",
                    "ss_type": "n",
                    "words": [{"lemma": "entity", "lex_id": 0}],
                    "pointers": [
                        {"symbol": "~", "offset": "00001930", "pos": "n", "source": 0, "target": 0}
                    ],
                    "gloss": (
                        "that which is perceived or known or inferred "
                        "to have its own distinct existence"
                    ),
                },
                {
                    "offset": "00001930",
                    "lex_filenum": 3,
                    "lex_filename": "noun.Tops",
                    "ss_type": "n",
                    "words": [{"lemma": "physical_entity", "lex_id": 0}],
                    "pointers": [
                        {"symbol": "@", "offset": "00001740", "pos": "n", "source": 0, "target": 0}
                    ],
                    "gloss": "an entity that has physical existence",
                },
                {
                    "offset": "00002325",
                    "lex_filenum": 29,
                    "lex_filename": "verb.body",
                    "ss_type": "v",
                    "words": [{"lemma": "run", "lex_id": 0}, {"lemma": "go", "lex_id": 1}],
                    "pointers": [],
                    "frames": [
                        {"frame_number": 1, "word_indices": [0]},
                        {"frame_number": 2, "word_indices": [0, 1]},
                    ],
                    "gloss": "move fast by using one's feet",
                },
            ]

            wordnet_file = data_path / "wordnet.jsonl"
            with open(wordnet_file, "w") as f:
                for synset in synsets_data:
                    f.write(json.dumps(synset) + "\n")

            # Create sense index (supplementary file alongside primary)
            sense_data = [
                {
                    "sense_key": "entity%1:03:00::",
                    "lemma": "entity",
                    "ss_type": "n",
                    "lex_filenum": 3,
                    "lex_id": 0,
                    "synset_offset": "00001740",
                    "sense_number": 1,
                    "tag_count": 0,
                },
                {
                    "sense_key": "run%2:38:00::",
                    "lemma": "run",
                    "ss_type": "v",
                    "lex_filenum": 38,
                    "lex_id": 0,
                    "synset_offset": "00002325",
                    "sense_number": 1,
                    "tag_count": 5,
                },
            ]

            with open(data_path / "wordnet_senses.jsonl", "w") as f:
                for sense in sense_data:
                    f.write(json.dumps(sense) + "\n")

            # Create exception entries (with pos field)
            exc_data = [
                {"inflected_form": "children", "base_forms": ["child"], "pos": "n"},
                {"inflected_form": "geese", "base_forms": ["goose"], "pos": "n"},
                {"inflected_form": "ran", "base_forms": ["run"], "pos": "v"},
            ]

            with open(data_path / "wordnet_exceptions.jsonl", "w") as f:
                for exc in exc_data:
                    f.write(json.dumps(exc) + "\n")

            yield wordnet_file

    def test_loader_initialization(self, temp_data_file):
        """Test loader initialization without autoload."""
        loader = WordNetLoader(temp_data_file, autoload=False)

        assert loader.data_path == temp_data_file
        assert loader.lazy is False
        assert loader.cache_size == 1000
        assert not loader._loaded
        assert len(loader.synsets) == 0

    def test_load_synsets(self, temp_data_file):
        """Test loading synsets from JSON Lines."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        # Check synsets loaded
        assert len(loader.synsets) == 3
        assert "00001740" in loader.synsets
        assert "00001930" in loader.synsets
        assert "00002325" in loader.synsets

        # Check synset content
        entity = loader.synsets["00001740"]
        assert entity.ss_type == "n"
        assert len(entity.words) == 1
        assert entity.words[0].lemma == "entity"
        assert len(entity.pointers) == 1

    def test_load_lemma_index(self, temp_data_file):
        """Test building lemma index from synset data."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        # Check lemma index built from synset words
        assert "entity" in loader.lemma_index
        assert "n" in loader.lemma_index["entity"]
        assert len(loader.lemma_index["entity"]["n"]) == 1

        # lemma_index values are SynsetOffset strings now
        offset = loader.lemma_index["entity"]["n"][0]
        assert offset == "00001740"

        # Check verb lemmas
        assert "run" in loader.lemma_index
        assert "v" in loader.lemma_index["run"]
        assert loader.lemma_index["run"]["v"][0] == "00002325"

        # "go" should also be indexed
        assert "go" in loader.lemma_index
        assert "v" in loader.lemma_index["go"]

    def test_load_sense_index(self, temp_data_file):
        """Test loading sense index."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        # Check sense index
        assert "entity%1:03:00::" in loader.sense_index
        sense = loader.sense_index["entity%1:03:00::"]
        assert sense.lemma == "entity"
        assert sense.synset_offset == "00001740"
        assert sense.sense_number == 1

    def test_load_exceptions(self, temp_data_file):
        """Test loading exception files."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        # Check noun exceptions
        assert "n" in loader.exceptions
        assert "children" in loader.exceptions["n"]
        assert loader.exceptions["n"]["children"] == ["child"]

        # Check verb exceptions
        assert "v" in loader.exceptions
        assert "ran" in loader.exceptions["v"]
        assert loader.exceptions["v"]["ran"] == ["run"]

    def test_build_relation_indices(self, temp_data_file):
        """Test building relation indices."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        # Check hypernym index
        assert "00001930" in loader.hypernym_index
        assert "00001740" in loader.hypernym_index["00001930"]

        # Check hyponym index
        assert "00001740" in loader.hyponym_index
        assert "00001930" in loader.hyponym_index["00001740"]

    def test_get_synset(self, temp_data_file):
        """Test getting synset by offset."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        synset = loader.get_synset("00001740")
        assert synset is not None
        assert synset.offset == "00001740"
        assert synset.words[0].lemma == "entity"

        # Test non-existent synset
        synset = loader.get_synset("99999999")
        assert synset is None

    def test_get_synsets_by_lemma(self, temp_data_file):
        """Test getting synsets by lemma."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        # Test noun
        synsets = loader.get_synsets_by_lemma("entity", "n")
        assert len(synsets) == 1
        assert synsets[0].offset == "00001740"

        # Test verb
        synsets = loader.get_synsets_by_lemma("run", "v")
        assert len(synsets) == 1
        assert synsets[0].offset == "00002325"

        # Test all POS
        synsets = loader.get_synsets_by_lemma("run")
        assert len(synsets) == 1

        # Test non-existent lemma
        synsets = loader.get_synsets_by_lemma("nonexistent")
        assert len(synsets) == 0

    def test_get_sense_by_key(self, temp_data_file):
        """Test getting sense by key."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        sense = loader.get_sense_by_key("entity%1:03:00::")
        assert sense is not None
        assert sense.lemma == "entity"
        assert sense.synset_offset == "00001740"

        # Test non-existent sense
        sense = loader.get_sense_by_key("nonexistent%1:00:00::")
        assert sense is None

    def test_get_senses_by_lemma(self, temp_data_file):
        """Test getting senses by lemma."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        senses = loader.get_senses_by_lemma("entity", "n")
        assert len(senses) == 1
        assert senses[0].sense_key == "entity%1:03:00::"

        senses = loader.get_senses_by_lemma("run", "v")
        assert len(senses) == 1
        assert senses[0].sense_key == "run%2:38:00::"

    def test_get_hypernyms(self, temp_data_file):
        """Test getting hypernyms."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        synset = loader.get_synset("00001930")
        hypernyms = loader.get_hypernyms(synset)
        assert len(hypernyms) == 1
        assert hypernyms[0].offset == "00001740"

    def test_get_hyponyms(self, temp_data_file):
        """Test getting hyponyms."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        synset = loader.get_synset("00001740")
        hyponyms = loader.get_hyponyms(synset)
        assert len(hyponyms) == 1
        assert hyponyms[0].offset == "00001930"

    def test_lazy_loading(self, temp_data_file):
        """Test lazy loading mode."""
        loader = WordNetLoader(temp_data_file, lazy=True, cache_size=2)
        loader.load()

        # Synsets should not be loaded yet
        assert len(loader.synsets) == 0

        # But file index should be built
        assert len(loader._synset_file_index) > 0

        # Loading a synset should work
        synset = loader.get_synset("00001740")
        assert synset is not None
        assert synset.offset == "00001740"

        # Check cache is working
        assert loader._cache is not None
        cached = loader._cache.get("00001740")
        assert cached is not None
        assert cached.offset == "00001740"

    def test_get_exceptions(self, temp_data_file):
        """Test getting morphological exceptions."""
        loader = WordNetLoader(temp_data_file)
        loader.load()

        noun_exc = loader.get_exceptions("n")
        assert "children" in noun_exc
        assert noun_exc["children"] == ["child"]

        verb_exc = loader.get_exceptions("v")
        assert "ran" in verb_exc
        assert verb_exc["ran"] == ["run"]

        # Non-existent POS
        adv_exc = loader.get_exceptions("r")
        assert len(adv_exc) == 0

    def test_load_wordnet_function(self, temp_data_file):
        """Test the convenience load_wordnet function."""
        wn = load_wordnet(temp_data_file)

        assert isinstance(wn, WordNetLoader)
        assert wn._loaded is True
        assert len(wn.synsets) > 0

        # Test functionality
        synset = wn.get_synset("00001740")
        assert synset is not None
