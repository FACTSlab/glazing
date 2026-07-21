"""Tests for adjective satellite handling and canonical synset keying.

Regression coverage for the converter dropping every ``ss_type == "s"`` synset
and for synsets being keyed by bare offset, which collides across parts of
speech. See https://github.com/factslab/glazing/issues/11.
"""

import pytest

from glazing.wordnet.converter import WordNetConverter
from glazing.wordnet.loader import WordNetLoader
from glazing.wordnet.models import Pointer, Synset, Word
from glazing.wordnet.relations import WordNetRelationTraverser
from glazing.wordnet.search import WordNetSearch
from glazing.wordnet.types import make_synset_key, normalize_pos

WN_LICENSE_HEADER = "  1 This software and database is being provided to you.\n"


def make_synset(offset: str, ss_type: str, lemma: str, **kwargs) -> Synset:
    """Build a minimal synset for testing."""
    defaults = {
        "lex_filenum": 0 if ss_type in ("a", "s") else 3,
        "lex_filename": "adj.all" if ss_type in ("a", "s") else "noun.Tops",
        "pointers": [],
        "gloss": f"gloss for {lemma}",
    }
    defaults.update(kwargs)
    return Synset(
        offset=offset,
        ss_type=ss_type,
        words=[Word(lemma=lemma, lex_id=0)],
        **defaults,
    )


class TestKeyPrimitives:
    """Tests for normalize_pos and make_synset_key."""

    @pytest.mark.parametrize(
        ("pos", "expected"),
        [("n", "n"), ("v", "v"), ("a", "a"), ("r", "r"), ("s", "a")],
    )
    def test_normalize_pos(self, pos, expected):
        """Satellites normalize to "a"; every other POS is unchanged."""
        assert normalize_pos(pos) == expected

    @pytest.mark.parametrize(
        ("offset", "pos", "expected"),
        [
            ("00001740", "n", "00001740n"),
            ("00001740", "v", "00001740v"),
            ("00014377", "a", "00014377a"),
            ("00014377", "s", "00014377a"),
            ("00001740", "r", "00001740r"),
        ],
    )
    def test_make_synset_key(self, offset, pos, expected):
        """Keys pair the offset with the normalized POS."""
        assert make_synset_key(offset, pos) == expected

    def test_satellite_record_and_pointer_agree(self):
        """A satellite's key matches the key of a pointer targeting it.

        WordNet writes "a" in the POS field of every pointer to an adjective,
        even when the target is a satellite, so this agreement is the whole
        point of normalizing.
        """
        satellite = make_synset("00014377", "s", "abounding")
        pointer = Pointer(symbol="&", offset="00014377", pos="a", source=0, target=0)

        assert satellite.ss_type == "s"
        assert satellite.key == pointer.target_key == "00014377a"

    def test_key_is_not_serialized(self):
        """key stays derived and out of the on-disk schema."""
        synset = make_synset("00014377", "s", "abounding")
        assert "key" not in synset.model_dump_json()
        assert Synset.model_validate_json(synset.model_dump_json()).key == synset.key


class TestConverterRetainsSatellites:
    """Tests that the converter no longer discards satellite synsets."""

    @pytest.fixture
    def converter(self):
        """Provide a converter instance."""
        return WordNetConverter()

    @pytest.fixture
    def data_adj(self, tmp_path):
        """Write a data.adj holding one head and two satellites."""
        path = tmp_path / "data.adj"
        path.write_text(
            WN_LICENSE_HEADER
            + "00001740 00 a 01 able 0 001 & 00014377 a 0000 | having the means\n"
            + "00014377 00 s 01 abounding 0 001 & 00001740 a 0000 | existing in abundance\n"
            + "00019769 00 s 01 handy 0 000 | easy to reach\n",
            encoding="utf-8",
        )
        return path

    def test_satellites_are_emitted(self, converter, data_adj):
        """The adjective pass returns heads and satellites alike."""
        synsets = converter.parse_data_file(data_adj, "a")

        assert len(synsets) == 3
        assert sorted(s.ss_type for s in synsets) == ["a", "s", "s"]

    def test_adjective_passes_are_equivalent(self, converter, data_adj):
        """ "a" and "s" both name the adjective data file, so both return all."""
        by_a = converter.parse_data_file(data_adj, "a")
        by_s = converter.parse_data_file(data_adj, "s")

        assert [s.key for s in by_a] == [s.key for s in by_s]

    def test_satellite_pointers_resolve(self, converter, data_adj):
        """No pointer among the emitted adjectives dangles."""
        synsets = converter.parse_data_file(data_adj, "a")
        keys = {s.key for s in synsets}

        targets = {p.target_key for s in synsets for p in s.pointers}
        assert targets <= keys

    def test_leading_punctuation_lemmas_survive(self, converter, tmp_path):
        """Lemmas opening with a dot or apostrophe no longer fail validation.

        A lemma that fails validation discards its entire synset, which lost
        ".22_caliber", "'hood", "'tween" and "'s_Gravenhage" from the output.
        """
        path = tmp_path / "data.adj"
        path.write_text(
            WN_LICENSE_HEADER
            + "03157978 01 a 02 .22_caliber 0 .22-caliber 0 000 | of a .22 bore\n"
            + "08659519 01 a 02 'hood 0 'tween 0 000 | elided forms\n",
            encoding="utf-8",
        )

        synsets = converter.parse_data_file(path, "a")
        assert len(synsets) == 2
        assert synsets[0].get_lemmas() == [".22_caliber", ".22-caliber"]
        assert synsets[1].get_lemmas() == ["'hood", "'tween"]


class TestSatelliteSenseEnrichment:
    """Tests that satellite senses receive sense numbers and tag counts."""

    def _wordnet_dir(self, tmp_path, index_sense, cntlist=""):
        """Write a minimal WordNet directory with two satellites."""
        wn_dir = tmp_path / "wn"
        wn_dir.mkdir()
        (wn_dir / "data.adj").write_text(
            WN_LICENSE_HEADER
            + "00014377 00 s 01 cardinal 0 000 | serving as a base\n"
            + "00019769 00 s 01 cardinal 0 000 | of a deep red\n",
            encoding="utf-8",
        )
        for name in ("data.noun", "data.verb", "data.adv"):
            (wn_dir / name).write_text(WN_LICENSE_HEADER, encoding="utf-8")
        (wn_dir / "index.sense").write_text(index_sense, encoding="utf-8")
        (wn_dir / "cntlist").write_text(cntlist, encoding="utf-8")
        return wn_dir

    def test_satellite_senses_are_enriched(self, tmp_path):
        """Satellites get sense_number and tag_count despite head-word keys.

        Both satellites share the head-less prefix ``cardinal%5:00:00:``, so a
        prefix match could not tell them apart; keying on the synset offset can.
        """
        wn_dir = self._wordnet_dir(
            tmp_path,
            index_sense=(
                "cardinal%5:00:00:fundamental:00 00014377 1 7\n"
                "cardinal%5:00:00:red:00 00019769 2 3\n"
            ),
        )
        out = tmp_path / "wordnet.jsonl"
        WordNetConverter().convert_wordnet_database(wn_dir, out)

        loader = WordNetLoader(out, autoload=False)
        loader.load()

        base = loader.synsets["00014377a"].words[0]
        assert (base.sense_number, base.tag_count) == (1, 7)

        red = loader.synsets["00019769a"].words[0]
        assert (red.sense_number, red.tag_count) == (2, 3)

    def test_cntlist_overrides_the_right_satellite(self, tmp_path):
        """A cntlist count lands on the satellite its sense key names."""
        wn_dir = self._wordnet_dir(
            tmp_path,
            index_sense=(
                "cardinal%5:00:00:fundamental:00 00014377 1 7\n"
                "cardinal%5:00:00:red:00 00019769 2 3\n"
            ),
            cntlist="99 cardinal%5:00:00:red:00 1\n",
        )
        out = tmp_path / "wordnet.jsonl"
        WordNetConverter().convert_wordnet_database(wn_dir, out)

        loader = WordNetLoader(out, autoload=False)
        loader.load()

        assert loader.synsets["00019769a"].words[0].tag_count == 99
        assert loader.synsets["00014377a"].words[0].tag_count == 7


class TestCrossPOSOffsetCollision:
    """Tests that synsets sharing an offset across POS both survive."""

    @pytest.fixture
    def colliding(self):
        """Provide a noun and a verb sharing offset 00001740."""
        return [
            make_synset("00001740", "n", "entity"),
            make_synset("00001740", "v", "breathe", lex_filenum=29, lex_filename="verb.body"),
        ]

    def test_search_accepts_both(self, colliding):
        """Indexing both no longer raises on the duplicate offset."""
        search = WordNetSearch(synsets=colliding)

        assert search.by_offset("00001740", "n").get_lemmas() == ["entity"]
        assert search.by_offset("00001740", "v").get_lemmas() == ["breathe"]

    def test_search_bare_offset_is_ambiguous(self, colliding):
        """An unqualified colliding offset resolves to nothing, not a guess."""
        search = WordNetSearch(synsets=colliding)
        assert search.by_offset("00001740") is None

    def test_loader_keeps_both(self, tmp_path, colliding):
        """Neither synset overwrites the other in the loader."""
        path = tmp_path / "wordnet.jsonl"
        path.write_text("".join(f"{s.model_dump_json()}\n" for s in colliding), encoding="utf-8")

        loader = WordNetLoader(path, autoload=False)
        loader.load()

        assert len(loader.synsets) == 2
        assert loader.get_synset("00001740", "n").get_lemmas() == ["entity"]
        assert loader.get_synset("00001740v").get_lemmas() == ["breathe"]
        assert loader.get_synset("00001740") is None


class TestSatelliteReachability:
    """Tests that satellites are findable through the adjective POS."""

    @pytest.fixture
    def satellite(self):
        """Provide a lone satellite synset."""
        return make_synset("00014377", "s", "abounding")

    def test_search_by_lemma(self, satellite):
        """A satellite answers to "a", to "s", and to no filter at all."""
        search = WordNetSearch(synsets=[satellite])

        assert [s.key for s in search.by_lemma("abounding", "a")] == ["00014377a"]
        assert [s.key for s in search.by_lemma("abounding", "s")] == ["00014377a"]
        assert [s.key for s in search.by_lemma("abounding")] == ["00014377a"]

    def test_search_by_offset_pos_filter(self, satellite):
        """ "a" reaches the satellite; ss_type still reports "s"."""
        search = WordNetSearch(synsets=[satellite])

        found = search.by_offset("00014377", "a")
        assert found is not None
        assert found.ss_type == "s"

    def test_search_pos_s_excludes_heads(self):
        """A "s" filter does not return adjective heads."""
        head = make_synset("00001740", "a", "able")
        search = WordNetSearch(synsets=[head])

        assert search.by_offset("00001740", "a") is not None
        assert search.by_offset("00001740", "s") is None

    def test_statistics_do_not_double_count(self, satellite):
        """Dual indexing under "s" and "a" must not inflate the counts."""
        head = make_synset("00001740", "a", "able")
        stats = WordNetSearch(synsets=[satellite, head]).get_statistics()

        assert stats["synset_count"] == 2
        assert stats["s_synsets"] == 1
        assert stats["a_synsets"] == 1

    def test_loader_by_lemma(self, tmp_path, satellite):
        """The loader reaches satellites from "a" without duplicating them."""
        path = tmp_path / "wordnet.jsonl"
        path.write_text(f"{satellite.model_dump_json()}\n", encoding="utf-8")

        loader = WordNetLoader(path, autoload=False)
        loader.load()

        assert [s.key for s in loader.get_synsets_by_lemma("abounding", "a")] == ["00014377a"]
        assert [s.key for s in loader.get_synsets_by_lemma("abounding", "s")] == ["00014377a"]
        assert [s.key for s in loader.get_synsets_by_lemma("abounding")] == ["00014377a"]


class TestSatelliteRelations:
    """Tests that relations traverse into satellites."""

    @pytest.fixture
    def head_and_satellite(self):
        """Provide an adjective head and the satellite it points at."""
        head = make_synset(
            "00001740",
            "a",
            "able",
            pointers=[Pointer(symbol="&", offset="00014377", pos="a", source=0, target=0)],
        )
        satellite = make_synset(
            "00014377",
            "s",
            "abounding",
            pointers=[Pointer(symbol="&", offset="00001740", pos="a", source=0, target=0)],
        )
        return head, satellite

    def test_similar_to_resolves(self, head_and_satellite):
        """A similar-to pointer reaches the satellite it names."""
        head, satellite = head_and_satellite
        traverser = WordNetRelationTraverser({s.key: s for s in (head, satellite)})

        assert [s.key for s in traverser.get_similar_to(head)] == ["00014377a"]
        assert [s.key for s in traverser.get_similar_to(satellite)] == ["00001740a"]

    def test_similarity_spans_head_and_satellite(self, head_and_satellite):
        """Heads and satellites count as one POS for path similarity."""
        head, satellite = head_and_satellite
        traverser = WordNetRelationTraverser({s.key: s for s in (head, satellite)})

        # Distinct ss_type values ("a" vs "s") must not short-circuit to 0.0.
        assert traverser.calculate_path_similarity(head, head) == 1.0
        assert traverser.calculate_path_similarity(satellite, satellite) == 1.0

    def test_relations_respect_pos(self):
        """A pointer resolves by POS, not by whichever synset shares an offset."""
        noun = make_synset(
            "00002325",
            "n",
            "entity",
            pointers=[Pointer(symbol="@", offset="00001740", pos="n", source=0, target=0)],
        )
        noun_target = make_synset("00001740", "n", "thing")
        verb_decoy = make_synset(
            "00001740", "v", "breathe", lex_filenum=29, lex_filename="verb.body"
        )
        traverser = WordNetRelationTraverser({s.key: s for s in (noun, noun_target, verb_decoy)})

        hypernyms = traverser.get_hypernyms(noun)
        assert [s.get_lemmas() for s in hypernyms] == [["thing"]]
