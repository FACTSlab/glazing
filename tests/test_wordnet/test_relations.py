"""Tests for WordNet relation traversal functionality."""

import pytest

from glazing.wordnet.models import Pointer, Synset, VerbFrame, Word
from glazing.wordnet.relations import WordNetRelationTraverser


class TestWordNetRelationTraverser:
    """Tests for WordNetRelationTraverser class."""

    @pytest.fixture
    def sample_synsets_dict(self):
        """Create sample synsets dictionary for testing relations."""
        synsets = {}

        # Create entity synset (root)
        entity_synset = Synset(
            offset="00001740",
            lex_filenum=3,
            lex_filename="noun.Tops",
            ss_type="n",
            words=[Word(lemma="entity", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="~", offset="00002137", pos="n", source=0, target=0
                ),  # hyponym: physical_entity
                Pointer(
                    symbol="~", offset="00001930", pos="n", source=0, target=0
                ),  # hyponym: abstraction
            ],
            gloss="that which is perceived or known or inferred to have its own distinct existence",
        )
        synsets[entity_synset.key] = entity_synset

        # Create physical_entity synset
        physical_synset = Synset(
            offset="00002137",
            lex_filenum=3,
            lex_filename="noun.Tops",
            ss_type="n",
            words=[Word(lemma="physical_entity", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="@", offset="00001740", pos="n", source=0, target=0
                ),  # hypernym: entity
                Pointer(
                    symbol="~", offset="00002684", pos="n", source=0, target=0
                ),  # hyponym: object
            ],
            gloss="an entity that has physical existence",
        )
        synsets[physical_synset.key] = physical_synset

        # Create object synset
        object_synset = Synset(
            offset="00002684",
            lex_filenum=3,
            lex_filename="noun.Tops",
            ss_type="n",
            words=[Word(lemma="object", lex_id=0), Word(lemma="physical_object", lex_id=1)],
            pointers=[
                Pointer(
                    symbol="@", offset="00002137", pos="n", source=0, target=0
                ),  # hypernym: physical_entity
                Pointer(
                    symbol="~", offset="00015388", pos="n", source=0, target=0
                ),  # hyponym: whole
                Pointer(
                    symbol="~", offset="02083346", pos="n", source=0, target=0
                ),  # hyponym: living_thing
            ],
            gloss="a tangible and visible entity",
        )
        synsets[object_synset.key] = object_synset

        # Create living_thing synset
        living_synset = Synset(
            offset="02083346",
            lex_filenum=3,
            lex_filename="noun.Tops",
            ss_type="n",
            words=[Word(lemma="living_thing", lex_id=0), Word(lemma="animate_thing", lex_id=1)],
            pointers=[
                Pointer(
                    symbol="@", offset="00002684", pos="n", source=0, target=0
                ),  # hypernym: object
                Pointer(symbol="~", offset="02084442", pos="n", source=0, target=0),  # hyponym: dog
                Pointer(symbol="~", offset="02121620", pos="n", source=0, target=0),  # hyponym: cat
                Pointer(
                    symbol="#p", offset="05220461", pos="n", source=0, target=0
                ),  # part holonym: body
            ],
            gloss="a living entity",
        )
        synsets[living_synset.key] = living_synset

        # Create dog synset
        dog_synset = Synset(
            offset="02084442",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=[Word(lemma="dog", lex_id=0), Word(lemma="domestic_dog", lex_id=1)],
            pointers=[
                Pointer(
                    symbol="@", offset="02083346", pos="n", source=0, target=0
                ),  # hypernym: living_thing
                Pointer(
                    symbol="#m", offset="08008335", pos="n", source=0, target=0
                ),  # member holonym: pack
                Pointer(
                    symbol="%p", offset="02159955", pos="n", source=0, target=0
                ),  # part meronym: paw
            ],
            gloss="a member of the genus Canis",
        )
        synsets[dog_synset.key] = dog_synset

        # Create cat synset
        cat_synset = Synset(
            offset="02121620",
            lex_filenum=5,
            lex_filename="noun.animal",
            ss_type="n",
            words=[Word(lemma="cat", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="@", offset="02083346", pos="n", source=0, target=0
                ),  # hypernym: living_thing
            ],
            gloss="feline mammal",
        )
        synsets[cat_synset.key] = cat_synset

        # Create pack synset (for holonym)
        pack_synset = Synset(
            offset="08008335",
            lex_filenum=14,
            lex_filename="noun.group",
            ss_type="n",
            words=[Word(lemma="pack", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="%m", offset="02084442", pos="n", source=0, target=0
                ),  # member meronym: dog
            ],
            gloss="a group of hunting animals",
        )
        synsets[pack_synset.key] = pack_synset

        # Create paw synset (for meronym)
        paw_synset = Synset(
            offset="02159955",
            lex_filenum=8,
            lex_filename="noun.body",
            ss_type="n",
            words=[Word(lemma="paw", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="#p", offset="02084442", pos="n", source=0, target=0
                ),  # part holonym: dog
            ],
            gloss="a clawed foot of an animal",
        )
        synsets[paw_synset.key] = paw_synset

        # Create run synset (verb)
        run_synset = Synset(
            offset="02092002",
            lex_filenum=30,
            lex_filename="verb.motion",
            ss_type="v",
            words=[Word(lemma="run", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="@", offset="01835496", pos="v", source=0, target=0
                ),  # hypernym: travel
                Pointer(
                    symbol="*", offset="02092309", pos="v", source=0, target=0
                ),  # entailment: move
                Pointer(symbol=">", offset="02093321", pos="v", source=0, target=0),  # cause: rush
                Pointer(symbol="$", offset="02091115", pos="v", source=0, target=0),  # verb group
            ],
            frames=[VerbFrame(frame_number=1, word_indices=[0])],
            gloss="move fast by using one's feet",
        )
        synsets[run_synset.key] = run_synset

        # Create travel synset (verb hypernym)
        travel_synset = Synset(
            offset="01835496",
            lex_filenum=30,
            lex_filename="verb.motion",
            ss_type="v",
            words=[Word(lemma="travel", lex_id=0), Word(lemma="go", lex_id=1)],
            pointers=[
                Pointer(symbol="~", offset="02092002", pos="v", source=0, target=0),  # hyponym: run
            ],
            gloss="change location; move, travel, or proceed",
        )
        synsets[travel_synset.key] = travel_synset

        # Create move synset (entailment)
        move_synset = Synset(
            offset="02092309",
            lex_filenum=30,
            lex_filename="verb.motion",
            ss_type="v",
            words=[Word(lemma="move", lex_id=0)],
            pointers=[],
            gloss="change position",
        )
        synsets[move_synset.key] = move_synset

        # Create rush synset (cause)
        rush_synset = Synset(
            offset="02093321",
            lex_filenum=30,
            lex_filename="verb.motion",
            ss_type="v",
            words=[Word(lemma="rush", lex_id=0)],
            pointers=[],
            gloss="move fast",
        )
        synsets[rush_synset.key] = rush_synset

        # Create good synset (adjective)
        good_synset = Synset(
            offset="01123148",
            lex_filenum=0,
            lex_filename="adj.all",
            ss_type="a",
            words=[Word(lemma="good", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="!", offset="01125429", pos="a", source=1, target=1
                ),  # antonym to bad
                Pointer(
                    symbol="&", offset="01124073", pos="a", source=0, target=0
                ),  # similar to nice
                Pointer(symbol="^", offset="01126456", pos="a", source=0, target=0),  # also see
                Pointer(
                    symbol="+", offset="05145118", pos="n", source=1, target=1
                ),  # derivation: goodness
            ],
            gloss="having desirable or positive qualities",
        )
        synsets[good_synset.key] = good_synset

        # Create bad synset (antonym)
        bad_synset = Synset(
            offset="01125429",
            lex_filenum=0,
            lex_filename="adj.all",
            ss_type="a",
            words=[Word(lemma="bad", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="!", offset="01123148", pos="a", source=1, target=1
                ),  # antonym to good
            ],
            gloss="having undesirable or negative qualities",
        )
        synsets[bad_synset.key] = bad_synset

        # Create nice synset (similar)
        nice_synset = Synset(
            offset="01124073",
            lex_filenum=0,
            lex_filename="adj.all",
            ss_type="a",
            words=[Word(lemma="nice", lex_id=0)],
            pointers=[],
            gloss="pleasant or pleasing",
        )
        synsets[nice_synset.key] = nice_synset

        # Create goodness synset (derivation)
        goodness_synset = Synset(
            offset="05145118",
            lex_filenum=7,
            lex_filename="noun.attribute",
            ss_type="n",
            words=[Word(lemma="goodness", lex_id=0)],
            pointers=[
                Pointer(
                    symbol="+", offset="01123148", pos="a", source=1, target=1
                ),  # derivation: good
            ],
            gloss="the quality of being good",
        )
        synsets[goodness_synset.key] = goodness_synset

        return synsets

    def test_get_hypernyms_direct(self, sample_synsets_dict):
        """Test getting direct hypernyms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        dog_synset = sample_synsets_dict["02084442n"]

        hypernyms = traverser.get_hypernyms(dog_synset, direct_only=True)
        assert len(hypernyms) == 1
        assert hypernyms[0].offset == "02083346"  # living_thing

    def test_get_hypernyms_all(self, sample_synsets_dict):
        """Test getting all hypernyms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        dog_synset = sample_synsets_dict["02084442n"]

        hypernyms = traverser.get_hypernyms(dog_synset, direct_only=False)
        assert len(hypernyms) == 4
        offsets = [h.offset for h in hypernyms]
        assert "02083346" in offsets  # living_thing
        assert "00002684" in offsets  # object
        assert "00002137" in offsets  # physical_entity
        assert "00001740" in offsets  # entity

    def test_get_hyponyms_direct(self, sample_synsets_dict):
        """Test getting direct hyponyms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        living_synset = sample_synsets_dict["02083346n"]

        hyponyms = traverser.get_hyponyms(living_synset, direct_only=True)
        assert len(hyponyms) == 2
        offsets = [h.offset for h in hyponyms]
        assert "02084442" in offsets  # dog
        assert "02121620" in offsets  # cat

    def test_get_hyponyms_all(self, sample_synsets_dict):
        """Test getting all hyponyms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        object_synset = sample_synsets_dict["00002684n"]

        hyponyms = traverser.get_hyponyms(object_synset, direct_only=False)
        assert len(hyponyms) == 3
        offsets = [h.offset for h in hyponyms]
        assert "02083346" in offsets  # living_thing
        assert "02084442" in offsets  # dog
        assert "02121620" in offsets  # cat

    def test_get_hypernym_paths(self, sample_synsets_dict):
        """Test getting hypernym paths to root."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        dog_synset = sample_synsets_dict["02084442n"]

        paths = traverser.get_hypernym_paths(dog_synset, max_depth=10)
        assert len(paths) == 1  # Only one path to root

        path = paths[0]
        assert len(path) == 5
        assert path[0].offset == "02084442"  # dog
        assert path[1].offset == "02083346"  # living_thing
        assert path[2].offset == "00002684"  # object
        assert path[3].offset == "00002137"  # physical_entity
        assert path[4].offset == "00001740"  # entity

    def test_get_common_hypernyms(self, sample_synsets_dict):
        """Test finding common hypernyms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        dog_synset = sample_synsets_dict["02084442n"]
        cat_synset = sample_synsets_dict["02121620n"]

        common = traverser.get_common_hypernyms(dog_synset, cat_synset)
        assert len(common) == 4
        offsets = [c.offset for c in common]
        assert "02083346" in offsets  # living_thing (lowest common hypernym)
        assert "00002684" in offsets  # object
        assert "00002137" in offsets  # physical_entity
        assert "00001740" in offsets  # entity

    def test_get_meronyms(self, sample_synsets_dict):
        """Test getting meronyms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        dog_synset = sample_synsets_dict["02084442n"]

        # Get all meronyms
        meronyms = traverser.get_meronyms(dog_synset)
        assert len(meronyms) == 1
        assert meronyms[0].offset == "02159955"  # paw

        # Get part meronyms
        part_meronyms = traverser.get_meronyms(dog_synset, meronym_type="part")
        assert len(part_meronyms) == 1
        assert part_meronyms[0].offset == "02159955"

        # Get member meronyms (none for dog)
        member_meronyms = traverser.get_meronyms(dog_synset, meronym_type="member")
        assert len(member_meronyms) == 0

    def test_get_holonyms(self, sample_synsets_dict):
        """Test getting holonyms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        dog_synset = sample_synsets_dict["02084442n"]

        # Get all holonyms
        holonyms = traverser.get_holonyms(dog_synset)
        assert len(holonyms) == 1
        assert holonyms[0].offset == "08008335"  # pack

        # Get member holonyms
        member_holonyms = traverser.get_holonyms(dog_synset, holonym_type="member")
        assert len(member_holonyms) == 1
        assert member_holonyms[0].offset == "08008335"

        # Get part holonyms (none for dog)
        part_holonyms = traverser.get_holonyms(dog_synset, holonym_type="part")
        assert len(part_holonyms) == 0

    def test_get_entailments(self, sample_synsets_dict):
        """Test getting entailments for verbs."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        run_synset = sample_synsets_dict["02092002v"]

        entailments = traverser.get_entailments(run_synset)
        assert len(entailments) == 1
        assert entailments[0].offset == "02092309"  # move

        # Non-verb synset
        dog_synset = sample_synsets_dict["02084442n"]
        entailments = traverser.get_entailments(dog_synset)
        assert len(entailments) == 0

    def test_get_causes(self, sample_synsets_dict):
        """Test getting causes for verbs."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        run_synset = sample_synsets_dict["02092002v"]

        causes = traverser.get_causes(run_synset)
        assert len(causes) == 1
        assert causes[0].offset == "02093321"  # rush

        # Non-verb synset
        dog_synset = sample_synsets_dict["02084442n"]
        causes = traverser.get_causes(dog_synset)
        assert len(causes) == 0

    def test_get_similar_to(self, sample_synsets_dict):
        """Test getting similar adjectives."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        good_synset = sample_synsets_dict["01123148a"]

        similar = traverser.get_similar_to(good_synset)
        assert len(similar) == 1
        assert similar[0].offset == "01124073"  # nice

        # Non-adjective synset
        dog_synset = sample_synsets_dict["02084442n"]
        similar = traverser.get_similar_to(dog_synset)
        assert len(similar) == 0

    def test_get_also_see(self, sample_synsets_dict):
        """Test getting also-see relations."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        good_synset = sample_synsets_dict["01123148a"]

        also_see = traverser.get_also_see(good_synset)
        assert len(also_see) == 0  # No also-see in our test data

    def test_get_antonyms(self, sample_synsets_dict):
        """Test getting antonyms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        good_synset = sample_synsets_dict["01123148a"]

        # Get all antonyms
        antonyms = traverser.get_antonyms(good_synset)
        assert len(antonyms) == 1
        assert antonyms[0][0].offset == "01125429"  # bad synset
        assert antonyms[0][1] == "bad"  # bad lemma

    def test_get_derivations(self, sample_synsets_dict):
        """Test getting derivationally related forms."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        good_synset = sample_synsets_dict["01123148a"]

        # Get derivations
        derivations = traverser.get_derivations(good_synset)
        assert len(derivations) == 1
        assert derivations[0][0].offset == "05145118"  # goodness synset
        assert derivations[0][1] == "goodness"  # goodness lemma

    def test_calculate_path_similarity(self, sample_synsets_dict):
        """Test calculating path similarity."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        dog_synset = sample_synsets_dict["02084442n"]
        cat_synset = sample_synsets_dict["02121620n"]

        # Dog and cat similarity
        similarity = traverser.calculate_path_similarity(dog_synset, cat_synset)
        assert 0 < similarity < 1  # Should be similar but not identical

        # Same synset
        similarity = traverser.calculate_path_similarity(dog_synset, dog_synset)
        assert similarity == 1.0

        # Different POS
        run_synset = sample_synsets_dict["02092002v"]
        similarity = traverser.calculate_path_similarity(dog_synset, run_synset)
        assert similarity == 0.0

    def test_calculate_depth(self, sample_synsets_dict):
        """Test calculating synset depth."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)

        # Entity (root) has depth 0
        entity_synset = sample_synsets_dict["00001740n"]
        depth = traverser.calculate_depth(entity_synset)
        assert depth == 0

        # Dog has depth 4
        dog_synset = sample_synsets_dict["02084442n"]
        depth = traverser.calculate_depth(dog_synset)
        assert depth == 4

    def test_get_verb_groups(self, sample_synsets_dict):
        """Test getting verb groups."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        run_synset = sample_synsets_dict["02092002v"]

        groups = traverser.get_verb_groups(run_synset)
        assert len(groups) == 0  # No verb group target in our test data

        # Non-verb synset
        dog_synset = sample_synsets_dict["02084442n"]
        groups = traverser.get_verb_groups(dog_synset)
        assert len(groups) == 0

    def test_get_all_relations(self, sample_synsets_dict):
        """Test getting all relations for a synset."""
        traverser = WordNetRelationTraverser(sample_synsets_dict)
        dog_synset = sample_synsets_dict["02084442n"]

        relations = traverser.get_all_relations(dog_synset)

        assert "hypernyms" in relations
        assert len(relations["hypernyms"]) == 1

        assert "meronyms" in relations
        assert len(relations["meronyms"]) == 1

        assert "holonyms" in relations
        assert len(relations["holonyms"]) == 1

        # Should not have verb-specific relations
        assert "entailments" not in relations
        assert "causes" not in relations
