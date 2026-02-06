"""WordNet database file parser.

This module provides parsing functionality for WordNet 3.1 database files
including index files, data files, sense index, and exception files.

Classes
-------
WordNetConverter
    Parse WordNet database files into JSON Lines format.

Functions
---------
parse_index_file
    Parse WordNet index file (index.noun, index.verb, etc.).
parse_data_file
    Parse WordNet data file (data.noun, data.verb, etc.).
parse_sense_index
    Parse WordNet sense index file.
parse_exception_file
    Parse morphological exception file.

Examples
--------
>>> from pathlib import Path
>>> from glazing.wordnet.converter import WordNetConverter
>>> converter = WordNetConverter()
>>> synsets = converter.parse_data_file("data.noun")
>>> index_entries = converter.parse_index_file("index.verb")

>>> # Convert entire WordNet database
>>> converter.convert_wordnet_database(
...     wordnet_dir="wordnet31/dict",
...     output_dir="wordnet_jsonl"
... )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, cast

from glazing.wordnet.models import (
    ExceptionEntry,
    IndexEntry,
    Pointer,
    Sense,
    Synset,
    VerbFrame,
    Word,
)
from glazing.wordnet.types import (
    LexFileName,
    PointerSymbol,
    VerbFrameNumber,
    WordNetPOS,
)


class WordNetConverter:
    """Parse WordNet database files into structured models.

    Handles parsing of WordNet 3.1 database files including index files,
    data files, sense index, and morphological exception files.

    Methods
    -------
    parse_data_file(filepath, pos)
        Parse WordNet data file into list of Synset models.
    parse_index_file(filepath, pos)
        Parse WordNet index file into list of IndexEntry models.
    parse_sense_index(filepath)
        Parse sense index file into list of Sense models.
    parse_exception_file(filepath)
        Parse morphological exception file.
    convert_wordnet_database(wordnet_dir, output_dir)
        Convert entire WordNet database to JSON Lines.
    """

    # Mapping from lexical file numbers to names
    LEX_FILE_NAMES: ClassVar[dict[int, LexFileName]] = {
        0: "adj.all",
        1: "adj.pert",
        2: "adj.ppl",
        3: "adv.all",
        4: "noun.Tops",
        5: "noun.act",
        6: "noun.animal",
        7: "noun.artifact",
        8: "noun.attribute",
        9: "noun.body",
        10: "noun.cognition",
        11: "noun.communication",
        12: "noun.event",
        13: "noun.feeling",
        14: "noun.food",
        15: "noun.group",
        16: "noun.location",
        17: "noun.motive",
        18: "noun.object",
        19: "noun.person",
        20: "noun.phenomenon",
        21: "noun.plant",
        22: "noun.possession",
        23: "noun.process",
        24: "noun.quantity",
        25: "noun.relation",
        26: "noun.shape",
        27: "noun.state",
        28: "noun.substance",
        29: "noun.time",
        30: "verb.body",
        31: "verb.change",
        32: "verb.cognition",
        33: "verb.communication",
        34: "verb.competition",
        35: "verb.consumption",
        36: "verb.contact",
        37: "verb.creation",
        38: "verb.emotion",
        39: "verb.motion",
        40: "verb.perception",
        41: "verb.possession",
        42: "verb.social",
        43: "verb.stative",
        44: "verb.weather",
    }

    def parse_data_file(self, filepath: Path | str, pos: WordNetPOS) -> list[Synset]:
        """Parse WordNet data file into list of Synset models.

        Parameters
        ----------
        filepath : Path | str
            Path to WordNet data file (e.g., data.noun).
        pos : WordNetPOS
            Part of speech for validation.

        Returns
        -------
        list[Synset]
            List of parsed Synset models.

        Raises
        ------
        FileNotFoundError
            If the data file does not exist.
        ValueError
            If line format is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"WordNet data file not found: {filepath}"
            raise FileNotFoundError(msg)

        synsets = []

        with filepath.open("r", encoding="utf-8") as f:
            for line_num, line_raw in enumerate(f, 1):
                # Skip license header (lines starting with two spaces)
                if line_raw.startswith("  "):
                    continue

                line = line_raw.strip()
                if not line:
                    continue

                try:
                    synset = self._parse_data_line(line)
                    if synset and synset.ss_type == pos:
                        synsets.append(synset)
                except ValueError as e:
                    # Log parsing error but continue
                    print(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue

        return synsets

    def parse_index_file(self, filepath: Path | str, pos: WordNetPOS) -> list[IndexEntry]:
        """Parse WordNet index file into list of IndexEntry models.

        Parameters
        ----------
        filepath : Path | str
            Path to WordNet index file (e.g., index.noun).
        pos : WordNetPOS
            Part of speech for validation.

        Returns
        -------
        list[IndexEntry]
            List of parsed IndexEntry models.

        Raises
        ------
        FileNotFoundError
            If the index file does not exist.
        ValueError
            If line format is invalid.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"WordNet index file not found: {filepath}"
            raise FileNotFoundError(msg)

        entries = []

        with filepath.open("r", encoding="utf-8") as f:
            for line_num, line_raw in enumerate(f, 1):
                # Skip license header (lines starting with two spaces)
                if line_raw.startswith("  "):
                    continue

                line = line_raw.strip()
                if not line:
                    continue

                try:
                    entry = self._parse_index_line(line, pos)
                    if entry:
                        entries.append(entry)
                except ValueError as e:
                    print(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue

        return entries

    def parse_sense_index(self, filepath: Path | str) -> list[Sense]:
        """Parse WordNet sense index file.

        Parameters
        ----------
        filepath : Path | str
            Path to sense index file (index.sense).

        Returns
        -------
        list[Sense]
            List of parsed Sense models.

        Raises
        ------
        FileNotFoundError
            If the sense index file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"WordNet sense index file not found: {filepath}"
            raise FileNotFoundError(msg)

        senses = []

        with filepath.open("r", encoding="utf-8") as f:
            for line_num, line_raw in enumerate(f, 1):
                # Skip license header
                if line_raw.startswith("  "):
                    continue

                line = line_raw.strip()
                if not line:
                    continue

                try:
                    sense = self._parse_sense_line(line)
                    if sense:
                        senses.append(sense)
                except ValueError as e:
                    print(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue

        return senses

    def parse_exception_file(self, filepath: Path | str) -> list[ExceptionEntry]:
        """Parse morphological exception file.

        Parameters
        ----------
        filepath : Path | str
            Path to exception file (e.g., verb.exc).

        Returns
        -------
        list[ExceptionEntry]
            List of parsed ExceptionEntry models.

        Raises
        ------
        FileNotFoundError
            If the exception file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            msg = f"WordNet exception file not found: {filepath}"
            raise FileNotFoundError(msg)

        entries = []

        with filepath.open("r", encoding="utf-8") as f:
            for line_num, line_raw in enumerate(f, 1):
                # Skip license header
                if line_raw.startswith("  "):
                    continue

                line = line_raw.strip()
                if not line:
                    continue

                try:
                    entry = self._parse_exception_line(line)
                    if entry:
                        entries.append(entry)
                except ValueError as e:
                    print(f"Error parsing line {line_num} in {filepath}: {e}")
                    continue

        return entries

    def parse_verb_framestext(self, filepath: Path | str) -> dict[int, str]:
        """Parse verb.Framestext into a mapping of frame number to template string.

        Parameters
        ----------
        filepath : Path | str
            Path to verb.Framestext file.

        Returns
        -------
        dict[int, str]
            Mapping from frame number to template string.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return {}

        frames: dict[int, str] = {}

        with filepath.open("r", encoding="utf-8") as f:
            for line_raw in f:
                line = line_raw.strip()
                if not line:
                    continue

                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue

                try:
                    frame_num = int(parts[0])
                    template = parts[1]
                    frames[frame_num] = template
                except ValueError:
                    continue

        return frames

    def parse_verb_sentences(self, filepath: Path | str) -> dict[int, str]:
        """Parse sents.vrb into a mapping of frame number to example sentence.

        Parameters
        ----------
        filepath : Path | str
            Path to sents.vrb file.

        Returns
        -------
        dict[int, str]
            Mapping from frame number to example sentence.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return {}

        sentences: dict[int, str] = {}

        with filepath.open("r", encoding="utf-8") as f:
            for line_raw in f:
                line = line_raw.strip()
                if not line:
                    continue

                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue

                try:
                    sent_num = int(parts[0])
                    sentence = parts[1]
                    sentences[sent_num] = sentence
                except ValueError:
                    continue

        return sentences

    def parse_cntlist(self, filepath: Path | str) -> dict[str, int]:
        """Parse cntlist into a mapping of sense key to frequency count.

        Parameters
        ----------
        filepath : Path | str
            Path to cntlist file.

        Returns
        -------
        dict[str, int]
            Mapping from sense key to frequency count.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return {}

        counts: dict[str, int] = {}

        with filepath.open("r", encoding="utf-8") as f:
            for line_raw in f:
                line = line_raw.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    count = int(parts[0])
                    sense_key = parts[1]
                    counts[sense_key] = count
                except ValueError:
                    continue

        return counts

    def convert_wordnet_database(
        self, wordnet_dir: Path | str, output_file: Path | str
    ) -> dict[str, int]:
        """Convert entire WordNet database to JSON Lines.

        Parameters
        ----------
        wordnet_dir : Path | str
            Directory containing WordNet database files.
        output_file : Path | str
            Output JSON Lines file path.

        Returns
        -------
        dict[str, int]
            Counts of processed items by file type.

        Raises
        ------
        FileNotFoundError
            If WordNet directory does not exist.
        """
        wordnet_dir = Path(wordnet_dir)
        output_file = Path(output_file)

        if not wordnet_dir.exists():
            msg = f"WordNet directory not found: {wordnet_dir}"
            raise FileNotFoundError(msg)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        counts = {}
        all_synsets = []

        # Process data files
        pos_mappings: list[tuple[str, WordNetPOS]] = [
            ("noun", "n"),
            ("verb", "v"),
            ("adj", "a"),
            ("adv", "r"),
        ]
        for pos_name, pos_code in pos_mappings:
            data_file = wordnet_dir / f"data.{pos_name}"
            if data_file.exists():
                synsets = self.parse_data_file(data_file, pos_code)
                all_synsets.extend(synsets)
                counts[f"synsets_{pos_name}"] = len(synsets)

        # Parse supplementary files for enrichment
        framestext = self.parse_verb_framestext(wordnet_dir / "verb.Framestext")
        sents = self.parse_verb_sentences(wordnet_dir / "sents.vrb")

        # Build sense_key → (sense_number, tag_count) map from index.sense
        sense_map: dict[str, tuple[int, int]] = {}
        sense_index_file = wordnet_dir / "index.sense"
        if sense_index_file.exists():
            with sense_index_file.open("r", encoding="utf-8") as f:
                for line_raw in f:
                    line = line_raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 4:
                        continue
                    try:
                        sk = parts[0]
                        sense_number = int(parts[2])
                        tag_count = int(parts[3])
                        sense_map[sk] = (sense_number, tag_count)
                    except ValueError:
                        continue

        # Parse cntlist to enhance tag_count data
        cntlist = self.parse_cntlist(wordnet_dir / "cntlist")
        for sk, count in cntlist.items():
            if sk in sense_map:
                sn, _ = sense_map[sk]
                sense_map[sk] = (sn, count)
            else:
                sense_map[sk] = (0, count)

        # ss_type to number mapping for sense key construction
        ss_type_num_map: dict[str, int] = {
            "n": 1,
            "v": 2,
            "a": 3,
            "r": 4,
            "s": 5,
        }

        # Enrich synsets with sense data and verb frame templates
        for synset in all_synsets:
            ss_num = ss_type_num_map.get(synset.ss_type, 1)

            # Enrich words with sense_number and tag_count
            for word in synset.words:
                lemma_lower = word.lemma.lower()
                sense_key = f"{lemma_lower}%{ss_num}:{synset.lex_filenum:02d}:{word.lex_id:02d}::"
                if sense_key in sense_map:
                    sn, tc = sense_map[sense_key]
                    if sn > 0:
                        word.sense_number = sn
                    word.tag_count = tc

            # Enrich verb frames with template and example_sentence
            if synset.frames:
                for frame in synset.frames:
                    fn = frame.frame_number
                    if fn in framestext:
                        frame.template = framestext[fn]
                    if fn in sents:
                        frame.example_sentence = sents[fn]

        # Write all synsets to single output file
        with output_file.open("w", encoding="utf-8") as f:
            for synset in all_synsets:
                f.write(f"{synset.model_dump_json()}\n")

        counts["total_synsets"] = len(all_synsets)

        return counts

    def convert_sense_index(self, wordnet_dir: Path | str, output_file: Path | str) -> int:
        """Parse index.sense and output Sense objects to JSONL.

        Parameters
        ----------
        wordnet_dir : Path | str
            Directory containing WordNet database files.
        output_file : Path | str
            Output JSON Lines file path.

        Returns
        -------
        int
            Number of sense entries written.

        Raises
        ------
        FileNotFoundError
            If index.sense file does not exist.
        """
        wordnet_dir = Path(wordnet_dir)
        output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        sense_file = wordnet_dir / "index.sense"
        senses = self.parse_sense_index(sense_file)

        with output_file.open("w", encoding="utf-8") as f:
            for sense in senses:
                f.write(f"{sense.model_dump_json()}\n")

        return len(senses)

    def convert_exceptions(self, wordnet_dir: Path | str, output_file: Path | str) -> int:
        """Parse *.exc files and output ExceptionEntry objects to JSONL.

        Parameters
        ----------
        wordnet_dir : Path | str
            Directory containing WordNet database files.
        output_file : Path | str
            Output JSON Lines file path.

        Returns
        -------
        int
            Number of exception entries written.
        """
        wordnet_dir = Path(wordnet_dir)
        output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        all_entries: list[ExceptionEntry] = []

        exc_files: list[tuple[str, WordNetPOS]] = [
            ("noun.exc", "n"),
            ("verb.exc", "v"),
            ("adj.exc", "a"),
            ("adv.exc", "r"),
        ]

        for exc_name, pos in exc_files:
            exc_path = wordnet_dir / exc_name
            if exc_path.exists():
                entries = self.parse_exception_file(exc_path)
                for entry in entries:
                    entry.pos = pos
                all_entries.extend(entries)

        with output_file.open("w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(f"{entry.model_dump_json()}\n")

        return len(all_entries)

    def _parse_data_line(self, line: str) -> Synset | None:
        """Parse a line from WordNet data file.

        Format: synset_offset lex_filenum ss_type w_cnt word lex_id [word lex_id...]
                p_cnt [ptr...] [frames...] | gloss

        Parameters
        ----------
        line : str
            Data file line to parse.

        Returns
        -------
        Synset | None
            Parsed synset or None if invalid.
        """
        # Find gloss separator
        gloss_idx = line.find(" | ")
        if gloss_idx == -1:
            return None

        data_part = line[:gloss_idx]
        gloss = line[gloss_idx + 3 :].strip()

        parts = data_part.split()
        if len(parts) < 6:
            return None

        try:
            # Parse basic fields
            offset = parts[0].zfill(8)  # Ensure 8 digits
            lex_filenum = int(parts[1])
            ss_type = parts[2]
            w_cnt = int(parts[3], 16)  # Hex count

            # Parse words
            words = []
            idx = 4
            for _ in range(w_cnt):
                if idx + 1 >= len(parts):
                    break
                lemma = parts[idx]
                lex_id = int(parts[idx + 1], 16)
                words.append(Word(lemma=lemma, lex_id=lex_id))
                idx += 2

            # Parse pointers
            if idx >= len(parts):
                return None

            p_cnt = int(parts[idx])
            idx += 1

            pointers = []
            for _ in range(p_cnt):
                if idx + 3 >= len(parts):
                    break

                symbol = parts[idx]
                target_offset = parts[idx + 1].zfill(8)
                target_pos = parts[idx + 2]
                source_target = parts[idx + 3]

                # Parse source/target word numbers
                if len(source_target) == 4:
                    source = int(source_target[:2], 16)
                    target = int(source_target[2:], 16)
                else:
                    source = 0
                    target = 0

                pointer = Pointer(
                    symbol=cast(PointerSymbol, symbol),
                    offset=target_offset,
                    pos=cast(WordNetPOS, target_pos),
                    source=source,
                    target=target,
                )
                pointers.append(pointer)
                idx += 4

            # Parse verb frames if present (for verbs only)
            frames = None
            if ss_type == "v" and idx < len(parts):
                frames = []

                # Skip frame count field
                idx += 1

                # Parse frames until no more "+" markers
                while idx + 2 < len(parts) and parts[idx] == "+":
                    frame_marker = parts[idx]  # "+"
                    frame_info = parts[idx + 1]  # Frame number
                    word_info = parts[idx + 2]  # Word index

                    if frame_marker == "+":
                        # Parse frame number
                        frame_num = int(frame_info)

                        # Parse word index
                        word_idx = int(word_info, 16)
                        word_indices = [word_idx]

                        if 1 <= frame_num <= 35:
                            frame = VerbFrame(
                                frame_number=cast(VerbFrameNumber, frame_num),
                                word_indices=word_indices,
                            )
                            frames.append(frame)

                    idx += 3

            # Get lexical file name
            lex_filename = self.LEX_FILE_NAMES.get(lex_filenum, "noun.Tops")

            return Synset(
                offset=offset,
                lex_filenum=lex_filenum,
                lex_filename=lex_filename,
                ss_type=cast(WordNetPOS, ss_type),
                words=words,
                pointers=pointers,
                frames=frames,
                gloss=gloss,
            )

        except (ValueError, IndexError):
            return None

    def _parse_index_line(self, line: str, pos: WordNetPOS) -> IndexEntry | None:
        """Parse a line from WordNet index file.

        Format: lemma pos synset_cnt p_cnt [ptr_symbol...] sense_cnt tagsense_cnt
        synset_offset [synset_offset...]

        Parameters
        ----------
        line : str
            Index file line to parse.
        pos : WordNetPOS
            Expected part of speech.

        Returns
        -------
        IndexEntry | None
            Parsed index entry or None if invalid.
        """
        parts = line.split()
        if len(parts) < 6:  # Minimum: lemma pos synset_cnt p_cnt sense_cnt tagsense_cnt
            return None

        try:
            lemma = parts[0]
            file_pos = parts[1]

            if file_pos != pos:
                return None

            synset_cnt = int(parts[2])
            p_cnt = int(parts[3])

            # Extract pointer symbols
            ptr_symbols = []
            for i in range(4, 4 + p_cnt):
                if i < len(parts):
                    ptr_symbols.append(parts[i])

            # Get sense and tagsense counts
            idx = 4 + p_cnt
            if idx + 1 >= len(parts):
                return None

            sense_cnt = int(parts[idx])
            tagsense_cnt = int(parts[idx + 1])

            # Extract synset offsets
            synset_offsets = []
            for i in range(idx + 2, len(parts)):
                offset = parts[i].zfill(8)
                synset_offsets.append(offset)

            return IndexEntry(
                lemma=lemma,
                pos=pos,
                synset_cnt=synset_cnt,
                p_cnt=p_cnt,
                ptr_symbols=[cast(PointerSymbol, s) for s in ptr_symbols],
                sense_cnt=sense_cnt,
                tagsense_cnt=tagsense_cnt,
                synset_offsets=synset_offsets,
            )

        except (ValueError, IndexError):
            return None

    def _parse_sense_line(self, line: str) -> Sense | None:
        """Parse a line from sense index file.

        Format: sense_key synset_offset sense_number tag_cnt

        Parameters
        ----------
        line : str
            Sense index line to parse.

        Returns
        -------
        Sense | None
            Parsed sense or None if invalid.
        """
        parts = line.split()
        if len(parts) != 4:
            return None

        try:
            sense_key = parts[0]
            synset_offset = parts[1].zfill(8)
            sense_number = int(parts[2])
            tag_count = int(parts[3])

            # Parse sense key components
            key_parts = sense_key.split("%")
            if len(key_parts) != 2:
                return None

            lemma = key_parts[0]
            rest = key_parts[1].split(":")
            if len(rest) < 3:
                return None

            ss_type_num = int(rest[0])
            lex_filenum = int(rest[1])
            lex_id = int(rest[2])

            # Extract head word info for satellites
            head_word = rest[3] if len(rest) > 3 and rest[3] else None
            head_id = int(rest[4]) if len(rest) > 4 and rest[4] else None

            # Map ss_type number to POS
            pos_map = {1: "n", 2: "v", 3: "a", 4: "r", 5: "s"}
            ss_type = pos_map.get(ss_type_num, "n")

            return Sense(
                sense_key=sense_key,
                lemma=lemma,
                ss_type=cast(WordNetPOS, ss_type),
                lex_filenum=lex_filenum,
                lex_id=lex_id,
                head_word=head_word,
                head_id=head_id,
                synset_offset=synset_offset,
                sense_number=sense_number,
                tag_count=tag_count,
            )

        except (ValueError, IndexError):
            return None

    def _parse_exception_line(self, line: str) -> ExceptionEntry | None:
        """Parse a line from morphological exception file.

        Format: inflected_form base_form1 [base_form2 ...]

        Parameters
        ----------
        line : str
            Exception line to parse.

        Returns
        -------
        ExceptionEntry | None
            Parsed exception entry or None if invalid.
        """
        parts = line.split()
        if len(parts) < 2:
            return None

        inflected_form = parts[0]
        base_forms = parts[1:]

        # Validate word forms
        if not self._is_valid_word_form(inflected_form):
            return None

        valid_base_forms = []
        for form in base_forms:
            if self._is_valid_word_form(form):
                valid_base_forms.append(form)

        if not valid_base_forms:
            return None

        return ExceptionEntry(
            inflected_form=inflected_form,
            base_forms=valid_base_forms,
        )

    def _is_valid_word_form(self, word: str) -> bool:
        """Check if a word form is valid.

        Parameters
        ----------
        word : str
            Word form to validate.

        Returns
        -------
        bool
            True if valid word form.
        """
        if not word:
            return False

        # Allow letters, underscores, hyphens, and apostrophes
        return bool(re.match(r"^[a-z][a-z0-9_'-]*$", word, re.IGNORECASE))


def parse_data_file(filepath: Path | str, pos: WordNetPOS) -> list[Synset]:
    """Parse WordNet data file into list of Synset models.

    Parameters
    ----------
    filepath : Path | str
        Path to WordNet data file.
    pos : WordNetPOS
        Part of speech code.

    Returns
    -------
    list[Synset]
        List of parsed synsets.
    """
    converter = WordNetConverter()
    return converter.parse_data_file(filepath, pos)


def parse_index_file(filepath: Path | str, pos: WordNetPOS) -> list[IndexEntry]:
    """Parse WordNet index file into list of IndexEntry models.

    Parameters
    ----------
    filepath : Path | str
        Path to WordNet index file.
    pos : WordNetPOS
        Part of speech code.

    Returns
    -------
    list[IndexEntry]
        List of parsed index entries.
    """
    converter = WordNetConverter()
    return converter.parse_index_file(filepath, pos)


def parse_sense_index(filepath: Path | str) -> list[Sense]:
    """Parse WordNet sense index file.

    Parameters
    ----------
    filepath : Path | str
        Path to sense index file.

    Returns
    -------
    list[Sense]
        List of parsed senses.
    """
    converter = WordNetConverter()
    return converter.parse_sense_index(filepath)


def parse_exception_file(filepath: Path | str) -> list[ExceptionEntry]:
    """Parse morphological exception file.

    Parameters
    ----------
    filepath : Path | str
        Path to exception file.

    Returns
    -------
    list[ExceptionEntry]
        List of parsed exception entries.
    """
    converter = WordNetConverter()
    return converter.parse_exception_file(filepath)


def convert_wordnet_database(wordnet_dir: Path | str, output_file: Path | str) -> dict[str, int]:
    """Convert entire WordNet database to JSON Lines.

    Parameters
    ----------
    wordnet_dir : Path | str
        WordNet database directory.
    output_file : Path | str
        Output JSON Lines file path.

    Returns
    -------
    dict[str, int]
        Processing counts by file type.
    """
    converter = WordNetConverter()
    return converter.convert_wordnet_database(wordnet_dir, output_file)
