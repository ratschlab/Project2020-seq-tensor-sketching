# Classes for Sequence and FastaFile.

from pathlib import Path

import numpy as np
import numba as nb
from numba import njit, types, typed
from numba.experimental import jitclass

# Map from sequence characters to internal integer representation.
_char_map: dict[str, int] = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Map from sequence characters to their complement, preserving casing.
_complement_map: dict[str, str] = {
    b'A': b'T',
    b'C': b'G',
    b'G': b'C',
    b'T': b'A',
    b'a': b't',
    b'c': b'g',
    b'g': b'c',
    b't': b'a',
    b'N': b'N',
}

# Map from integer representation back to character. Uses index -1 for a space.
_id_to_char_map = np.array(['A', 'C', 'G', 'T'])

# Given the char_map above, returns an array of length 256 mapping bytes to
# internal integers. -1 signals unknown bytes.
def _char_map_to_int_list(char_map) -> np.ndarray:
    char_list = np.full(256, -1, np.int8)
    for k in char_map:
        char_list[ord(k)] = char_map[k]
    return char_list


def _char_map_to_char_list(char_map) -> np.ndarray:
    char_list = np.full(256, -1, np.byte)
    for k in char_map:
        char_list[ord(k)] = ord(char_map[k])
    return char_list


# Map 256 bytes to integers; built from the map above.
# -1 signals unknown bytes.
_char_list: np.ndarray = _char_map_to_int_list(_char_map)
_complement_list = _char_map_to_char_list(_complement_map)


# Returns a string version of an int8 encoded sequence.
# -1 signals space
@njit
def to_char(c):
    return _id_to_char_map[c]


@njit
def to_chars(seq):
    return ''.join([_id_to_char_map[c] for c in seq])


# Class that contains a single sequence. The full_seq member contains the
# original byte-representation of the sequence as read from the Fasta file. The
# seq member contains the processed internal int8 representation.
@jitclass(
    [
        # The full string header of the sequence in the fasta file.
        ('id', types.unicode_type),
        # The decoded metadata in the id.
        ('metadata', types.DictType(types.unicode_type, types.unicode_type)),
        # The position in full_seq where this substring starts.
        # 0 for the original sequence.
        # Note that taking subsequences into full_seq and seq (before/after
        # removing repeats) is sometimes mixed, so this number may not be
        # accurate.
        # Exons are taken as indices into full_seq, and repeats are pruned.
        # From this (or from the original seq), minimizer positions/subsequences are taken directly from seq.
        # This offset is mostly useful as an ID to identify subsequences.
        ('offset', nb.int32),
        # C-layout 1-dimensional arrays.
        ('full_seq', nb.byte[::1]),
        ('seq', nb.int8[::1]),
    ]
    # NOTE: cache=True is sadly not supported for @jitclass.
)
class Sequence:
    # Given an ID of the form key:value|otherkey:othervalue, parse it.
    @staticmethod
    def id_to_map(id):
        data = typed.Dict()
        if id:
            for kv in id.split('|'):
                k, v = kv.split(':')
                data[k] = v
        return data

    # Remap characters by char_map. Removes other (lower case) characters.
    @staticmethod
    def remap(s):
        return np.array([_char_list[c] for c in s if _char_list[c] != -1], dtype=np.int8)

    # Reverse complement of an array of integers.
    @staticmethod
    def reverse_complement(seq: np.ndarray):
        seqr = np.flip(seq)
        return np.array([(c ^ 3) for c in seqr], dtype=np.int8)

    # Reverse complement of a string.
    @staticmethod
    def string_reverse_complement(seq: np.ndarray):
        seqr = np.flip(seq)
        return np.array([_complement_list[c] for c in seqr], dtype=nb.byte)

    def __init__(self, id: str, s: bytes):
        # String: header/name/id of this sequence in the Fasta file.
        self.id = id
        # Metadata encoded in the header.
        self.metadata = self.id_to_map(id)
        # This sequence doesn't have an offset.
        self.offset = 0
        # The original sequence, corrected for reverse complement in case this is the negative strand.
        self.full_seq = np.array([c for c in s], dtype=nb.byte)
        # The sequence with masked repeats (lower case characters) removed, and mapped to integers.
        self.seq = self.remap(s)

        if 'strand' in self.metadata and self.metadata['strand'] == '-':
            # NOTE: Although the README.md in the homology dataset states that
            # sequences marked with strand:- are the reverse complement of the
            # top strand, in practice this doesn't seem to be the case!
            # Hence, we just ignore it and do *not* take the reverse complement of those sequences here.

            # self.full_seq = self.string_reverse_complement(self.full_seq)
            # self.seq = self.reverse_complement(self.seq)
            pass

    def len(self):
        return len(self.seq)

    # Creates a new Sequence object that is a subsequence of the raw sequence with the given start and length.
    # The full_seq is reconstructed from the raw subsequence.
    # TODO: Convert to (start, end) args.
    def subsequence(self, start: int, length: int):
        subseq = Sequence(self.id, b'')
        subseq.seq = self.seq[start : start + length]
        subseq.offset = self.offset + start
        # Convert the raw subsequence back to chars.
        # ord(A, C, G, T)
        d = np.array([65, 67, 71, 84], dtype=nb.byte)
        subseq.full_seq = np.array([d[c] for c in subseq.seq], dtype=nb.byte)
        return subseq

    # Creates a new Sequence object that is a subsequence of the original full sequence with the given start and length.
    # Repeats are dropped for the raw int8 sequence, and the string sequence is reconstructed from this.
    def subsequence_of_full(self, start: int, end: int):
        subseq = Sequence(self.id, b'')
        subseq.seq = subseq.remap(self.full_seq[start:end])
        subseq.offset = self.offset + start
        # Convert the raw subsequence back to chars.
        # ord(A, C, G, T)
        d = np.array([65, 67, 71, 84], dtype=nb.byte)
        subseq.full_seq = np.array([d[c] for c in subseq.seq], dtype=nb.byte)
        return subseq


# Convert seq.full_seq from list of int8 to bytes to string.
# NOTE: bytes() is not supported by Numba.
def to_string(seq):
    return str(bytes(seq.full_seq), 'ascii')


Sequence_type = Sequence.class_type.instance_type


class FastaFile:
    def __init__(self, path):
        # The Path to the current file.
        self.path = Path(path)
        # The name of the current file.
        self.name = self.path.name
        # A list of Sequence objects in this file.
        self.seqs = []

        self.read()

    def read(self):
        header = None
        seq = []

        def flush():
            nonlocal header, seq
            if header is None:
                return
            assert seq
            sequence = Sequence(header, b''.join(seq))
            self.seqs.append(sequence)
            header = None
            seq = []

        # Sequences are read in binary mode; ids are decoded as ascii.
        with self.path.open('br') as f:
            for line in f:
                if line[0] == ord('>'):
                    flush()
                    header = line[1:].decode('ascii').strip()
                else:
                    # Remove trailing whitespace
                    line = line.strip()
                    seq.append(line)
            flush()


# Contains a map from ids to sequences, constructed from a list of FastaFiles.
def make_sequence_dict(fastafiles):
    by_id = dict()
    for file in fastafiles:
        for seq in file.seqs:
            by_id[seq.id] = seq
    return by_id
