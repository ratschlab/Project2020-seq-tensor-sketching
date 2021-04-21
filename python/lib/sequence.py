# Classes for Sequence and FastaFile.

from pathlib import Path

import numpy as np
import numba as nb
from numba import njit, types, typed
from numba.experimental import jitclass

# Map from sequence characters to internal integer representation.
_char_map: dict[str, int] = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


# Given the char_map above, returns an array of length 256 mapping bytes to
# internal integers. -1 signals unknown bytes.
def _compute_char_list() -> np.ndarray:
    char_list = np.full(256, -1, np.int8)
    for k in _char_map:
        char_list[ord(k)] = _char_map[k]
    return char_list


# Map 256 bytes to integers; built from the map above.
# -1 signals unknown bytes.
_char_list: np.ndarray = _compute_char_list()


# Class that contains a single sequence. The full_seq member contains the
# original byte-representation of the sequence as read from the Fasta file. The
# seq member contains the processed internal int8 representation.
@jitclass(
    [
        ('id', types.unicode_type),
        ('metadata', types.DictType(types.unicode_type, types.unicode_type)),
        # C-layout 1-dimensional arrays.
        ('full_seq', nb.byte[::1]),
        ('seq', nb.int8[::1]),
    ]
)
class Sequence:

    # Given an ID of the form key:value|otherkey:othervalue, parse it.
    @staticmethod
    def id_to_map(id):
        data = typed.Dict()
        for kv in id.split('|'):
            k, v = kv.split(':')
            data[k] = v
        return data

    # Remap characters by char_map. Removes other (lower case) characters.
    @staticmethod
    def remap(s):
        return np.array([_char_list[c] for c in s if _char_list[c] != -1], dtype=np.int8)

    @staticmethod
    def reverse_complement(seq: np.ndarray):
        seqr = np.flip(seq)
        return np.array([(c ^ 3) for c in seqr], dtype=np.int8)

    def __init__(self, id: str, s: bytes):
        # String: header/name/id of this sequence in the Fasta file.
        self.id = id
        # Metadata encoded in the header.
        self.metadata = self.id_to_map(id)
        # The original sequence.
        self.full_seq = np.array([c for c in s], dtype=nb.byte)
        # The sequence with masked repeats (lower case characters) removed, and mapped to integers.
        self.seq = self.remap(s)
        if 'strand' in self.metadata and self.metadata['strand'] == '-':
            self.seq = self.reverse_complement(self.seq)

    def len(self):
        return len(self.seq)


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
                    seq.append(line)
            flush()


# Contains a map from ids to sequences, constructed from a list of FastaFiles.
class SequenceDict:
    def __init__(self, fastafiles):
        self.by_id = dict()
        for file in fastafiles:
            for seq in file.seqs:
                self.by_id[seq.id] = seq

    # Returns None if key not found.
    def __getitem__(self, key):
        return self.by_id.get(key)
