# Contains functions that read all Fasta/json data into memory.

# Note: Do not do `from data import *`, but instead always access as `data.files`.

import json

from pathlib import Path
from collections import defaultdict

from lib.util import *
from lib.sequence import *
from lib.minimizers import *
from lib.align import *

# This is set in the call to read_data().
data_dir = None

# Constants.
file_names = ['hetGla2.fa', 'hg38.fa', 'macFas5.fa', 'mm10.fa']
orthologs_file = 'orthologs.json'

# List of FastaFile objects.
files = None
# Map from sequence id to Sequence objects.
by_id = None
# Map from sequence tid to Sequence objects. Needed since exons only contain tid.
by_tid = None
# Map from sequence id to list of ortholog sequence ids.
orthologs = None
# Map from sequence id to list of Sequence where the sequence is a subsequence of the full sequence.
exons = None
# Map from exon key (id, exon_offset) to orthologs exon keys.
# This must be filled manually after calling align_exons.
exon_orthologs = defaultdict(list)


# Read all data
def read(data_dir_=Path('/home/philae/git/eth/data/homology/'), file_names_=file_names):
    global data_dir, file_names, files, by_id, orthologs, exons
    data_dir = Path(data_dir_)
    file_names = file_names_
    file_paths = [data_dir / name for name in file_names]
    files = timeit(lambda: [FastaFile(path) for path in file_paths], "read files")
    by_id = make_sequence_dict(files)

    orthologs = json.loads((data_dir / orthologs_file).read_text())

    # Temporary map from tid to sequence id.
    by_tid = dict()
    for id in by_id:
        seq = by_id[id]
        by_tid[seq.metadata['tid']] = seq

    exons = dict()
    for f in file_paths:
        data = json.loads(f.with_suffix('.CDS.json').read_text())
        for tid in data:
            seq = by_tid[tid]

            # Ensure that exons are sorted.
            es = sorted(data[tid], key=lambda x: x['rstart'])

            # Occasionally multiple exons cover the same stretch. In this case keep the longer ones.
            filtered = []
            for exon in es:
                if len(filtered) > 0:
                    last = filtered[-1]
                    if exon['rstart'] <= last['rend']:
                        same = exon['rstart'] == last['rstart'] and exon['rend'] == last['rend']
                        if not same:
                            # Take the longer one
                            if exon['rend'] - exon['rstart'] > last['rend'] - last['rstart']:
                                filtered[-1] = exon
                        continue
                filtered.append(exon)

            exons[seq.id] = [
                seq.exon_subsequence(exon_meta['rstart'], exon_meta['rend'] + 1)
                for exon_meta in filtered
            ]


_seqid_map = dict()


def seqid(seq):
    if not isinstance(seq, str):
        seq = seq.id
    if seq not in _seqid_map:
        _seqid_map[seq] = len(_seqid_map)
    return _seqid_map[seq]


# Return the orthologs for the given sequence id, including the sequence itself.
def get_orthologs(id):
    os = [id] + orthologs[id]
    # Filter for sequences that are actually read.
    return sorted([by_id[o] for o in os if o in by_id], key=lambda s: s.id)


# Are (sub)sequences s1 and s2 orthologs?
def is_match(s1, s2):
    if s1.id not in orthologs:
        return False
    return s2.id in orthologs[s1.id]


# ============ EXON PROCESSING ===============

# Given an exon, its minimizers, and its aligned string (to another exon), find the aligned minimizer positions.
# Returns a list of [aligned_minimizer_pos]
def align_minimizer_pos(exon, minimizer_pos, aligned):
    # Put a [ at the start of each minimizer segment.
    # i: position in original sequence
    # j: position in aligned sequence
    i = 0
    j = 0
    minimizer_pos.sort()
    aligned_minimizer_pos = []
    while aligned[j] == ' ':
        j += 1
    for s in minimizer_pos:
        while i < s:
            i += 1
            j += 1
            while aligned[j] == ' ':
                j += 1
        aligned_minimizer_pos.append(j)
    return aligned_minimizer_pos


def print_exon(exon, minimizer_params=None, *, aligned=None):
    # Remove colours from aligned before taking indices.
    RED = '\033[91m'
    END = '\033[0m'

    chars = aligned or to_string(exon)
    print(f' {chars}')

    if minimizer_params is None:
        return
    # Print the minimizers
    minimizer_pos = minimizers(exon, *minimizer_params)
    # Just to be sure.
    minimizer_pos.sort()

    # Remove colours from align before taking indices.
    if aligned:
        stripped_aligned = aligned.replace(RED, '')
        stripped_aligned = stripped_aligned.replace(END, '')
        chars = stripped_aligned

    # Put a [ at the start of each minimizer segment.
    x = [' '] * len(chars)
    if aligned:
        for j in align_minimizer_pos(exon, minimizer_pos, stripped_aligned):
            x[j] = '['
    else:
        for j in minimizer_pos:
            x[j] = '['

    x = ''.join(x)
    print(f'{"":4} {x}')


# Given a sequence id, print its exons.
def print_exons(id):
    # Print the exons in format:
    # <start> <end> <len> <basepairs>
    # where start and end are relative to the gene start.
    seq = by_id[id]
    print(f'Exons for sequence {id}')
    for exon in exons[id]:
        print_exon(exon)
    print()


def align_exon_pair(e1, e2, minimizer_params=None):
    exon_pair = (e1, e2)
    if e1 is not None and e2 is not None:
        dist, aligned = align(e1, e2)
    else:
        aligned = (None, None)

    for e, a in zip(exon_pair, aligned):
        if e is None:
            continue
        print(f'{seqid(e):4}', end='')
        print_exon(e, minimizer_params, aligned=a)
    print()


# Given a sequence id, print the exons for this sequence and all orthologs.
def compare_exons(id, minimizer_params=None):
    seqs = get_orthologs(id)
    print('Exons for sequences:')
    for s in seqs:
        print(f'{seqid(s)} {s.id}')

    print()
    if len(seqs) != 2:
        seq_exons = [exons[seq.id] for seq in seqs]
        for number in range(max(len(exons) for exons in seq_exons)):
            for i in range(len(os)):
                if number < len(seq_exons[i]):
                    print(i, end='')
                    exon = get_exon(exon_meta, seqs[i])
                    print_exon(exon, minimizer_params)
            print()
    else:
        # Align the exons.
        exon_pairs = align_exons(seqs[0], exons[seqs[0].id], seqs[1], exons[seqs[1].id])
        for exon_pair in exon_pairs:
            align_exon_pair(*exon_pair, minimizer_params=minimizer_params)
    print()


# Given 2 matching exons and their corresponding subsequences with minimizer
# positions, find all matching (seq, offset) pairs.
# Exons are given as Sequence.
# Subsequences are a list of pairs: [(minimizer position, sequence)]
#
# Returns a list of tuples of matching sequences.
def find_matching_minimizers(exon1, subsequences1, exon2, subsequences2):
    # 1. align the exons
    (d, (a1, a2)) = align(exon1, exon2, color=False)

    # 2. find aligned minimizer positions
    # [((mp, seq), amp)]
    amps1 = align_minimizer_pos(exon1, [mp for mp, seq in subsequences1], a1)
    amps2 = align_minimizer_pos(exon2, [mp for mp, seq in subsequences2], a2)

    assert len(amps1) == len(subsequences1)
    assert len(amps2) == len(subsequences2)

    matching_seqs = []

    # 3. Return a dictionary: (seq id, offset) -> [(seq id, offset)]
    # i: index into mp1
    # j: index into mp2
    i = 0
    j = 0
    while True:
        if i == len(amps1):
            break
        if j == len(amps2):
            break

        # Store if equal aligned position.
        if amps1[i] == amps2[j]:
            matching_seqs.append((subsequences1[i][1], subsequences2[j][1]))
            i += 1
            j += 1
        else:
            if amps1[i] < amps2[j]:
                i += 1
            else:
                j += 1

    return matching_seqs
