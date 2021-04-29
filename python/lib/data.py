# Contains functions that read all Fasta/json data into memory.

# Note: Do not do `from data import *`, but instead always access as `data.files`.

import json

from pathlib import Path

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
# Map from sequence id to list of (exon_meta, Sequence), where the sequence is a subsequence of the full sequence.
exons = None

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
                (exon_meta, seq.subsequence_of_full(exon_meta['rstart'], exon_meta['rend'] + 1))
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


# ============ EXON PROCESSING ===============


def print_exon(exon_meta, exon, minimizer_params=None, *, aligned=None):
    # Remove colours from aligned before taking indices.
    RED = '\033[91m'
    END = '\033[0m'

    chars = aligned or to_string(exon)
    print(f' {exon_meta["number"]:3} {chars}')

    if minimizer_params is None:
        return
    # Print the minimizers
    starts = minimizers(exon, *minimizer_params)
    # Just to be sure.
    starts.sort()

    # Remove colours from align before taking indices.
    if aligned:
        stripped_aligned = aligned.replace(RED, '')
        stripped_aligned = stripped_aligned.replace(END, '')
        chars = stripped_aligned

    # Put a [ at the start of each minimizer segment.
    x = [' '] * len(chars)
    if aligned:
        # i: position in original sequence
        # j: position in aligned sequence
        i = 0
        j = 0
        while stripped_aligned[j] == ' ':
            j += 1
        for s in starts:
            while i < s:
                i += 1
                j += 1
                while stripped_aligned[j] == ' ':
                    j += 1
            x[j] = '['
    else:
        for s in starts:
            x[s] = '['

    x = ''.join(x)
    print(f'{"":4} {"":3} {x}')


# Given a sequence id, print its exons.
def print_exons(id):
    # Print the exons in format:
    # <start> <end> <len> <basepairs>
    # where start and end are relative to the gene start.
    seq = by_id[id]
    print(f'Exons for sequence {id}')
    for exon_meta in exons[id]:
        print_exon(exon, exon_meta)
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
                    exon_meta = seq_exons[i][number]
                    exon = get_exon(exon_meta, seqs[i])
                    print_exon(exon_meta, exon, minimizer_params)
            print()
    else:
        # Align the exons.
        exon_pairs = align_exons(seqs[0], exons[seqs[0].id], seqs[1], exons[seqs[1].id])
        for exon_pair in exon_pairs:
            e1, e2 = exon_pair
            if e1 is not None and e2 is not None:
                dist, aligned = align(exon_pair[0][1], exon_pair[1][1])
            else:
                aligned = (None, None)

            for i in range(2):
                if exon_pair[i] is None:
                    continue
                print(f'{seqid(seqs[i]):4}', end='')
                print_exon(*exon_pair[i], minimizer_params, aligned=aligned[i])
            print()
    print()
