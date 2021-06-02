#!/usr/bin/env python3
import os
import json
from pathlib import Path
from collections import defaultdict

import scipy.spatial
from colorama import Fore, Style

from lib.tensor_sketch_gpu import *
from lib.base import *
from lib.util import *
from lib.minimizers import *
from lib.align import *

import lib.data as data


# Dists is a list of (distance, sequence, sequence)
def ROC_curve(dists, print_distance=False, sketch_exons=False, matching_subsequences=None):
    dists.sort(key=lambda tup: tup[0])
    total = 0
    match = 0
    ratio = 0
    done = False
    matched_seqs = set()
    num_printed = 0

    # Histogram of matches: dist -> {no match, exon match, minimizer match}
    edit_dists = []
    sketch_dists = []
    types = []
    for d, s1, s2 in dists:
        total += 1
        newmatch = False

        # True when the sketched [sub]sequences come from homologues sequences/exons.
        sequence_match = False
        # True when the sketched subsequences have matching minimizer positions.
        # Only meaningful for exon sketching.
        minimizer_match = False
        if sketch_exons:
            key1 = s1.seq.exon_key()
            key2 = s2.seq.exon_key()

            # Check whether the exons are the same.
            sequence_match = key1 in data.exon_orthologs[key2]
            # Check whether these correspond to matching minimizers as well.
            minimizer_match = s1.seq.key() in matching_subsequences[s2.seq.key()]
            if minimizer_match:
                assert sequence_match
        else:
            key1 = s1.seq.id
            key2 = s2.seq.id
            sequence_match = data.is_match(s1.seq, s2.seq)

        if sequence_match:
            match += 1
            done = False
            if key1 not in matched_seqs:
                newmatch = True
            matched_seqs.add(key1)
            # matched_seqs.add(key2)

            if print_distance and d > 0 and sketch_exons and newmatch:
                print(f'{Fore.GREEN}New exon match:{Style.RESET_ALL}')
                data.align_exon_pair(s1.seq.parent, s2.seq.parent, minimizer_params)
                print()

        # Add to histogram
        ed = edit_distance(s1.seq, s2.seq)
        edit_dists.append(ed)
        types.append('match' if sequence_match else 'no_match')
        sketch_dists.append(d)
        # if minimizer_match:
        # edit_dists.append(ed)
        # types.append('minimizer_match')
        # sketch_dists.append(d)

        # Print the first 200 pairs with distance > 0.
        if print_distance and d > 0 and (num_printed < 200 or (sketch_exons and newmatch)):
            num_printed += 1
            edit_dist, (x, y) = align(s1.seq, s2.seq)
            print(f'Seq {data.seqid(s1.seq):3}: {x}')
            print(f'Seq {data.seqid(s2.seq):3}: {y}')
            sequence_match_text = (
                Fore.GREEN + "exon match" + Style.RESET_ALL if sequence_match else ''
            )
            minimizer_match_text = (
                Fore.GREEN + "minimizer match" + Style.RESET_ALL if minimizer_match else ''
            )

            print(
                f'sketch dist {d:0.6f}    | edit dist {edit_dist: 2}    | {sequence_match_text} {minimizer_match_text}'
            )
            print()

        # Print stats for powers of 2 and when new pairs of sequences are matched
        if match > 0 and (match & (match - 1)) == 0 and not done:
            print(
                f'{match:7} {total:7} {match / total:.3f} for dist threshold {d:.4f} with {len(matched_seqs):5} matched seqs'
            )
            done = True
    print(
        f'{match:7} {total:7} {match / total:.3f} for dist threshold {d:.4f} with {len(matched_seqs):5} matched seqs'
    )

    print('Number of matched sequences: ', len(matched_seqs))

    return (edit_dists, types, sketch_dists)


def full_sketching():
    # Sketch all sequences

    params = SketchParams(A=4, t=4, D=10, normalize=True, L=2)
    gts = GTS(params)

    sketches = timeit(lambda: gts.sketch(seqs), "gts")

    # Compute all pairwise distances
    dists = timeit(lambda: pairwise_dists(sketches), "distances")
    ROC_curve(dists)


# For each pair of sequences, count the number of pairs of substrings with small distance.
# TODO: Weigh this by number of subsequences.
def count_matches(distances, num_subsequences):
    d = collections.defaultdict(int)
    cnt_1 = collections.defaultdict(int)
    cnt_2 = collections.defaultdict(int)

    id_to_sketch = dict()

    for _, s1, s2 in distances:
        if s1.seq.id == s2.seq.id:
            continue
        cnt_1[s1.seq.id] += 1
        cnt_2[s2.seq.id] += 1
        d[(s1.seq.id, s2.seq.id)] += 1
        id_to_sketch[s1.seq.id] = s1
        id_to_sketch[s2.seq.id] = s2

    # 2 sequences are similar if d[(s1, s2)] is high compared to the count of s1.
    # The distance s1->s2 is given as 1 - match_cnt / s2_cnt
    dists = []
    for s1id in cnt_1:
        for s2id in cnt_2:
            if (s1id, s2id) not in d:
                continue
            match_freq = d[(s1id, s2id)]
            s1_cnt = cnt_1[s1id]
            s2_cnt = cnt_2[s2id]
            # TODO: Experiment with min/max/avg here.
            # Maybe it's even better to take a product instead.
            dists.append(
                (
                    -match_freq / max(num_subsequences[s1id], num_subsequences[s2id]),
                    id_to_sketch[s1id],
                    id_to_sketch[s2id],
                )
            )

    dists.sort(key=lambda tup: tup[0])

    print('COUNT MATCHES')
    print(
        [
            ((data.seqid(s[0]), data.seqid(s[1])), y)
            for s, y in sorted(d.items(), key=lambda tup: tup[1], reverse=True)
        ]
    )
    print([(f'{dd:.3e}', data.seqid(s0.seq), data.seqid(s1.seq)) for dd, s0, s1 in dists])
    return dists


def minimizer_sketching(
    seqs,
    minimizer_k,
    l,
    window,
    params,
    r,
    sketch_exons_only=False,
    matching_subsequences=defaultdict(list),
):

    print(f'MINIMIZER PARAMS: k={minimizer_k}, window={window}')
    print(f'SUBSEQUENCE LEN:  l={l}')
    print(f'SKETCH PARAMS:    t={params.t}, D={params.D}')
    print(f'KDTREE PARAMS:    r={r}')
    print()

    # 1. Find minimizers for each sequence and the corresponding subsequences.

    # [ [subsequences] ]
    # Note: the subsequences of each input fastafile are stored separately, so we can focus on cross-file matches.
    subsequences = [List(), List()]

    # Map from sequence id to number of subsequences taken for this sequence.
    num_exons = 0
    num_matching_exons = 0
    num_subsequences = defaultdict(int)
    num_matching_subsequences = 0

    exon_lengths = []
    seq_density = []
    exon_edit_distance = []

    if sketch_exons_only:
        # A dictionary: seq_key -> [seq_key], matching a
        # subsequence to homologues subsequences.
        # Add homologues for hetGla2 sequences.
        for s in seqs:
            if s.metadata['genome'] != 'hetGla2':
                continue

            seqs = data.get_orthologs(s.id)

            exons1 = data.exons[seqs[0].id]
            exons2 = data.exons[seqs[1].id]

            exon_pairs = align_exons(seqs[0], exons1, seqs[1], exons2)
            for exon_pair in exon_pairs:
                e1, e2 = exon_pair

                subseqs = [None, None]

                # Compute subsequences for these exons.
                for i, e in enumerate(exon_pair):
                    if e is None:
                        continue
                    num_exons += 1
                    positions = minimizers(e, minimizer_k, window)
                    # print(positions)
                    subseqs[i] = make_subsequences(e, minimizer_k, positions, l)
                    num_subsequences[e.id] += len(subseqs)
                    for pos, subseq in subseqs[i]:
                        subsequences[i].append(subseq)
                    exon_lengths.append(e.len())
                    seq_density.append(len(subseqs) / e.len())

                if e1 is None or e2 is None:
                    continue

                exon_dist = edit_distance(e1, e2) / max(e1.len(), e2.len())
                exon_edit_distance.append(exon_dist)

                # Print far away exons.
                if exon_dist > 0.5:
                    data.align_exon_pair(e1, e2, minimizer_params)

                num_matching_exons += 1
                data.exon_orthologs[e1.exon_key()].append(e2.exon_key())
                data.exon_orthologs[e2.exon_key()].append(e1.exon_key())

                # Find matching minimizer positions/subsequences .
                matches = data.find_matching_minimizers(e1, subseqs[0], e2, subseqs[1])
                for s1, s2 in matches:
                    # print(s1.exon_offset, s1.subseq_offset, to_string(s1))
                    # print(s2.exon_offset, s2.subseq_offset, to_string(s2))
                    # print()
                    key1 = s1.key()
                    key2 = s2.key()
                    matching_subsequences[key1].append(key2)
                    matching_subsequences[key2].append(key1)
                    num_matching_subsequences += 1
        # return

    else:
        # Add all sequences for the first 2 files.
        for s in seqs:
            idx = None
            if s.metadata['genome'] == 'hetGla2':
                idx = 0
            elif s.metadata['genome'] == 'hg38':
                idx = 1
            else:
                continue

            minimizer_pos = minimizers(s, minimizer_k, window)
            subseqs = make_subsequences(s, minimizer_k, minimizer_pos, l)
            num_subsequences[s.id] = len(subseqs)
            for pos, subseq in subseqs:
                subsequences[idx].append(subseq)

    print('num exons:        ', num_exons)
    print('num matching exs: ', num_matching_exons)
    print('num subsequences: ', sum(len(ss) for ss in subsequences))
    print('num matching sss: ', num_matching_subsequences)
    print('file0: ', len(subsequences[0]))
    print('file1: ', len(subsequences[1]))

    print('Exon lengths')
    sns.displot(x=exon_lengths)
    print('Exon minimizer density')
    sns.displot(x=seq_density)
    print('Matching exon edit distance')
    sns.displot(x=exon_edit_distance)

    # 2. Sketch all subsequences.
    gts = TS(params)
    sketches = timeit(lambda: [gts.sketch(ss) for ss in subsequences], "GTS")

    raw_sketches = [[s.sketch for s in ss] for ss in sketches]

    print_sketch_statistics([s for ss in raw_sketches for s in ss])

    # 3. Find close subsequences using kd-trees.
    kdtrees = timeit(
        lambda: [scipy.spatial.cKDTree(ss) for ss in raw_sketches],
        "build KDTree",
    )

    # Distance between sketch i from file 0 and sketch j from file 1.
    def point_dist(i, j):
        return dist(sketches[0][i], sketches[1][j])

    # Find all pairs within a given distance of each other.
    counted_matches = timeit(
        lambda: kdtrees[0].count_neighbors(kdtrees[1], [r / 3, r / 2, r, 1.5 * r]), 'Count nbs'
    )
    print('counted   pairs: ', counted_matches)
    matches = timeit(lambda: kdtrees[0].query_ball_tree(kdtrees[1], r), 'Query KDTree')
    num_matches = sum(len(x) for x in matches)
    print('number of pairs: ', num_matches)

    dists = []
    for i in range(len(sketches[0])):
        for j in matches[i]:
            dists.append((point_dist(i, j), sketches[0][i], sketches[1][j]))
    return dists


def print_all_exons(minimizer_params=None):
    for s in seqs:
        if s.metadata['genome'] != 'hetGla2':
            continue
        data.compare_exons(s.id, minimizer_params)


# =============================================================

# This must be 2 for exon ortholog related code to work.
NUM_FILES = 2
NUM_ORTHOLOGS = 50

# k for minimizers
minimizer_k = 4
# Length of subsequences to sketch.
l = 20
# Window size to look for minimizers
window = 20
# Sketch params
params = SketchParams(A=4, t=3, D=10, normalize=True, L=2)
sketch_exons_only = True
# Max query distance, map from (t, D) to distance
r_map = {
    (2, 10, False): 0.015,
    (3, 10, False): 0.009,
    (3, 10, True): 0.1,
    (3, 20, False): 0.009,
    (6, 10, False): 0.0018,
}
r = r_map[(params.t, params.D, sketch_exons_only)]


seqs = List()
minimizer_params = (minimizer_k, window)
matching_subsequences = defaultdict(list)
dists = None


def run():
    global seqs, minimizer_params, matching_subsequences, dists

    data.read(file_names_=data.file_names[:NUM_FILES])

    # Sequences corresponding to the first 10 orthologs.
    for i, key in enumerate(data.orthologs):
        for o in data.get_orthologs(key):
            seqs.append(o)
        if len(seqs) >= NUM_FILES * NUM_ORTHOLOGS:
            break

    total_len = 0
    for s in seqs:
        total_len += s.len()

    print('Total length of sequences: ', total_len)

    # print_stats('Sequence Length', [s.len() for s in seqs], False)
    # for s in seqs:
    # print(f'{data.seqid(s.id):3} {s.len():7} {s.id}')

    print()

    print_all_exons(minimizer_params)

    dists = minimizer_sketching(
        seqs,
        minimizer_k,
        l,
        window,
        params,
        r,
        sketch_exons_only,
        matching_subsequences=matching_subsequences,
    )


if __name__ == '__main__':
    run()

    ROC_curve(
        dists, True, sketch_exons=sketch_exons_only, matching_subsequences=matching_subsequences
    )
    if not sketch_exons_only:
        ROC_curve(count_matches(dists, num_subsequences))
