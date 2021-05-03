#!/usr/bin/env python3
import os
import json
from pathlib import Path

import scipy.spatial

from lib.tensor_sketch_gpu import *
from lib.base import *
from lib.util import *
from lib.minimizers import *
from lib.align import *

import lib.data as data


# Dists is a list of (distance, sequence, sequence)
def ROC_curve(dists, print_distance=False):
    dists.sort(key=lambda tup: tup[0])
    total = 0
    match = 0
    ratio = 0
    done = False
    matched_seqs = set()
    num_printed = 0
    for d, s1, s2 in dists:
        total += 1
        newmatch = False
        if data.is_match(s1.seq, s2.seq):
            match += 1
            done = False
            if s1.seq.id not in matched_seqs:
                newmatch = True
            matched_seqs.add(s1.seq.id)

        # Print the first 200 pairs with distance > 0.
        if print_distance and d > 0 and num_printed < 200:
            num_printed += 1
            edit_dist, (x, y) = align(s1.seq, s2.seq)
            print(f'Seq {data.seqid(s1.seq): 2}: {x}')
            print(f'Seq {data.seqid(s2.seq): 2}: {y}')
            print(f'sketch dist {d:0.6f}    | edit dist {edit_dist: 2}')
            print()

        # Print stats for powers of 2 and when new pairs of sequences are matched
        if newmatch or (match > 0 and (match & (match - 1)) == 0 and not done):
            print(
                f'{match:7} {total:7} {match / total:.3f} for dist threshold {d:.4f} with {len(matched_seqs):5} matched seqs'
            )
            done = True
    print(
        f'{match:7} {total:7} {match / total:.3f} for dist threshold {d:.4f} with {len(matched_seqs):5} matched seqs'
    )

    print('Number of matched sequences: ', len(matched_seqs))


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


def minimizer_sketching(minimizer_k, l, window, params, r, sketch_exons_only=False):

    print(f'MINIMIZER PARAMS: k={minimizer_k}, window={window}')
    print(f'SUBSEQUENCE LEN:  l={l}')
    print(f'SKETCH PARAMS:    t={params.t}, D={params.D}')
    print(f'KDTREE PARAMS:    r={r}')
    print()

    # 1. Find minimizers for each sequence and the corresponding subsequences.
    # Note: the subsequences of each input fastafile are stored separately, so we can focus on cross-file matches.
    subsequences = []

    # Sort sequences by genome
    sequences = [[], []]
    for s in seqs:
        if sketch_exons_only:
            exon_seqs = [e[1] for e in data.exons[s.id]]
            if s.metadata['genome'] == 'hetGla2':
                sequences[0] += exon_seqs
            if s.metadata['genome'] == 'hg38':
                sequences[1] += exon_seqs
        else:
            if s.metadata['genome'] == 'hetGla2':
                sequences[0].append(s)
            if s.metadata['genome'] == 'hg38':
                sequences[1].append(s)

    num_subsequences = dict()

    for file in sequences:
        ss = List()
        for s in file:
            minimizer_pos = minimizers(s, minimizer_k, window)
            subseqs = make_subsequences(s, minimizer_k, minimizer_pos, l)
            num_subsequences[s.id] = len(subseqs)
            for subseq in subseqs:
                ss.append(subseq)
        subsequences.append(ss)

    print('num subsequences: ', sum(len(ss) for ss in subsequences))
    print('file0: ', len(subsequences[0]))
    print('file1: ', len(subsequences[1]))

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

    ROC_curve(dists, True)
    ROC_curve(count_matches(dists, num_subsequences))


def print_all_exons(minimizer_params=None):
    for s in seqs:
        if s.metadata['genome'] != 'hetGla2':
            continue
        data.compare_exons(s.id, minimizer_params)


# =============================================================

NUM_FILES = 2
NUM_ORTHOLOGS = 5

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
    (3, 10, True): 0.9,
    (3, 20, False): 0.009,
    (6, 10, False): 0.0018,
}
r = r_map[(params.t, params.D, sketch_exons_only)]


data.read(file_names_=data.file_names[:NUM_FILES])


# Sequences corresponding to the first 10 orthologs.
seqs = List()
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
for s in seqs:
    print(f'{data.seqid(s.id):3} {s.len():7} {s.id}')
print()

print_all_exons((minimizer_k, window))
minimizer_sketching(minimizer_k, l, window, params, r, sketch_exons_only)
