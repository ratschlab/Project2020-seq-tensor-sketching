# Returns the alignment of two sequences.

import numpy as np
from numba import njit

from lib.sequence import *

# Return the edit distance between two sequences.
# See `align` below which also returns the aligned strings.
# This is copied from the c++ implementation in utils.hpp.
@njit
def edit_distance(s1: Sequence, s2: Sequence, color=True):
    # This should only be used for relatively small sequences.
    l1 = s1.len()
    l2 = s2.len()
    assert l1 * l2 <= 1000000

    if l1 == 0:
        return l2
    if l2 == 0:
        return l1

    costs = np.zeros((l1 + 1, l2 + 1), dtype=np.int32)
    for i in range(l2 + 1):
        costs[0][i] = i

    for i, c1 in enumerate(s1.seq):
        costs[i + 1][0] = i + 1

        for j, c2 in enumerate(s2.seq):
            if c1 == c2:
                costs[i + 1][j + 1] = costs[i][j]
            else:
                costs[i + 1][j + 1] = 1 + min(costs[i][j], costs[i + 1][j], costs[i][j + 1])

    return costs[l1][l2]


# Given two sequences, find the pairwise matching exons.
# Returns (edit_distance, (string, string)) where the 2 strings contain the
# original characters interleaved with spaces.
def align(s1: Sequence, s2: Sequence, color=True):
    # This should only be used for relatively small sequences.
    l1 = s1.len()
    l2 = s2.len()
    assert l1 * l2 <= 1000000

    costs = np.zeros((l1 + 1, l2 + 1), dtype=np.int32)
    for i in range(l2 + 1):
        costs[0][i] = i

    for i, c1 in enumerate(s1.seq):
        costs[i + 1][0] = i + 1

        for j, c2 in enumerate(s2.seq):
            if c1 == c2:
                costs[i + 1][j + 1] = costs[i][j]
            else:
                costs[i + 1][j + 1] = 1 + min(costs[i][j], costs[i + 1][j], costs[i][j + 1])

    # Color substituted characters differently.
    RED = '\033[91m' if color else ''
    END = '\033[0m' if color else ''

    # Reconstruct optimal path
    state = (l1, l2)
    a1 = []
    a2 = []
    while state != (0, 0):
        best = (1000000000, (i, j))
        i, j = state
        if i > 0 and j > 0 and s1.seq[i - 1] == s2.seq[j - 1]:
            state = (i - 1, j - 1)
            c1 = s1.seq[i - 1]
            c2 = s2.seq[j - 1]
            a1.append(to_char(c1))
            a2.append(to_char(c2))
            assert costs[i][j] == costs[i - 1][j - 1]
        elif i > 0 and j > 0 and costs[i][j] == costs[i - 1][j - 1] + 1:
            state = (i - 1, j - 1)
            c1 = s1.seq[i - 1]
            c2 = s2.seq[j - 1]
            # Highlight substitutions in red.
            a1.append(RED + to_char(c1) + END)
            a2.append(RED + to_char(c2) + END)
        elif i > 0 and costs[i][j] == costs[i - 1][j] + 1:
            state = (i - 1, j)
            c1 = s1.seq[i - 1]
            a1.append(RED + to_char(c1) + END)
            a2.append(' ')
        elif j > 0 and costs[i][j] == costs[i][j - 1] + 1:
            state = (i, j - 1)
            a1.append(' ')
            c2 = s2.seq[j - 1]
            a2.append(RED + to_char(c2) + END)
        else:
            assert False

    a1.reverse()
    a2.reverse()
    return (costs[l1][l2], (''.join(a1), ''.join(a2)))


# Return a list of aligned (Exon/None, Exon/None) pairs.
# The cost of an unaligned exon is its length multiplied by C=0.75.
# exons1 and exons2 are lists of Sequence objects.
def align_exons(s1, exons1, s2, exons2, color=True, C=0.75):
    l1 = len(exons1)
    l2 = len(exons2)

    dists = np.zeros((l1, l2), dtype=np.float32)
    for i, e1 in enumerate(exons1):
        for j, e2 in enumerate(exons2):
            dists[i][j] = edit_distance(e1, e2)

    costs = np.zeros((l1 + 1, l2 + 1), dtype=np.float32)
    for j, e2 in enumerate(exons2):
        costs[0][j + 1] = costs[0][j] + C * e2.len()

    for i, e1 in enumerate(exons1):
        costs[i + 1][0] = costs[i][0] + C * e1.len()

        for j, e2 in enumerate(exons2):
            costs[i + 1][j + 1] = min(
                costs[i][j] + dists[i][j],
                costs[i][j + 1] + C * e1.len(),
                costs[i + 1][j] + C * e2.len(),
            )

    # Color substituted characters differently.
    RED = '\033[91m' if color else ''
    END = '\033[0m' if color else ''

    # Reconstruct optimal path
    state = (l1, l2)
    exon_pairs = []
    while state != (0, 0):
        best = (1000000000, (i, j))
        i, j = state
        if i > 0 and j > 0 and costs[i][j] == costs[i - 1][j - 1] + dists[i - 1][j - 1]:
            state = (i - 1, j - 1)
            exon_pairs.append((exons1[i - 1], exons2[j - 1]))
        elif i > 0 and costs[i][j] == costs[i - 1][j] + C * exons1[i - 1].len():
            state = (i - 1, j)
            exon_pairs.append((exons1[i - 1], None))
        elif j > 0 and costs[i][j] == costs[i][j - 1] + C * exons2[j - 1].len():
            state = (i, j - 1)
            exon_pairs.append((None, exons2[j - 1]))
        else:
            assert False

    exon_pairs.reverse()
    return exon_pairs
