# This file contains various methods to find start positions for subsequences to sketch.
# This includes:
# - strided start positions
# - random start positions
# - minimizer start positions
#
# Each method takes a string and some parameters, and returns a list of start positions.
#
# TODO: Confirm the intuition that starting a new sketch every ~L/2 positions works well, where L is the length of each sketched sequence.

import numpy.random
import collections

from numba import jit

import lib.hashing


rng = numpy.random.default_rng()

# Starting positions are exactly `stride` apart.
def strided_positions(s, stride, offset=0):
    return list(range(offset, s.len(), stride))


# Each position is a start position with probability p.
def random_positions(s, p):
    pos = []
    i = -1
    while True:
        i += rng.geometric(p=p)
        if i >= s.len():
            break
        pos.append(i)
    return pos


# Return the positions of the length k minimizers of each window of kmers of the given size.
# The window size does not include k: the possible kmers span a total of window+k-1 positions.
# The expected density of returned positions is 2/(w+1) [1], and the distance
# between consecutive minimizers is distributed uniformly on [1, window].
#
# Minimizers that start a subsequence of length l that go over the end of s are ignored.
#
# Ideally, A^k is larger than window to ensure uniqueness of the minimizer.
#
# [1]:Winnowing: local algorithms for document fingerprinting
#
# TODO: Add option to return the top t minimizers from each window size.
# TODO: Add option for force spacing between minimizers: discard/ignore minimizers within distance d of each other.
# TODO: Only return the first or last occurrence in case of equality.
def minimizers(s, k, window, A=4):
    # Keep a queue of minimizers. The hashes in the queue are always increasing.
    # We collect all positions that are at the front of the queue at some point in time.
    pos = []

    # Queue contains tuples (pos, hash), where both pos and hash are always increasing.
    q = collections.deque()

    Ak = A ** k
    # Value corresponding to the kmer.
    val = 0
    for i in range(k - 1):
        val = (A * val + s.seq[i]) % Ak

    hash_array = lib.hashing.get_hash(k, A)

    # For each kmer start position i:
    for i in range(0, s.len() - k + 1):
        # - Remove out-of-window items from the front of the queue.
        #   These are minimizer of the window ending just before position i, so store them in `pos`.
        if len(q) > 0 and q[0][0] == i - window - 1:
            # print('+ left pop', q[0])
            pos.append(q[0][0])
            q.popleft()
        # - Update the kmer value and compute the hash.
        val = (A * val + s.seq[i + k - 1]) % Ak
        h = lib.hashing.hash_int(val, k, cache=hash_array)

        # - remove earlier positions with a larger hash
        while len(q) > 0 and q[-1][1] >= h:
            if len(q) == 1 and i >= window:
                # print('+ last pop', q[-1])
                pos.append(q[0][0])
            else:
                # print('right pop', q[-1])
                pass
            q.pop()
        q.append((i, h))
        # print('push', q[-1])

    # Add the rest of the queue, until the entire sequence is covered.
    while len(q) > 0:
        p, h = q.popleft()
        pos.append(p)
        if p > s.len() - window - k:
            break

    return pos


# Takes a sequence s, the minimizer k, the minimizer positions, and the subsequence length.
# Returns lists of [(position, subsequence)].
def make_subsequences(s, k, positions, l, method=2):
    length = s.len()
    if method == 1:
        # Option 1: **start** windows at minimizers.
        return [(p, s.subsequence(p, l)) for p in positions if p + l <= length]
    if method == 2:
        # Option 2: **center** windows around minimizers.
        offset = (l - k) // 2
        # print('OFFSET', offset)
        return [
            (p, s.subsequence(p - offset, l))
            for p in positions
            if 0 <= p - offset and p - offset + l <= length
        ]
    assert False
