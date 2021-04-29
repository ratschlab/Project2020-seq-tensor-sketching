#!/usr/bin/env python3

# Playground for experimenting with the density of minimizers on uniform sequences.

import random
import numpy as np

from lib.minimizers import *


# Takes a sequence of hashes x, and a function S : A^w -> i and applies it to all w-mers.
def generic_minimizers(x, w, S):
    pos = set()
    for i in range(len(x) - w + 1):
        pos.add(S(x[i : i + w]) + i)

    return list(sorted(pos))


# First minimum
def S_minimum(l):
    return np.argmin(l)


# First minimum of maximum of k adjacent values.
def S_maxmin(l):
    l2 = [max(x, y) for x, y in zip(l, l[1:])]
    i = np.argmin(l2)
    v = l2[i]
    # version A: return i, the left of the min max pair.
    # version B: return the position of v, the max of the min max pair.
    if l[i] != v:
        i += 1
        assert l[i] == v

    return i


def S_median(l):
    l2 = sorted(l)
    w = len(l)
    median = l2[w // 2]
    for i, x in enumerate(l):
        if x == median:
            return i


def S_cyclic_minimum(l, A=1000000):
    cnt = [0] * len(l)
    for i in range(len(l)):
        for j in range(len(l)):
            d = l[i] - l[j]
            if d < 0:
                d += A
            if d <= A // 2:
                cnt[i] += 1
    return np.argmin(cnt)


def uniform_seq(A, l):
    return list(random.choices(range(A), k=l))


l = 1000000
w = 9
A = 1000000
x = uniform_seq(A, l)


def test(S):
    num_pos = len(generic_minimizers(x, w, S))
    density = num_pos / (l - w + 1)
    expected_density = 2 / (w + 1)
    print('Test')
    print(num_pos, density, expected_density, sep='\n')
    print()


test(S_minimum)
test(S_maxmin)
test(S_median)
test(S_cyclic_minimum)
