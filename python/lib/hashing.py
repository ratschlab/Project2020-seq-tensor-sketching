# Methods for hashing k-mers.
# Each of them uses a global cache.

import random

_hash_cache = dict()


def _init_hash(k, A):
    # TODO: Hashing for k > 5, where we have to use random hash functions
    # instead of a full map.
    assert A ** k <= 1000000

    # For k <= 4, shuffle an array of size A^k.
    vals = list(range(A ** k))
    random.shuffle(vals)
    _hash_cache[(k, A)] = vals


def get_hash(k, A):
    if (k, A) not in _hash_cache:
        _init_hash(k, A)
    return _hash_cache[(k, A)]


# Takes an integer in [0, A^k) and returns its hash.
# Optionally pass the hash array directly to skip the additional lookup.
# kmers are hashed into values by reading them in base A, with the rightmost character being the units.
def hash_int(kmer_val, k, A=4, cache=None):
    if cache is None:
        cache = get_hash(k, A)
    return cache[kmer_val]


# Takes the kmer as a list of ints and returns the hash.
def hash_string(kmer, k, A=4):
    pass
