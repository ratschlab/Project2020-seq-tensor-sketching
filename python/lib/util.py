import time
import collections

import numpy as np

# Time the duration of the given lambda and print it.
def timeit(f, name=''):
    s = time.time()
    ret = f()
    e = time.time()
    print(f'Duration [{name}]: {e-s:4.4f}')
    return ret


# Print: min, mean (std), median, max, 5 lowest + 5 largest values with counts, 5 most common values
def print_stats(name, data, extended=True):
    if len(data) == 0:
        print(f'{name} has no data')
        print()
        return
    name = name.upper()
    minval = min(data)
    mean = np.mean(data)
    std = np.std(data)
    maxval = max(data)
    sumval = sum(data)

    counts = collections.defaultdict(int)
    for x in data:
        counts[x] += 1

    counts = list(counts.items())

    # Sort by value, get low and high 5.
    counts = sorted(counts, key=lambda x: x[0])
    if len(counts) <= 10:
        lowhigh = counts
    else:
        lowhigh = counts[:5] + ['...'] + counts[-5:]

    # Sort by count, get the 5 most frequent values.
    counts = sorted(counts, key=lambda x: x[1])
    maxcount = reversed(counts[-5:])

    print(name)
    print(f'{minval: 6} <= {mean: 8.1f}=μ ({std: 8.1f}=σ) <= {maxval: 6};     sum={sumval}')
    if extended:
        print('Low/high vals:', *lowhigh)
        print('Frequent vals:', *maxcount)

    if len(data) <= 20:
        print(*data)
    print()
