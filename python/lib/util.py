import time
import collections

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def print_sketch_statistics(sketches):
    # For each coordinate print and plot: the min, avg, std, max
    # min/avg/std/max for the norm.
    # Distribution over 3D space in the first 3 coordinates.

    n = len(sketches)
    D = len(sketches[0])
    print(f'n = {n}  D = {D}')

    indexes = []
    values = []
    values_per_index = [[] for _ in range(D)]
    norms = []
    for i, sketch in enumerate(sketches):
        for j, x in enumerate(sketch):
            indexes.append(str(j))
            values.append(x)
            values_per_index[j].append(x)
        norms.append(np.linalg.norm(sketch))

    volume = 1

    q1 = 5
    q2 = 95

    for j, vs in enumerate(values_per_index):
        qs = np.percentile(vs, q1)
        qe = np.percentile(vs, q2)
        print(f'Index {j}: {q1}% {qs: .3f}  {q2}% {qe: .3f}  width {qe-qs: .3f}')
        volume *= qe - qs

    print(f'Volume          : {volume: .4e}  [contains ~{(q2-q1)/100}^{D} of the points]')
    per_point_vol = volume / n
    scaled_per_point_vol = volume / (n * pow((q2 - q1) / 100, D))
    print(f'Volume per point: {per_point_vol: .4e}   or scaled {scaled_per_point_vol: .4e}')
    point_distance = pow(scaled_per_point_vol, 1.0 / D)
    print(f'Uniform distance: {point_distance: .4f}')

    print('Distribution of value per sketch dimension.')
    sns.displot(x=values, hue=indexes, kind='kde')
    print('Distribution of norm of sketches.')
    sns.displot(x=norms, kind='kde')
