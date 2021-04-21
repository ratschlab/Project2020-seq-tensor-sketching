# Helper functions to parse the .CDS files in the homology dataset.

import os
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict

# Print: min, mean (std), median, max, 5 lowest + 5 largest values with counts, 5 most common values
def _print_stats(name, data):
    name = name.upper()
    minval = min(*data)
    mean = np.mean(data)
    std = np.std(data)
    maxval = max(*data)
    sumval = sum(data)

    counts = defaultdict(int)
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
    print('Low/high vals:', *lowhigh)
    print('Frequent vals:', *maxcount)

    if len(data) <= 20:
        print(*data)
    print()


def exon_stats(fasta_paths, sequences):
    exon_lengths = []
    num_exons = []
    total_exon_lengths = []

    id_to_exons = dict()
    for f in fasta_paths:
        data = json.loads(f.with_suffix('.CDS.json').read_text())
        id_to_exons |= data

    print(len(id_to_exons))

    i = 0
    for s in sequences:
        exons = id_to_exons[s.metadata['tid']]
        total_exon_length = 0

        num_exons.append(len(exons))

        for exon in exons:
            l = exon['end'] - exon['start']
            exon_lengths.append(l)
            total_exon_length += l

        _print_stats('Exon length', exon_lengths)
        _print_stats('Exons per gene', num_exons)
        _print_stats('Total exon length per gene', total_exon_lengths)

    sns.displot(exon_lengths)
    plt.show()
