from itertools import product
import os, math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn import metrics
from matplotlib import pyplot as plt
from glob import glob
import seaborn as sns

sns.set(context="paper", style="white", font_scale=1)


def load_results(data_dir, thresh):
    flags = pd.read_csv(os.path.join(data_dir, 'flags'), delimiter='=', header=None, names=['name', 'value'])
    flags = {row['name'].strip('-'): row['value'] for _, row in flags.iterrows()}
    flags['pairs'] = float(flags['num_seqs']) / float(flags['group_size'])
    flags['pairs'] = str(int(flags['pairs']))
    seq_len = float(flags['seq_len'])

    dists = pd.read_csv(os.path.join(data_dir, 'dists.csv'))
    columns = [l for l in dists.columns]
    methods = columns[3:8]
    methods.append(columns[2])  # put ED at the end

    times = pd.read_csv(os.path.join(data_dir, 'timing.csv'), skipinitialspace=True)
    times = {row['short name']: row['time'] for _, row in times.iterrows()}
    # times_rel = {k : (v/times['ED']) for k,v in times.items()}
    times_abs = [times[m] for m in methods]
    times_rel = [times[m] / times['ED'] for m in methods]

    auc = [[] for _ in thresh]
    for thi, th in enumerate(thresh):
        for m in methods:
            fpr, tpr, thresholds = metrics.roc_curve(dists['ED'] < th * seq_len, dists[m], pos_label=0)
            auc[thi].append(metrics.auc(fpr, tpr))

    sp_corr = []
    for m in methods:
        if len(pd.unique(dists[m])) == 1:  # check if dists[m] is a constant vector
            sr = 0
        else:
            sr = spearmanr(dists['ED'], dists[m]).correlation
        sp_corr.append(sr)

    stats = {'method': methods, 'Sp': sp_corr}
    stats.update({'AUC{}'.format(i): val for i, val in enumerate(auc)})
    stats.update({'AbsTime': times_abs, 'RelTime': times_rel})
    return flags, dists, stats


def load_sub_results(data_dir, grid_flags=None, thresh=None):
    if grid_flags is None:
        grid_flags = []
    if thresh is None:
        thresh = []
    dirs = glob(os.path.join(data_dir, '*'))
    data = pd.DataFrame()
    for path in dirs:
        flags, dists, stats = load_results(data_dir=path, thresh=thresh)
        for flag in grid_flags:
            stats[flag] = [int(flags[flag])] * len(stats['method'])
        data = pd.concat([data, pd.DataFrame(stats)])
    return data


def gen_table1(data_dir, save_dir, thresh):
    flags, dists, stats = load_results(data_dir=data_dir, thresh=thresh)
    # best Sp corr, AUC values (higher better), exclude edit distance
    best_row = {k: np.argmax(v[:-1]) for k, v in stats.items()}
    # best times (lower better), excluce edit distance
    best_row['AbsTime'] = np.argmin(stats['AbsTime'][:-1])
    best_row['RelTime'] = np.argmin(stats['RelTime'][:-1])
    for name, col in stats.items():
        if name == 'method':
            continue
        for i, v in enumerate(col):
            v = '{:.3f}'.format(v)
            if best_row[name] == i:
                stats[name][i] = '\\textbf{' + v + '}'
            else:
                stats[name][i] = v

    table_body = 'Method  & Spearman  & {} & Abs. ($10^{{-3}}$ sec) & Rel.(1/ED) \\\\\n\hline\n'.format(
        ' & '.join(str(th) for th in thresh))
    table_body = table_body + '\\\\\n\hline\n'.join(
        [' & '.join(col[row] for method, col in stats.items()) for row in range(6)])

    Min = float(flags['min_mutation_rate'])
    Max = float(flags['max_mutation_rate'])
    assert (Min <= Max)
    if Min < Max:
        mutation_rate = "mutation rate uniformly drawn from $[{:.2f},{:.2f}]$".format(Min, Max)
    else:
        mutation_rate = "mutation rate set to {:.2f}".format(Min)
    caption = """
\\caption{{${flags[pairs]}$ sequence pairs of length ${flags[seq_len]}$
were generated over an alphabet of size ${flags[alphabet_size]}$,
 with the {mutation_rate}.
The time column shows normalized time in microseconds, i.e., total time divided by number of sequences,
while the relative time shows the ratio of sketch-based time to the time for computing exact edit distance.
The embedding dimension is set to $D={flags[embed_dim]}$, and individual model parameters are
(a) MinHash $k = {flags[mh_kmer_size]}$,
(b) Weighted MinHash $k={flags[wmh_kmer_size]}$,
(c) Ordered MinHash $k={flags[omh_kmer_size]},t={flags[omh_tuple_length]}$,
(d) Tensor Sketch $t={flags[ts_tuple_length]}$,
(e) Tensor Slide Sketch $w={flags[tss_window_size]},t={flags[tss_tuple_length]}$. }}
    """
    caption = caption.format(flags=flags, mutation_rate=mutation_rate)

    table_latex = """
\\begin{table}[h!]
    """ + caption + """
\\centering
\\begin{tabular}{ |c|c|""" + 'c|' * len(thresh) + """c|c|}
\\hline
\\multicolumn{1}{|c|}{\\textbf{}} &
\\multicolumn{1}{|c|}{\\textbf{Correlation}} &
\\multicolumn{""" + str(len(thresh)) + """}{|c|}{\\textbf{AUROC ($\\ED \\le \\cdot $)}} &
\\multicolumn{2}{c|}{\\textbf{Time}} \\\\
\\hline
    """ + table_body + """\\\\
\\hline
\\end{tabular}
\\label{tabel1}
\\end{table}"""
    fout = open(os.path.join(save_dir, 'table1.tex'), 'w')
    fout.write(table_latex)
    fout.close()
    return table_latex


def gen_fig_s1(data_dir, save_dir):
    flags, dists, _ = load_results(data_dir=data_dir, thresh=[])
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    dists['ED'] = dists['ED'] / int(flags['seq_len'])     # normalize
    dists['ED_quant'] = pd.qcut(dists['ED'], q=100)

    cols = dists.columns[3:8]
    for mi, method in enumerate(cols):
        g = sns.scatterplot(ax=axes[int(mi / 3), mi % 3],
                            x=dists['ED'],
                            y=dists[method])
        g.set(xlabel='Normalized edit dist.',
              ylabel='Normalized sketch dist.',
              title=('({}) {}'.format(chr(ord('a') + mi), method)))



    fig.delaxes(axes[1][2])
    caption = """\\caption{{Sketch distances (normalizied by maximum) versus edit distance 
    (normalized by the sequence length). Overall, ${flags[pairs]}$ pairs of sequences, each with the fixed length 
    ${flags[seq_len]}$ were generated over ${flags[alphabet_size]}$ alphabets. One sequence was generated randomly, and the second was mutated, with the mutation rate uniformly drawn 
    from $({flags[min_mutation_rate]},{flags[max_mutation_rate]})$, to generate a spectrum of edit distances. Subplot 
    (a-e) show the sketch-based distances, normalized by their max value vs. edit distances, normalized by the 
    sequence length. The embedding dimension is set to $D={flags[embed_dim]}$, and models parameters are
    (a) MinHash $k = {flags[mh_kmer_size]}$,
    (b) Weighted MinHash $k={flags[wmh_kmer_size]}$,
    (c) Ordered MinHash $k={flags[omh_kmer_size]},t={flags[omh_tuple_length]}$,
    (d) Tensor Sketch $t={flags[ts_tuple_length]}$,
    (e) Tensor Slide Sketch $w={flags[tss_window_size]},t={flags[tss_tuple_length]}$.
    }} """
    caption = caption.format(flags=flags)
    plt.savefig(os.path.join(save_dir, 'FigS1.pdf'),bbox_inches='tight')
    fout = open(os.path.join(save_dir, 'FigS1.tex'), 'w')
    fout.write(caption)
    fout.close()


def gen_fig_s2(data_dir, save_dir, ed_th):
    flags, dists, stat = load_results(data_dir=data_dir, thresh=ed_th)
    data = {'fpr': [], 'tpr': [], 'method': [], 'th': []}
    for th in ed_th:
        seq_len = int(flags['seq_len'])
        cols = dists.columns[3:8]
        for mi, method in enumerate(cols):
            fpr, tpr, thresholds = metrics.roc_curve(dists['ED'] < th * seq_len, dists[method], pos_label=0)
            data['fpr'].extend(fpr)
            data['tpr'].extend(tpr)
            data['method'].extend([method] * len(fpr))
            data['th'].extend([th] * len(fpr))
    data = pd.DataFrame(data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for thi, th in enumerate(ed_th):
        ax = axes[int(thi / 2), thi % 2]
        g = sns.lineplot(ax=ax, data=data[data.th == th], x='fpr', y='tpr', hue='method')
        g.set(xlabel='False Positive',
              ylabel='True Positive',
              title='ROC to detect ED<{}'.format(th))

    Min = float(flags['min_mutation_rate'])
    Max = float(flags['max_mutation_rate'])
    assert (Min <= Max)
    if Min < Max:
        mutation_rate = "mutation rate uniformly drawn from $[{:.2f},{:.2f}]$".format(Min, Max)
    else:
        mutation_rate = "mutation rate set to {:.2f}".format(Min)
    caption = """\\caption{{ {flags[pairs]} sequence pairs of length ${flags[seq_len]}$ were generated over an 
    alphabet of size $\\#\\Abc={flags[alphabet_size]}$. with the {mutation_rate}. 
    Subplots (a)-(e) show the ROC curve for detecting pairs with edit distance (normalized by length) 
    less than ${th[0]},{th[1]},{th[2]},$ and ${th[3]}$ respectively. }} """
    caption = caption.format(flags=flags, th=ed_th, mutation_rate=mutation_rate)
    fo = open(os.path.join(save_dir, 'FigS2.tex'), 'w')
    plt.savefig(os.path.join(save_dir, 'FigS2.pdf'),bbox_inches='tight')
    fo.write(caption)
    fo.close()


def gen_fig1(data_dir, save_dir):
    figure_size = (5, 5)
    flags, dists, stats = load_results(data_dir=data_dir, thresh=[])

    data = {'auc': [], 'method': [], 'th': []}
    for th in np.linspace(.05, .5, 10):
        seq_len = int(flags['seq_len'])
        methods = dists.columns[3:8]
        for mi, method in enumerate(methods):
            fpr, tpr, thresholds = metrics.roc_curve(dists['ED'] < th * seq_len, dists[method], pos_label=0)
            data['auc'].append(metrics.auc(fpr, tpr))
            data['method'].append(method)
            data['th'].append(th)
    data = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=figure_size)
    g = sns.lineplot(ax=ax, data=data, x='th', y='auc', hue='method')
    g.set(xlabel='Edit distance threshold', ylabel='AUROC')
    plt.savefig(os.path.join(save_dir, 'Fig1a.pdf'))

    dirs = glob(os.path.join(data_dir, 'seq_len', '*'))
    data = pd.DataFrame()
    for path in dirs:
        flags, dists, stats = load_results(data_dir=path, thresh=[])
        stats['seq_len'] = [int(flags['seq_len'])] * len(stats['method'])
        data = pd.concat([data, pd.DataFrame(stats)])
    data = data[data.method != 'ED']
    fig, ax = plt.subplots(figsize=figure_size)
    g = sns.lineplot(ax=ax, data=data, x='seq_len', y='Sp', hue='method')
    g.set(xlabel='Sequence length', ylabel='Spearman Corr.')
    ax.set_xscale('log')
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    plt.savefig(os.path.join(save_dir, 'Fig1b.pdf'))

    fig, ax = plt.subplots(figsize=figure_size)
    g = sns.lineplot(ax=ax, data=data, x='seq_len', y='AbsTime', hue='method')
    g.set(xlabel='Sequence length', ylabel='Absolute Time (ms)')
    ax.set_xscale('log')
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    ax.set_yscale('log')
    ax.get_yaxis().get_major_formatter().labelOnlyBase = False
    plt.savefig(os.path.join(save_dir, 'Fig1c.pdf'))


    dirs = glob(os.path.join(data_dir, 'embed_dim', '*'))
    data = pd.DataFrame()
    for path in dirs:
        flags, dists, stats = load_results(data_dir=path, thresh=[])
        stats['embed_dim'] = [int(flags['embed_dim'])] * len(stats['method'])
        data = pd.concat([data, pd.DataFrame(stats)])
    data = data[data.method != 'ED']
    fig, ax = plt.subplots(figsize=figure_size)
    g = sns.lineplot(ax=ax, data=data, x='embed_dim', y='Sp', hue='method')
    ax.set_xscale('log')
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    g.set(xlabel='Embedding dimension', ylabel='Spearman Corr.')
    plt.savefig(os.path.join(save_dir, 'Fig1d.pdf'))

    caption = """
    \\caption{{
The dataset for these experiments consisted of ${flags[num_seqs]}$ sequence pairs independently generated 
over an alphabet of size ${flags[alphabet_size]}$. The embedding dimension is set to $D={flags[embed_dim]}$, 
and model-specific parameters are MH $k = {flags[mh_kmer_size]}$, WMH $k={flags[wmh_kmer_size]}$,
OMH $k={flags[omh_kmer_size]},t={flags[omh_tuple_length]}$,
TS $t={flags[ts_tuple_length]}$,
TSS $w={flags[tss_window_size]},t={flags[tss_tuple_length]}$.
(\\ref{{fig:AUROC}}) Area Under the ROC Curve (AUROC), for detection of edit distances below a threshold using the sketch-based approximations.  
The x-axis, shows which edit distance (normalized) is used, and the y axis shows AUROC for various sketch-based distances.  
(\\ref{{fig:Spearman_vs_len}}) The Spearman's rank correlation is plotted against the sequence length (logarithmic scale). 
(\\ref{{fig:Time_vs_len}}) Similar setting to (\\ref{{fig:Spearman_vs_len}}), plotting the execution time 
of each sketching method (y-axis, logarithmic scale) as a function of sequence length (x-axis, logarithmic scale). 
The reported times are normalized, i.e., average sketching time plus average distance computation time for each method. 
(\\ref{{fig:Spearman_vs_embed}}) Spearman rank correlation of each sketching method as a function 
of the embedding dimension $D$ (x-axis, logarithmic scale). 
}}
    """.format(flags=flags)
    fo = open(os.path.join(save_dir, 'Fig1.tex'), 'w')
    fo.write(caption)
    fo.close()


def gen_fig2(data_dir, save_dir):
    flags, dists, _ = load_results(data_dir=data_dir, thresh=[.1, .2, .5])
    cols = dists.columns[2:8]
    num_seqs = int(flags['num_seqs'])
    d_sq = np.zeros((num_seqs, num_seqs))
    s1 = dists['s1'].astype(int)
    s2 = dists['s2'].astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for mi, method in enumerate(cols):
        d_rank = rankdata(dists[method])
        for i, d in enumerate(d_rank):
            d_sq[s1[i], s2[i]] = d
            d_sq[s2[i], s1[i]] = d

        g = sns.heatmap(ax=axes[int(mi / 3), mi % 3], data=d_sq, cbar=False, xticklabels=[], yticklabels=[])
        g.set(xlabel='seq #', ylabel='seq #', title='({}) {}'.format(chr(ord('a') + mi), method))

    Min = float(flags['min_mutation_rate'])
    Max = float(flags['max_mutation_rate'])
    assert (Min <= Max)
    if Min < Max:
        mutation_rate = "mutation rate uniformly drawn from $[{:.2f},{:.2f}]$".format(Min, Max)
    else:
        mutation_rate = "mutation rate set to {:.2f}".format(Min)
    num_generations = int(math.log2(num_seqs))
    caption = """\\caption{{ The subplot (a) illustrate the exact edit distance matrix, while the subplots (b)-(f) 
    demonstrate the approximate distance matrices based on sketching methods. To highlight how well each method 
    preserves the rank of distances, In all plots, the color-code indicates the rank of each distance (darker, 
    smaller distance). The phylogeny was simulated by an initial random sequence of length $\\SLen={flags[seq_len]}$, 
    over an alphabet of size $\\#\\Abc={flags[alphabet_size]}$. Afterwards, for ${num_generations}$ generations, 
    each sequence in the gene pool was mutated and added back to the pool, resulting in ${flags[num_seqs]}$ sequences 
    overall. The {mutation_rate}, to produce a diverse set of distances. For all sketching algorithms, 
    embedding dimension is set to $\\EDim={flags[embed_dim]}$, and individual model parameters are set to 
    (b) MinHash $k = {flags[mh_kmer_size]}$, 
    (c) Weighted MinHash $k={flags[wmh_kmer_size]}$, 
    (d) Ordered MinHash $k={flags[omh_kmer_size]},t={flags[omh_tuple_length]}$, 
    (e) Tensor Sketch $t={flags[ts_tuple_length]}$, 
    (f) Tensor Slide Sketch $w={flags[tss_window_size]},t={flags[tss_tuple_length]}, D={flags[tss_dim]}$. }} """
    caption = caption.format(flags=flags, num_generations=num_generations, mutation_rate=mutation_rate)
    fo = open(os.path.join(save_dir, 'Fig2.tex'), 'w')
    plt.savefig(os.path.join(save_dir, 'Fig2.pdf'), bbox_inches='tight')
    fo.write(caption)
    fo.close()


def default_params_pairs():
    return {'alphabet_size': 4,
            'num_seqs': 2000,
            'group_size': 2,
            'seq_len': 10000,
            'min_mutation_rate': 0.0,
            'max_mutation_rate': 1.0,
            'phylogeny_shape': 'path',
            'embed_dim': 64,
            'hash_alg': 'crc32',
            'num_threads': 0}


def default_params_tree():
    return {'alphabet_size': 4,
            'num_seqs': 64,
            'group_size': 64,
            'seq_len': 10000,
            'min_mutation_rate': 0.10,
            'max_mutation_rate': 0.10,
            'phylogeny_shape': 'tree',
            'embed_dim': 64,
            'hash_alg': 'crc32',
            'num_threads': 0}


def opts2flags(flags: dict):
    opts = " "
    for flag, val in flags.items():
        opts += '--{flag} {val} '.format(flag=flag,val=val)
    return opts


def find_optimal_params(data_dir):
    data = load_sub_results(data_dir=data_dir, grid_flags=['kmer_size', 'tuple_length'])

    params = dict()
    # find optimal parameters according to Spearman Corr.
    Metric = "Sp"
    med_acc = data[data.method == "OMH"].groupby(["kmer_size", "tuple_length"])[Metric].median()
    k, t = med_acc.idxmax()
    params.update({'omh_kmer_size': k, 'omh_tuple_length': t})
    med_acc = data[data.method == "MH"].groupby(["kmer_size"])[Metric].median()
    k = med_acc.idxmax()
    params.update({'mh_kmer_size': k})
    med_acc = data[data.method == "WMH"].groupby(["kmer_size"])[Metric].median()
    k = med_acc.idxmax()
    params.update({'wmh_kmer_size': k})
    med_acc = data[data.method == "TS"].groupby(["tuple_length"])[Metric].median()
    t = med_acc.idxmax()
    params.update({'ts_tuple_length': t})
    med_acc = data[data.method == "TSS"].groupby(["tuple_length"])[Metric].median()
    t = med_acc.idxmax()
    params.update({'tss_tuple_length': t})

    return params


def test_optimal_params(experiments_dir, binary_path, num_runs=None, seq_lens=None, embed_dims=None):
    if num_runs is None:
        num_runs = 5
    if embed_dims is None:
        embed_dims = [4, 16, 32, 64, 256]
    if seq_lens is None:
        seq_lens = [2000, 4000, 8000, 16000]
    params = find_optimal_params(data_dir=os.path.join(experiments_dir, 'data', 'grid_search_pairs'))
    print('optimal params (pairs): ', params)
    params.update(default_params_pairs())
    params['o'] = os.path.join(experiments_dir, 'data', 'pairs')
    os.system(binary_path + opts2flags(params))

    for ri in range(10,10+num_runs):
        for seq_len in seq_lens:
            params.update(
                {'o': os.path.join(experiments_dir, 'data', 'pairs', 'seq_len', 'len{}_r{}'.format(seq_len, ri)),
                 'seq_len': seq_len})
            os.system(binary_path + opts2flags(params))

    params.update(default_params_pairs())
    for ri in range(10,10+num_runs):
        for embed_dim in embed_dims:
            params.update(
                {'o': os.path.join(experiments_dir, 'data', 'pairs', 'embed_dim', 'dim{}_r{}'.format(embed_dim, ri)),
                 'embed_dim': embed_dim})
            os.system(binary_path + opts2flags(params))

    tree_params = find_optimal_params(data_dir=os.path.join(experiments_dir, 'data', 'grid_search_tree'))
    print('optimal params (tree): ', tree_params)
    tree_params.update(default_params_tree())
    tree_params.update({'o': os.path.join(experiments_dir, 'data', 'tree')})
    os.system(binary_path + opts2flags(tree_params))


def gen_plots(experiments_dir, plots_dir):
    gen_table1(data_dir=os.path.join(experiments_dir, 'data', 'pairs'),
               save_dir=plots_dir, thresh=[.1, .2, .3, .5])

    gen_fig_s1(data_dir=os.path.join(experiments_dir, 'data', 'pairs'),
               save_dir=plots_dir)

    gen_fig_s2(data_dir=os.path.join(experiments_dir, 'data', 'pairs'),
               save_dir=plots_dir, ed_th=[.1, .2, .3, .5])

    gen_fig1(data_dir=os.path.join(experiments_dir, 'data', 'pairs'),
             save_dir=plots_dir)

    gen_fig2(data_dir=os.path.join(experiments_dir, 'data', 'tree'),
             save_dir=plots_dir)


def grid_search(experiments_dir, binary_path,
                num_runs = None, k_range=None, t_range=None):
    if k_range is None:
        k_range = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    if t_range is None:
        t_range = range(2,11)
    if num_runs is None:
        num_runs = 3
    params = {'pairs': default_params_pairs(), 'tree': default_params_tree()}

    for grid_type, param in params.items():
        for run in range(num_runs):
            for k in k_range:
                for t in t_range:
                    path = os.path.join(experiments_dir, 'data',
                                        'grid_search_{}'.format(grid_type),
                                        'run{}_k{}_t{}'.format(run,k,t))
                    params.update({'kmer_size': k, 'tuple_length': t, 'o': path})
                    os.system(binary_path + opts2flags(param) )


if __name__ == '__main__':

    experiments_dir = './experiments'
    binary_path = './cmake-build-release/experiments'
    plots_dir = os.path.join(experiments_dir, 'figures')

    # grid_search(experiments_dir=experiments_dir, binary_path=binary_path)
    #
    # test_optimal_params(binary_path=binary_path, experiments_dir=experiments_dir, num_runs=10)

    gen_plots(experiments_dir=experiments_dir, plots_dir=plots_dir)
