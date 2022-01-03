# Similarity Estimation via Tensor Sketching
This repository contains the reference implementation for the Tensor Sketching method, which can be used to estimate sequence similarity without needing to align the sequences.

The method is described in the paper by Amir Joudaki et al. [`Fast Alignment-Free Similarity Estimation By Tensor
 Sketching`](https://www.biorxiv.org/content/10.1101/2020.11.13.381814v5.full).

## Download and build
```
git clone https://github.com/ratschlab/Project2020-seq-tensor-sketching
cd Project2020-seq-tensor-sketching
git submodule update --init --
mkdir build; cd build
cmake ..
make -j
```

## Run 
The `sketch` binary expects as input a directory containing fasta files (with extension `.fa`, `.fasta` or `.fna`), 
each fasta file containing a single sequence:
```bash
./sketch -i /tmp/  -o /tmp/sketch_triangle
```

The output file will contain the number of sequences on the first line and the pairwise distances between each 
sequence on the following lines, e.g.:
```
        4
test2.fa
test3.fa        0.28125
test4.fa        1.06314 0.915816
test1.fa        0       0.28125 1.06314
```
For example, the distance between test1.fa and test2.fa is 0 (the lower the distance the more similar the sequences).

### Flags
To see all available flags, run:
```
./sketch --help
```
Here are the most important flags:

`-m`, `--sketch_method`: the sketching method to use; can be one of `MH, WMH, OMH, TS, TSB or TSS`, which corresponds to
min-hash, weigheted-min-has, ordered-min-hash, tensor-sketch, tensor-block and tensor-slide-sketch, respectively.

`-k`, `--kmer_length`: the length of the k-mer used in the sketching method (default=3)

`--embed_dim`: the dimension of the embedded space used in all sketching methods (default=4)

`-t, --tuple-length`: the ordered tuple length, not used in Min-hash and Weighted-min-hash (default=3)

`--block_size`: only consider tuples made out of block-size continuous characters for Tensor sketch (default=1)

`-w, --window_size`: the size of sliding window in Tensor Slide Sketch (default=32)

`--max_len`: the maximum accepted sequence length for Ordered and Weighted min-hash (default=32)

`-s, --stride`: stride for sliding window: shift step for sliding window (default=8)
## Contributing

- The python code in the repository is formatted using [black](https://github.com/psf/black).
  To enable the pre-commit hook, install [pre-commit](https://pre-commit.com/)
  with `pip` or your package manager (Arch: `python-pre-commit`) and run
  `pre-commit install` from the repository root. All python code will now automatically be formatted
  on each commit.
