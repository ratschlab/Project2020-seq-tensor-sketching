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
