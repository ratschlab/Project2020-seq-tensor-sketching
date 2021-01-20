# default parameters
SEQGEN_PARAMS="--alphabet_size 4 --num_seqs 4000 --seq_len 10000 --min_mutation_rate 0.0 --max_mutation_rate 1.0 --phylogeny_shape path --group_size 2"
MODEL_PARAMS="--kmer_size 8 --tuple_length 3 --embed_dim 64 --tss_dim 8 --stride 100 --window_size 1000 --hash_alg crc32 --num_threads 0"
EXP=$(echo ./cmake-build-release/experiments $SEQGEN_PARAMS  $MODEL_PARAMS)
DATA_DIR=./experiments/data/grid_search


## table 1
for K in 1 2 3 4 6 8 10 12 14 16 18 20
do
  for T in 2 3 4 5
  do
    $EXP -o $DATA_DIR/k${K}_t${T}  --kmer_size $K --tuple_length $T
  done
done
