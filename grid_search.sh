# grid search: independet pairs
SEQGEN_PARAMS="--alphabet_size 4 --num_seqs 2000 --seq_len 10000 --min_mutation_rate 0.0 --max_mutation_rate 1.0 --phylogeny_shape path --group_size 2"
MODEL_PARAMS=" --embed_dim 64 --hash_alg crc32 --num_threads 0"
EXP="./cmake-build-release/experiments $SEQGEN_PARAMS  $MODEL_PARAMS"
DATA_DIR=./experiments/data/grid_search_pairs


for RUN in 1 2 3
  do
  for T in 2 3 4 5 6 7 8 9 10
    do
    for K in 1 2 3 4 6 8 10 12 14 16
    do
      $EXP -o $DATA_DIR/k${K}_t${T}_RUN${RUN}  --kmer_size $K --tuple_length $T
    done
  done
done


# grid search: tree
SEQGEN_PARAMS="--alphabet_size 4 --seq_len 10000 --min_mutation_rate 0.15 --max_mutation_rate 0.15 --phylogeny_shape tree --num_seqs 64  --group_size 64"
MODEL_PARAMS="--embed_dim 64 --hash_alg crc32 --num_threads 0"
EXP="./cmake-build-release/experiments $SEQGEN_PARAMS  $MODEL_PARAMS"
DATA_DIR=./experiments/data/grid_search_tree


for RUN in 1 2 3
  do
  for T in 2 3 4 5 6 7 8 9 10
    do
    for K in 1 2 3 4 6 8 10 12 14 16
    do
      $EXP -o $DATA_DIR/k${K}_t${T}_RUN${RUN}  --kmer_size $K --tuple_length $T
    done
  done
done

