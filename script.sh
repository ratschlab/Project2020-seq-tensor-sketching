# default parameters
SEQGEN_PARAMS="--alphabet_size 4 --num_seqs 4000 --seq_len 10000 --min_mutation_rate 0.0 --max_mutation_rate 1.0 --phylogeny_shape path --group_size 2"
MODEL_PARAMS="--kmer_size 8 --tuple_length 3 --embed_dim 64 --tss_dim 8 --stride 100 --window_size 1000 --hash_alg crc32 --num_threads 0"
EXP=$(echo ./cmake-build-release/experiments $SEQGEN_PARAMS  $MODEL_PARAMS)
DATA_DIR=./experiments/k8t3/data

# clean directory & create sub-directories
P=$(pwd)
cd $DATA_DIR || return
rm -Rf table1 fig1 fig2
mkdir table1 fig1 fig2
cd $P || return

## table 1
$EXP -o $DATA_DIR/table1 --hash_alg crc32

## fig 1a
$EXP -o $DATA_DIR/fig1/a
#
## fig 1b: stride set to seq_len/10, and window_size=2*stride
SAVE_DIR=$DATA_DIR/fig1/b
$EXP -o $SAVE_DIR/4 --seq_len 100 --stride 1 --window_size 10
$EXP -o $SAVE_DIR/5 --seq_len 200 --stride 2 --window_size 20
$EXP -o $SAVE_DIR/6 --seq_len 400 --stride 4 --window_size 30
$EXP -o $SAVE_DIR/7 --seq_len 800 --stride 8 --window_size 80
$EXP -o $SAVE_DIR/8 --seq_len 1600 --stride 16 --window_size 160
$EXP -o $SAVE_DIR/9 --seq_len 3200 --stride 32 --window_size 320
$EXP -o $SAVE_DIR/10 --seq_len 6400 --stride 64 --window_size 640

## fig 1b: stride set to seq_len/10, and window_size=2*stride
SAVE_DIR=$DATA_DIR/fig1/c
$EXP -o $SAVE_DIR/4 --seq_len 100 --stride 1 --window_size 10
$EXP -o $SAVE_DIR/5 --seq_len 200 --stride 2 --window_size 20
$EXP -o $SAVE_DIR/6 --seq_len 400 --stride 4 --window_size 30
$EXP -o $SAVE_DIR/7 --seq_len 800 --stride 8 --window_size 80
$EXP -o $SAVE_DIR/8 --seq_len 1600 --stride 16 --window_size 160
$EXP -o $SAVE_DIR/9 --seq_len 3200 --stride 32 --window_size 320
$EXP -o $SAVE_DIR/10 --seq_len 6400 --stride 64 --window_size 640
#
# fig 1d
SAVE_DIR=$DATA_DIR/fig1/d
$EXP -o $SAVE_DIR/1 --embed_dim 4 --tss_dim 2
$EXP -o $SAVE_DIR/2 --embed_dim 9 --tss_dim 3
$EXP -o $SAVE_DIR/3 --embed_dim 16 --tss_dim 4
$EXP -o $SAVE_DIR/4 --embed_dim 25 --tss_dim 5
$EXP -o $SAVE_DIR/6 --embed_dim 36 --tss_dim 6
$EXP -o $SAVE_DIR/8 --embed_dim 100 --tss_dim 10
$EXP -o $SAVE_DIR/10 --embed_dim 225 --tss_dim 15
$EXP -o $SAVE_DIR/11 --embed_dim 400 --tss_dim 20
$EXP -o $SAVE_DIR/12 --embed_dim 900 --tss_dim 30
#
## fig 2
$EXP -o $DATA_DIR/fig2 --phylogeny_shape tree --num_seqs 64 --group_size 64  --min_mutation_rate .1 --max_mutation_rate .1
