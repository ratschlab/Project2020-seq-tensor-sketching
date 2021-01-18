EXP=$(echo ./cmake-build-release/experiments --flagfile experiments_flags)
ROOT_DIR=./experiments/data

# clean directory & create sub-directories
P=$(pwd)
cd $ROOT_DIR || return
rm -Rf table1 fig1a fig1b fig1c fig1d fig2
mkdir table1 fig1a fig1b fig1c fig1d fig2
cd $P || return

## table 1
$EXP -o $ROOT_DIR/table1 --num_seqs 4000 --seq_len 10000 --stride 1000 --window_size 2000

## Figure 1
FIG1=$(echo $EXP --num_seqs 2000)

# fig 1a
$FIG1 --seq_len 10000 --stride 1000 --window_size 2000

# fig 1b
SAVE_DIR=$ROOT_DIR/fig1/b
$FIG1 -o $SAVE_DIR/0 --seq_len 10 --stride 1 --window_size 2
$FIG1 -o $SAVE_DIR/1 --seq_len 20 --stride 2 --window_size 4
$FIG1 -o $SAVE_DIR/2 --seq_len 40 --stride 4 --window_size 8
$FIG1 -o $SAVE_DIR/3 --seq_len 80 --stride 8 --window_size 16
$FIG1 -o $SAVE_DIR/4 --seq_len 160 --stride 16 --window_size 32
$FIG1 -o $SAVE_DIR/5 --seq_len 320 --stride 32 --window_size 64
$FIG1 -o $SAVE_DIR/6 --seq_len 640 --stride 64 --window_size 128
$FIG1 -o $SAVE_DIR/7 --seq_len 1280 --stride 128 --window_size 256
$FIG1 -o $SAVE_DIR/8 --seq_len 2560 --stride 256 --window_size 512
$FIG1 -o $SAVE_DIR/9 --seq_len 5120 --stride 512 --window_size 1024
$FIG1 -o $SAVE_DIR/10 --seq_len 10240 --stride 1024 --window_size 2048

# fig 1c: use crc32 to measure time
FIG1_time=$(echo $FIG1 --hash_alg crc32)
SAVE_DIR=$ROOT_DIR/fig1/c
$FIG1_time -o $SAVE_DIR/0 --seq_len 10 --stride 1 --window_size 2
$FIG1_time -o $SAVE_DIR/1 --seq_len 20 --stride 2 --window_size 4
$FIG1_time -o $SAVE_DIR/2 --seq_len 40 --stride 4 --window_size 8
$FIG1_time -o $SAVE_DIR/3 --seq_len 80 --stride 8 --window_size 16
$FIG1_time -o $SAVE_DIR/4 --seq_len 160 --stride 16 --window_size 32
$FIG1_time -o $SAVE_DIR/5 --seq_len 320 --stride 32 --window_size 64
$FIG1_time -o $SAVE_DIR/6 --seq_len 640 --stride 64 --window_size 128
$FIG1_time -o $SAVE_DIR/7 --seq_len 1280 --stride 128 --window_size 256
$FIG1_time -o $SAVE_DIR/8 --seq_len 2560 --stride 256 --window_size 512
$FIG1_time -o $SAVE_DIR/9 --seq_len 5120 --stride 512 --window_size 1024
$FIG1_time -o $SAVE_DIR/10 --seq_len 10240 --stride 1024 --window_size 2048

# fig 1d
SAVE_DIR=$ROOT_DIR/fig1/d
$FIG1 -o $SAVE_DIR/1 --embed_dim 10
$FIG1 -o $SAVE_DIR/2 --embed_dim 20
$FIG1 -o $SAVE_DIR/3 --embed_dim 30
$FIG1 -o $SAVE_DIR/4 --embed_dim 40
$FIG1 -o $SAVE_DIR/6 --embed_dim 60
$FIG1 -o $SAVE_DIR/8 --embed_dim 80
$FIG1 -o $SAVE_DIR/10 --embed_dim 100

# fig 2
$EXP -o $ROOT_DIR/fig2 --embed_dim 100 --phylogeny_shape tree --num_seqs 128 --group_size 128  --min_mutation_rate .1 --max_mutation_rate .1
