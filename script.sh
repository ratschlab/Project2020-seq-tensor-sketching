EXP=$(echo ./cmake-build-release/experiments --flagfile experiments_flags)
ROOT_DIR=./experiments/data

# clean directory
P=$(pwd)
cd $ROOT_DIR || return
rm -Rf table1 fig1a fig1b fig1c fig1d fig2
mkdir table1 fig1a fig1b fig1c fig1d fig2
cd $P || return

# table 1
$EXP -o $ROOT_DIR/table1 --num_seqs 1000 --seq_len 10000 --stride 1000 --window_size=2000

# fig 1
FIG1=$(echo $EXP --num_seqs 1000 --seq_len 10000 --stride 1000 --window_size=2000)

$FIG1 -o $ROOT_DIR/fig1/a

SAVE_DIR=$ROOT_DIR/fig1/b
$FIG1 -o $SAVE_DIR/1 --seq_len 2000 --stride 200 --window_size 400
$FIG1 -o $SAVE_DIR/2 --seq_len 3000 --stride 300 --window_size 600
$FIG1 -o $SAVE_DIR/3 --seq_len 4000 --stride 400 --window_size 800
$FIG1 -o $SAVE_DIR/4 --seq_len 5000 --stride 500 --window_size 1000

SAVE_DIR=$ROOT_DIR/fig1/c
$FIG1 -o $SAVE_DIR/1 --seq_len 2000 --stride 200 --window_size 400 --hash_alg crc32
$FIG1 -o $SAVE_DIR/2 --seq_len 3000 --stride 300 --window_size 600 --hash_alg crc32
$FIG1 -o $SAVE_DIR/3 --seq_len 4000 --stride 400 --window_size 800 --hash_alg crc32
$FIG1 -o $SAVE_DIR/4 --seq_len 5000 --stride 500 --window_size 1000 --hash_alg crc32

SAVE_DIR=$ROOT_DIR/fig1/d
$FIG1 -o $SAVE_DIR/1 --embed_dim 10
$FIG1 -o $SAVE_DIR/2 --embed_dim 20
$FIG1 -o $SAVE_DIR/3 --embed_dim 30
$FIG1 -o $SAVE_DIR/4 --embed_dim 40
$FIG1 -o $SAVE_DIR/6 --embed_dim 60
$FIG1 -o $SAVE_DIR/8 --embed_dim 80
$FIG1 -o $SAVE_DIR/10 --embed_dim 100

##fig 2
$EXP -o $ROOT_DIR/fig2 --embed_dim 100 --phylogeny_shape tree --num_seqs 64 --group_size 64  --min_mutation_rate=.1 --max_mutation_rate=.1
