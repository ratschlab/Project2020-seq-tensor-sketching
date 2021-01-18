EXP=$(echo ./cmake-build-release/experiments --flagfile experiments_flags)
SAVE_DIR=/tmp/experiments

# clean directory
P=$(pwd)
cd $SAVE_DIR || return
rm -Rf table1 fig1a fig1b fig1c fig1d fig2
mkdir table1 fig1a fig1b fig1c fig1d fig2
cd $P || return

# table 1
$EXP -o $SAVE_DIR/table1 --num_seqs 1000 --seq_len 10000 --stride 1000 --window_size=2000

# fig 1
FIG1=$(echo $EXP --num_seqs 1000 --seq_len 10000 --stride 1000 --window_size=2000)

$FIG1 -o $SAVE_DIR/fig1a

$FIG1 -o $SAVE_DIR/fig1b/1 --seq_len 2000 --stride 200 --window_size 400
$FIG1 -o $SAVE_DIR/fig1b/2 --seq_len 3000 --stride 300 --window_size 600
$FIG1 -o $SAVE_DIR/fig1b/3 --seq_len 4000 --stride 400 --window_size 800
$FIG1 -o $SAVE_DIR/fig1b/4 --seq_len 5000 --stride 500 --window_size 1000

$FIG1 -o $SAVE_DIR/fig1c/1 --seq_len 2000 --stride 200 --window_size 400 --hash_alg crc32
$FIG1 -o $SAVE_DIR/fig1c/2 --seq_len 3000 --stride 300 --window_size 600 --hash_alg crc32
$FIG1 -o $SAVE_DIR/fig1c/3 --seq_len 4000 --stride 400 --window_size 800 --hash_alg crc32
$FIG1 -o $SAVE_DIR/fig1c/4 --seq_len 5000 --stride 500 --window_size 1000 --hash_alg crc32

$FIG1 -o $SAVE_DIR/fig1d/1 --embed_dim 10
$FIG1 -o $SAVE_DIR/fig1d/2 --embed_dim 20
$FIG1 -o $SAVE_DIR/fig1d/3 --embed_dim 30
$FIG1 -o $SAVE_DIR/fig1d/4 --embed_dim 40
$FIG1 -o $SAVE_DIR/fig1d/6 --embed_dim 60
$FIG1 -o $SAVE_DIR/fig1d/8 --embed_dim 80
$FIG1 -o $SAVE_DIR/fig1d/10 --embed_dim 100

##fig 2
$EXP -o $SAVE_DIR/fig2 --embed_dim 10 --phylogeny_shape=tree group_size=128  --min_mutation_rate=.1 --max_mutation_rate=.3
