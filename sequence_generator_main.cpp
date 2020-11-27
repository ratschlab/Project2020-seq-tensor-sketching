#include "sequence/sequence_generator.hpp"
#include "sequence/fasta_io.hpp"
#include "util/utils.hpp"

#include <gflags/gflags.h>

#include <memory>

DEFINE_int32(alphabet_size, 4, "Size of the alphabet for generated sequences");
DEFINE_int32(A, 4, "Short hand for --alphabet_size");

DEFINE_bool(fix_len, false, "Force generated sequence length to be equal");
DEFINE_bool(F, false, "Short hand for --fix_len");

DEFINE_int32(max_num_blocks, 4, "Maximum number of blocks for block permutation");
DEFINE_int32(B, 4, "Short hand for --max_num_blocks");

DEFINE_int32(min_num_blocks, 2, "Minimum number of blocks for block permutation");
DEFINE_int32(b, 4, "Short hand for --min_num_blocks");

DEFINE_uint32(num_seqs, 200, "Number of sequences to be generated");
DEFINE_uint32(N, 200, "Short hand for --num_seqs");

DEFINE_uint32(seq_len, 256, "The length of sequence to be generated");
DEFINE_uint32(L, 256, "Short hand for --seq_len");

DEFINE_double(mutation_rate, 0.015, "Rate of point mutation rate for sequence generation");
DEFINE_double(r, 0.015, "Short hand for --mutation_rate");

DEFINE_double(block_mutation_rate, 0.02, "The probability of having a block permutation");
DEFINE_double(R, 0.02, "Short hand for --block_mutation_rate");

DEFINE_uint32(sequence_seeds, 1, "Number of initial random sequences");
DEFINE_uint32(s, 1, "Short hand for --sequence_seeds");

DEFINE_string(output, "./seqs.fa", "File name where the generated sequence should be written");
DEFINE_string(o, "./seqs.fa", "Short hand for --output");

static bool ValidateMutationPattern(const char *flagname, const std::string &value) {
    if (value == "linear" || value == "tree" || value == "pairs")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(mutation_pattern, "linear", "the mutational pattern, can be 'linear', or 'tree'");
DEFINE_validator(mutation_pattern, &ValidateMutationPattern);


void adjust_short_names() {
    if (!gflags::GetCommandLineFlagInfoOrDie("A").is_default) {
        FLAGS_alphabet_size = FLAGS_A;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("B").is_default) {
        FLAGS_max_num_blocks = FLAGS_B;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("b").is_default) {
        FLAGS_min_num_blocks = FLAGS_b;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("N").is_default) {
        FLAGS_num_seqs = FLAGS_N;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("L").is_default) {
        FLAGS_seq_len = FLAGS_L;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("r").is_default) {
        FLAGS_mutation_rate = FLAGS_r;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("R").is_default) {
        FLAGS_block_mutation_rate = FLAGS_R;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("s").is_default) {
        FLAGS_sequence_seeds = FLAGS_s;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("o").is_default) {
        FLAGS_output = FLAGS_o;
    }
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    adjust_short_names();

    ts::Vec2D<int> seqs;
    ts::Vec<std::string> seq_names;
    std::string test_id;

    ts::SeqGen seq_gen(FLAGS_alphabet_size,
                       FLAGS_fix_len,
                       FLAGS_max_num_blocks,
                       FLAGS_min_num_blocks,
                       FLAGS_num_seqs,
                       FLAGS_seq_len,
                       (float)FLAGS_mutation_rate,
                       (float)FLAGS_block_mutation_rate);

    if (FLAGS_mutation_pattern == "linear") {
        seq_gen.genseqs_linear(seqs);
    } else if (FLAGS_mutation_pattern == "tree") {
        seq_gen.genseqs_tree(seqs, FLAGS_sequence_seeds);
    } else {
        assert(false);
    }
    ts::write_fasta(FLAGS_output, seqs);
}
