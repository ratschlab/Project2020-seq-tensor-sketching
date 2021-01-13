#include "sequence/sequence_generator.hpp"
#include "sequence/fasta_io.hpp"
#include "util/utils.hpp"

#include <gflags/gflags.h>

#include <filesystem>
#include <memory>

DEFINE_int32(alphabet_size, 4, "Size of the alphabet for generated sequences");
DEFINE_int32(A, 4, "Short hand for --alphabet_size");

DEFINE_bool(fix_len, false, "Force generated sequence length to be equal");
DEFINE_bool(F, false, "Short hand for --fix_len");


DEFINE_uint32(num_seqs, 200, "Number of sequences to be generated");
DEFINE_uint32(N, 200, "Short hand for --num_seqs");

DEFINE_uint32(seq_len, 256, "The length of sequence to be generated");
DEFINE_uint32(L, 256, "Short hand for --seq_len");

DEFINE_uint32(group_size, 2, "The number of sequences in each group");
DEFINE_uint32(G, 2, "Short hand for --group_size");

DEFINE_double(max_mutation_rate, 0.3, "Maximum rate of point mutation for sequence generation");
DEFINE_double(R, 0.3, "Short hand for --max_mutation_rate");

DEFINE_double(min_mutation_rate, 0.0, "min rate for sequence mutation for sequence generation");
DEFINE_double(r, 0.00, "Short hand for --min_mutation_rate");


DEFINE_string(output_dir, "/tmp/", "File name where the generated sequence should be written");
DEFINE_string(o, "./seqs.fa", "Short hand for --output");


static bool validatePhylogenyShape(const char *flagname, const std::string &value) {
    if (value == "path" || value == "tree" || value == "star" || value == "pair")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(phylogeny_shape,
              "path",
              "shape of the phylogeny can be 'path', 'tree', 'star', or 'pair'");
DEFINE_validator(phylogeny_shape, &validatePhylogenyShape);



void adjust_short_names() {
    if (!gflags::GetCommandLineFlagInfoOrDie("A").is_default) {
        FLAGS_alphabet_size = FLAGS_A;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("N").is_default) {
        FLAGS_num_seqs = FLAGS_N;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("L").is_default) {
        FLAGS_seq_len = FLAGS_L;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("R").is_default) {
        FLAGS_max_mutation_rate = FLAGS_R;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("r").is_default) {
        FLAGS_min_mutation_rate = FLAGS_r;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("o").is_default) {
        FLAGS_output_dir = FLAGS_o;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("G").is_default) {
        FLAGS_group_size = FLAGS_G;
    }
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    adjust_short_names();

    ts::Vec2D<uint8_t> seqs;
    std::vector<std::string> seq_names;
    std::string test_id;

    ts::SeqGen seq_gen(FLAGS_alphabet_size, FLAGS_fix_len, FLAGS_num_seqs, FLAGS_seq_len,
                       FLAGS_group_size, FLAGS_max_mutation_rate, FLAGS_min_mutation_rate, FLAGS_phylogeny_shape);

    seqs = seq_gen.generate_seqs<uint8_t>();
    ts::write_fasta(std::filesystem::path(FLAGS_output_dir) / "seqs.fa", seqs);
}
