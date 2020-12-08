#include "sequence/fasta_io.hpp"
#include "sequence/sequence_generator.hpp"
#include "sketch/hash_min.hpp"
#include "sketch/hash_ordered.hpp"
#include "sketch/hash_weighted.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"
#include "util/multivec.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include <filesystem>
#include <fstream>
#include <memory>


DEFINE_uint32(kmer_size, 1, "The sketching method to use: MH, WMH, OMH, TenSketch or TenSlide");
DEFINE_uint32(K, 3, "Short hand for --kmer_size");

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

DEFINE_int32(embed_dim, 128, "Embedding dimension, used for all sketching methods");
DEFINE_int32(M, 128, "Short hand for --embed_dim");

DEFINE_bool(tuple_on_kmer,
            false,
            "Apply tuple-based methods (OMH, TensorSketch, and TenSlide), on kmer sequence");
DEFINE_bool(tk, false, "Short hand for --tuple_on_kmer");

DEFINE_int32(tup_len, 2, "Ordered tuple length, used in ordered MinHash and Tensor-based sketches");
DEFINE_int32(T, 2, "Short hand for --tup_len");

DEFINE_int32(num_phases,
             2,
             "Number of phases to be used for modular arithmetic in tensor sketching");
DEFINE_int32(P, 2, "Short hand for --num_phases");

DEFINE_int32(num_bins, 255, "Number of bins for discretization after tensor sketching");
DEFINE_int32(n, 255, "Short hand for --num_bins");

DEFINE_int32(win_len, 32, "Window length: the size of sliding window in Tensor Slide Sketch");
DEFINE_int32(W, 32, "Short hand for --win_len");

DEFINE_int32(max_len, 32, "The maximum accepted sequence length for Ordered and Weighted min-hash");

DEFINE_int32(stride, 8, "Stride for sliding window: shift step for sliding window");
DEFINE_int32(S, 8, "Short hand for --stride");

DEFINE_int32(offset, 0, "Initial index to start the sliding window");
DEFINE_int32(O, 0, "Short hand for --offset");


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
    if (!gflags::GetCommandLineFlagInfoOrDie("K").is_default) {
        FLAGS_kmer_size = FLAGS_K;
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
    if (!gflags::GetCommandLineFlagInfoOrDie("F").is_default) {
        FLAGS_fix_len = FLAGS_F;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("tk").is_default) {
        FLAGS_tuple_on_kmer = FLAGS_tk;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("M").is_default) {
        FLAGS_embed_dim = FLAGS_M;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("T").is_default) {
        FLAGS_tup_len = FLAGS_T;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("P").is_default) {
        FLAGS_num_phases = FLAGS_P;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("n").is_default) {
        FLAGS_num_bins = FLAGS_n;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("o").is_default) {
        FLAGS_output = FLAGS_o;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("W").is_default) {
        FLAGS_win_len = FLAGS_W;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("S").is_default) {
        FLAGS_stride = FLAGS_S;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("O").is_default) {
        FLAGS_offset = FLAGS_O;
    }
}

namespace fs = std::filesystem;
using namespace ts;

template <class char_type, class kmer_type, class embed_type>
struct SeqGenModule {
    Vec2D<char_type> seqs;
    Vec<std::string> seq_names;
    std::string test_id;
    Vec2D<kmer_type> kmer_seqs;
    Vec2D<kmer_type> mh_sketch;
    Vec2D<kmer_type> wmh_sketch;
    Vec2D<kmer_type> omh_sketch;
    Vec2D<embed_type> ten_sketch;
    Vec3D<embed_type> slide_sketch;
    Vec3D<embed_type> dists;

    std::string output;

    void generate_sequences() {
        ts::SeqGen seq_gen(FLAGS_alphabet_size, FLAGS_fix_len, FLAGS_max_num_blocks,
                           FLAGS_min_num_blocks, FLAGS_num_seqs, FLAGS_seq_len,
                           (float)FLAGS_mutation_rate, (float)FLAGS_block_mutation_rate);

        if (FLAGS_mutation_pattern == "pairs") {
            seq_gen.genseqs_pairs(seqs);
        } else if (FLAGS_mutation_pattern == "linear") {
            seq_gen.genseqs_linear(seqs);
        } else if (FLAGS_mutation_pattern == "tree") {
            seq_gen.genseqs_tree(seqs, FLAGS_sequence_seeds);
        }
    }

    void compute_sketches() {
        embed_type set_size = int_pow<size_t>(FLAGS_alphabet_size, FLAGS_kmer_size);
        MinHash<kmer_type> min_hash(set_size, FLAGS_embed_dim);
        WeightedMinHash<kmer_type> wmin_hash(set_size, FLAGS_embed_dim, FLAGS_max_len);
        // TODO(dd) - this is fishy - there is no reason to compute omh on characters
        kmer_type omh_set_size = FLAGS_tuple_on_kmer ? set_size : FLAGS_alphabet_size;
        OrderedMinHash<kmer_type> omin_hash(omh_set_size, FLAGS_embed_dim, FLAGS_max_len,
                                            FLAGS_tup_len);
        Tensor<char_type> tensor_sketch(set_size, FLAGS_embed_dim, FLAGS_tup_len);
        embed_type slide_sketch_dim = FLAGS_embed_dim / FLAGS_stride + 1;
        TensorSlide<char_type> tensor_slide(set_size, slide_sketch_dim, FLAGS_tup_len,
                                            FLAGS_win_len, FLAGS_stride);

        size_t num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ten_sketch.resize(num_seqs);
        slide_sketch.resize(num_seqs);
        for (size_t si = 0; si < num_seqs; si++) {
            kmer_seqs[si] = seq2kmer<char_type, kmer_type>(seqs[si], FLAGS_kmer_size,
                                                           FLAGS_alphabet_size);
            mh_sketch[si] = min_hash.compute(kmer_seqs[si]);
            wmh_sketch[si] = wmin_hash.compute(kmer_seqs[si]);
            omh_sketch[si] = omin_hash.compute_flat(kmer_seqs[si]);
            ten_sketch[si] = tensor_sketch.compute(seqs[si]);
            slide_sketch[si] = tensor_slide.compute(seqs[si]);
        }
    }

    void compute_pairwise_dists() {
        int num_seqs = seqs.size();
        if (FLAGS_mutation_pattern == "pairs") {
            dists = new3D<double>(8, num_seqs, 1, -1);
            for (size_t i = 0; i < seqs.size(); i += 2) {
                int j = i + 1;
                dists[0][i][0] = edit_distance(seqs[i], seqs[j]);
                dists[1][i][0] = hamming_dist(mh_sketch[i], mh_sketch[j]);
                dists[2][i][0] = hamming_dist(wmh_sketch[i], wmh_sketch[j]);
                dists[3][i][0] = hamming_dist(omh_sketch[i], omh_sketch[j]);
                dists[4][i][0] = l1_dist(ten_sketch[i], ten_sketch[j]);
                dists[5][i][0] = l1_dist2D_minlen(slide_sketch[i], slide_sketch[j]);
            }
        } else {
            dists = new3D<double>(8, num_seqs, num_seqs, 0);
            for (size_t i = 0; i < seqs.size(); i++) {
                for (size_t j = i + 1; j < seqs.size(); j++) {
                    dists[0][i][j] = edit_distance(seqs[i], seqs[j]);
                    dists[1][i][j] = hamming_dist(mh_sketch[i], mh_sketch[j]);
                    dists[2][i][j] = hamming_dist(wmh_sketch[i], wmh_sketch[j]);
                    dists[3][i][j] = hamming_dist(omh_sketch[i], omh_sketch[j]);
                    dists[4][i][j] = l1_dist(ten_sketch[i], ten_sketch[j]);
                    dists[5][i][j] = l1_dist2D_minlen(slide_sketch[i], slide_sketch[j]);
                }
            }
        }
    }

    void save_output() {
        Vec<std::string> method_names
                = { "ED", "MH", "WMH", "OMH", "TenSketch", "TenSlide", "Ten2", "Ten2Slide" };
        std::ofstream fo;

        fs::remove_all(fs::path(output));
        fs::create_directories(fs::path(output + "/dists"));
        fs::create_directories(fs::path(output + "/sketches"));

        fo.open(output + "conf.csv");
        assert(fo.is_open());
        fo << flag_values();
        fo.close();

        fo.open(output + "timing.csv");
        assert(fo.is_open());
        fo << Timer::summary();
        fo.close();

        write_fasta(output + "seqs.fa", seqs);

        size_t num_seqs = seqs.size();
        for (int m = 0; m < 6; m++) {
            fo.open(output + "dists/" + method_names[m] + ".txt");
            assert(fo.is_open());
            if (FLAGS_mutation_pattern == "pairs") {
                for (size_t i = 0; i < num_seqs; i += 2) {
                    size_t j = i + 1;
                    fo << i << ", " << j << ", " << dists[m][i][0] << "\n";
                }
            } else {
                for (size_t i = 0; i < num_seqs; i++) {
                    for (size_t j = i + 1; j < seqs.size(); j++) {
                        fo << i << ", " << j << ", " << dists[m][i][j] << "\n";
                    }
                }
            }
            fo.close();
        }

        fo.open(output + "sketches/mh.txt");
        assert(fo.is_open());
        for (size_t si = 0; si < num_seqs; si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : mh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output + "sketches/wmh.txt");
        assert(fo.is_open());
        for (size_t si = 0; si < num_seqs; si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : wmh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output + "sketches/omh.txt");
        assert(fo.is_open());
        for (size_t si = 0; si < num_seqs; si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : omh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output + "sketches/ten.txt");
        assert(fo.is_open());
        for (size_t si = 0; si < seqs.size(); si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : ten_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output + "sketches/ten_slide.txt");
        for (size_t si = 0; si < seqs.size(); si++) {
            auto &sk = slide_sketch[si];
            for (size_t dim = 0; dim < sk.size(); dim++) {
                fo << ">> seq: " << si << ", dim: " << dim << "\n";
                for (auto &item : sk[dim])
                    fo << item << ", ";
                fo << "\n";
            }
            fo << "\n";
        }
        fo.close();
    }
};

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    SeqGenModule<uint8_t, uint64_t, double> experiment;
    experiment.generate_sequences();
    experiment.compute_sketches();
    experiment.compute_pairwise_dists();
    experiment.save_output();
    return 0;
}
