#include "sequence/fasta_io.hpp"
#include "sequence/sequence_generator.hpp"
#include "sketch/hash_min.hpp"
#include "sketch/hash_ordered.hpp"
#include "sketch/hash_weighted.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"
#include "util/multivec.hpp"
#include "util/spearman.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include <filesystem>
#include <fstream>
#include <memory>


DEFINE_uint32(kmer_size, 3, "Kmer size for MH, OMH, WMH");
DEFINE_uint32(k, 3, "Short hand for --kmer_size");

DEFINE_string(alphabet,
              "dna4",
              "The alphabet over which sequences are defined (dna4, dna5, protein)");

DEFINE_bool(fix_len, false, "Force generated sequences length to be equal");

DEFINE_int32(max_num_blocks, 4, "Maximum number of blocks for block permutation");

DEFINE_int32(min_num_blocks, 2, "Minimum number of blocks for block permutation");

DEFINE_uint32(num_seqs, 50, "Number of sequences to be generated");

DEFINE_uint32(seq_len, 1024, "The length of sequence to be generated");

DEFINE_double(mutation_rate, 0.015, "Rate of point mutation rate for sequence generation");

DEFINE_double(block_mutation_rate, 0.02, "The probability of having a block permutation");

DEFINE_uint32(sequence_seeds, 1, "Number of initial random sequences");

DEFINE_string(o, "/tmp/out", "Directory where the generated sequence should be written");

DEFINE_int32(embed_dim, 16, "Embedding dimension, used for all sketching methods");

DEFINE_int32(tuple_length,
             3,
             "Ordered tuple length, used in ordered MinHash and Tensor-based sketches");
DEFINE_int32(t, 3, "Short hand for --tuple_length");

DEFINE_int32(window_size, 32, "Window length: the size of sliding window in Tensor Slide Sketch");
DEFINE_int32(w, 32, "Short hand for --window_size");

DEFINE_int32(
        max_len,
        36,
        "The maximum accepted sequence length for Ordered and Weighted min-hash. Must be larger "
        "than seq_len + delta, where delta is the number of random insertions");

DEFINE_int32(stride, 16, "Stride for sliding window: shift step for sliding window");
DEFINE_int32(s, 16, "Short hand for --stride");

static bool ValidateMutationPattern(const char *flagname, const std::string &value) {
    if (value == "linear" || value == "tree" || value == "uniform")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(mutation_pattern,
              "uniform",
              "the mutational pattern, can be 'linear', 'uniform' or 'tree'");
DEFINE_validator(mutation_pattern, &ValidateMutationPattern);

void adjust_short_names() {
    if (!gflags::GetCommandLineFlagInfoOrDie("K").is_default) {
        FLAGS_kmer_size = FLAGS_k;
    }

    if (!gflags::GetCommandLineFlagInfoOrDie("T").is_default) {
        FLAGS_tuple_length = FLAGS_t;
    }

    if (!gflags::GetCommandLineFlagInfoOrDie("W").is_default) {
        FLAGS_window_size = FLAGS_w;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("S").is_default) {
        FLAGS_stride = FLAGS_w;
    }
}

namespace fs = std::filesystem;
using namespace ts;

template <class char_type, class kmer_type, class embed_type>
struct SeqGenModule {
    Vec2D<char_type> seqs;
    std::vector<std::string> seq_names;
    std::string test_id;
    Vec2D<kmer_type> kmer_seqs;
    Vec2D<kmer_type> mh_sketch;
    Vec2D<kmer_type> wmh_sketch;
    Vec2D<kmer_type> omh_sketch;
    Vec2D<embed_type> ten_sketch;
    Vec3D<embed_type> slide_sketch;
    Vec3D<double> dists;

    std::filesystem::path output_dir;

    SeqGenModule(const std::string &out_dir) : output_dir(out_dir) {}

    void generate_sequences() {
        ts::SeqGen seq_gen(alphabet_size, FLAGS_fix_len, FLAGS_max_num_blocks, FLAGS_min_num_blocks,
                           FLAGS_num_seqs, FLAGS_seq_len, (float)FLAGS_mutation_rate,
                           (float)FLAGS_block_mutation_rate);

        if (FLAGS_mutation_pattern == "uniform") {
            seqs = seq_gen.genseqs_uniform<char_type>();
        } else if (FLAGS_mutation_pattern == "linear") {
            seqs = seq_gen.genseqs_linear<char_type>();
        } else if (FLAGS_mutation_pattern == "tree") {
            seqs = seq_gen.genseqs_tree<char_type>(FLAGS_sequence_seeds);
        }
    }

    void compute_sketches() {
        embed_type set_size = int_pow<size_t>(alphabet_size, FLAGS_kmer_size);
        MinHash<kmer_type> min_hash(set_size, FLAGS_embed_dim);
        WeightedMinHash<kmer_type> wmin_hash(set_size, FLAGS_embed_dim, FLAGS_max_len);
        OrderedMinHash<kmer_type> omin_hash(set_size, FLAGS_embed_dim, FLAGS_max_len,
                                            FLAGS_tuple_length);
        Tensor<char_type> tensor_sketch(alphabet_size, FLAGS_embed_dim, FLAGS_tuple_length);
        // embed_type slide_sketch_dim = FLAGS_embed_dim / FLAGS_stride + 1;
        TensorSlide<char_type> tensor_slide(alphabet_size, FLAGS_embed_dim, FLAGS_tuple_length,
                                            FLAGS_window_size, FLAGS_stride);

        size_t num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ten_sketch.resize(num_seqs);
        slide_sketch.resize(num_seqs);
        for (size_t si = 0; si < num_seqs; si++) {
            kmer_seqs[si]
                    = seq2kmer<char_type, kmer_type>(seqs[si], FLAGS_kmer_size, alphabet_size);
            mh_sketch[si] = min_hash.compute(kmer_seqs[si]);
            wmh_sketch[si] = wmin_hash.compute(kmer_seqs[si]);
            omh_sketch[si] = omin_hash.compute_flat(kmer_seqs[si]);
            ten_sketch[si] = tensor_sketch.compute(seqs[si]);
            slide_sketch[si] = tensor_slide.compute(seqs[si]);
            std::cout << "." << std::flush;
        }
        std::cout << std::endl;
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
                std::cout << "." << std::flush;
            }
        }
        std::cout << std::endl;
    }

    void print_spearman(const std::string &output) {
        std::vector<double> dists_ed;
        std::vector<double> dists_mh;
        std::vector<double> dists_wmh;
        std::vector<double> dists_omh;
        std::vector<double> dists_tensor_sketch;
        std::vector<double> dists_tensor_slide_sketch;
        for (size_t i = 0; i < seqs.size(); i++) {
            dists_ed.insert(dists_ed.end(), dists[0][i].begin(), dists[0][i].end());
            dists_mh.insert(dists_mh.end(), dists[1][i].begin(), dists[1][i].end());
            dists_wmh.insert(dists_wmh.end(), dists[2][i].begin(), dists[2][i].end());
            dists_omh.insert(dists_omh.end(), dists[3][i].begin(), dists[3][i].end());
            dists_tensor_sketch.insert(dists_tensor_sketch.end(), dists[4][i].begin(),
                                       dists[4][i].end());
            dists_tensor_slide_sketch.insert(dists_tensor_slide_sketch.end(), dists[5][i].begin(),
                                             dists[5][i].end());
        }
        std::ofstream f(output, std::ios::app);
        f << FLAGS_mutation_rate << "\t" << FLAGS_block_mutation_rate << "\t" << FLAGS_kmer_size << "\t";
        f << spearman(dists_ed, dists_mh) << "\t" << spearman(dists_ed, dists_wmh) << "\t"
          << spearman(dists_ed, dists_omh) << "\t" << spearman(dists_ed, dists_tensor_sketch)
          << "\t" << spearman(dists_ed, dists_tensor_slide_sketch) << std::endl;
    }
};


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    init_alphabet(FLAGS_alphabet);

    std::filesystem::remove(FLAGS_o);

    std::ofstream f(FLAGS_o);
    f << "M\tBM\tK\tMH\tWMH\tOMH\tTS\tTSS" << std::endl;
    f.close();


    for (double mutation_prob : { 0.0, 0.02, 0.1 }) {
        FLAGS_mutation_rate = mutation_prob;
        for (double block_mutation_prob : { 0.0, 0.02, 0.1 }) {
            if (mutation_prob == 0 && block_mutation_prob == 0) {
                continue;
            }
            FLAGS_block_mutation_rate = block_mutation_prob;
            for (uint32_t kmer_length : { 4, 8, 16 }) {
                std::cout << "Mutation prob: " << mutation_prob
                          << " Block mutation prob: " << block_mutation_prob
                          << " Kmer size: " << kmer_length << std::endl;
                for (uint32_t repeat = 0; repeat < 3; ++ repeat) {
                    FLAGS_kmer_size = kmer_length;
                    FLAGS_tuple_length = kmer_length;

                    SeqGenModule<uint8_t, uint64_t, double> experiment(FLAGS_o);
                    std::cout << "Generating sequences..." << std::endl;
                    experiment.generate_sequences();
                    std::cout << "Computing sketches";
                    experiment.compute_sketches();
                    std::cout << "Computing distances";
                    experiment.compute_pairwise_dists();
                    experiment.print_spearman(FLAGS_o);
                }
            }
        }
    }
    return 0;
}