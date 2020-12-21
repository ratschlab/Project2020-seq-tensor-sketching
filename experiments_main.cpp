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
#include "util/progress.hpp"

#include <filesystem>
#include <fstream>
#include <memory>


DEFINE_uint32(kmer_size, 3, "Kmer size for MH, OMH, WMH");
DEFINE_uint32(k, 3, "Short hand for --kmer_size");

//DEFINE_string(alphabet,
//              "dna4",
//              "The alphabet over which sequences are defined (dna4, dna5, protein)");
DEFINE_int32(alphabet_size, 4, "size of alphabet for synthetic sequence generation");

DEFINE_bool(fix_len, false, "Force generated sequences length to be equal");

DEFINE_int32(max_num_blocks, 4, "Maximum number of blocks for block permutation");

DEFINE_int32(min_num_blocks, 2, "Minimum number of blocks for block permutation");

DEFINE_uint32(num_seqs, 200, "Number of sequences to be generated");

DEFINE_uint32(seq_len, 256, "The length of sequence to be generated");

DEFINE_double(mutation_rate, 0.015, "Rate of point mutation rate for sequence generation");

DEFINE_double(block_mutation_rate, 0.02, "The probability of having a block permutation");

DEFINE_uint32(sequence_seeds, 1, "Number of initial random sequences");

DEFINE_string(o, "/tmp", "Directory where the generated sequence should be written");

DEFINE_int32(embed_dim, 16, "Embedding dimension, used for all sketching methods");

DEFINE_int32(num_bins, -1, "Number of bins used to discretize the sketch output"
             ", use num_bins=-1 to use the raw sketch without binning");

DEFINE_int32(tuple_length,
             3,
             "Ordered tuple length, used in ordered MinHash and Tensor-based sketches");
DEFINE_int32(t, 3, "Short hand for --tuple_length");

DEFINE_int32(window_size, 32, "Window length: the size of sliding window in Tensor Slide Sketch");
DEFINE_int32(w, 32, "Short hand for --window_size");

DEFINE_int32(
        max_len,
        -1,
        "The maximum accepted sequence length for Ordered and Weighted min-hash. Must be larger "
        "than seq_len + delta, where delta is the number of random insertions, if max_len=-1, "
        "its value will be set to seq_len (default=-1)");

DEFINE_int32(stride, 8, "Stride for sliding window: shift step for sliding window");
DEFINE_int32(s, 8, "Short hand for --stride");

static bool ValidateMutationPattern(const char *flagname, const std::string &value) {
    if (value == "linear" || value == "tree" || value == "uniform" || value == "pairs")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(mutation_pattern,
              "linear",
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
        ts::SeqGen seq_gen(FLAGS_alphabet_size, FLAGS_fix_len, FLAGS_max_num_blocks, FLAGS_min_num_blocks,
                           FLAGS_num_seqs, FLAGS_seq_len, (float)FLAGS_mutation_rate,
                           (float)FLAGS_block_mutation_rate);

        if (FLAGS_mutation_pattern == "uniform") {
            seqs = seq_gen.genseqs_uniform<char_type>();
        } else if (FLAGS_mutation_pattern == "pairs") {
            seqs = seq_gen.genseqs_independent_pairs<char_type>();
        } else if (FLAGS_mutation_pattern == "linear") {
            seqs = seq_gen.genseqs_linear<char_type>();
        } else if (FLAGS_mutation_pattern == "tree") {
            seqs = seq_gen.genseqs_tree<char_type>(FLAGS_sequence_seeds);
        }
    }

    void compute_sketches() {
        embed_type set_size = int_pow<size_t>(FLAGS_alphabet_size, FLAGS_kmer_size);
        MinHash<kmer_type> min_hash(set_size, FLAGS_embed_dim);
        WeightedMinHash<kmer_type> wmin_hash(set_size, FLAGS_embed_dim, FLAGS_max_len);
        OrderedMinHash<kmer_type> omin_hash(set_size, FLAGS_embed_dim, FLAGS_max_len,
                                            FLAGS_tuple_length);
        Tensor<char_type> tensor_sketch(FLAGS_alphabet_size, FLAGS_embed_dim, FLAGS_tuple_length);
        // embed_type slide_sketch_dim = FLAGS_embed_dim / FLAGS_stride + 1;
        TensorSlide<char_type> tensor_slide(FLAGS_alphabet_size, FLAGS_embed_dim, FLAGS_tuple_length,
                                            FLAGS_window_size, FLAGS_stride, FLAGS_seq_len);

        size_t num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ten_sketch.resize(num_seqs);
        slide_sketch.resize(num_seqs);
        start_progress_bar(seqs.size());
//#pragma omp parallel for default(shared) private(min_hash, wmin_hash, omin_hash)
        for (size_t si = 0; si < num_seqs; si++) {
            kmer_seqs[si]
                    = seq2kmer<char_type, kmer_type>(seqs[si], FLAGS_kmer_size, FLAGS_alphabet_size);
            mh_sketch[si] = min_hash.compute(kmer_seqs[si]);
            wmh_sketch[si] = wmin_hash.compute(kmer_seqs[si]);
            omh_sketch[si] = omin_hash.compute_flat(kmer_seqs[si]);
            ten_sketch[si] = tensor_sketch.compute(seqs[si]);
            slide_sketch[si] = tensor_slide.compute(seqs[si]);
            iterate_progress_bar();
        }
        std::cout << std::endl;
    }

    void compute_pairwise_dists() {
        int num_seqs = seqs.size();
        if (FLAGS_mutation_pattern == "pairs") {
            dists = new3D<double>(8, num_seqs, 1, -1);
            start_progress_bar(seqs.size()/2);
#pragma omp parallel for default(shared) schedule(dynamic)
            for (size_t i = 0; i < seqs.size(); i += 2) {
                int j = i + 1;
                dists[0][i][0] = edit_distance(seqs[i], seqs[j]);
                dists[1][i][0] = hamming_dist(mh_sketch[i], mh_sketch[j]);
                dists[2][i][0] = hamming_dist(wmh_sketch[i], wmh_sketch[j]);
                dists[3][i][0] = hamming_dist(omh_sketch[i], omh_sketch[j]);
                dists[4][i][0] = l1_dist(ten_sketch[i], ten_sketch[j]);
                dists[5][i][0] = l1_dist2D_minlen(slide_sketch[i], slide_sketch[j]);
                iterate_progress_bar();
            }
        } else {
            dists = new3D<double>(8, num_seqs, num_seqs, 0);
            start_progress_bar(seqs.size());
#pragma omp parallel for default(shared) schedule(dynamic)
            for (size_t i = 0; i < seqs.size(); i++) {
                for (size_t j = i + 1; j < seqs.size(); j++) {
                    dists[0][i][j] = edit_distance(seqs[i], seqs[j]);
                    dists[1][i][j] = hamming_dist(mh_sketch[i], mh_sketch[j]);
                    dists[2][i][j] = hamming_dist(wmh_sketch[i], wmh_sketch[j]);
                    dists[3][i][j] = hamming_dist(omh_sketch[i], omh_sketch[j]);
                    dists[4][i][j] = l1_dist(ten_sketch[i], ten_sketch[j]);
                    dists[5][i][j] = l1_dist2D_minlen(slide_sketch[i], slide_sketch[j]);
                }
                iterate_progress_bar();
            }
        }
        std::cout << std::endl;
    }

    void print_spearman() {
        std::vector<double> dists_ed;
        std::vector<double> dists_mh;
        std::vector<double> dists_wmh;
        std::vector<double> dists_omh;
        std::vector<double> dists_tensor_sketch;
        std::vector<double> dists_tensor_slide_sketch;
        if (FLAGS_mutation_pattern != "pairs") {
            for (size_t i = 0; i < seqs.size(); i++) {
                dists_ed.insert(dists_ed.end(), dists[0][i].begin()+i+1, dists[0][i].end());
                dists_mh.insert(dists_mh.end(), dists[1][i].begin()+i+1, dists[1][i].end());
                dists_wmh.insert(dists_wmh.end(), dists[2][i].begin()+i+1, dists[2][i].end());
                dists_omh.insert(dists_omh.end(), dists[3][i].begin()+i+1, dists[3][i].end());
                dists_tensor_sketch.insert(dists_tensor_sketch.end(), dists[4][i].begin()+i+1,
                                           dists[4][i].end());
                dists_tensor_slide_sketch.insert(dists_tensor_slide_sketch.end(),
                                                 dists[5][i].begin()+i+1, dists[5][i].end());
            }
        } else {
            for (size_t i = 0; i < seqs.size(); i+=2) {
                dists_ed.push_back(dists[0][i][0]);
                dists_mh.push_back(dists[1][i][0]);
                dists_wmh.push_back(dists[2][i][0]);
                dists_omh.push_back(dists[3][i][0]);
                dists_tensor_sketch.push_back(dists[4][i][0]);
                dists_tensor_slide_sketch.push_back(dists[5][i][0]);
            }
        }
        std::cout << "Spearman correlation MH: " << spearman(dists_ed, dists_mh) << std::endl;
        std::cout << "Spearman correlation WMH: " << spearman(dists_ed, dists_wmh) << std::endl;
        std::cout << "Spearman correlation OMH: " << spearman(dists_ed, dists_omh) << std::endl;
        std::cout << "Spearman correlation TensorSketch: "
                  << spearman(dists_ed, dists_tensor_sketch) << std::endl;
        std::cout << "Spearman correlation TensorSlide: "
                  << spearman(dists_ed, dists_tensor_slide_sketch) << std::endl;
    }

    void save_output() {
        std::vector<std::string> method_names
                = { "ED", "MH", "WMH", "OMH", "TenSketch", "TenSlide", "Ten2", "Ten2Slide" };
        std::ofstream fo;

        fs::create_directories(fs::path(output_dir / "dists"));
        fs::create_directories(fs::path(output_dir / "sketches"));

        fo.open(output_dir / "conf.csv");
        assert(fo.is_open());
        fo << flag_values();
        fo.close();

        fo.open(output_dir / "timing.csv");
        assert(fo.is_open());
        fo << Timer::summary(FLAGS_num_seqs);
        fo.close();

        write_fasta(output_dir / "seqs.fa", seqs);

        size_t num_seqs = seqs.size();
        for (int m = 0; m < 6; m++) {
            fo.open(output_dir / "dists" / (method_names[m] + ".txt"));
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

        fo.open(output_dir / "sketches/mh.txt");
        assert(fo.is_open());
        for (size_t si = 0; si < num_seqs; si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : mh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output_dir / "sketches/wmh.txt");
        assert(fo.is_open());
        for (size_t si = 0; si < num_seqs; si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : wmh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output_dir / "sketches/omh.txt");
        assert(fo.is_open());
        for (size_t si = 0; si < num_seqs; si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : omh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output_dir / "sketches/ten.txt");
        assert(fo.is_open());
        for (size_t si = 0; si < seqs.size(); si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : ten_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output_dir / "sketches/ten_slide.txt");
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
    if (FLAGS_max_len<0) {
        FLAGS_max_len = FLAGS_seq_len;
    }

    SeqGenModule<uint8_t, uint64_t, double> experiment(FLAGS_o);
    std::cout << "Generating sequences ..." << std::endl;
    experiment.generate_sequences();
    std::cout << "Computing sketches ... " << std::endl;
    experiment.compute_sketches();
    std::cout << "Computing distances ... " << std::endl;
    experiment.compute_pairwise_dists();
    std::cout << "Writing output to " << FLAGS_o << std::endl;
    experiment.save_output();
    experiment.print_spearman();
    return 0;
}
