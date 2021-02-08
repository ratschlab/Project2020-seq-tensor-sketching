#include "sequence/fasta_io.hpp"
#include "sketch/edit_distance.hpp"
#include "sketch/hash_base.hpp"
#include "sketch/hash_min.hpp"
#include "sketch/hash_ordered.hpp"
#include "sketch/hash_weighted.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_block.hpp"
#include "sketch/tensor_embedding.hpp"
#include "sketch/tensor_slide.hpp"
#include "util/multivec.hpp"
#include "util/progress.hpp"
#include "util/utils.hpp"

#include <gflags/gflags.h>

#include <filesystem>
#include <memory>
#include <random>
#include <sstream>
#include <utility>

using namespace ts;

// The main command this program should perform.
// Triangle: compute a triangular distance matrix.
// More actions will be added.
DEFINE_string(action, "", "Which action to do. One of: triangle, none");

DEFINE_string(alphabet,
              "dna4",
              "The alphabet over which sequences are defined (dna4, dna5, protein)");

DEFINE_string(sketch_method,
              "TensorSlide",
              "The sketching method to use: MH, WMH, OMH, TS, TSB or TSS");
DEFINE_string(m, "TSS", "Short hand for --sketch_method");

DEFINE_uint32(kmer_length, 1, "The kmer length for: MH, WMH, OMH");
DEFINE_uint32(k, 3, "Short hand for --kmer_size");

DEFINE_string(o, "", "Output file, containing the sketches for each sequence");

DEFINE_string(i,
              "",
              "Input file or directory, containing the sequences to be sketched in .fa format");

DEFINE_string(input_format, "fasta", "Input format: 'fasta', 'csv'");
DEFINE_string(f, "fasta", "Short hand for --input_format");

DEFINE_int32(embed_dim, 4, "Embedding dimension, used for all sketching methods");

DEFINE_int32(tuple_length,
             3,
             "Ordered tuple length, used in ordered MinHash and Tensor-based sketches");
DEFINE_int32(t, 3, "Short hand for --tuple_length");

static bool ValidateBlockSize(const char *flagname, int32_t value) {
    if (FLAGS_tuple_length % value == 0 || FLAGS_t % value == 0) {
        return true;
    }
    printf("Invalid value for --%s: %d. Must be a divisor of --tuple_len\n", flagname, value);
    return false;
}
DEFINE_int32(block_size,
             1,
             "Only consider tuples made out of block-size continuous characters for Tensor sketch");
DEFINE_validator(block_size, &ValidateBlockSize);

DEFINE_int32(window_size, 32, "Window length: the size of sliding window in Tensor Slide Sketch");
DEFINE_int32(w, 32, "Short hand for --window_size");

DEFINE_int32(max_len, 32, "The maximum accepted sequence length for Ordered and Weighted min-hash");

DEFINE_int32(stride, 8, "Stride for sliding window: shift step for sliding window");
DEFINE_int32(s, 8, "Short hand for --stride");

DEFINE_uint32(num_threads, 0, "number of OpenMP threads, default: use all available cores");

static bool ValidateInput(const char * /*unused*/, const std::string &value) {
    if (!value.empty()) {
        return true;
    }
    std::cerr << "Please specify a fasta input file using '-i <input_file>'" << std::endl;
    return false;
}
DEFINE_validator(i, &ValidateInput);

static bool ValidateOutput(const char * /*unused*/, const std::string &value) {
    if (value.empty()) {
        FLAGS_o = FLAGS_i + "." + FLAGS_sketch_method;
    }
    return true;
}
DEFINE_validator(o, &ValidateOutput);

void adjust_short_names() {
    if (!gflags::GetCommandLineFlagInfoOrDie("m").is_default) {
        FLAGS_sketch_method = FLAGS_m;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("k").is_default) {
        FLAGS_kmer_length = FLAGS_k;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("f").is_default) {
        FLAGS_input_format = FLAGS_f;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("w").is_default) {
        FLAGS_window_size = FLAGS_w;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("s").is_default) {
        FLAGS_stride = FLAGS_s;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("t").is_default) {
        FLAGS_tuple_length = FLAGS_t;
    }
}

template <typename seq_type, class kmer_type, class embed_type>
class SketchHelper {
  public:
    SketchHelper(std::function<std::vector<embed_type>(const std::vector<kmer_type> &)> sketcher,
                 std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher)
        : sketcher(std::move(sketcher)), slide_sketcher(std::move(slide_sketcher)) {}

    void compute_sketches() {
        size_t num_seqs = seqs.size();
        sketches = new3D<embed_type>(seqs.size(), FLAGS_embed_dim, 0);
        for (size_t si = 0; si < num_seqs; si++) {
            std::vector<kmer_type> kmers
                    = seq2kmer<seq_type, kmer_type>(seqs[si], FLAGS_kmer_length, alphabet_size);

            for (size_t i = 0; i < kmers.size(); i += FLAGS_stride) {
                auto end = std::min(kmers.begin() + i + FLAGS_window_size, kmers.end());
                std::vector<kmer_type> kmer_slice(kmers.begin() + i, end);
                std::vector<embed_type> embed_slice = sketcher(kmer_slice);
                for (int m = 0; m < FLAGS_embed_dim; m++) {
                    sketches[si][m].push_back(embed_slice[m]);
                }
            }
        }
    }

    void compute_slide() {
        sketches = new3D<double>(seqs.size(), FLAGS_embed_dim, 0);

        for (size_t si = 0; si < seqs.size(); si++) {
            std::vector<uint64_t> kmers
                    = seq2kmer<uint8_t, uint64_t>(seqs[si], FLAGS_kmer_length, alphabet_size);
            sketches[si] = slide_sketcher(kmers);
        }
    }

    void read_input() {
        FastaFile<seq_type> file = read_fasta<seq_type>(FLAGS_i, FLAGS_input_format);
        seqs = std::move(file.sequences);
        seq_names = std::move(file.comments);
    }

    void save_output() {
        std::filesystem::path ofile = std::filesystem::absolute(std::filesystem::path(FLAGS_o));
        std::filesystem::path opath = ofile.parent_path();
        if (!std::filesystem::exists(opath) && !std::filesystem::create_directories(opath)) {
            std::cerr << "Could not create output directory: " << opath << std::endl;
            std::exit(1);
        }

        std::ofstream fo(FLAGS_o);
        if (!fo.is_open()) {
            std::cerr << "Could not open " << ofile << " for writing." << std::endl;
            std::exit(1);
        }
        std::cout << "Writing sketches to: " << FLAGS_o << std::endl;

        for (size_t si = 0; si < seqs.size(); si++) {
            fo << seq_names[si] << std::endl;
            for (size_t m = 0; m < sketches[si].size(); m++) {
                for (size_t i = 0; i < sketches[si][m].size(); i++) {
                    fo << sketches[si][m][i] << ",";
                }
            }
            fo << '\b' << std::endl;
        }
        fo.close();
    }

  private:
    Vec2D<seq_type> seqs;
    std::vector<std::string> seq_names;
    Vec3D<embed_type> sketches;

    std::function<std::vector<embed_type>(const std::vector<kmer_type> &)> sketcher;
    std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher;
};

// Some global constant types.
using seq_type = uint8_t;

// Run the given sketch method on input specified by the command line arguments, and write a
// triangular distance matrix to the output file.
template <class SketchAlgorithm>
void run_triangle(const std::vector<FastaFile<seq_type>> &files, SketchAlgorithm &algorithm) {
    const size_t n = files.size();

    std::vector<typename SketchAlgorithm::sketch_type> sketches(n);

    std::cerr << "Sketching .." << std::endl;
    progress_bar::init(n);
#pragma omp parallel for default(shared) num_threads(FLAGS_num_threads)
    for (size_t i = 0; i < n; ++i) {
        assert(files[i].sequences.size() == 1
               && "Each input file must contain exactly one sequence!");
        sketches[i] = algorithm.compute(files[i].sequences[0]);
        progress_bar::iter();
    }

    std::cerr << "Computing all pairwise distances .." << std::endl;

    std::vector<std::pair<int, int>> pairs;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < i; ++j)
            pairs.emplace_back(i, j);

    std::vector<std::vector<double>> distances(n);
    for (size_t i = 0; i < n; ++i)
        distances[i].resize(i);

    progress_bar::init(n * (n - 1) / 2);
#pragma omp parallel for default(shared) num_threads(FLAGS_num_threads)
    for (auto it = pairs.begin(); it < pairs.end(); ++it) { // NOLINT
        auto [i, j] = *it;
        distances[i][j] = algorithm.dist(sketches[i], sketches[j]);
        progress_bar::iter();
    }


    std::string suffix
            = "_" + std::to_string(FLAGS_tuple_length) + "_" + std::to_string(FLAGS_block_size);
    std::filesystem::path ofile
            = std::filesystem::absolute(std::filesystem::path(FLAGS_o + suffix));
    std::cerr << "Writing distances triangle to " << ofile << " .." << std::endl;


    write_output_meta();
    std::ofstream fo(ofile);
    if (!fo.is_open()) {
        std::cerr << "Could not open " << FLAGS_o << " for writing." << std::endl;
        std::exit(1);
    }

    for (size_t i = 0; i < n; ++i) {
        fo << files[i].filename;
        for (size_t j = 0; j < i; ++j) {
            fo << '\t' << distances[i][j];
        }
        fo << '\n';
    }
    fo.close();
};

// Runs function f on the sketch method specified by the command line options.
template <typename F>
void run_function_on_algorithm(F f) {
    using kmer_type = uint64_t;

    auto kmer_word_size = int_pow<kmer_type>(alphabet_size, FLAGS_kmer_length);

    std::random_device rd;
    if (FLAGS_sketch_method == "ED") {
        f(EditDistance<seq_type>());
        return;
    }
    if (FLAGS_sketch_method == "TE") {
        f(TensorEmbedding<seq_type>(kmer_word_size, FLAGS_tuple_length));
        return;
    }
    if (FLAGS_sketch_method == "TS") {
        f(Tensor<seq_type>(kmer_word_size, FLAGS_embed_dim, FLAGS_tuple_length, rd()));
        return;
    }
    if (FLAGS_sketch_method == "TSB") {
        f(TensorBlock<seq_type>(kmer_word_size, FLAGS_embed_dim, FLAGS_tuple_length,
                                FLAGS_block_size, rd()));
        return;
    }
    if (FLAGS_sketch_method == "TSS") {
        f(TensorSlide<seq_type>(kmer_word_size, FLAGS_embed_dim, FLAGS_tuple_length,
                                FLAGS_window_size, FLAGS_stride, rd()));
        return;
    }
}


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    adjust_short_names();

    init_alphabet(FLAGS_alphabet);

    if (std::pow(alphabet_size, FLAGS_kmer_length) > (double)std::numeric_limits<uint64_t>::max()) {
        std::cerr << "Kmer size is too large to fit in 64 bits " << std::endl;
        std::exit(1);
    }

    auto kmer_word_size = int_pow<uint64_t>(alphabet_size, FLAGS_kmer_length);

    std::random_device rd;

    if (FLAGS_action == "triangle") {
        std::cerr << "Reading input .." << std::endl;
        std::vector<FastaFile<seq_type>> files = read_directory<seq_type>(FLAGS_i);
        std::sort(files.begin(), files.end(),
                  [](const auto &a, const auto &b) { return a.filename < b.filename; });
        std::cerr << "Read " << files.size() << " files" << std::endl;

        for (uint32_t k = 3; k < 42; ++k) {
            for (uint32_t b = 1; b <= k / 2; ++b) {
                if (k % b != 0) {
                    continue;
                }
                FLAGS_tuple_length = k;
                FLAGS_block_size = b;
                run_function_on_algorithm([&files](auto x) { run_triangle(files, x); });
            }
        }
        return 0;
    }

    if (FLAGS_sketch_method.substr(FLAGS_sketch_method.size() - 2, 2) == "MH") {
        std::function<std::vector<uint64_t>(const std::vector<uint64_t> &)> sketcher;

        if (FLAGS_sketch_method == "MH") {
            // The hash function is part of the lambda state.
            sketcher = [&,
                        min_hash
                        = MinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim, HashAlgorithm::uniform,
                                            rd())](const std::vector<uint64_t> &seq) mutable {
                return min_hash.compute(seq);
            };
        } else if (FLAGS_sketch_method == "WMH") {
            sketcher = [&,
                        wmin_hash
                        = WeightedMinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim, FLAGS_max_len,
                                                    HashAlgorithm::uniform, rd())](
                               const std::vector<uint64_t> &seq) mutable {
                return wmin_hash.compute(seq);
            };
        } else if (FLAGS_sketch_method == "OMH") {
            sketcher = [&,
                        omin_hash = OrderedMinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim,
                                                             FLAGS_max_len, FLAGS_tuple_length,
                                                             HashAlgorithm::uniform, rd())](
                               const std::vector<uint64_t> &seq) mutable {
                return omin_hash.compute(seq);
            };
        }
        std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher
                = [&](const std::vector<uint64_t> & /*unused*/) { return new2D<double>(0, 0); };
        SketchHelper<uint8_t, uint64_t, uint64_t> sketch_helper(sketcher, slide_sketcher);
        sketch_helper.read_input();
        sketch_helper.compute_sketches();
        sketch_helper.save_output();
    } else if (FLAGS_sketch_method.rfind("TS", 0) == 0) {
        Tensor<uint64_t> tensor_sketch(kmer_word_size, FLAGS_embed_dim, FLAGS_tuple_length, rd());
        TensorBlock<uint64_t> tensor_block(kmer_word_size, FLAGS_embed_dim, FLAGS_tuple_length,
                                           FLAGS_block_size, rd());
        TensorSlide<uint64_t> tensor_slide(kmer_word_size, FLAGS_embed_dim, FLAGS_tuple_length,
                                           FLAGS_window_size, FLAGS_stride, rd());
        std::function<std::vector<double>(const std::vector<uint64_t> &)> sketcher
                = [&](const std::vector<uint64_t> &seq) {
                      return FLAGS_block_size == 1 ? tensor_sketch.compute(seq)
                                                   : tensor_block.compute(seq);
                  };
        std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher
                = [&](const std::vector<uint64_t> &seq) { return tensor_slide.compute(seq); };
        SketchHelper<uint8_t, uint64_t, double> sketch_helper(sketcher, slide_sketcher);
        sketch_helper.read_input();
        FLAGS_sketch_method == "TSS" ? sketch_helper.compute_slide()
                                     : sketch_helper.compute_sketches();
        sketch_helper.save_output();
    } else {
        std::cerr << "Unkknown method: " << FLAGS_sketch_method << std::endl;
        exit(1);
    }
}
