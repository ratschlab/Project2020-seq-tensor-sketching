#include "sequence/alphabets.hpp"
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
#include <numeric>
#include <random>
#include <sstream>
#include <utility>

using namespace ts;

DEFINE_string(alphabet,
              "dna4",
              "The alphabet over which sequences are defined (dna4, dna5, protein)");

DEFINE_string(sketch_method, "TSS", "The sketching method to use: MH, WMH, OMH, TS, TSB or TSS");
DEFINE_string(m, "TSS", "Short hand for --sketch_method");

DEFINE_uint32(kmer_length, 1, "The kmer length for: MH, WMH, OMH");
DEFINE_uint32(k, 3, "Short hand for --kmer_length");

DEFINE_string(o, "", "Output file, containing the sketches for each sequence");

DEFINE_string(i,
              "",
              "Input file or directory, containing the sequences to be sketched in .fa format");

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

// Some global constant types.
using seq_type = uint8_t;

// Run the given sketch method on input specified by the command line arguments, and write a
// triangular distance matrix to the output file.
template <class SketchAlgorithm>
void run_triangle(SketchAlgorithm &algorithm) {
    std::cerr << "Reading input .." << std::endl;
    std::vector<FastaFile<seq_type>> files = read_directory<seq_type>(FLAGS_i);
    std::cerr << "Read " << files.size() << " files" << std::endl;

    const size_t n = files.size();

    std::vector<typename SketchAlgorithm::sketch_type> sketches(n);

    std::cerr << "Sketching .." << std::endl;
    progress_bar::init(n);
#pragma omp parallel for default(shared)
    for (size_t i = 0; i < n; ++i) {
        assert(files[i].sequences.size() == 1
               && "Each input file must contain exactly one sequence!");
        if constexpr (SketchAlgorithm::kmer_input) {
            sketches[i]
                    = algorithm.compute(files[i].sequences[0], FLAGS_kmer_length, alphabet_size);
        } else {
            sketches[i] = algorithm.compute(files[i].sequences[0]);
        }
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
#pragma omp parallel for default(shared)
    for (auto it = pairs.begin(); it < pairs.end(); ++it) { // NOLINT
        auto [i, j] = *it;
        distances[i][j] = algorithm.dist(sketches[i], sketches[j]);
        progress_bar::iter();
    }

    std::cerr << "Writing distances triangle to " << FLAGS_o << " .." << std::endl;
    std::filesystem::path ofile = std::filesystem::absolute(std::filesystem::path(FLAGS_o));

    write_output_meta();
    std::ofstream fo(ofile);
    if (!fo.is_open()) {
        std::cerr << "Could not open " << FLAGS_o << " for writing." << std::endl;
        std::exit(1);
    }

    // MASH adds an extra tab before the number of lines, so mirror that.
    fo << "\t" << n << '\n';
    for (size_t i = 0; i < n; ++i) {
        fo << files[i].filename;
        for (size_t j = 0; j < i; ++j)
            fo << '\t' << distances[i][j];
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
    if (FLAGS_sketch_method == "MH") {
        f(MinHash<kmer_type>(kmer_word_size, FLAGS_embed_dim, HashAlgorithm::murmur, rd()));
        return;
    }
    if (FLAGS_sketch_method == "WMH") {
        f(WeightedMinHash<kmer_type>(kmer_word_size, FLAGS_embed_dim, FLAGS_max_len,
                                     HashAlgorithm::murmur, rd()));
        return;
    }
    if (FLAGS_sketch_method == "OMH") {
        f(OrderedMinHash<kmer_type>(kmer_word_size, FLAGS_embed_dim, FLAGS_max_len,
                                    FLAGS_tuple_length, HashAlgorithm::murmur, rd()));
        return;
    }
    if (FLAGS_sketch_method == "ED") {
        f(EditDistance<seq_type>());
        return;
    }
    if (FLAGS_sketch_method == "TE") {
        f(TensorEmbedding<seq_type>(alphabet_size, FLAGS_tuple_length, "TensorEmbedding"));
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
    std::cerr << "Unknown sketch method: " << FLAGS_sketch_method << "\n";
}


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    adjust_short_names();

    init_alphabet(FLAGS_alphabet);

    if (std::pow(alphabet_size, FLAGS_kmer_length) > (double)std::numeric_limits<uint64_t>::max()) {
        std::cerr << "Kmer size is too large to fit in 64 bits " << std::endl;
        std::exit(1);
    }

    run_function_on_algorithm([](auto x) { run_triangle(x); });
}
