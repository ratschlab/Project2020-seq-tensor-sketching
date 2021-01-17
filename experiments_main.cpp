#include "sequence/fasta_io.hpp"
#include "sequence/sequence_generator.hpp"
#include "sketch/dim_reduce.hpp"
#include "sketch/hash_base.hpp"
#include "sketch/hash_min.hpp"
#include "sketch/hash_ordered.hpp"
#include "sketch/hash_weighted.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"
#include "sketch/tensor_slide_flat.hpp"
#include "util/multivec.hpp"
#include "util/progress.hpp"
#include "util/spearman.hpp"
#include "util/timer.hpp"
#include "util/transformer.hpp"
#include "util/utils.hpp"

#include <gflags/gflags.h>

#include <array>
#include <filesystem>
#include <memory>
#include <numeric>
#include <omp.h>
#include <sys/types.h>
#include <type_traits>

DEFINE_uint32(kmer_size, 4, "Kmer size for MH, OMH, WMH");

DEFINE_int32(alphabet_size, 4, "size of alphabet for synthetic sequence generation");

DEFINE_int32(max_num_blocks, 4, "Maximum number of blocks for block permutation");

DEFINE_int32(min_num_blocks, 2, "Minimum number of blocks for block permutation");

DEFINE_uint32(num_seqs, 200, "Number of sequences to be generated");

DEFINE_uint32(seq_len, 256, "The length of sequence to be generated");

DEFINE_bool(fix_len, false, "Force generated sequences length to be equal");

DEFINE_double(max_mutation_rate, 0.5, "Maximum rate of point mutation for sequence generation");

DEFINE_double(min_mutation_rate, 0.0, "Minimum rate of point mutation for sequence generation");


DEFINE_double(block_mutation_rate, 0.00, "The probability of having a block permutation");

DEFINE_uint32(group_size, 2, "Number of sequences in each independent group");

DEFINE_string(o, "/tmp", "Directory where the generated sequence should be written");

DEFINE_int32(embed_dim, 16, "Embedding dimension, used for all sketching methods");

DEFINE_int32(tuple_length,
             3,
             "Ordered tuple length, used in ordered MinHash and Tensor-based sketches");

DEFINE_int32(window_size, 32, "Window length: the size of sliding window in Tensor Slide Sketch");

DEFINE_int32(
        max_len,
        -1,
        "The maximum accepted sequence length for Ordered and Weighted min-hash. Must be larger "
        "than seq_len + delta, where delta is the number of random insertions, if max_len=-1, "
        "its value will be set to seq_len (default=-1)");

DEFINE_int32(stride, 8, "Stride for sliding window: shift step for sliding window");

static bool validatePhylogenyShape(const char *flagname, const std::string &value) {
    if (value == "path" || value == "tree" || value == "star" || value == "pair")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(phylogeny_shape,
              "path",
              "shape of the phylogeny can be 'path', 'tree', 'star', 'pair'");
DEFINE_validator(phylogeny_shape, &validatePhylogenyShape);


static bool ValidateTransformation(const char *flagname, const std::string &value) {
    if (value == "none" || value == "atan" || value == "disc")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(transform, "none", "transform TS and TSS output, can be 'none', 'atan' or 'disc'");
DEFINE_validator(transform, &ValidateTransformation);


static bool ValidateHashAlg(const char *flagname, const std::string &value) {
    if (value == "uniform" || value == "crc32")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
// Use CRC32 only for comparing speed
// TODO: implement a proper iterative permutation generation for hash*
DEFINE_string(hash_alg,
              "uniform",
              "hash algorithm to be used as basis, can be 'uniform', or 'crc32'");
DEFINE_validator(hash_alg, &ValidateHashAlg);

DEFINE_uint32(num_bins, 256, "Number of bins used to discretize, if --transform=disc");

DEFINE_uint32(num_threads,
              1,
              "number of OpenMP threads, default: 1, "
              "use --num_threads=0 to use all available cores");


using namespace ts;

/**
 * The main class that takes care of running and comparing multiple sketch algorithms.
 * Takes a variable number of algorithms of potentially different types and runs them on the same
 * synthetically generated set of sequences.
 *
 * @tparam char_type the type of the characters in a sequence.
 * @tparam kmer_type the type of the characters in the sequence of kmers.
 * @tparam SketchAlgorithms the types of the sketch algorithms.
 */
template <class char_type, class kmer_type, class... SketchAlgorithms>
class ExperimentRunner {
    static constexpr size_t NumAlgorithms = sizeof...(SketchAlgorithms);
    std::tuple<SketchAlgorithms...> algorithms;

    Vec2D<char_type> seqs;
    std::vector<std::pair<uint32_t, uint32_t>> ingroup_pairs;

    std::vector<double> edit_dists;
    using Distances = std::array<std::vector<double>, NumAlgorithms>;
    Distances dists;

  public:
    ExperimentRunner() = delete;
    ExperimentRunner(ExperimentRunner &) = delete;

    explicit ExperimentRunner(SketchAlgorithms... algorithms) : algorithms(algorithms...) {}


    // Return the Spearman coefficient.
    template <class SketchAlgorithm>
    double run_sketch_algorithm(SketchAlgorithm *algorithm, std::vector<double> *dist) const {
        assert(algorithm != nullptr);
        assert(dist != nullptr);
        std::cout << "Running " << algorithm->name << std::endl;

        std::vector<typename SketchAlgorithm::sketch_type> sketch(seqs.size());

        // Compute sketches.
        std::cout << "\t" << "Compute sketches ... " << std::endl;
        progress_bar::init(seqs.size());
#pragma omp parallel for default(shared)
        for (uint32_t si = 0; si < seqs.size(); si++) {
            if constexpr (SketchAlgorithm::kmer_input) {
                sketch[si] = algorithm->compute(seqs[si], FLAGS_kmer_size, FLAGS_alphabet_size);
            } else {
                sketch[si] = algorithm->compute(seqs[si]);
            }
            progress_bar::iter();
        }

        // If needed, transform the sketch output.
        // Currently this is only done for TensorSketch variants when the command line flag is set.
        if constexpr (SketchAlgorithm::transform_sketches) {
            if (FLAGS_transform == "disc") {
                discretize<double> disc(FLAGS_num_bins);
                apply(sketch, disc);
            } else if (FLAGS_transform == "atan") {
                atan_scaler<double> atan;
                apply(sketch, atan);
            }
        }


        // Compute pairwise distances.
        std::cout << "\t" << "Compute distances ... " << std::endl;
        dist->resize(ingroup_pairs.size());
        progress_bar::init(ingroup_pairs.size());
#pragma omp parallel for default(shared)
        for (size_t i = 0; i < ingroup_pairs.size(); i++) {
            auto [si, sj] = ingroup_pairs[i];

            (*dist)[i] = algorithm->dist(sketch[si], sketch[sj]);
            progress_bar::iter();
        }


        // Print summary.
        auto spearman_coefficient = spearman(edit_dists, *dist);
        std::cout << "\t" << "Spearman Corr.: " << spearman_coefficient
                  << std::endl;

        // TODO: Save output.
        return spearman_coefficient;
    }


    void run() {
        std::cout << "Generating sequences ..." << std::endl;
        generate_sequences();
        std::cout << "Computing edit distances ..." << std::endl;
        compute_edit_distance();
        apply_tuple([&](auto &algorithm, auto &dist) { run_sketch_algorithm(&algorithm, &dist); },
                    algorithms, dists);
        std::cout << "Writing output to ... " << FLAGS_o << std::endl;
        save_output();
    }

    void generate_sequences() {
        ts::SeqGen seq_gen(FLAGS_alphabet_size, FLAGS_fix_len, FLAGS_num_seqs, FLAGS_seq_len,
                           FLAGS_group_size, FLAGS_max_mutation_rate, FLAGS_min_mutation_rate,
                           FLAGS_phylogeny_shape);

        seqs = seq_gen.generate_seqs<char_type>();
        seq_gen.ingroup_pairs(ingroup_pairs);
    }

    void compute_edit_distance() {
        edit_dists.resize(ingroup_pairs.size());
        progress_bar::init(ingroup_pairs.size());
#pragma omp parallel for default(shared)
        for (size_t i = 0; i < ingroup_pairs.size(); i++) {
            size_t si = ingroup_pairs[i].first, sj = ingroup_pairs[i].second;
            edit_dists[i] = edit_distance(seqs[si], seqs[sj]);
            progress_bar::iter();
        }
        std::cout << endl;
    }

    void save_output() {
        const std::filesystem::path output_dir(FLAGS_o);

        std::ofstream fo;
        fo.open(output_dir / "flags");
        assert(fo.is_open());
        fo << flag_values('\n', true);
        fo.close();

        fo.open(output_dir / "timing.csv");
        assert(fo.is_open());
        fo << Timer::summary(FLAGS_num_seqs, ingroup_pairs.size());
        fo.close();

        write_fasta(output_dir / "seqs.fa", seqs);

        fo.open(output_dir / "dists.csv");
        fo << "s1,s2,ED";
        apply_tuple([&](const auto &algo) { fo << "," << algo.name; }, algorithms);
        fo << "\n";
        for (uint32_t pi = 0; pi < ingroup_pairs.size(); pi++) {
            fo << ingroup_pairs[pi].first << "," << ingroup_pairs[pi].second; // seq 1 & 2 indices

            fo << "," << edit_dists[pi];
            for (const auto &dist : dists)
                fo << "," << dist[pi];
            fo << "\n";
        }
    }
};

// A small wrapper around the ExperimentRunner constructor that can be called without specifying the
// SketchAlgorithm types.
template <class char_type, class kmer_type, class... SketchAlgorithms>
ExperimentRunner<char_type, kmer_type, SketchAlgorithms...>
MakeExperimentRunner(SketchAlgorithms... algorithms) {
    return ExperimentRunner<char_type, kmer_type, SketchAlgorithms...>(algorithms...);
}


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_max_len == 0) { // 0: automatic computation, based on seq_len
        FLAGS_max_len = FLAGS_seq_len * 2;
    }
    if (FLAGS_num_threads > 0) { // 0: default: use all threads
        omp_set_num_threads(FLAGS_num_threads);
    }

    using char_type = uint8_t;
    using kmer_type = uint64_t;
    std::random_device rd;
    auto experiment = MakeExperimentRunner<char_type, kmer_type>(
            MinHash<kmer_type>(int_pow<uint32_t>(FLAGS_alphabet_size, FLAGS_kmer_size),
                               FLAGS_embed_dim, parse_hash_algorithm(FLAGS_hash_alg), rd(), "MH"),
            WeightedMinHash<kmer_type>(int_pow<uint32_t>(FLAGS_alphabet_size, FLAGS_kmer_size),
                                       FLAGS_embed_dim, FLAGS_max_len,
                                       parse_hash_algorithm(FLAGS_hash_alg), rd(), "WMH"),
            OrderedMinHash<kmer_type>(int_pow<uint32_t>(FLAGS_alphabet_size, FLAGS_kmer_size),
                                      FLAGS_embed_dim, FLAGS_max_len, FLAGS_tuple_length,
                                      parse_hash_algorithm(FLAGS_hash_alg), rd(), "OMH"),
            Tensor<char_type>(FLAGS_alphabet_size, FLAGS_embed_dim, FLAGS_tuple_length, rd(), "TS"),
            TensorSlide<char_type>(FLAGS_alphabet_size, ceil(sqrt(FLAGS_embed_dim)),
                                   FLAGS_tuple_length, FLAGS_window_size, FLAGS_stride, rd(),
                                   "TSS"),
            TensorSlideFlat<char_type, Int32Flattener>(
                    FLAGS_alphabet_size, ceil(sqrt(FLAGS_embed_dim)), FLAGS_tuple_length,
                    FLAGS_window_size, FLAGS_stride,
                    Int32Flattener(FLAGS_embed_dim, ceil(sqrt(FLAGS_embed_dim)), FLAGS_seq_len,
                                   rd()),
                    rd(), "TSS_flat_int32"),
            TensorSlideFlat<char_type, DoubleFlattener>(
                    FLAGS_alphabet_size, ceil(sqrt(FLAGS_embed_dim)), FLAGS_tuple_length,
                    FLAGS_window_size, FLAGS_stride,
                    DoubleFlattener(FLAGS_embed_dim, ceil(sqrt(FLAGS_embed_dim)), FLAGS_seq_len,
                                    rd()),
                    rd(), "TSS_flat_double"));
    experiment.run();

    return 0;
}
