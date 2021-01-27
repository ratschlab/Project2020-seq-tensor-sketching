#include "sequence/fasta_io.hpp"
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
#include <random>
#include <sys/types.h>

DEFINE_string(alphabet,
              "dna4",
              "The alphabet over which sequences are defined (dna4, dna5, protein)");

DEFINE_double(length_ratio, 2, "Group sequences with similar length ");

DEFINE_uint32(kmer_length, 4, "Kmer size for MH, OMH, WMH");

DEFINE_string(o, "/tmp", "Directory where the generated sequence should be written");

DEFINE_string(i, "./data", "File where exact distances are precomputed, if left empty they're computed");

DEFINE_string(dist_file, "", "Directory where the generated sequence should be written");

DEFINE_int32(embed_dim, 30, "Embedding dimension, used for all sketching methods");

DEFINE_int32(tuple_length,
             3,
             "Ordered tuple length, used in ordered MinHash and Tensor-based sketches");

DEFINE_int32(
        max_kmer_count,
        10000,
        "The maximum number of kmers, used in WMH & OMH for hash computation");

DEFINE_uint32(seed, 0, "Seed for randomizes hashes. If 0, std::random_device is used.");


static bool ValidateTransformation(const char *flagname, const std::string &value) {
    if (value == "none" || value == "atan" || value == "disc")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(transform, "none", "transform TS and TSS output, can be 'none', 'atan' or 'disc'");
DEFINE_validator(transform, &ValidateTransformation);


static bool ValidateHashAlg(const char *flagname, const std::string &value) {
    if (value == "uniform" || value == "crc32" || value == "murmur") {
        return true;
    }
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
// Use CRC32 only for comparing speed
// TODO: implement a proper iterative permutation generation for hash*
DEFINE_string(hash_alg,
              "murmur",
              "hash algorithm to be used as basis, can be 'murmur', 'uniform', or 'crc32'");
DEFINE_validator(hash_alg, &ValidateHashAlg);

DEFINE_uint32(num_bins, 256, "Number of bins used to discretize, if --transform=disc");

DEFINE_uint32(num_threads, 0, "number of OpenMP threads, default: use all available cores");


DEFINE_uint32(reruns, 1, "The number of times to rerun sketch algorithms on the same data");

// individual flags, use global values if 0 (default)

DEFINE_uint32(mh_kmer_length, 0, "Kmer size for MH, default: kmer_length");
DEFINE_uint32(wmh_kmer_length, 0, "Kmer size for WMH, default: kmer_length");
DEFINE_uint32(omh_kmer_length, 0, "Kmer size for OMH, default: kmer_length");

DEFINE_int32(ts_dim, 0, "embedding dimension for TS sketch output, default: emed_dim");
DEFINE_int32(tss_dim, 0, "embedding dimension for TSS, default: square root of embed_dim");
DEFINE_int32(mh_dim, 0, "embedding dimension for MH, default: embed_dim");
DEFINE_int32(wmh_dim, 0, "embedding dimension for WMH, default: embed_dim");
DEFINE_int32(omh_dim, 0, "embedding dimension for OMH, default: embed_dim");

DEFINE_uint32(omh_tuple_length, 0, "Tuple length for OMH, default: tup_len");
DEFINE_uint32(ts_tuple_length, 0, "Tuple length for TS, default: tup_len");
DEFINE_uint32(tss_tuple_length, 0, "Tuple length for TSS, default: tup_len");

DEFINE_int32(tss_stride,32, "shift step for TSS sliding window, ");
DEFINE_int32(tss_window_size,256, "Window length: the size of sliding window in Tensor Slide Sketch");


void set_default_flags() {
    // kmer size
    if (FLAGS_omh_kmer_length == 0) {
        FLAGS_omh_kmer_length = FLAGS_kmer_length;
    }
    if (FLAGS_mh_kmer_length == 0) {
        FLAGS_mh_kmer_length = FLAGS_kmer_length;
    }
    if (FLAGS_wmh_kmer_length == 0) {
        FLAGS_wmh_kmer_length = FLAGS_kmer_length;
    }
    // embed dim
    if (FLAGS_tss_dim == 0) {
        FLAGS_tss_dim = ceil(sqrt(FLAGS_embed_dim));
    }
    if (FLAGS_ts_dim == 0) {
        FLAGS_ts_dim = FLAGS_embed_dim;
    }
    if (FLAGS_omh_dim == 0) {
        FLAGS_omh_dim = FLAGS_embed_dim;
    }
    if (FLAGS_wmh_dim == 0) {
        FLAGS_wmh_dim = FLAGS_embed_dim;
    }
    if (FLAGS_mh_dim == 0) {
        FLAGS_mh_dim = FLAGS_embed_dim;
    }
    // tuple length
    if (FLAGS_omh_tuple_length == 0) {
        FLAGS_omh_tuple_length = FLAGS_tuple_length;
    }
    if (FLAGS_ts_tuple_length == 0) {
        FLAGS_ts_tuple_length = FLAGS_tuple_length;
    }
    if (FLAGS_tss_tuple_length == 0) {
        FLAGS_tss_tuple_length = FLAGS_tuple_length;
    }
}

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
    std::map<std::string, size_t> seq_names;

    std::vector<std::pair<uint32_t, uint32_t>> cmp_pairs;

    std::vector<double> edit_dists;
    using Distances = std::array<std::vector<double>, NumAlgorithms>;
    Distances dists;

  public:
    ExperimentRunner() = delete;
    ExperimentRunner(ExperimentRunner &) = delete;

    explicit ExperimentRunner(SketchAlgorithms... algorithms) : algorithms(algorithms...) {}


    // Return the Spearman coefficient.
    template <class SketchAlgorithm>
    double run_sketch_algorithm(SketchAlgorithm *algorithm, std::vector<double> *store_dist) const {
        assert(algorithm != nullptr);

        // Initialize the algorithm.
        algorithm->init();

        std::vector<typename SketchAlgorithm::sketch_type> sketch(seqs.size());

        // Compute sketches.
        std::cout << "\t"
                  << "Compute sketches ... ";
        progress_bar::init(seqs.size());
#pragma omp parallel for default(shared)
        for (uint32_t si = 0; si < seqs.size(); si++) {
            if constexpr (algorithm->kmer_input) {
                sketch[si]
                        = algorithm->compute(seqs[si], algorithm->kmer_length, alphabet_size);
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

        std::cout << "\r";

        // Compute pairwise distances.
        std::cout << "\t"
                  << "Compute distances ... ";
        std::vector<double> dists(cmp_pairs.size(), 0);
        progress_bar::init(cmp_pairs.size());
#pragma omp parallel for default(shared)
        for (size_t i = 0; i < cmp_pairs.size(); i++) {
            auto [si, sj] = cmp_pairs[i];

            dists[i] += algorithm->dist(sketch[si], sketch[sj]);
            progress_bar::iter();
        }


        // Print summary.
        auto spearman_coefficient = spearman(edit_dists, dists);
        std::cout << "\t"
                  << "Spearman Corr.: " << spearman_coefficient << std::endl;

        if (store_dist) {
            store_dist->resize(dists.size());
            for (size_t i = 0; i < dists.size(); ++i)
                (*store_dist)[i] += dists[i];
        }


        return spearman_coefficient;
    }


    void run() {
        std::cout << "Reading Input ..." << std::endl;
        read_input();;

        std::cout << "Computing edit distances ... ";
        compute_edit_distance();
        apply_tuple(
                [&](auto &algorithm, auto &dist) {
                    std::cout << "Running " << algorithm.name << std::endl;

                    // Run the algorithms FLAGS_reruns times, storing the distances and Spearman
                    // coefficient computed in each run. The average and standard deviation of the
                    // Spearman coefficients is reported, as well as the Spearman coefficient
                    // obtained from using the median and average of the distances of all runs.
                    std::vector<double> spearman_coefficients(FLAGS_reruns);
                    Vec2D<double> dists_per_run = new2D<double>(FLAGS_reruns, cmp_pairs.size());

                    for (uint32_t rerun = 0; rerun < FLAGS_reruns; ++rerun) {
                        spearman_coefficients[rerun]
                                = run_sketch_algorithm(&algorithm, &dists_per_run[rerun]);
                    }

                    if (FLAGS_reruns > 1) {
                        // Transpose of dists_per_run.
                        Vec2D<double> runs_per_dist
                                = new2D<double>(cmp_pairs.size(), FLAGS_reruns);
                        for (size_t i = 0; i < FLAGS_reruns; ++i)
                            for (size_t j = 0; j < dists_per_run[i].size(); ++j)
                                runs_per_dist[j][i] = dists_per_run[i][j];

                        for (std::vector<double> &distances : runs_per_dist)
                            sort(begin(distances), end(distances));

                        const auto [avg, sd] = avg_stddev(spearman_coefficients);

                        std::cout << "\t"
                                  << "Average  Corr.: " << avg << " \t (σ=" << sd
                                  << ", n=" << FLAGS_reruns << ")" << std::endl;

                        dist.resize(cmp_pairs.size());

                        for (size_t i = 0; i < cmp_pairs.size(); ++i)
                            dist[i] = median(runs_per_dist[i]);
                        double sc_on_med_dist = spearman(edit_dists, dist);
                        std::cout << "\t"
                                  << "SC on med conv_dist: " << sc_on_med_dist << std::endl;

                        for (size_t i = 0; i < cmp_pairs.size(); ++i)
                            dist[i] = std::accumulate(begin(runs_per_dist[i]),
                                                      end(runs_per_dist[i]), 0.0)
                                    / FLAGS_reruns;
                        double sc_on_avg_dist = spearman(edit_dists, dist);
                        std::cout << "\t"
                                  << "SC on avg conv_dist: " << sc_on_avg_dist << std::endl;

                        std::cout << std::endl;
                    } else {
                        dist = dists_per_run[0];
                    }
                },
                algorithms, dists);
        std::cout << "Writing output to ... " << FLAGS_o << std::endl;
        save_output();
    }

    void read_input() {
        std::cerr << "Reading input ... " << std::endl;
        std::vector<FastaFile<char_type>> files = read_directory<char_type>(FLAGS_i);
        for (auto & fastfile : files) {
            assert(fastfile.sequences.size() == fastfile.comments.size()); // every part is preceded with a comment
            if (fastfile.sequences.size()>1 || fastfile.sequences[0].size()>10000 || fastfile.sequences[0].size()<500) // filter out
                continue;

            for (size_t i=0; i<fastfile.sequences.size(); i++) {
                seq_names[fastfile.filename] = seqs.size();
                seqs.push_back(fastfile.sequences[i]);
            }

        }
        std::cerr << "Read " << files.size() << " files" << std::endl;

        for (size_t i=0; i<seqs.size(); i++) {
            for (size_t j = i+1; j<seqs.size(); j++) {
                size_t len1 = seqs[i].size(), len2 = seqs[j].size();
                if ((double)std::max(len1,len2)/std::min(len1,len2) < FLAGS_length_ratio)
                    cmp_pairs.emplace_back(i, j);
            }
        }

    }


    void compute_edit_distance() {
        if (!FLAGS_dist_file.empty()) {
            cmp_pairs.clear();
            edit_dists.clear();
            auto data = read_csv(FLAGS_dist_file);
            for (auto &row : data) {
                size_t si = seq_names[row[0].substr(0,row[0].size()-3)],    // remove .gz name
                sj = seq_names[row[1].substr(0,row[1].size()-3)];
                double d = std::stod(row[2]);
                size_t l1 = seqs[si].size(), l2 = seqs[sj].size();
                d /= std::max(seqs[si].size(), seqs[sj].size());    // normalize to max length
//                d = d - (double)(l1-l2>0? l1-l2 : l2-l1); // subtract length diff
//                d = d / std::min(l1,l2);                    // divide by min length

                if (std::max(l1,l2) * 1.0 / std::min(l1,l2) < FLAGS_length_ratio ) {
                    edit_dists.push_back(d);
                    cmp_pairs.emplace_back(si, sj);
                }
            }
        } else {
            edit_dists.resize(cmp_pairs.size());
            progress_bar::init(cmp_pairs.size());
#pragma omp parallel for default(shared)
            for (size_t i = 0; i < cmp_pairs.size(); i++) {
                size_t si = cmp_pairs[i].first, sj = cmp_pairs[i].second;
                edit_dists[i] = edit_distance(seqs[si], seqs[sj]);
                edit_dists[i] /= std::max(seqs[si].size(), seqs[sj].size());    // normalize to max length
                progress_bar::iter();
            }
        }

        std::cout << std::endl;
    }

    void save_output() {
        const std::filesystem::path output_dir(FLAGS_o);
        std::filesystem::create_directories(FLAGS_o);

        std::ofstream fo;
        fo.open(output_dir / "flags");
        assert(fo.is_open());
        fo << flag_values('\n', true);
        fo.close();

        fo.open(output_dir / "timing.csv");
        assert(fo.is_open());
        fo << Timer::summary();
        fo.close();

//        fo.open(output_dir / "dists.csv");
//        fo << "s1,s2,ED";
//        apply_tuple([&](const auto &algo) { fo << "," << algo.name; }, algorithms);
//        fo << "\n";
//        for (uint32_t pi = 0; pi < cmp_pairs.size(); pi++) {
//            fo << cmp_pairs[pi].first << "," << cmp_pairs[pi].second; // seq 1 & 2 indices
//
//            fo << "," << edit_dists[pi];
//            for (const auto &dist : dists)
//                fo << "," << dist[pi];
//            fo << "\n";
//        }
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
    if (FLAGS_num_threads > 0) { // 0: default: use all threads
        omp_set_num_threads(FLAGS_num_threads);
    }
    if (std::pow(alphabet_size, FLAGS_kmer_length) > (double)std::numeric_limits<uint64_t>::max()) {
        std::cerr << "Kmer size is too large to fit in 64 bits " << std::endl;
        std::exit(1);
    }
    init_alphabet(FLAGS_alphabet);

    set_default_flags();

    using char_type = uint8_t;
    using kmer_type = uint64_t;
    std::random_device random_device;
    auto rd = [&] {
        if (FLAGS_seed)
            return FLAGS_seed;
        return random_device();
    };

    auto experiment = MakeExperimentRunner<char_type, kmer_type>(
            MinHash<kmer_type>(int_pow<uint32_t>(alphabet_size, FLAGS_mh_kmer_length),
                               FLAGS_mh_dim, parse_hash_algorithm(FLAGS_hash_alg), rd(), "MH",
                               FLAGS_mh_kmer_length),
//            WeightedMinHash<kmer_type>(int_pow<uint32_t>(alphabet_size, FLAGS_wmh_kmer_length),
//                                       FLAGS_wmh_dim, FLAGS_max_kmer_count,
//                                       parse_hash_algorithm(FLAGS_hash_alg), rd(), "WMH",
//                                       FLAGS_wmh_kmer_length),
//            OrderedMinHash<kmer_type>(int_pow<uint32_t>(alphabet_size, FLAGS_omh_kmer_length),
//                                      FLAGS_omh_dim, FLAGS_max_kmer_count, FLAGS_omh_tuple_length,
//                                      parse_hash_algorithm(FLAGS_hash_alg), rd(), "OMH",
//                                      FLAGS_omh_kmer_length),
            Tensor<char_type>(alphabet_size, FLAGS_ts_dim, FLAGS_ts_tuple_length, rd(), "TS"),
            TensorSlide<char_type>(alphabet_size, FLAGS_tss_dim, FLAGS_tss_tuple_length,
                                   FLAGS_tss_window_size, FLAGS_tss_stride, rd(), "TSS"));
    experiment.run();

    return 0;
}
