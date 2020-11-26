#include "sequence/fasta_io.hpp"
#include "sequence/sequence_generator.hpp"
#include "util/args.hpp"
#include "util/modules.hpp"
#include "util/multivec.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include <filesystem>
#include <fstream>
#include <memory>

DEFINE_int32(alphabet_size, 4, "Size of the alphabet for generated sequences");
DEFINE_bool(fix_len, false, "");
DEFINE_int32(max_num_blocks, 4, "Maximum number of blocks for block permutation");
DEFINE_int32(min_num_blocks, 2, "Minimum number of blocks for block permutation");
DEFINE_uint32(num_seqs, 200, "Number of sequences to be generated");
DEFINE_uint32(seq_len, 256, "The length of sequence to be generated");
DEFINE_double(mutation_rate, 0.015, "Rate of point mutation rate for sequence generation");
DEFINE_double(block_mutate_rate, 0.02, "The probability of having a block permutation");
DEFINE_uint32(sequence_seeds, 1, "Number of initial random sequences");

DEFINE_string(output, "./seqs.fa", "File name where the generated sequence should be written");

static bool ValidateMutationPattern(const char *flagname, const std::string &value) {
    if (value == "linear" || value == "tree" || value == "pairs")
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(mutation_pattern, "linear", "the mutational pattern, can be 'linear', or 'tree'");
DEFINE_validator(mutation_pattern, &ValidateMutationPattern);

namespace fs = std::filesystem;
using namespace ts;

struct KmerModule : public BasicModule {
    void override_module_params() override {
        alphabet_size = int_pow<size_t>(alphabet_size, kmer_size);
    }
};

template <class seq_type, class embed_type>
struct SeqGenModule {
    Vec2D<seq_type> seqs;
    Vec<std::string> seq_names;
    std::string test_id;
    Vec2D<seq_type> kmer_seqs;
    Vec2D<embed_type> mh_sketch;
    Vec2D<embed_type> wmh_sketch;
    Vec2D<embed_type> omh_sketch;
    Vec2D<embed_type> ten_sketch;
    Vec3D<embed_type> slide_sketch;
    Vec3D<embed_type> dists;

    BasicModule basicModules;
    KmerModule kmerModules;
    std::string output;

    void parse(int argc, char **argv) {
        basicModules.parse(argc, argv);
        basicModules.models_init();
        kmerModules.parse(argc, argv);
        kmerModules.models_init();
        output = basicModules.directory + basicModules.output;
    }

    void generate_sequences() {
        ts::SeqGen::Config config = { FLAGS_alphabet_size,
                                      FLAGS_fix_len,
                                      FLAGS_max_num_blocks,
                                      FLAGS_min_num_blocks,
                                      FLAGS_num_seqs,
                                      FLAGS_seq_len,
                                      (float)FLAGS_mutation_rate,
                                      (float)FLAGS_block_mutate_rate };
        ts::SeqGen seq_gen(config);

        if (FLAGS_mutation_pattern == "pairs") {
            seq_gen.genseqs_pairs(seqs);
        } else if (FLAGS_mutation_pattern == "linear") {
            seq_gen.genseqs_linear(seqs);
        } else if (FLAGS_mutation_pattern == "tree") {
            seq_gen.genseqs_tree(seqs, basicModules.sequence_seeds);
        }
    }

    void compute_sketches() {
        size_t num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ten_sketch.resize(num_seqs);
        slide_sketch.resize(num_seqs);
        for (size_t si = 0; si < num_seqs; si++) {
            kmer_seqs[si] = seq2kmer<seq_type, seq_type>(seqs[si], basicModules.kmer_size,
                                                         basicModules.alphabet_size);
            minhash(kmer_seqs[si], mh_sketch[si], kmerModules.mh_params);
            weighted_minhash(kmer_seqs[si], wmh_sketch[si], kmerModules.wmh_params);
            if (basicModules.tuple_on_kmer) {
                ordered_minhash_flat(kmer_seqs[si], omh_sketch[si], kmerModules.omh_params);
                //                tensor_sketch(kmer_seqs[si], ten_sketch[si],
                //                kmerModules.tensor_params); tensor_slide_sketch(kmer_seqs[si],
                //                slide_sketch[si], kmerModules.tensor_slide_params);
            } else {
                ordered_minhash_flat(seqs[si], omh_sketch[si], basicModules.omh_params);
            }
            tensor_sketch(seqs[si], ten_sketch[si], basicModules.tensor_params);
            tensor_slide_sketch(seqs[si], slide_sketch[si], basicModules.tensor_slide_params);
        }
    }

    void compute_pairwise_dists() {
        int num_seqs = seqs.size();
        if (basicModules.mutation_pattern == "pairs") {
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

        // std::filesystem::remove_all(std::filesystem::path(output));
        // std::filesystem::create_directories(std::filesystem::path(output + "/dists"));
        // std::filesystem::create_directories(std::filesystem::path(output + "/sketches"));
        fs::remove_all(fs::path(output));
        fs::create_directories(fs::path(output + "/dists"));
        fs::create_directories(fs::path(output + "/sketches"));

        fo.open(output + "conf.csv");
        assert(fo.is_open());
        fo << basicModules.config();
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
            if (basicModules.mutation_pattern == "pairs") {
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

    SeqGenModule<int, double> experiment;
    experiment.parse(argc, argv);
    if (experiment.basicModules.show_help) {
        std::cout << experiment.basicModules.description();
    } else {
        experiment.generate_sequences();
        experiment.compute_sketches();
        experiment.compute_pairwise_dists();
        experiment.save_output();
    }
    return 0;
}
