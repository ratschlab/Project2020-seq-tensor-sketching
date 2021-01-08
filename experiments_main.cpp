#include "sequence/fasta_io.hpp"
#include "sequence/sequence_generator.hpp"
#include "sketch/hash_min.hpp"
#include "sketch/hash_ordered.hpp"
#include "sketch/hash_weighted.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"
#include "util/multivec.hpp"
#include "util/spearman.hpp"
#include "util/Timer.hpp"
#include "util/progress.hpp"
#include "util/utils.hpp"
#include "util/transformer.hpp"

#include <filesystem>
#include <memory>
#include <omp.h>


DEFINE_uint32(kmer_size, 3, "Kmer size for MH, OMH, WMH");
DEFINE_uint32(k, 3, "Short hand for --kmer_size");

//DEFINE_string(alphabet,
//              "dna4",
//              "The alphabet over which sequences are defined (dna4, dna5, protein)");
DEFINE_int32(alphabet_size, 4, "size of alphabet for synthetic sequence generation");

DEFINE_int32(max_num_blocks, 4, "Maximum number of blocks for block permutation");

DEFINE_int32(min_num_blocks, 2, "Minimum number of blocks for block permutation");

DEFINE_uint32(num_seqs, 200, "Number of sequences to be generated");

DEFINE_uint32(seq_len, 256, "The length of sequence to be generated");

DEFINE_bool(fix_len, false, "Force generated sequences length to be equal");

DEFINE_double(mutation_rate, 0.3, "Maximum rate of point mutation for sequence generation");

DEFINE_double(min_mutation_rate, 0.0, "Minimum rate of point mutation for sequence generation");


DEFINE_double(block_mutation_rate, 0.00, "The probability of having a block permutation");

DEFINE_uint32(group_size, 2, "Number of sequences in each independent group");

DEFINE_string(o, "/tmp", "Directory where the generated sequence should be written");

DEFINE_int32(embed_dim, 16, "Embedding dimension, used for all sketching methods");

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

static bool validatePhylogenyShape(const char *flagname, const std::string &value) {
    if (value == "path" || value == "tree" )
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(phylogeny_shape,
              "path",
              "shape of the phylogeny can be 'path', 'tree'");
DEFINE_validator(phylogeny_shape, &validatePhylogenyShape);


static bool validateMutationType(const char *flagname, const std::string &value) {
    if (value == "rate" || value == "edit" )
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(mutation_type,
              "rate",
              "basic method used for mutating sequences can be 'rate', 'edit'");
DEFINE_validator(mutation_type, &validateMutationType);


static bool ValidateTransformation(const char *flagname, const std::string &value) {
    if (value == "none" || value == "atan" || value == "disc" )
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(transform,
              "none",
              "transform TS and TSS output, can be 'none', 'atan' or 'disc'");
DEFINE_validator(transform, &ValidateTransformation);


static bool ValidateHashAlg(const char *flagname, const std::string &value) {
    if (value == "uniform" || value == "crc32" )
        return true;
    printf("Invalid value for --%s: %s\n", flagname, value.c_str());
    return false;
}
DEFINE_string(hash_alg,
              "uniform",
              "hash algorithm to be used as basis, can be 'uniform', or 'crc32'");
DEFINE_validator(hash_alg, &ValidateHashAlg);

DEFINE_uint32(num_bins, 256, "Number of bins used to discretize, if --transform=disc");

DEFINE_uint32(num_threads, 1, "number of OpenMP threads, default: 1, "
                              "use --num_threads=0 to use all available cores");


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
    Vec2D<double> dists;
    std::vector<std::pair<size_t,size_t>> pairs;


    std::filesystem::path output_dir;

    SeqGenModule(const std::string &out_dir) : output_dir(out_dir) {}

    void generate_sequences() {
        ts::SeqGen seq_gen(FLAGS_alphabet_size, FLAGS_fix_len,
                           FLAGS_max_num_blocks,
                           FLAGS_min_num_blocks,
                           FLAGS_num_seqs,
                           FLAGS_seq_len,
                           FLAGS_group_size,
                           (float)FLAGS_mutation_rate,
                           (float)FLAGS_min_mutation_rate,
                           (float)FLAGS_block_mutation_rate,
                           FLAGS_mutation_type);


        if (FLAGS_phylogeny_shape == "path") {
            seq_gen.generate_path(seqs);
        } else if (FLAGS_phylogeny_shape == "tree") {
            seq_gen.generate_tree(seqs);
        }


        size_t num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ten_sketch.resize(num_seqs);
        slide_sketch.resize(num_seqs);

    }

    void compute_sketches() {
        embed_type set_size = int_pow<size_t>(FLAGS_alphabet_size, FLAGS_kmer_size);
        MinHash<kmer_type> min_hash(set_size, FLAGS_embed_dim);
        WeightedMinHash<kmer_type> wmin_hash(set_size, FLAGS_embed_dim, FLAGS_max_len);
        OrderedMinHash<kmer_type> omin_hash(set_size, FLAGS_embed_dim, FLAGS_max_len,
                                            FLAGS_tuple_length);
        Tensor<char_type> tensor_sketch(FLAGS_alphabet_size, FLAGS_embed_dim, FLAGS_tuple_length);
        TensorSlide<char_type> tensor_slide(FLAGS_alphabet_size, FLAGS_embed_dim, FLAGS_tuple_length,
                                            FLAGS_window_size, FLAGS_stride, FLAGS_seq_len);
        min_hash.set_hash_algorithm(FLAGS_hash_alg);
        wmin_hash.set_hash_algorithm(FLAGS_hash_alg);
        omin_hash.set_hash_algorithm(FLAGS_hash_alg);


        PB_init(seqs.size());
#pragma omp parallel for
        for (size_t si = 0; si < seqs.size(); si++) {
            kmer_seqs[si] = seq2kmer<char_type, kmer_type>(
                    seqs[si], FLAGS_kmer_size, FLAGS_alphabet_size);
            mh_sketch[si] = min_hash.compute(kmer_seqs[si]);
            wmh_sketch[si] = wmin_hash.compute(kmer_seqs[si]);
            omh_sketch[si] = omin_hash.compute_flat(kmer_seqs[si]);
            ten_sketch[si] = tensor_sketch.compute(seqs[si]);
            slide_sketch[si] = tensor_slide.compute(seqs[si]);
            PB_iter();
        }

        std::cout << std::endl;

    }

    // bin or scale sketch output of tensor sketch and tensor slide sketch
    void transform_sketches() {
        if (FLAGS_transform == "disc") {
            discretize<double> disc(FLAGS_num_bins);
            apply2D(ten_sketch, disc);
            apply3D(slide_sketch, disc);
        } else if (FLAGS_transform == "atan") {
            atan_scaler<double> atan;
            apply2D(ten_sketch, atan);
            apply3D(slide_sketch, atan);
        }
    }

    void compute_pairwise_dists() {
        for (size_t g =0; g < seqs.size(); g += FLAGS_group_size ) { // g: group offset
            for (size_t i = 0; i < FLAGS_group_size ; i++) {
                for (size_t j = i + 1; j < FLAGS_group_size && g + j < seqs.size(); j++) {
                    pairs.push_back({ g + i, g + j }); // g+i, g+i -> index i & j in the group
                }
            }
        }
        dists = new2D<double>(6, pairs.size());
        PB_init(pairs.size());
#pragma omp parallel for default(shared)
        for (size_t pi=0; pi< pairs.size(); pi++ ) {
            size_t si = pairs[pi].first, sj = pairs[pi].second;
            dists[0][pi]=(edit_distance(seqs[si], seqs[sj]));
            dists[1][pi]=(hamming_dist(mh_sketch[si], mh_sketch[sj]));
            dists[2][pi]=(hamming_dist(wmh_sketch[si], wmh_sketch[sj]));
            dists[3][pi]=(hamming_dist(omh_sketch[si], omh_sketch[sj]));
            dists[4][pi]=(l1_dist(ten_sketch[si], ten_sketch[sj]));
            dists[5][pi]=(l1_dist2D_minlen(slide_sketch[si], slide_sketch[sj]));
            PB_iter();
        }

        std::cout << std::endl;
    }

    void print_spearman() {
        std::cout << "Spearman correlation MH: " << spearman(dists[0], dists[1]) << std::endl;
        std::cout << "Spearman correlation WMH: " << spearman(dists[0], dists[2]) << std::endl;
        std::cout << "Spearman correlation OMH: " << spearman(dists[0], dists[3]) << std::endl;
        std::cout << "Spearman correlation TensorSketch: "
                  << spearman(dists[0], dists[4]) << std::endl;
        std::cout << "Spearman correlation TensorSlide: "
                  << spearman(dists[0], dists[5]) << std::endl;
    }

    void save_output() {
        std::vector<std::string> method_names
                = { "ED", "MH", "WMH", "OMH", "TenSketch", "TenSlide", "Ten2", "Ten2Slide" };
        std::ofstream fo;

        fs::create_directories(fs::path(output_dir / "dists"));
        fs::create_directories(fs::path(output_dir / "sketches"));

        fo.open(output_dir / "conf");
        assert(fo.is_open());
        fo << flag_values();
        fo.close();

        // legacy: to be able to run old matlab scripts
        fo.open(output_dir / "legacy_conf.csv");
        assert(fo.is_open());
        fo << legacy_config();
        fo.close();

        fo.open(output_dir / "timing.csv");
        assert(fo.is_open());
        fo << timer_summary(FLAGS_num_seqs);
        fo.close();

        write_fasta(output_dir / "seqs.fa", seqs);

        size_t num_seqs = seqs.size();
        for (int m = 0; m < 6; m++) {
            fo.open(output_dir / "dists" / (method_names[m] + ".txt"));
            assert(fo.is_open());
            for (size_t i=0; i<pairs.size(); i++) {
                fo << pairs[i].first << ", " << pairs[i].second << ", " << dists[m][i] << "\n";
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
    if (FLAGS_num_threads > 0)
        omp_set_num_threads(FLAGS_num_threads);

    timer_start("main_func"); // start measuring the main execution time
    SeqGenModule<uint8_t, uint64_t, double> experiment(FLAGS_o);
    std::cout << "Generating sequences ..." << std::endl;
    experiment.generate_sequences();
    std::cout << "Computing sketches ... " << std::endl;
    experiment.compute_sketches();
    std::cout << "Transform sketches ... " << std::endl;
    experiment.transform_sketches();
    std::cout << "Computing distances ... " << std::endl;
    experiment.compute_pairwise_dists();
    experiment.print_spearman();
    timer_stop(); // stop measuring time

    std::cout << "Writing output to " << FLAGS_o << std::endl;
    experiment.save_output();


    return 0;
}
