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
#include "util/progress.hpp"
#include "util/utils.hpp"
#include "util/transformer.hpp"

#include <filesystem>
#include <memory>
#include <omp.h>


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



namespace fs = std::filesystem;
using namespace ts;

template <class char_type, class kmer_type, class embed_type>
struct SeqGenModule {
    SeqGenModule() {}

    void run() {
        std::cout << "Generating sequences ..." << std::flush;
        generate_sequences();
        std::cout << "\nComputing sketches ... " << std::flush;
        compute_sketches();
        std::cout << "\nTransform sketches ... " << std::flush;
        transform_sketches();
        std::cout << "\nComputing distances ... " << std::flush;
        compute_pairwise_dists();
        std::cout << "\nComputing Spearman correlation ... \n" << std::flush;
        print_spearman();
        std::cout << "Writing output to ... " << FLAGS_o  <<  std::endl;
        save_output();
    }

    void generate_sequences() {
        ts::SeqGen seq_gen(FLAGS_alphabet_size, FLAGS_fix_len,
                           FLAGS_max_num_blocks,
                           FLAGS_min_num_blocks,
                           FLAGS_num_seqs,
                           FLAGS_seq_len,
                           FLAGS_group_size,
                           (double)FLAGS_max_mutation_rate,
                           (double)FLAGS_min_mutation_rate,
                           (double)FLAGS_block_mutation_rate,
                           FLAGS_mutation_type,
                           FLAGS_phylogeny_shape);


        seq_gen.generate_seqs(seqs);
        seq_gen.ingroup_pairs(ingroup_pairs);


        uint32_t num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ten_sketch.resize(num_seqs);
        slide_sketch.resize(num_seqs);

    }

    void compute_sketches() {
        embed_type set_size = int_pow<uint32_t>(FLAGS_alphabet_size, FLAGS_kmer_size);
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


        progress_bar::init(seqs.size());
#pragma omp parallel for
        for (uint32_t si = 0; si < seqs.size(); si++) {
            kmer_seqs[si] = seq2kmer<char_type, kmer_type>(
                    seqs[si], FLAGS_kmer_size, FLAGS_alphabet_size);
            mh_sketch[si] = min_hash.compute(kmer_seqs[si]);
            wmh_sketch[si] = wmin_hash.compute(kmer_seqs[si]);
            omh_sketch[si] = omin_hash.compute_flat(kmer_seqs[si]);
            ten_sketch[si] = tensor_sketch.compute(seqs[si]);
            slide_sketch[si] = tensor_slide.compute(seqs[si]);
            progress_bar::iter();
        }


    }

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
        dists = new2D<double>(6, ingroup_pairs.size());
        progress_bar::init(ingroup_pairs.size());
#pragma omp parallel for default(shared)
        for (size_t pi=0; pi< ingroup_pairs.size(); pi++ ) {
            size_t si = ingroup_pairs[pi].first, sj = ingroup_pairs[pi].second;
            dists[0][pi]=(edit_distance(seqs[si], seqs[sj]));
            dists[1][pi]=(hamming_dist(mh_sketch[si], mh_sketch[sj]));
            dists[2][pi]=(hamming_dist(wmh_sketch[si], wmh_sketch[sj]));
            dists[3][pi]=(hamming_dist(omh_sketch[si], omh_sketch[sj]));
            dists[4][pi]=(l1_dist(ten_sketch[si], ten_sketch[sj]));
            dists[5][pi]=(l1_dist2D_minlen(slide_sketch[si], slide_sketch[sj]));
            progress_bar::iter();
        }


    }

    void print_spearman() {
        std::cout << "\tMH: " << spearman(dists[0], dists[1]) << std::endl;
        std::cout << "\tWMH: " << spearman(dists[0], dists[2]) << std::endl;
        std::cout << "\tOMH: " << spearman(dists[0], dists[3]) << std::endl;
        std::cout << "\tTensorSketch: " << spearman(dists[0], dists[4]) << std::endl;
        std::cout << "\tTensorSlide: " << spearman(dists[0], dists[5]) << std::endl;
    }

    void save_output() {
        const std::filesystem::path output_dir(FLAGS_o);

        std::vector<std::string> method_names
                = { "ED", "MH", "WMH", "OMH", "TenSketch", "TenSlide"};
        std::ofstream fo;

        fs::create_directories(fs::path(output_dir / "sketches"));

        fo.open(output_dir / "flags");
        assert(fo.is_open());
        fo << flag_values('\n', true);
        fo.close();

        fo.open(output_dir / "timing.csv");
        assert(fo.is_open());
        fo << timer_summary(FLAGS_num_seqs, ingroup_pairs.size());
        fo.close();

        write_fasta(output_dir / "seqs.fa", seqs);

        fo.open(output_dir / "dists.csv");
        fo << "s1,s2";
        for (int m=0; m<6; m++) { // table header
            fo << "," << method_names[m];
        }
        fo << "\n";
        for (uint32_t pi=0; pi< ingroup_pairs.size(); pi++) {
            fo << ingroup_pairs[pi].first << "," << ingroup_pairs[pi].second; // seq 1 & 2 indices
            for (int m=0; m<6; m++) { // distance based on each method
                fo << "," << dists[m][pi];
            }
            fo << "\n";
        }

        fo.open(output_dir / "sketches" /  "MH.txt");
        assert(fo.is_open());
        for (uint32_t si = 0; si < mh_sketch.size(); si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : mh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output_dir / "sketches" /  "WMH.txt");
        assert(fo.is_open());
        for (uint32_t si = 0; si < wmh_sketch.size(); si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : wmh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output_dir / "sketches" /  "OMH.txt");
        assert(fo.is_open());
        for (uint32_t si = 0; si < omh_sketch.size(); si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : omh_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output_dir / "sketches" /  "TenSketch.txt");
        assert(fo.is_open());
        for (uint32_t si = 0; si < ten_sketch.size(); si++) {
            fo << ">> seq " << si << "\n";
            for (const auto &e : ten_sketch[si]) {
                fo << e << ", ";
            }
            fo << "\n";
        }
        fo.close();

        fo.open(output_dir / "sketches" /  "TenSlide.txt");
        for (uint32_t si = 0; si < seqs.size(); si++) {
            auto &sk = slide_sketch[si];
            for (uint32_t dim = 0; dim < sk.size(); dim++) {
                fo << ">> seq: " << si << ", dim: " << dim << "\n";
                for (auto &item : sk[dim])
                    fo << item << ", ";
                fo << "\n";
            }
            fo << "\n";
        }
        fo.close();
    }



  private:
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
    std::vector<std::pair<uint32_t,uint32_t>> ingroup_pairs;
};


int main(int argc, char *argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_max_len<0) {
        FLAGS_max_len = FLAGS_seq_len;
    }
    if (FLAGS_num_threads > 0) {
        omp_set_num_threads(FLAGS_num_threads);
    }

    SeqGenModule<uint8_t, uint64_t, double> experiment;
    experiment.run();

    return 0;
}
