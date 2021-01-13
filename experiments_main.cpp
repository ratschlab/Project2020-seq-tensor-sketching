#include "sequence/fasta_io.hpp"
#include "sequence/sequence_generator.hpp"
#include "sketch/hash_min.hpp"
#include "sketch/hash_ordered.hpp"
#include "sketch/hash_weighted.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"
#include "sketch/dim_reduce.h"
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
// Use CRC32 only for comparing speed
// TODO: implement a proper iterative permutation generation for hash*
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
        print_summary();
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
                           FLAGS_max_mutation_rate,
                           FLAGS_min_mutation_rate,
                           FLAGS_block_mutation_rate,
                           FLAGS_mutation_type,
                           FLAGS_phylogeny_shape);


        seqs = seq_gen.generate_seqs<char_type>();
        seq_gen.ingroup_pairs(ingroup_pairs);


        size_t num_seqs = seqs.size();
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ts_sketch.resize(num_seqs);
        tss_sketch.resize(num_seqs);
        tss_sketch_flat.resize(num_seqs);
        tss_sketch_binary.resize(num_seqs);

        embed_type set_size = int_pow<uint32_t>(FLAGS_alphabet_size, FLAGS_kmer_size);
        min_hash = MinHash<kmer_type>(set_size, FLAGS_embed_dim, FLAGS_hash_alg);
        wmin_hash = WeightedMinHash<kmer_type>(set_size, FLAGS_embed_dim, FLAGS_max_len, FLAGS_hash_alg);
        omin_hash = OrderedMinHash<kmer_type>(set_size, FLAGS_embed_dim, FLAGS_max_len,
                                            FLAGS_tuple_length, FLAGS_hash_alg);
        tensor_sketch = Tensor<char_type>(FLAGS_alphabet_size, FLAGS_embed_dim, FLAGS_tuple_length);
        auto inner_dim = ceil(sqrt(FLAGS_embed_dim));
        tensor_slide = TensorSlide<char_type>(FLAGS_alphabet_size, inner_dim, FLAGS_tuple_length,
                                            FLAGS_window_size, FLAGS_stride);
        l1SketchBin32 = flatten_int32(FLAGS_embed_dim, inner_dim, FLAGS_seq_len);
        l1Sketch = flatten_double(FLAGS_embed_dim, inner_dim, FLAGS_seq_len);

    }

    void compute_sketches() {
        progress_bar::init(seqs.size());
#pragma omp parallel for default(shared)
        for (uint32_t si = 0; si < seqs.size(); si++) {
            auto kmer_seq = seq2kmer<char_type, kmer_type>(
                    seqs[si], FLAGS_kmer_size, FLAGS_alphabet_size);
            mh_sketch[si] = min_hash.compute(kmer_seq);
            wmh_sketch[si] = wmin_hash.compute(kmer_seq);
            omh_sketch[si] = omin_hash.compute_flat(kmer_seq);
            ts_sketch[si] = tensor_sketch.compute(seqs[si]);
            tss_sketch[si] = tensor_slide.compute(seqs[si]);
            tss_sketch_flat[si] = l1Sketch.flatten(tss_sketch[si]);
            tss_sketch_binary[si] = l1SketchBin32.flatten(tss_sketch[si]);
            progress_bar::iter();
        }


    }

    void transform_sketches() {
        if (FLAGS_transform == "disc") {
            discretize<double> disc(FLAGS_num_bins);
            apply2D(ts_sketch, disc);
            apply3D(tss_sketch, disc);
        } else if (FLAGS_transform == "atan") {
            atan_scaler<double> atan;
            apply2D(ts_sketch, atan);
            apply3D(tss_sketch, atan);
        }
    }

    void compute_pairwise_dists() {
        dists = new2D<double>(8, ingroup_pairs.size());
        progress_bar::init(ingroup_pairs.size());
#pragma omp parallel for default(shared)
        for (size_t pi=0; pi< ingroup_pairs.size(); pi++ ) {
            size_t si = ingroup_pairs[pi].first, sj = ingroup_pairs[pi].second;
            dists[0][pi]=(edit_distance(seqs[si], seqs[sj]));
            dists[1][pi]= min_hash.dist(mh_sketch[si], mh_sketch[sj]);
            dists[2][pi]= wmin_hash.dist(wmh_sketch[si], wmh_sketch[sj]);
            dists[3][pi]= omin_hash.dist(omh_sketch[si], omh_sketch[sj]);
            dists[4][pi]= tensor_sketch.dist(ts_sketch[si], ts_sketch[sj]);
            dists[5][pi]= tensor_slide.dist(tss_sketch[si], tss_sketch[sj]);
            dists[6][pi]= l1Sketch.dist(tss_sketch_flat[si], tss_sketch_flat[sj]);
            dists[7][pi]= l1SketchBin32.dist(tss_sketch_binary[si], tss_sketch_binary[sj]);
            progress_bar::iter();
        }

    }

    void print_summary() {
        std::cout << "\tMH: " << spearman(dists[0], dists[1]) << std::endl;
        std::cout << "\tWMH: " << spearman(dists[0], dists[2]) << std::endl;
        std::cout << "\tOMH: " << spearman(dists[0], dists[3]) << std::endl;
        std::cout << "\tTensorSketch: " << spearman(dists[0], dists[4]) << std::endl;
        std::cout << "\tTensorSlide: " << spearman(dists[0], dists[5]) << std::endl;
        std::cout << "\tTensorSlideFlat: " << spearman(dists[0], dists[6]) << std::endl;
        std::cout << "\tTensorSlideFlat32: " << spearman(dists[0], dists[7]) << std::endl;
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
        fo << timer_summary(FLAGS_num_seqs, ingroup_pairs.size());
        fo.close();

        write_fasta(output_dir / "seqs.fa", seqs);

        std::vector<std::string> method_names = { "ED", "MH", "WMH", "OMH", "TS", "TSS", "TSS2"};
        fo.open(output_dir / "dists.csv");
        fo << "s1,s2";
        for (size_t m=0; m < method_names.size(); m++) { // table header
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

    }



  private:
    Vec2D<char_type> seqs;
    std::vector<std::string> seq_names;
    Vec2D<kmer_type> mh_sketch;
    Vec2D<kmer_type> wmh_sketch;
    Vec2D<kmer_type> omh_sketch;
    Vec2D<embed_type> ts_sketch;
    Vec3D<embed_type> tss_sketch;
    Vec2D<embed_type> tss_sketch_flat;
    Vec2D<uint32_t> tss_sketch_binary;

    MinHash<kmer_type> min_hash;
    WeightedMinHash<kmer_type> wmin_hash;
    OrderedMinHash<kmer_type> omin_hash;
    Tensor<char_type> tensor_sketch;
    TensorSlide<char_type> tensor_slide;
    flatten_int32 l1SketchBin32;
    flatten_double l1Sketch;

    Vec2D<double> dists;
    std::vector<std::pair<uint32_t,uint32_t>> ingroup_pairs;
};


int main(int argc, char *argv[]) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_max_len==0) { // 0: automatic computation, based on seq_len
        FLAGS_max_len = FLAGS_seq_len * 2;
    }
    if (FLAGS_num_threads > 0) { // 0: default: use all threads
        omp_set_num_threads(FLAGS_num_threads);
    }

    SeqGenModule<uint8_t, uint64_t, double> experiment;
    experiment.run();

    return 0;
}
