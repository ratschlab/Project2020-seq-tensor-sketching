#include "sequence/fasta_io.hpp"

#include "sketch/hash_min.hpp"
#include "sketch/hash_ordered.hpp"
#include "sketch/hash_weighted.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"

#include "util/multivec.hpp"
#include "util/utils.hpp"

#include <filesystem>
#include <memory>
#include <sstream>

DEFINE_string(sketch_method,
              "TenSlide",
              "The sketching method to use: MH, WMH, OMH, TenSketch or TenSlide");
DEFINE_string(m, "TenSlide", "Short hand for --sketch_method");

DEFINE_uint32(kmer_size, 1, "The sketching method to use: MH, WMH, OMH, TenSketch or TenSlide");
DEFINE_uint32(K, 3, "Short hand for --kmer_size");

// TODO: this should be determined by sequence/alphabets.hpp
DEFINE_int32(alphabet_size, 5, "Size of the alphabet for generated sequences");
DEFINE_int32(A, 4, "Short hand for --alphabet_size");

DEFINE_string(output_dir, "/tmp/", "Output directory for sketches");
DEFINE_string(o, "./seqs.fa", "Short hand for --output");

DEFINE_string(input,
              "./data/fasta/seqs.fa",
              "File name where the generated sequence should be written");
DEFINE_string(i, "./seqs.fa", "Short hand for --input");

DEFINE_string(format_input, "fasta", "Input format: 'fasta', 'csv'");
DEFINE_string(f, "fasta", "Short hand for --format_input");

DEFINE_int32(embed_dim, 128, "Embedding dimension, used for all sketching methods");
DEFINE_int32(M, 128, "Short hand for --embed_dim");

DEFINE_int32(tup_len, 2, "Ordered tuple length, used in ordered MinHash and Tensor-based sketches");
DEFINE_int32(T, 2, "Short hand for --tup_len");

DEFINE_int32(num_phases,
             2,
             "Number of phases to be used for modular arithmetic in tensor sketching");
DEFINE_int32(P, 2, "Short hand for --num_phases");

DEFINE_int32(num_bins, 255, "Number of bins for discretization after tensor sketching");
DEFINE_int32(n, 255, "Short hand for --num_bins");

DEFINE_int32(win_len, 32, "Window length: the size of sliding window in Tensor Slide Sketch");
DEFINE_int32(W, 32, "Short hand for --win_len");

DEFINE_int32(max_len, 32, "The maximum accepted sequence length for Ordered and Weighted min-hash");

DEFINE_int32(stride, 8, "Stride for sliding window: shift step for sliding window");
DEFINE_int32(S, 8, "Short hand for --stride");

DEFINE_int32(offset, 0, "Initial index to start the sliding window");
DEFINE_int32(O, 0, "Short hand for --offset");

void adjust_short_names() {
    if (!gflags::GetCommandLineFlagInfoOrDie("A").is_default) {
        FLAGS_alphabet_size = FLAGS_A;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("m").is_default) {
        FLAGS_sketch_method = FLAGS_m;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("K").is_default) {
        FLAGS_kmer_size = FLAGS_K;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("i").is_default) {
        FLAGS_input = FLAGS_i;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("M").is_default) {
        FLAGS_embed_dim = FLAGS_M;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("f").is_default) {
        FLAGS_format_input = FLAGS_f;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("T").is_default) {
        FLAGS_tup_len = FLAGS_T;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("P").is_default) {
        FLAGS_num_phases = FLAGS_P;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("n").is_default) {
        FLAGS_num_bins = FLAGS_n;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("o").is_default) {
        FLAGS_output_dir = FLAGS_o;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("W").is_default) {
        FLAGS_win_len = FLAGS_W;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("S").is_default) {
        FLAGS_stride = FLAGS_S;
    }
    if (!gflags::GetCommandLineFlagInfoOrDie("O").is_default) {
        FLAGS_offset = FLAGS_O;
    }
}

using namespace ts;

template <typename seq_type, class kmer_type, class embed_type>
class SketchHelper {
  public:
    SketchHelper(const std::function<std::vector<embed_type>(const std::vector<kmer_type> &)> &sketcher,
                 const std::function<Vec2D<double>(const std::vector<uint64_t> &)> &slide_sketcher)
        : sketcher(sketcher), slide_sketcher(slide_sketcher) {}

    void compute_sketches() {
        size_t num_seqs = seqs.size();
        slide_sketch = new3D<embed_type>(seqs.size(), FLAGS_embed_dim, 0);
        for (size_t si = 0; si < num_seqs; si++) {
            std::vector<kmer_type> kmers
                    = seq2kmer<seq_type, kmer_type>(seqs[si], FLAGS_kmer_size, FLAGS_alphabet_size);

            for (int i = FLAGS_offset; i < sketch_end(FLAGS_offset, kmers.size());
                 i += FLAGS_stride) {
                auto end = std::min(kmers.begin() + i + FLAGS_win_len, kmers.end());
                std::vector<kmer_type> kmer_slice(kmers.begin() + i, end);
                std::vector<embed_type> embed_slice = sketcher(kmer_slice);
                for (int m = 0; m < FLAGS_embed_dim; m++) {
                    slide_sketch[si][m].push_back(embed_slice[m]);
                }
            }
        }
    }

    void compute_slide() {
        slide_sketch = new3D<double>(seqs.size(), FLAGS_embed_dim, 0);

        for (size_t si = 0; si < seqs.size(); si++) {
            std::vector<uint64_t> kmers
                    = seq2kmer<uint8_t, uint64_t>(seqs[si], FLAGS_kmer_size, FLAGS_alphabet_size);
            slide_sketch[si] = slide_sketcher(kmers);
        }
    }

    void read_input() {
        std::tie(test_id, seqs, seq_names) = read_fasta<seq_type>(FLAGS_input, FLAGS_format_input);
    }

    void save_output() {
        std::ofstream fo(std::filesystem::path(FLAGS_output_dir) / (FLAGS_sketch_method + ".txt"));
        if (!fo.is_open()) {
            std::cerr << "output file not opened\n";
        }
        fo << test_id << "\n";
        fo << "# " << flag_values() << "\n";
        for (size_t si = 0; si < slide_sketch.size(); si++) {
            for (size_t m = 0; m < slide_sketch[si].size(); m++) {
                fo << seq_names[si] << ">" << std::dec << m << "\n";
                for (size_t i = 0; i < slide_sketch[si][m].size(); i++) {
                    if (FLAGS_num_bins == 0) {
                        fo << slide_sketch[si][m][i] << ",";
                    } else {
                        fo << std::hex << (int)slide_sketch[si][m][i] << ",";
                    }
                }
                fo << "\n";
            }
            fo << "\n";
        }
        fo.close();
    }

  private:
    Vec2D<seq_type> seqs;
    std::vector<std::string> seq_names;
    Vec3D<embed_type> slide_sketch;
    std::string test_id;

    std::function<std::vector<embed_type>(const std::vector<kmer_type> &)> sketcher;
    std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher;
};

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    adjust_short_names();

    if (std::pow(FLAGS_alphabet_size, FLAGS_kmer_size)
        > (double)std::numeric_limits<uint64_t>::max()) {
        std::cerr << "Kmer size is too large to fit in 64 bits " << std::endl;
        std::exit(1);
    }

    uint64_t kmer_word_size = int_pow<uint64_t>(FLAGS_alphabet_size, FLAGS_kmer_size);

    if (FLAGS_sketch_method.ends_with("MH")) {
        std::function<std::vector<uint64_t>(const std::vector<uint64_t> &)> sketcher;
        MinHash<uint64_t> min_hash;
        WeightedMinHash<uint64_t> wmin_hash;
        OrderedMinHash<uint64_t> omin_hash;

        if (FLAGS_sketch_method == "MH") {
            min_hash = MinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim);
            sketcher = [&](const std::vector<uint64_t> &seq) { return min_hash.compute(seq); };
        } else if (FLAGS_sketch_method == "WMH") {
            wmin_hash = WeightedMinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim, FLAGS_max_len);
            sketcher = [&](const std::vector<uint64_t> &seq) { return wmin_hash.compute(seq); };
        } else if (FLAGS_sketch_method == "OMH") {
            omin_hash = OrderedMinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim, FLAGS_max_len,
                                                 FLAGS_tup_len);
            sketcher
                    = [&](const std::vector<uint64_t> &seq) { return omin_hash.compute_flat(seq); };
        }
        std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher
                = [&](const std::vector<uint64_t> &) { return new2D<double>(0, 0); };
        SketchHelper<uint8_t, uint64_t, uint64_t> sketch_helper(sketcher, slide_sketcher);
        sketch_helper.read_input();
        sketch_helper.compute_sketches();
        sketch_helper.save_output();
    } else if (FLAGS_sketch_method.starts_with("TenS")) {
        Tensor<uint64_t> tensor_sketch(kmer_word_size, FLAGS_embed_dim, FLAGS_tup_len);
        TensorSlide<uint64_t> tensor_slide(kmer_word_size, FLAGS_embed_dim, FLAGS_tup_len,
                                           FLAGS_win_len, FLAGS_stride);
        std::function<std::vector<double>(const std::vector<uint64_t> &)> sketcher
                = [&](const std::vector<uint64_t> &seq) { return tensor_sketch.compute(seq); };
        std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher
                = [&](const std::vector<uint64_t> &seq) { return tensor_slide.compute(seq); };
        SketchHelper<uint8_t, uint64_t, double> sketch_helper(sketcher, slide_sketcher);
        sketch_helper.read_input();
        FLAGS_sketch_method == "TenSlide" ? sketch_helper.compute_slide()
                                          : sketch_helper.compute_sketches();
        sketch_helper.save_output();
    } else {
        std::cerr << "Unkknown method: " << FLAGS_sketch_method << std::endl;
        exit(1);
    }
}
