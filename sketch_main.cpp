#include "sequence/fasta_io.hpp"
#include "sketch/hash_base.hpp"
#include "sketch/hash_min.hpp"
#include "sketch/hash_ordered.hpp"
#include "sketch/hash_weighted.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_slide.hpp"
#include "util/multivec.hpp"
#include "util/utils.hpp"

#include <gflags/gflags.h>

#include <filesystem>
#include <memory>
#include <sstream>

DEFINE_string(alphabet,
              "dna4",
              "The alphabet over which sequences are defined (dna4, dna5, protein)");

DEFINE_string(sketch_method,
              "TensorSlide",
              "The sketching method to use: MH, WMH, OMH, TensorSketch or TensorSlide");
DEFINE_string(m, "TensorSlide", "Short hand for --sketch_method");

DEFINE_uint32(kmer_length, 1, "The kmer length for: MH, WMH, OMH");
DEFINE_uint32(k, 3, "Short hand for --kmer_size");

DEFINE_string(o, "", "Output file, containing the sketches for each sequence");

DEFINE_string(i, "", "Input file, containing the sequences to be sketched in .fa format");

DEFINE_string(input_format, "fasta", "Input format: 'fasta', 'csv'");
DEFINE_string(f, "fasta", "Short hand for --input_format");

DEFINE_int32(embed_dim, 4, "Embedding dimension, used for all sketching methods");

DEFINE_int32(tuple_length,
             3,
             "Ordered tuple length, used in ordered MinHash and Tensor-based sketches");
DEFINE_int32(t, 3, "Short hand for --tuple_length");

DEFINE_int32(window_size, 32, "Window length: the size of sliding window in Tensor Slide Sketch");
DEFINE_int32(w, 32, "Short hand for --window_size");

DEFINE_int32(max_len, 32, "The maximum accepted sequence length for Ordered and Weighted min-hash");

DEFINE_int32(stride, 8, "Stride for sliding window: shift step for sliding window");
DEFINE_int32(s, 8, "Short hand for --stride");

static bool ValidateInput(const char *, const std::string &value) {
    if (!value.empty()) {
        return true;
    }
    std::cerr << "Please specify a fasta input file using '-i <input_file>'" << std::endl;
    return false;
}
DEFINE_validator(i, &ValidateInput);

static bool ValidateOutput(const char *, const std::string &value) {
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
}

using namespace ts;

template <typename seq_type, class kmer_type, class embed_type>
class SketchHelper {
  public:
    SketchHelper(
            const std::function<std::vector<embed_type>(const std::vector<kmer_type> &)> &sketcher,
            const std::function<Vec2D<double>(const std::vector<uint64_t> &)> &slide_sketcher)
        : sketcher(sketcher), slide_sketcher(slide_sketcher) {}

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
        std::tie(seqs, seq_names) = read_fasta<seq_type>(FLAGS_i, FLAGS_input_format);
    }

    void save_output() {
        std::filesystem::path ofile(FLAGS_o);
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

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    adjust_short_names();

    init_alphabet(FLAGS_alphabet);

    if (std::pow(alphabet_size, FLAGS_kmer_length) > (double)std::numeric_limits<uint64_t>::max()) {
        std::cerr << "Kmer size is too large to fit in 64 bits " << std::endl;
        std::exit(1);
    }

    uint64_t kmer_word_size = int_pow<uint64_t>(alphabet_size, FLAGS_kmer_length);

    if (FLAGS_sketch_method.substr(FLAGS_sketch_method.size() - 2, 2) == "MH") {
        std::function<std::vector<uint64_t>(const std::vector<uint64_t> &)> sketcher;

        if (FLAGS_sketch_method == "MH") {
            // The hash function is part of the lambda state.
            sketcher = [&,
                        min_hash = MinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim,
                                                     HashAlgorithm::uniform)](
                               const std::vector<uint64_t> &seq) mutable {
                return min_hash.compute(seq);
            };
        } else if (FLAGS_sketch_method == "WMH") {
            sketcher = [&,
                        wmin_hash
                        = WeightedMinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim, FLAGS_max_len,
                                                    HashAlgorithm::uniform)](
                               const std::vector<uint64_t> &seq) mutable {
                return wmin_hash.compute(seq);
            };
        } else if (FLAGS_sketch_method == "OMH") {
            sketcher = [&,
                        omin_hash
                        = OrderedMinHash<uint64_t>(kmer_word_size, FLAGS_embed_dim, FLAGS_max_len,
                                                   FLAGS_tuple_length, HashAlgorithm::uniform)](
                               const std::vector<uint64_t> &seq) mutable {
                return omin_hash.compute(seq);
            };
        }
        std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher
                = [&](const std::vector<uint64_t> &) { return new2D<double>(0, 0); };
        SketchHelper<uint8_t, uint64_t, uint64_t> sketch_helper(sketcher, slide_sketcher);
        sketch_helper.read_input();
        sketch_helper.compute_sketches();
        sketch_helper.save_output();
    } else if (FLAGS_sketch_method.rfind("Tensor", 0) == 0) {
        Tensor<uint64_t> tensor_sketch(kmer_word_size, FLAGS_embed_dim, FLAGS_tuple_length);
        TensorSlide<uint64_t> tensor_slide(kmer_word_size, FLAGS_embed_dim, FLAGS_tuple_length,
                                           FLAGS_window_size, FLAGS_stride);
        std::function<std::vector<double>(const std::vector<uint64_t> &)> sketcher
                = [&](const std::vector<uint64_t> &seq) { return tensor_sketch.compute(seq); };
        std::function<Vec2D<double>(const std::vector<uint64_t> &)> slide_sketcher
                = [&](const std::vector<uint64_t> &seq) { return tensor_slide.compute(seq); };
        SketchHelper<uint8_t, uint64_t, double> sketch_helper(sketcher, slide_sketcher);
        sketch_helper.read_input();
        FLAGS_sketch_method == "TensorSlide" ? sketch_helper.compute_slide()
                                             : sketch_helper.compute_sketches();
        sketch_helper.save_output();
    } else {
        std::cerr << "Unkknown method: " << FLAGS_sketch_method << std::endl;
        exit(1);
    }
}
