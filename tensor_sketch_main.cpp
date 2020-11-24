#include <fstream>
#include <memory>

#include "util/args.hpp"
#include "util/modules.hpp"
#include "util/seqgen.h"
#include "util/utils.h"
#include <sstream>

using namespace ts;

template <typename seq_type, class embed_type>
struct SketchModule : public BasicModule {
    int original_sig_len {};

    void override_module_params() override {
        original_sig_len = sig_len;
        sig_len = int_pow<size_t>(sig_len, kmer_size);
        wmh_params.max_len = win_len;
        omh_params.max_len = win_len;
    }

    void override_model_params() override { tensor_slide_params.embed_dim = embed_dim; }

    SketchModule() {
        directory = "./";
        output = "data/sketches/";
        input = "data/fasta/seqs.fa";
        sig_len = 5;
        kmer_size = 1;
        embed_dim = 128;
        tup_len = 2;
        num_phases = 2;
        num_bins = 255;
        win_len = 32;
        stride = 8;
        offset = 0;
        show_help = false;
        sketch_method = "TenSlide";
    }

    Vec2D<seq_type> seqs;
    Vec<std::string> seq_names;
    string test_id;


    std::map<char, seq_type> chr2int
            = { { 'a', 1 }, { 'c', 2 }, { 'g', 3 }, { 't', 4 }, { 'n', 0 },
                { 'A', 1 }, { 'C', 2 }, { 'G', 3 }, { 'T', 4 }, { 'N', 0 } };
    std::map<char, seq_type> chr2int_mask
            = { { 'a', -1 }, { 'c', -2 }, { 'g', -3 }, { 't', -4 }, { 'n', 0 },
                { 'A', 1 },  { 'C', 2 },  { 'G', 3 },  { 'T', 4 },  { 'N', 0 } };
    void read_fasta() {
        seqs.clear();
        string file = (directory + input);
        std::ifstream infile = std::ifstream(file);
        assert(infile.is_open());

        string line;
        std::getline(infile, line);
        if (line[0] == '#') {
            test_id = line;
            std::getline(infile, line);
        }
        while (line[0] != '>') {
            std::getline(infile, line);
        }
        string name = line;
        Vec<seq_type> seq;
        while (std::getline(infile, line)) {
            if (line[0] == '>') {
                seqs.push_back(seq);
                seq_names.push_back(name);
                seq.clear();
                name = line;
            } else if (!line.empty()) {
                if (format_input == "fasta") {
                    for (char c : line) {
                        seq.push_back(chr2int[c]);
                    }
                } else if (format_input == "csv") {
                    std::stringstream ss(line);
                    string item;
                    while (std::getline(ss, item, ',')) {
                        seq.push_back(std::stoi(item, 0, 16));
                    }
                } else {
                    std::cerr << " input format `" << format_input << "` does not exist\n";
                    exit(1);
                }
            }
        }
    }

    void sketch_slice(Seq<seq_type> seq, Vec<embed_type> &embed) {
        if (sketch_method == "MH") {
            minhash(seq, embed, mh_params);
        } else if (sketch_method == "WMH") {
            weighted_minhash(seq, embed, wmh_params);
        } else if (sketch_method == "OMH") {
            ordered_minhash_flat(seq, embed, omh_params);
        } else if (sketch_method == "TenSketch") {
            tensor_sketch(seq, embed, tensor_params);
        } else {
            std::cerr << "method not recognized\n";
            exit(1);
        }
    }

    Vec3D<embed_type> slide_sketch;
    void compute_sketches() {
        //        Vec<string> args = {"MH", "WMH", "OMH", "TenSketch", "TenSlide"};
        int num_seqs = seqs.size();
        slide_sketch = new3D<embed_type>(seqs.size(), embed_dim, 0);
        for (int si = 0; si < num_seqs; si++) {
            Vec<seq_type> kmers;
            seq2kmer(seqs[si], kmers, kmer_size, original_sig_len);
            if (sketch_method == "TenSlide") {
                tensor_slide_sketch(kmers, slide_sketch[si], tensor_slide_params);
            } else {
                for (int i = offset; i < sketch_end(offset, kmers.size()); i += stride) {
                    Vec<embed_type> embed_slice;
                    Vec<seq_type> kmer_slice(kmers.begin() + i,
                                             std::min(kmers.begin() + i + win_len, kmers.end()));
                    sketch_slice(kmer_slice, embed_slice);
                    for (int m = 0; m < embed_dim; m++) {
                        slide_sketch[si][m].push_back(embed_slice[m]);
                    }
                }
            }
        }
    }


    void save_output() {
        std::ofstream fo;
        fo.open(directory + output + sketch_method + ".txt");
        if (!fo.is_open()) {
            std::cerr << "otuput file not opened\n";
        }
        fo << test_id << "\n";
        fo << "# " << argvals() << "\n";
        for (size_t si = 0; si < slide_sketch.size(); si++) {
            for (size_t m = 0; m < slide_sketch[si].size(); m++) {
                fo << seq_names[si] << ">" << std::dec << m << "\n";
                for (size_t i = 0; i < slide_sketch[si][m].size(); i++) {
                    if (num_bins == 0) {
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
};

int main(int argc, char *argv[]) {
    SketchModule<int, double> sketchModule;
    sketchModule.parse(argc, argv);
    sketchModule.models_init();
    sketchModule.read_fasta();
    sketchModule.compute_sketches();
    sketchModule.save_output();
}
