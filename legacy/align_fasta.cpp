#include <fstream>
#include <memory>


#include "util/modules.hpp"
#include "util/seqgen.hpp"
#include "util/utils.hpp"

using namespace ts;
using namespace BasicTypes;

struct KmerModule : public BasicModules {
    int original_alphabet_size {};

    void override_pre() override {
        alphabet_size = 5;
        original_alphabet_size = alphabet_size;
        alphabet_size = int_pow<size_t>(alphabet_size, kmer_size);
    }

    void override_post() override {
        //        tensor_slide_params.alphabet_size = original_alphabet_size;
        //        tensor_slide_params.tup_len = 2;
        tensor_slide_params.embed_dim = 50;
        tensor_slide_params.num_bins = 250;
    }
};

struct HGModule {
    Vec3D<int> dists;
    std::ifstream infile;

    std::map<char, int> chr2int = { { 'a', 1 }, { 'c', 2 }, { 'g', 3 }, { 't', 4 }, { 'n', 0 },
                                    { 'A', 1 }, { 'C', 2 }, { 'G', 3 }, { 'T', 4 }, { 'N', 0 } };
    std::map<char, int> chr2int_mask
            = { { 'a', -1 }, { 'c', -2 }, { 'g', -3 }, { 't', -4 }, { 'n', 0 },
                { 'A', 1 },  { 'C', 2 },  { 'G', 3 },  { 'T', 4 },  { 'N', 0 } };

    BasicModules basicModules;
    KmerModule kmerModules;

    void parse(int argc, char **argv) {
        basicModules.parse(argc, argv);
        basicModules.alphabet_size = 5;
        basicModules.models_init();
        kmerModules.parse(argc, argv);
        kmerModules.models_init();
    }


    string read_first() {
        string hg_file = "data/sub2.fa";
        infile = std::ifstream(hg_file);
        string line;
        std::getline(infile, line);
        return line;
    }

    template <typename seq_type>
    string read_next_seq(std::vector<seq_type> &seq, std::vector<bool> mask) {
        seq.clear();
        string line;
        while (std::getline(infile, line)) {
            if (line[0] == '>') {
                return line;
            } else {
                for (char c : line) {
                    seq.push_back(chr2int[c]);
                    mask.push_back((chr2int[c] > 0));
                }
            }
        }
        return "";
    }

    void compute_sketches() {
        Vec2D<int> slide_sketch;
        std::vector<int> seq, kmer_seq;
        std::vector<bool> mask;
        string name = read_first(), next_name;
        while (not name.empty()) {
            next_name = read_next_seq(seq, mask);
            seq2kmer(seq, kmer_seq, basicModules.kmer_size, basicModules.alphabet_size);
            tensor_slide_sketch(kmer_seq, slide_sketch, kmerModules.tensor_slide_params);
            save_output(name, slide_sketch);
            name = next_name;
        }
    }


    void save_output(string seq_name, const Vec2D<int> &sketch) {
        std::ofstream fo;
        seq_name = string("data/sketch_") + seq_name.substr(1) + "_" + std::to_string(sketch.size())
                + "_" + std::to_string(sketch[0].size()) + ".txt";
        fo.open(seq_name);
        //        fo << sketch.size() << ", " << sketch[0].size() << "\n";
        for (int m = 0; m < sketch.size(); m++) {
            for (int i = 0; i < sketch[m].size(); i++) {
                fo << sketch[m][i] << ",";
            }
            fo << "\n";
        }
        fo.close();
    }
};

int main(int argc, char *argv[]) {
    HGModule experiment;
    experiment.parse(argc, argv);
    experiment.compute_sketches();
    //    experiment.save_output();
}
