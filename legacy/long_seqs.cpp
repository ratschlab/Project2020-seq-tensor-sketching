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
        original_alphabet_size = alphabet_size;
        alphabet_size = int_pow<size_t>(alphabet_size, kmer_size);
    }

    void override_post() override {
        //        tensor_slide_params.alphabet_size = original_alphabet_size;
        //        tensor_slide_params.tup_len = 2;
    }
};

struct TestModule1 {
    Vec2D<int> seqs;
    Vec2D<int> kmer_seqs;
    Vec2D<int> wmh_sketch;
    Vec2D<int> mh_sketch;
    Vec3D<int> omh_sketch;
    Vec2D<int> ten_sketch;
    Vec3D<int> slide_sketch;
    Vec3D<int> dists;

    BasicModules basicModules;
    KmerModule kmerModules;

    void parse(int argc, char **argv) {
        basicModules.parse(argc, argv);
        basicModules.models_init();
        kmerModules.parse(argc, argv);
        kmerModules.models_init();
    }

    void generate_sequences() { basicModules.seq_gen.gen_seqs(seqs); }

    void compute_sketches() {
        int num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        slide_sketch.resize(num_seqs);
        for (int si = 0; si < num_seqs; si++) {
            seq2kmer(seqs[si], kmer_seqs[si], basicModules.kmer_size, basicModules.alphabet_size);
            tensor_slide_sketch(kmer_seqs[si], slide_sketch[si], kmerModules.tensor_slide_params);
        }
    }
    void compute_dists() {
        int num_seqs = seqs.size();
        dists = new3D<int>(2, num_seqs, num_seqs, 0);
        for (int i = 0; i < seqs.size(); i++) {
            for (int j = i + 1; j < seqs.size(); j++) {
                dists[0][i][j] = edit_distance(seqs[i], seqs[j]);
                dists[1][i][j] = l1_dist2D_minlen(slide_sketch[i], slide_sketch[j]);
            }
        }
    }

    void save_output() {
        std::ofstream fo;
        fo.open("long_seq_output.txt");
        for (int i = 0; i < seqs.size(); i++) {
            for (int j = i + 1; j < seqs.size(); j++) {
                fo << dists[0][i][j] << ", " << dists[1][i][j] << "\n";
            }
        }
        fo.close();
    }
};

int main(int argc, char *argv[]) {
    TestModule1 experiment;
    experiment.parse(argc, argv);
    experiment.generate_sequences();
    experiment.compute_sketches();
    experiment.compute_dists();
    experiment.save_output();
}
