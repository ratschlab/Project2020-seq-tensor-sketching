#include <memory>
#include <fstream>

#include "common.hpp"
#include "modules.hpp"
#include "seq_tools.hpp"
#include "sketch.hpp"


using namespace Sketching;
using namespace Types;

struct TestModule1 {

    SeqGen seq_gen;
    Vec2D<int> seqs;
    Vec2D<int> kmer_seqs;
    Vec2D<int> wmh_sketch;
    Vec2D<int> mh_sketch;
    Vec3D<int> omh_sketch;
    Vec2D<int> t_sketch;
    Vec3D<int> tss_sketch;
    Vec3D<int> dists;

    BasicModules &basicModules;
    KmerModules &kmerModules;

    TestModule1(BasicModules &bm, KmerModules &km) : basicModules(bm), kmerModules(km) {}

    void generate_sequences() {
        basicModules.init_seqgen(seq_gen);
        seq_gen.gen_seqs(seqs);
    }

    void compute_sketches() {
        int num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        t_sketch.resize(num_seqs);
        tss_sketch.resize(num_seqs);
        for (int si = 0; si < num_seqs; si++) {
            seq2kmer(seqs[si], kmer_seqs[si], basicModules.kmer_size, basicModules.sig_len);
            minhash(kmer_seqs[si], mh_sketch[si], kmerModules.mh_params);
            weighted_minhash(kmer_seqs[si], wmh_sketch[si], kmerModules.wmh_params);
            ordered_minhash(kmer_seqs[si], omh_sketch[si], kmerModules.omh_params);
            tensor_sketch(kmer_seqs[si], t_sketch[si], kmerModules.tensor_params);
            tensor_sketch_slide(seqs[si], tss_sketch[si], basicModules.tensor_slide_params);
        }
    }
    void compute_dists() {
        int num_seqs = seqs.size();
        dists = new3D<int>(6, num_seqs, num_seqs, 0);
        for (int i = 0; i < seqs.size(); i++) {
            for (int j = i + 1; j < seqs.size(); j++) {
                dists[0][i][j] = edit_distance(seqs[i], seqs[j]);
                dists[1][i][j] = hamming_dist(mh_sketch[i], mh_sketch[j]);
                dists[2][i][j] = hamming_dist(wmh_sketch[i], wmh_sketch[j]);
                dists[3][i][j] = hamming_dist2D(omh_sketch[i], omh_sketch[j]);
                dists[4][i][j] = l1_dist(t_sketch[i], t_sketch[j]);
                dists[5][i][j] = l1_dist2D_minlen(tss_sketch[i], tss_sketch[j]);
            }
        }
    }

    void save_output() {
        std::ofstream fo;
        fo.open("output.txt");
        //    fo << to_string(argc, argv) << "\n";
        for (int i = 0; i < seqs.size(); i++) {
            for (int j = i + 1; j < seqs.size(); j++) {
                fo << dists[0][i][j] << ", " << dists[1][i][j] << ", " << dists[2][i][j] << ", " << dists[3][i][j] << ", " << dists[4][i][j] << ", " << dists[5][i][j] << "\n";
            }
        }
        fo.close();
    }
};

int main(int argc, char *argv[]) {
    BasicModules bm(argc, argv);
    KmerModules km(argc, argv);
    km.models_init();
    bm.models_init();
    TestModule1 experiment(bm, km);
    experiment.generate_sequences();
    experiment.compute_sketches();
    experiment.compute_dists();
    experiment.save_output();

}
