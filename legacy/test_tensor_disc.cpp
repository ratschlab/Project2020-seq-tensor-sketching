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
        tensor_slide_params.alphabet_size = original_alphabet_size;
        tensor_slide_params.tup_len = 2;
    }
};

struct DiscModules : public BasicModules {
    std::vector<int> dims = { 16, 64, 256 };
    std::vector<int> win2stride = { 2, 2, 2 };
    std::vector<int> tup_lens = { 4, 4, 4 };
    std::vector<int> num_phases = { 5, 5, 5 };
    std::vector<int> strid2dim = { 2, 2, 2 };

    void override_post() override {}

    TensorSlideParams layer0() {
        TensorSlideParams tensorParams;
        init_tensor_slide_params(tensorParams);
        tensorParams.tup_len = tup_lens[0];
        tensorParams.win_len = dims[0];
        tensorParams.stride = dims[0] / strid2dim[0];
        tensorParams.embed_dim = dims[0];
        tensorParams.num_phases = num_phases[0];
        return tensorParams;
    }

    std::vector<TensorSlideParams> layers(int l) {
        assert(0 < l and l <= 2);
        std::vector<TensorSlideParams> param_vec;
        for (int i = 0; i < dims[l - 1]; i++) {
            auto params = layer0();
            params.tup_len = tup_lens[l];
            params.alphabet_size = num_phases[l];
            params.num_phases = num_phases[l];
            params.embed_dim = dims[l] / dims[l - 1];
            params.stride = (dims[l] / dims[l - 1]);
            params.win_len = params.stride * win2stride[l];
            param_vec.push_back(params);
        }
        return param_vec;
    }
};

struct TestModule1 {
    Vec2D<int> seqs;
    Vec2D<int> kmer_seqs;
    Vec2D<int> wmh_sketch;
    Vec2D<int> mh_sketch;
    Vec3D<int> omh_sketch;
    Vec2D<int> ten_sketch;
    Vec3D<double> ten_disc_sketch;
    Vec4D<double> slide_disc_sketch1;
    Vec4D<double> slide_disc_sketch2;
    Vec4D<double> slide_disc_sketch3;
    Vec2D<double> slide_disc_flat;
    Vec3D<int> slide_sketch;
    Vec3D<double> dists;

    BasicModules basicModules;
    KmerModule kmerModules;
    DiscModules discModules;

    void parse(int argc, char **argv) {
        basicModules.parse(argc, argv);
        basicModules.models_init();
        kmerModules.parse(argc, argv);
        kmerModules.models_init();
        discModules.parse(argc, argv);
        discModules.models_init();
    }

    void generate_sequences() { basicModules.seq_gen.gen_seqs(seqs); }

    void compute_sketches() {
        int num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ten_sketch.resize(num_seqs);
        ten_disc_sketch.resize(num_seqs);
        slide_disc_sketch1.resize(num_seqs);
        slide_disc_sketch2.resize(num_seqs);
        slide_disc_sketch3.resize(num_seqs);
        slide_disc_flat.resize(num_seqs);
        slide_sketch.resize(num_seqs);
        auto lay0 = discModules.layer0();
        lay0.rand_init();
        auto lay1 = discModules.layers(1);
        for (auto &l : lay1)
            l.rand_init();
        auto lay2 = discModules.layers(2);
        for (auto &l : lay2)
            l.rand_init();
        for (int si = 0; si < num_seqs; si++) {
            seq2kmer(seqs[si], kmer_seqs[si], basicModules.kmer_size, basicModules.alphabet_size);
            minhash(kmer_seqs[si], mh_sketch[si], kmerModules.mh_params);
            weighted_minhash(kmer_seqs[si], wmh_sketch[si], kmerModules.wmh_params);
            ordered_minhash(kmer_seqs[si], omh_sketch[si], kmerModules.omh_params);
            //            tensor_sketch(seqs[si], ten_sketch[si], longseqModule.tensor_params);
            //            tensor_disc_sketch<int, double>(seqs[si], ten_disc_sketch[si],
            //            discModules.tensor_params); Vec3D<double> in, out;
            slide_disc_sketch1[si] = tensor_disc_slide<int, double>(seqs[si], lay0);
            slide_disc_sketch2[si]
                    = tensor_disc_slide3<double, double>(slide_disc_sketch1[si], lay1);
            slide_disc_sketch3[si]
                    = tensor_disc_slide3<double, double>(slide_disc_sketch2[si], lay2);
            slide_disc_flat[si] = squeeze_tensor(slide_disc_sketch3[si]);
            tensor_slide_sketch(seqs[si], slide_sketch[si], kmerModules.tensor_slide_params);
        }
    }
    void compute_dists() {
        int num_seqs = seqs.size();
        dists = new3D<double>(7, num_seqs, num_seqs, 0);
        for (int i = 0; i < seqs.size(); i++) {
            for (int j = i + 1; j < seqs.size(); j++) {
                dists[0][i][j] = edit_distance(seqs[i], seqs[j]);
                dists[1][i][j] = hamming_dist(mh_sketch[i], mh_sketch[j]);
                dists[2][i][j] = hamming_dist(wmh_sketch[i], wmh_sketch[j]);
                dists[3][i][j] = hamming_dist2D(omh_sketch[i], omh_sketch[j]);
                dists[4][i][j] = l1_dist(ten_sketch[i], ten_sketch[j]);
                dists[5][i][j] = l1_dist(slide_disc_flat[i], slide_disc_flat[j]);
                dists[6][i][j] = l1_dist2D_minlen(slide_sketch[i], slide_sketch[j]);
            }
        }
    }

    void save_output() {
        std::ofstream fo;
        fo.open("output.txt");
        for (int i = 0; i < seqs.size(); i++) {
            for (int j = i + 1; j < seqs.size(); j++) {
                fo << dists[0][i][j] << ", " << dists[1][i][j] << ", " << dists[2][i][j] << ", "
                   << dists[3][i][j] << ", " << dists[4][i][j] << ", " << dists[5][i][j] << ", "
                   << dists[6][i][j] << "\n";
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
