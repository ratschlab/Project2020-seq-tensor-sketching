#include <memory>

#include "../include/seq_tools.hpp"
#include "../include/sketch.hpp"
#include <fstream>

// TODO write Google test units

int main() {
    using namespace SeqTools;
    using namespace SeqSketch;
    using namespace Types;

    // sequence generatioin parameters
    bool fix_len = true;
    size_t sig_len = 4, num_seqs = 100, seq_len = 256;
    // embedding parameters
    size_t tup_len = 2, embed_dim = 200, kmer_size = 2, num_phases = 5;
    double mutate_rate = .05, block_mutate = .1;

    // generate sequences
    SeqGen gen;
    Vec<Seq<int>> seqs;
    gen.sig_len = sig_len;
    gen.num_seqs = num_seqs;
    gen.seq_len = seq_len;
    gen.mutation_rate = mutate_rate;
    gen.block_mutate_rate = block_mutate;
    gen.fix_len = fix_len;
    gen.gen_seqs(seqs);

    // transform to kmer_seqs
    Vec<Seq<int>> kmer_seqs(num_seqs);
    for (int si = 0; si < num_seqs; si++) {
        seq2kmer(seqs[si], kmer_seqs[si], kmer_size, sig_len);
    }
    auto ksig_len = int_pow(sig_len, kmer_size);


    // min hash
    MinHashParams MHparams(embed_dim, ksig_len);
    Vec2D<int> MHembed(num_seqs);

    // weighted min hash
    WeightedMinHashParams WMHparams(embed_dim, ksig_len, seq_len * 4);
    Vec2D<int> WMHembed(num_seqs);


    // init OMP embedding
    OMP_Params omp_params;
    omp_params.sig_len = ksig_len;
    omp_params.tup_len = tup_len;
    omp_params.embed_dim = embed_dim;
    omp_params.max_seq_len = seq_len * 2;
    auto perms = omp_rand_perms(omp_params);
    Vec3D<int> omp_embeddings(num_seqs);

    // init tensor embed
    TensorParams tensor_params;
    tensor_params.sig_len = ksig_len;
    tensor_params.tup_len = tup_len;
    tensor_params.embed_dim = embed_dim;
    tensor_params.num_bins = 245;
    tensor_params.num_phases = num_phases;
    tensor_params.rand_init();
    Vec2D<int> tensor_embeddings(num_seqs);

    // init tensor slide embed
    TensorSlideParams tensor_slide_params;
    tensor_slide_params.sig_len = ksig_len;
    tensor_slide_params.tup_len = tup_len;
    tensor_slide_params.num_phases = num_phases;
    tensor_slide_params.num_bins = 64;
    tensor_slide_params.win_len = 32;
    tensor_slide_params.stride = 8;
    tensor_slide_params.embed_dim = embed_dim/tensor_slide_params.stride;
    tensor_slide_params.rand_init();
    Vec3D<int> tensor_slide_embeddings(num_seqs);


    for (int si = 0; si < kmer_seqs.size(); si++) {
        const auto &kseq = kmer_seqs[si];
        minhash(kseq, MHembed[si], MHparams);
        weighted_minhash(kseq, WMHembed[si], WMHparams);
        omp_sketch(kseq, omp_embeddings[si], perms, omp_params);
        tensor_sketch(kseq, tensor_embeddings[si], tensor_params);
        tensor_sketch_slide(kseq, tensor_slide_embeddings[si], tensor_slide_params);
    }

    auto dists = new3D<int>(6, num_seqs, num_seqs, 0);
    for (int i = 0; i < seqs.size(); i++) {
        for (int j = i + 1; j < seqs.size(); j++) {
            dists[0][i][j] = SeqTools::edit_distance(seqs[i], seqs[j]);
            dists[1][i][j] = VecTools::hamming_dist(MHembed[i], MHembed[j]);
            dists[2][i][j] = VecTools::hamming_dist(WMHembed[i], WMHembed[j]);
            dists[3][i][j] = VecTools::hamming_dist2D(omp_embeddings[i], omp_embeddings[j]);
            dists[4][i][j] = VecTools::l1_dist(tensor_embeddings[i], tensor_embeddings[j]);
            dists[5][i][j] = VecTools::l1_dist2D_minlen(tensor_slide_embeddings[i] , tensor_slide_embeddings[j]);
//            dists[5][i][j] = VecTools::l1_dist(tensor_slide_embeddings[i] , tensor_slide_embeddings[j]);
        }
    }

    std::ofstream fo;
    fo.open("output.txt");
    for (int i = 0; i < seqs.size(); i++) {
        for (int j = i + 1; j < seqs.size(); j++) {
            fo << dists[0][i][j] << ", " << dists[1][i][j] << ", " << dists[2][i][j] << ", " << dists[3][i][j] << ", " << dists[4][i][j] << ", " << dists[5][i][j] << "\n";
        }
    }
    fo.close();
}
