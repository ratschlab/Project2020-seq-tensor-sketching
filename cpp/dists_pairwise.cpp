#include <memory>

#include "../include/args.hpp"
#include "../include/seq_tools.hpp"
#include "../include/sketch.hpp"
#include <fstream>

// TODO write Google test units


struct dists_arg_group : public tensor_embed_args, public kmer_args, public seq_args {
    int ksig_len;
    dists_arg_group(int argc, char* argv[])  : tensor_embed_args(argc, argv),
                                       kmer_args(argc, argv),
                                       seq_args(argc, argv){
       ksig_len = VecTools::int_pow<size_t>(sig_len, kmer_size);
    }
};

int main(int argc, char* argv[]) {
    using namespace SeqSketch;
    using namespace Types;

    dists_arg_group opts(argc, argv);


// generate sequences
    SeqSketch::SeqGen gen(opts);
    Vec<Seq<int>> seqs;
    gen.gen_seqs(seqs);

    // transform to kmer_seqs
    Vec<Seq<int>> kmer_seqs(opts.num_seqs);
    for (int si = 0; si < opts.num_seqs; si++) {
        seq2kmer(seqs[si], kmer_seqs[si], opts.kmer_size, opts.sig_len);
    }


    // min hash
    MinHashParams MHparams(opts.embed_dim, opts.ksig_len);
    Vec2D<int> MHembed(opts.num_seqs);

    // weighted min hash
    WeightedMinHashParams WMHparams(opts.embed_dim, opts.ksig_len, opts.seq_len * 2);
    Vec2D<int> WMHembed(opts.num_seqs);


    // init OMP embedding
    OMP_Params omp_params(argc, argv);
    omp_params.sig_len = opts.ksig_len;
    omp_params.init_rand();
    Vec3D<int> omp_embeddings(opts.num_seqs);

    // init tensor embed
    TensorParams tensor_params(argc, argv);
    tensor_params.sig_len = opts.ksig_len;
    tensor_params.rand_init();
    Vec2D<int> tensor_embeddings(opts.num_seqs);

    // init tensor slide embed
    TensorSlideParams tensor_slide_params(argc, argv);
    tensor_slide_params.num_bins = 64;
    tensor_slide_params.sig_len = opts.ksig_len;
    tensor_slide_params.embed_dim = opts.embed_dim/tensor_slide_params.stride;
    tensor_slide_params.rand_init();
    Vec3D<int> tensor_slide_embeddings(opts.num_seqs);


    for (int si = 0; si < kmer_seqs.size(); si++) {
        const auto &kseq = kmer_seqs[si];
        minhash(kseq, MHembed[si], MHparams);
        weighted_minhash(kseq, WMHembed[si], WMHparams);
        omp_sketch(kseq, omp_embeddings[si], omp_params);
        tensor_sketch(kseq, tensor_embeddings[si], tensor_params);
        tensor_sketch_slide(kseq, tensor_slide_embeddings[si], tensor_slide_params);
    }

    auto dists = new3D<int>(6, opts.num_seqs, opts.num_seqs, 0);
    for (int i = 0; i < seqs.size(); i++) {
        for (int j = i + 1; j < seqs.size(); j++) {
            dists[0][i][j] = SeqSketch::edit_distance(seqs[i], seqs[j]);
            dists[1][i][j] = VecTools::hamming_dist(MHembed[i], MHembed[j]);
            dists[2][i][j] = VecTools::hamming_dist(WMHembed[i], WMHembed[j]);
            dists[3][i][j] = VecTools::hamming_dist2D(omp_embeddings[i], omp_embeddings[j]);
            dists[4][i][j] = VecTools::l1_dist(tensor_embeddings[i], tensor_embeddings[j]);
            dists[5][i][j] = VecTools::l1_dist2D_minlen(tensor_slide_embeddings[i] , tensor_slide_embeddings[j]);
        }
    }

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
