#include <memory>

#include "../include/seqgen.hpp"
#include <fstream>

// TODO write Google test units

int main() {
    using namespace SeqTools;
    using namespace SeqSketch;
    using namespace Types;
    // objects

    // global parameters
    size_t sig_len = 2, tup_len = 3, num_seqs = 50, embed_dim = 100, seq_len = 200;
    float mutate_rate = .1;

    // generate sequences
    SeqGen gen;
    Vec<Seq<int>> seqs;
    gen.sig_len = sig_len;
    gen.num_seqs = num_seqs;
    gen.seq_len = seq_len;
    gen.mutation_rate = mutate_rate;
    gen.gen_seqs(seqs);

    // tuple direct embedding
    Vec2D<int> tup_embeddings(num_seqs);

    // init OMP embedding
    OMP_Params omp_params;
    omp_params.sig_len = sig_len;
    omp_params.tup_len = tup_len;
    omp_params.max_seq_len = seq_len*3;
    auto perms = omp_rand_perms(omp_params);
    Vec3D<int> omp_embeddings(num_seqs);

    // init tensor embed
    TensorParams tensor_params;
    tensor_params.sig_len = sig_len;
    tensor_params.tup_len = tup_len;
    tensor_params.rand_init();
    Vec2D<int> tensor_embeddings(num_seqs);

    // init tensor slide embed
    TensorSlideParams tensor_slide_params;
    tensor_slide_params.sig_len = sig_len;
    tensor_slide_params.tup_len = tup_len;
    tensor_slide_params.num_bins = 10;
    tensor_slide_params.win_len = 20;
    tensor_slide_params.stride = 10;
    tensor_slide_params.rand_init();
    Vec3D<int> tensor_slide_embeddings(num_seqs);

    // hierarchical tensor embed
    TensorSlideParams tp1;
    tp1.sig_len = sig_len;
    tp1.tup_len = 3;
    tp1.num_bins = 11;
    tp1.win_len = 4;
    tp1.stride = 1;
    tp1.rand_init();
    TensorSlideParams tp2;
    tp2.sig_len = tp1.num_bins+1;
    tp2.tup_len = 3;
    tp2.num_bins = 11;
    tp2.win_len = 20;
    tp2.stride = 10;
    tp2.rand_init();
    TensorSlideParams tp3;
    tp3.sig_len = tp2.num_bins+1;
    tp3.tup_len = 5;
    tp3.num_bins = 1<<15;
    tp3.win_len = 10;
    tp3.stride = 5;
    tp3.rand_init();
    Vec2D<int> tensor_multi_embed(num_seqs);

    // hierarchical embedding

    for (int si = 0; si < seqs.size(); si++) {
        const auto &seq = seqs[si];
        tup_embed(seq, tup_embeddings[si], sig_len, tup_len);
        omp_sketch(seq, omp_embeddings[si], perms, omp_params);
        tensor_sketch(seq, tensor_embeddings[si], tensor_params);
        tensor_sketch_slide(seq, tensor_slide_embeddings[si], tensor_slide_params);

        Vec2D<int> embed1, embed2, embed3;
        tensor_sketch_slide(seq, embed1, tp1);
        tensor_sketch_slide2(embed1, embed2, tp2);
        tensor_sketch_slide2(embed2, embed3, tp3);
        for (int m=0; m < tp3.embed_dim; m++)
            tensor_multi_embed[si].insert(tensor_multi_embed[si].end(), embed3[m].begin(), embed3[m].begin() + 2);
    }

    Vec3D<int> dists(6,Vec2D<int>(num_seqs, Vec<int>(num_seqs,0)));
    for (int i=0; i<seqs.size(); i++) {
        for (int j=i+1; j<seqs.size(); j++) {
            dists[0][i][j] = SeqTools::edit_distance(seqs[i], seqs[j]);
            dists[1][i][j] = VecTools::l1<int>(tup_embeddings[i] - tup_embeddings[j]);
            dists[2][i][j] = VecTools::hamming_dist2D<int>(omp_embeddings[i] , omp_embeddings[j]);
//            dists[2][i][j] = VecTools::l1_dist2D<int>(omp_embeddings[i] , omp_embeddings[j]);
            dists[3][i][j] = VecTools::l1(tensor_embeddings[i] - tensor_embeddings[j]);
            dists[4][i][j] = VecTools::l1_dist2D_minlen(tensor_slide_embeddings[i] , tensor_slide_embeddings[j]);
            dists[5][i][j] = VecTools::l1(tensor_multi_embed[i] - tensor_multi_embed[j]);
            // TODO write [D]X[D] -> [D] matrix for the cauchy bins l1 distance
            // TODO test if tensor slide embedding = tensor embedding at the end of each window
        }
    }

    std::ofstream fo;
    fo.open("output.txt");
    for (int i=0; i<seqs.size(); i++) {
        for (int j=i+1; j<seqs.size(); j++) {
            fo << dists[0][i][j] << ", " << dists[1][i][j] << ", " << dists[2][i][j] << ", " << dists[3][i][j] << ", " << dists[4][i][j] << ", " << dists[5][i][j] << "\n";
        }
    }
    fo.close();
}
