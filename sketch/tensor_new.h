#pragma once

namespace ts { // ts = Tensor Sketch


struct Tensor2Params {
    int alphabet_size;
    int embed_dim;
    int num_phases;
    int tup_len;
    int hash_len;

    Vec2D<int> hash;

    virtual void rand_init() {
        hash_len = num_phases * embed_dim;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> rand_hash2(0, hash_len - 1);

        hash = new2D<int>(tup_len, alphabet_size);
        for (int h = 0; h < tup_len; h++) {
            for (int c = 0; c < alphabet_size; c++) {
                hash[h][c] = rand_hash2(gen);
            }
        }
    }
};


template <class T>
void shift_sum(Vec<T> &a, const Vec<T> &b, int sh, double z) {
    assert(a.size() == b.size());
    int len = a.size();
    for (int i = 0; i < a.size(); i++) {
        a[i] = (1 - z) * a[i] + z * b[(i - sh + len) % len];
    }
}

template <class seq_type, class embed_type>
void tensor2_sketch(const Seq<seq_type> &seq,
                    Vec<embed_type> &embedding,
                    const Tensor2Params &params) {
    auto M = new2D<double>(params.tup_len + 1, params.hash_len, 0);
    M[0][0] = 1;
    for (int i = 0; i < seq.size(); i++) {
        for (int t = params.tup_len - 1; t >= 0; t--) {
            double z = (double)(t + 1) / (i + 1);
            auto r = params.hash[t][seq[i]];
            shift_sum(M[t + 1], M[t], r, z);
        }
    }
    embedding = Vec<embed_type>(params.embed_dim, 0);
    for (int m = 0; m < params.embed_dim; m++) {
        embed_type prod = 0;
        for (int r = 0; r < params.num_phases; r++) {
            prod += ((r % 2 == 0) ? 1 : -1) * M[params.tup_len][m * params.num_phases + r];
        }
        int exp;
        frexp(prod, &exp);
        embedding[m] = exp * sgn(prod);
        embedding[m] = prod;
    }
}

struct Tensor2_slide_Params : Tensor2Params {
    int win_len;
    int stride;
};

template <class T>
void conv_sum(Vec<T> &a, const Vec<T> &b, double z) {
    assert(a.size() == b.size());
    int len = a.size;
    Vec<T> result(len, 0);
    for (int i = 0; i < len; i++) {
        T val = 0;
        for (int j = 0; j < len; j++) {
            val += a[(i + j) % len] * b[(i - j + len) % len];
        }
        a[i] = (1 - z) * a[i] + z * val;
    }
    std::swap(a, result);
}

template <class seq_type, class embed_type>
void tensor2_slide_sketch(const Vec<seq_type> &seq,
                          Vec2D<embed_type> &embedding,
                          const Tensor2_slide_Params &params) {
    auto M = new3D<double>(params.tup_len + 1, params.tup_len + 1, params.hash_len, 0);
    for (int p = 0; p < params.tup_len; p++) {
        M[p + 1][p][0] = 1;
    }

    for (int i = 0; i < seq.size(); i++) {
        for (int p = 0; p < params.tup_len; p++) {
            for (int q = params.tup_len - 1; q >= p; q--) {
                double z = (double)(q - p + 1) / std::min(i + 1, params.win_len);
                auto r = params.hash[q][seq[i]];
                shift_sum(M[p + 1][q + 1], M[p + 1][q], r, z);
            }
        }

        if ((i + 1) % params.stride == 0) {
            Vec<embed_type> em(params.embed_dim);
            for (int m = 0; m < params.embed_dim; m++) {
                embed_type prod = 0;
                for (int r = 0; r < params.num_phases; r++) {
                    prod += ((r % 2 == 0) ? 1 : -1)
                            * M[1][params.tup_len][m * params.num_phases + r];
                }
                em[m] = prod;
            }
            embedding.push_back(em);
        }

        if (i < params.win_len) {
            continue;
        }

        for (int p = 0; p < params.tup_len; p++) {
            for (int q = p; q < params.tup_len; q++) {
                double z = (double)(q - p + 1) / (params.win_len - q + p);
                auto r = params.hash[q][seq[i]];
                shift_sum(M[p + 1][q + 1], M[p + 1][q], r, -z);
            }
        }
    }
}

template <class seq_type, class embed_type>
void conv_slide_sketch(const Vec2D<seq_type> &seq,
                       Vec2D<embed_type> &embedding,
                       const Tensor2_slide_Params &params) {
    auto M = new3D<double>(params.tup_len + 1, params.tup_len + 1, params.hash_len, 0);
    for (int p = 0; p < params.tup_len; p++) {
        M[p + 1][p][0] = 1;
    }

    for (int i = 0; i < seq.size(); i++) {
        assert(seq[i].size() == params.alphabet_size);
        for (int p = 0; p < params.tup_len; p++) {
            for (int q = params.tup_len - 1; q >= p; q--) {
                double z = (double)(q - p + 1) / std::min(i + 1, params.win_len);
                auto r = params.hash[q][seq[i]];
                shift_sum(M[p + 1][q + 1], M[p + 1][q], r, z);
            }
        }

        if ((i + 1) % params.stride == 0) {
            Vec<embed_type> em(params.embed_dim);
            for (int m = 0; m < params.embed_dim; m++) {
                embed_type prod = 0;
                for (int r = 0; r < params.num_phases; r++) {
                    prod += ((r % 2 == 0) ? 1 : -1)
                            * M[1][params.tup_len][m * params.num_phases + r];
                }
                em[m] = prod;
            }
            embedding.push_back(em);
        }

        if (i < params.win_len) {
            continue;
        }

        for (int p = 0; p < params.tup_len; p++) {
            for (int q = p; q < params.tup_len; q++) {
                double z = (double)(q - p + 1) / (params.win_len - q + p);
                auto r = params.hash[q][seq[i]];
                shift_sum(M[p + 1][q + 1], M[p + 1][q], r, -z);
            }
        }
    }
}


} // namespace ts
