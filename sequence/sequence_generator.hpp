#pragma once

#include "util/utils.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <random>

namespace ts { // ts = Tensor Sketch

class SeqGen {
  public:
    SeqGen(uint8_t alphabet_size,
           bool fix_len,
           uint32_t max_num_blocks,
           uint32_t min_num_blocks,
           uint32_t num_seqs,
           uint32_t seq_len,
           uint32_t  group_size,
           double mutation_rate,
           double min_mutation_rate,
           double block_mutation_rate,
           std::string mutation_type)
        : alphabet_size(alphabet_size),
          fix_len(fix_len),
          max_num_blocks(max_num_blocks),
          min_num_blocks(min_num_blocks),
          num_seqs(num_seqs),
          seq_len(seq_len),
          group_size(group_size),
          mutation_rate(mutation_rate),
          min_mutation_rate(min_mutation_rate),
          block_mutate_rate(block_mutation_rate),
          mutation_type(mutation_type) {}


    template <class T>
    void generate_path(Vec2D<T> &seqs) {
        seqs.resize(num_seqs + group_size);
        for (size_t gi=0; gi<num_seqs; gi += group_size) {
            random_sequence(seqs[gi], seq_len);
            for (size_t i=0; i<group_size-1; i++) {
                mutate(seqs[gi+i], seqs[gi+i+1]);
            }
        }
        seqs.resize(num_seqs);
    }


    template <class T>
    void generate_tree(Vec2D<T> seqs) {
        seqs.reserve(num_seqs);
        while (seqs.size() < num_seqs) {
            Vec2D<T> group(1), children;
            random_sequence(group[0], seq_len);
            while (group.size() < group_size) {
                children.clear();
                for (auto &seq : group) {
                    std::vector<T> ch;
                    mutate(seq, ch);
                    children.push_back(seq);
                    children.push_back(ch);
                }
                std::swap(group, children);
            }
            group.resize(group_size);
            seqs.insert(seqs.end(), group.begin(), group.end());
        }
        seqs.resize(num_seqs);
    }

  private:

    template <class T>
    void mutate(const std::vector<T> &ref, std::vector<T> &seq) {
        std::uniform_real_distribution<double> unif(min_mutation_rate, mutation_rate);
        if (mutation_type == "edit") {
            mutate_edits(ref, seq, unif(gen));
        } else if (mutation_type == "rate") {
            mutate_rate(ref, seq, unif(gen));
        } else {
            exit(1);
        }
        mutate_block(seq);
        if (fix_len)
            make_fix_len(seq);
    }


    /**
     * mutate seq from ref, by mutating each position with the probability = `rate`
     * @tparam T : character type
     * @param ref : reference
     * @param seq : mutated sequence
     * @param rate : probability of mutation at each index
     */
    template <class T>
    void mutate_rate(const std::vector<T> &ref, std::vector<T> &seq, double rate) {
        assert((rate>=0.0) && (rate<= 1.0) && " rate must be strictly in the range [0,1]");
        // probabilities for each index position: no mutation, insert, delete, substitute
        std::discrete_distribution<int> mut { 1 - rate, rate / 3, rate / 3, rate / 3 };
        // the range chosen such that (sub_char+ref % alphabet_size) will different from ref
        std::uniform_int_distribution<T> sub_char(1, alphabet_size - 1);
        // random character from the alphabet
        std::uniform_int_distribution<T> rand_char(0, alphabet_size - 1);
        for (size_t i = 0; i < ref.size(); i++) {
            switch (mut(gen)) {
                case 0: { // no mutation
                    seq.push_back(ref[i]);
                    break;
                }
                case 1: { // insert
                    seq.push_back(rand_char(gen));
                    i--; // init_tensor_slide_params negate the increment
                    break;
                }
                case 2: { // delete
                    break;
                }
                case 3: { // substitute
                    seq.push_back( (sub_char(gen) + ref[i]) % alphabet_size);
                    break;
                }
            }
        }
    }


    /**
     * generate seq by mutating it from ref with `ed_norm * seq_len` random edit operations,
     * with number of insert, substitute, and deletes chosen randomly to sum to 'edit'
     * @tparam T : char type
     * @param ref : reference
     * @param seq : mutated sequence
     * @param ed_norm : normalized edit operations
     */
    template <class T>
    void mutate_edits(const std::vector<T> &ref, std::vector<T> &seq, double ed_norm) {
        assert(ed_norm>=0 && ed_norm<=1 && "ed_norm argument must be always in [0,1] range");
        std::uniform_real_distribution<double> r(0.0,1.0);
        double ins = r(gen), del =r(gen), sub = r(gen), S = (ins + del + sub)/(ed_norm *seq_len);
        ins = (size_t)ins / S;
        del = (size_t) del / S;
        sub = (ed_norm *seq_len)- ins - del;
        random_edits(ref, seq, (size_t)ins, (size_t)del, (size_t)sub);
    }


    /**
     * generate seq based on ref by mutating it with random edit operations, number of each edit
     * is given bby the input arguments ins, del, sub
     * @tparam T : char type
     * @param ref : reference sequence
     * @param seq : sequence to be generated by mutation from ref
     * @param ins : number of insert operation
     * @param del : number of delete operations
     * @param sub : number of substitute operations
     */
    template <class T>
    void random_edits(const std::vector<T> &ref, std::vector<T> &seq, size_t ins, size_t del, size_t sub) {
        assert(ref.size() + ins >= del && " can't delete more than ref len + inserted chars");
        random_sequence(seq, ref.size() + ins - del);   // calc. length of the resulting sequence

        // generate random indices not part of deleted nor inserted indices
        auto idx1 = rand_n_choose_k<size_t>(ref.size(), ref.size() - del),
             idx2 = rand_n_choose_k<size_t>(seq.size(), seq.size() - ins );
        for (size_t i = 0; i < idx2.size(); i++) {
            seq[idx2[i]] = ref[idx1[i]];
        }

        // generate substituted indices randomly
        std::uniform_int_distribution<T> sub_char(1, alphabet_size - 1);
        auto sub_idx = rand_n_choose_k<size_t>(idx2.size(), sub);
        for (auto & i : sub_idx) {
            seq[idx2[i]] = (seq[idx2[i]] + sub_char(gen) ) % alphabet_size;
        }
    }


    template <class T>
    void mutate_block(std::vector<T> &seq) {
        std::uniform_real_distribution<double> mute(0, 1);
        if (mute(gen) > block_mutate_rate) {
            return;
        }
        std::uniform_int_distribution<T> unif(0, alphabet_size - 1);
        std::uniform_int_distribution<T> blocks(min_num_blocks, max_num_blocks);
        int num_blocks = blocks(gen);
        std::vector<size_t> perm(num_blocks);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), gen);
        while (seq.size() % num_blocks != 0) { // make length divisible by num_blocks
            seq.push_back(unif(gen));
        }
        std::vector<T> result(seq.size());
        size_t block_size = seq.size() / num_blocks;
        for (size_t i = 0; i < block_size; i++) {
            for (int pi = 0; pi < num_blocks; pi++) {
                size_t bi = pi * block_size + i;
                size_t bj = perm[pi] * block_size + i;
                result[bj] = seq[bi];
            }
        }
        seq = result;
    }

    template <class T>
    void make_fix_len(std::vector<T> &seq) {
        std::uniform_int_distribution<T> rand_char(0, alphabet_size - 1),
                blocks(min_num_blocks, max_num_blocks);
        if (seq.size() > seq_len) {
            seq = std::vector<T>(seq.begin(), seq.end());
        } else if (seq.size() < seq_len) {
            while (seq.size() < seq_len) {
                seq.push_back(rand_char(gen));
            }
        }
    }

    /**
     * Generate a random sequence of length `len`
     * @tparam T
     * @param seq : the result will be stored in `seq`
     * @param len : length of the random sequence
     */
    template <class T>
    void random_sequence(std::vector<T> &seq, size_t len) {
        seq.resize(len);
        std::uniform_int_distribution<T> rand_char(0, alphabet_size - 1);
        for (uint32_t i = 0; i < len; i++) {
            seq[i] = rand_char(gen);
        }
    }

    /**
     * randomly select k items from {0, ..., n-1}
     * @param n: size of set to choose from {0, ..., n-1}
     * @param k: size of the output set permutation
     * @return the set containing the randomly selected indices, sorted
     */
    template <class T>
    std::vector<T> rand_n_choose_k( size_t n, size_t k) {
        assert(k <= n);
        std::vector<T> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), gen);
        idx.resize(k);
        std::sort(idx.begin(), idx.end());
        return idx;
    }

  private:
    std::mt19937 gen = std::mt19937(341234);

    uint8_t alphabet_size;
    bool fix_len;
    uint32_t max_num_blocks;
    uint32_t min_num_blocks;
    uint32_t num_seqs;
    uint32_t seq_len;
    uint32_t group_size;
    double mutation_rate;
    double min_mutation_rate;
    double block_mutate_rate;
    std::string mutation_type;
};

} // namespace ts
