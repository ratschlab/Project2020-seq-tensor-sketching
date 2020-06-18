//
// Created by Amir Joudaki on 6/17/20.
//

#ifndef SEQUENCE_SKETCHING_MODULES_H

#include "common.hpp"
#include "seqgen.hpp"
#include "sketch/minhash.hpp"
#include "sketch/omh.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tuple.hpp"
#include "sketch/wminhash.hpp"

namespace Sketching {

    struct BasicParams : public ArgParser {
        bool fix_len;
        int sig_len;
        int max_num_blocks;
        int min_num_blocks;
        int num_seqs;
        int seq_len;
        int kmer_size;
        int embed_dim;
        int tup_len;
        int num_phases;
        int num_bins;
        int win_len;
        int stride;
        float mutation_rate;
        float block_mutate_rate;

        BasicParams() {
            add(&fix_len, true, ArgNames::FIX_LEN, ArgNames::FIX_LEN2);
            add(&sig_len, 4, ArgNames::SIG_LEN, ArgNames::SIG_LEN2);
            add(&max_num_blocks, 4, ArgNames::MAX_NUM_BLOCKS, ArgNames::MAX_NUM_BLOCKS2);
            add(&min_num_blocks, 2, ArgNames::MIN_NUM_BLOCKS, ArgNames::MIN_NUM_BLOCKS2);
            add(&num_seqs, 200, ArgNames::NUM_SEQS, ArgNames::NUM_SEQS2);
            add(&seq_len, 256, ArgNames::SEQ_LEN, ArgNames::SEQ_LEN2);
            add(&mutation_rate, 0.015, ArgNames::MUTATION_RATE, ArgNames::MUTATION_RATE2);
            add(&block_mutate_rate, 0.02, ArgNames::BLOCK_MUTATION_RATE, ArgNames::BLOCK_MUTATION_RATE2);
            add(&kmer_size, 2, ArgNames::KMER_SIZE, ArgNames::KMER_SIZE2);
            add(&embed_dim, 100, ArgNames::EMBED_DIM, ArgNames::EMBED_DIM2);
            add(&tup_len, 2, ArgNames::TUP_LEN, ArgNames::TUP_LEN2);
            add(&num_phases, 5, ArgNames::NUM_PHASES, ArgNames::NUM_PHASES2);
            add(&num_bins, 255, ArgNames::NUM_BINS, ArgNames::NUM_BINS2);
            add(&win_len, 32, ArgNames::WIN_LEN, ArgNames::WIN_LEN2);
            add(&stride, 8, ArgNames::STRIDE, ArgNames::STRIDE2);
        }
    };

    struct BasicModules : public BasicParams {
        // modudles
        MHParams mh_params;
        WMHParams wmh_params;
        OMHParams omh_params;
        TensorParams tensor_params;
        TensorSlideParams tensor_slide_params;

        BasicModules() = default;
        BasicModules(int argc, char* argv[]) {
            parse(argc, argv);
//            models_init();
        }

        void init_seqgen(SeqGen &seqgen) const {
            seqgen.sig_len = sig_len;
            seqgen.fix_len = fix_len;
            seqgen.max_num_blocks = max_num_blocks;
            seqgen.min_num_blocks = min_num_blocks;
            seqgen.num_seqs = num_seqs;
            seqgen.seq_len = seq_len;
            seqgen.mutation_rate = mutation_rate;
            seqgen.block_mutate_rate = block_mutate_rate;
        }

        void init_omh(OMHParams &params) const {
            params.tup_len = tup_len;
            params.sig_len = sig_len;
            params.embed_dim = embed_dim;
            params.max_seq_len = 2 * seq_len;
        }

        void init_tensor_params(TensorParams &params) const {
            params.embed_dim = embed_dim;
            params.sig_len = sig_len;
            params.tup_len = tup_len;
            params.num_phases = num_phases;
            params.num_bins = num_bins;
        }

        void init_tensor_slide_params(TensorSlideParams &params) const {
            init_tensor_params(params);
            params.win_len = win_len;
            params.stride = stride;
            params.embed_dim = embed_dim / stride;
        }

        void init_mh_params(MHParams &params) const {
            params.sig_len = sig_len;
            params.embed_dim = embed_dim;
        }

        void init_wmh_params(WMHParams &params) const {
            params.embed_dim = embed_dim;
            params.sig_len = sig_len;
            params.max_len = 2 * seq_len;
        }

        void rand_init() {
            mh_params.rand_init();
            wmh_params.rand_init();
            omh_params.init_rand();
            tensor_params.rand_init();
            tensor_slide_params.rand_init();
        }

        virtual void pre() {}

        virtual void post() {
            tensor_slide_params.sig_len = 4;
            tensor_slide_params.tup_len = 2;
        }

        void models_init() {
            pre();
            init_mh_params(mh_params);
            init_wmh_params(wmh_params);
            init_omh(omh_params);
            init_tensor_params(tensor_params);
            init_tensor_slide_params(tensor_slide_params);
            post();
            rand_init();
        }
    };

    struct KmerModules : public BasicModules {

        KmerModules() = default;
        KmerModules(int argc, char* argv[]) : BasicModules(argc, argv) {}

        void pre() override {
            sig_len = int_pow<size_t>(sig_len, kmer_size);
        }
    };


}// namespace Sketching

#define SEQUENCE_SKETCHING_MODULES_H

#endif//SEQUENCE_SKETCHING_MODULES_H
