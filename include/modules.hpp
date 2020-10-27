//
// Created by Amir Joudaki on 6/17/20.
//

#ifndef SEQUENCE_SKETCHING_MODULES_H

#include "../cpp/experimental/tensor_slide2.hpp"
#include "args.hpp"
#include "seqgen.hpp"
#include "sketch/minhash.hpp"
#include "sketch/omh.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_disc.hpp"
#include "sketch/tensor_new.h"
#include "sketch/tensor_slide.hpp"
#include "sketch/tuple.hpp"
#include "sketch/wminhash.hpp"

namespace SeqSketch {

    struct BasicModules : public BasicParams {
        SeqGen seq_gen;
        MHParams mh_params;
        WMHParams wmh_params;
        OMHParams omh_params;
        TensorParams tensor_params;
        TensorSlideParams tensor_slide_params;

    protected:
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

        void init_mh_params(MHParams &params) const {
            params.sig_len = sig_len;
            params.embed_dim = embed_dim;
        }

        void init_wmh_params(WMHParams &params) const {
            params.embed_dim = embed_dim;
            params.sig_len = sig_len;
            params.max_len = 2 * seq_len;
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
            assert(seq_len % stride == 0);
            params.embed_dim = embed_dim / stride + 1;
        }


        void rand_init() {
            mh_params.rand_init();
            wmh_params.rand_init();
            omh_params.init_rand();
            tensor_params.rand_init();
            tensor_slide_params.rand_init();
        }

        /**
         * runs before module params are set, affect params in all modules
         */
        virtual void override_pre() {}

        /**
         *  runs after modules parameters are set, but before rand_init()
         */
        virtual void override_post() {}

    public:
        void models_init() {
            override_pre();
            init_seqgen(seq_gen);
            init_mh_params(mh_params);
            init_wmh_params(wmh_params);
            init_omh(omh_params);
            init_tensor_params(tensor_params);
            init_tensor_slide_params(tensor_slide_params);
            override_post();
            rand_init();
        }
    };

    struct NewModules : public BasicParams {
        Tensor2Params ten_2_params;
        Tensor2_slide_Params ten_2_slide_params;

        void rand_init() {
            ten_2_params.rand_init();
            ten_2_slide_params.rand_init();
        }

        void init_ten_2_params(Tensor2Params &params) const {
            params.tup_len = tup_len;
            params.sig_len = sig_len;
            params.embed_dim = embed_dim;
            params.num_phases = num_phases;
        }
        void init_ten_2_slide_params(Tensor2_slide_Params &params) const {
            init_ten_2_params(params);
            params.win_len = win_len;
            params.stride = stride;
            int embed = embed_dim * stride / seq_len;
            params.embed_dim = embed + 1;
        }

        void model_init() {
            init_ten_2_params(ten_2_params);
            init_ten_2_slide_params(ten_2_slide_params);
            rand_init();
        }
    };

}// namespace SeqSketch

#define SEQUENCE_SKETCHING_MODULES_H

#endif//SEQUENCE_SKETCHING_MODULES_H
