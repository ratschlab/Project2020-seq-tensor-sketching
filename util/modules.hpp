#pragma once

#include "util/args.hpp"
#include "util/seqgen.h"
#include "sketch/minhash.hpp"
#include "sketch/omh.hpp"
#include "sketch/tensor.hpp"
#include "sketch/tensor_new.h"
#include "sketch/tensor_slide.hpp"
#include "sketch/tuple.hpp"
#include "sketch/wminhash.hpp"

namespace ts { // ts = Tensor Sketch

struct BasicModule : public ArgSet {
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
        params.max_len = 2 * seq_len;
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
        params.offset = offset;
        //            assert(seq_len % stride == 0);
    }


    void rand_init() {
        mh_params.rand_init();
        wmh_params.rand_init();
        omh_params.init_rand();
        tensor_params.rand_init();
        tensor_slide_params.rand_init();
    }

    /**
     * runs after parsing the arguments, before model hyperparameters
     * hence, it affects hyperparameters in all modules.
     * Example: embed_dim = embed_dim/2; // embedding dimension halves for all
     */
    virtual void override_module_params() {}

    /**
     *  runs after modules parameters are set, but before rand_init(), hence, it
     *  will override individual model parameters, and their random initialization
     *  Example: OMH.max_len = win_len
     */
    virtual void override_model_params() { tensor_slide_params.embed_dim = embed_dim / stride + 1; }

  public:
    void models_init() {
        override_module_params();
        init_seqgen(seq_gen);
        init_mh_params(mh_params);
        init_wmh_params(wmh_params);
        init_omh(omh_params);
        init_tensor_params(tensor_params);
        init_tensor_slide_params(tensor_slide_params);
        override_model_params();
        rand_init();
    }
};

struct ComboModules_v2 : public ArgSet {
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
        int embed = embed_dim / stride + 1;
        params.embed_dim = embed + 1;
    }

    void model_init() {
        init_ten_2_params(ten_2_params);
        init_ten_2_slide_params(ten_2_slide_params);
        rand_init();
    }
};

} // namespace ts
