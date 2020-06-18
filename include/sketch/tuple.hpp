//
// Created by Amir Joudaki on 6/18/20.
//

#ifndef SEQUENCE_SKETCHING_TUPLE_H
#define SEQUENCE_SKETCHING_TUPLE_H

namespace SeqSearch {

    template<class seq_type, class size_type>
    size_type subseq2ind(const Vec<seq_type> &seq, const Vec<size_type> &sub, size_type sig_len) {
        size_type ind = 0, coef = 1;
        for (size_type i = 0; i < sub.size(); i++) {
            ind += seq[sub[i]] * coef;
            coef *= sig_len;
        }
        return ind;
    }

    template<class seq_type, class embed_type, class size_type = std::size_t>
    void tup_embed(const Seq<seq_type> &seq, Vec<embed_type> &embed,
                   size_type sig_len, size_type tup_len) {
        size_type seq_len = seq.size();
        size_type cnt = 0, size = int_pow(sig_len, tup_len);
        embed = Vec<embed_type>(size, 0);
        Vec<size_type> sub(tup_len, 0);
        do {
            if (is_ascending(sub)) {
                auto ind = subseq2ind(seq, sub, sig_len);
                embed[ind]++;
            }
        } while (increment_sub(sub, seq_len));
    }

}// namespace SeqSearch

#endif//SEQUENCE_SKETCHING_TUPLE_H
