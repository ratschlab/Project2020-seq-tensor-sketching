//
// Created by Amir Joudaki on 6/17/20.
//

#ifndef SEQUENCE_SKETCHING_ARGS_HPP
#define SEQUENCE_SKETCHING_ARGS_HPP

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace SeqSketch {

    namespace BasicTypes {
        template<class T>
        using is_u_integral = typename std::enable_if<std::is_unsigned<T>::value>::type;
        using Index = std::size_t;
        using Size_t = std::size_t;
        template<class T>
        using Vec = std::vector<T>;
        template<class T>
        using Vec2D = Vec<Vec<T>>;
        template<class T>
        using Vec3D = Vec<Vec2D<T>>;
        template<class T>
        using Vec4D = Vec<Vec3D<T>>;
        template<class T>
        using Seq = std::vector<T>;

    }// namespace BasicTypes

    template<typename T>
    int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    enum ArgType {
        STRING = 1,
        INT = 2,
        FLOAT = 3,
        BOOL = 4,
        INDIC = 5
    };

    struct Argument {
        const char *long_name;
        const char *short_name;
        ArgType type;
        void *ptr = nullptr;
    };
    struct Parser {
        using string = std::string;
        std::map<string, Argument> name2arg;

        bool not_exists(const string &name) {
            if (name2arg.find(name) != name2arg.end())
                return false;
            return true;
        }

        // TODO add regex checks for validity of argnames or values
        void add(void *ptr, Argument arg) {
            assert(not_exists(arg.long_name));
            assert(not_exists(arg.short_name));
            arg.ptr = ptr;
            name2arg[arg.long_name] = arg;
            name2arg[arg.short_name] = arg;
        }
        void parse(int argc, char *argv[]) {
            for (int i = 1; i < argc; i++) {
                Argument arg;
                if (name2arg.find(argv[i]) != name2arg.end()) {
                    arg = name2arg[argv[i]];
                } else {
                    std::string A = argv[i];
                    std::cerr << "invalid arg long_name";
                    exit(1);
                }
                switch (name2arg[argv[i]].type) {
                    case FLOAT:
                        *(float *) arg.ptr = std::stof(argv[++i]);
                        break;
                    case STRING:
                        *(string *) arg.ptr = argv[++i];
                        break;
                    case INT:
                        *(int *) arg.ptr = std::stoi(argv[++i]);
                        break;
                    case BOOL:
                        *(bool *) arg.ptr = std::stoi(argv[++i]);
                        break;
                    case INDIC:
                        break;
                }
            }
        }
    };

    //    using namespace Argument;

    struct BasicParams : public Parser {
        const char *const L_FIX_LEN = "--fix_len";
        const char *const L_NUM_SEQS = "--num_seqs";
        const char *const L_SIG_LEN = "--sig_len";
        const char *const L_SEQ_LEN = "--seq_len";
        const char *const L_MAX_NUM_BLOCKS = "--max_num_blocks";
        const char *const L_MIN_NUM_BLOCKS = "--min_num_blocks";
        const char *const L_KMER_SIZE = "--kmer_size";
        const char *const L_TUP_LEN = "--tup_len";
        const char *const L_EMBED_DIM = "--embed_dim";
        const char *const L_NUM_PHASES = "--num_phases";
        const char *const L_NUM_BINS = "--num_bins";
        const char *const L_WIN_LEN = "--win_len";
        const char *const L_STRIDE = "--stride";
        const char *const L_MUTATION_RATE = "--mutation_rate";
        const char *const L_BLOCK_MUTATION_RATE = "--block_mutate_rate";
        const char *const L_METHOD_NAME = "--method_name";
        const char *const L_HELP = "--help";

        const char *const S_FIX_LEN = "-F";
        const char *const S_NUM_SEQS = "-N";
        const char *const S_SIG_LEN = "-A";
        const char *const S_SEQ_LEN = "-L";
        const char *const S_MAX_NUM_BLOCKS = "-B";
        const char *const S_MIN_NUM_BLOCKS = "-b";
        const char *const S_NUM_PHASES = "-P";
        const char *const S_NUM_BINS = "-n";
        const char *const S_KMER_SIZE = "-K";
        const char *const S_TUP_LEN = "-T";
        const char *const S_EMBED_DIM = "-M";
        const char *const S_WIN_LEN = "-W";
        const char *const S_STRIDE = "-S";
        const char *const S_MUTATION_RATE = "-r";
        const char *const S_BLOCK_MUTATION_RATE = "-R";
        const char *const S_METHOD_NAME = "--m";
        const char *const S_HELP = "-h";

        const Argument FIX_LEN = {L_FIX_LEN, S_FIX_LEN, BOOL};
        const Argument NUM_SEQS = {L_NUM_SEQS, S_NUM_SEQS, INT};
        const Argument SIG_LEN = {L_SIG_LEN, S_SIG_LEN, INT};
        const Argument SEQ_LEN = {L_SEQ_LEN, S_SEQ_LEN, INT};
        const Argument MAX_NUM_BLOCKS = {L_MAX_NUM_BLOCKS, S_MAX_NUM_BLOCKS, INT};
        const Argument MIN_NUM_BLOCKS = {L_MIN_NUM_BLOCKS, S_MIN_NUM_BLOCKS, INT};
        const Argument NUM_PHASES = {L_NUM_PHASES, S_NUM_PHASES, INT};
        const Argument NUM_BINS = {L_NUM_BINS, S_NUM_BINS, INT};
        const Argument KMER_SIZE = {L_KMER_SIZE, S_KMER_SIZE, INT};
        const Argument TUP_LEN = {L_TUP_LEN, S_TUP_LEN, INT};
        const Argument EMBED_DIM = {L_EMBED_DIM, S_EMBED_DIM, INT};
        const Argument WIN_LEN = {L_WIN_LEN, S_WIN_LEN, INT};
        const Argument STRIDE = {L_STRIDE, S_STRIDE, INT};
        const Argument MUTATION_RATE = {L_MUTATION_RATE, S_MUTATION_RATE, FLOAT};
        const Argument BLOCK_MUTATION_RATE = {L_BLOCK_MUTATION_RATE, S_BLOCK_MUTATION_RATE, FLOAT};
        const Argument METHOD_NAME = {L_METHOD_NAME, S_METHOD_NAME, STRING};

        bool fix_len = true;
        int sig_len = 4;
        int max_num_blocks = 4;
        int min_num_blocks = 2;
        int num_seqs = 200;
        int seq_len = 256;
        int kmer_size = 2;
        int embed_dim = 128;
        int tup_len = 2;
        int num_phases = 6;
        int num_bins = 255;
        int win_len = 32;
        int stride = 8;
        float mutation_rate = 0.015;
        float block_mutate_rate = 0.02;

        BasicParams() {
            add(&fix_len, FIX_LEN);
            add(&sig_len, SIG_LEN);
            add(&max_num_blocks, MAX_NUM_BLOCKS);
            add(&min_num_blocks, MIN_NUM_BLOCKS);
            add(&num_seqs, NUM_SEQS);
            add(&seq_len, SEQ_LEN);
            add(&mutation_rate, MUTATION_RATE);
            add(&block_mutate_rate, BLOCK_MUTATION_RATE);
            add(&kmer_size, KMER_SIZE);
            add(&embed_dim, EMBED_DIM);
            add(&tup_len, TUP_LEN);
            add(&num_phases, NUM_PHASES);
            add(&num_bins, NUM_BINS);
            add(&win_len, WIN_LEN);
            add(&stride, STRIDE);
        }
    };
}// namespace SeqSketch


#endif//SEQUENCE_SKETCHING_ARGS_HPP
