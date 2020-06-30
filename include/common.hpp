//
// Created by Amir Joudaki on 6/17/20.
//

#ifndef SEQUENCE_SKETCHING_COMMON_HPP
#define SEQUENCE_SKETCHING_COMMON_HPP

#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace SeqSearch {

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


    namespace Argument {
        namespace ArgNames {
            static const char *const L_FIX_LEN = "--fix_len";
            static const char *const L_NUM_SEQS = "--num_seqs";
            static const char *const L_SIG_LEN = "--sig_len";
            static const char *const L_SEQ_LEN = "--seq_len";
            static const char *const L_MAX_NUM_BLOCKS = "--max_num_blocks";
            static const char *const L_MIN_NUM_BLOCKS = "--min_num_blocks";
            static const char *const L_KMER_SIZE = "--kmer_size";
            static const char *const L_TUP_LEN = "--tup_len";
            static const char *const L_EMBED_DIM = "--embed_dim";
            static const char *const L_NUM_PHASES = "--num_phases";
            static const char *const L_NUM_BINS = "--num_bins";
            static const char *const L_WIN_LEN = "--win_len";
            static const char *const L_STRIDE = "--stride";
            static const char *const L_MUTATION_RATE = "--mutation_rate";
            static const char *const L_BLOCK_MUTATION_RATE = "--block_mutate_rate";
            // short names
            static const char *const S_FIX_LEN = "-F";
            static const char *const S_NUM_SEQS = "-N";
            static const char *const S_SIG_LEN = "-A";
            static const char *const S_SEQ_LEN = "-L";
            static const char *const S_MAX_NUM_BLOCKS = "-B";
            static const char *const S_MIN_NUM_BLOCKS = "-b";
            static const char *const S_NUM_PHASES = "-P";
            static const char *const S_NUM_BINS = "-n";
            static const char *const S_KMER_SIZE = "-K";
            static const char *const S_TUP_LEN = "-T";
            static const char *const S_EMBED_DIM = "-M";
            static const char *const S_WIN_LEN = "-W";
            static const char *const S_STRIDE = "-S";
            static const char *const S_MUTATION_RATE = "-r";
            static const char *const S_BLOCK_MUTATION_RATE = "-R";

        };// namespace ArgNames
        using namespace ArgNames;

        enum StoreType {
            STRING = 1,
            INT = 2,
            FLOAT = 3,
            BOOL = 4
        };

        struct Type {
            const char *long_name;
            const char *short_name;
            StoreType type;
            void *ptr = nullptr;
        };
        static const Type FIX_LEN = {L_FIX_LEN, S_FIX_LEN, BOOL};
        static const Type NUM_SEQS = {L_NUM_SEQS, S_NUM_SEQS, INT};
        static const Type SIG_LEN = {L_SIG_LEN, S_SIG_LEN, INT};
        static const Type SEQ_LEN = {L_SEQ_LEN, S_SEQ_LEN, INT};
        static const Type MAX_NUM_BLOCKS = {L_MAX_NUM_BLOCKS, S_MAX_NUM_BLOCKS, INT};
        static const Type MIN_NUM_BLOCKS = {L_MIN_NUM_BLOCKS, S_MIN_NUM_BLOCKS, INT};
        static const Type NUM_PHASES = {L_NUM_PHASES, S_NUM_PHASES, INT};
        static const Type NUM_BINS = {L_NUM_BINS, S_NUM_BINS, INT};
        static const Type KMER_SIZE = {L_KMER_SIZE, S_KMER_SIZE, INT};
        static const Type TUP_LEN = {L_TUP_LEN, S_TUP_LEN, INT};
        static const Type EMBED_DIM = {L_EMBED_DIM, S_EMBED_DIM, INT};
        static const Type WIN_LEN = {L_WIN_LEN, S_WIN_LEN, INT};
        static const Type STRIDE = {L_STRIDE, S_STRIDE, INT};
        static const Type MUTATION_RATE = {L_MUTATION_RATE, S_MUTATION_RATE, FLOAT};
        static const Type BLOCK_MUTATION_RATE = {L_BLOCK_MUTATION_RATE, S_BLOCK_MUTATION_RATE, FLOAT};
    }// namespace Argument

    struct Parser {
        using string = std::string;
        std::map<string, Argument::Type> name2arg;

        bool not_exists(const string &name) {
            if (name2arg.find(name) != name2arg.end())
                return false;
            return true;
        }

        // TODO add regex checks for validity of argnames or values
        void add(void *ptr, Argument::Type arg) {
            assert(not_exists(arg.long_name));
            assert(not_exists(arg.short_name));
            arg.ptr = ptr;
            name2arg[arg.long_name] = arg;
            name2arg[arg.short_name] = arg;
        }
        void parse(int argc, char *argv[]) {
            for (int i = 1; i < argc; i++) {
                Argument::Type arg;
                if (name2arg.find(argv[i]) != name2arg.end()) {
                    arg = name2arg[argv[i]];
                } else {
                    std::string A = argv[i];
                    std::cerr << "invalid arg long_name";
                    exit(1);
                }
                switch (name2arg[argv[i]].type) {
                    case Argument::FLOAT:
                        *(float *) arg.ptr = std::stof(argv[++i]);
                        break;
                    case Argument::STRING:
                        *(string *) arg.ptr = argv[++i];
                        break;
                    case Argument::INT:
                        *(int *) arg.ptr = std::stoi(argv[++i]);
                        break;
                    case Argument::BOOL:
                        *(bool *) arg.ptr = std::stoi(argv[++i]);
                        break;
                }
            }
        }
    };
}// namespace SeqSearch


#endif//SEQUENCE_SKETCHING_COMMON_HPP
