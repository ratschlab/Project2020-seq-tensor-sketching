//
// Created by Amir Joudaki on 6/17/20.
//

#ifndef SEQUENCE_SKETCHING_COMMON_HPP
#define SEQUENCE_SKETCHING_COMMON_HPP
#include <string>
#include <vector>
#include <map>
#include <iostream>

namespace Sketching {

    namespace Types {
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
        using Seq = std::vector<T>;

    }// namespace Types

    namespace ArgNames {
        static const char *const FIX_LEN = "--fix_len";
        static const char *const NUM_SEQS = "--num_seqs";
        static const char *const SIG_LEN = "--sig_len";
        static const char *const SEQ_LEN = "--seq_len";
        static const char *const MUTATION_RATE = "--mutation_rate";
        static const char *const BLOCK_MUTATION_RATE = "--block_mutate_rate";
        static const char *const MAX_NUM_BLOCKS = "--max_num_blocks";
        static const char *const MIN_NUM_BLOCKS = "--min_num_blocks";
        static const char *const KMER_SIZE = "--kmer_size";
        static const char *const TUP_LEN = "--tup_len";
        static const char *const EMBED_DIM = "--embed_dim";
        static const char *const NUM_PHASES = "--num_phases";
        static const char *const NUM_BINS = "--num_bins";
        static const char *const WIN_LEN = "--win_len";
        static const char *const STRIDE = "--stride";
        // short names
        static const char *const FIX_LEN2 = "-F";
        static const char *const NUM_SEQS2 = "-N";
        static const char *const SIG_LEN2 = "-A";
        static const char *const SEQ_LEN2 = "-L";
        static const char *const MUTATION_RATE2 = "-r";
        static const char *const BLOCK_MUTATION_RATE2 = "-R";
        static const char *const MAX_NUM_BLOCKS2 = "-B";
        static const char *const MIN_NUM_BLOCKS2 = "-b";
        static const char *const NUM_PHASES2 = "-P";
        static const char *const NUM_BINS2 = "-n";
        static const char *const KMER_SIZE2 = "-K";
        static const char *const TUP_LEN2 = "-T";
        static const char *const EMBED_DIM2 = "-M";
        static const char *const WIN_LEN2 = "-W";
        static const char *const STRIDE2 = "-S";

    };
    enum ArgTypes {
        STRING = 1, INT = 2, FLOAT = 3, BOOL = 4
    };

    struct ArgParser {
        using string = std::string;
//        int argc;
//        char **argv;

        std::map<string, void*> arg2ptr;
        std::map<string, ArgTypes> arg2type;

//        ArgParser(int argc, char *argv[]) : argc(argc), argv(argv) {}

        bool not_exists(const string &name) {
            if (arg2ptr.find(name) != arg2ptr.end())
                return false;
            return true;
        }

        void basic_add(void* ptr, const string& name, const string &name2, ArgTypes argType) {
            assert(not_exists(name));
            assert(not_exists(name2));
            arg2ptr[name] = ptr;
            arg2ptr[name2] = ptr;
            arg2type[name] = argType;
            arg2type[name2] = argType;
        }

        void add(int *ptr, int val, const string& name, const string &name2) {
            basic_add(ptr, name, name2, ArgTypes::INT);
            *ptr = val;
        }
        void add(float *ptr, float val, const string &name, const string &name2) {
            basic_add(ptr, name, name2, ArgTypes::FLOAT);
            *ptr = val;
        }
        void add(string *ptr, string val, const string &name, const string &name2) {
            basic_add(ptr, name, name2, ArgTypes::STRING);
            *ptr = val;
        }
        void add(bool *ptr, bool val, const string& name, const string& name2) {
            basic_add(ptr, name, name2, ArgTypes::BOOL);
            *ptr = val;
        }

        void parse(int argc, char* argv[]) {
            for (int i=1; i<argc; i++) {
                void *ptr = nullptr;
                if (arg2ptr.find(argv[i]) != arg2ptr.end()) {
                    ptr = arg2ptr[argv[i]];
                } else {
                    std::string A = argv[i];
                    std::cerr << "invalid arg name";
                    exit(1);
                }
                switch (arg2type[argv[i]]) {
                    case ArgTypes::FLOAT:
                        *(float*)ptr = std::stof(argv[++i]);
                        break;
                    case ArgTypes::STRING:
                        *(string*)ptr = argv[++i];
                        break;
                    case ArgTypes::INT:
                        *(int*)ptr = std::stoi(argv[++i]);
                        break;
                    case ArgTypes::BOOL:
                        *(bool*)ptr = std::stoi(argv[++i]);
                        break;
                }
            }
        }
    };
}// namespace Sketching


#endif//SEQUENCE_SKETCHING_COMMON_HPP
