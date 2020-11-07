//
// Created by Amir Joudaki on 6/17/20.
//

#ifndef SEQUENCE_SKETCHING_ARGS_HPP
#define SEQUENCE_SKETCHING_ARGS_HPP

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "utils.h"
#include <string_view>

namespace SeqSketch {
    using namespace BasicTypes;

    struct Parser {
        enum ArgType {
            STRING = 1,
            INT = 2,
            FLOAT = 3,
            BOOL = 4,
            INDIC = 5

        };

        struct Argument {
            using string = std::string;
            const char *long_name;
            const char *short_name;
            ArgType type;
            string description;
            void *ptr = nullptr;

            int num_inputs() const {
                if (type == INDIC) {
                    return 0;
                } else {
                    return 1;
                }
            }

            void set(char *argv) {
                switch (type) {
                    case FLOAT:
                        *(float *) ptr = std::stof(argv);
                        break;
                    case STRING:
                        *(string *) ptr = argv;
                        break;
                    case INT:
                        *(int *) ptr = std::stoi(argv);
                        break;
                    case BOOL:
                        *(bool *) ptr = std::stoi(argv);
                        break;
                    case INDIC:
                        *(bool *) ptr = true;
                        break;
                }
            }

            std::string type2string() const {
                switch (type) {
                    case STRING:
                        return "string";
                    case INT:
                        return "int";
                    case FLOAT:
                        return "float";
                    case BOOL:
                    case INDIC:
                        return "bool";
                }
            }

            std::string to_string() const {
                switch (type) {
                    case STRING:
                        return *(string *) ptr;
                    case INT:
                        return std::to_string(*(int *) ptr);
                    case FLOAT:
                        return std::to_string(*(float *) ptr);
                    case BOOL:
                    case INDIC:
                        return std::to_string(*(bool *) ptr);
                }
            }

            std::string help() const {
                std::string str;
                str += long_name;
                str += ",";
                str += short_name;
                while (str.size() < 25)
                    str += " ";
                str += "" + type2string();
                while (str.size() < 35)
                    str += " ";
                str += ' ' + description;
                str += " (default: " + this->to_string() + ")";
                return str;
            }
        };
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

        string argvals(bool long_names = true, char sep = ' ', char tab = ' ') {
            string str;
            for (auto const &[arg_name, arg] : name2arg) {
                if ((arg_name.find("--") != std::string::npos) == long_names) {
                    str += " " + arg_name + tab + arg.to_string() + sep;
                }
            }
            return str;
        }

        string config() {
            string str;
            for (auto const &[arg_name, arg] : name2arg) {
                if (arg_name.find("--") != std::string::npos) {
                    str += " " + arg_name + ",\t" + arg.type2string() + ",\t" + arg.to_string() + '\n';
                }
            }
            return str;
        }

        string description() {
            string str;
            for (auto const &[x, arg] : name2arg) {
                str += arg.help() + '\n';
            }
            return str;
        }

        void parse(int argc, char *argv[]) {
            for (int i = 1; i < argc; i++) {
                //                auto aptr = name2arg.find(argv[i]);
                if (name2arg.find(argv[i]) == name2arg.end()) {
                    std::cerr << argv[i] << " is an invalid arg name\n";
                    exit(1);
                }
                auto &aptr = name2arg[argv[i]];
                if (i + aptr.num_inputs() >= argc) {
                    std::cerr << "argument " << argv[i] << " requires an input\n";
                    exit(1);
                }
                i += aptr.num_inputs();
                aptr.set(argv[i]);
            }
        }
    };

    namespace ArgDefs {
        using Argument = Parser::Argument;
        using ArgType = Parser::ArgType;
        static const char *const L_FIX_LEN = "--fix-len";
        static const char *const L_NUM_SEQS = "--num-seqs";
        static const char *const L_SIG_LEN = "--sig-len";
        static const char *const L_SEQ_LEN = "--seq-len";
        static const char *const L_MAX_NUM_BLOCKS = "--max-num-blocks";
        static const char *const L_MIN_NUM_BLOCKS = "--min-num-blocks";
        static const char *const L_KMER_SIZE = "--kmer-size";
        static const char *const L_TUPLE_ON_KMER = "--tuple-on-kmer";
        static const char *const L_TUP_LEN = "--tup-len";
        static const char *const L_EMBED_DIM = "--embed-dim";
        static const char *const L_NUM_PHASES = "--num-phases";
        static const char *const L_NUM_BINS = "--num-bins";
        static const char *const L_WIN_LEN = "--win-len";
        static const char *const L_STRIDE = "--stride";
        static const char *const L_OFFSET = "--offset";
        static const char *const L_MUTATION_RATE = "--mutation-rate";
        static const char *const L_BLOCK_MUTATION_RATE = "--block-mutate-rate";
        static const char *const L_SKETCH_METHOD = "--sketch-method";
        static const char *const L_OUTPUT = "--output";
        static const char *const L_INPUT = "--input";
        static const char *const L_FORMAT_INPUT = "--format-input";
        static const char *const L_DIRECTORY = "--directory";
        static const char *const L_MUTATION_PATTERN = "--mutation-pattern";
        static const char *const L_SEQUENCE_SEEDS = "--sequence-seeds";
        static const char *const L_SHOW_HELP = "--help";

        static const char *const S_FIX_LEN = "-F";
        static const char *const S_NUM_SEQS = "-N";
        static const char *const S_SIG_LEN = "-A";
        static const char *const S_SEQ_LEN = "-L";
        static const char *const S_MAX_NUM_BLOCKS = "-B";
        static const char *const S_MIN_NUM_BLOCKS = "-b";
        static const char *const S_NUM_PHASES = "-P";
        static const char *const S_NUM_BINS = "-n";
        static const char *const S_KMER_SIZE = "-K";
        static const char *const S_TUPLE_ON_KMER = "-tk";
        static const char *const S_TUP_LEN = "-T";
        static const char *const S_EMBED_DIM = "-M";
        static const char *const S_WIN_LEN = "-W";
        static const char *const S_STRIDE = "-S";
        static const char *const S_OFFSET = "-O";
        static const char *const S_MUTATION_RATE = "-r";
        static const char *const S_BLOCK_MUTATION_RATE = "-R";
        static const char *const S_SKETCH_METHOD = "-m";
        static const char *const S_SHOW_HELP = "-h";
        static const char *const S_OUTPUT = "-o";
        static const char *const S_INPUT = "-i";
        static const char *const S_FORMAT_INPUT = "-f";
        static const char *const S_DIRECTORY = "-d";
        static const char *const S_MUTATION_PATTERN = "-mp";
        static const char *const S_SEQUENCE_SEEDS = "-s";


        static const Argument FIX_LEN = {L_FIX_LEN, S_FIX_LEN, ArgType::BOOL, "force generated sequence length to be equal"};
        static const Argument NUM_SEQS = {L_NUM_SEQS, S_NUM_SEQS, ArgType::INT, "number of sequences to be generated"};
        static const Argument SIG_LEN = {L_SIG_LEN, S_SIG_LEN, ArgType::INT, "sigma, size of the alphabet"};
        static const Argument SEQ_LEN = {L_SEQ_LEN, S_SEQ_LEN, ArgType::INT, "sequence length: the length of sequence to be generated"};
        static const Argument MAX_NUM_BLOCKS = {L_MAX_NUM_BLOCKS, S_MAX_NUM_BLOCKS, ArgType::INT, "maximum number of blocks for block permutation"};
        static const Argument MIN_NUM_BLOCKS = {L_MIN_NUM_BLOCKS, S_MIN_NUM_BLOCKS, ArgType::INT, "minimum number of blocks for block permutation"};
        static const Argument SEQUENCE_SEEDS = {L_SEQUENCE_SEEDS, S_SEQUENCE_SEEDS, ArgType::INT, "number of initial random sequences "};
        static const Argument MUTATION_PATTERN = {L_MUTATION_PATTERN, S_MUTATION_PATTERN, ArgType::STRING, "the mutational pattern, can be 'linear', or 'tree'"};
        static const Argument BLOCK_MUTATION_RATE = {L_BLOCK_MUTATION_RATE, S_BLOCK_MUTATION_RATE, ArgType::FLOAT, "the probability of having a block permutation"};
        static const Argument MUTATION_RATE = {L_MUTATION_RATE, S_MUTATION_RATE, ArgType::FLOAT, "rate of point mutation rate for sequence generation"};
        static const Argument NUM_PHASES = {L_NUM_PHASES, S_NUM_PHASES, ArgType::INT, "number of phase to be used for modular arithmetic in tensor sketching"};
        static const Argument NUM_BINS = {L_NUM_BINS, S_NUM_BINS, ArgType::INT, "number of bins for descritization after tensor sketching"};
        static const Argument KMER_SIZE = {L_KMER_SIZE, S_KMER_SIZE, ArgType::INT, "kmer length, used for sequence to kmer transformation"};
        static const Argument TUPLE_ON_KMER = {L_TUPLE_ON_KMER, S_TUPLE_ON_KMER, ArgType::INDIC, "apply tuple-based methods (OMH, TensorSketch, and TenSlide), on kmer sequence"};
        static const Argument TUP_LEN = {L_TUP_LEN, S_TUP_LEN, ArgType::INT, "ordered tuple length, used in ordered MinHash and Tensor-based sketches"};
        static const Argument EMBED_DIM = {L_EMBED_DIM, S_EMBED_DIM, ArgType::INT, "embedding dimension, used for all sketching methods"};
        static const Argument WIN_LEN = {L_WIN_LEN, S_WIN_LEN, ArgType::INT, "window length: the size of sliding window, which will be all sketched"};
        static const Argument STRIDE = {L_STRIDE, S_STRIDE, ArgType::INT, "stride for sliding window: shift step for sliding window"};
        static const Argument OFFSET = {L_OFFSET, S_OFFSET, ArgType::INT, "initial index to start the sliding window"};
        static const Argument SKETCH_METHOD = {L_SKETCH_METHOD, S_SKETCH_METHOD, ArgType::STRING, "the sketching method to use, options are MH, WMH, OMH, TenSketch, TenSlide"};
        static const Argument DIRECTORY = {L_DIRECTORY, S_DIRECTORY, ArgType::STRING, "working directory for input/output reading/writing"};
        static const Argument OUTPUT = {L_OUTPUT, S_OUTPUT, ArgType::STRING, "output file path"};
        static const Argument INPUT = {L_INPUT, S_INPUT, ArgType::STRING, "input file path"};
        static const Argument FORMAT_INPUT = {L_FORMAT_INPUT, S_FORMAT_INPUT, ArgType::STRING, "input format, options: 'fasta', 'csv'"};
        static const Argument SHOW_HELP = {L_SHOW_HELP, S_SHOW_HELP, ArgType::INDIC, "show this help "};
    }// namespace ArgDefs

    //    using namespace Argument;

    /**
     * It's safer to extend this class, since default values
     * must be set explicitly, but ArgSet is a shortcut
     */
    struct VirtualArgSet : public Parser {
        Vec<string> method_names_short = {"ED", "MH", "WMH", "OMH", "TenSketch", "TenSlide", "Ten2", "Ten2Slide"};
        Vec<string> method_names_long = {"edit distance", "MinHash", "Weighted MinHash", "Ordered MinHash", "Tensor Sketch", "Slide Sketch", "Tensor Sketch v2", "Slide Sketch v2"};

        // generic
        bool show_help;
        string directory;
        string output;
        string input;
        string format_input;
        int sig_len;
        // sequence generation
        int num_seqs;
        int seq_len;
        bool fix_len;
        int max_num_blocks;
        int min_num_blocks;
        float mutation_rate;
        float block_mutate_rate;
        int sequence_seeds;
        string mutation_pattern;
        // sketching
        int kmer_size;
        string sketch_method;
        int embed_dim;
        int tup_len;
        bool tuple_on_kmer;
        int num_phases;
        int num_bins;
        int win_len;
        int stride;
        int offset;

        VirtualArgSet() {
            // generic
            add(&show_help, ArgDefs::SHOW_HELP);
            add(&output, ArgDefs::OUTPUT);
            add(&input, ArgDefs::INPUT);
            add(&format_input, ArgDefs::FORMAT_INPUT);
            add(&directory, ArgDefs::DIRECTORY);
            add(&sig_len, ArgDefs::SIG_LEN);

            // sequence generation
            add(&num_seqs, ArgDefs::NUM_SEQS);
            add(&seq_len, ArgDefs::SEQ_LEN);
            add(&fix_len, ArgDefs::FIX_LEN);
            add(&max_num_blocks, ArgDefs::MAX_NUM_BLOCKS);
            add(&min_num_blocks, ArgDefs::MIN_NUM_BLOCKS);
            add(&mutation_rate, ArgDefs::MUTATION_RATE);
            add(&block_mutate_rate, ArgDefs::BLOCK_MUTATION_RATE);
            add(&sequence_seeds, ArgDefs::SEQUENCE_SEEDS);
            add(&mutation_pattern, ArgDefs::MUTATION_PATTERN);

            // sequence analysis
            add(&kmer_size, ArgDefs::KMER_SIZE);
            add(&sketch_method, ArgDefs::SKETCH_METHOD);
            add(&embed_dim, ArgDefs::EMBED_DIM);
            add(&tup_len, ArgDefs::TUP_LEN);
            add(&tuple_on_kmer, ArgDefs::TUPLE_ON_KMER);
            add(&num_phases, ArgDefs::NUM_PHASES);
            add(&num_bins, ArgDefs::NUM_BINS);
            add(&win_len, ArgDefs::WIN_LEN);
            add(&stride, ArgDefs::STRIDE);
            add(&offset, ArgDefs::OFFSET);
        }
    };

    struct ArgSet : public VirtualArgSet {
        ArgSet() : VirtualArgSet() {
            fix_len = true;
            sig_len = 4;
            max_num_blocks = 4;
            min_num_blocks = 2;
            num_seqs = 200;
            seq_len = 256;
            sequence_seeds = 1;
            kmer_size = 2;
            embed_dim = 128;
            tup_len = 2;
            tuple_on_kmer = false;
            num_phases = 2;
            num_bins = 255;
            win_len = 32;
            stride = 8;
            offset = 0;
            mutation_rate = 0.015;
            mutation_pattern = "linear";
            block_mutate_rate = 0.02;
            show_help = false;
            directory = "./";
            output = "data/";
            input = "data/";
            format_input = "fasta";
        }
    };
}// namespace SeqSketch


#endif//SEQUENCE_SKETCHING_ARGS_HPP
