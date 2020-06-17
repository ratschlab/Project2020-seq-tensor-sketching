//
// Created by Amir Joudaki on 6/17/20.
//

#ifndef SEQUENCE_SKETCHING_ARGS_HPP
#define SEQUENCE_SKETCHING_ARGS_HPP
#include <string>

struct basic_args {
    int argc;
    char **argv;

    basic_args(int argc, char *argv[]) : argc(argc), argv(argv) {}

};


std::string to_string(int argc, char* argv[]) {
    std::string s;
    for (int i = 0; i < argc; i++) {
        s += argv[i];
        s += " ";
    }
    return s;
}


struct seq_args : public basic_args {
    bool fix_len = true;
    int sig_len = 4,
        max_blocks = 4,
        num_seqs = 200,
        seq_len = 256;
    double mutation_rate = .02,
           block_mutate_rate = .05;

    seq_args(int argc, char *argv[]) : basic_args(argc, argv) {
        for (int i = 0; i < argc; i++) {
            if (std::strcmp(argv[i], "--num_seqs") == 0) {// number of sequences
                num_seqs = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--sig_len") == 0 or std::strcmp(argv[i], "-A") == 0) {// alphabet size
                sig_len = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--seq_len") == 0) {
                seq_len = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--mutation_rate") == 0) {
                mutation_rate = std::stod(argv[++i]);
            } else if (std::strcmp(argv[i], "--block_mutate_rate") == 0) {
                block_mutate_rate = std::stod(argv[++i]);
            } else if (std::strcmp(argv[i], "--max_blocks") == 0) {
                max_blocks = std::stoi(argv[++i]);
            }
        }
    }
};

struct kmer_args : public basic_args {
    int kmer_size = 2;
    kmer_args(int argc, char *argv[]) : basic_args(argc, argv) {
        for (int i = 0; i < argc; i++) {
            if (std::strcmp(argv[i], "--kmer_size") == 0 or std::strcmp(argv[i], "-K") == 0) {
                kmer_size = std::stoi(argv[++i]);
            }
        }
    }
};

struct embed_args : public basic_args {
    int embed_dim = 200;
    embed_args(int argc, char *argv[]) : basic_args(argc, argv) {
        for (int i = 0; i < argc; i++) {
            if (std::strcmp(argv[i], "--embed_dim") == 0 or std::strcmp(argv[i], "-M") == 0) {
                embed_dim = std::stoi(argv[++i]);
            }
        }
    }
};

struct tuple_embed_args : public embed_args {
    int tup_len = 2;
    tuple_embed_args(int argc, char *argv[]) : embed_args(argc, argv) {
        for (int i = 0; i < argc; i++) {
            if (std::strcmp(argv[i], "--tup_len") == 0 or std::strcmp(argv[i], "-T") == 0) {
                tup_len = std::stoi(argv[++i]);
            }
        }
    }
};

struct tensor_embed_args : public tuple_embed_args {
    int num_phases = 5, num_bins = 255;
    tensor_embed_args(int argc, char *argv[]) : tuple_embed_args(argc, argv) {
        for (int i = 0; i < argc; i++) {
            if (std::strcmp(argv[i], "--num_phases") == 0) {
                num_phases = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--num_bins") == 0) {
                num_bins = std::stoi(argv[++i]);
            }
        }
    }
};

struct tensor_slide_args : public tensor_embed_args {
    int win_len = 32, stride = 8;
    tensor_slide_args(int argc, char *argv[]) : tensor_embed_args(argc, argv) {
        for (int i = 0; i < argc; i++) {
            if (std::strcmp(argv[i], "--win_len") == 0) {
                win_len = std::stoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--stride") == 0) {
                stride = std::stoi(argv[++i]);
            }
        }
    }
};

struct tensor_embed_opts : public tensor_embed_args, public seq_args {
    tensor_embed_opts(int argc, char *argv[]) : tensor_embed_args(argc, argv), seq_args(argc, argv) {}
};

struct tensor_slide_opts : public tensor_slide_args, public seq_args {
    tensor_slide_opts(int argc, char *argv[]) : tensor_slide_args(argc, argv), seq_args(argc, argv) {}
};

struct omp_embed_opts : public tuple_embed_args, public seq_args {
    omp_embed_opts(int argc, char *argv[]) : tuple_embed_args(argc, argv), seq_args(argc, argv) {}
};

struct minhash_opts : public kmer_args, public seq_args {
    minhash_opts(int argc, char *argv[]) : kmer_args(argc, argv), seq_args(argc, argv) {}
};



#endif//SEQUENCE_SKETCHING_ARGS_HPP
