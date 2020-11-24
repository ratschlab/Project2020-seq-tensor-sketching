#include <fstream>
#include <memory>

#include "util/args.hpp"
#include "util/modules.hpp"
#include "util/utils.h"

using namespace ts;

struct SeqGenModule : BasicModule {
    Vec2D<int> seqs;
    Vec<std::string> seq_names;
    string test_id;

    SeqGenModule() {
        directory = "./";
        output = "seqs.fa";
    }

    template <class seq_type>
    void write_fasta(const Vec2D<seq_type> &sequences, bool Abc = false) {
        std::ofstream fo(directory + output);
        test_id = "#" + std::to_string(random());
        fo << test_id << "\n";
        fo << "# " << argvals() << "\n";
        for (uint32_t si = 0; si < sequences.size(); si++) {
            fo << ">s" << si << "\n";
            auto &seq = sequences[si];
            for (auto &c : seq) {
                if (Abc) {
                    fo << (char)(c + (int)'A');
                } else {
                    fo << c << ",";
                }
            }
            fo << "\n\n";
        }
    }

    void generate_sequences() {
        if (mutation_pattern == "linear") {
            seq_gen.genseqs_linear(seqs);
        } else if (mutation_pattern == "tree") {
            seq_gen.genseqs_tree(seqs, sequence_seeds);
        } else {
            std::cerr << ("mutation pattern `" + mutation_pattern + "` does not exist\n");
            exit(1);
        }
        write_fasta(seqs);
    }
};

int main(int argc, char *argv[]) {
    SeqGenModule seqGen;
    seqGen.parse(argc, argv);
    if (seqGen.show_help) {
        std::cout << seqGen.description();
    } else {
        seqGen.models_init();
        seqGen.generate_sequences();
    }
}
