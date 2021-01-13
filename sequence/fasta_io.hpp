#pragma once

#include "sequence/alphabets.hpp"
#include "util/utils.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

namespace ts { // ts = Tensor Sketch

/**
 * Reads a fasta file and returns its contents as a tuple of <test_id, sequences, names>.
 * @tparam seq_type type used for storing a character of the fasta file, typically uint8_t
 */
template <typename seq_type>
std::pair<Vec2D<seq_type>, std::vector<std::string>> read_fasta(const std::string &file_name,
                                                                const std::string &input_format) {
    if (!std::filesystem::exists(file_name)) {
        std::cerr << "Input file does not exist: " << file_name << std::endl;
        std::exit(1);
    }
    std::string test_id;
    Vec2D<seq_type> seqs;
    std::vector<std::string> seq_names;
    std::ifstream infile(file_name);
    if (!infile.is_open()) {
        std::cout << "Could not open " + file_name << std::endl;
        std::exit(1);
    }
    std::string line;
    std::vector<seq_type> seq;
    while (std::getline(infile, line)) {
        if (line[0] == '>') {
            seq_names.push_back(line);
        } else if (!line.empty()) {
            if (input_format == "fasta") {
                for (char c : line) {
                    seq.push_back(char2int(c));
                }
                seqs.push_back(seq);
            } else if (input_format == "csv") {
                std::stringstream ss(line);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    seq.push_back(std::stoi(item, 0, 16));
                }
                seq_names.push_back("seq" + std::to_string(seqs.size()));
                seqs.push_back(seq);
            } else {
                std::cerr << "Invalid input foramt: " << input_format << std::endl;
                exit(1);
            }
            seq.clear();
        }
    }
    assert(seqs.size() == seq_names.size());
    return { seqs, seq_names };
}

template <class seq_type>
void write_fasta(const std::string &file_name, const Vec2D<seq_type> &sequences, bool Abc = false) {
    std::ofstream fo(file_name);
    fo << "#" + std::to_string(random()) << std::endl;
    fo << "# " << flag_values(' ') << std::endl;
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

} // namespace ts
