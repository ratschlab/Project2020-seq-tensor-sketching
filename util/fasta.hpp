#pragma once

#include "util/alphabets.hpp"
#include "util/utils.hpp"

#include <string>

namespace ts { // ts = Tensor Sketch

/**
 * Reads a fasta file and returns its contents as a tuple of <test_id, sequences, names>.
 * @tparam seq_type type used for storing a character of the fasta file, typically uint8_t
 */
template <typename seq_type>
std::tuple<std::string, Vec2D<seq_type>, Vec<std::string>>
read_fasta(const std::string &file_name, const std::string &format_input) {
    std::string test_id;
    Vec2D<seq_type> seqs;
    Vec<std::string> seq_names;
    std::ifstream infile(file_name);
    assert(infile.is_open() && ("Could not open " + file_name).c_str());

    std::string line;
    std::getline(infile, line);
    if (line[0] == '#') {
        test_id = line;
        std::getline(infile, line);
    }
    while (line[0] != '>') {
        std::getline(infile, line);
    }
    std::string name = line;
    Vec<seq_type> seq;
    while (std::getline(infile, line)) {
        if (line[0] == '>') {
            seqs.push_back(seq);
            seq_names.push_back(name);
            seq.clear();
            name = line;
        } else if (!line.empty()) {
            if (format_input == "fasta") {
                for (char c : line) {
                    seq.push_back(char2int(c));
                }
            } else if (format_input == "csv") {
                std::stringstream ss(line);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    seq.push_back(std::stoi(item, 0, 16));
                }
            } else {
                std::cerr << "Invalid input foramt: " << format_input << std::endl;
                exit(1);
            }
        }
    }
    return { test_id, seqs, seq_names };
}

} // namespace ts
