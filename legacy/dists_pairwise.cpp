#include <fstream>
#include <memory>


#include "util/modules.hpp"
#include "util/seqgen.hpp"
#include "util/utils.hpp"

using namespace ts;
using namespace BasicTypes;

struct KmerModule : public BasicModules {
    int original_alphabet_size {};

    void override_pre() override {
        original_alphabet_size = alphabet_size;
        alphabet_size = int_pow<size_t>(alphabet_size, kmer_size);
    }

    void override_post() override {
        tensor_slide_params.alphabet_size = original_alphabet_size;
        tensor_slide_params.tup_len = 2;
    }
};

struct TestModule1 {
    Vec2D<int> seqs;
    std::vector<std::string> seq_names;
    string test_id;
    Vec2D<int> kmer_seqs;
    Vec2D<int> wmh_sketch;
    Vec2D<int> mh_sketch;
    Vec3D<int> omh_sketch;
    Vec2D<double> ten_sketch;
    Vec3D<int> slide_sketch;
    Vec2D<double> ten_new_sketch;
    Vec3D<double> ten_new_slide_sketch;
    Vec3D<double> dists;

    BasicModules basicModules;
    KmerModule kmerModules;
    NewModules newModules;

    void parse(int argc, char **argv) {
        basicModules.parse(argc, argv);
        basicModules.models_init();
        kmerModules.parse(argc, argv);
        kmerModules.models_init();
        newModules.parse(argc, argv);
        newModules.model_init();
    }


    template <class seq_type>
    void write_fasta(Vec2D<seq_type> &seq_vec) {
        std::ofstream fo;
        fo.open(out_path + "/seqs.fa");
        test_id = "#" + std::to_string(random());
        fo << test_id << "\n";
        for (int si = 0; si < seq_vec.size(); si++) {
            fo << "> " << si << "\n";
            for (int i = 0; i < seq_vec[i].size(); i++) {
                fo << (char)(seq_vec[si][i] + (int)'A');
            }
            fo << "\n\n";
        }
        fo.close();
    }

    template <typename seq_type>
    void read_fasta(Vec2D<seq_type> &seq_vec) {
        seq_vec.clear();
        string file = (out_path + "/seqs.fa");
        std::ifstream infile = std::ifstream(file);
        string line;

        std::getline(infile, line);
        if (line[0] == '#') {
            test_id = line;
            std::getline(infile, line);
        }
        while (line[0] != '>') {
            std::cout << line << "\n";
            std::getline(infile, line);
        }
        string name = line;
        std::vector<seq_type> seq;
        while (std::getline(infile, line)) {
            if (line[0] == '>') {
                seq_vec.push_back(seq);
                seq_names.push_back(name);
                seq.clear();
                name = line;
            } else if (!line.empty()) {
                for (char c : line) {
                    int ic = c - (int)'A';
                    seq.push_back(ic);
                }
            }
        }
    }

    void generate_sequences() {
        basicModules.seq_gen.gen_seqs(seqs);
        write_fasta(seqs);
        //        read_fasta(seqs);
    }

    void compute_sketches() {
        int num_seqs = seqs.size();
        kmer_seqs.resize(num_seqs);
        wmh_sketch.resize(num_seqs);
        mh_sketch.resize(num_seqs);
        omh_sketch.resize(num_seqs);
        ten_sketch.resize(num_seqs);
        slide_sketch.resize(num_seqs);
        ten_new_sketch.resize(num_seqs);
        ten_new_slide_sketch.resize(num_seqs);
        for (int si = 0; si < num_seqs; si++) {
            seq2kmer(seqs[si], kmer_seqs[si], basicModules.kmer_size, basicModules.alphabet_size);
            minhash(kmer_seqs[si], mh_sketch[si], kmerModules.mh_params);
            weighted_minhash(kmer_seqs[si], wmh_sketch[si], kmerModules.wmh_params);
            ordered_minhash(kmer_seqs[si], omh_sketch[si], kmerModules.omh_params);
            tensor_sketch(kmer_seqs[si], ten_sketch[si], kmerModules.tensor_params);
            tensor_slide_sketch(seqs[si], slide_sketch[si], kmerModules.tensor_slide_params);

            tensor2_sketch<int, double>(kmer_seqs[si], ten_new_sketch[si], newModules.ten_2_params);
            tensor2_slide_sketch<int, double>(kmer_seqs[si], ten_new_slide_sketch[si],
                                              newModules.ten_2_slide_params);
        }
        std::ofstream fo;
        fo.open(out_path + "/sketches_Ten2.txt");
        fo << test_id << "\n";
        for (int si = 0; si < num_seqs; si++) {
            for (int i = 0; i < ten_new_sketch[si].size(); i++) {
                fo << ten_new_sketch[si][i];
            }
            fo << "\n";
        }
        fo.close();
        fo.open(out_path + "/sketches_Ten2_slide.txt");
        fo << test_id << "\n";
        for (int si = 0; si < num_seqs; si++) {
            fo << ">> " << si << "\n";
            for (int i = 0; i < ten_new_slide_sketch[si].size(); i++) {
                for (int j = 0; j < ten_new_slide_sketch[si][i].size(); j++)
                    fo << ten_new_slide_sketch[si][i][j] << ", ";
                fo << "\n";
            }
            fo << "\n";
        }
        fo.close();
    }
    void compute_dists() {
        std::ofstream fo;
        int num_seqs = seqs.size();
        dists = new3D<double>(8, num_seqs, num_seqs, 0);
        for (int i = 0; i < seqs.size(); i++) {
            for (int j = i + 1; j < seqs.size(); j++) {
                dists[0][i][j] = edit_distance(seqs[i], seqs[j]);
                dists[1][i][j] = hamming_dist(mh_sketch[i], mh_sketch[j]);
                dists[2][i][j] = hamming_dist(wmh_sketch[i], wmh_sketch[j]);
                dists[3][i][j] = hamming_dist2D(omh_sketch[i], omh_sketch[j]);
                dists[4][i][j] = l2_sq_dist(ten_sketch[i], ten_sketch[j]);
                dists[5][i][j] = l1_dist2D_minlen(slide_sketch[i], slide_sketch[j]);
                dists[6][i][j] = l2_sq_dist(ten_new_sketch[i], ten_new_sketch[j]);
                dists[7][i][j] = l1_dist2D_minlen(ten_new_slide_sketch[i], ten_new_slide_sketch[j]);
                //                dists[6][i][j] = cosine_sim(ten_new_sketch[i], ten_new_sketch[j]);
                //                dists[6][i][j] = l1_dist(ten_new_sketch[i], ten_new_sketch[j]);
            }
        }
        std::vector<string> method_names
                = { "ED", "MH", "WMH", "OMH", "TenSketch", "TenSlide", "Ten2", "Ten2Slide" };
        for (int m = 0; m < 8; m++) {
            fo.open(out_path + "/dists_" + method_names[m] + ".txt");
            fo << test_id << "\n";
            for (int i = 0; i < num_seqs; i++) {
                for (int j = i + 1; j < num_seqs; j++) {
                    fo << i << ", " << j << ", " << dists[m][i][j] << "\n";
                }
            }
            fo.close();
        }
    }

    void save_output() {
        std::ofstream fo;
        //        fo.open("output.txt");
        fo.open(out_path + "/matlab_output.txt");
        for (int i = 0; i < seqs.size(); i++) {
            for (int j = i + 1; j < seqs.size(); j++) {
                fo << dists[0][i][j] << ", " << dists[1][i][j] << ", " << dists[2][i][j] << ", "
                   << dists[3][i][j] << ", " << dists[4][i][j] << ", " << dists[5][i][j] << ", "
                   << dists[6][i][j] << ", " << dists[7][i][j] << "\n";
            }
        }
        fo.close();
    }
};

int main(int argc, char *argv[]) {
    TestModule1 experiment;
    experiment.parse(argc, argv);
    experiment.generate_sequences();
    experiment.compute_sketches();
    experiment.compute_dists();
    experiment.save_output();
}
