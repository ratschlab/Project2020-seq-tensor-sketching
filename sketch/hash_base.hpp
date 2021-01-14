#pragma once

#include "util/timer.hpp"
#include "util/utils.hpp"

#include <algorithm>
#include <immintrin.h>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace ts {

enum class HashAlgorithm { uniform, crc32 };

HashAlgorithm parse_hash_algorithm(const std::string &name);

template <typename T>
class HashBase {
  public:
    HashBase(T set_size, size_t sketch_dim, size_t hash_size, HashAlgorithm hash_algorithm)
        : set_size(set_size),
          sketch_dim(sketch_dim),
          hash_size(2 * hash_size),
          hash_algorithm(hash_algorithm),
          hashes(std::vector<std::unordered_map<T, T>>(sketch_dim)),
          hash_values(std::vector<std::unordered_set<T>>(sketch_dim)) {
        std::random_device rd;
        rng = std::mt19937(rd());
        rand = std::uniform_int_distribution<T>(0, this->hash_size - 1);
        crc32_base = rand(rng);
    }

    void set_hashes_for_testing(const std::vector<std::unordered_map<T, T>> &h) { hashes = h; }


  protected:
    T set_size;
    size_t sketch_dim;
    size_t hash_size;

    /**
     * Returns the hash value for the index-th hash function.
     * Since the Hashes are generated on demand.
     */
    uint32_t hash(size_t index, size_t key) {
        switch (hash_algorithm) {
            case HashAlgorithm::uniform: {
                T val;
                // TODO multiple read Semaphore instead of critical
#pragma omp critical
                {
                    auto [it, inserted] = hashes[index].insert({ key, -1 });
                    if (not inserted) {
                        val = it->second;
                    } else {
                        do {
                            val = rand(rng);
                        } while (hash_values[index].find(val) != hash_values[index].end());
                        it->second = val;
                    }
                }
                assert(val >= 0 && val < hash_size
                       && " Hash values are not in [0,set_size-1] range");
                return val;
            }
            case HashAlgorithm::crc32: {
                T val = _mm_crc32_u32((unsigned int)crc32_base, (unsigned int)key);
                return _mm_crc32_u32((unsigned int)val, (unsigned int)index);
            }
            default:
                return -1;
        }
    }

  private:
    HashAlgorithm hash_algorithm;

    /** Contains the sketch_dim permutations (hashes) that are used to compute the min-hash */
    std::vector<std::unordered_map<T, T>> hashes;
    /** Contains the values used so far for each on-demand permutation */
    std::vector<std::unordered_set<T>> hash_values;
    std::uniform_int_distribution<T> rand;
    std::mt19937 rng;
    uint32_t crc32_base = 341234;
};


} // namespace ts
