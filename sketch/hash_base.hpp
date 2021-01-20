#pragma once

#include "sketch/sketch_base.hpp"
#include "util/timer.hpp"
#include "util/utils.hpp"

#include <murmur_hash3.hpp>

#include <algorithm>
#include <immintrin.h>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace ts {

enum class HashAlgorithm { uniform, crc32, murmur };

HashAlgorithm parse_hash_algorithm(const std::string &name);

/**
 * @tparam T the type of elements in the hash. Can't be larger than a 32-bit int.
 *
 * //TODO(ddanciu) - this template parameter makes no sense, as CRC and Murmur always return int32
 * // remove it.
 */
template <typename T>
class HashBase : public SketchBase<std::vector<T>, true> {
  public:
    HashBase(T set_size,
             size_t sketch_dim,
             size_t hash_size,
             HashAlgorithm hash_algorithm,
             uint32_t seed,
             const std::string &name = "HashBase")
        : SketchBase<std::vector<T>, true>(name),
          set_size(set_size),
          sketch_dim(sketch_dim),
          hash_size(2 * hash_size),
          hash_algorithm(hash_algorithm),
          rand(0, this->hash_size - 1),
          rng(seed) {
        init();
    }

    void init() {
        hash_seed = rand(rng);
        hashes.assign(sketch_dim, {});
        hash_values.assign(sketch_dim, {});
    }

    void set_hashes_for_testing(const std::vector<std::unordered_map<T, T>> &h) { hashes = h; }

  protected:
    T set_size;
    size_t sketch_dim;
    size_t hash_size;

    /**
     * Returns the hash value for the given #key of the #index-th hash function.
     */
    uint32_t hash(size_t index, size_t key) {
        switch (hash_algorithm) {
            case HashAlgorithm::uniform: {
                T val;
                // TODO multiple read Semaphore instead of critical
#pragma omp critical
                {
                    auto [it, inserted] = hashes[index].insert({ key, -1 });
                    if (!inserted) {
                        val = it->second;
                    } else {
                        do {
                            val = rand(rng);
                        } while (!hash_values[index].insert(val).second);
                        it->second = val;
                    }
                }
                assert(val >= 0 && val < hash_size
                       && " Hash values are not in [0,set_size-1] range");
                return val;
            }
            case HashAlgorithm::crc32: {
                uint32_t val = _mm_crc32_u32(hash_seed, (uint32_t)key);
                return _mm_crc32_u32((uint32_t)val, (uint32_t)index);
            }
            case HashAlgorithm::murmur:
                uint32_t val;
                MurmurHash3_x86_32(&key, sizeof(key), hash_seed, &val);
                MurmurHash3_x86_32(&index, sizeof(key), val, &val);
                return val;
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
    uint32_t hash_seed;
};


} // namespace ts
