/*****************************************
@file
@brief Miscellaneous utils for the N2 pipeline.
*****************************************/
#ifndef N2_UTIL_HPP
#define N2_UTIL_HPP

#include <complex>     // for complex, imag, real
#include <cstdint>     // for uint32_t, uint16_t, int64_t, int32_t, uint64_t
#include <cstdlib>     // for size_t
#include <time.h>      // for timespec, clock_gettime

namespace N2
{

/// Define an alias for the single precision complex type
using cfloat = typename std::complex<float>;

/**
 * @brief Get the number of products for a given number of elements.
 */
inline size_t get_num_prod(size_t num_elements) {
    return num_elements * (num_elements + 1) / 2;
}

/**
 * @brief Index into a flattened upper matrix triangle.
 * @param  i Row index.
 * @param  j Column index.
 * @param  n Size of matrix.
 * @return   Index into flattend matrix.
 */
inline uint32_t cmap(uint32_t i, uint32_t j, uint32_t n) {
    return (n * (n + 1) / 2) - ((n - i) * (n - i + 1) / 2) + (j - i);
}

/**
 * @brief Product index map type.
 */
struct prod_ctype {
    /// Index of input A
    uint16_t input_a;
    /// Index of input B
    uint16_t input_b;
};

/**
 * @brief Convert a product index to an input pair.
 * @param k Product index.
 * @param n Total number of inputs.
 * @return Product pair indices.
 *
 * @todo This is super inefficient.
 **/
inline prod_ctype icmap(uint32_t k, uint16_t n) {
    uint16_t ii;
    for (ii = 0; ii < n; ii++) {
        if (cmap(ii, n - 1, n) >= k) {
            break;
        }
    }

    uint16_t j = k - cmap(ii, ii, n) + ii;
    return {ii, j};
}

/**
 * Get the index of a particular product into the GPU blocked output.
 * @param  i     Row index.
 * @param  j     Column index.
 * @param  block Block size.
 * @param  N     Number if inputs.
 * @return       Index into blocked array.
 */
inline uint32_t prod_index(uint32_t i, uint32_t j, uint32_t block, uint32_t N) {
    uint32_t num_blocks1 = ((N - 1) / block) + 1; // Blocks needed to tile 1D
    uint32_t b_ix = cmap(i / block, j / block, num_blocks1);

    return block * block * b_ix + (i % block) * block + (j % block);
}


/**
 * @brief Convert timespec type into total nanoseconds as an uint64.
 * @param  ts Time as timespec.
 * @return    Time as an uint64.
 */
inline uint64_t ts_to_uint64(const timespec& ts) {
    return 1000000000L * (uint64_t) ts.tv_sec + (uint64_t) ts.tv_nsec;
}

/**
 * @brief A class for modular arithmetic. Used for holding ring buffer indices.
 *
 * This implements comparison and arithmetic operators for modular arithmetic.
 *
 * @note The binary arithmetic operators only work adding/subtracting normal numbers
 *       to a modular number. They are also *asymmetric*.
 **/
template<typename T>
class modulo {

public:
    // Use an unsigned type for the base
    using Tu = typename std::make_unsigned<T>::type;

    /**
     * @brief Create a new modular number.
     **/
    modulo(Tu n) : _n(n){};

    // Default constructor
    modulo() : modulo(0){};

    /// Assignment of a number into the modular number.
    modulo<T>& operator=(const T& i) {
        _i = i;
        return *this;
    }

    // Increment and decrement
    modulo<T>& operator++() {
        _i++;
        return *this;
    }
    modulo<T>& operator--() {
        _i--;
        return *this;
    }
    modulo<T> operator++(int) {
        modulo<T> t(*this);
        operator++();
        return t;
    }
    modulo<T> operator--(int) {
        modulo<T> t(*this);
        operator--();
        return t;
    }

    template<typename V, typename std::enable_if_t<std::is_integral<V>::value>* = nullptr>
    modulo<T>& operator+=(const V& rhs) {
        _i += rhs;
        return *this;
    }

    template<typename V, typename std::enable_if_t<std::is_integral<V>::value>* = nullptr>
    modulo<T>& operator-=(const V& rhs) {
        _i -= rhs;
        return *this;
    }

    // Add and subtract are *asymmetric*. Must be always be modulo<T> +/- T
    template<typename V, typename std::enable_if_t<std::is_integral<V>::value>* = nullptr>
    friend modulo<T> operator+(modulo<T> lhs, const V& rhs) {
        modulo<T> t(lhs);
        t += rhs;
        return t;
    }
    template<typename V, typename std::enable_if_t<std::is_integral<V>::value>* = nullptr>
    friend modulo<T> operator-(modulo<T> lhs, const V& rhs) {
        modulo<T> t(lhs);
        t -= rhs;
        return t;
    }

    // Comparisons are always false if the bases don't match
    friend bool operator==(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() == rhs.norm());
    }
    friend bool operator!=(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() != rhs.norm());
    }
    friend bool operator<(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() < rhs.norm());
    }
    friend bool operator>(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() > rhs.norm());
    }
    friend bool operator<=(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() <= rhs.norm());
    }
    friend bool operator>=(const modulo<T>& lhs, const modulo<T>& rhs) {
        return (lhs._n == rhs._n) && (lhs.norm() >= rhs.norm());
    }

    /**
     * @brief Return the normalised modular number.
     *
     * @returns The modular number.
     **/
    T norm() const {
        return _i % _n;
    }

    /// Conversion back to type T
    operator T() const {
        return norm();
    }

private:
    // Internally we don't actually keep bother mod'ing the number when
    // we do arithmetic, only at output time.
    T _i = 0;

    // The modular base.
    Tu _n;
};

/// Stream output for modular types
template<typename T>
std::ostream& operator<<(std::ostream& os, const modulo<T>& m) {
    return (os << m.norm());
}

/**
 * @brief Class to hold buffer indices.
 **/
class frameID : public modulo<int> {
public:
    /**
     * @brief Create a frameID for a given buffer.
     *
     * @param buf   Buffer to use.
     **/
    frameID(const Buffer* buf) : modulo<int>(buf->num_frames) {}
};

/**
 * @brief Return the next aligned location for a given type size
 * @param  offset Start offset.
 * @param  size   Item size.
 * @return        Next aligned offset.
 */
inline size_t member_alignment(size_t offset, size_t size) {
    return (((size - (offset % size)) % size) + offset);
}

} // N2

/**
 * @brief FMT formatter that casts frameIDs to int so that `format("{:d}", frame_id)` works.
 */
namespace fmt {
template<>
struct formatter<N2::frameID> : formatter<int> {
    auto format(const N2::frameID id, format_context& ctx) {
        return formatter<int>::format((int)id, ctx);
    }
};
} // namespace fmt


#endif
