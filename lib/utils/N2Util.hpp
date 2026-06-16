/*****************************************
@file
@brief Miscellaneous utils for the N2 pipeline.
*****************************************/
#ifndef N2_UTIL_HPP
#define N2_UTIL_HPP

#include "buffer.hpp"
#include "div.hpp" // for div_ceil
#include "timeUtil.hpp"

#include <cmath>
#include <complex> // for complex, imag, real
#include <cstdint> // for uint32_t, uint16_t, int64_t, int32_t, uint64_t
#include <cstdlib> // for size_t
#include <time.h>  // for timespec, clock_gettime

namespace N2 {

/// Define an alias for the single precision complex type
using cfloat = typename std::complex<float>;

/**
 * @brief Frequency index map type for the N2 pipeline.
 */
struct freq_ctype {
    /// Centre of frequency channel in MHz
    double centre;
    /// Width of frequency channel in MHz
    double width;
};

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
 *
 * input_a is the row index into the full visibility matrix, and input_b is the column index into
 * the full visibility matrix.  That is, for visibility matrix entry V_{ij}, input_a = i and input_b
 * = j.
 */
struct prod_ctype {
    /// Index of input A
    uint16_t input_a;
    /// Index of input B
    uint16_t input_b;

    /// Equality comparison
    bool operator==(const prod_ctype& other) const {
        return input_a == other.input_a && input_b == other.input_b;
    }
};

/**
 * @brief Convert a product index to an input pair.
 * @param k Product index.
 * @param n Total number of inputs.
 * @return Product pair indices.
 *
 * @details Given an index k into a flattened upper-triangular matrix of size n x n, return the
 *  corresponding row and column indices (i, j) such that i <= j. Uses an approximate
 *  triangular root, then iteratively corrects for roundoff.
 **/
inline prod_ctype icmap(uint32_t k, uint16_t n) {
    // Calculate number of elements left from k to the end of the triangular matrix
    uint64_t rem = (uint64_t)n * (n + 1) / 2 - k;

    // Use triangular root to estimate height of the remaining triangle
    uint64_t x = (uint64_t)((std::sqrt((long double)8 * rem) + 1) / 2);

    // Lambda to compute Triangular number T(t) = t*(t+1)/2
    auto T = [](uint64_t t) { return t * (t + 1) / 2; };
    // If estimate is too high (the triangle below x-1 is still >= rem), decrease x
    while (x > 0 && T(x - 1) >= rem)
        --x;
    // If estimate is too low (the triangle at x is smaller than rem), increase x
    while (T(x) < rem)
        ++x;

    // Convert x back to row i and column j
    uint16_t i = n - (uint16_t)x;
    uint16_t j = i + (uint16_t)(T(x) - rem);

    return {i, j};
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
    uint32_t num_blocks1 = kotekan::div_ceil(N, block); // Blocks needed to tile 1D
    uint32_t b_ix = cmap(i / block, j / block, num_blocks1);

    return block * block * b_ix + (i % block) * block + (j % block);
}

/**
 * @brief Correlator input type
 */
struct input_ctype {
    input_ctype(uint16_t idx, std::string input_id) {
        // Check input_id length
        if (input_id.size() >= 32) {
            ERROR_NON_OO("input_id {} length exceeds 31 characters", input_id);
        }
        element_idx = idx;
        std::memset(input_id_str, 0, 32);
        input_id.copy(input_id_str, 31); // Ensure null termination
    }

    /// element index
    uint16_t element_idx;
    /// input identifier
    char input_id_str[32];
};


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

/**
 * @class movingAverage
 * @brief Calculate an exponentially weighted moving average of a time series.
 *
 * @author Richard Shaw
 **/
class movingAverage {

public:
    /**
     * @brief Create a moving average calculation.
     *
     * @param  length  The length scale to average over. This is defined as
     *                 the lag at which all newer samples carry the same weight as all earlier
     *                 samples. Or equivalently the distance at which the weight per sample has
     *                 decreased by a factor of two.
     **/
    movingAverage(double length = 4.0) {
        // Calculate the coefficient for the moving average as a halving of the weight
        alpha = 1.0 - pow(2, -1.0 / length);
    }

    /**
     * @brief Add a new sample in the time series.
     *
     * @param  value  The sample to add.
     **/
    void add_sample(uint64_t value) {

        // Special case for the first sample.
        if (!initialised) {
            current_value = value;
            initialised = true;
        } else {
            current_value = alpha * value + (1 - alpha) * current_value;
        }
    }

    /**
     * @brief Return the moving average of the current set of samples.
     *
     * @returns  The current moving average.
     **/
    double average() {
        if (!initialised) {
            return NAN;
        }
        return current_value;
    }

private:
    double current_value;
    double alpha;

    bool initialised = false;
};

/**
 * @brief Get the current time in nanoseconds.
 *
 * @returns  The current time in nanoseconds.
 **/
inline int64_t current_system_time_ns() {
    timespec output_ts;
    timespec_get(&output_ts, TIME_UTC);
    return timespec_to_nanosec_i64(output_ts);
}

} // namespace N2

// Include json.hpp for JSON serialization support
#include "json.hpp"

// ADL-compatible JSON serialization for N2::prod_ctype
// Must be in the N2 namespace for ADL to work with nlohmann::json
namespace N2 {

inline void to_json(nlohmann::json& j, const prod_ctype& p) {
    j = nlohmann::json::array({p.input_a, p.input_b});
}

inline void from_json(const nlohmann::json& j, prod_ctype& p) {
    p.input_a = j.at(0).get<uint16_t>();
    p.input_b = j.at(1).get<uint16_t>();
}

} // namespace N2

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
