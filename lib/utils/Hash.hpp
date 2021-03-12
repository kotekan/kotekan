#ifndef _HASH_HPP
#define _HASH_HPP

#include "MurmurHash3.hpp" // for MurmurHash3_x64_128

#include <fmt.hpp>      // for formatter
#include <gsl-lite.hpp> // for span
#include <iostream>     // for istream, ostream
#include <json.hpp>     // for json
#include <stdint.h>     // for uint64_t
#include <string>       // for string

// Set a value for the hash seed
#define _SEED 1420


/**
 * @brief Container for 128-bit hashes.
 *
 * These are stored as two 64bit unsigned ints, which should be equivalent in
 * memory to a little endian 128 bit uint.
 **/
struct Hash {

    /// The low bytes
    uint64_t l = 0;

    /// The high bytes
    uint64_t h = 0;

    /**
     * @brief Load a hash from a string value.
     *
     * @param  s  The string to parse.
     * @returns   The loaded hash.
     **/
    static Hash from_string(const std::string& s);

    /**
     * @brief Set the hash from a string value.
     *
     * @param  s  The string to parse.
     **/
    void set_from_string(const std::string& s);


    /**
     * @brief Return a hexstring of the Hash.
     *
     * @returns  A string of the Hash.
     **/
    std::string to_string() const;

    /// Type for an undefined hash
    const static Hash null;
};


/**
 * @brief Hash an array-like type. Hashed by memory value.
 *
 * @param  s  The range to hash.
 *
 * @returns   The hashed value.
 **/
template<typename T>
Hash hash(gsl::span<const T> s) {
    Hash t;
    MurmurHash3_x64_128((void*)s.as_bytes(), s.size_bytes(), _SEED, (void*)&t);
    return t;
}


/**
 * @brief Hash a string.
 *
 * @param  s  The string to hash.
 *
 * @returns   The hashed value.
 **/
inline Hash hash(const std::string& s) {
    Hash t;
    MurmurHash3_x64_128((void*)s.c_str(), s.size(), _SEED, (void*)&t);
    return t;
}

/**
 * @brief Comparison of two hash types.
 *
 * @param  a  Hash a.
 * @param  b  Hash b.
 * @return    The comparison result.
 **/
inline bool operator<(const Hash& a, const Hash& b) {
    return (b.h > a.h) || ((b.h == a.h) && (b.l > a.l));
}

/**
 * @brief Test equality of two hash types.
 *
 * @param  a  Hash a.
 * @param  b  Hash b.
 * @return    The comparison result.
 **/
inline bool operator==(const Hash& a, const Hash& b) {
    return (b.h == a.h) && (b.l == a.l);
}


/**
 * @brief Test in equality of two hash types.
 *
 * @param  a  Hash a.
 * @param  b  Hash b.
 * @return    The comparison result.
 **/
inline bool operator!=(const Hash& a, const Hash& b) {
    return (b.h != a.h) || (b.l != a.l);
}


// Define a custom fmt formatter for the type
template<>
struct fmt::formatter<Hash> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const Hash& h, FormatContext& ctx) {
        return format_to(ctx.out(), "{:016x}{:016x}", h.h, h.l);
    }
};

// Define the IO stream operators
std::istream& operator>>(std::istream& is, Hash& h);
std::ostream& operator<<(std::ostream& os, const Hash& h);

// Define a to_string method to use ADL
std::string to_string(const Hash& h);

// Conversions of the index types to json
void to_json(nlohmann::json& j, const Hash& h);
void from_json(const nlohmann::json& j, Hash& h);

#endif
