/**
 * @file
 * @brief JSON helpers.
 *  - json_base64_to_array
 */

#ifndef JSON_UTIL_HPP
#define JSON_UTIL_HPP

#include "modp_b64.hpp" // for modp_b64_decode, modp_b64_decode_len, MODP_B64_ERROR

#include "fmt.hpp"  // for format
#include "json.hpp" // for json

#include <cstdint>   // for uint32_t, uint64_t
#include <cstring>   // for memcpy
#include <stdexcept> // for runtime_error
#include <utility>   // for pair, move
#include <vector>    // for vector

/**
 * @brief Decode a JSON-serialized array `{"shape": [...], "data": "<base64>"}`.
 *
 * `data` is the base64 encoding of the array's bytes in row-major order, with the value type
 * given by `T`; `shape` gives the extents. The caller interprets `dtype` or `encoding` keys.
 *
 * @return The values, flat and row-major, and the shape.
 * @throws std::runtime_error if the data is not base64, or its length does not match the shape.
 */
template<typename T>
std::pair<std::vector<T>, std::vector<uint32_t>> json_base64_to_array(const nlohmann::json& j) {

    auto shape = j.at("shape").get<std::vector<uint32_t>>();
    const auto& data_json = j.at("data");

    if (!data_json.is_string()) {
        throw std::runtime_error("Value for data key must be a string.");
    }

    auto string_data = data_json.get_ref<const nlohmann::json::string_t&>();
    auto enc_size = string_data.size();

    // Must be a multiple of 4 bytes to be base64 decodable
    if (enc_size % 4 != 0) {
        throw std::runtime_error(fmt::format(
            "base64 encoded data must be a multiple of four bytes. Length: {}.", enc_size));
    }

    // Decode the base64 data into a temporary buffer
    std::vector<char> decoded_data(modp_b64_decode_len(enc_size));
    auto dec_len = modp_b64_decode(decoded_data.data(), string_data.c_str(), string_data.size());

    if (dec_len == MODP_B64_ERROR) {
        throw std::runtime_error("Could not base64 decode the array data.");
    }

    // Calculate the total array size
    uint64_t size = 1;
    for (const auto& axsize : shape)
        size *= axsize;
    auto size_bytes = size * sizeof(T);

    if (dec_len != size_bytes) {
        throw std::runtime_error(fmt::format("Size of decoded data different from expectation."
                                             "Got {}, expected {}",
                                             dec_len, size_bytes));
    }

    // Copy data into a vector of the correct length and alignment,
    // (decoded_data likely has padding bytes at the end)
    std::vector<T> arr(size);
    std::memcpy(arr.data(), decoded_data.data(), size_bytes);

    return {std::move(arr), std::move(shape)};
}

#endif // JSON_UTIL_HPP
