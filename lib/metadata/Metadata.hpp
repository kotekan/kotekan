/**
 * @file
 * @brief Contains Metadata class.
 */

#ifndef METADATA_HPP
#define METADATA_HPP

#include <cstdint>           // for int64_t
#include <initializer_list>  // for initializer_list
#include <iostream>          // for size_t, ostream
#include <string>            // for string
#include <type_traits>       // for enable_if_t
#include <unordered_map>     // for unordered_map
#include <vector>            // for vector

namespace kotekan {

// The Metadata class is roughly similar to a Python dictionary. It
// holds typed key/value pairs. The keys are strings, and the values
// are either boolen, integer, float, string, or a vector of these.
//
// Values cannot be overridden once set. This is a write-once data
// structure.
//
// Internally, integers and floats are stored using 64 bits.
class Metadata {
public:
    // Possible attribute value keys
    enum type {
        type_bool,
        type_int,
        type_float,
        type_string,
        type_bool_vector,
        type_int_vector,
        type_float_vector,
        type_string_vector,
    };

private:
    std::unordered_map<std::string, type> m_type;

    std::unordered_map<std::string, bool> m_bool;
    std::unordered_map<std::string, std::int64_t> m_int;
    std::unordered_map<std::string, double> m_float;
    std::unordered_map<std::string, std::string> m_string;
    std::unordered_map<std::string, std::vector<bool>> m_bool_vector;
    std::unordered_map<std::string, std::vector<std::int64_t>> m_int_vector;
    std::unordered_map<std::string, std::vector<double>> m_float_vector;
    std::unordered_map<std::string, std::vector<std::string>> m_string_vector;

public:
    Metadata();
    Metadata(Metadata&&);
    Metadata& operator=(Metadata&&);

    Metadata(const Metadata&) = default;
    Metadata& operator=(const Metadata&) = default;

    // Total number of metadata attributes
    std::size_t size() const noexcept;
    // Whether an attribute key exists
    bool has_key(const std::string& key) const noexcept;
    // All attribute keys
    std::vector<std::string> keys() const;
    // The value type of a key
    type value_type(const std::string& key) const;

    // Number of bool attributes
    std::size_t bool_size() const noexcept;
    // Number of int attributes
    std::size_t int_size() const noexcept;
    // Number of float attributes
    std::size_t float_size() const noexcept;
    // Number of string attributes
    std::size_t string_size() const noexcept;
    // Number of bool vector attributes
    std::size_t bool_vector_size() const noexcept;
    // Number of int vector attributes
    std::size_t int_vector_size() const noexcept;
    // Number of float vector attributes
    std::size_t float_vector_size() const noexcept;
    // Number of string vector attributes
    std::size_t string_vector_size() const noexcept;

    // All bool attribute keys
    std::vector<std::string> bool_keys() const;
    // All int attribute keys
    std::vector<std::string> int_keys() const;
    // All float attribute keys
    std::vector<std::string> float_keys() const;
    // All string attribute keys
    std::vector<std::string> string_keys() const;
    // All bool vector attribute keys
    std::vector<std::string> bool_vector_keys() const;
    // All int vector attribute keys
    std::vector<std::string> int_vector_keys() const;
    // All float vector attribute keys
    std::vector<std::string> float_vector_keys() const;
    // All string vector attribute keys
    std::vector<std::string> string_vector_keys() const;

    // Whether a bool attribute key exists
    bool has_bool(const std::string& key) const noexcept;
    // Whether an int attribute key exists
    bool has_int(const std::string& key) const noexcept;
    // Whether a float attribute key exists
    bool has_float(const std::string& key) const noexcept;
    // Whether a string attribute key exists
    bool has_string(const std::string& key) const noexcept;
    // Whether a bool vector attribute key exists
    bool has_bool_vector(const std::string& key) const noexcept;
    // Whether an int vector attribute key exists
    bool has_int_vector(const std::string& key) const noexcept;
    // Whether a float vector attribute key exists
    bool has_float_vector(const std::string& key) const noexcept;
    // Whether a string vector attribute key exists
    bool has_string_vector(const std::string& key) const noexcept;

    // Set a bool attribute
    void set_bool(const std::string& key, bool value);
    // Set an int attribute
    void set_int(const std::string& key, std::int64_t value);
    template<typename T>
    std::enable_if_t<std::is_integral_v<T>, void> set_int(const std::string& key, T value) {
        set_int(key, std::int64_t(value));
    }
    // Set a float attribute
    void set_float(const std::string& key, double value);
    template<typename T>
    std::enable_if_t<std::is_floating_point_v<T>, void> set_float(const std::string& key, T value) {
        set_float(key, double(value));
    }
    // Set a string attribute
    void set_string(const std::string& key, std::string value);
    void set_string(const std::string& key, const char* value);
    // Set a bool vector attribute
    void set_bool_vector(const std::string& key, std::vector<bool> value);
    // Set an int vector attribute
    void set_int_vector(const std::string& key, std::vector<std::int64_t> value);
    template<typename T>
    std::enable_if_t<std::is_integral_v<T>, void> set_int_vector(const std::string& key,
                                                                 const std::vector<T>& value) {
        set_int_vector(key, std::vector<std::int64_t>(value));
    }
    // Set a float vector attribute
    void set_float_vector(const std::string& key, std::vector<double> value);
    template<typename T>
    std::enable_if_t<std::is_floating_point_v<T>, void>
    set_float_vector(const std::string& key, const std::vector<T>& value) {
        set_float_vector(key, std::vector<double>(value));
    }
    // Set a string vector attribute
    void set_string_vector(const std::string& key, std::vector<std::string> value);
    void set_string_vector(const std::string& key, const std::vector<const char*>& value);
    void set_string_vector(const std::string& key, const std::vector<char*>& value);
    void set_string_vector(const std::string& key, const std::initializer_list<const char*>& value);

    // Get a bool attribute
    bool get_bool(const std::string& key) const;
    // Get an int attribute
    std::int64_t get_int(const std::string& key) const;
    // Get a float attribute
    double get_float(const std::string& key) const;
    // Get a string attribute
    std::string get_string(const std::string& key) const;
    // Get a bool vector attribute
    std::vector<bool> get_bool_vector(const std::string& key) const;
    // Get an int vector attribute
    std::vector<std::int64_t> get_int_vector(const std::string& key) const;
    // Get a float vector attribute
    std::vector<double> get_float_vector(const std::string& key) const;
    // Get a string vector attribute
    std::vector<std::string> get_string_vector(const std::string& key) const;

    // Output all attributes
    friend std::ostream& operator<<(std::ostream& os, const Metadata& meta);
};

} // namespace kotekan

#endif // #ifndef METADATA_HPP
