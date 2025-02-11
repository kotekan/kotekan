#ifndef METADATA_HPP
#define METADATA_HPP

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

class Metadata {
    std::map<std::string, bool> m_bool;
    std::map<std::string, std::int64_t> m_int;
    std::map<std::string, double> m_real;
    std::map<std::string, std::string> m_string;
    std::map<std::string, std::vector<bool>> m_bool_vector;
    std::map<std::string, std::vector<std::int64_t>> m_int_vector;
    std::map<std::string, std::vector<double>> m_real_vector;
    std::map<std::string, std::vector<std::string>> m_string_vector;

public:
    Metadata();
    Metadata(Metadata&&);
    Metadata& operator=(Metadata&&);

    Metadata(const Metadata&) = delete;
    Metadata& operator=(const Metadata&) = delete;

    std::size_t size() const noexcept;

    std::size_t bool_size() const noexcept;
    std::size_t int_size() const noexcept;
    std::size_t real_size() const noexcept;
    std::size_t string_size() const noexcept;
    std::size_t bool_vector_size() const noexcept;
    std::size_t int_vector_size() const noexcept;
    std::size_t real_vector_size() const noexcept;
    std::size_t string_vector_size() const noexcept;

    std::vector<std::string> bool_keys() const;
    std::vector<std::string> int_keys() const;
    std::vector<std::string> real_keys() const;
    std::vector<std::string> string_keys() const;
    std::vector<std::string> bool_vector_keys() const;
    std::vector<std::string> int_vector_keys() const;
    std::vector<std::string> real_vector_keys() const;
    std::vector<std::string> string_vector_keys() const;

    bool has_bool(const std::string& key) const noexcept;
    bool has_int(const std::string& key) const noexcept;
    bool has_real(const std::string& key) const noexcept;
    bool has_string(const std::string& key) const noexcept;
    bool has_bool_vector(const std::string& key) const noexcept;
    bool has_int_vector(const std::string& key) const noexcept;
    bool has_real_vector(const std::string& key) const noexcept;
    bool has_string_vector(const std::string& key) const noexcept;

    void set_bool(const std::string& key, bool value);
    void set_int(const std::string& key, std::int64_t value);
    template<typename T>
    std::enable_if_t<std::is_integral_v<T>, void> set_int(const std::string& key, T value) {
        set_int(key, std::int64_t(value));
    }
    void set_real(const std::string& key, double value);
    template<typename T>
    std::enable_if_t<std::is_floating_point_v<T>, void> set_real(const std::string& key, T value) {
        set_real(key, double(value));
    }
    void set_string(const std::string& key, std::string value);
    void set_string(const std::string& key, const char* value);
    void set_bool_vector(const std::string& key, std::vector<bool> value);
    void set_int_vector(const std::string& key, std::vector<std::int64_t> value);
    template<typename T>
    std::enable_if_t<std::is_integral_v<T>, void> set_int_vector(const std::string& key,
                                                                 const std::vector<T>& value) {
        set_int_vector(key, std::vector<std::int64_t>(value));
    }
    void set_real_vector(const std::string& key, std::vector<double> value);
    template<typename T>
    std::enable_if_t<std::is_floating_point_v<T>, void>
    set_real_vector(const std::string& key, const std::vector<T>& value) {
        set_real_vector(key, std::vector<double>(value));
    }
    void set_string_vector(const std::string& key, std::vector<std::string> value);
    void set_string_vector(const std::string& key, const std::vector<const char*>& value);
    void set_string_vector(const std::string& key, const std::vector<char*>& value);
    void set_string_vector(const std::string& key, const std::initializer_list<const char*>& value);

    bool get_bool(const std::string& key) const;
    std::int64_t get_int(const std::string& key) const;
    double get_real(const std::string& key) const;
    std::string get_string(const std::string& key) const;
    std::vector<bool> get_bool_vector(const std::string& key) const;
    std::vector<std::int64_t> get_int_vector(const std::string& key) const;
    std::vector<double> get_real_vector(const std::string& key) const;
    std::vector<std::string> get_string_vector(const std::string& key) const;

    friend std::ostream& operator<<(std::ostream& os, const Metadata& meta);
};

#endif // #ifndef METADATA_HPP
