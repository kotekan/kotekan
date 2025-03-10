#include <Metadata.hpp>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>

namespace chord {

Metadata::Metadata() = default;
Metadata::Metadata(Metadata&&) = default;
Metadata& Metadata::operator=(Metadata&&) = default;

std::size_t Metadata::size() const noexcept {
    return bool_size() + int_size() + float_size() + string_size() + bool_vector_size()
           + int_vector_size() + float_vector_size() + string_vector_size();
}
bool Metadata::has_key(const std::string& key) const noexcept {
    return m_type.count(key);
}
std::vector<std::string> Metadata::keys() const {
    std::vector<std::string> keys1;
    keys1.reserve(m_type.size());
    // Ubuntu 18 outputs a spurious fatal warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    for (const auto& [key, _] : m_type)
        keys1.push_back(key);
#pragma GCC diagnostic pop
    return keys1;
}
Metadata::type Metadata::value_type(const std::string& key) const {
    assert(has_key(key));
    return m_type.at(key);
}

std::size_t Metadata::bool_size() const noexcept {
    return m_bool.size();
}
std::size_t Metadata::int_size() const noexcept {
    return m_int.size();
}
std::size_t Metadata::float_size() const noexcept {
    return m_float.size();
}
std::size_t Metadata::string_size() const noexcept {
    return m_string.size();
}
std::size_t Metadata::bool_vector_size() const noexcept {
    return m_bool_vector.size();
}
std::size_t Metadata::int_vector_size() const noexcept {
    return m_int_vector.size();
}
std::size_t Metadata::float_vector_size() const noexcept {
    return m_float_vector.size();
}
std::size_t Metadata::string_vector_size() const noexcept {
    return m_string_vector.size();
}

bool Metadata::has_bool(const std::string& key) const noexcept {
    return m_bool.count(key);
}
bool Metadata::has_int(const std::string& key) const noexcept {
    return m_int.count(key);
}
bool Metadata::has_float(const std::string& key) const noexcept {
    return m_float.count(key);
}
bool Metadata::has_string(const std::string& key) const noexcept {
    return m_string.count(key);
}
bool Metadata::has_bool_vector(const std::string& key) const noexcept {
    return m_bool_vector.count(key);
}
bool Metadata::has_int_vector(const std::string& key) const noexcept {
    return m_int_vector.count(key);
}
bool Metadata::has_float_vector(const std::string& key) const noexcept {
    return m_float_vector.count(key);
}
bool Metadata::has_string_vector(const std::string& key) const noexcept {
    return m_string_vector.count(key);
}

// Ubuntu 18 outputs a spurious fatal warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
std::vector<std::string> Metadata::bool_keys() const {
    std::vector<std::string> keys;
    keys.reserve(m_bool.size());
    for (const auto& [key, _] : m_bool)
        keys.push_back(key);
    return keys;
}
std::vector<std::string> Metadata::int_keys() const {
    std::vector<std::string> keys;
    keys.reserve(m_int.size());
    for (const auto& [key, _] : m_int)
        keys.push_back(key);
    return keys;
}
std::vector<std::string> Metadata::float_keys() const {
    std::vector<std::string> keys;
    keys.reserve(m_float.size());
    for (const auto& [key, _] : m_float)
        keys.push_back(key);
    return keys;
}
std::vector<std::string> Metadata::string_keys() const {
    std::vector<std::string> keys;
    keys.reserve(m_string.size());
    for (const auto& [key, _] : m_string)
        keys.push_back(key);
    return keys;
}
std::vector<std::string> Metadata::bool_vector_keys() const {
    std::vector<std::string> keys;
    keys.reserve(m_bool_vector.size());
    for (const auto& [key, _] : m_bool_vector)
        keys.push_back(key);
    return keys;
}
std::vector<std::string> Metadata::int_vector_keys() const {
    std::vector<std::string> keys;
    keys.reserve(m_int_vector.size());
    for (const auto& [key, _] : m_int_vector)
        keys.push_back(key);
    return keys;
}
std::vector<std::string> Metadata::float_vector_keys() const {
    std::vector<std::string> keys;
    keys.reserve(m_float_vector.size());
    for (const auto& [key, _] : m_float_vector)
        keys.push_back(key);
    return keys;
}
std::vector<std::string> Metadata::string_vector_keys() const {
    std::vector<std::string> keys;
    keys.reserve(m_string_vector.size());
    for (const auto& [key, _] : m_string_vector)
        keys.push_back(key);
    return keys;
}
#pragma GCC diagnostic push

void Metadata::set_bool(const std::string& key, bool value) {
    assert(!has_key(key));
    m_type[key] = type_bool;
    m_bool[key] = value;
}
void Metadata::set_int(const std::string& key, std::int64_t value) {
    assert(!has_key(key));
    m_type[key] = type_int;
    m_int[key] = value;
}
void Metadata::set_float(const std::string& key, double value) {
    assert(!has_key(key));
    m_type[key] = type_float;
    m_float[key] = value;
}
void Metadata::set_string(const std::string& key, std::string value) {
    assert(!has_key(key));
    m_type[key] = type_string;
    m_string[key] = std::move(value);
}
void Metadata::set_string(const std::string& key, const char* value) {
    set_string(key, std::string(value));
}
void Metadata::set_bool_vector(const std::string& key, std::vector<bool> value) {
    assert(!has_key(key));
    m_type[key] = type_bool_vector;
    m_bool_vector[key] = std::move(value);
}
void Metadata::set_int_vector(const std::string& key, std::vector<std::int64_t> value) {
    assert(!has_key(key));
    m_type[key] = type_int_vector;
    m_int_vector[key] = std::move(value);
}
void Metadata::set_float_vector(const std::string& key, std::vector<double> value) {
    assert(!has_key(key));
    m_type[key] = type_float_vector;
    m_float_vector[key] = std::move(value);
}
void Metadata::set_string_vector(const std::string& key, std::vector<std::string> value) {
    assert(!has_key(key));
    m_type[key] = type_string_vector;
    m_string_vector[key] = std::move(value);
}
void Metadata::set_string_vector(const std::string& key, const std::vector<const char*>& value) {
    std::vector<std::string> value2(value.size());
    std::copy(value.begin(), value.end(), value2.begin());
    set_string_vector(key, std::move(value2));
}
void Metadata::set_string_vector(const std::string& key, const std::vector<char*>& value) {
    std::vector<std::string> value2(value.size());
    std::copy(value.begin(), value.end(), value2.begin());
    set_string_vector(key, std::move(value2));
}
void Metadata::set_string_vector(const std::string& key,
                                 const std::initializer_list<const char*>& value) {
    std::vector<std::string> value2(value.size());
    std::copy(value.begin(), value.end(), value2.begin());
    set_string_vector(key, std::move(value2));
}

bool Metadata::get_bool(const std::string& key) const {
    assert(has_bool(key));
    return m_bool.at(key);
}
std::int64_t Metadata::get_int(const std::string& key) const {
    assert(has_int(key));
    return m_int.at(key);
}
double Metadata::get_float(const std::string& key) const {
    assert(has_float(key));
    return m_float.at(key);
}
std::string Metadata::get_string(const std::string& key) const {
    assert(has_string(key));
    return m_string.at(key);
}
std::vector<bool> Metadata::get_bool_vector(const std::string& key) const {
    assert(has_bool_vector(key));
    return m_bool_vector.at(key);
}
std::vector<std::int64_t> Metadata::get_int_vector(const std::string& key) const {
    assert(has_int_vector(key));
    return m_int_vector.at(key);
}
std::vector<double> Metadata::get_float_vector(const std::string& key) const {
    assert(has_float_vector(key));
    return m_float_vector.at(key);
}
std::vector<std::string> Metadata::get_string_vector(const std::string& key) const {
    assert(has_string_vector(key));
    return m_string_vector.at(key);
}

namespace {

const char* format_bool(bool value) {
    return value ? "true" : "false";
}

template<typename T>
std::enable_if_t<std::is_integral_v<T>, T> format_int(T value) {
    // No special formatting required
    return value;
}

template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, std::string> format_float(T value) {
    // Use a stringstream to avoid changing the properties of the output stream `os`
    std::ostringstream buf;
    buf << std::setprecision(std::numeric_limits<T>::max_digits10) << std::showpoint << value;
    return buf.str();
}

std::string format_string(const char* value) {
    // Use a stringstream to avoid changing the properties of the output stream `os`
    std::ostringstream buf;
    buf << '"';
    for (const char* p = value; *p; ++p) {
        const char ch = *p;
        if (ch > 0 && ch < 32) {
            switch (ch) {
                case '\a':
                    buf << "\\a";
                    break;
                case '\b':
                    buf << "\\b";
                    break;
                    // '\e' is not part of the standard
                case '\f':
                    buf << "\\f";
                    break;
                case '\n':
                    buf << "\\n";
                    break;
                case '\r':
                    buf << "\\r";
                    break;
                case '\t':
                    buf << "\\t";
                    break;
                case '\v':
                    buf << "\\v";
                    break;
                    // ignoring '\?' because we don't care about trigraphs
                default:
                    // can't use hex because hex doesn't terminate after 2 digits
                    buf << "\\" << std::oct << std::setfill('0') << std::setw(3) << ch;
            }
        } else {
            switch (ch) {
                case '"':
                case '\\':
                    buf << '\\' << ch;
                    // fall-through
                default:
                    // this will do the right thing for UTF-8 as well
                    buf << ch;
            }
        }
    }
    buf << '"';
    return buf.str();
}
std::string format_string(const std::string& value) {
    return format_string(value.c_str());
}

bool is_valid_identifier(const char* value) {
    const char* p = value;
    if (!*p)
        // empty string is not a valid identifier
        return false;
    const char ch = *p++;
    if (!isalpha(ch) && ch != '_')
        return false;
    while (*p) {
        const char ch = *p++;
        if (!isalnum(ch) && ch != '_')
            return false;
    }
    return true;
}
bool is_valid_identifier(const std::string& value) {
    return is_valid_identifier(value.c_str());
}
std::string format_key(const std::string& value) {
    if (is_valid_identifier(value))
        return value;
    return format_string(value);
}
[[maybe_unused]] std::string format_key(const char* value) {
    if (is_valid_identifier(value))
        return value;
    return format_string(value);
}

} // namespace

std::ostream& operator<<(std::ostream& os, const Metadata& meta) {
    for (const auto& key : meta.bool_keys())
        os << format_key(key) << ": " << format_bool(meta.get_bool(key)) << "\n";
    for (const auto& key : meta.int_keys())
        os << format_key(key) << ": " << format_int(meta.get_int(key)) << "\n";
    for (const auto& key : meta.float_keys())
        os << format_key(key) << ": " << format_float(meta.get_float(key)) << "\n";
    for (const auto& key : meta.string_keys())
        os << format_key(key) << ": " << format_string(meta.get_string(key)) << "\n";
    for (const auto& key : meta.bool_vector_keys()) {
        os << format_key(key) << ": [";
        bool isfirst = true;
        for (const auto& value : meta.get_bool_vector(key))
            os << (isfirst ? "" : ", ") << format_bool(value), isfirst = false;
        os << "]\n";
    }
    for (const auto& key : meta.int_vector_keys()) {
        os << format_key(key) << ": [";
        bool isfirst = true;
        for (const auto& value : meta.get_int_vector(key))
            os << (isfirst ? "" : ", ") << format_int(value), isfirst = false;
        os << "]\n";
    }
    for (const auto& key : meta.float_vector_keys()) {
        os << format_key(key) << ": [";
        bool isfirst = true;
        for (const auto& value : meta.get_float_vector(key))
            os << (isfirst ? "" : ", ") << format_float(value), isfirst = false;
        os << "]\n";
    }
    for (const auto& key : meta.string_vector_keys()) {
        os << format_key(key) << ": [";
        bool isfirst = true;
        for (const auto& value : meta.get_string_vector(key))
            os << (isfirst ? "" : ", ") << format_string(value), isfirst = false;
        os << "]\n";
    }
    return os;
}

} // namespace chord
