#include <Metadata.hpp>

Metadata::Metadata() = default;
Metadata::Metadata(Metadata&&) = default;
Metadata& Metadata::operator=(Metadata&&) = default;

bool Metadata::has_bool(const std::string& key) const {
    return m_bool.count(key);
}
bool Metadata::has_int(const std::string& key) const {
    return m_int.count(key);
}
bool Metadata::has_real(const std::string& key) const {
    return m_real.count(key);
}
bool Metadata::has_string(const std::string& key) const {
    return m_string.count(key);
}
bool Metadata::has_bool_vector(const std::string& key) const {
    return m_bool_vector.count(key);
}
bool Metadata::has_int_vector(const std::string& key) const {
    return m_int_vector.count(key);
}
bool Metadata::has_real_vector(const std::string& key) const {
    return m_real_vector.count(key);
}
bool Metadata::has_string_vector(const std::string& key) const {
    return m_string_vector.count(key);
}

void Metadata::set_bool(const std::string& key, bool value) {
    assert(!has_bool(key));
    m_bool[key] = value;
}
void Metadata::set_int(const std::string& key, std::int64_t value) {
    assert(!has_int(key));
    m_int[key] = value;
}
void Metadata::set_real(const std::string& key, double value) {
    assert(!has_real(key));
    m_real[key] = value;
}
void Metadata::set_string(const std::string& key, std::string value) {
    assert(!has_string(key));
    m_string[key] = std::move(value);
}
void Metadata::set_string(const std::string& key, const char* value) {
    set_string(key, std::string(value));
}
void Metadata::set_bool_vector(const std::string& key, std::vector<bool> value) {
    assert(!has_bool_vector(key));
    m_bool_vector[key] = std::move(value);
}
void Metadata::set_int_vector(const std::string& key, std::vector<std::int64_t> value) {
    assert(!has_int_vector(key));
    m_int_vector[key] = std::move(value);
}
void Metadata::set_real_vector(const std::string& key, std::vector<double> value) {
    assert(!has_real_vector(key));
    m_real_vector[key] = std::move(value);
}
void Metadata::set_string_vector(const std::string& key, std::vector<std::string> value) {
    assert(!has_string_vector(key));
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

bool Metadata::get_bool(const std::string& key) const {
    assert(has_bool(key));
    return m_bool.at(key);
}
std::int64_t Metadata::get_int(const std::string& key) const {
    assert(has_int(key));
    return m_int.at(key);
}
double Metadata::get_real(const std::string& key) const {
    assert(has_real(key));
    return m_real.at(key);
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
std::vector<double> Metadata::get_real_vector(const std::string& key) const {
    assert(has_real_vector(key));
    return m_real_vector.at(key);
}
std::vector<std::string> Metadata::get_string_vector(const std::string& key) const {
    assert(has_string_vector(key));
    return m_string_vector.at(key);
}
