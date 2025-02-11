#include <Symbol.hpp>

std::mutex Symbol::mutex;
std::unordered_map<std::string, Symbol::value_type> Symbol::values;
std::vector<std::string> Symbol::strings;

Symbol::value_type Symbol::lookup_or_insert(const std::string& str) const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto iter = values.find(str);
    if (iter != values.end())
        return iter->second;
    const value_type val = strings.size();
    values[str] = val;
    strings.push_back(str);
    return val;
}

Symbol::Symbol(value_type value) : value(value) {
    std::lock_guard<std::mutex> lock(mutex);
    if (value >= strings.size())
        throw std::invalid_argument("Value out of range");
}

std::string Symbol::get_string() const {
    if (!valid())
        throw std::invalid_argument("Invalid symbol");
    std::lock_guard<std::mutex> lock(mutex);
    return strings.at(value);
}

std::ostream& operator<<(std::ostream& os, Symbol sym) {
    return os << sym.get_string();
}
