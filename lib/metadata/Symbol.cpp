#include <Symbol.hpp>
#include <json.hpp>  // for json, basic_json
#include <stdexcept> // for invalid_argument

namespace kotekan {

std::mutex Symbol::mutex;
owning_unordered_set<std::string_view> Symbol::known_symbols;

// Look up a string in the known symbols. If it is not known, insert
// it.
const char* Symbol::lookup_or_insert(const std::string_view& str) {
    // An emptry string turns into an invalid symbol
    if (str.empty())
        return nullptr;
    // From now on we need to lock the global mutex. This is the
    // expensive part.
    std::lock_guard<std::mutex> lock(mutex);
    const auto iter = known_symbols.find(str);
    // If the symbol is known, return it
    if (iter != known_symbols.end())
        return iter->data();
    // The symbol is not known. Insert it.
    char* symbol = new char[str.size() + 1];
    // build a C string (NUL terminated) from an array of chars, that may not be
    // NUL terminated
    std::memcpy(symbol, str.data(), str.size());
    symbol[str.size()] = '\0';
    known_symbols.insert(std::string_view(symbol));
    return symbol;
}

Symbol::Symbol(const std::string_view& str) : value(lookup_or_insert(str)) {}
Symbol::Symbol(const std::string& str) : value(lookup_or_insert(str)) {}
Symbol::Symbol(const char* str) : value(lookup_or_insert(str ? str : "")) {}

std::string Symbol::get_string() const {
    if (!valid())
        throw std::invalid_argument("Invalid symbol");
    return std::string(get_c_string());
}

std::ostream& operator<<(std::ostream& os, Symbol sym) {
    // An unset (invalid) Symbol has no string; render it as empty rather than
    // streaming the null get_c_string().
    return os << (sym.valid() ? sym.get_c_string() : "");
}

// An unset (invalid) Symbol serializes as JSON `null`: it keeps a deliberately
// unset label distinguishable from a set one and round-trips (an empty string
// is itself an invalid Symbol). get_string() would otherwise throw on invalid.
void to_json(nlohmann::json& j, const Symbol& s) {
    if (s.valid())
        j = s.get_string();
    else
        j = nullptr;
}

void from_json(const nlohmann::json& j, Symbol& s) {
    s = j.is_null() ? Symbol() : Symbol(j.get<std::string>());
}
} // namespace kotekan
