#include <Symbol.hpp>
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
    return os << sym.get_c_string();
}

} // namespace kotekan
