#ifndef SYMBOL_HPP
#define SYMBOL_HPP

#include <array>         // for array
#include <cstring>       // for size_t
#include <functional>    // for equal_to, less
#include <iostream>      // for ostream
#include <mutex>         // for mutex
#include <string>        // for string, basic_string
#include <string_view>   // for string_view, hash
#include <unordered_set> // for unordered_set

namespace kotekan {

// A symbol is similar to a string: A symbol can be created from a
// string (this is expensive). As pay-off, comparing symbols to each
// other is very cheap, it has the same cost as comparing two
// integers.
class Symbol {
    // We keep a set of all symbols that have been created
    // (`known_symbols). It is protected by a mutex.
    static std::mutex mutex;
    static std::unordered_set<std::string_view> known_symbols;

    // A symbol's value is a pointer to a C string.
    const char* value;

    // Look up a string in the known symbols. If it is not known,
    // insert it.
    const char* lookup_or_insert(const char* str);

public:
    // Default constructure, returning an invalid symbol. Think null
    // pointer.
    Symbol() : value() {}

    // Create a symbol from a string
    Symbol(const std::string_view& str);
    Symbol(const std::string& str);
    Symbol(const char* str);

    Symbol(const Symbol&) = default;
    Symbol(Symbol&&) = default;
    Symbol& operator=(const Symbol&) = default;
    Symbol& operator=(Symbol&&) = default;

    // Check whether a symbol is valid. Operations on invalid symbols
    // may raise exceptions.
    bool valid() const noexcept {
        return bool(value);
    }

    // Conver a symbol to a string. This is cheap.
    const char* get_c_string() const noexcept {
        return value;
    }
    operator const char*() const noexcept {
        return get_c_string();
    }
    std::string get_string() const;
    operator std::string() const {
        return get_string();
    }

    // Compare symbols. These operations are cheap.
    friend bool operator==(Symbol sym1, Symbol sym2) {
        return sym1.value == sym2.value;
    }
    friend bool operator!=(Symbol sym1, Symbol sym2) {
        return !(sym1 == sym2);
    }

    // Arbitrary order
    friend bool operator<(Symbol sym1, Symbol sym2) {
        return sym1.value < sym2.value;
    }
    friend bool operator>(Symbol sym1, Symbol sym2) {
        return sym2 < sym1;
    }
    friend bool operator<=(Symbol sym1, Symbol sym2) {
        return !(sym1 > sym2);
    }
    friend bool operator>=(Symbol sym1, Symbol sym2) {
        return !(sym1 < sym2);
    }

    // Output a symbol
    friend std::ostream& operator<<(std::ostream& os, Symbol sym);
};

template<std::size_t D>
std::array<kotekan::Symbol, D> strings_to_symbols(const std::array<std::string, D>& strings) {
    std::array<kotekan::Symbol, D> symbols;
    for (std::size_t d = 0; d < D; ++d)
        symbols[d] = strings[d];
    return symbols;
}
template<std::size_t D>
std::array<kotekan::Symbol, D> strings_to_symbols(const std::array<const char*, D>& strings) {
    std::array<kotekan::Symbol, D> symbols;
    for (std::size_t d = 0; d < D; ++d)
        symbols[d] = strings[d];
    return symbols;
}

} // namespace kotekan

namespace std {
// Comparison functions for Symbol
template<>
struct equal_to<kotekan::Symbol> {
    bool operator()(const kotekan::Symbol& lhs, const kotekan::Symbol& rhs) const noexcept {
        return lhs == rhs;
    }
};
template<>
struct less<kotekan::Symbol> {
    bool operator()(const kotekan::Symbol& lhs, const kotekan::Symbol& rhs) const noexcept {
        return lhs < rhs;
    }
};
// Hash function for Symbol
template<>
struct hash<kotekan::Symbol> {
    std::size_t operator()(const kotekan::Symbol& sym) const noexcept {
        return std::size_t(sym.get_c_string());
    }
};
} // namespace std

#endif // #ifndef SYMBOL_HPP
