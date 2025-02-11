#ifndef SYMBOL_HPP
#define SYMBOL_HPP

#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>

class Symbol {
    static std::mutex mutex;
    static std::unordered_set<std::string_view> known_symbols;

    const char* value;

    const char* lookup_or_insert(const char* str);

public:
    Symbol() : value() {}

    Symbol(const std::string_view& str);
    Symbol(const std::string& str);
    Symbol(const char* str);

    Symbol(const Symbol&) = default;
    Symbol(Symbol&&) = default;
    Symbol& operator=(const Symbol&) = default;
    Symbol& operator=(Symbol&&) = default;

    bool valid() const noexcept {
        return bool(value);
    }

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

    friend std::ostream& operator<<(std::ostream& os, Symbol sym);
};

namespace std {
template<>
struct equal_to<::Symbol> {
    bool operator()(const ::Symbol& lhs, const ::Symbol& rhs) const noexcept {
        return lhs == rhs;
    }
};
template<>
struct less<::Symbol> {
    bool operator()(const ::Symbol& lhs, const ::Symbol& rhs) const noexcept {
        return lhs < rhs;
    }
};
template<>
struct hash<::Symbol> {
    std::size_t operator()(const ::Symbol& sym) const noexcept {
        return std::size_t(sym.get_c_string());
    }
};
} // namespace std

#endif // #ifndef SYMBOL_HPP
