#ifndef SYMBOL_HPP
#define SYMBOL_HPP

#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class Symbol {
public:
    using value_type = std::size_t;

private:
    static std::mutex mutex;
    static std::unordered_map<std::string, value_type> values;
    static std::vector<std::string> strings;

    value_type value;

    value_type lookup_or_insert(const std::string& str) const;

    Symbol(value_type value);

public:
    Symbol() : value(std::numeric_limits<value_type>::max()) {}

    Symbol(const std::string& str) : value(lookup_or_insert(str)) {}
    Symbol(const char* str) : value(lookup_or_insert(str)) {}

    Symbol(const Symbol&) = default;
    Symbol(Symbol&&) = default;
    Symbol& operator=(const Symbol&) = default;
    Symbol& operator=(Symbol&&) = default;

    static Symbol from_value(value_type value) {
        return Symbol(value);
    }

    bool valid() const noexcept {
        return value != std::numeric_limits<value_type>::max();
    }

    value_type get_value() const {
        if (!valid())
            throw std::invalid_argument("Invalid symbol");
        return value;
    }
    std::string get_string() const;
    operator std::string() const {
        return get_string();
    }

    friend bool operator==(Symbol sym1, Symbol sym2) {
        return sym1.get_value() == sym2.get_value();
    }
    friend bool operator!=(Symbol sym1, Symbol sym2) {
        return !(sym1 == sym2);
    }

    // Arbitrary order
    friend bool operator<(Symbol sym1, Symbol sym2) {
        return sym1.get_value() < sym2.get_value();
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
struct hash<::Symbol> {
    std::size_t operator()(const ::Symbol& sym) const noexcept {
        return std::hash<Symbol::value_type>()(sym.get_value());
    }
};
} // namespace std

#endif // #ifndef SYMBOL_HPP
