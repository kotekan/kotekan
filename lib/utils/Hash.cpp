#include "Hash.hpp"

#include "fmt.hpp" // for format

#include <inttypes.h> // for SCNx64
#include <iostream>   // for istream, ostream, basic_istream::read
#include <stdexcept>  // for invalid_argument
#include <stdio.h>    // for sscanf


using nlohmann::json;


// Define the null Hash
const Hash Hash::null = {0, 0};


Hash Hash::from_string(const std::string& s) {
    Hash t;
    t.set_from_string(s);
    return t;
}

std::string Hash::to_string() const {
    return fmt::format("{}", *this);
}

void Hash::set_from_string(const std::string& s) {

    // C++ doesn't provide any reasonable string parsing routines, need to use
    // some old school C ones instead.
    int ret = sscanf(s.c_str(), "%016" SCNx64 "%016" SCNx64, &h, &l);
    if (ret != 2) {
        throw std::invalid_argument(
            fmt::format("Could not parse \"{}\" as a 128-bit hash. Length != 32.", s));
    }
}

// Conversions of the index types to json
void to_json(json& j, const Hash& h) {
    j = h.to_string();
}

void from_json(const json& j, Hash& h) {
    h.set_from_string(j.get<std::string>());
}

// Implement the stream operator
std::istream& operator>>(std::istream& is, Hash& h) {
    char t[16];
    is.read(t, 16);
    h.set_from_string(std::string(t, 16));
    return is;
}

// Implement the stream operator using fmt because iostreams are so ghastly.
std::ostream& operator<<(std::ostream& os, const Hash& h) {
    os << fmt::format("{}", h);
    return os;
}

// Implement to_string to use ADL
std::string to_string(const Hash& h) {
    return fmt::format("{}", h);
}
