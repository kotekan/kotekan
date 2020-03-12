#include "type.hpp"

#include <cstdlib>
#include <cxxabi.h>
#include <memory>

// NOTE: this works on GCC and clang, but might be fragile elsewhere
std::string demangle(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status),
                                               std::free};

    return (status == 0) ? res.get() : name;
}
