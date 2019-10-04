#ifndef TYPE_HPP
#define TYPE_HPP

#include <string>
#include <typeinfo>

std::string demangle(const char* name);

template<class T>
std::string type_demangle(const T& t) {

    return demangle(typeid(t).name());
}

template<class T>
std::string type_demangle() {

    return demangle(typeid(T).name());
}

#endif
