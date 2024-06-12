#ifndef _GPUTILS_STRING_UTILS_HPP
#define _GPUTILS_STRING_UTILS_HPP

#include <string>
#include <vector>
#include <sstream>
#include <typeinfo>
#include <stdexcept>

#ifdef __GNUG__
#include <cxxabi.h>
#endif

namespace gputils {
#if 0
}   // pacify editor auto-indent
#endif


// ---------------------------------------   type_name()   -----------------------------------------


// Returns a string typename, e.g. type_name<int>() -> "int"
template<typename T>
static std::string type_name()
{
    const char *s = typeid(T).name();

#ifdef __GNUG__
    // Reference: https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
    int status = -1;
    char *t = abi::__cxa_demangle(s, nullptr, nullptr, &status);
    std::string ret((t && !status) ? t : s);
    free(t);
    return ret;
#else
    return std::string(s);
#endif
}


// -----------------------------------  to_str(), from_str()  --------------------------------------


template<typename T>
static std::string to_str(const T &x)
{
    std::stringstream ss;
    ss << x;
    return ss.str();
}


// Returns false if (string -> T) conversion fails.
template<typename T>
static bool _from_str(const std::string &s, T &val)
{
    std::stringstream ss;
    ss << s;

    ss >> val;  // shouldn't fail, if 's' is valid.
    if (ss.fail())
	return false;

    std::string t;
    ss >> t;    // should fail, if 's' is valid.
    return ss.fail();
}


// Throws an exception if (string -> T) conversion fails.
template<typename T>
static T from_str(const std::string &s)
{
    T ret = 0;
    if (_from_str(s, ret))
	return ret;
    
    std::stringstream err;
    err << "couldn't convert string \"" << s << "\" to type " << type_name<T>();
    throw std::runtime_error(err.str());
}


// ---------------------------------------   tuple_str()   -----------------------------------------


// Returns a formatted tuple, e.g. "(1,2,3)"
// To insert spaces (e.g. "(1, 2, 3)"), call with space=" ".
template<typename T>
static std::string tuple_str(int nelts, const T *tuple, const char *space="")
{
    if (nelts == 0)
	return "()";
	
    std::stringstream ss;
    ss << "(" << tuple[0];

    if (nelts == 1) {
	ss << ",)";
	return ss.str();
    }

    for (int d = 1; d < nelts; d++)
	ss << "," << space << tuple[d];

    ss << ")";
    return ss.str();
}


template<typename T>
static std::string tuple_str(const std::vector<T> &tuple, const char *space="")
{
    return tuple_str(tuple.size(), &tuple[0], space);
}


// ---------------------------  nbytes_to_str(), nbytes_from_str()  --------------------------------


// Converts integer byte count to a string such as "1.5 MB" or "320 bytes".
extern std::string nbytes_to_str(ssize_t nbytes);

// Converts a string such as "1.5 MB" or "320 bytes" to a byte count.
extern ssize_t nbytes_from_str(const std::string &s);


} // namespace gputils

#endif  // _GPUTILS_STRING_UTILS_HPP
