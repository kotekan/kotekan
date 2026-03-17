#ifndef _KSGPU_XASSERT_HPP
#define _KSGPU_XASSERT_HPP

#include <string>  // std::to_string()
#include <stdexcept>

// Note: xassert_* macros are implemented with #define, and therefore are outside the ksgpu namespace.

#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif


// xassert(): like assert(), but throws an exception in order to work smoothly with python.
#define xassert(cond) _xassert(cond,__LINE__)
#define _xassert(cond,line) \
    do { \
        if (_unlikely(!(cond))) \
	    throw std::runtime_error("C++ assertion '" __STRING(cond) "' failed (" __FILE__ ":" __STRING(line) ")"); \
    } while (0)


// xassert_msg(): use customized error message.
// The 'msg' argument can either be a (const char *) or a (const std::string &).
#define xassert_msg(cond, msg) \
    do { \
        if (_unlikely(!(cond))) \
	    throw std::runtime_error(msg); \
    } while (0)


// xassert_eq(), xassert_ne(), xassert_lt(), xassert_le(), xassert_ge(), xassert_gt():
// Compare two arguments, and show their values if the assertion fails.

#define xassert_eq(lhs,rhs) _xassert_eq(lhs,rhs,__LINE__)
#define _xassert_eq(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) != (rhs))) \
	    throw std::runtime_error("C++ assertion (" __STRING(lhs) ") == (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): " \
				     "lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs)); \
    } while (0)

#define xassert_ne(lhs,rhs) _xassert_ne(lhs,rhs,__LINE__)
#define _xassert_ne(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) != (rhs))) \
	    throw std::runtime_error("C++ assertion (" __STRING(lhs) ") != (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): " \
				     "lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs)); \
    } while (0)

#define xassert_lt(lhs,rhs) _xassert_lt(lhs,rhs,__LINE__)
#define _xassert_lt(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) >= (rhs))) \
	    throw std::runtime_error("C++ assertion (" __STRING(lhs) ") < (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): " \
				     "lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs)); \
    } while (0)

#define xassert_le(lhs,rhs) _xassert_le(lhs,rhs,__LINE__)
#define _xassert_le(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) > (rhs))) \
	    throw std::runtime_error("C++ assertion (" __STRING(lhs) ") <= (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): " \
				     "lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs)); \
    } while (0)

#define xassert_ge(lhs,rhs) _xassert_ge(lhs,rhs,__LINE__)
#define _xassert_ge(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) < (rhs))) \
	    throw std::runtime_error("C++ assertion (" __STRING(lhs) ") >= (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): " \
				     "lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs)); \
    } while (0)

#define xassert_gt(lhs,rhs) _xassert_gt(lhs,rhs,__LINE__)
#define _xassert_gt(lhs,rhs,line) \
    do { \
	if (_unlikely((lhs) <= (rhs))) \
	    throw std::runtime_error("C++ assertion (" __STRING(lhs) ") > (" __STRING(rhs) ") failed (" __FILE__ ":" __STRING(line) "): " \
				     "lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs)); \
    } while (0)


#endif // _KSGPU_XASSERT_HPP
