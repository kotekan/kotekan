// Wrap fmt into a single header file

#ifndef _FMT_HPP_
#define _FMT_HPP_

#ifndef FMT_HEADER_ONLY
    #define FMT_HEADER_ONLY
#endif

#ifndef FMT_OVERRIDE
    #define FMT_OVERRIDE override
#endif

// Enable the fmt() macro for compile time string format checking
#define FMT_STRING_ALIAS 1

#include <fmt/format.h>
#include <fmt/chrono.h>
#include <fmt/printf.h>

#endif // _FMT_HPP_
