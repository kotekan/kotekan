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
#define fmt(s) FMT_STRING(s)

#include <fmt/format.h>
#include <fmt/chrono.h>
#include <fmt/printf.h>
#include <fmt/ostream.h>

#endif // _FMT_HPP_
