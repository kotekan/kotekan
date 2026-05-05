#include "../include/ksgpu/string_utils.hpp"
#include "../include/ksgpu/xassert.hpp"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <iostream>

using namespace std;

namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// ----------------------------------------  nbytes_to_str()  --------------------------------------


static string _nbytes_to_str(double nbytes, double nunits, const char *units)
{
    stringstream ss;
    ss << setprecision(4) << (nbytes/nunits) << " " << units;
    return ss.str();
}


string nbytes_to_str(long nbytes)
{
    constexpr long kilo = 1024L;
    constexpr long mega = 1024L * 1024L;
    constexpr long giga = 1024L * 1024L * 1024L;
    constexpr long tera = 1024L * 1024L * 1024L * 1024L;

    xassert(nbytes >= 0);
        
    if (nbytes >= tera)
        return _nbytes_to_str(nbytes, tera, "TB");
    else if (nbytes >= giga)
        return _nbytes_to_str(nbytes, giga, "GB");
    else if (nbytes >= mega)
        return _nbytes_to_str(nbytes, mega, "MB");
    else if (nbytes >= kilo)
        return _nbytes_to_str(nbytes, kilo, "KB");
    else 
        return _nbytes_to_str(nbytes, 1, "bytes");
}


// ---------------------------------------  nbytes_from_str()  -------------------------------------


static std::string _nbytes_from_str_err(const string &s)
{
    stringstream ss;
    ss << "nbytes_from_str(): couldn't convert string \"" << s << "\" to a byte count";
    return ss.str();
}


// Parses string such as "1.5GB" or "256B"
long nbytes_from_str(const string &s)
{
    // FIXME: some day I'll learn how to use std::string
    const char *cs = s.c_str();

    int n3 = s.size();
    while ((n3 > 0) && (isspace(cs[n3-1])))
        n3--;

    int n2 = n3;
    while ((n2 > 0) && (isalpha(cs[n2-1])))
        n2--;

    int n1 = n2;
    while ((n1 > 0) && (isspace(cs[n1-1])))
        n1--;

    if ((n1==0) || (n2==n3))
        throw runtime_error(_nbytes_from_str_err(s));
    
    string s_numeric = s.substr(0,n1);
    string s_units = s.substr(n2,n3-n2);
    
    long units;
    const char *cs_units = s_units.c_str();
    
    if (!strcasecmp(cs_units, "bytes") || !strcasecmp(cs_units, "B"))
        units = 1;
    else if (!strcasecmp(cs_units, "KB"))
        units = 1024L;
    else if (!strcasecmp(cs_units, "MB"))
        units = 1024L * 1024L;
    else if (!strcasecmp(cs_units, "GB"))
        units = 1024L * 1024L * 1024L;
    else if (!strcasecmp(cs_units, "TB"))
        units = 1024L * 1024L * 1024L * 1024L;
    else
        throw runtime_error(_nbytes_from_str_err(s));

    long i_numeric = 0;
    if (_from_str(s_numeric, i_numeric))
        return i_numeric * units;

    double d_numeric = 0.0;
    if (_from_str(s_numeric, d_numeric))
        return d_numeric * units;

    throw runtime_error(_nbytes_from_str_err(s));
}


}  // namespace ksgpu
