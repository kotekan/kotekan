#ifndef UTIL_H
#define UTIL_H

#include <stdint.h> // for int64_t, int32_t

#define EVER                                                                                       \
    ;                                                                                              \
    ;

#ifndef MAC_OSX
#define MIN(a, b)                                                                                  \
    ({                                                                                             \
        __typeof__(a) _a = (a);                                                                    \
        __typeof__(b) _b = (b);                                                                    \
        _a < _b ? _a : _b;                                                                         \
    })
#endif

#ifdef __cplusplus
#include <string>

/**
 * @brief Returns the last N characters of a string
 *
 * @param str The string to get the last characters from
 * @param N The number of characters to take form the end of the string
 * @return std::string
 */
inline std::string string_tail(std::string const& str, size_t const N) {
    if (N >= str.size()) {
        return str;
    }
    return str.substr(str.size() - N);
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

void make_rfi_dirs(int streamID, const char* write_to, const char* time_dir);
int make_dir(const char* dir_name);
void make_dirs(char* disk_base, char* data_set, int num_disks);
void make_raw_dirs(const char* disk_base, const char* disk_set, const char* data_set,
                   int num_disks);
int cp(const char* to, const char* from);
int64_t mod(int64_t a, int64_t b);
double e_time(void);
void hex_dump(const int rows, void* addr, int len);

//! A complex integer datatype.
typedef struct {
    int32_t real; //!< The real component.
    int32_t imag; //!< The imaginary component.
} complex_int_t;

#ifdef __cplusplus
}
#endif

#endif
