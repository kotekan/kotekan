#ifndef UTIL_H
#define UTIL_H

#include <stdint.h> // for int64_t, int32_t
#include <sys/stat.h> // for mode_t

#define EVER                                                                                       \
    ;                                                                                              \
    ;

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
void make_dirs(char* disk_base, char* data_set, int num_disks);
void make_raw_dirs(const char* disk_base, const char* disk_set, const char* data_set,
                   int num_disks);
int cp(const char* to, const char* from);
int64_t mod(int64_t a, int64_t b);
double e_time(void);
void hex_dump(const int rows, void* addr, int len);

// Recursively create directories (mkdir -p behavior). Returns 0 on success
// (including when the directory already exists), and -1 on failure with errno set.
int mkdir_p(const char* path, mode_t mode);

//! A complex integer datatype.
typedef struct {
    int32_t real; //!< The real component.
    int32_t imag; //!< The imaginary component.
} complex_int_t;

#ifdef __cplusplus
}
#endif

#endif
