#ifndef UTIL_H
#define UTIL_H

#define EVER ;;

#define MIN(a,b) \
    ({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

#ifdef __cplusplus
extern "C" {
#endif

void make_dirs(char * disk_base, char * data_set, int num_disks);
void make_raw_dirs(const char * disk_base, const char * disk_set, const char * data_set, int num_disks);
int cp(const char *to, const char *from);
int64_t mod(int64_t a, int64_t b);
double e_time(void);
void hex_dump (const int rows, void *addr, int len);

//! A complex integer datatype.
typedef struct {
    int32_t real; //!< The real component.
    int32_t imag; //!< The imaginary component.
} complex_int_t;

#ifdef __cplusplus
}
#endif

#endif
