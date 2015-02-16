#ifndef UTIL_H
#define UTIL_H

#define EVER ;;

#define MIN(a,b) \
    ({ __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

char *strclone(const char *s);
double tv_difference(const struct timeval *tv1, const struct timeval *tv2);
int64_t mod(int64_t a, int64_t b);

#endif
