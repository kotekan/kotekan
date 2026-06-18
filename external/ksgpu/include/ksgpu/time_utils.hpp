#ifndef _KSGPU_TIME_UTILS_HPP
#define _KSGPU_TIME_UTILS_HPP

#include <stdexcept>
#include <sys/time.h>

#ifndef _unlikely
#define _unlikely(cond)  (__builtin_expect(cond,0))
#endif


namespace ksgpu {
#if 0
}   // pacify editor auto-indent
#endif


// Note: one call to get_time() takes around 25 ns.
inline struct timeval get_time()
{
    struct timeval t;
    
    int err = gettimeofday(&t, NULL);
    if (_unlikely(err != 0))
        throw std::runtime_error("gettimeofday() failed?!");

    return t;
}


inline double time_diff(const struct timeval &t0, const struct timeval &t1)
{
    return (t1.tv_sec - t0.tv_sec) + 1.0e-6 * (t1.tv_usec - t0.tv_usec);
}


inline double time_since(const struct timeval &t0)
{
    return time_diff(t0, get_time());
}


}  // namespace ksgpu

#endif // _KSGPU_TIME_UTILS_HPP
