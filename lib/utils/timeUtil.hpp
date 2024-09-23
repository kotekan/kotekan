/*****************************************
@file
@brief Miscellaneous utils for time, ERA, LST, etc.
*****************************************/
#ifndef TIME_UTIL_HPP
#define TIME_UTIL_HPP

#include <time.h>   // for timespec

/**
 * @brief Compute TAI time from GPS time
 * @param   gps_time The GPS time to convert, const reference timespec
 * @return  TAI time as a timespec
 */
struct timespec get_TAI_from_GPS(const struct timespec &gps_time);

#endif
