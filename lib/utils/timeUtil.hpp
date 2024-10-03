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

timespec get_TAI_from_GPS(const struct timespec &gps_time);

/**
 * @brief Compute UT1 "time" Julian Date (in seconds, nanoseconds) from GPS time
 * @param   gps_time The GPS time to convert, const reference timespec
 * @param   dAT double Value of TAI-UTC at gps_time, seconds
 * @param   dUT dobule Value of UT1-UTC at gps_time, seconds
 * @return  UT1 "time" (really: angle) as a timespec
 */
timespec get_UT1_from_GPS(const timespec &gps_time, double dAT, double dUT);

/**
 * @brief Compute Earth Rotation Angle (ERA) from UT1
 * @param   ut1 const ref timespec The UT1 time to convert, since JD=0.
 * @return  ERA in degrees
 */
double get_ERA_from_UT1(const timespec &ut1);

/**
 * @brief Compute Earth Rotation Angle (ERA) from GPS time
 * @param   gps_time The GPS time to convert, const reference timespec
 * @param   dAT double Value of TAI-UTC at gps_time, seconds
 * @param   dUT dobule Value of UT1-UTC at gps_time, seconds
 * @return  ERA in degrees
 */
double get_ERA_from_GPS(const timespec &gpstime, double dAT, double dUT);


#endif
