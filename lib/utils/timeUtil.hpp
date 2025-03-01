/*****************************************
@file
@brief Miscellaneous utils for time, ERA, LST, etc.
*****************************************/
#ifndef TIME_UTIL_HPP
#define TIME_UTIL_HPP

#include <time.h>   // for timespec
#include <vector>


/**
 * @brief Compute UT1 time Julian Date (in seconds, nanoseconds) from GPS time
 * @param   t The instrument time to convert, const reference timespec
 * @param   dAT double Difference between TAI-UTC at t and value at frame0.
 * @param   dUT dobule Value of UT1-UTC at t, seconds
 * @return  UT1 time as a timespec
 */
timespec get_UT1_from_time(const timespec &t, double dAT, double dUT);

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
double get_ERA_from_time(const timespec &t, double dAT, double dUT);

#endif
