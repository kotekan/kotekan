/*****************************************
@file
@brief Miscellaneous utils for time, ERA, LST, etc.
*****************************************/
#ifndef TIME_UTIL_HPP
#define TIME_UTIL_HPP

#include <inttypes.h>
#include <time.h> // for timespec

int64_t timespec_to_nanosec_i64(timespec t);
timespec nanosec_i64_to_timespec(int64_t t);

/**
 * @brief Compute UT1 time Julian Date (in seconds, nanoseconds) from GPS time
 * @param   t The instrument time to convert, const reference timespec
 * @param   dAT double Difference between TAI-UTC at t and value at frame0.
 * @param   dUT dobule Value of UT1-UTC at t, seconds
 * @return  UT1 time as a timespec
 */
int64_t get_UT1_from_time(const timespec& t, double delta_UT1_inst);

/**
 * @brief Compute UT1 time Julian Date (in seconds, nanoseconds) from GPS time
 * @param   t The instrument time to convert, const reference timespec
 * @param   dAT double Difference between TAI-UTC at t and value at frame0.
 * @param   dUT dobule Value of UT1-UTC at t, seconds
 * @return  UT1 time as a timespec
 */
timespec get_time_from_UT1(int64_t t_ut1, double delta_UT1_inst);

/**
 * @brief Compute Earth Rotation Angle (ERA) from UT1
 * @param   ut1 const ref timespec The UT1 time to convert, since JD=0.
 * @param   num_rot int64_t pointer Optional location to store number of rotations since UT1 2451545
 * JD
 * @return  ERA in degrees, [0.0, 360.0)
 */
double get_ERA_from_UT1(int64_t ut1, int64_t* num_rot);

/**
 * @brief Compute UT1 time from Earth Rotation Angle (ERA).
 * @param   num_rot int64_t Number of Earth rotations since UT1 2451545 JD
 * @param   ERA_deg double ERA in degrees
 * @return  timespec containing the UT1 time with JD epoch.
 */
int64_t get_UT1_from_ERA(int64_t num_rot, double ERA_deg);

/**
 * @brief Compute Earth Rotation Angle (ERA) from GPS time
 * @param   t_inst The instrument time to convert, const reference timespec
 * @param   delta_UT1_inst double Value of UT1-t_inst at t_inst, seconds
 *          Equal to UT1 - UTC plus the number of leap seconds that have
 *          occured since instrument start up.
 * @return  ERA in degrees, [0.0, 360.0)
 */
double get_ERA_from_time(const timespec& t_inst, double delta_UT1_inst);

#endif
