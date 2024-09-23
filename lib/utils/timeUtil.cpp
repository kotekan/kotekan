#include "timeUtil.hpp"

timespec get_TAI_from_GPS(const timespec &gps_time) {
    return timespec{
                .tv_sec=(gps_time.tv_sec+19),
                .tv_nsec=gps_time.tv_nsec};
}
