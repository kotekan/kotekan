/*****************************************
@file
@brief Miscellaneous utils for time, ERA, LST, etc.

There are several time and time-like variables we track in Kotekan:

- *Instrument time* (often denoted `t`, `t_inst`, `t_inst_ns`, `INST`, or INST).  The raw internal
time variable, calculated as the time of instrument start (a UNIX time) plus the elapsed TAI time
since instrument start. This is equivalent to a UTC time unless a leap second occurs on the UTC day
the instrument starts (possibly before instrument start) or during a run.  Instrument time
accurately and monotonically tracks time elapsed since instrument start, and two instrument times
_from the same run_ can be differenced to obtain the TAI time difference between them.  Instrument
times from different runs can only be differenced if a leap second did not occur between the runs.

- *UT1 time* (often denoted `ut1`, `t_ut1`, `UT1`, or UT1).  Defined by the IERS Conventions (2010),
a time-like variable which exactly tracks the rotation of the Earth.  To convert to/from UT1, the
IERS distributes a value UT1-UTC which can be added to UTC to get the UT1 time. Because Kotekan's
instrument time is not exactly UTC, we rely on a value `delta_UT1_inst` to give us the conversion
from instrument time to UT1 directly. This value should be computed taking into account leap seconds
and the time at which the instrument started.  Internally UT1 time is represented as nanoseconds
since 2451545.0 JD(UT1), which we call J2000(UT1).

- *Earth Rotation Angle* (denoted `era`, `ERA`, or ERA). The rotational phase of the Earth,
internally stored as a double in degrees in the interval [0.0, 360).  Computed from UT1 via IERS
Covenventions (2010, 2012 update), Chapter 5, Eq. 5.14).

- *Earth Rotation number* (denoted `nrot`, or `num_rot`). An integer, the number of full rotations
(sidereal) of the earth since out UT1 epoch, J2000(UT1).  At this time `num_rot` was 0, and it
increments by 1 when ERA rolls over from 360.0 to 0.0 degrees.  Used to enable conversions from ERA
back to UT1.
*****************************************/
#ifndef TIME_UTIL_HPP
#define TIME_UTIL_HPP

#include <inttypes.h>  // for int64_t
#include <time.h>      // for timespec
#include <iosfwd>      // for ostream
#include <string>      // for string

#include "json.hpp"    // for json


/**
 * @brief   Directly convert timespec fields into a count of nanoseconds in an int64_t. Overflows
 *          if the timespec represents a time more than 2^63-1 nanoseconds (~292 years) past the
 * epoch.
 * @param   t   The timespec to convert.
 * @return  The time as an int64_t in nanoseconds.
 */
int64_t timespec_to_nanosec_i64(const timespec& t);

/**
 * @brief   Directly convert an int64_t nanosecond time into a timespec, keeping fields in their
 * spec ranges.
 * @param   t   The time in nanoseconds to convert.
 * @return  The time as a timespec.
 */
timespec nanosec_i64_to_timespec(int64_t t);

/**
 * @brief   Compute UT1 time (J2000(UT1) epoch, nanoseconds) from instrument time.
 * @param   t The instrument time to convert, const reference timespec
 * @param   delta_UT1_inst Value of UT1-INST at t, seconds
 * @return  UT1 time (int64_t nanoseconds since J2000(UT1))
 */
int64_t get_UT1_from_time(const timespec& t, double delta_UT1_inst);

/**
 * @brief   Compute instrument time from UT1 time (J2000(UT1) epoch, nanoseconds).
 * @param   t_ut1 The instrument time to convert, const reference timespec
 * @param   delta_UT1_inst Value of UT1-INST at ut1, seconds
 * @return  Instrument time as a timespec
 */
timespec get_time_from_UT1(int64_t t_ut1, double delta_UT1_inst);

/**
 * @brief   Compute UT1 time (J2000(UT1) epoch, nanoseconds) from instrument time.
 * @param   t_ns The instrument time to convert in nanoseconds
 * @param   delta_UT1_inst Value of UT1-INST at t, seconds
 * @return  UT1 time (int64_t nanoseconds since J2000(UT1))
 */
int64_t get_UT1_from_time_ns(int64_t t_ns, double delta_UT1_inst);

/**
 * @brief   Compute instrument time from UT1 time (J2000(UT1) epoch, nanoseconds).
 * @param   t_ut1 The UT1 time to convert, nanoseconds since J2000(UT1)
 * @param   delta_UT1_inst Value of UT1-INST at t_ut1, seconds
 * @return  Instrument time in nanoseconds
 */
int64_t get_time_ns_from_UT1(int64_t t_ut1, double delta_UT1_inst);

/**
 * @brief   Compute Earth Rotation Angle (ERA) from UT1
 * @param   ut1  The UT1 time to convert, nanoseconds since J2000(UT1)
 * @param   num_rot int64_t pointer, optional location to store number of rotations since UT1
 * 2451545 JD (J2000(UT1)). Set to `nullptr` to ignore.
 * @return  ERA in degrees, [0.0, 360.0)
 */
double get_ERA_from_UT1(int64_t ut1, int64_t* num_rot);

/**
 * @brief   Compute UT1 time from Earth Rotation Angle (ERA).
 * @param   num_rot Number of Earth rotations since UT1 2451545 JD
 * @param   ERA_deg ERA in degrees
 * @return  UT1 time, nanoseconds since J2000(UT1)
 */
int64_t get_UT1_from_ERA(int64_t num_rot, double ERA_deg);

/**
 * @brief   Compute Earth Rotation Angle (ERA) from instrument time
 * @param   t_inst The instrument time to convert, const reference timespec
 * @param   delta_UT1_inst double Value of UT1-t_inst at t_inst, seconds
 *          Equal to UT1 - UTC plus the number of leap seconds that have
 *          occured since instrument start up.
 * @return  ERA in degrees, [0.0, 360.0)
 */
double get_ERA_from_time(const timespec& t_inst, double delta_UT1_inst);


/**
 * @brief   Simple struct for containing Earth Orientation Parameter (EOP) data
 *
 * @param   t_inst_ns       int64_t Instrument time, nanoseconds, UNIX epoch.
 * @param   t_ut1_ns        int64_t UT1 time, nanoseconds, J2000(UT1) epoch.
 * @param   delta_UT1_inst  double  Difference between UT1 and Instrument time, seconds.
 * @param   ERA_deg         double  Earth Rotation Angle, degrees.
 * @param   xp_as           double  Polar Motion x', arcseconds.
 * @param   yp_as           double  Polar Motion y', arcseconds.
 */
struct EOP {
    int64_t t_inst_ns;     // Instrument time, nanoseconds, UNIX epoch.
    int64_t t_ut1_ns;      // UT1 time, nanoseconds, J2000(UT1) epoch.
    double delta_UT1_inst; // Diff between UT1 and Instrument time, seconds
    double ERA_deg;        // Earth Rotation Angle, degrees
    double xp_as;          // Polar Motion x', in arcseconds.
    double yp_as;          // Polar Motion y', in arcseconds.

    /// Return a simple string representation of the EOP for printing purposes
    std::string to_string() const;
};

static constexpr EOP eop_null = {.t_inst_ns = 0,
                                 .t_ut1_ns = 0,
                                 .delta_UT1_inst = 0.0,
                                 .ERA_deg = 0.0,
                                 .xp_as = 0.0,
                                 .yp_as = 0.0};

/// JSON serialization functions
void to_json(nlohmann::json& j, const EOP& m);
void from_json(const nlohmann::json& j, EOP& m);

/// Simple equality comparison
inline bool operator==(const EOP& lhs, const EOP& rhs) {
    return (lhs.t_inst_ns == rhs.t_inst_ns && lhs.t_ut1_ns == rhs.t_ut1_ns
            && lhs.delta_UT1_inst == rhs.delta_UT1_inst && lhs.ERA_deg == rhs.ERA_deg
            && lhs.xp_as == rhs.xp_as && lhs.yp_as == rhs.yp_as);
}

/// Simple string printing.
std::ostream& operator<<(std::ostream& os, const EOP& eop);

/// Printing with fmt {}
std::string format_as(const EOP& eop);

/**
 * @brief   Comparison function for searching/sorting an EOP table. Compares
 *          EOP based on t_inst, orders chronologically.
 *
 * @params  eop1    First EOP to compare.
 * @params  eop2    Second EOP to compare.
 **/
inline bool EOP_comp_time(const EOP& eop1, const EOP& eop2) {
    return eop1.t_inst_ns < eop2.t_inst_ns;
}

/**
 * @brief   Comparison function for searching/sorting an EOP table. Compares
 *          EOP based on t_ut1, orders by increasing rotation. Will produce the
 *          same order as t_inst, unless something is apocalyptically wrong.
 *
 * @params  eop1    First EOP to compare.
 * @params  eop2    Second EOP to compare.
 **/
inline bool EOP_comp_ut1(const EOP& eop1, const EOP& eop2) {
    return eop1.t_ut1_ns < eop2.t_ut1_ns;
}

/**
 * @brief A struct used for updating the EOP Table via REST which contains minimal EOP data. A full
 * EOP can be produced from a BareEOP.
 */
struct BareEOP {
    int64_t t_inst_ns;     /// Instrument time INST in nanoseconds for this moment.
    double delta_UT1_inst; /// UT1 - INST at this moment.
    double xp_as;          /// x polar motion parameter at this moment, arcseconds.
    double yp_as;          /// y polar motion parameter at this moment, arcseconds.

    /// Produce an EOP from this BareEOP, computing the redundant fields in the full EOP.
    inline EOP to_EOP() const {
        int64_t ut1 = get_UT1_from_time_ns(t_inst_ns, delta_UT1_inst);
        double era = get_ERA_from_UT1(ut1, nullptr);
        return {.t_inst_ns = t_inst_ns,
                .t_ut1_ns = ut1,
                .delta_UT1_inst = delta_UT1_inst,
                .ERA_deg = era,
                .xp_as = xp_as,
                .yp_as = yp_as};
    }

    /// Produce a BareEOP from an EOP, stripping the redundant fields.
    static inline BareEOP from_EOP(const EOP& eop) {
        return {.t_inst_ns = eop.t_inst_ns,
                .delta_UT1_inst = eop.delta_UT1_inst,
                .xp_as = eop.xp_as,
                .yp_as = eop.yp_as};
    }

    /// Return a simple string representation of the BareEOP for printing purposes
    std::string to_string() const;
};

static constexpr BareEOP bare_eop_null = {
    .t_inst_ns = 0, .delta_UT1_inst = 0, .xp_as = 0, .yp_as = 0};

/// Simple equality for BareEOP
inline bool operator==(const BareEOP& lhs, const BareEOP& rhs) {
    return (lhs.t_inst_ns == rhs.t_inst_ns && lhs.delta_UT1_inst == rhs.delta_UT1_inst
            && lhs.xp_as == rhs.xp_as && lhs.yp_as == rhs.yp_as);
}

/// Simple printing to string
std::ostream& operator<<(std::ostream& os, const BareEOP& eop);

/// Printing with fmt {}
std::string format_as(const BareEOP& eop);

/// JSON serialization
void to_json(nlohmann::json& j, const BareEOP& eop);
void from_json(const nlohmann::json& j, BareEOP& eop);


#endif
