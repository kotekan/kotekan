#include "timeUtil.hpp"

#include <boost/multiprecision/cpp_int.hpp>
#include <math.h>

using boost::multiprecision::int128_t;
    
const int128_t era_A =  77'905'727'326'400'000L;
const int128_t era_B = 100'273'781'191'135'448L;
const int128_t day_ns = 86'400'000'000'000L;
const int128_t e17 = 100'000'000'000'000'000L;
const int64_t GIGA = 1'000'000'000L;

int64_t timespec_to_nanosec_i64(const timespec &t) {
    return (int64_t) (t.tv_sec * GIGA + t.tv_nsec);
}

timespec nanosec_i64_to_timespec(int64_t ns) {

    // Compute the number of seconds in (less than) t. If t is negative,
    // ensure t_s is always rounded down.  This ensures the remainder is
    // positive.
    int64_t t_s = ns / GIGA;
    int64_t t_ns = ns - GIGA * t_s;
    if(ns < 0 && t_ns != 0)
    {
        t_s -= 1;
        t_ns += GIGA;
    }

    return {.tv_sec = (time_t) t_s, .tv_nsec = t_ns};
}
    
int64_t get_UT1_from_time(const timespec& t, double delta_UT1_inst) {
    /*
     * Compute the UT1 time at given instrument time t.
     *
     * t is a UNIX time (non leap seconds since 1970-01-01 00:00:00 UTC)
     *
     * We produce ut1 in a timespec (seconds and nanoseconds) with the Julian
     * Date (JD) epoch.
     *
     * To convert to JD we use the J2000 as a reference point with a known
     * conversion.
     *
     * J2000 = 2000-01-01 12:00:00 TT
     *       = 2000-01-01 11:58:55.816 UTC  (64.184 s behind TT)
     *
     * In JD this is:
     *       = 2'451'545.0 TT
     *       = 2'451'545.0 - 64.184 / 86400 UTC
     *
     * In UNIX this is:
     *       = 946'728'000.0 - 64.184
     *       = 946'727'935.816
     */

    // UNIX time of J2000
    int64_t J2000_s_unix = 946'727'935L;
    int64_t J2000_ns_unix = 816'000'000L;

    // JD of J2000 in s,ns
    int64_t J2000_s_jd = 2'451'545L * 86400L - 64;
    int64_t J2000_ns_jd = -184'000'000L;

    // Our internal UT1 epoch (2'451'545 JD UT1)
    int64_t ut1_t0_s_jd = 2'451'545L * 86400L;

    // Instrument time relative to J2000 (in UNIX (UTC) seconds)
    int64_t t_J2000_s = t.tv_sec - J2000_s_unix;
    int64_t t_J2000_ns = t.tv_nsec - J2000_ns_unix;

    // Get dUT1 in nanoseconds
    int64_t dUT1_ns = (int64_t)round(1e9 * delta_UT1_inst);

    // Finally, compute the UT1 time with UT1 2451545 epoch.
    int64_t ut1_s = J2000_s_jd + t_J2000_s - ut1_t0_s_jd;
    int64_t ut1_ns = J2000_ns_jd + t_J2000_ns + dUT1_ns;

    // Combine into a single nanosecond count.
    ut1_ns += GIGA * ut1_s;

    return ut1_ns;
}

timespec get_time_from_UT1(int64_t t_ut1, double delta_UT1_inst) {

    int64_t t0_ut1_s_jd = 2'451'545L * 86400L;

    // UNIX time of J2000
    int64_t J2000_s_unix = 946'727'935L;
    int64_t J2000_ns_unix = 816'000'000L;

    // JD of J2000 in s,ns
    int64_t J2000_s_jd = 2'451'545L * 86400L - 64;
    int64_t J2000_ns_jd = -184'000'000L;

    // Get dUT1 in nanoseconds
    int64_t dUT1_ns = (int64_t)round(1e9 * delta_UT1_inst);


    // Convert UT1 to SI seconds (subtract dUT1), and re-base to J2000
    // (by subtracting J2000 in jy)
    int64_t t_J2000_ns = t_ut1 - dUT1_ns - J2000_ns_jd
                            + GIGA * (t0_ut1_s_jd - J2000_s_jd);

    // Now re-base to the unix epoch for instrument time.
    int64_t t_ns = t_J2000_ns + J2000_ns_unix + GIGA * J2000_s_unix;

    return nanosec_i64_to_timespec(t_ns);
}

double get_ERA_from_UT1(int64_t ut1, int64_t* n_rot) {
    /*
     * To handle the large constants and maintain nanosecond precision we
     * compute the ERA using integer arithmetic.  The large constants are
     * expressed as rational numbers with a denominator of 1e17 to ensure
     * the numerator is an integer.
     *
     * The ut1 time since 2,451,545 JD is expressed as an integer number of 
     * nanoseconds which can have a magnitude ~ few x 10^18 ~ few x 2^60.
     * The product of this with the integral ERA constants requires a 128 bit
     * variable.
     *
     * ERA = 2pi * (0.7790572732640 + 1.00273781191135448 * (Tu(d) - 2451545))
     *     
     *     = 2pi * (0.7790572732640 + 1.00273781191135448 * ut1(d) )
     *     
     *     = 2pi * (07790572732640 + 100273781191135448 * ut1(d) ) / 1e17
     *     
     *     = 2pi * (A + B * ut1(d) ) / 1e17             (defining A & B)
     *     
     *     = 2pi * (A * day_ns + B * ut1_ns) / (1e17 * day_ns)
     *
     *     day_ns = number of ns in a day.
     */

    // Cast ut1 time (already with correct epoch) to an i128
    int128_t t_ns = (int128_t) ut1;

    // Denominator 
    int128_t denom = ((int128_t)e17) * day_ns;

    // Numerator
    int128_t tot_17ns = era_A * day_ns + era_B * t_ns;

    // Number of rotations since UT1 T0
    int64_t num_rot = (int64_t)(tot_17ns / denom);
    
    // Fraction of rotation, this is the ERA.
    int128_t f_17ns = tot_17ns % denom;
    double f = ((double)f_17ns) / ((double)denom);

    // Correct if time was negative
    if (f < 0.0) {
        f += 1;
        num_rot -= 1;
    }

    if (n_rot != nullptr)
        *n_rot = num_rot;

    // Return ERA in degrees.
    return 360.0 * f;
}

int64_t get_UT1_from_ERA(int64_t n_rot, double ERA_deg) {

    /*
     * For details see comment for get_ERA_from_UT1()
     *
     * rots = a + b * td;
     * td = (rots - a) / b
     *    = (1e17 * rots - A) / B
     * tns = (day_ns * (1e17 * rots - A)) / B
     */
   
    // the amount of rotation since UT1 T0, multiplied by 1e17 * day_ns
    int128_t rot_17ns = day_ns * (e17 * n_rot
                                  + ((int128_t)(1e17 * ERA_deg)) / 360);

    // Calculate the UT1 time for the amount of rotation.
    int128_t numer = rot_17ns - era_A * day_ns;
    int128_t t_ns = numer / era_B;

    // Compute the correction for integer rounding. Will be < 1 ns, but may
    // round to a full nanosecond.
    double frac_ns = (double)(numer % era_B) / ((double)era_B);

    return (int64_t) t_ns + (int64_t) round(frac_ns);
}

double get_ERA_from_time(const timespec& time, double dUT) {

    int64_t ut1 = get_UT1_from_time(time, dUT);
    return get_ERA_from_UT1(ut1, nullptr);
}
