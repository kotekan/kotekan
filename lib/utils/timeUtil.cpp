#include <math.h>
#include "timeUtil.hpp"

timespec get_UT1_from_time(const timespec &t, double delta_UT1_inst) {
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

    long GIGA = 1'000'000'000;

    // UNIX time of J2000
    long J2000_s_unix = 946'728'935L;
    long J2000_ns_unix = 816'000'000L;

    // JD of J2000 in s,ns
    long J2000_s_jd = 2'451'545L * 86400L - 64;
    long J2000_ns_jd = -184'000'000L;

    // Instrument time relative to J2000 (in UNIX (UTC) seconds)
    long t_J2000_s = t.tv_sec - J2000_s_unix;
    long t_J2000_ns = t.tv_nsec - J2000_ns_unix;

    // extract the seconds and fractional seconds from dUT1
    double dUT1_secd;
    double dUT1_fracsec = modf(delta_UT1_inst, &dUT1_secd);

    // convert from floating point sec & frac sec to integral s & ns

    // The seconds are already guaranteed to be integral, so a simple cast
    // works here.
    long dUT1_s = (long) dUT1_secd;

    // The fracsec are doubles, with precision ~ 1e-16 s, so rounding 
    // will produce the correct number of ns.
    // NOTE: dUT1 is only accurate to ~microseconds at best, so
    // nanoseconds is comfortably overkill.
    long dUT1_ns = (long) round(1e9 * dUT1_fracsec);

    // Finally, compute the UT1 time with JD epoch.  
    long ut1_s = J2000_s_jd + t_J2000_s + dUT1_s;
    long ut1_ns = J2000_ns_jd + t_J2000_ns + dUT1_ns;

    // Our ns count may be negative or greater than 1'000'000'000,
    // need to normalize for putting in a timespec.
    long full_sec_over = ut1_ns / GIGA;

    // if ns count is negative, the division above will undercount by 1 the
    // the number of seconds to shift;
    if(ut1_ns < 0)
        full_sec_over -= 1;

    ut1_s += full_sec_over;
    ut1_ns -= full_sec_over * GIGA;

    return timespec{.tv_sec=ut1_s, .tv_nsec=ut1_ns};
}

timespec get_time_from_UT1(const timespec &t_ut1, double delta_UT1_inst) {

    long GIGA = 1'000'000'000;

    // UNIX time of J2000
    long J2000_s_unix = 946'728'935L;
    long J2000_ns_unix = 816'000'000L;

    // JD of J2000 in s,ns
    long J2000_s_jd = 2'451'545L * 86400L - 64;
    long J2000_ns_jd = -184'000'000L;

    // extract the seconds and fractional seconds from dUT1
    double dUT1_secd;
    double dUT1_fracsec = modf(delta_UT1_inst, &dUT1_secd);

    // convert from floating point sec & frac sec to integral s & ns

    // The seconds are already guaranteed to be integral, so a simple cast
    // works here.
    long dUT1_s = (long) dUT1_secd;

    // The fracsec are doubles, with precision ~ 1e-16 s, so rounding 
    // will produce the correct number of ns.
    // NOTE: dUT1 is only accurate to ~microseconds at best, so
    // nanoseconds is comfortably overkill.
    long dUT1_ns = (long) round(1e9 * dUT1_fracsec);


    // Convert UT1 to SI seconds (subtract dUT1), and re-base to J2000
    // (by subtracting J2000 in jy)
    int64_t t_J2000_s  = t_ut1.tv_sec  - J2000_s_jd  - dUT1_s;
    int64_t t_J2000_ns = t_ut1.tv_nsec - J2000_ns_jd - dUT1_ns;

    // Now re-base to the unix epoch for instrument time.
    int64_t t_s  = t_J2000_s  + J2000_s_unix;
    int64_t t_ns = t_J2000_ns + J2000_ns_unix;

    // Make sure nanoseconds >= 0 and < 1'000'000'000
    int64_t full_sec_over = t_ns / GIGA;
    if(t_ns < 0)
        full_sec_over -= 1;
    
    t_s += full_sec_over;
    t_ns -= full_sec_over * GIGA;

    return timespec{.tv_sec=t_s, .tv_nsec=t_ns};
}

double get_ERA_from_UT1(const timespec &ut1) {
    // ERA = 2pi * (0.7790572732640 + 1.00273781191135448 * (Tu(d) - 2451545))
    //
    //     = 2pi * (0.7790572732640
    //              + Tu(d)_frac
    //              + 0.00273781191135448 * (Tu(d) - 2451545)

    long t0_sec = 2'451'545L * 86400L;

    long t_sec = ut1.tv_sec - t0_sec;
    long t_nsec = ut1.tv_nsec;

    //auto dv = std::div(t_sec, 86400);

    double temp;
    double day_sec = t_sec / 86400.0;
    double day_nsec = 1.0e-9 * (t_nsec / 86400.0);
    double dayfrac_sec = modf(day_sec, &temp);
    double dayfrac_nsec = modf(day_nsec, &temp);
    double sidcor_frac_sec =  modf(2.73781191135448e-3 * day_sec, &temp);
    double sidcor_frac_nsec = modf(2.73781191135448e-3 * day_nsec, &temp);

    double f = modf(0.7790572732640 + dayfrac_sec + dayfrac_nsec
                    + sidcor_frac_sec + sidcor_frac_nsec, &temp);
    if(f < 0.0)
        f += 1;

    return 360.0 * f;
}

double get_ERA_from_time(const timespec &time, double dUT) {

    timespec ut1 = get_UT1_from_time(time, dUT);
    return get_ERA_from_UT1(ut1);
}

