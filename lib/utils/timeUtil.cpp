#include <math.h>
#include "timeUtil.hpp"

static constexpr double C = 2.99792458e8;

timespec get_TAI_from_GPS(const timespec &gps_time) {
    return timespec{
                .tv_sec=(gps_time.tv_sec+19),
                .tv_nsec=gps_time.tv_nsec};
}

timespec get_UT1_from_GPS(const timespec &gps_time, double dAT, double dUT) {
    
    // This is the Julian Date of the GPS epoch in seconds.
    // The GPS epoch is 1980-01-06 00:00:00 UTC,
    // this is 1980-10-06 00:00:19 TAI,
    // which is Julian Date 2444244.5002199074074074 (days),
    // which in seconds (x86400) is 211182724819.0
    long gps_t0_jd_sec = 211182724819;
    long gps_t0_jd_nsec = 0;

    double dAT_secd, dUT_secd;
    double dAT_fracsec = modf(dAT, &dAT_secd);
    double dUT_fracsec = modf(dUT, &dUT_secd);

    long GIGA = 1000000000;

    //These conversions truncate instead of round, might lose a nanosecond
    //in the conversion. --> 15 nas in ERA
    long dAT_sec = (long) dAT_secd;
    long dUT_sec = (long) dUT_secd;

    // fracsec values are < 1.0 s and have ~16 digits precision, so they
    // can represent ns with "exactly".
    long dT_nsec = (long) round(1e9 * (dUT_fracsec - dAT_fracsec));

    long ut1_sec = gps_t0_jd_sec + gps_time.tv_sec + dUT_sec - dAT_sec;
    long ut1_nsec = gps_t0_jd_nsec + gps_time.tv_nsec + dT_nsec;

    while(ut1_nsec < 0) {
        ut1_nsec += GIGA;
        ut1_sec -= 1;
    }

    while(ut1_nsec >= GIGA) {
        ut1_nsec -= GIGA;
        ut1_sec += 1;
    }

    return timespec{.tv_sec=ut1_sec, .tv_nsec=ut1_nsec};
}

double get_ERA_from_UT1(const timespec &ut1) {
    // ERA = 2pi * (0.7790572732640 + 1.00273781191135448 * (Tu(d) - 2451545))
    //
    //     = 2pi * (0.7790572732640
    //              + Tu(d)_frac
    //              + 0.00273781191135448 * (Tu(d) - 2451545)

    long t0_sec = 2451545L * 86400L;

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

double get_ERA_from_GPS(const timespec &gpstime, double dAT, double dUT) {

    timespec ut1 = get_UT1_from_GPS(gpstime, dAT, dUT);
    return get_ERA_from_UT1(ut1);
}

void fringestopping_phases_linear(std::vector<double>& phases,
                                  double nu, double era, double era0,
                                  const std::vector<int> &dish_indices,
                                  const std::vector<float> &dish_positions,
                                  int n_elements, int n_dish,
                                  double lat) {

    double two_pi_k = 2*M_PI*C/nu * cos(lat);

    int el = 0;

    for(int i = 0; i < n_elements; i++) {
        for(int j = 0; j <= i; j++) {
            int i_dish = i % n_dish;
            int j_dish = j % n_dish;
            //Assuming here dish_positions is Ndish x 2, with E/W in the
            //first position and N/S in the second.
            float dish_sep_ew = dish_positions[2*i_dish]
                                - dish_positions[2*j_dish];

            // In principle we could compute the precise component of the
            // baseline transverse to the local meridian and the Earth's
            // current rotational axis, but instead we'll assume
            // for now this is just the E/W distance.

            double Phi = two_pi_k * dish_sep_ew * (era - era0);

            phases[2*el] = cos(Phi);
            phases[2*el+1] = sin(Phi);
            el++;
        }
    }
}

void fringestopping_phases(std::vector<double>& phases,
                           double nu, double era, double era0,
                           const std::vector<int> &dish_indices,
                           const std::vector<float> &dish_positions,
                           int n_elements, int n_dish,
                           double lat) {

    double two_pi_k = 2*M_PI*C/nu * cos(lat);

    int el = 0;

    for(int i = 0; i < n_elements; i++) {
        for(int j = 0; j <= i; j++) {
            int i_dish = i % n_dish;
            int j_dish = j % n_dish;
            //Assuming here dish_positions is Ndish x 2, with E/W in the
            //first position and N/S in the second.
            float dish_sep_ew = dish_positions[2*i_dish]
                                - dish_positions[2*j_dish];

            // In principle we could compute the precise component of the
            // baseline transverse to the local meridian and the Earth's
            // current rotational axis, but instead we'll assume
            // for now this is just the E/W distance.

            double Phi = two_pi_k * dish_sep_ew * (era - era0);

            phases[2*el] = cos(Phi);
            phases[2*el+1] = sin(Phi);
            el++;
        }
    }
}
