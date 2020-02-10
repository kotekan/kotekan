
#ifndef PULSAR_TIMING_HPP
#define PULSAR_TIMING_HPP

#include <time.h>  // for timespec
#include <utility> // for pair
#include <vector>  // for vector


/**
 * @class Polyco
 * @brief Calculate pulsar timing given polynomial expansion of phase.
 *
 * Based on work by Tara Akhoundsadegh, Carol Ng and Kiyo Masui.
 *
 * @author Tristan Pinsonneault-Marotte
 **/
class Polyco {

public:
    /**
     * @brief Constructor.
     * @param tmid       Reference time in MJD (days).
     * @param dm         Dispersion measure (cm^-3 pc).
     * @param phase_ref  Reference phase at tmid.
     * @param rot_freq   Rotation frequency (Hz), i.e. (period)^-1.
     * @param coeff      List of polynomial coefficients in Tempo format.
     **/
    Polyco(double tmid, float dm, double phase_ref, double rot_freq, std::vector<float> coeff);

    Polyco();

    /**
     * @brief Calculate pulsar phase from time in MJD.
     * @param t    Time in MJD (days).
     * @returns    Phase in number of rotations since reference.
     **/
    double mjd2phase(double t) const;

    /**
     * @brief Calculate pulsar phase from UNIX time.
     * @param t    timespec.
     * @returns    Phase in number of rotations since reference.
     **/
    double unix2phase(timespec t) const;

    /**
     * @brief Calculate next pulse time of arrival.
     * @param t      timespec.
     * @param freq   The frequency (MHz) to use for dispersion delay.
     * @returns      Time after t in seconds.
     **/
    double next_toa(timespec t, float freq) const;

private:
    double tmid;
    float dm;
    double phase_ref;
    double rot_freq;
    std::vector<float> coeff;
};

/**
 * @class SegmentedPolyco
 * @brief Collection of Polyco objects representing timing solutions
 *        over a segmented span of time.
 *
 * @author Tristan Pinsonneault-Marotte
 **/
class SegmentedPolyco {

public:
    /**
     * @brief Constructor.
     *
     * @param rot_freq   Rotation frequency (Hz), i.e. (period)^-1.
     * @param dm         Dispersion measure (cm^-3 pc).
     * @param seg    Length in time (s) of polyco segments.
     * @param tmid       Reference times in MJD (days). i.e. centres of segments.
     * @param phase_ref  Reference phases at tmid
     * @param coeff      Polynomial coefficients in Tempo format.
     *
     * @par Exceptions
     * @raises           std::runtime_error if parameter vectors have different lenghts.
     **/
    SegmentedPolyco(double rot_freq, float dm, float seg, std::vector<double> tmid,
                    std::vector<double> phase_ref, std::vector<std::vector<float>> coeff);

    SegmentedPolyco();

    /**
     * @brief Get the polyco that is valid at a given time.
     * @param t    Time to when polyco is required.
     * @returns    The corresponding Polyco object.
     *
     * @par Exceptions
     * @raises     std::runtime_error if no polyco is found.
     **/
    const Polyco& get_polyco(timespec t) const;

    /**
     * @brief Get the polyco at the given index.
     * @param i    Index of polyco list. Negative values will index backwards.
     * @returns    The corresponding Polyco object.
     *
     * @par Exceptions
     * @raises     std::runtime_error if no polyco is found.
     **/
    const Polyco& get_polyco(int i) const;

private:
    std::vector<Polyco> polycos;
    std::vector<std::pair<double, double>> segments;
};

inline double ts2mjd(timespec t) {
    // number of days between UNIX epoch and MJD epoch is 40587
    return ((double)t.tv_sec / 86400.) + ((double)t.tv_nsec / 86400 / 1e9) + 40587;
};

#endif
