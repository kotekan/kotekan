#ifndef BEAM_UTIL_HPP
#define BEAM_UTIL_HPP

#include "json.hpp" // for nlohmann::json

#include <iostream>

/**
 *  Simple structures for inputting, outputting, and passing around Beam metadata:
 *      - ID (a uint64_t)
 *      - Position on the sky (a pair of doubles)
 *
 *
 *  ## Fixed Beams ##
 *
 *  Beams fixed relative to the telescope are given as the x & y components of the beam's normalized
 *  pointing vector `n` in the GRID frame (ie. direction cosines, for more on frames see
 *  lib/util/Telescope.hpp).
 *
 *  The GRID frame is an orthonormal basis whose:
 *      - x-axis is parallel to fiducial East/West feed separation vector (pointing ~East).
 *      - y-axis is parallel to fiducial North/South feed separation vector (pointing ~North).
 *      - z-axis is normal to the fiducial feed plane (pointing ~Up).
 *
 *  Note: the actual telescope feed grid is not perfectly square.  The GRID frame is an orthonormal
 *  frame which fits the physical feed locations as well as possible, as determined by the telescope
 *  developers.
 *
 *  In the GRID frame, a pointing vector n has components n = (nx, ny, nz). n is normalizaed (|n| =
 *1), and the components nx, ny, nz are dimensionless. An upward-looking beam position will have nz
 *> 0, which can be computed from nz = sqrt(1 - nx^2 - ny^2).
 *
 *  The FRBBeam and FixedBBBeam hold exactly `nx` and `ny` in their `x_dir_grid` and `y_dir_grid`
 *  fields.
 *
 *
 *  ## Tracking Beams ##
 *
 *  Tracking beams are fixed to a position on the sky, and move with respect to the telescope.  Our
 *  tracking beams are fixed in CIRS coordinates, which takes into account Earth rotation and proper
 *  motion
 *  but *not* precession/nutation.  Beam positions are given as Right Ascention (RA) and Declination
 *  (Dec) in degrees, in the `ra_cirs_deg` and `dec_cirs_deg` fields.
 *
 **/
namespace Beams {

struct FRBBeam {
    uint64_t id;       /// Beam ID
    double x_dir_grid; /// x component of beam pointing vector in GRID frame
    double y_dir_grid; /// x component of beam pointing vector in GRID frame
};

struct FixedBBBeam {
    uint64_t id;       /// Beam ID
    double x_dir_grid; /// x component of beam pointing vector in GRID frame
    double y_dir_grid; /// y component of beam pointing vector in GRID frame
};

struct TrackingBBBeam {
    uint64_t id;         /// Beam ID
    double ra_cirs_deg;  /// Right Ascension of beam target in CIRS coordinates, in degrees.
    double dec_cirs_deg; /// Declination of beam target in CIRS coordinates, in degrees.
};

/**
 * For outputting the beam directly with `{}` in a `fmt` functions
 **/
std::string format_as(const FRBBeam& b);
std::string format_as(const FixedBBBeam& b);
std::string format_as(const TrackingBBBeam& b);

/**
 * For outputting the beam via std::cout and other streams. Needed by BOOST
 **/
std::ostream& operator<<(std::ostream& os, const FRBBeam& b);
std::ostream& operator<<(std::ostream& os, const FixedBBBeam& b);
std::ostream& operator<<(std::ostream& os, const TrackingBBBeam& b);

/**
 * Serialize beams into JSON
 **/
void to_json(nlohmann::json& j, const FRBBeam& b);
void to_json(nlohmann::json& j, const FixedBBBeam& b);
void to_json(nlohmann::json& j, const TrackingBBBeam& b);


/**
 * Deserialize beams from JSON
 **/
void from_json(const nlohmann::json& j, FRBBeam& b);
void from_json(const nlohmann::json& j, FixedBBBeam& b);
void from_json(const nlohmann::json& j, TrackingBBBeam& b);
} // namespace Beams

#endif
