#ifndef GEO_UTIL_HPP
#define GEO_UTIL_HPP

#include "kotekanLogging.hpp" // for kotekanLogging

#include <array>

// Shorthand for common coordinate vectors and matrices.
using vec3d_t = std::array<double, 3>;
using vec2d_t = std::array<double, 2>;
using mat3x3d_t = std::array<std::array<double, 3>, 3>;

/**
 * @brief A class which defines a cartesian reference frame on the surface of the Earth
 * and facilitates transformations between this frame and the ITRS.
 *
 * The frame exists at a particular point on the Earth specified by geodetic Latitude and
 * Longitude in ITRS. This object is concerned with three reference frames:
 *
 * 1) ITRS (International Terrestrial Reference System). The coordinate system for the planet
 * Earth, where geodetic Latitude and Longitude live. The x-, y-, and z- axes are at latitude and
 * longitude (0, 0), (0, 90), and (90, *), the origin is at Earth Barycenter. The IAU defines the
 * transformations between this frame and sky (e.g. ICRS).
 *
 * 2) TOPO (Topocentric coordinates). A local, Cartesian coordinate system with origin at a point
 * on the Earth, aligned with geodetic latitude, longitude, and altitude.  The x-, y-, and z- axes
 * are directed exactly East, North, and Up at the origin point and are mutually orthogonal.
 *
 * 3) This Frame (name). The frame defined by this object. A local, Cartesian coordinate system,
 * with origin at a point on the Earth, and x-, y-, and z- axes mutually orthonormal but otherwise
 * arbitrary.  This frame's origin may be displaced from the TOPO origin. The axes of this frame
 * are provided in the TOPO basis.
 *
 * In practice this object is often defining the axes of an interferometric array, but it is also
 * used to define the orientation of a dish pointing, and may serve other uses in the future.
 *
 * @author Geoffrey Ryan
 **/
class GeoFrame : public kotekan::kotekanLogging {
public:
    /**
     * @brief Explicit constructor for a GeoFrame
     *
     * @param log_level     The log level for this stage.
     * @param name          The name for this frame, e.g. "grid" or "dish".
     * @param itrs_lat_deg  The ITRS latitude of the origin point for the TOPO frame, in degrees.
     * @param itrs_lon_deg  The ITRS longitude of the origin point for the TOPO frame, in degrees.
     * @param offset_m      This frame's origin point in the TOPO frame in meters.
     *                      frame_origin = topo_origin + offset
     * @param x_axis        This frame's x-axis, in the TOPO basis. Must be normalized
     *                      (|x| = 1.0) and orthogonal to the y- and z-axes.
     * @param y_axis        This frame's y-axis, in the TOPO basis. Must be normalized
     *                      (|y| = 1.0) and orthogonal to the z- and x-axes.
     * @param z_axis        This frame's z-axis, in the TOPO basis. Must be normalized
     *                      (|z| = 1.0) and orthogonal to the x- and y-axes.
     *
     * Will produce a FATAL_ERROR if x, y, and z are not mutually orthonormal
     */
    GeoFrame(const std::string& log_level, const std::string& name, double itrs_lat_deg,
             double itrs_lon_deg, const vec3d_t& offset_m, const vec3d_t& x_axis,
             const vec3d_t& y_axis, const vec3d_t& z_axis);

    /**
     * @brief Return this frame's name.
     */
    std::string get_name() const;

    /**
     * @brief Return this frame's ITRS latitude in degrees.
     */
    double get_itrs_lat_deg() const;

    /**
     * @brief Return this frame's ITRS longitude in degrees.
     */
    double get_itrs_lon_deg() const;

    /**
     * @brief Return this frame's offset from TOPO in meters.
     */
    vec3d_t get_offset_m() const;

    /**
     * @brief Return this frame's x-axis in TOPO coordinates.
     */
    vec3d_t get_x_axis() const;

    /**
     * @brief Return this frame's y-axis in TOPO coordinates.
     */
    vec3d_t get_y_axis() const;

    /**
     * @brief Return this frame's z-axis in TOPO coordinates.
     */
    vec3d_t get_z_axis() const;

    /**
     * @brief Return the rotation matrix R between the ITRS and TOPO bases.
     *
     * v_topo_i = R_ij * v_itrs_j
     * R_ij = R[i][j]
     *
     * Since these frames all have orthonormal bases, the inverse transform is given
     * by the transpose of R.
     */
    mat3x3d_t get_R_itrs_to_topo() const;

    /**
     * @brief Return the rotation matrix R between the TOPO and frame bases.
     *
     * v_frame_i = R_ij * v_topo_j
     * R_ij = R[i][j]
     *
     * Since these frames all have orthonormal bases, the inverse transform is given
     * by the transpose of R.
     */
    mat3x3d_t get_R_topo_to_frame() const;

    /**
     * @brief Transform the given vector in the ITRS frame to the TOPO frame
     */
    vec3d_t vec_itrs_to_topo(const vec3d_t& v_itrs) const;

    /**
     * @brief Transform the given vector in the TOPO frame to the ITRS frame
     */
    vec3d_t vec_topo_to_itrs(const vec3d_t& v_topo) const;

    /**
     * @brief Transform the given vector in the TOPO frame to this frame
     */
    vec3d_t vec_topo_to_frame(const vec3d_t& v_topo) const;

    /**
     * @brief Transform the given vector in this frame to the TOPO frame
     */
    vec3d_t vec_frame_to_topo(const vec3d_t& v_frame) const;

    /**
     * @brief Transform the given vector in the ITRS frame to this frame (via TOPO)
     */
    vec3d_t vec_itrs_to_frame(const vec3d_t& v_itrs) const;

    /**
     * @brief Transform the given vector in this frame to the ITRS frame (via TOPO)
     */
    vec3d_t vec_frame_to_itrs(const vec3d_t& v_frame) const;

private:
    /**
     * @brief Construct the TOPO -> this frame rotation matrix from this frame's basis vectors.
     * Basis vectors must be orthonormal. The matrix is simply:
     *
     *     (  x^T  )
     * R = (  y^T  )
     *     (  z^T  )
     */
    static mat3x3d_t make_R_topo_to_frame(const vec3d_t& x, const vec3d_t& y, const vec3d_t& z);

    /**
     * @brief Construct the ITRS -> TOPO rotation matrix from the latitude and longitude
     * of the origin.
     */
    static mat3x3d_t make_R_itrs_to_topo(double lat_deg, double lon_deg);

    std::string name;                /// This frame's name
    const double itrs_lat_deg;       /// Latitude of TOPO origin in ITRS
    const double itrs_lon_deg;       /// Longitude of TOPO origin in ITRS
    const vec3d_t offset_m;          /// Offset of frame origin from TOPO
    const vec3d_t x_axis;            /// Frame x-axis in TOPO
    const vec3d_t y_axis;            /// Frame y-axis in TOPO
    const vec3d_t z_axis;            /// Frame z-axis in TOPO
    const mat3x3d_t R_topo_to_frame; /// Rotation matrix TOPO -> frame
    const mat3x3d_t R_itrs_to_topo;  /// Rotation matrix ITRX -> TOPO
};

/**
 * @brief Return the cross product of the two input vectors.
 */
vec3d_t vec3d_cross(const vec3d_t& a, const vec3d_t& b);

/**
 * @brief Return the dot product of the two input vectors.
 */
double vec3d_dot(const vec3d_t& a, const vec3d_t& b);

/**
 * @brief Perform the multiplication A * x for 3x3 matrix A and 3-vector x.
 *
 * A's first index is the row, second is the column, so if y = A * x, then:
 *
 * y[i] = A[i][j] * x[j]  // summed over j
 */
vec3d_t mat3x3d_vecmul(const mat3x3d_t A, const vec3d_t& x);

/**
 * @brief Perform the multiplication A^T * x for 3x3 matrix A and 3-vector x.
 *
 * A's first index is the row, second is the column, so if y = A^T * x, then:
 *
 * y[i] = A[j][i] * x[j]  // summed over j
 */
vec3d_t mat3x3d_vecmul_transpose(const mat3x3d_t A, const vec3d_t& x);

/**
 * @brief   Transform the given vector to a frame where the basis has
 *          rotated about the x axis by an angle theta.
 *
 * @param   v       Input vector
 * @param   theta   Angle basis is rotated by, in radians.
 **/
vec3d_t vec3d_axes_rotation_R1(const vec3d_t& v, double theta);

/**
 * @brief   Transform the given vector to a frame where the basis has
 *          rotated about the y axis by an angle theta.
 *
 * @param   v       Input vector
 * @param   theta   Angle basis is rotated by, in radians.
 **/
vec3d_t vec3d_axes_rotation_R2(const vec3d_t& v, double theta);

/**
 * @brief   Transform the given vector to a frame where the basis has
 *          rotated about the z axis by an angle theta.
 *
 * @param   v       Input vector
 * @param   theta   Angle basis is rotated by, in radians.
 **/
vec3d_t vec3d_axes_rotation_R3(const vec3d_t& v, double theta);

#endif
