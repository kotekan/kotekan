#ifndef GEO_UTIL_HPP
#define GEO_UTIL_HPP

#include "kotekanLogging.hpp" // for kotekanLogging

#include <array>

using vec3d_t = std::array<double, 3>;
using mat3x3d_t = std::array<std::array<double, 3>, 3>;

class GeoFrame : public kotekan::kotekanLogging {
public:
    GeoFrame(const std::string& log_level, const std::string& name, double itrs_lat_deg,
             double itrs_lon_deg, const vec3d_t& offset_m, const vec3d_t& x_axis,
             const vec3d_t& y_axis, const vec3d_t& z_axis);

private:
    static mat3x3d_t make_R_topo_to_frame(const vec3d_t& x, const vec3d_t& y, const vec3d_t& z);
    static mat3x3d_t make_R_itrs_to_topo(double lat_deg, double lon_deg);

    std::string name;
    const double itrs_lat_deg;
    const double itrs_lon_deg;
    const vec3d_t offset_m;
    const vec3d_t x_axis;
    const vec3d_t y_axis;
    const vec3d_t z_axis;
    const mat3x3d_t R_topo_to_frame;
    const mat3x3d_t R_itrs_to_topo;
};

vec3d_t vec3d_cross(const vec3d_t& a, const vec3d_t& b);
double vec3d_dot(const vec3d_t& a, const vec3d_t& b);

#endif
