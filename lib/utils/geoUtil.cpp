#include "geoUtil.hpp"

#include "kotekanLogging.hpp" // for set_log_level, set_log_prefix

#include "fmt.hpp"

#include <iostream>

static constexpr double deg2rad = M_PI / 180.0;

GeoFrame::GeoFrame(const std::string& log_level, const std::string& name, double itrs_lat_deg,
                   double itrs_lon_deg, const vec3d_t& offset_m, const vec3d_t& x_axis,
                   const vec3d_t& y_axis, const vec3d_t& z_axis) :
    name(name), itrs_lat_deg(itrs_lat_deg), itrs_lon_deg(itrs_lon_deg), offset_m(offset_m),
    x_axis(x_axis), y_axis(y_axis), z_axis(z_axis),
    R_topo_to_frame(make_R_topo_to_frame(x_axis, y_axis, z_axis)),
    R_itrs_to_topo(make_R_itrs_to_topo(itrs_lat_deg, itrs_lon_deg)) {

    set_log_level(log_level);
    set_log_prefix(fmt::format("GeoFrame[{:s}]", name));

    // FATAL ERROR if frame is not orthonormal.
    // TODO: Allow for non-orthornormal frames? Would have to compute R inverse
    // instead of assuming R^-1 = R^T
    const double tol = 1.0e-14;

    if (std::abs(vec3d_dot(x_axis, x_axis) - 1.0) > tol) {
        FATAL_ERROR("x_axis is not normalized, length^2 = {}", vec3d_dot(x_axis, x_axis));
    }
    if (std::abs(vec3d_dot(y_axis, y_axis) - 1.0) > tol) {
        FATAL_ERROR("y_axis is not normalized, length^2 = {}", vec3d_dot(y_axis, y_axis));
    }
    if (std::abs(vec3d_dot(z_axis, z_axis) - 1.0) > tol) {
        FATAL_ERROR("z_axis is not normalized, length^2 = {}", vec3d_dot(z_axis, z_axis));
    }
    if (std::abs(vec3d_dot(x_axis, y_axis)) > tol) {
        FATAL_ERROR("x & y axes are not orthogonal, dot product: {}", vec3d_dot(x_axis, y_axis));
    }
    if (std::abs(vec3d_dot(x_axis, z_axis)) > tol) {
        FATAL_ERROR("x & z axes are not orthogonal, dot product: {}", vec3d_dot(x_axis, z_axis));
    }
    if (std::abs(vec3d_dot(y_axis, z_axis)) > tol) {
        FATAL_ERROR("y & z axes are not orthogonal, dot product: {}", vec3d_dot(y_axis, z_axis));
    }
}

std::string GeoFrame::get_name() const {
    return name;
}

double GeoFrame::get_itrs_lat_deg() const {
    return itrs_lat_deg;
}

double GeoFrame::get_itrs_lon_deg() const {
    return itrs_lon_deg;
}

vec3d_t GeoFrame::get_offset_m() const {
    return offset_m;
}

vec3d_t GeoFrame::get_x_axis() const {
    return x_axis;
}

vec3d_t GeoFrame::get_y_axis() const {
    return y_axis;
}

vec3d_t GeoFrame::get_z_axis() const {
    return z_axis;
}

mat3x3d_t GeoFrame::get_R_topo_to_frame() const {
    return R_topo_to_frame;
}

mat3x3d_t GeoFrame::get_R_itrs_to_topo() const {
    return R_itrs_to_topo;
}

vec3d_t GeoFrame::vec_itrs_to_topo(const vec3d_t& v_itrs) const {
    return mat3x3d_vecmul(R_itrs_to_topo, v_itrs);
}

vec3d_t GeoFrame::vec_topo_to_itrs(const vec3d_t& v_topo) const {
    return mat3x3d_vecmul_transpose(R_itrs_to_topo, v_topo);
}

vec3d_t GeoFrame::vec_topo_to_frame(const vec3d_t& v_topo) const {
    return mat3x3d_vecmul(R_topo_to_frame, v_topo);
}

vec3d_t GeoFrame::vec_frame_to_topo(const vec3d_t& v_frame) const {
    return mat3x3d_vecmul_transpose(R_topo_to_frame, v_frame);
}

vec3d_t GeoFrame::vec_itrs_to_frame(const vec3d_t& v_itrs) const {
    // itrs and frame are connected through topo
    vec3d_t v_topo = vec_itrs_to_topo(v_itrs);
    vec3d_t v_frame = vec_topo_to_frame(v_topo);
    return v_frame;
}

vec3d_t GeoFrame::vec_frame_to_itrs(const vec3d_t& v_frame) const {
    // itrs and frame are connected through topo
    vec3d_t v_topo = vec_frame_to_topo(v_frame);
    vec3d_t v_itrs = vec_topo_to_itrs(v_topo);
    return v_itrs;
}

mat3x3d_t GeoFrame::make_R_topo_to_frame(const vec3d_t& x_axis, const vec3d_t& y_axis,
                                         const vec3d_t& z_axis) {
    mat3x3d_t R;

    for (int j = 0; j < 3; j++) {
        R[0][j] = x_axis[j];
        R[1][j] = y_axis[j];
        R[2][j] = z_axis[j];
    }

    return R;
}

mat3x3d_t GeoFrame::make_R_itrs_to_topo(double lat_deg, double lon_deg) {

    double cos_lon = cos(deg2rad * lon_deg);
    double sin_lon = sin(deg2rad * lon_deg);
    double cos_lat = cos(deg2rad * lat_deg);
    double sin_lat = sin(deg2rad * lat_deg);

    mat3x3d_t R;

    // Topocentric X (East) in ITRS (Earth-centered, Earth-fixed) coords
    R[0][0] = -sin_lon;
    R[0][1] = cos_lon;
    R[0][2] = 0.0;

    // Topocentric Y (North) in ITRS (Earth-centered, Earth-fixed) coords
    R[1][0] = -sin_lat * cos_lon;
    R[1][1] = -sin_lat * sin_lon;
    R[1][2] = cos_lat;

    // Topocentric Z (Up) in ITRS (Earth-centered, Earth-fixed) coords
    R[2][0] = cos_lat * cos_lon;
    R[2][1] = cos_lat * sin_lon;
    R[2][2] = sin_lat;

    return R;
}

vec3d_t vec3d_cross(const vec3d_t& a, const vec3d_t& b) {
    vec3d_t c;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}

double vec3d_dot(const vec3d_t& a, const vec3d_t& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

vec3d_t mat3x3d_vecmul(const mat3x3d_t A, const vec3d_t& x) {
    vec3d_t y = {0, 0, 0};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            y[i] += A[i][j] * x[j];

    return y;
}

vec3d_t mat3x3d_vecmul_transpose(const mat3x3d_t A, const vec3d_t& x) {
    vec3d_t y = {0, 0, 0};

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            y[i] += A[j][i] * x[j];

    return y;
}

vec3d_t vec3d_axes_rotation_R1(const vec3d_t& v, double theta) {
    // Return coordinates of vector v in frame rotated by theta about x-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    vec3d_t v_rot = {v[0], cos_th * v[1] + sin_th * v[2], -sin_th * v[1] + cos_th * v[2]};

    return v_rot;
}

vec3d_t vec3d_axes_rotation_R2(const vec3d_t& v, double theta) {
    // Return coordinates of vector v in frame rotated by theta about y-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    vec3d_t v_rot = {cos_th * v[0] - sin_th * v[2], v[1], sin_th * v[0] + cos_th * v[2]};

    return v_rot;
}

vec3d_t vec3d_axes_rotation_R3(const vec3d_t& v, double theta) {
    // Return coordinates of vector v in frame rotated by theta about z-axis

    double cos_th = cos(theta);
    double sin_th = sin(theta);

    vec3d_t v_rot = {cos_th * v[0] + sin_th * v[1], -sin_th * v[0] + cos_th * v[1], v[2]};

    return v_rot;
}
