#include "fmt.hpp"  // for fmt
#include "json.hpp" // for nlohmann::json

#include <iostream>

struct FRBBeam {
    uint64_t id;
    double x_dir_grid;
    double y_dir_grid;
};

struct FixedBBBeam {
    uint64_t id;
    double x_dir_grid;
    double y_dir_grid;
};

struct TrackingBBBeam {
    uint64_t id;
    double ra_cirs_deg;
    double dec_cirs_deg;
};

inline std::string format_as(const FRBBeam& b) {
    return fmt::format("FRBBeam(id: {:d}, x_dir_grid: {:f}, y_dir_grid: {:f})", b.id, b.x_dir_grid, b.y_dir_grid);
}

inline std::string format_as(const FixedBBBeam& b) {
    return fmt::format("FixedBBBeam(id: {:d}, x_dir_grid: {:f}, y_dir_grid: {:f})", b.id, b.x_dir_grid, b.y_dir_grid);
}

inline std::string format_as(const TrackingBBBeam& b) {
    return fmt::format("TrackingBBBeam(id: {:d}, ra_cirs_deg: {:f}, dec_cirs_deg: {:f})", b.id, b.ra_cirs_deg, b.dec_cirs_deg);
}


inline std::ostream& operator<<(std::ostream& os, const FRBBeam& b) {
    os << format_as(b);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const FixedBBBeam& b) {
    os << format_as(b);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const TrackingBBBeam& b) {
    os << format_as(b);
    return os;
}

inline void to_json(nlohmann::json& j, const FRBBeam& b) {
    j = {};
    j.emplace("id", b.id);
    j.emplace("x_dir_grid", b.x_dir_grid);
    j.emplace("y_dir_grid", b.y_dir_grid);
}

inline void to_json(nlohmann::json& j, const FixedBBBeam& b) {
    j = {};
    j.emplace("id", b.id);
    j.emplace("x_dir_grid", b.x_dir_grid);
    j.emplace("y_dir_grid", b.y_dir_grid);
}

inline void to_json(nlohmann::json& j, const TrackingBBBeam& b) {
    j = {};
    j.emplace("id", b.id);
    j.emplace("ra_cirs_deg", b.ra_cirs_deg);
    j.emplace("dec_cirs_deg", b.dec_cirs_deg);
}

inline void from_json(const nlohmann::json& j, FRBBeam& b) {
    b.id = j.at("id");
    b.x_dir_grid = j.at("x_dir_grid");
    b.y_dir_grid = j.at("y_dir_grid");
}

inline void from_json(const nlohmann::json& j, FixedBBBeam& b) {
    b.id = j.at("id");
    b.x_dir_grid = j.at("x_dir_grid");
    b.y_dir_grid = j.at("y_dir_grid");
}

inline void from_json(const nlohmann::json& j, TrackingBBBeam& b) {
    b.id = j.at("id");
    b.ra_cirs_deg = j.at("ra_cirs_deg");
    b.dec_cirs_deg = j.at("dec_cirs_deg");
}
