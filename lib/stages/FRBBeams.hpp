#include "fmt.hpp"  // for fmt
#include "json.hpp" // for nlohmann::json

#include <iostream>

struct FRBBeam {
    uint64_t id;
    double x_dir_grid;
    double y_dir_grid;
};

inline std::string format_as(const FRBBeam& b) {
    return fmt::format("FRBBeam(id: {:d}, x_dir_grid: {:f}, y_dir_grid: {:f})", b.id, b.x_dir_grid, b.y_dir_grid);
}

inline std::ostream& operator<<(std::ostream& os, const FRBBeam& b) {
    os << format_as(b);
    return os;
}

inline void to_json(nlohmann::json& j, const FRBBeam& b) {
    j = {};
    j.emplace("id", b.id);
    j.emplace("x_dir_grid", b.x_dir_grid);
    j.emplace("y_dir_grid", b.y_dir_grid);
}

inline void from_json(const nlohmann::json& j, FRBBeam& b) {
    b.id = j.at("id");
    b.x_dir_grid = j.at("x_dir_grid");
    b.y_dir_grid = j.at("y_dir_grid");
}
