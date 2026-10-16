/**
 * @file
 * @brief Placement of the dishes on the dish grid of the FRB1 beamformers
 */

#ifndef FRB_DISH_LAYOUT_HPP
#define FRB_DISH_LAYOUT_HPP

#include "Telescope.hpp"      // for grid_idx_2d_t
#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO

#include <cstddef> // for size_t
#include <vector>  // for vector

namespace kotekan {

/**
 * @brief The dish at each cell of the FRB1 dish grid
 *
 * The FRB1 kernels (`julia/kernels/frb.jl`) place every dish at a cell of an `M` x `N` grid
 * (their dish layout `S`) and multiply it by the weight `W` of that cell. Main array dishes sit at
 * their grid indices. The remaining cells are filled in order (`m` slow, `n` fast) with the other
 * dishes: first the input dishes outside the main array (e.g. RFI antennas, Missing dishes, whose
 * grid indices are negative), then padding dishes `num_dishes`, `num_dishes + 1`, ... that do not
 * exist in the input.
 *
 * @param grid_indices The grid indices (m, n) of the input dishes; its size is `num_dishes`
 * @param M            The size of the dish grid in the fast direction
 * @param N            The size of the dish grid in the slow direction
 * @return The dish index at cell `m + M * n`
 */
inline std::vector<int> frb_dish_index_at_location(const std::vector<grid_idx_2d_t>& grid_indices,
                                                   const int M, const int N) {
    const int num_dishes = grid_indices.size();
    if (num_dishes > M * N)
        FATAL_ERROR_NON_OO("{:d} dishes do not fit the {:d}x{:d} dish grid", num_dishes, M, N);

    std::vector<int> dish_index_at_loc(M * N, -1);
    std::vector<int> surplus_dishes;
    for (int d = 0; d < num_dishes; ++d) {
        const grid_idx_2d_t grid_idx = grid_indices.at(d);
        if (grid_idx.at(0) < 0 || grid_idx.at(1) < 0) {
            surplus_dishes.push_back(d);
            continue;
        }
        if (grid_idx.at(0) >= M || grid_idx.at(1) >= N)
            FATAL_ERROR_NON_OO("Dish {:d} has grid indices ({:d}, {:d}) outside the {:d}x{:d} "
                               "dish grid",
                               d, grid_idx.at(0), grid_idx.at(1), M, N);
        const int loc_idx = grid_idx.at(0) + M * grid_idx.at(1);
        if (dish_index_at_loc.at(loc_idx) >= 0)
            FATAL_ERROR_NON_OO("Dishes {:d} and {:d} have the same grid indices ({:d}, {:d})",
                               dish_index_at_loc.at(loc_idx), d, grid_idx.at(0), grid_idx.at(1));
        dish_index_at_loc.at(loc_idx) = d;
    }
    for (int d = num_dishes; d < M * N; ++d)
        surplus_dishes.push_back(d);

    // Every dish is either on the grid or surplus, and there are M * N of them, so the surplus
    // dishes exactly fill the empty cells
    std::size_t surplus_idx = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            if (dish_index_at_loc.at(m + M * n) < 0)
                dish_index_at_loc.at(m + M * n) = surplus_dishes.at(surplus_idx++);

    return dish_index_at_loc;
}

} // namespace kotekan

#endif
