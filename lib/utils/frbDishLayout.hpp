/**
 * @file
 * @brief Placement of the dishes on the dish grid of the FRB1 beamformers
 */

#ifndef FRB_DISH_LAYOUT_HPP
#define FRB_DISH_LAYOUT_HPP

#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO

#include <cstddef> // for size_t
#include <vector>  // for vector

namespace kotekan {

/**
 * @brief The dish at each cell of the FRB1 dish grid
 *
 * The FRB1 kernels (`julia/kernels/frb.jl`) place every dish at a cell of an `M` x `N` grid
 * (their dish layout `S`) and multiply it by the weight `W` of that cell. Main array dishes sit at
 * their grid location (see `CHORDTelescope::get_dish_idx_at_grid_indices`). The remaining cells
 * are filled in order (`m` slow, `n` fast) with the other dishes: first the input dishes outside
 * the main array (e.g. RFI antennas, Missing dishes), then padding dishes `num_dishes`,
 * `num_dishes + 1`, ... that do not exist in the input.
 *
 * @param array_dish_at_loc The main array dish at cell `m + M * n`, or -1 if there is none
 * @param num_dishes        The number of input dishes
 * @param M                 The size of the dish grid in the fast direction
 * @param N                 The size of the dish grid in the slow direction
 * @return The dish index at cell `m + M * n`
 */
inline std::vector<int> frb_dish_index_at_location(const std::vector<int>& array_dish_at_loc,
                                                   const int num_dishes, const int M, const int N) {
    if (num_dishes > M * N)
        FATAL_ERROR_NON_OO("{:d} dishes do not fit the {:d}x{:d} dish grid", num_dishes, M, N);
    if (array_dish_at_loc.size() != std::size_t(M * N))
        FATAL_ERROR_NON_OO("The dish table has {:d} cells, but the dish grid is {:d}x{:d}",
                           array_dish_at_loc.size(), M, N);

    std::vector<bool> placed(M * N, false);
    for (const int d : array_dish_at_loc) {
        if (d < 0)
            continue;
        if (d >= num_dishes || placed.at(d))
            FATAL_ERROR_NON_OO("Dish {:d} is not a unique input dish (num_dishes {:d})", d,
                               num_dishes);
        placed.at(d) = true;
    }

    // Every dish is either in the main array or surplus, and there are M * N of them, so the
    // surplus dishes exactly fill the empty cells
    std::vector<int> dish_index_at_loc = array_dish_at_loc;
    int surplus_dish = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            if (dish_index_at_loc.at(m + M * n) >= 0)
                continue;
            while (placed.at(surplus_dish))
                ++surplus_dish;
            dish_index_at_loc.at(m + M * n) = surplus_dish++;
        }
    }

    return dish_index_at_loc;
}

} // namespace kotekan

#endif
