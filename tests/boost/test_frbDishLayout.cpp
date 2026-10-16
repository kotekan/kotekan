#define BOOST_TEST_MODULE "test_frbDishLayout"

#include "Telescope.hpp"             // for grid_idx_2d_t
#include "frbDishLayout.hpp"         // for frb_dish_index_at_location
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture

#include <algorithm>                         // for sort
#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_CHECK_EQUAL_COLLECTIONS
#include <numeric>                           // for iota
#include <vector>                            // for vector

using kotekan::frb_dish_index_at_location;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

BOOST_AUTO_TEST_CASE(_full_grid) {
    // Every cell holds the dish whose grid indices point there
    const int M = 3, N = 4;
    std::vector<grid_idx_2d_t> grid_indices;
    for (int d = 0; d < M * N; ++d)
        grid_indices.push_back({(5 * d) % M, (5 * d / M) % N});
    const std::vector<int> dish_at_loc = frb_dish_index_at_location(grid_indices, M, N);
    BOOST_REQUIRE_EQUAL(dish_at_loc.size(), std::size_t(M * N));
    for (int d = 0; d < M * N; ++d)
        BOOST_CHECK_EQUAL(dish_at_loc.at(grid_indices.at(d)[0] + M * grid_indices.at(d)[1]), d);
}

BOOST_AUTO_TEST_CASE(_surplus_dishes) {
    // A 2x3 grid with three array dishes, one dish outside the main array, and two padding
    // dishes. The empty cells are filled in the order (m, n) = (0, 0), (0, 1), (1, 2).
    const int M = 2, N = 3;
    const std::vector<grid_idx_2d_t> grid_indices = {{1, 0}, {-1, -1}, {0, 2}, {1, 1}};
    const std::vector<int> dish_at_loc = frb_dish_index_at_location(grid_indices, M, N);
    const std::vector<int> expected = {1, 0, 4, 3, 2, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(dish_at_loc.begin(), dish_at_loc.end(), expected.begin(),
                                  expected.end());
}

BOOST_AUTO_TEST_CASE(_no_array_dishes) {
    // Without a dish table no dish is in the main array; the dishes then fill the grid in order
    const int M = 2, N = 2;
    const std::vector<grid_idx_2d_t> grid_indices(3, {-1, -1});
    const std::vector<int> dish_at_loc = frb_dish_index_at_location(grid_indices, M, N);
    // Cells in fill order (m, n) = (0, 0), (0, 1), (1, 0), (1, 1)
    const std::vector<int> expected = {0, 2, 1, 3};
    BOOST_CHECK_EQUAL_COLLECTIONS(dish_at_loc.begin(), dish_at_loc.end(), expected.begin(),
                                  expected.end());

    std::vector<int> sorted = dish_at_loc;
    std::sort(sorted.begin(), sorted.end());
    std::vector<int> all(M * N);
    std::iota(all.begin(), all.end(), 0);
    BOOST_CHECK_EQUAL_COLLECTIONS(sorted.begin(), sorted.end(), all.begin(), all.end());
}
