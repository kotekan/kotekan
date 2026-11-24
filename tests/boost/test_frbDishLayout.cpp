#define BOOST_TEST_MODULE "test_frbDishLayout"

#include "frbDishLayout.hpp"         // for frb_dish_index_at_location
#include "kotekanLoggingFixture.hpp" // for kotekan_logging_fixture

#include <boost/test/included/unit_test.hpp> // for BOOST_PP_IIF_1, BOOST_CHECK_EQUAL_COLLECTIONS
#include <vector>                            // for vector

using kotekan::frb_dish_index_at_location;

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

BOOST_AUTO_TEST_CASE(_full_grid) {
    // A full main array is left as it is
    const std::vector<int> array_dish_at_loc = {3, 0, 5, 1, 4, 2};
    const std::vector<int> dish_at_loc = frb_dish_index_at_location(array_dish_at_loc, 6, 2, 3);
    BOOST_CHECK_EQUAL_COLLECTIONS(dish_at_loc.begin(), dish_at_loc.end(), array_dish_at_loc.begin(),
                                  array_dish_at_loc.end());
}

BOOST_AUTO_TEST_CASE(_surplus_dishes) {
    // A 2x3 grid with three main array dishes, one dish (1) outside the main array, and two
    // padding dishes. The empty cells are filled in the order (m, n) = (0, 0), (0, 1), (1, 2).
    const std::vector<int> array_dish_at_loc = {-1, 0, -1, 3, 2, -1};
    const std::vector<int> dish_at_loc = frb_dish_index_at_location(array_dish_at_loc, 4, 2, 3);
    const std::vector<int> expected = {1, 0, 4, 3, 2, 5};
    BOOST_CHECK_EQUAL_COLLECTIONS(dish_at_loc.begin(), dish_at_loc.end(), expected.begin(),
                                  expected.end());
}

BOOST_AUTO_TEST_CASE(_no_array_dishes) {
    // Without main array dishes the dishes fill the grid in order (m, n) = (0, 0), (0, 1), (1, 0),
    // (1, 1)
    const std::vector<int> dish_at_loc = frb_dish_index_at_location({-1, -1, -1, -1}, 3, 2, 2);
    const std::vector<int> expected = {0, 2, 1, 3};
    BOOST_CHECK_EQUAL_COLLECTIONS(dish_at_loc.begin(), dish_at_loc.end(), expected.begin(),
                                  expected.end());
}
