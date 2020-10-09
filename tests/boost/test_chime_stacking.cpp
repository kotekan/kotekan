#define BOOST_TEST_MODULE "test_chime_stacking"

#include <boost/test/included/unit_test.hpp>  // for BOOST_PP_IIF_1, BOOST_PP_IIF_0, BOOST_PP_BO...
#include <algorithm>                          // for copy, max, transform
#include <cstdint>                            // for uint32_t, uint16_t
#include <memory>                             // for allocator_traits<>::value_type
#include <numeric>                            // for iota
#include <ostream>                            // for operator<<, ostream, basic_ostream, basic_o...
#include <stdexcept>                          // for invalid_argument
#include <string>                             // for string
#include <utility>                            // for pair
#include <vector>                             // for vector, vector<>::iterator

#include "Stack.hpp"                          // for stack_chime_in_cyl, chimeFeed, CYL_A, CYL_D
#include "datasetState.hpp"                   // for invert_stack
#include "visUtil.hpp"                        // for input_ctype, prod_ctype, rstack_ctype, stac...


// Teach boost to understand how to print the stack types...
std::ostream& operator<<(std::ostream& os, rstack_ctype const& pr) {
    os << "<" << pr.stack << "," << pr.conjugate << ">";
    return os;
}

std::ostream& operator<<(std::ostream& os, stack_ctype const& pr) {
    os << "<" << pr.prod << "," << pr.conjugate << ">";
    return os;
}

BOOST_AUTO_TEST_CASE(_chimeFeed) {
    // Pick a few input value and compare to them decode by hand
    chimeFeed f1 = chimeFeed::from_input(input_ctype(10, ""));
    BOOST_CHECK_EQUAL(f1.cylinder, CYL_A);
    BOOST_CHECK_EQUAL(f1.polarisation, POL_Y);
    BOOST_CHECK_EQUAL(f1.feed_location, 10);

    f1 = chimeFeed::from_input(input_ctype(1873, ""));
    BOOST_CHECK_EQUAL(f1.cylinder, CYL_D);
    BOOST_CHECK_EQUAL(f1.polarisation, POL_X);
    BOOST_CHECK_EQUAL(f1.feed_location, 81);

    BOOST_CHECK_THROW(chimeFeed::from_input(input_ctype(2049, "")), std::invalid_argument);
}


BOOST_AUTO_TEST_CASE(chimeStacking) {
    // Set up some inputs and products.
    // Here we do three feeds in a line in a single cylinder and same pol
    std::vector<input_ctype> inputs = {{0, ""}, {1, ""}, {2, ""}};
    std::vector<prod_ctype> prods = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
    // Hand calculated answer
    uint32_t nstack = 3;
    std::vector<rstack_ctype> stack_map = {{0, false}, {1, false}, {2, false},
                                           {0, false}, {1, false}, {0, false}};

    auto stack = stack_chime_in_cyl(inputs, prods);
    BOOST_CHECK_EQUAL(stack.first, nstack);
    BOOST_CHECK_EQUAL_COLLECTIONS(stack.second.begin(), stack.second.end(), stack_map.begin(),
                                  stack_map.end());

    // Here we do two feeds in a line but with both pols
    inputs = {{0, ""}, {1, ""}, {256, ""}, {257, ""}};
    prods = {{0, 2}, {1, 3}};
    // Hand calculated answer
    nstack = 1;
    stack_map = {{0, true}, {0, true}};

    stack = stack_chime_in_cyl(inputs, prods);
    BOOST_CHECK_EQUAL(stack.first, nstack);
    BOOST_CHECK_EQUAL_COLLECTIONS(stack.second.begin(), stack.second.end(), stack_map.begin(),
                                  stack_map.end());

    // ... try across cylinders
    inputs = {{1, ""}, {2, ""}, {512, ""}, {514, ""}, {515, ""}};
    prods = {{0, 2}, {0, 3}, {1, 4}};
    // Hand calculated answer
    nstack = 2;
    stack_map = {{0, false}, {1, false}, {1, false}};

    stack = stack_chime_in_cyl(inputs, prods);
    BOOST_CHECK_EQUAL(stack.first, nstack);
    BOOST_CHECK_EQUAL_COLLECTIONS(stack.second.begin(), stack.second.end(), stack_map.begin(),
                                  stack_map.end());

    // .. check that the ordering is correct (i.e. pol is slowest)
    inputs = {{0, ""}, {1, ""}, {770, ""}, {771, ""}};
    prods = {{0, 1}, {2, 3}};
    // Hand calculated answer
    nstack = 2;
    stack_map = {{1, false}, {0, false}};

    stack = stack_chime_in_cyl(inputs, prods);
    BOOST_CHECK_EQUAL(stack.first, nstack);
    BOOST_CHECK_EQUAL_COLLECTIONS(stack.second.begin(), stack.second.end(), stack_map.begin(),
                                  stack_map.end());


    // Try a full CHIME example where we'll test the total number and a few obvious indices
    inputs.clear();
    prods.clear();
    for (uint16_t i = 0; i < 2048; i++) {
        inputs.emplace_back(i, "");
        for (uint16_t j = i; j < 2048; j++) {
            prods.push_back({i, j});
        }
    }

    stack_map = {{}};
    nstack = 16356;

    stack = stack_chime_in_cyl(inputs, prods);
    BOOST_CHECK_EQUAL(stack.first, nstack);

    // Check a few indices that we have calculated by hand
    BOOST_CHECK_EQUAL(stack.second[491648].stack, 0u); // Auto of Cyl A Feed 0 pol X
    BOOST_CHECK_EQUAL(stack.second[493440].stack, 0u); // Auto of Cyl A Feed 1 pol X
    BOOST_CHECK_EQUAL(stack.second[491649].stack, 1u); // Delta 1 sep for XX in Cyl A


    // Check the reverse mapping
    auto istack = invert_stack(stack.first, stack.second);

    // Check the precalculated values above (though this relies on the sort order)
    BOOST_CHECK_EQUAL(istack[0].prod, 491648u);
    BOOST_CHECK_EQUAL(istack[1].prod, 491649u);

    // Check that all the entries are the inverses
    std::vector<uint32_t> stack_ind1(stack.first);
    std::vector<uint32_t> stack_ind2(stack.first);
    std::iota(stack_ind1.begin(), stack_ind1.end(), 0);
    std::transform(stack_ind1.begin(), stack_ind1.end(), stack_ind2.begin(),
                   [&](uint32_t ind) -> uint32_t { return stack.second[istack[ind].prod].stack; });

    BOOST_CHECK_EQUAL_COLLECTIONS(stack_ind1.begin(), stack_ind1.end(), stack_ind2.begin(),
                                  stack_ind2.end());
}
