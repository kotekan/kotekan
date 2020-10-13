#include "Stack.hpp"

#include "visUtil.hpp" // for rstack_ctype, prod_ctype, input_ctype

#include "fmt.hpp" // for format, fmt

#include <algorithm>  // for copy, sort, transform, max
#include <cstdint>    // for uint32_t, int8_t, int16_t
#include <functional> // for _Bind_helper<>::type, bind, _1, placeholders
#include <iterator>   // for back_insert_iterator, begin, end, back_inserter
#include <math.h>     // for abs
#include <memory>     // for allocator_traits<>::value_type
#include <numeric>    // for iota
#include <stdexcept>  // for invalid_argument
#include <stdlib.h>   // for abs
#include <string>     // for operator<<
#include <tuple>      // for tuple, make_tuple, operator!=, operator<
#include <utility>    // for pair
#include <vector>     // for vector, __alloc_traits<>::value_type

using namespace std::placeholders;


// Stack along the band diagonals
std::pair<uint32_t, std::vector<rstack_ctype>>
stack_diagonal(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods) {
    uint32_t num_elements = inputs.size();
    std::vector<rstack_ctype> stack_def;

    for (auto& p : prods) {
        uint32_t stack_ind = abs(p.input_b - p.input_a);
        bool conjugate = p.input_a > p.input_b;

        stack_def.push_back({stack_ind, conjugate});
    }

    return {num_elements, stack_def};
}


chimeFeed chimeFeed::from_input(input_ctype input) {
    chimeFeed feed;

    if (input.chan_id >= 2048) {
        throw std::invalid_argument("Channel ID is not a valid CHIME feed.");
    }

    feed.cylinder = (input.chan_id / 512);
    feed.polarisation = ((input.chan_id / 256 + 1) % 2);
    feed.feed_location = input.chan_id % 256;

    return feed;
}

std::ostream& operator<<(std::ostream& os, const chimeFeed& f) {
    char cyl_name[4] = {'A', 'B', 'C', 'D'};
    char pol_name[2] = {'X', 'Y'};
    return os << fmt::format(fmt("{:c}{:03d}{:c}"), cyl_name[f.cylinder], f.feed_location,
                             pol_name[f.polarisation]);
}

using feed_diff = std::tuple<int8_t, int8_t, int8_t, int8_t, int16_t>;

// Calculate the baseline parameters and whether the product must be
// conjugated to get canonical ordering
std::pair<feed_diff, bool> calculate_chime_vis(const prod_ctype& p,
                                               const std::vector<input_ctype>& inputs) {

    chimeFeed fa = chimeFeed::from_input(inputs[p.input_a]);
    chimeFeed fb = chimeFeed::from_input(inputs[p.input_b]);

    bool is_wrong_cylorder = (fa.cylinder > fb.cylinder);
    bool is_same_cyl_wrong_feed_order =
        ((fa.cylinder == fb.cylinder) && (fa.feed_location > fb.feed_location));
    bool is_same_feed_wrong_pol_order =
        ((fa.cylinder == fb.cylinder) && (fa.feed_location == fb.feed_location)
         && (fa.polarisation > fb.polarisation));

    bool conjugate = false;

    // Check if we need to conjugate/transpose to get the correct order
    if (is_wrong_cylorder || is_same_cyl_wrong_feed_order || is_same_feed_wrong_pol_order) {

        chimeFeed t = fa;
        fa = fb;
        fb = t;
        conjugate = true;
    }

    return {std::make_tuple(fa.polarisation, fb.polarisation, fa.cylinder, fb.cylinder,
                            fb.feed_location - fa.feed_location),
            conjugate};
}

// Stack along the band diagonals
std::pair<uint32_t, std::vector<rstack_ctype>>
stack_chime_in_cyl(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods) {
    // Calculate the set of baseline properties
    std::vector<std::pair<feed_diff, bool>> bl_prop;
    std::transform(std::begin(prods), std::end(prods), std::back_inserter(bl_prop),
                   std::bind(calculate_chime_vis, _1, inputs));

    // Create an index array for doing the sorting
    std::vector<uint32_t> sort_ind(prods.size());
    std::iota(std::begin(sort_ind), std::end(sort_ind), 0);

    auto sort_fn = [&](const uint32_t& ii, const uint32_t& jj) -> bool {
        return (bl_prop[ii].first < bl_prop[jj].first);
    };
    std::sort(std::begin(sort_ind), std::end(sort_ind), sort_fn);

    std::vector<rstack_ctype> stack_map(prods.size());

    feed_diff cur = bl_prop[sort_ind[0]].first;
    uint32_t cur_stack_ind = 0;

    for (auto& ind : sort_ind) {
        if (bl_prop[ind].first != cur) {
            cur = bl_prop[ind].first;
            cur_stack_ind++;
        }
        stack_map[ind] = {cur_stack_ind, bl_prop[ind].second};
    }

    return {++cur_stack_ind, stack_map};
}
