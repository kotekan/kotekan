#ifndef STACK_HPP
#define STACK_HPP

#include "visUtil.hpp" // for rstack_ctype

#include <cstdint> // for uint32_t
#include <utility> // for pair
#include <vector>  // for vector


/**
 * @brief Combine all feeds on the same diagonal of the correlation matrix.
 *
 * This is mostly useful for testing as in most cases it will combine
 * non-redundant visibilities.
 *
 * @param inputs The set of inputs.
 * @param prods  The products we are stacking.
 *
 * @returns Stack definition.
 **/
std::pair<uint32_t, std::vector<rstack_ctype>>
stack_diagonal(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods);

/**
 * @brief Stack redundant baselines between cylinder pairs for CHIME.
 *
 * This stacks together redundant baselines between cylinder pairs, but does
 * not stack distinct pairs together. For instance A1,B2 will be stacked in the
 * same group as A2,B3, but not the same group as B1,C2.
 *
 * This will give back stacks ordered by (polarisation pair, cylinder pair, NS
 * feed separation).
 *
 * @param inputs The set of inputs.
 * @param prods  The products we are stacking.
 *
 * @returns Stack definition.
 **/
std::pair<uint32_t, std::vector<rstack_ctype>>
stack_chime_in_cyl(const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods);

#define CYL_A 0
#define CYL_B 1
#define CYL_C 2
#define CYL_D 3

#define POL_X 0
#define POL_Y 1


/**
 * @brief The properties of a CHIME feed
 **/
struct chimeFeed {

    /// The cylinder the feed is on.
    int8_t cylinder;

    /// The polarisation of the feed
    int8_t polarisation;

    /// The feed location running South to North
    int16_t feed_location;

    /**
     * @brief Get the CHIME feed properties from an input.
     *
     * @param input The input to calculate.
     *
     * @returns The feed definition.
     **/
    static chimeFeed from_input(input_ctype input);
};

/**
 * @brief Implement an output operator to help debugging.
 **/
std::ostream& operator<<(std::ostream& os, const chimeFeed& f);
#endif // STACK_HPP
