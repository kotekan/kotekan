/*****************************************
@file
@brief Processes for compressing visibility data.
- baselineCompression
*****************************************/
#ifndef VIS_COMPRESSION_HPP
#define VIS_COMPRESSION_HPP

#include <cstdint>
#include <vector>
#include <tuple>

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"

/**
 * @class baselineCompression
 * @brief Compress visibility data by stacking together equivalent baselines.
 *
 * This task takes a stream of uncompressed visibility data and performs a
 * (weighted) average over the equivalent baselines.
 *
 * @par Buffers
 * @buffer in_bufs The input uncompressed data (must be full N^2).
 *         @buffer_format visBuffer structured.
 *         @buffer_metadata visMetadata
 * @buffer out_buf The merged and transformed buffer
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @author Richard Shaw
 */
class baselineCompression : public KotekanProcess {

public:

    // Default constructor
    baselineCompression(Config &config,
                        const string& unique_name,
                        bufferContainer &buffer_container);

    void apply_config(uint64_t fpga_seq) override;

    // Main loop for the process
    void main_thread() override;

private:

    // TODO: remove this when dataset manager arrives
    uint32_t num_elements;

    // Alias for the type of a stack definition function. Damn C++ is verbose :)
    // Return is a pair of (num stacks total, stack_map), where stack_map is a
    // vector of (stack_output_index, conjugate) pairs. In this conjugate
    // describes whether the input must be complex conjugated prior to stacking.
    // Inputs are the usual input map and product map.
    using stack_def_fn = std::function<
        std::pair<uint32_t, std::vector<std::pair<uint32_t, bool>>>(
            std::vector<input_ctype>&, std::vector<prod_ctype>&
        )
    >;

    /// Map from the name of a stack to its definiting function.
    std::map<std::string, stack_def_fn> stack_type_defs;

    /// The stack function to use.
    stack_def_fn calculate_stack;

    // Buffers to read/write
    Buffer* in_buf;
    Buffer* out_buf;
};

// Stack along the band diagonals
std::pair<uint32_t, std::vector<std::pair<uint32_t, bool>>> stack_diagonal(
    std::vector<input_ctype>& inputs, std::vector<prod_ctype>& prods
);

// Stack along the band diagonals
std::pair<uint32_t, std::vector<std::pair<uint32_t, bool>>> stack_chime_in_cyl(
    std::vector<input_ctype>& inputs, std::vector<prod_ctype>& prods
);

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
     * @params input The input to calculate.
     *
     * @returns The feed definition.
     **/
    static chimeFeed from_input(input_ctype input);
};

/**
 * @brief Implement an output operator to help debugging.
 **/
std::ostream & operator<<(std::ostream &os, const chimeFeed& f);





#endif