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

#include "json.hpp"

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"
#include "datasetManager.hpp"

// This type is used a lot so let's use an alias
using json = nlohmann::json;


/**
 * @brief Compress visibility data by stacking together equivalent baselines.
 *
 * This task takes a stream of uncompressed visibility data and performs a
 * (weighted) average over the equivalent baselines.
 *
 * @note This task requires there to be an inputState and prodState registered
 *       for the incoming dataset.
 *
 * @par Buffers
 * @buffer in_bufs The input uncompressed data.
 *         @buffer_format visBuffer structured.
 *         @buffer_metadata visMetadata
 * @buffer out_buf The merged and transformed buffer
 *         @buffer_format visBuffer structured
 *         @buffer_metadata visMetadata
 *
 * @conf stack_type      String. Type of stacking to apply to the data. Look at
 *                       documentation of stack_X functions for details.
 * @conf exclude_inputs  List of ints. Extra inputs to exclude from stack.
 *
 * @par Metrics
 * @metric kotekan_baselinecompression_residuals
 *      The variance of the residuals.
 * @metric kotekan_baselinecompression_time_seconds
 *      The time elapsed to process one frame.
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

    // Main loop for the process: Creates n threads that do the compression.
    void main_thread() override;

private:

	/// Entrancepoint for n threads. Each thread takes frames with a
	/// different frame_id from the buffer and compresses them.
    void compress_thread(int thread_id);

    /// Tracks input dataset ID and gets output dataset IDs from manager
    void get_states(dset_id_t ds_id);

    /// Vector to hold the thread handles
    std::vector<std::thread> thread_handles;

    // The extra inputs we are excluding
    std::vector<uint32_t> exclude_inputs;

    // Alias for the type of a stack definition function. Damn C++ is verbose :)
    // Return is a pair of (num stacks total, stack_map), where stack_map is a
    // vector of (stack_output_index, conjugate) pairs. In this conjugate
    // describes whether the input must be complex conjugated prior to stacking.
    // Inputs are the usual input map and product map.
    using stack_def_fn = std::function<
        std::pair<uint32_t, std::vector<rstack_ctype>>(
            const std::vector<input_ctype>&, const std::vector<prod_ctype>&
        )
    >;

    /// Map from the name of a stack to its definiting function.
    std::map<std::string, stack_def_fn> stack_type_defs;

    /// The stack function to use.
    stack_def_fn calculate_stack;

    /// Number of parallel threads accessing the same buffers (default 1)
    uint32_t num_threads;

    // Buffers to read/write
    Buffer* in_buf;
    Buffer* out_buf;

    // dataset states and IDs
    dset_id_t output_dset_id;
    const prodState* prod_state_ptr;
    const stackState* stack_state_ptr;
};


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
std::pair<uint32_t, std::vector<rstack_ctype>> stack_diagonal(
    const std::vector<input_ctype>& inputs,
    const std::vector<prod_ctype>& prods
);

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
std::pair<uint32_t, std::vector<rstack_ctype>> stack_chime_in_cyl(
    const std::vector<input_ctype>& inputs,
    const std::vector<prod_ctype>& prods
);

/**
 * @brief Take an a rstack map and generate a stack->prod mapping.
 *
 * @param num_stack Total number of stacks.
 * @param stack_map The prod->stack mapping.
 *
 * @returns The stack->prod mapping.
 **/
std::vector<stack_ctype> invert_stack(
    uint32_t num_stack, const std::vector<rstack_ctype>& stack_map);


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