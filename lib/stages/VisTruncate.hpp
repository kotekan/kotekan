#ifndef VISTRUNCATE
#define VISTRUNCATE

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.h"
#include "bufferContainer.hpp"

#include <string> // for string

/**
 * @class VisTruncate
 * @brief Truncates visibility, eigenvalue and weight values.
 *
 * eigenvalues and weights are truncated with a fixed precision that is set in
 * the config. visibility values are truncated to a precision based on their
 * weight.
 *
 * @warning Don't run this anywhere but on the transpose (gossec) node.
 * The OpenMP calls could cause issues on systems using kotekan pin
 * priority threads (likely the GPU nodes).
 *
 * @par Buffers
 * @buffer in_buf The input stream.
 *         @buffer_format VisBuffer.
 *         @buffer_metadata VisMetadata
 * @buffer out_buf The output stream with truncated values.
 *         @buffer_format VisBuffer.
 *         @buffer_metadata VisMetadata
 *
 * @conf   err_sq_lim               Limit for the error of visibility truncation.
 * @conf   weight_fixed_precision   Fixed precision for weight truncation.
 * @conf   data_fixed_precision     Fixed precision for eigenvector and visibility truncation (if
 * weights are zero).
 *
 * @author Tristan Pinsonneault-Marotte, Rick Nitsche
 */
class VisTruncate : public kotekan::Stage {
public:
    /// Constructor; loads parameters from config
    VisTruncate(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~VisTruncate() = default;

    /// Main loop over buffer frames
    void main_thread() override;

private:
    // Buffers
    Buffer* in_buf;
    Buffer* out_buf;

    // Truncation parameters
    float err_sq_lim;
    float w_prec;
    float vis_prec;

    // Flag for frame with a zero weight
    bool zero_weight_found;
};

#endif
