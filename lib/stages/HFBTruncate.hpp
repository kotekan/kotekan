#ifndef HFBTRUNCATE
#define HFBTRUNCATE

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <string> // for string

/**
 * @class HFBTruncate
 * @brief Truncates absorber data and weight values.
 *
 * Absorber values are truncated to a precision based on their
 * weight.
 *
 * @warning Don't run this anywhere but on the transpose (gossec) node.
 * The OpenMP calls could cause issues on systems using kotekan pin
 * priority threads (likely the GPU nodes).
 *
 * @par Buffers
 * @buffer in_buf The input stream.
 *         @buffer_format HFBBuffer.
 *         @buffer_metadata HFBMetadata
 * @buffer out_buf The output stream with truncated values.
 *         @buffer_format HFBBuffer.
 *         @buffer_metadata HFBMetadata
 *
 * @conf   err_sq_lim               Limit for the error of absorber truncation.
 * @conf   weight_fixed_precision   Fixed precision for weight truncation.
 * @conf   data_fixed_precision     Fixed precision for absorber truncation (if
 *                                  weights are zero).
 *
 * @author James Willis
 */
class HFBTruncate : public kotekan::Stage {
public:
    /// Constructor; loads parameters from config
    HFBTruncate(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    ~HFBTruncate() = default;

    /// Main loop over buffer frames
    void main_thread() override;

private:
    // Buffers
    Buffer* in_buf;
    Buffer* out_buf;

    // Truncation parameters
    float err_sq_lim;
    float w_prec;
    float hfb_prec;

    // Flag for frame with a zero weight
    bool zero_weight_found;
};

#endif
