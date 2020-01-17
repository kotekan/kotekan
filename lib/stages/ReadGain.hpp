/**
 * @file
 * @brief read in new gain file for FRB/PSR when available
 *  - ReadGain : public kotekan::Stage
 */

#ifndef READ_GAIN
#define READ_GAIN

#include "Stage.hpp"
#include "restServer.hpp"

#include <condition_variable>
#include <mutex>
#include <vector>

using std::vector;

/**
 * @class ReadGain
 * @brief read in new gain files for FRB/PSR when available
 *
 * @par Buffers
 * @buffer gain_frb_buf Array of gains size 2048*2
 *     @buffer_format Array of @c floats
 *     @buffer_metadata none
 * @buffer gain_psr_buf Array of gains size 2048*2*nbeam
 *     @buffer_format Array of @c floats
 *     @buffer_metadata none
 *
 * @conf   num_elements         Int (default 2048). Number of elements
 * @conf   num_pulsar           Int (default 10). Number of pulsar beams to be formed
 * @conf   scaling              Float (default 1.0). Scaling factor on gains
 * @conf   default_gains        Float array (default 1+1j). Default gain value if gain file is
 * missing
 *
 * The gain path is registered as a subscriber to an updatable config block.
 * For the FRB, it is one directory path: '{"frb_gain_dir":"the_new_path"}'
 * For the PSR, it is an array of 10 paths for each of the 10 beams:
 * '{"pulsar_gain_dir":["path0","path1","path2","path3","path4","path5","path6","path7","path8","path9"]}'
 *
 * @author Cherry Ng
 *
 */
class ReadGain : public kotekan::Stage {
public:
    /// Constructor.
    ReadGain(kotekan::Config& config_, const string& unique_name,
             kotekan::bufferContainer& buffer_container);

    void main_thread() override;

    /// Endpoint for providing new directory path for FRB gain updates
    bool update_gains_frb_callback(nlohmann::json& json);

    /// Endpoint for providing new directory path for PSR gain updates
    bool update_gains_psr_callback(nlohmann::json& json);

    /// Read gain file for frb
    void read_gain_frb();
    /// Read gain file for psr
    void read_gain_psr();

private:
    std::condition_variable cond_var;
    /// Lock when change status of update_gains
    std::mutex mux;

    struct Buffer* gain_frb_buf;
    int32_t gain_frb_buf_id;
    struct Buffer* gain_psr_buf;
    int32_t gain_psr_buf_id;

    /// Directory path where gain files are
    string _gain_dir_frb;
    vector<string> _gain_dir_psr;
    /// Default gain values if gain file is missing for this freq, currently set to 1+1j
    vector<float> default_gains;

    /// Buffer for accessing metadata
    Buffer* metadata_buf;
    /// Metadata buffer ID
    int32_t metadata_buffer_id;
    /// Metadata buffer precondition ID
    int32_t metadata_buffer_precondition_id;

    /// Freq bin index, where the 0th is at 800MHz
    int32_t freq_idx;
    /// Freq in MHz
    float freq_MHz;

    /// Scaling factor to be applied on the gains, currently set to 1.0 and somewhat deprecated?
    float scaling;

    /// Flag to control gains to be only loaded on request.
    bool update_gains_frb;
    bool update_gains_psr;

    /// Number of elements, should be 2048
    uint32_t _num_elements;
    /// Number of pulsar beams, should be 10
    int16_t _num_beams;
};


#endif
