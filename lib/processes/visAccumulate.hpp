
/*****************************************
@file
@brief Accumulation and gating of visibility data.
- visAccumulate : public KotekanProcess
*****************************************/
#ifndef VIS_ACCUMULATE_HPP
#define VIS_ACCUMULATE_HPP

#include <cstdint>
#include <fstream>
#include <functional>
#include <time.h>

#include "buffer.h"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"
#include "pulsarTiming.hpp"


class gateSpec {
    std::string name;
    bool enabled;
};


class gateInternalState {
private:
    Buffer* buf;
    int frame_id;
    gateSpec spec;
    std::function<float(timespec, float)> weightfunc;

    friend visAccumulate;
}


/**
 * @class visAccumulate
 * @brief Accumulate the high rate GPU output into integrated visBuffers.
 *
 * This process will accumulate the GPU output and calculate the within sample
 * variance for weights.
 *
 * @par Buffers
 * @buffer in_buf
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 * @buffer out_buf
 *         @buffer_format visBuffer
 *         @buffer_metadata visMetadata
 *
 * @conf  samples_per_data_set  Int. The number of samples each GPU buffer has
 *                              been integrated for.
 * @conf  num_gpu_frames        Int. The number of GPU frames to accumulate over.
 * @conf  integration_time      Float. Requested integration time in seconds.
 *                              This can be used as an alterative to
 *                              `num_gpu_frames` (which it overrides).
 *                              Internally it picks the nearest acceptable value
 *                              of `num_gpu_frames`.
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in the
 *                              correlator data.
 * @conf  num_freq_in_frame     Int. Number of frequencies in each GPU frame.
 * @conf  block_size            Int. The block size of the packed data.
 * @conf  num_ev                Int. The number of eigenvectors to be stored
 * @conf  input_reorder         Array of [int, int, string]. The reordering mapping.
 *                              Only the first element of each sub-array is used and it is the the index of
 *                              the input to move into this new location. The remaining elements of the
 *                              subarray are for correctly labelling the input in ``visWriter``.
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte
 */
class visAccumulate : public KotekanProcess {
public:
    visAccumulate(Config& config,
                  const string& unique_name,
                  bufferContainer &buffer_container);
    ~visAccumulate();
    void apply_config(uint64_t fpga_seq) override;
    void main_thread() override;

private:

    // Buffers to read/write
    Buffer* in_buf;
    Buffer* out_buf;

    // Parameters saved from the config files
    size_t num_elements;
    size_t num_freq_in_frame;
    size_t num_eigenvectors;
    size_t block_size;
    size_t samples_per_data_set;
    size_t num_gpu_frames;

    // Config parameters for pulsar gating
    double rot_freq;  // in Hz
    float pulse_width;  // in s
    Polyco * polyco;

    // The mapping from buffer element order to output file element ordering
    std::vector<uint32_t> input_remap;

    // Helper methods to make code clearer

    // Set initial values of visBuffer
    void initialise_output(Buffer* out_buf, int out_frame_id,
                           int in_frame_id, int freq_ind);

    // Fill in data sections of visBuffer
    void finalise_output(Buffer* out_buf, int out_frame_id,
                         cfloat* vis1, float* vis2, int freq_ind,
                         uint32_t total_samples);

    // Abstraction of gating accumulation
    // TODO: does it need more that input_frame_id ?
    typedef void(visAccumulate::*gating_func)(int);
    // List of gates and whether they are enabled
    std::map<std::string, gating_func> gating;
    std::map<std::string, bool> gating_enabled;

    void pulsar_gating(int in_frame_id);
};

#endif