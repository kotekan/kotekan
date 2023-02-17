#include "rfiVDIF.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, mark_frame_full, register_consumer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO
#include "util.h"              // for e_time
#include "vdif_functions.h"    // for VDIFHeader

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <string.h>   // for memset, memcpy
#include <vector>     // for vector


using std::string;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(rfiVDIF);

rfiVDIF::rfiVDIF(Config& config, const std::string& unique_name,
                 bufferContainer& buffer_containter) :
    Stage(config, unique_name, buffer_containter, std::bind(&rfiVDIF::main_thread, this)) {
    // Get relevant buffers
    buf_in = get_buffer("vdif_in");
    buf_out = get_buffer("rfi_out");
    // Register stage as consumer and producer
    register_consumer(buf_in, unique_name.c_str());
    register_producer(buf_out, unique_name.c_str());

    // General data paramters
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // Rfi parameters
    _rfi_combined = config.get_default<bool>(unique_name, "rfi_combined", true);
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
}

rfiVDIF::~rfiVDIF() {}

void rfiVDIF::main_thread() {
    // Frame parameters
    uint32_t frame_in_id = 0;
    uint32_t frame_out_id = 0;
    uint8_t* in_frame = nullptr;
    uint8_t* out_frame = nullptr;
    // Set the VDIF block size
    uint32_t VDIF_BLOCK_SIZE = _num_local_freq + sizeof(VDIFHeader);
    // Counters and indices
    uint32_t i, j, k, block_counter, power, rfi_index;
    size_t ptr_counter;
    // Holders for real/imag componenets
    char real, imag;
    // Total integration length
    float M;
    // Declare power arrays
    float power_arr[_num_elements][_num_local_freq];
    float power_sq_arr[_num_elements][_num_local_freq];
    // Invalid Data array
    uint32_t invalid_data_counter[_num_elements];
    uint32_t RFI_Buffer_Size = _num_elements * _num_local_freq * (_samples_per_data_set / _sk_step);
    float S2[_num_local_freq];
    // Create empty Buffer for RFI Values
    if (_rfi_combined) {
        RFI_Buffer_Size /= _num_elements;
    }
    // Buffer to hold kurtosis estimates
    float RFI_Buffer[RFI_Buffer_Size];
    // Initialize Arrays for a single block
    uint8_t block[VDIF_BLOCK_SIZE];
    // Value of current block's element index
    int32_t current_element;
    // Endless Loop
    while (!stop_thread) {
        // Get a new frame
        in_frame = wait_for_full_frame(buf_in, unique_name.c_str(), frame_in_id);
        if (in_frame == nullptr)
            break;
        // Start timer
        double start_time = e_time();
        // Reset Counters
        rfi_index = 0;
        block_counter = 0;
        ptr_counter = 0;
        // Loop through frame
        while (ptr_counter < buf_in->frame_size) {
            // Reset after each _sk_step
            if (block_counter == 0) {
                memset(power_arr, (float)0, sizeof(power_arr));
                memset(power_sq_arr, (float)0, sizeof(power_sq_arr));
                memset(invalid_data_counter, (uint32_t)0, sizeof(invalid_data_counter));
            }
            // Read in first block
            memcpy(block, in_frame + ptr_counter, VDIF_BLOCK_SIZE);
            // Update number of blocks read
            block_counter++;
            // Update the buffer location
            ptr_counter += VDIF_BLOCK_SIZE;
            // Find current input number
            current_element = (int32_t)block[14];

            // Check Validity of the block
            // TODO Why doesn't this work?
            // if((block[3] & 0x1) == 0x1){
            //      invalid_data_counter[current_element]++;
            //      continue;
            //}

            // Sum Across Time
            for (i = 0; i < _num_local_freq; i++) {
                real = ((block[sizeof(VDIFHeader) + i] >> 4) & 0xF) - 8;
                imag = (block[sizeof(VDIFHeader) + i] & 0xF) - 8;
                power = real * real + imag * imag; // Compute power
                power_arr[current_element][i] += power;
                power_sq_arr[current_element][i] += power * power;
            }
            // After a certain amount of timesteps
            if (block_counter == _num_elements * _sk_step) {
                if (_rfi_combined) {
                    // Compute the correct value for M
                    M = _num_elements * _sk_step;
                    for (i = 0; i < _num_elements; i++) {
                        M -= invalid_data_counter[i];
                        for (j = 0; j < _num_local_freq; j++) {
                            // Normalize
                            power_sq_arr[i][j] /=
                                (power_arr[i][j] / _sk_step) * (power_arr[i][j] / _sk_step);
                        }
                    }
                    for (i = 0; i < _num_local_freq; i++) {
                        S2[i] = 0; // Intialize
                        // Sum Across Input
                        for (j = 0; j < _num_elements; j++) {
                            S2[i] += power_sq_arr[j][i];
                        }
                        // Compute Kurtosis for each frequency
                        RFI_Buffer[rfi_index] = ((M + 1) / (M - 1)) * (S2[i] / M - 1);
                        rfi_index++;
                    }
                } else {
                    // For each element
                    for (k = 0; k < _num_elements; k++) {
                        // Compute the correct value for M
                        M = _sk_step - (float)invalid_data_counter[k];
                        for (i = 0; i < _num_local_freq; i++) {
                            // Compute Kurtosis for each frequency
                            RFI_Buffer[rfi_index] =
                                (((M + 1) / (M - 1))
                                 * ((M * power_sq_arr[k][i]) / (power_arr[k][i] * power_arr[k][i])
                                    - 1));
                            INFO("SK value {:f}", RFI_Buffer[rfi_index]);
                            rfi_index++;
                        }
                    }
                }
                // Reset Block Counter
                block_counter = 0;
            }
        }
        // Wait for output frame
        out_frame = wait_for_empty_frame(buf_out, unique_name.c_str(), frame_out_id);
        if (out_frame == nullptr)
            break;
        // Copy results to output frame
        memcpy(out_frame, RFI_Buffer, RFI_Buffer_Size * sizeof(float));
        // Mark output frame full and input frame empty
        mark_frame_full(buf_out, unique_name.c_str(), frame_out_id);
        mark_frame_empty(buf_in, unique_name.c_str(), frame_in_id);
        // Move forward one frame
        frame_out_id = (frame_out_id + 1) % buf_out->num_frames;
        frame_in_id = (frame_in_id + 1) % buf_in->num_frames;
        INFO("Frame {:d} Complete Time {:f}ms", frame_in_id, (e_time() - start_time) * 1000);
    }
}
