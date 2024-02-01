#include "rfiAVXVDIF.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for mark_frame_empty, mark_frame_full, register_consumer, reg...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG
#include "vdif_functions.h"    // for VDIFHeader

#ifdef DEBUGGING
#include "util.h" // for e_time
#endif

#include <atomic>     // for atomic_bool
#include <cstdint>    // for uint32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#ifdef __AVX2__
#include <immintrin.h> // for __m256i, _mm256_loadu_si256, _mm256_add_epi32, _mm256_mul...
#endif
#include <pthread.h> // for pthread_setaffinity_np
#include <regex>     // for match_results<>::_Base_type
#include <sched.h>   // for cpu_set_t, CPU_SET, CPU_ZERO
#include <stdexcept> // for runtime_error
#include <stdlib.h>  // for srand
#include <string.h>  // for memset
#include <thread>    // for thread
#include <time.h>    // for time
#include <vector>    // for vector

#ifdef MAC_OSX
#include "osxBindCPU.hpp"
#endif

#define PACKET_LEN (_num_local_freq + VDIF_HEADER_LEN)
#define VDIF_HEADER_LEN sizeof(VDIFHeader)

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(rfiAVXVDIF);

rfiAVXVDIF::rfiAVXVDIF(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rfiAVXVDIF::main_thread, this)) {
    // Retrieve buffers
    buf_in = get_buffer("vdif_in");
    buf_out = get_buffer("rfi_out");
    // Register Consumer/Producer
    register_consumer(buf_in, unique_name.c_str());
    register_producer(buf_out, unique_name.c_str());

    // Apply config.
    // Standard config variables
    _num_local_freq = config.get<uint32_t>(unique_name, "num_local_freq");
    _num_elements = config.get<uint32_t>(unique_name, "num_elements");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    // RFI config variables
    _sk_step = config.get_default<uint32_t>(unique_name, "sk_step", 256);
    _rfi_combined = config.get_default<bool>(unique_name, "rfi_combined", true);
}

rfiAVXVDIF::~rfiAVXVDIF() {}

void rfiAVXVDIF::main_thread() {
    // Random Seed
    srand(time(nullptr));
    // Declare frame IDs
    uint32_t frame_in_id = 0;
    uint32_t frame_out_id = 0;
    // Declare number of threads and number of loops each thread will make
    uint32_t nthreads = 1;
    uint32_t nloop = (_samples_per_data_set / _sk_step) / nthreads;
    std::thread this_thread[nthreads];
    // Endless Loop
    while (!stop_thread) {
        // Get input frame
        in_local = (uint8_t*)wait_for_full_frame(buf_in, unique_name.c_str(), frame_in_id);
        if (in_local == nullptr)
            break;
        // Get output frame
        out_local = (uint8_t*)wait_for_empty_frame(buf_out, unique_name.c_str(), frame_out_id);
        if (out_local == nullptr)
            break;
#ifdef DEBUGGING
        // Start timer
        double start_time = e_time();
#endif
        // Create threads to do parallel spectral kurtosis
        for (uint32_t j = 0; j < nthreads; j++) {
            this_thread[j] = std::thread(&rfiAVXVDIF::parallelSpectralKurtosis, this, j, nloop);
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
                CPU_SET(i, &cpuset);
            pthread_setaffinity_np(this_thread[j].native_handle(), sizeof(cpu_set_t), &cpuset);
        }
        // Wait for threads to end
        for (uint32_t j = 0; j < nthreads; j++) {
            this_thread[j].join();
        }
        // Log/display time results
        DEBUG("Time used for kurtosis calculation: {:f}ms\n", (e_time() - start_time) * 1000);
        // Mark Frames empty/full
        mark_frame_empty(buf_in, unique_name.c_str(), frame_in_id);
        mark_frame_full(buf_out, unique_name.c_str(), frame_out_id);
        // Update frame ids
        frame_in_id = (frame_in_id + 1) % buf_in->num_frames;
        frame_out_id = (frame_out_id + 1) % buf_out->num_frames;
    }
}

void rfiAVXVDIF::parallelSpectralKurtosis(uint32_t loop_idx, uint32_t loop_length) {
    // Declare arrays to hold power and sq power
    uint32_t temp_buffer[_num_local_freq * _num_elements];
    uint32_t sq_temp_buffer[_num_local_freq * _num_elements];
    // Perform fast SK measurement
    for (uint32_t i = loop_idx * loop_length; i < (loop_idx + 1) * loop_length; i++) {
        fastSKVDIF(in_local + (i * _sk_step * PACKET_LEN * _num_elements), temp_buffer,
                   sq_temp_buffer, (float*)(out_local + i * _num_local_freq * sizeof(float)));
    }
}

#ifdef __AVX2__
inline void rfiAVXVDIF::fastSKVDIF(uint8_t* data, uint32_t* temp_buf, uint32_t* sq_temp_buf,
                                   float* out) {
    // Reset integration count
    uint32_t integration_count[_num_elements];
    memset(integration_count, (uint32_t)0, sizeof(integration_count));
    // Loop through pckates and elements (polarizations)
    for (uint32_t packet = 0; packet < _sk_step; ++packet) {
        for (uint32_t pol = 0; pol < _num_elements; ++pol) {
            // Compute index in the input data
            uint32_t idx_header = PACKET_LEN * (packet * _num_elements + pol);
            // Ignore faulty packets
            if (((struct VDIFHeader*)&data[idx_header])->invalid)
                continue;
            // Kepp track of how many good packets there are
            integration_count[pol]++;
            // Loop through frequencies
            for (uint32_t freq = 0; freq < _num_local_freq / 32; freq++) {
                // Compute current index in data
                uint32_t index = idx_header + freq * 32 + VDIF_HEADER_LEN;
                uint32_t out_index = pol * _num_local_freq + freq * 32;
                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, sq0, sq1, sq2, sq3;
                // Load 64 4 bit numbers
                ymm0 = _mm256_loadu_si256((__m256i const*)&data[index]);
                // Shift the high 4-bits to the low 4-bits in each 8 bit block
                ymm1 = _mm256_srli_epi64(ymm0, 4); // real
                // Mask out the lower 4 bits
                ymm2 = _mm256_set1_epi32(0x0f0f0f0f);
                ymm0 = _mm256_and_si256(ymm0, ymm2); // imag
                ymm1 = _mm256_and_si256(ymm1, ymm2); // real
                // This packs (real, imag) (8+8) pairs together
                ymm3 = _mm256_unpacklo_epi8(ymm0, ymm1);
                ymm4 = _mm256_unpackhi_epi8(ymm0, ymm1);
                // subtract 8 to make the 8-bit numbers twos complement
                ymm2 = _mm256_set1_epi8(8);
                ymm3 = _mm256_sub_epi8(ymm3, ymm2);
                ymm4 = _mm256_sub_epi8(ymm4, ymm2);
                // Take the abs value since the multi is unsigned
                ymm3 = _mm256_abs_epi8(ymm3);
                ymm4 = _mm256_abs_epi8(ymm4);
                // Multiply and add real and imaginary 8+8-bit pairs into 16-bit ints
                ymm5 = _mm256_maddubs_epi16(ymm3, ymm3); // hi
                ymm6 = _mm256_maddubs_epi16(ymm4, ymm4); // lo
                // Extend to 32-bit
                ymm7 = _mm256_set1_epi32(0);
                ymm0 = _mm256_unpacklo_epi16(ymm5, ymm7);
                ymm1 = _mm256_unpackhi_epi16(ymm5, ymm7);
                ymm2 = _mm256_unpacklo_epi16(ymm6, ymm7);
                ymm3 = _mm256_unpackhi_epi16(ymm6, ymm7);
                // If not the first packet
                if (packet != 0) {
                    // Load integrated power**2
                    ymm4 = _mm256_loadu_si256((__m256i const*)&sq_temp_buf[out_index + 0 * 8]);
                    ymm5 = _mm256_loadu_si256((__m256i const*)&sq_temp_buf[out_index + 1 * 8]);
                    ymm6 = _mm256_loadu_si256((__m256i const*)&sq_temp_buf[out_index + 2 * 8]);
                    ymm7 = _mm256_loadu_si256((__m256i const*)&sq_temp_buf[out_index + 3 * 8]);
                    // Get the current power**2 and add to integration
                    sq0 = _mm256_add_epi32(_mm256_mullo_epi32(ymm0, ymm0), ymm4);
                    sq1 = _mm256_add_epi32(_mm256_mullo_epi32(ymm1, ymm1), ymm5);
                    sq2 = _mm256_add_epi32(_mm256_mullo_epi32(ymm2, ymm2), ymm6);
                    sq3 = _mm256_add_epi32(_mm256_mullo_epi32(ymm3, ymm3), ymm7);
                    // Load Integrated Power
                    ymm4 = _mm256_loadu_si256((__m256i const*)&temp_buf[out_index + 0 * 8]);
                    ymm5 = _mm256_loadu_si256((__m256i const*)&temp_buf[out_index + 1 * 8]);
                    ymm6 = _mm256_loadu_si256((__m256i const*)&temp_buf[out_index + 2 * 8]);
                    ymm7 = _mm256_loadu_si256((__m256i const*)&temp_buf[out_index + 3 * 8]);
                    // Add current power to integrated sum
                    ymm0 = _mm256_add_epi32(ymm0, ymm4);
                    ymm1 = _mm256_add_epi32(ymm1, ymm5);
                    ymm2 = _mm256_add_epi32(ymm2, ymm6);
                    ymm3 = _mm256_add_epi32(ymm3, ymm7);
                } else {
                    // Compute power squared if it's the first packet
                    sq0 = _mm256_mullo_epi32(ymm0, ymm0);
                    sq1 = _mm256_mullo_epi32(ymm1, ymm1);
                    sq2 = _mm256_mullo_epi32(ymm2, ymm2);
                    sq3 = _mm256_mullo_epi32(ymm3, ymm3);
                }
                // Store new integrated power sum
                _mm256_storeu_si256((__m256i*)&temp_buf[out_index + 0 * 8], ymm0);
                _mm256_storeu_si256((__m256i*)&temp_buf[out_index + 1 * 8], ymm1);
                _mm256_storeu_si256((__m256i*)&temp_buf[out_index + 2 * 8], ymm2);
                _mm256_storeu_si256((__m256i*)&temp_buf[out_index + 3 * 8], ymm3);
                // Store new integrated power squared sum
                _mm256_storeu_si256((__m256i*)&sq_temp_buf[out_index + 0 * 8], sq0);
                _mm256_storeu_si256((__m256i*)&sq_temp_buf[out_index + 1 * 8], sq1);
                _mm256_storeu_si256((__m256i*)&sq_temp_buf[out_index + 2 * 8], sq2);
                _mm256_storeu_si256((__m256i*)&sq_temp_buf[out_index + 3 * 8], sq3);
            }
        }
    }
    for (uint32_t i = 0; i < _num_local_freq; ++i) {
        // Fix stupid index problem
        uint32_t m32 = i % 32;
        if (m32 < 16)
            m32 = (m32 / 4) * 4;
        else
            m32 = -12 + ((m32 - 16) / 4) * 4;
        // Reset Element sum for each freq
        float M = 0;
        float sq_power_across_element = 0;
        // Sum both elements
        for (uint32_t p = 0; p < _num_elements; p++) {
            float mean =
                ((float)temp_buf[i + m32 + p * (_num_local_freq + 1)] / integration_count[p]);
            sq_power_across_element +=
                (float)sq_temp_buf[i + m32 + p * (_num_local_freq + 1)] / (mean * mean);
            M += (float)integration_count[p];
        }
        // Calculate Kurtosis and output
        out[i] = ((M + 1) / (M - 1)) * (sq_power_across_element / M - 1);
    }
}
#else
inline void rfiAVXVDIF::fastSKVDIF(uint8_t* data, uint32_t* temp_buf, uint32_t* sq_temp_buf,
                                   float* out) {
    (void)data;
    (void)temp_buf;
    (void)sq_temp_buf;
    (void)out;
    ERROR("This system does not support AVX2, fast spectral kurtosis will not work");
}
#endif
