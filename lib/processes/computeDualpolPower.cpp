#include "computeDualpolPower.hpp"

#include "Config.hpp"
#include "buffer.h"
#include "errors.h"
#include "nt_memcpy.h"
#include "time_tracking.h"
#include "util.h"
#include "vdif_functions.h"
#ifdef MAC_OSX
#include "osxBindCPU.hpp"

#include <immintrin.h>
#endif
#include <arpa/inet.h>
#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <functional>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <memory.h>
#include <pmmintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
#include <time.h>
#include <unistd.h>

//#define PACKET_OFFSET 58
//#define NUM_POL 2
#define PACKET_LEN (num_freq + VDIF_HEADER_LEN)
//#define PACKET_LEN 1056
#define VDIF_HEADER_LEN sizeof(VDIFHeader)
//#define NUM_FREQ 1024

REGISTER_KOTEKAN_PROCESS(computeDualpolPower);

computeDualpolPower::computeDualpolPower(Config& config, const string& unique_name,
                                         bufferContainer& buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&computeDualpolPower::main_thread, this)) {
    buf_in = get_buffer("vdif_in_buf");
    register_consumer(buf_in, unique_name.c_str());
    buf_out = get_buffer("power_out_buf");
    register_producer(buf_out, unique_name.c_str());

    timesteps_in = config.get<int>(unique_name, "samples_per_data_set");
    integration_length = config.get<int>(unique_name, "power_integration_length");
    timesteps_out = timesteps_in / integration_length;
    num_freq = config.get<int>(unique_name, "num_freq");
    num_elem = config.get<int>(unique_name, "num_elements");

    if (timesteps_in % timesteps_out) {
        ERROR("BAD COMBINATION OF BUFFER & INTEGRATION LENGTHS!");
    }

    integration_count = (uint*)malloc(num_elem * sizeof(uint));
}

computeDualpolPower::~computeDualpolPower() {
    free(integration_count);
}

void computeDualpolPower::main_thread() {
    srand(time(NULL));

    int buf_in_id = 0;
    int buf_out_id = 0;

    const int nthreads = 1;
    int nloop = timesteps_out / nthreads;
    std::thread this_thread[nthreads];

    while (!stop_thread) {
        in_local = (unsigned char*)wait_for_full_frame(buf_in, unique_name.c_str(), buf_in_id);
        if (in_local == NULL)
            break;
        out_local = (unsigned char*)wait_for_empty_frame(buf_out, unique_name.c_str(), buf_out_id);
        if (out_local == NULL)
            break;
#ifdef DEBUGGING
        double start_time = e_time();
#endif
        for (int j = 0; j < nthreads; j++) {
            this_thread[j] = std::thread(&computeDualpolPower::parallelSqSumVdif, this, j, nloop);
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            for (auto& i : config.get<std::vector<int>>(unique_name, "cpu_affinity"))
                CPU_SET(i, &cpuset);
            pthread_setaffinity_np(this_thread[j].native_handle(), sizeof(cpu_set_t), &cpuset);
        }
        for (int j = 0; j < nthreads; j++)
            this_thread[j].join();

        DEBUG("TIME USED FOR INTEGRATION: %fms\n", (e_time() - start_time) * 1000);

        mark_frame_empty(buf_in, unique_name.c_str(), buf_in_id);
        mark_frame_full(buf_out, unique_name.c_str(), buf_out_id);
        buf_in_id = (buf_in_id + 1) % buf_in->num_frames;
        buf_out_id = (buf_out_id + 1) % (buf_out->num_frames);
    }
}

void computeDualpolPower::parallelSqSumVdif(int loop_idx, int loop_length) {
    uint temp_buffer[num_freq * num_elem];
    for (int i = loop_idx * loop_length; i < (loop_idx + 1) * loop_length; i++)
        fastSqSumVdif(in_local + (i * integration_length * PACKET_LEN * num_elem), temp_buffer,
                      (uint*)(out_local + i * (1 + num_freq) * num_elem * sizeof(uint)));
}

#ifdef __AVX2__
inline void computeDualpolPower::fastSqSumVdif(unsigned char* data, uint* temp_buf, uint* out) {
    memset((void*)integration_count, 0, num_elem * sizeof(uint));
    for (int packet = 0; packet < integration_length; ++packet) {
        for (int pol = 0; pol < num_elem; ++pol) {
            const int idx_header = packet * PACKET_LEN * num_elem + pol * PACKET_LEN;
            if (((struct VDIFHeader*)&data[idx_header])->invalid)
                continue;
            integration_count[pol]++;

            for (int freq = 0; freq < num_freq / 32; freq++) {
                const int index =
                    packet * PACKET_LEN * num_elem + pol * PACKET_LEN + freq * 32 + VDIF_HEADER_LEN;

                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;

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

                int out_index = pol * num_freq + freq * 32;

                if (packet != 0) {

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
                }
                // Store new integrated power sum
                _mm256_storeu_si256((__m256i*)&temp_buf[out_index + 0 * 8], ymm0);
                _mm256_storeu_si256((__m256i*)&temp_buf[out_index + 1 * 8], ymm1);
                _mm256_storeu_si256((__m256i*)&temp_buf[out_index + 2 * 8], ymm2);
                _mm256_storeu_si256((__m256i*)&temp_buf[out_index + 3 * 8], ymm3);
            }
        }
    }

    //    INFO("INTEGRATION COUNT: %d %d", integration_count[0], integration_count[1]);
    // Reorder the numbers.
    for (int p = 0; p < num_elem; p++) {
        for (int i = 0; i < num_freq; ++i) {
            // Fix stupid index problem
            int m32 = i % 32;
            if (m32 < 16)
                m32 = (m32 / 4) * 4;
            else
                m32 = -12 + ((m32 - 16) / 4) * 4;
            out[i + p * (num_freq + 1)] = temp_buf[i + m32 + p * (num_freq + 1)];
        }
        out[p * (num_freq + 1) + num_freq] = integration_count[p];
    }
}
#else
inline void computeDualpolPower::fastSqSumVdif(unsigned char* data, uint* temp_buf, uint* out) {
    ERROR("This system does not support AVX2, fast square-and-sum will not work");
}
#endif
