#include "computeDualpolPower.hpp"
#include "buffers.h"
#include "errors.h"
#include "test_data_generation.h"
#include "error_correction.h"
#include "nt_memcpy.h"
#include "Config.hpp"
#include "util.h"
#include "vdif_functions.h"
#include "time_tracking.h"

#include <dirent.h>
#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <memory.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <inttypes.h>
#include <functional>
#include <thread>

#define PACKET_OFFSET 58
#define NUM_POL 2
#define PACKET_LEN 1056
#define VDIF_HEADER_LEN 32
//#define NUM_FREQ 1024

computeDualpolPower::computeDualpolPower(Config &config_,
                     struct Buffer &buf_in_, struct Buffer &buf_out_) :
                  KotekanProcess(config_, std::bind(&computeDualpolPower::main_thread, this)),
                  buf_in(buf_in_),
                  buf_out(buf_out_)
{
    timesteps_in = config.get_int("/processing/samples_per_data_set");
    integration_length = config.get_int("/raw_capture/integration_length");
    timesteps_out = timesteps_in / integration_length;
    num_freq = config.get_int("/processing/num_local_freq");

    if (timesteps_in % timesteps_out)
    {
        ERROR("BAD COMBINATION OF BUFFER & INTEGRATION LENGTHS!");
    }
}


void computeDualpolPower::apply_config(uint64_t fpga_seq) {
}


computeDualpolPower::~computeDualpolPower() {
}

void computeDualpolPower::main_thread() {
//    in_local = (unsigned char*)malloc(buf_in.buffer_size);
  //  out_local = (unsigned char*)malloc(buf_out.buffer_size);

    int buf_in_id=0;
    int buf_out_id=0;

    static int nthreads=8;
    int nloop = timesteps_out/nthreads;
    std::thread this_thread[nthreads];

    for (EVER) {
/*
        buf_in_id = get_full_buffer_from_list(&buf_in, &buf_in_id, 1);
        memcpy(in_local,buf_in.data[buf_in_id],buf_in.buffer_size);
        mark_buffer_empty(&buf_in, buf_in_id);
        buf_in_id = ( buf_in_id + 1 ) % buf_in.num_buffers;

        for (int j=0; j<nthreads; j++)
            this_thread[j] = std::thread(&computeDualpolPower::parallelSqSumVdif, this, j, nloop);
        for (int j=0; j<nthreads; j++)
            this_thread[j].join();
  
        wait_for_empty_buffer(&buf_out, buf_out_id);
        memcpy(buf_out.data[buf_out_id],out_local,buf_out.buffer_size);
        mark_buffer_full(&buf_out, buf_out_id);
        buf_out_id = (buf_out_id + 1) % (buf_out.num_buffers);
*/
        buf_in_id = get_full_buffer_from_list(&buf_in, &buf_in_id, 1);
        wait_for_empty_buffer(&buf_out, buf_out_id);
        in_local = buf_in.data[buf_in_id];
        out_local = buf_out.data[buf_out_id];

    double start_time = e_time();

        for (int j=0; j<nthreads; j++)
            this_thread[j] = std::thread(&computeDualpolPower::parallelSqSumVdif, this, j, nloop);
        for (int j=0; j<nthreads; j++)
            this_thread[j].join();

    double stop_time = e_time(); 
//    INFO("TIME USED FOR INTEGRATION: %fms\n",(stop_time-start_time)*1000);

        mark_buffer_empty(&buf_in, buf_in_id);
        mark_buffer_full(&buf_out, buf_out_id);
        buf_in_id = ( buf_in_id + 1 ) % buf_in.num_buffers;
        buf_out_id = (buf_out_id + 1) % (buf_out.num_buffers);

   }

    mark_producer_done(&buf_out, 0);

//    free(in_local);
  //  free(out_local);
}


void computeDualpolPower::parallelSqSumVdif(int loop_idx, int loop_length){
    int temp_buffer[num_freq*2];
    for (int i=loop_idx*loop_length; i<(loop_idx+1)*loop_length; i++)
        fastSqSumVdif(in_local+(i*integration_length*PACKET_LEN*NUM_POL),
                temp_buffer, (float*)(out_local+i*num_freq*sizeof(float)));
}

#ifdef __AVX2__
void computeDualpolPower::fastSqSumVdif(unsigned char * data, int * temp_buf, float *out) {
    int *integration_count=(int*)calloc(NUM_POL,sizeof(int));

    for (int packet = 0; packet < integration_length; ++packet) {
        for (int pol = 0; pol < NUM_POL; ++pol) {
            const int idx_header = packet * PACKET_LEN * NUM_POL +
                                    pol * PACKET_LEN;
            if (((struct VDIFHeader*)&data[idx_header])->invalid) continue;
            integration_count[pol]++;

            for (int freq = 0; freq < num_freq / 32; ++freq) {
                const int index = packet * PACKET_LEN * NUM_POL +
                                    pol * PACKET_LEN
                                     + freq*32 + VDIF_HEADER_LEN;

                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;

                // Load 64 4 bit numbers
                ymm0 = _mm256_loadu_si256((__m256i const *)&data[index]);

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
                    ymm4 = _mm256_loadu_si256((__m256i const *)&temp_buf[out_index + 0*8]);
                    ymm5 = _mm256_loadu_si256((__m256i const *)&temp_buf[out_index + 1*8]);
                    ymm6 = _mm256_loadu_si256((__m256i const *)&temp_buf[out_index + 2*8]);
                    ymm7 = _mm256_loadu_si256((__m256i const *)&temp_buf[out_index + 3*8]);

                    ymm0 = _mm256_add_epi32(ymm0, ymm4);
                    ymm1 = _mm256_add_epi32(ymm1, ymm5);
                    ymm2 = _mm256_add_epi32(ymm2, ymm6);
                    ymm3 = _mm256_add_epi32(ymm3, ymm7);
                }

                _mm256_storeu_si256((__m256i *)&temp_buf[out_index + 0*8], ymm0);
                _mm256_storeu_si256((__m256i *)&temp_buf[out_index + 1*8], ymm1);
                _mm256_storeu_si256((__m256i *)&temp_buf[out_index + 2*8], ymm2);
                _mm256_storeu_si256((__m256i *)&temp_buf[out_index + 3*8], ymm3);
            }
        }
    }

//    INFO("INTEGRATION COUNT: %d %d", integration_count[0], integration_count[1])
    // Reorder the numbers.
    for (int i = 0; i < num_freq; ++i) {
        // Fix stupid index problem
        int m32 = i % 32;
        if (m32 < 16) m32 = (m32/4)*4;
        else m32 = -12 + ((m32 - 16)/4)*4;
        out[i         ] = ((float)temp_buf[i+m32         ]) / integration_count[0]; //XX
    }

//INFO("Value: %i, Count: %i\n",temp_buf[0], integration_count[0]);

    for (int i = 0; i < num_freq; ++i) {
        // Fix stupid index problem
        int m32 = i % 32;
        if (m32 < 16) m32 = (m32/4)*4;
        else m32 = -12 + ((m32 - 16)/4)*4;
        out[i+num_freq] = ((float)temp_buf[i+m32+num_freq]) / integration_count[1]; //YY
    }
}
#else
void computeDualpolPower::fastSqSumVdif(int integration_time,
        unsigned char * data, int * temp_buf, int * xx, int * yy)
{
    ERROR("This system does not support AVX2, fast square-and-sum will not work");
}
#endif

