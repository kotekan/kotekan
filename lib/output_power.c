
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include "output_power.h"
#include "buffers.h"
#include "errors.h"

#include <immintrin.h>

#define PACKET_OFFSET 58

// Hard coded, to be fixed.
#define NUM_POL 2
#define PACKET_LEN 1056
#define VDIF_HEADER_LEN 32
#define NUM_FREQ 1024

void fast_square_and_sum_vdif(int integration_time,
        unsigned char * data, int * temp_buf, int * xx, int * yy) {


    for (int packet = 0; packet < integration_time; packet++) {
        for (int pol = 0; pol < NUM_POL; ++pol) {
            for (int freq = 0; freq < NUM_FREQ / 32; ++freq) {

                const int index = packet * PACKET_LEN * NUM_POL +
                                    pol * PACKET_LEN
                                    + VDIF_HEADER_LEN + freq*32;

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

                int out_index = pol * NUM_FREQ + freq * 32;

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

    // Reorder the numbers.
    for (int i = 0; i < NUM_FREQ; ++i) {
        // Fix stupid index problem
        int m32 = i % 32;
        if (m32 < 16) {
            m32 = (m32/4)*4;
        } else  {
            m32 = -12 + ((m32 - 16)/4)*4;
        }

        xx[i] = temp_buf[i + m32];
    }

    for (int i = 0; i < NUM_FREQ; ++i) {
        // Fix stupid index problem
        int m32 = i % 32;
        if (m32 < 16) {
            m32 = (m32/4)*4;
        } else  {
            m32 = -12 + ((m32 - 16)/4)*4;
        }

        yy[i] = temp_buf[NUM_FREQ + i + m32];
    }

}


void *output_power_thread(void * arg)
{
    struct output_power_thread_arg * args = (struct output_power_thread_arg *) arg;

    FILE * fd;
    FILE * fd_legacy;
    int useableBufferIDs[1] = {0};
    int bufferID = 0;

    // Open the file to write
    const int file_name_len = 100;
    char file_name[file_name_len];

    snprintf(file_name, file_name_len, "%s/power_data.dat", args->ram_disk);

    const int integration_time = args->integration_samples;

    const int head_size = 3;
    const int line_head_size = 2;

    int line_idx = 0;
    int num_rolls = 0;
    int line_size = args->num_freq*2 + line_head_size;
    const int num_entries = 400000;

    int out_buf[line_size];
    int temp_buf[args->num_freq*2];

    int * xx = out_buf + line_head_size; // Pointer math
    int * yy = out_buf + args->num_freq + line_head_size;

    int out_buf_legacy[args->num_freq*2];
    int * xx_legacy = out_buf_legacy; // Pointer math
    int * yy_legacy = out_buf_legacy + args->num_freq;

    // Delete the file before first use
    unlink(file_name);

    fd = fopen(file_name, "w+");

    if (fd == NULL) {
        ERROR("Cannot open file");
        ERROR("File name was: %s", file_name);
        exit(errno);
    }

    if (args->legacy_output == 1) {
        snprintf(file_name, file_name_len, "%s/power_data_legacy.dat", args->ram_disk);

        unlink(file_name);

        fd_legacy = fopen(file_name, "w+");

        if (fd_legacy == NULL) {
            ERROR("Cannot open legacy file");
            ERROR("File name was: %s", file_name);
            exit(errno);
        }
    }

    // Create a second free running file.

    // Grow the file to full size with zeros
    memset((void *)out_buf, 0, line_size*sizeof(int));
    fwrite((void *)out_buf, sizeof(int), head_size, fd);
    for (int i = 0; i < num_entries ; ++i) {
        fwrite((void *) out_buf, sizeof(int), line_size, fd);
    }
    fseek(fd, 0, SEEK_SET);

    for (;;) {

	// This call is blocking.
        bufferID = get_full_buffer_from_list(args->buf, useableBufferIDs, 1);

        //printf("Got buffer, id: %d", bufferID);

        // Check if the producer has finished, and we should exit.
        if (bufferID == -1) {
            int ret;
            if (fclose(fd) == -1) {
                fprintf(stderr, "Cannot close file");
            }
            pthread_exit((void *) &ret);
        }

        if (args->legacy_output == 1) {
            memset((void*)out_buf_legacy, 0, args->num_freq*2*sizeof(int));
        }

        unsigned char * data = (unsigned char *) args->buf->data[bufferID];

        for (int integration = 0;
             integration < args->num_timesamples/integration_time;
             ++integration) {

            // Get the FPGA count
            // TODO fix this somehow!
            //*((uint32_t *) out_buf) = *(uint32_t *)&data[integration*packet_len*integration_time/args->num_frames + 54];
            //*((uint32_t *) out_buf + 1) = *(uint32_t *)&data[integration*packet_len*integration_time/args->num_frames + 46];

            fast_square_and_sum_vdif(integration_time,
                                    &data[integration * integration_time * PACKET_LEN * NUM_POL],
                                    temp_buf, xx, yy);

            fseek(fd, sizeof(int) * (line_size * line_idx  + head_size), SEEK_SET);
            ssize_t ints_written = fwrite(out_buf, sizeof(int), line_size, fd);

            if (ints_written != line_size) {
                ERROR("Failed to write power data to ram disk!!!");
                fclose(fd);
            }

            // Update the file header
            fseek(fd, 0, SEEK_SET);
            fwrite((void*)&line_idx, sizeof(int), 1, fd);
            fwrite((void*)&num_rolls, sizeof(int), 1, fd);
            fwrite((void*)&integration_time, sizeof(int), 1, fd);
            line_idx += 1;
            if (line_idx == num_entries) {
                line_idx = 0;
                num_rolls += 1;
            }

            if (args->legacy_output == 1) {
                for (int i = 0; i < args->num_freq; ++i) {
                    xx_legacy[i] += xx[i];
                    yy_legacy[i] += yy[i];
                }
            }
        }

        if (args->legacy_output == 1) {
            ssize_t ints_written = fwrite((void*)out_buf_legacy, sizeof(int), args->num_freq*2, fd_legacy);
            if (ints_written != args->num_freq*2) {
                ERROR("Failed to write power data to legacy ram disk!!!");
                fclose(fd);
            }
        }

        mark_buffer_empty(args->buf, bufferID);

        useableBufferIDs[0] = ( useableBufferIDs[0] + 1 ) % ( args->buf->num_buffers );
    }

}
