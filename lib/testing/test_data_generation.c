
#include "test_data_generation.h"

#include "fpga_header_functions.h" // for bin_number, bin_number_16_elem, stream_id_t

// TODO replace these functions with C++11 style lambda functions.

void generate_full_range_data_set(int offset, int num_time_steps, int num_freq, int num_elem,
                                  unsigned char* out_data) {
    unsigned char real = 0;
    unsigned char imag = offset;
    int idx = 0;

    for (int time_step = 0; time_step < num_time_steps; ++time_step) {
        for (int freq = 0; freq < num_freq; ++freq) {
            for (int elem = 0; elem < num_elem; ++elem) {
                idx = time_step * num_elem * num_freq + freq * num_elem + elem;
                out_data[idx] = ((real << 4) & 0xF0) + (imag & 0x0F);

                // Note this is the same as doing [0,255] in the char
                // but this is more clear as to what the components are doing.
                if (imag == 15) {
                    real = (real + 1) % 16;
                }
                imag = (imag + 1) % 16;
            }
        }
    }
}

void generate_const_data_set(unsigned char real, unsigned char imag, int num_time_steps,
                             int num_freq, int num_elem, unsigned char* out_data) {
    int idx = 0;

    for (int time_step = 0; time_step < num_time_steps; ++time_step) {
        for (int freq = 0; freq < num_freq; ++freq) {
            for (int elem = 0; elem < num_elem; ++elem) {
                idx = time_step * num_elem * num_freq + freq * num_elem + elem;
                out_data[idx] = ((real << 4) & 0xF0) + (imag & 0x0F);
            }
        }
    }
}

void generate_complex_sine_data_set(stream_id_t stream_id, int num_time_steps, int num_freq,
                                    int num_elem, unsigned char* out_data) {
    int idx = 0;
    int imag = 0;
    int real = 0;

    for (int time_step = 0; time_step < num_time_steps; ++time_step) {
        for (int freq = 0; freq < num_freq; ++freq) {
            for (int elem = 0; elem < num_elem; ++elem) {
                idx = time_step * num_elem * num_freq + freq * num_elem + elem;
                if (num_elem == 16) {
                    imag = bin_number_16_elem(&stream_id, freq) % 16;
                } else {
                    imag = bin_number(&stream_id, freq) % 16;
                }
                real = 9;
                out_data[idx] = ((real << 4) & 0xF0) + (imag & 0x0F);
            }
        }
    }
}
