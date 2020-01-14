#ifndef TEST_DATA_GENERATION
#define TEST_DATA_GENERATION

#include "fpga_header_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

void generate_full_range_data_set(int offset, int num_time_steps, int num_freq, int num_elem,
                                  unsigned char* out_data);

void generate_const_data_set(unsigned char real, unsigned char imag, int num_time_steps,
                             int num_freq, int num_elem, unsigned char* out_data);

void generate_complex_sine_data_set(stream_id_t stream_id, int num_time_steps, int num_freq,
                                    int num_elem, unsigned char* out_data);

#ifdef __cplusplus
}
#endif

#endif