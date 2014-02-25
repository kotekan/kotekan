
#include "chrx.h"

#ifndef OUTPUT_FORMATING
#define OUTPUT_FORMATING

void reorganize_32_to_16_feed_GPU_Correlated_Data_Interleaved(int actual_num_frequencies, 
                                                              int actual_num_elements, 
                                                              int *correlated_data);

void reorganize_32_to_16_feed_GPU_Correlated_Data(int actual_num_frequencies,
                                                  int actual_num_elements,
                                                  int *correlated_data);

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion(
        int num_frequencies_final, int num_frequencies,
        int *input_data, complex_int_t *output_data);


void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8_pairs(
    int num_frequencies_final, int num_frequencies,
    int *input_data, complex_int_t *output_data, int link_num);

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8(
    int num_frequencies_final, int num_frequencies,
    int *input_data, complex_int_t *output_data, int link_num);

#endif