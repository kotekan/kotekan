#ifndef OUTPUT_FORMATING
#define OUTPUT_FORMATING

#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

void reorganize_32_to_16_feed_GPU_Correlated_Data_Interleaved(int actual_num_frequencies,
                                                              int actual_num_elements,
                                                              int* correlated_data);

void reorganize_32_to_16_feed_GPU_Correlated_Data(int actual_num_frequencies,
                                                  int actual_num_elements, int* correlated_data);

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion(
    int num_frequencies_final, int num_frequencies, int* input_data, complex_int_t* output_data);


void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8_pairs(
    int num_frequencies, int* input_data, complex_int_t* output_data, int link_num);

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8(
    int num_frequencies, int* input_data, complex_int_t* output_data, int link_num);

void full_16_element_matrix_to_upper_triangle_skip_8(int num_frequencies, int* input_data,
                                                     complex_int_t* output_data, int link_num);

void full_16_element_matrix_to_upper_triangle(int num_frequencies, int* input_data,
                                              complex_int_t* output_data);

void reorganize_32_to_16_element_GPU_correlated_data_with_shuffle(int actual_num_frequencies,
                                                                  int actual_num_elements,
                                                                  int num_data_sets,
                                                                  int* correlated_data, int* map);

void reorganize_32_to_16_element_UT_GPU_correlated_data_with_shuffle(int actual_num_frequencies,
                                                                     int actual_num_elements,
                                                                     int num_data_sets,
                                                                     int* correlated_data,
                                                                     int* map);

void reorganize_GPU_to_upper_triangle(int block_side_length, int num_blocks,
                                      int actual_num_frequencies, int actual_num_elements,
                                      int num_data_sets, int* gpu_data,
                                      complex_int_t* final_matrix);

void reorganize_GPU_to_upper_triangle_remap(int block_side_length, int num_blocks,
                                            int actual_num_frequencies, int actual_num_elements,
                                            int num_data_sets, int* gpu_data,
                                            complex_int_t* final_matrix, int* map);

#ifdef __cplusplus
}
#endif

#endif