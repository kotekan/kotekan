
#include "output_formating.h"

#include <stdlib.h> // for free, malloc

void reorganize_32_to_16_feed_GPU_Correlated_Data(int actual_num_frequencies,
                                                  int actual_num_elements, int* correlated_data) {
    // data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements
    // exist. This is equivalent to processing 2 elements at the same time, where the desired
    // correlations live in the first and fourth quadrants This function is to reorganize the data
    // so that comparisons can be done more easily

    // The input dataset is larger than the output, so can reorganize in the same array

    int input_frequencies = actual_num_frequencies / 2;
    int input_elements = actual_num_elements * 2;
    int address = 0;
    int address_out = 0;
    for (int freq = 0; freq < input_frequencies; freq++) {
        for (int element_y = 0; element_y < input_elements; element_y++) {
            for (int element_x = 0; element_x < input_elements; element_x++) {
                if (element_x < actual_num_elements && element_y < actual_num_elements) {
                    correlated_data[address_out++] = correlated_data[address++];
                    correlated_data[address_out++] =
                        correlated_data[address++]; // real and imaginary at each spot
                } else if (element_x >= actual_num_elements && element_y >= actual_num_elements) {
                    correlated_data[address_out++] = correlated_data[address++];
                    correlated_data[address_out++] = correlated_data[address++];
                } else
                    address += 2;
            }
        }
    }
    return;
}

void reorganize_32_to_16_feed_GPU_Correlated_Data_Interleaved(int actual_num_frequencies,
                                                              int actual_num_elements,
                                                              int* correlated_data) {
    // data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements
    // exist. There are two frequencies interleaved...  Need to sort the output values properly This
    // function is to reorganize the data so that comparisons can be done more easily

    // The input dataset is larger than the output, so can reorganize in the same array
    int* temp_output = (int*)malloc(actual_num_elements * actual_num_elements
                                    * actual_num_frequencies * 2 * sizeof(int));

    int input_elements = actual_num_elements * 2;
    int address = 0;
    int address_out = 0;
    for (int freq = 0; freq < actual_num_frequencies; freq++) {
        address = (freq >> 1) * input_elements * input_elements * 2;
        for (int element_y = 0; element_y < input_elements; element_y++) {
            for (int element_x = 0; element_x < input_elements; element_x++) {
                if (freq & 1) { // odd frequencies
                    if ((element_x & 1) && (element_y & 1)) {
                        temp_output[address_out++] = correlated_data[address++];
                        temp_output[address_out++] = correlated_data[address++];
                    } else {
                        address += 2;
                    }
                } else { // even frequencies
                    if ((!(element_x & 1)) && (!(element_y & 1))) {
                        temp_output[address_out++] = correlated_data[address++];
                        temp_output[address_out++] = correlated_data[address++];
                    } else {
                        address += 2;
                    }
                }
            }
        }
    }

    // copy the results back into correlated_data
    for (int i = 0; i < actual_num_frequencies * actual_num_elements * actual_num_elements * 2; i++)
        correlated_data[i] = temp_output[i];

    free(temp_output);
    return;
}

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion(
    int num_frequencies_final, int num_frequencies, int* input_data, complex_int_t* output_data) {
    // input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of
    // complex visibilities for frequencies output array will be sparsely to moderately filled, so
    // loop such that writing is done in sequential order int num_complex_visibilities =
    // 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_counter = 0;
    for (int y = 0; y < 16; y++) {
        for (int x = y; x < 16; x++) {
            for (int freq_count = 0; freq_count < num_frequencies_final; freq_count++) {
                if (freq_count < num_frequencies) {
                    int input_index = (freq_count * 256 + y * 16 + x) * 2;
                    output_data[output_counter].real = input_data[input_index];
                    output_data[output_counter++].imag = input_data[input_index + 1];
                } else {
                    output_data[output_counter].real = 0;
                    output_data[output_counter++].imag = 0;
                }
            }
        }
    }
    return;
}

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8_pairs(
    int num_frequencies, int* input_data, complex_int_t* output_data, int link_num) {
    // input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of
    // complex visibilities for frequencies output array will be sparsely to moderately filled, so
    // loop such that writing is done in sequential order int num_complex_visibilities =
    // 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_counter = link_num * 2;
    for (int y = 0; y < 16; y++) {
        for (int x = y; x < 16; x++) {
            for (int freq_count = 0; freq_count < num_frequencies / 2; freq_count++) {
                int input_index = (freq_count * 2 * 256 + y * 16 + x) * 2;
                output_data[output_counter].real = input_data[input_index];
                output_data[output_counter].imag = input_data[input_index + 1];
                output_data[output_counter + 1].real = input_data[input_index + 256 * 2];
                output_data[output_counter + 1].imag = input_data[input_index + 256 * 2 + 1];
            }
            output_counter += 16;
        }
    }
    return;
}

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8(
    int num_frequencies, int* input_data, complex_int_t* output_data, int link_num) {
    // input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of
    // complex visibilities for frequencies output array will be sparsely to moderately filled, so
    // loop such that writing is done in sequential order int num_complex_visibilities =
    // 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_counter = link_num;
    for (int y = 0; y < 16; y++) {
        for (int x = y; x < 16; x++) {
            for (int freq_count = 0; freq_count < num_frequencies; freq_count++) {

                int input_index = (freq_count * 256 + y * 16 + x) * 2;
                output_data[output_counter].real = input_data[input_index];
                output_data[output_counter].imag = input_data[input_index + 1];
                output_counter += 8;
            }
        }
    }
    return;
}

void full_16_element_matrix_to_upper_triangle_skip_8(int num_frequencies, int* input_data,
                                                     complex_int_t* output_data, int link_num) {
    // input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of
    // complex visibilities for frequencies output array will be sparsely to moderately filled, so
    // loop such that writing is done in sequential order int num_complex_visibilities =
    // 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_address = link_num * 136;

    for (int freq_count = 0; freq_count < num_frequencies; ++freq_count) {
        for (int y = 0; y < 16; ++y) {
            for (int x = y; x < 16; ++x) {
                int input_index = (freq_count * 256 + y * 16 + x) * 2;
                output_data[output_address].real = input_data[input_index];
                output_data[output_address].imag = input_data[input_index + 1];
                output_address++;
            }
        }
        output_address += 136 * 8;
    }
}

void full_16_element_matrix_to_upper_triangle(int num_frequencies, int* input_data,
                                              complex_int_t* output_data) {
    // input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of
    // complex visibilities for frequencies output array will be sparsely to moderately filled, so
    // loop such that writing is done in sequential order int num_complex_visibilities =
    // 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_address = 0;
    for (int freq_count = 0; freq_count < num_frequencies; ++freq_count) {
        for (int y = 0; y < 16; ++y) {
            for (int x = y; x < 16; ++x) {
                int input_index = (freq_count * 256 + y * 16 + x) * 2;
                output_data[output_address].real = input_data[input_index];
                output_data[output_address].imag = input_data[input_index + 1];
                output_address++;
            }
        }
    }
}

void reorganize_32_to_16_element_GPU_correlated_data_with_shuffle(int actual_num_frequencies,
                                                                  int actual_num_elements,
                                                                  int num_data_sets,
                                                                  int* correlated_data, int* map) {
    // data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements
    // exist. This is equivalent to processing 2 elements at the same time, where the desired
    // correlations live in the first and fourth quadrants This function is to reorganize the data
    // so that comparisons can be done more easily

    // The input dataset is larger than the output, so can reorganize in the same array

    int input_frequencies = actual_num_frequencies / 2;
    int input_elements = actual_num_elements * 2;
    int address = 0;
    int address_out = 0;
    int temp_address;
    int temp_array[1024]; // 16 elements x 16 elements x 2 complex vals x 2 sets at once
    for (int m = 0; m < num_data_sets; m++) {
        for (int freq = 0; freq < input_frequencies; freq++) {
            for (int element_y = 0; element_y < input_elements; element_y++) {
                for (int element_x = 0; element_x < input_elements; element_x++) {
                    if (element_x < actual_num_elements && element_y < actual_num_elements) {
                        temp_address = (map[element_y] * 16 + map[element_x]) * 2;
                        temp_array[temp_address++] = correlated_data[address++];
                        temp_array[temp_address] =
                            correlated_data[address++]; // real and imaginary at each spot
                    } else if (element_x >= actual_num_elements
                               && element_y >= actual_num_elements) {
                        temp_address = (map[element_y - 16] * 16 + map[element_x - 16]) * 2 + 512;
                        temp_array[temp_address++] = correlated_data[address++];
                        temp_array[temp_address] = correlated_data[address++];
                    } else
                        address += 2;
                }
            }
            for (int i = 0; i < 1024; i++) {
                correlated_data[address_out++] = temp_array[i];
            }
        }
    }
    return;
}

void reorganize_32_to_16_element_UT_GPU_correlated_data_with_shuffle(int actual_num_frequencies,
                                                                     int actual_num_elements,
                                                                     int num_data_sets,
                                                                     int* correlated_data,
                                                                     int* map) {
    // data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements
    // exist. This is equivalent to processing 2 elements at the same time, where the desired
    // correlations live in the first and fourth quadrants This function is to reorganize the data
    // so that comparisons can be done more easily

    // The input dataset is larger than the output, so can reorganize in the same array

    int input_frequencies = actual_num_frequencies / 2;
    int input_elements = actual_num_elements * 2;
    int address = 0;
    int address_out = 0;
    int temp_address;
    int temp_array[544]; //(16 x 17 / 2 ) elements x 2 complex vals x 2 sets at once
    int mapped_x, mapped_y;
    for (int m = 0; m < num_data_sets; m++) {
        for (int freq = 0; freq < input_frequencies; freq++) {
            for (int element_y = 0; element_y < input_elements; element_y++) {
                for (int element_x = 0; element_x < input_elements; element_x++) {
                    if (element_x < actual_num_elements && element_y < actual_num_elements) {
                        mapped_x = map[element_x];
                        mapped_y = map[element_y];
                        if (mapped_x >= mapped_y) {
                            temp_address =
                                (mapped_y * 16 + mapped_x - (mapped_y * (mapped_y + 1) / 2)) * 2;
                            temp_array[temp_address++] = correlated_data[address++];
                            temp_array[temp_address] =
                                correlated_data[address++]; // real and imaginary at each spot
                        } else
                            address += 2;
                    } else if (element_x >= actual_num_elements
                               && element_y >= actual_num_elements) {
                        mapped_x = map[element_x - 16];
                        mapped_y = map[element_y - 16];
                        if (mapped_x >= mapped_y) {
                            temp_address =
                                (mapped_y * 16 + mapped_x - (mapped_y * (mapped_y + 1) / 2)) * 2
                                + 272; // 16*17/2*2: offset to next set of data
                            temp_array[temp_address++] = correlated_data[address++];
                            temp_array[temp_address] = correlated_data[address++];
                        } else
                            address += 2;
                    } else
                        address += 2;
                }
            }
            for (int i = 0; i < 544; i++) {
                correlated_data[address_out++] = temp_array[i];
            }
        }
    }
    return;
}

void reorganize_GPU_to_upper_triangle(int block_side_length, int num_blocks,
                                      int actual_num_frequencies, int actual_num_elements,
                                      int num_data_sets, int* gpu_data,
                                      complex_int_t* final_matrix) {

    for (int m = 0; m < num_data_sets; m++) {
        // We go through the gpu data sequentially and
        // map it to the proper locations in the output array
        int GPU_address =
            m
            * (actual_num_frequencies * (num_blocks * (block_side_length * block_side_length * 2)));

        for (int frequency_bin = 0; frequency_bin < actual_num_frequencies; frequency_bin++) {
            int block_x_ID = 0;
            int block_y_ID = 0;
            int num_blocks_x = actual_num_elements / block_side_length;
            int block_check = num_blocks_x;
            int frequency_offset =
                m * actual_num_frequencies * (actual_num_elements * (actual_num_elements + 1)) / 2
                + frequency_bin * (actual_num_elements * (actual_num_elements + 1)) / 2;
            // frequency_bin * number of items in an upper triangle

            for (int block_ID = 0; block_ID < num_blocks; block_ID++) {
                if (block_ID == block_check) { // at the end of a row in the upper triangle
                    num_blocks_x--;
                    block_check += num_blocks_x;
                    block_y_ID++;
                    block_x_ID = block_y_ID;
                }

                for (int y_ID_local = 0; y_ID_local < block_side_length; y_ID_local++) {

                    for (int x_ID_local = 0; x_ID_local < block_side_length; x_ID_local++) {

                        int x_ID_global = block_x_ID * block_side_length + x_ID_local;
                        int y_ID_global = block_y_ID * block_side_length + y_ID_local;

                        // address_1d_output = frequency_offset, plus the number of entries
                        // in the rectangle area (y_ID_global*actual_num_elements),
                        // minus the number of elements in lower triangle to that row:
                        // (((y_ID_global-1)*y_ID_global)/2),
                        // plus the contributions to the address from the current row:
                        // (x_ID_global - y_ID_global)
                        int address_1d_output = frequency_offset + y_ID_global * actual_num_elements
                                                - ((y_ID_global - 1) * y_ID_global) / 2
                                                + (x_ID_global - y_ID_global);

                        if (block_x_ID != block_y_ID) { // when we are not in the diagonal blocks
                            final_matrix[address_1d_output].real = gpu_data[GPU_address++];
                            final_matrix[address_1d_output].imag = gpu_data[GPU_address++];
                        } else { // the special case needed to deal with the diagonal pieces
                            if (x_ID_local >= y_ID_local) {
                                final_matrix[address_1d_output].real = gpu_data[GPU_address++];
                                final_matrix[address_1d_output].imag = gpu_data[GPU_address++];
                            } else {
                                GPU_address += 2;
                            }
                        }
                    }
                }
                // offset_GPU += (block_side_length*block_side_length);
                // update block offset values
                block_x_ID++;
            }
        }
    }
}


void reorganize_GPU_to_upper_triangle_remap(int block_side_length, int num_blocks,
                                            int actual_num_frequencies, int actual_num_elements,
                                            int num_data_sets, int* gpu_data,
                                            complex_int_t* final_matrix, int* map) {
    for (int m = 0; m < num_data_sets; m++) {
        // we go through the gpu data sequentially and
        // map it to the proper locations in the output array
        int GPU_address =
            m
            * (actual_num_frequencies * (num_blocks * (block_side_length * block_side_length * 2)));
        for (int frequency_bin = 0; frequency_bin < actual_num_frequencies; frequency_bin++) {
            int block_x_ID = 0;
            int block_y_ID = 0;
            int num_blocks_x = actual_num_elements / block_side_length;
            int block_check = num_blocks_x;
            // frequency_bin * number of items in an upper triangle
            int frequency_offset =
                m * actual_num_frequencies * (actual_num_elements * (actual_num_elements + 1)) / 2
                + frequency_bin * (actual_num_elements * (actual_num_elements + 1)) / 2;

            for (int block_ID = 0; block_ID < num_blocks; block_ID++) {
                if (block_ID == block_check) { // at the end of a row in the upper triangle
                    num_blocks_x--;
                    block_check += num_blocks_x;
                    block_y_ID++;
                    block_x_ID = block_y_ID;
                }

                for (int y_ID_local = 0; y_ID_local < block_side_length; y_ID_local++) {

                    for (int x_ID_local = 0; x_ID_local < block_side_length; x_ID_local++) {

                        int x_ID_global = block_x_ID * block_side_length + x_ID_local;
                        int y_ID_global = block_y_ID * block_side_length + y_ID_local;
                        // would reduce to the unmapped case if map[x_ID_global] = x_ID_global
                        int mapped_x = map[x_ID_global];
                        int mapped_y = map[y_ID_global];
                        int imag_multiplier = 1;
                        if (mapped_x < mapped_y) { // i.e. lower triangle position after remap
                            int temp_mapped_x = mapped_x;
                            mapped_x = mapped_y;
                            mapped_y = temp_mapped_x;

                            imag_multiplier = -1;
                        }

                        /// address_1d_output = frequency_offset,
                        /// plus the number of entries in the rectangle area
                        /// (y_ID_global*actual_num_elements), minus the number of elements in
                        /// lower triangle to that row (((y_ID_global-1)*y_ID_global)/2),
                        /// plus the contributions to the address from the current row
                        /// (x_ID_global - y_ID_global)
                        int address_1d_output = frequency_offset + mapped_y * actual_num_elements
                                                - ((mapped_y - 1) * mapped_y) / 2
                                                + (mapped_x - mapped_y);

                        if (block_x_ID != block_y_ID) { // when we are not in the diagonal blocks
                            final_matrix[address_1d_output].real = gpu_data[GPU_address++];
                            final_matrix[address_1d_output].imag =
                                imag_multiplier * gpu_data[GPU_address++];
                        } else { // the special case needed to deal with the diagonal pieces
                            if (x_ID_local >= y_ID_local) {
                                final_matrix[address_1d_output].real = gpu_data[GPU_address++];
                                final_matrix[address_1d_output].imag =
                                    imag_multiplier * gpu_data[GPU_address++];
                            } else {
                                GPU_address += 2;
                            }
                        }
                    }
                }
                // update block offset values
                block_x_ID++;
            }
        }
    }
    return;
}
