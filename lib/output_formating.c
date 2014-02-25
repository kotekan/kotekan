
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>


#include "output_formating.h"

void reorganize_32_to_16_feed_GPU_Correlated_Data(int actual_num_frequencies, int actual_num_elements, int *correlated_data) {
    //data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements exist.
    //This is equivalent to processing 2 elements at the same time, where the desired correlations live in the first and fourth quadrants
    //This function is to reorganize the data so that comparisons can be done more easily

    //The input dataset is larger than the output, so can reorganize in the same array

    int input_frequencies = actual_num_frequencies/2;
    int input_elements = actual_num_elements*2;
    int address = 0;
    int address_out = 0;
    for (int freq = 0; freq < input_frequencies; freq++){
        for (int element_y = 0; element_y < input_elements; element_y++){
            for (int element_x = 0; element_x < input_elements; element_x++){
                if (element_x < actual_num_elements && element_y < actual_num_elements){
                    correlated_data[address_out++] = correlated_data[address++];
                    correlated_data[address_out++] = correlated_data[address++]; //real and imaginary at each spot
                }
                else if (element_x >=actual_num_elements && element_y >=actual_num_elements){
                    correlated_data[address_out++] = correlated_data[address++];
                    correlated_data[address_out++] = correlated_data[address++];
                }
                else
                    address += 2;
            }
        }
    }
    return;
}

void reorganize_32_to_16_feed_GPU_Correlated_Data_Interleaved(int actual_num_frequencies,
                                                              int actual_num_elements,
                                                              int *correlated_data)
{
    //data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements exist.
    //There are two frequencies interleaved...  Need to sort the output values properly
    //This function is to reorganize the data so that comparisons can be done more easily

    //The input dataset is larger than the output, so can reorganize in the same array
    int *temp_output = (int *)malloc(actual_num_elements*actual_num_elements*actual_num_frequencies*2*sizeof(int));

    int input_elements = actual_num_elements*2;
    int address = 0;
    int address_out = 0;
    for (int freq = 0; freq < actual_num_frequencies; freq++){
        address = (freq >>1) * input_elements*input_elements *2;
        for (int element_y = 0; element_y < input_elements; element_y++){
            for (int element_x = 0; element_x < input_elements; element_x++){
                if (freq & 1){//odd frequencies
                    if ((element_x & 1) && (element_y & 1)){
                        temp_output[address_out++] = correlated_data[address++];
                        temp_output[address_out++] = correlated_data[address++];
                    }
                    else{
                        address += 2;
                    }
                }
                else{ // even frequencies
                    if ((!(element_x & 1)) && (!(element_y & 1))){
                        temp_output[address_out++] = correlated_data[address++];
                        temp_output[address_out++] = correlated_data[address++];
                    }
                    else{
                        address += 2;
                    }
                }
            }
        }
    }

    //copy the results back into correlated_data
    for (int i = 0; i < actual_num_frequencies * actual_num_elements*actual_num_elements * 2; i++)
        correlated_data[i] = temp_output[i];

    free(temp_output);
    return;
}

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion(
        int num_frequencies_final, int num_frequencies, int *input_data, complex_int_t *output_data)
{
    //input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of complex visibilities for frequencies
    //output array will be sparsely to moderately filled, so loop such that writing is done in sequential order
    //int num_complex_visibilities = 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_counter = 0;
    for (int y = 0; y < 16; y++){
        for (int x = y; x < 16; x++){
            for (int freq_count = 0; freq_count < num_frequencies_final; freq_count++){
                if (freq_count < num_frequencies){
                    int input_index = (freq_count * 256 + y*16 + x)*2;
                    output_data[output_counter].real = input_data[input_index];
                    output_data[output_counter++].imag = input_data[input_index+1];
                }
                else{
                    assert(output_counter < 139264); // 16*(16+1)/2 * 1024
                    output_data[output_counter].real = 0;
                    output_data[output_counter++].imag = 0;
                }
            }
        }
    }
    return;
}

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8_pairs(int num_frequencies_final, int num_frequencies, int *input_data, complex_int_t *output_data, int link_num)
{
    //input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of complex visibilities for frequencies
    //output array will be sparsely to moderately filled, so loop such that writing is done in sequential order
    //int num_complex_visibilities = 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_counter = link_num*2;
    for (int y = 0; y < 16; y++){
        for (int x = y; x < 16; x++){
            for (int freq_count = 0; freq_count < 64; freq_count++){
                int input_index = (freq_count * 2 * 256 + y*16 + x)*2;
                output_data[output_counter].real = input_data[input_index];
                output_data[output_counter].imag = input_data[input_index+1];
                output_data[output_counter+1].real = input_data[input_index+256*2];
                output_data[output_counter+1].imag = input_data[input_index+256*2+1];
                //output_counter += 16;
                assert(output_counter <= 139264);
            }
            output_counter += 16;
        }
    }
    return;
}

void shuffle_data_to_frequency_major_output_16_element_with_triangle_conversion_skip_8(int num_frequencies_final, int num_frequencies, int *input_data, complex_int_t *output_data, int link_num)
{
    //input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of complex visibilities for frequencies
    //output array will be sparsely to moderately filled, so loop such that writing is done in sequential order
    //int num_complex_visibilities = 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_counter = link_num;
    for (int y = 0; y < 16; y++){
        for (int x = y; x < 16; x++){
            for (int freq_count = 0; freq_count < 128; freq_count++){
                assert(output_counter <= 139264);
                
                int input_index = (freq_count * 256 + y*16 + x)*2;
                output_data[output_counter].real = input_data[input_index];
                output_data[output_counter].imag = input_data[input_index+1];
                output_counter += 8;
            }
            //output_counter += 8;
        }
    }
    return;
}
