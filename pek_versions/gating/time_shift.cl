//time shifting kernel
//P Klages 2014
//Global Work Size = (num_elements_to_shift,frequency_bands,timesteps)
//the idea is quite simple.  If at an element that needs to be shiftedData is copied from
//#pragma OPENCL EXTENSION cl_amd_printf : enable


#define NUM_BUFFERS 2
__kernel void time_shift(__global unsigned char *data,
                         __global unsigned char *output_data,
                                  int            num_elements_to_shift,
                                  int            element_offset, //start # of elements to shift
                                  int            num_time_bins_to_shift,
                                  int            input_data_offset){

    //__local unsigned char loaded_data [64];
    //determine if the element represented by the work item id should be from the shift group or not.
    int element_should_shift_a = (get_global_id(0) >= element_offset) ? 1:0;
    int element_should_shift_b = (get_global_id(0) <  element_offset + num_elements_to_shift) ? 1:0;
//printf(".\n");
    int input_address = input_data_offset * NUM_TIMESAMPLES * ACTUAL_NUM_FREQUENCIES * ACTUAL_NUM_ELEMENTS
                        + (get_global_id(2) + (element_should_shift_a*element_should_shift_b)*(-num_time_bins_to_shift)) * ACTUAL_NUM_FREQUENCIES * ACTUAL_NUM_ELEMENTS
                        + get_global_id(1) * ACTUAL_NUM_ELEMENTS
                        + (get_global_id(0));

    //compare performance to the following: I think this second one might be faster: same number of comparisons, and one assignment, no multiplies
//
    //int element_time_shift;
    //if (get_global_id (0) >= element_offset && get_global_id(0) < element_offset + num_elements_to_shift)
    //    element_time_shift = num_time_bins_to_shift;
    //else
    //    element_time_shift = 0;
    //int input_address = input_data_offset * NUM_TIMESAMPLES * NUM_FREQUENCIES * NUM_ELEMENTS
    //                    + (get_global_id(2) + element_time_shift) * NUM_FREQUENCIES * NUM_ELEMENTS
    //                    + (get_global_id(1) * NUM_ELEMENTS
    //                    + (get_global_id(0));
//

    ///correct the input address, making sure it lies in array space
    //if the % operator works as I expect it, then it is a one line check, since negative numbers would have array_size = 2*NUM_TIMESAMPLES*NUM_FREQUENCIES*NUM_ELEMENTS added to them, and numbers greater than array_size would have array_size subtracted
//    input_address = input_address % (NUM_BUFFERS*NUM_TIMESAMPLES*NUM_FREQUENCIES*NUM_ELEMENTS);

    //speed compare the simple alternates

    if (input_address<0)
        input_address += NUM_BUFFERS*NUM_TIMESAMPLES*ACTUAL_NUM_FREQUENCIES*ACTUAL_NUM_ELEMENTS;
    if (input_address >= NUM_BUFFERS*NUM_TIMESAMPLES*ACTUAL_NUM_FREQUENCIES*ACTUAL_NUM_ELEMENTS)
        input_address -= NUM_BUFFERS*NUM_TIMESAMPLES*ACTUAL_NUM_FREQUENCIES*ACTUAL_NUM_ELEMENTS;


    ///load data
//     unsigned int l_id = get_local_id(0)+get_local_id(1)*get_local_size(0)+get_local_id(2)*get_local_size(0)*get_local_size(1);
//     //printf("Local_id: %d \n",l_id);
//     loaded_data [l_id]= data[input_address];
//     barrier(CLK_LOCAL_MEM_FENCE);
//
//     ///write data out
//     int output_address = get_global_id(2)  * ACTUAL_NUM_ELEMENTS * ACTUAL_NUM_FREQUENCIES
//                         + get_global_id(1) * ACTUAL_NUM_ELEMENTS
//                         + get_global_id(0);
//     output_data[output_address] = loaded_data[l_id];

    //speed compare
//
    //load data
    unsigned char temp = data[input_address];

    //write data out
    int output_address = get_global_id(2)  * ACTUAL_NUM_ELEMENTS * ACTUAL_NUM_FREQUENCIES
                       + get_global_id(1) * ACTUAL_NUM_ELEMENTS
                       + get_global_id(0);
    output_data[output_address] = temp;
//
}
