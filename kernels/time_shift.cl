// Timeshift

// need a NUM_BUFFERS for the ring buffer, "data"

__kernel void time_shift(__global unsigned char *data, // Input data starting at the front of the ring buffer
                         __global unsigned char *output_data, // Transfer to local buffer for next stage kernel
                                  int            num_elements_to_shift, // The number of element starting at element_offset to shift
                                  int            element_offset, // index of the first element to shift
                                  int            num_time_bins_to_shift, // e.g. -1 for 26 m lag cable
                                  int            input_data_offset){ // Buffer number

    //determine if the element represented by the work item id should be from the shift group or not.
    int element_should_shift_a = (get_global_id(0) >= element_offset) ? 1:0;
    int element_should_shift_b = (get_global_id(0) <  element_offset + num_elements_to_shift) ? 1:0;

    int input_address = input_data_offset * NUM_TIMESAMPLES * ACTUAL_NUM_FREQUENCIES * ACTUAL_NUM_ELEMENTS
                        + (get_global_id(2) + (element_should_shift_a*element_should_shift_b)*(-num_time_bins_to_shift)) * ACTUAL_NUM_FREQUENCIES * ACTUAL_NUM_ELEMENTS
                        + get_global_id(1) * ACTUAL_NUM_ELEMENTS
                        + (get_global_id(0));

    ///correct the input address, making sure it lies in array space
    if (input_address<0)
        input_address += NUM_BUFFERS*NUM_TIMESAMPLES*ACTUAL_NUM_FREQUENCIES*ACTUAL_NUM_ELEMENTS;
    if (input_address >= NUM_BUFFERS*NUM_TIMESAMPLES*ACTUAL_NUM_FREQUENCIES*ACTUAL_NUM_ELEMENTS)
        input_address -= NUM_BUFFERS*NUM_TIMESAMPLES*ACTUAL_NUM_FREQUENCIES*ACTUAL_NUM_ELEMENTS;

    //load data
    unsigned char temp = data[input_address];

    //write data out
    int output_address = get_global_id(2) * ACTUAL_NUM_ELEMENTS * ACTUAL_NUM_FREQUENCIES
                       + get_global_id(1) * ACTUAL_NUM_ELEMENTS
                       + get_global_id(0);
    output_data[output_address] = temp;
//
}
