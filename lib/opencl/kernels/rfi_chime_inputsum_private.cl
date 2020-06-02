/*****************************************************
Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: January 2018
File Purpose: OpenCL Kernel for kurtosis calculation
Details:
        Sums square power across inputs
        Computes Kurtosis value
******************************************************/
__kernel void
rfi_chime_input_sum(
     __global float *input,
     __global float *output,
     const uint num_elements,
     const uint M
)
{
    //Get work ID's
    short gx = get_global_id(0);
    short gy = get_global_id(1);
    short gz = get_global_id(2);
    short gx_size = get_global_size(0); //#8
    short gy_size = get_global_size(1); //#32
    short gz_size = get_global_size(2); //#4
    //Declare Local Memory
    float sq_power_across_input = (float)0;
    //Partial Sum across inputs
    uint address = gx*num_elements + gy*gx_size*num_elements + gz*gy_size*gx_size*num_elements;
    for(int i = 0; i < num_elements; i++){
        sq_power_across_input += input[address + i];
    }
    //Compute address in output and add SK value to output
    address = gx + gy*gx_size + gz*gy_size*gx_size;
    output[address] = (((float)M+1)/((float)M-1))*((sq_power_across_input/M) - 1);
}
