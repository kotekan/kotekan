/*********************************************************************************
Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: January 2018
File Purpose: OpenCL Kernel for kurtosis calculation
Details:
        Sums square power across inputs
        Computes Kurtosis value
**********************************************************************************/
__kernel void
rfi_bad_input(
     __global float *input,
     __global float *output,
     const uint M,
     const uint num_sk
)
{
    //Get work ID's
    short gx = get_global_id(0); 
    short gy = get_global_id(1); 
    short gx_size = get_global_size(0); //num_elem
    short gy_size = get_global_size(1); //num_local_freq
    //Compute index in input array
    uint base_index = gx + gy*gx_size;
    //Compute and Sum single receiver SK estimates
    float total_sk = 0;
    for(int i = 0; i < num_sk; i++){
        total_sk += (((float)M+1)/((float)M-1))*((input[base_index + i*gx_size*gy_size]/M) - 1);
    }
    //Add to output array
    output[base_index] =  total_sk/num_sk;
}
