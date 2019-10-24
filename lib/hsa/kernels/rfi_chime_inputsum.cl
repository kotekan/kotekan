/*********************************************************************************
Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: July 2018
File Purpose: OpenCL Kernel for kurtosis calculation
Details:
        Sums square power across inputs
        Computes Kurtosis value
**********************************************************************************/
__kernel void
rfi_chime_inputsum(
     __global float *input,
     __global float *output,
     __global uchar *InputMask,
     __global uchar *OutputMask,
//     __global uint *LostSampleCorrection,
     const uint num_elements,
     const uint num_bad_inputs,
     const uint sk_step,
     const uint rfi_sigma_cut
)
{
    //Get work ID's
    short gx = get_global_id(0);
    short gy = get_global_id(1);
    short gz = get_global_id(2);
    short lx = get_local_id(0);
    short gx_size = get_global_size(0); //#256
    short gy_size = get_global_size(1); //#num_freq
    short gz_size = get_global_size(2); //#timesteps/sk_step
    short lx_size = get_local_size(0);
    //Declare Local Memory
    __local float sq_power_across_input[256];
    //Compute index in input array
    uint base_index = gx + gy*num_elements + gz*num_elements*gy_size;
    sq_power_across_input[lx] = InputMask[lx]*input[base_index];
    //Partial sum if more than 256 inputs
    for(int i = 1; i < num_elements/lx_size; i++){
        sq_power_across_input[lx] += InputMask[lx + i*lx_size]*input[base_index + i*lx_size];
    }
    //Sum Across Input in local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int j = lx_size/2; j>0; j >>= 1){
        if(lx < j){
            sq_power_across_input[lx] += sq_power_across_input[lx + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //Compute spectral kurtosis estimate and add to output
    if(lx == 0){
        float n = (float)sk_step; // - LostSampleCorrection[gz];
        float N = (float)(num_elements-num_bad_inputs);
        uint address = gy + gz*gy_size;
        if(n*N == 0){
            output[address] = -1.0;
            OutputMask[address] = 1;
        } 
        else{
            float SK =  ((n+1)/(n-1))*((sq_power_across_input[0]/(n*N)) - 1);
            output[address] = SK;
            float sigma = sqrt((double)((4*n*n)/(N*(n-1)*(n+2)*(n+3))));
            if(SK > 1 + rfi_sigma_cut*sigma || SK < 1 - rfi_sigma_cut*sigma) OutputMask[address] = 1;
            else OutputMask[address] = 0;
        }
    }
}
