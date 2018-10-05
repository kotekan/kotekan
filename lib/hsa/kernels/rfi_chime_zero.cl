/***********************************************
Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: September 2018
File Purpose: OpenCL Kernel for zeroing data with substantial RFI
Details:
    - Reads in input data and mask
    - Zeros neccesary timesteps
************************************************/
__kernel void
rfi_chime_zero(
     __global uint *input,
     __global uchar *mask,
     const uint sk_step
)
{
    //Get work id's
    short gx = get_global_id(0);
    short gy = get_global_id(1);
    short gx_size = get_global_size(0); //num_elements/4
    short gy_size = get_global_size(1); //_samples_per_data_set/_sk_step
    //Compute current address in data
    uint address_input = gx + gy*gx_size*sk_step;
    //If we need to zero some data
    if(mask[gy] == 1){
        //Zero all of the data for sk_step timesteps
        for(int i = 0; i < sk_step; i++){
            input[address_input + i*gx_size] = 0x88888888;
        }
    }
}
