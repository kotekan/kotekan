/*********************************************************************************
Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: January 2018
File Purpose: OpenCL Kernel for kurtosis presum
Details:
        Sums power and square power across time
        Normalizes by the mean power
**********************************************************************************/

__kernel void
rfi_chime_timesum(
     __global char *input,
     __global float *output,
     __constant uchar *InputMask,
     const uint num_bad_inputs,
     const uint sk_step
)
{
    short gx = get_global_id(0); //Get Work Id's
    short gy = get_global_id(1);
    short gz = get_global_id(2);
    short ly = get_local_id(1);
    short gx_size = get_global_size(0); //#2048
    short gy_size = get_global_size(1); //#256
    short gz_size = get_global_size(2); //#128
    short ly_size = get_local_size(1);

    //Declare Local Memory
    __local uint power_across_time[256];
    __local uint sq_power_across_time[256];
    
    //Compute Power and Sq Power
    uchar data_val = input[gx + gy*gx_size + gz*gx_size*gy_size];
    char real = ((data_val >> 4) & 0xF) - 8;
    char imag = (data_val & 0xF) - 8;
    uchar power = real*real + imag*imag;
    power_across_time[ly] = power;
    sq_power_across_time[ly] = power*power;    

    //Sum Across Time 
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int j = ly_size/2; j>0; j >>= 1){
        if(ly < j){
            power_across_time[ly] += power_across_time[ly + j];
            sq_power_across_time[ly] += sq_power_across_time[ly + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(ly == 0){
        float mean = (float)power_across_time[0]/ly_size + 0.00000001;
        output[gx + gz*gx_size] = (1-InputMask[gx])*sq_power_across_time[0]/(mean*mean);
        //printf("%d Reg Power: %d Sq Power %d Normed Sq %f\n",gx + gz*gx_size, power_across_time[0], sq_power_across_time[0], output[gx + gz*gx_size]);
    }
}
