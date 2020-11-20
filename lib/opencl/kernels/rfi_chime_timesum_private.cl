/***********************************************
Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: January 2018
File Purpose: OpenCL Kernel for kurtosis presum
Details:
        Sums power and square power across time
        Normalizes by the mean power
************************************************/
__kernel void
rfi_chime_time_sum(
     __global uint *input,
     __global float *output,
     __constant uchar *InputMask,
     const uint sk_step,
     const uint num_elements
)
{
    //Get work id's
    short gx = get_global_id(0);
    short gy = get_global_id(1);
    short gz = get_global_id(2);
    short gx_size = get_global_size(0); //num_elements/4
    short gy_size = get_global_size(1); //num_local_freq
    short gz_size = get_global_size(2); //samples_per_dataset/sk_step
    //Declare Local Memory
    uint4 power_across_time;
    uint4 sq_power_across_time;
    //Initialize local memory
    power_across_time = (uint4)(0u,0u,0u,0u);
    sq_power_across_time = (uint4)(0u,0u,0u,0u);
    //Compute current address in data
    uint address = gx + gy*gx_size + gz*gx_size*gy_size*sk_step;
    //Compute current element
    uint current_element = (4*gx + gy*4*gx_size)%num_elements;
    uint data; 
    uint4 temp;
    uint4 power;
    //Sum across time
    for(int i = 0; i < sk_step; i++){
        //Read input data
        data = input[address + i*gx_size*gy_size];
        //Compute power
        temp.s0 = ((data & 0x000000f0) << 12u) | ((data & 0x0000000f) >>   0u);
        temp.s1 = ((data & 0x0000f000) <<  4u) | ((data & 0x00000f00) >>   8u);
        temp.s2 = ((data & 0x00f00000) >>  4u) | ((data & 0x000f0000) >>  16u);
        temp.s3 = ((data & 0xf0000000) >> 12u) | ((data & 0x0f000000) >>  24u);
        power = ((temp>>16) - 8)*((temp>>16) - 8) + ((temp&0x0000ffff)- 8)*((temp&0x0000ffff) - 8);
        //Integrate
        power_across_time += power;
        sq_power_across_time += power*power;
    }
    //Compute mean of power sum and normalize square power sum
    float4 mean = convert_float4(power_across_time)/sk_step + (float4)0.00000001;
    float4 tmp = convert_float4(sq_power_across_time)/(mean*mean);
    //Compute address in output data and add sum to output array
    address = 4*gx + gy*4*gx_size + gz*4*gx_size*gy_size;
    output[0 + address] = (1-InputMask[0 + current_element])*tmp.s0;
    output[1 + address] = (1-InputMask[1 + current_element])*tmp.s1;
    output[2 + address] = (1-InputMask[2 + current_element])*tmp.s2;
    output[3 + address] = (1-InputMask[3 + current_element])*tmp.s3;
}
