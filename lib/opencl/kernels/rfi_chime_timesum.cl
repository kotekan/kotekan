/*********************************************************************************
Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: April 2018
File Purpose: OpenCL Kernel for kurtosis presum across time
Details:
        Sums power and square power across time
        Normalizes square power sum by the mean power
**********************************************************************************/
__kernel void
rfi_chime_time_sum(
     __global uint *input,
     __global float *output,
     __constant uchar *InputMask,
     const uint sk_step,
     const uint num_elements
)
{
    short gx = get_global_id(0); //Get Work Id's
    short gy = get_global_id(1);
    short gz = get_global_id(2);
    short ly = get_local_id(1);
    short gx_size = get_global_size(0); //num_elements*num_freq/4
    short gy_size = get_global_size(1); //256
    short gz_size = get_global_size(2); //timesteps/sk_step
    short ly_size = get_local_size(1);
    //Declare Local Memory
    __local uint4 power_across_time[256];
    __local uint4 sq_power_across_time[256];
    //Initialize local memory
    power_across_time[ly] = (uint4)0;
    sq_power_across_time[ly] = (uint4)0;
    //Compute current index in input
    uint base_index = gx + gy*gx_size + gz*sk_step*gx_size;
    uint data;
    uint4 temp;
    uint4 power;
    //Loop if integration size is larger than 256
    for(int i = 0; i < sk_step/ly_size; i++){
        //Retrieve Input (4 uint8's as 1 float)
        data = input[base_index + i*gx_size*ly_size];
        //Decompose input (float) into 4 values (uint8)
        temp.s0 = ((data & 0x000000f0) << 12u) | ((data & 0x0000000f) >>   0u);
        temp.s1 = ((data & 0x0000f000) <<  4u) | ((data & 0x00000f00) >>   8u);
        temp.s2 = ((data & 0x00f00000) >>  4u) | ((data & 0x000f0000) >>  16u);
        temp.s3 = ((data & 0xf0000000) >> 12u) | ((data & 0x0f000000) >>  24u);
        //Compute Power and add to local array
        power = ((temp>>16) - 8)*((temp>>16) - 8) + ((temp&0x0000ffff)- 8)*((temp&0x0000ffff) - 8);
        power_across_time[ly] += power;
        sq_power_across_time[ly] += power*power;
    }
    //Parallel sum of power and square power in local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int j = ly_size/2; j>0; j >>= 1){
        if(ly < j){
            power_across_time[ly] += power_across_time[ly + j];
            sq_power_across_time[ly] += sq_power_across_time[ly + j];
        }
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    //Normalize square power sum and place in output array
    if(ly == 0){
        //Compute mean of power sum
        float4 mean = convert_float4(power_across_time[0])/sk_step + (float4)0.00000001;
        //Normalize square power sum by the mean
        float4 tmp = convert_float4(sq_power_across_time[0])/(mean*mean);
        //Output
        output[4*gx + 0 + gz*4*gx_size] = (1-InputMask[4*gx%num_elements + 0])*tmp.s0;
        output[4*gx + 1 + gz*4*gx_size] = (1-InputMask[4*gx%num_elements + 1])*tmp.s1;
        output[4*gx + 2 + gz*4*gx_size] = (1-InputMask[4*gx%num_elements + 2])*tmp.s2;
        output[4*gx + 3 + gz*4*gx_size] = (1-InputMask[4*gx%num_elements + 3])*tmp.s3;
    }
}
