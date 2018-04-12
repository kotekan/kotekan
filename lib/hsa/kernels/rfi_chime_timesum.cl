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
     __global uint *input,
     __global float *output,
     __constant uchar *InputMask,
     const uint sk_step
)
{
    short gx = get_global_id(0); //Get Work Id's
    short gy = get_global_id(1);
    short gz = get_global_id(2);
    short ly = get_local_id(1);
    short gx_size = get_global_size(0); //num_elements*num_freq/4
    short gy_size = get_global_size(1); //256
    short gz_size = get_global_size(2); //32768/sk_step
    short ly_size = get_local_size(1);

    //Declare Local Memory
    __local uint4 power_across_time[256];
    __local uint4 sq_power_across_time[256];
         
    power_across_time[ly] = (uint4)0;
    sq_power_across_time[ly] = (uint4)0;

    uint data; 
    uint base_index = gx + gy*gx_size + gz*sk_step*gx_size;
    
    uint4 temp;
    uint4 power;

    for(int i = 0; i < sk_step/ly_size; i++){
	
        data = input[base_index + i*gx_size*ly_size];

        temp.s0 = ((data & 0x000000f0) << 12u) | ((data & 0x0000000f) >>   0u);
        temp.s1 = ((data & 0x0000f000) <<  4u) | ((data & 0x00000f00) >>   8u);
        temp.s2 = ((data & 0x00f00000) >>  4u) | ((data & 0x000f0000) >>  16u);
        temp.s3 = ((data & 0xf0000000) >> 12u) | ((data & 0x0f000000) >>  24u);

        power = ((temp>>16) - 8)*((temp>>16) - 8) + ((temp&0x0000ffff)- 8)*((temp&0x0000ffff) - 8);
        power_across_time[ly] += power;
        sq_power_across_time[ly] += power*power;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for(int j = ly_size/2; j>0; j >>= 1){
        if(ly < j){
            power_across_time[ly] += power_across_time[ly + j];
            sq_power_across_time[ly] += sq_power_across_time[ly + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(ly == 0){

        float4 mean = convert_float4(power_across_time[0])/sk_step + (float4)0.00000001;
        float4 tmp = convert_float4(sq_power_across_time[0])/(mean*mean);

        output[4*gx + 0 + gz*4*gx_size] = (1-InputMask[4*gx + 0])*tmp.s0;
        output[4*gx + 1 + gz*4*gx_size] = (1-InputMask[4*gx + 1])*tmp.s3;
        output[4*gx + 2 + gz*4*gx_size] = (1-InputMask[4*gx + 2])*tmp.s2;
        output[4*gx + 3 + gz*4*gx_size] = (1-InputMask[4*gx + 3])*tmp.s1;
    }
}
