/***********************************************
Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: July 2018
File Purpose: OpenCL Kernel for kurtosis presum
Details:
        Sums power and square power across time
        Normalizes by the mean power
************************************************/
__kernel void
rfi_chime_timesum(
     __global uint *input,
     __global float *output,
     //__global float *output_var,
//     __constant uchar *InputMask,
//     __constant uchar *LostSamples,
//     __global uint *LostSamplesCorrection,
     const uint sk_step,
     const uint num_elements
     //const uint element_index
)
{
    //Get work id's
    short gx = get_global_id(0);
    short gy = get_global_id(1);
    short gx_size = get_global_size(0); //_samples_per_data_set/_sk_step
    short gy_size = get_global_size(1); //num_local_freq
    //Declare Local Memory
    uint4 power_across_time;
    uint4 sq_power_across_time;
    //Initialize local memory
    power_across_time = (uint4)(0u,0u,0u,0u);
    sq_power_across_time = (uint4)(0u,0u,0u,0u);
    //Compute current address in data
    uint address = gx + gy*gx_size*sk_step;
    //Compute current element
    uint current_element = (4*gx)%num_elements;
    uint data; 
    uint4 temp;
    uint4 power;
    //uint lost_sample_counter = 0;
    //Sum across time
    for(int i = 0; i < sk_step; i++){
        //Read input data
        data = input[address + i*gx_size];
        //Compute power
        temp.s0 = ((data & 0x000000f0) << 12u) | ((data & 0x0000000f) >>   0u);
        temp.s1 = ((data & 0x0000f000) <<  4u) | ((data & 0x00000f00) >>   8u);
        temp.s2 = ((data & 0x00f00000) >>  4u) | ((data & 0x000f0000) >>  16u);
        temp.s3 = ((data & 0xf0000000) >> 12u) | ((data & 0x0f000000) >>  24u);
        power = ((temp>>16) - 8)*((temp>>16) - 8) + ((temp&0x0000ffff)- 8)*((temp&0x0000ffff) - 8);
        //Integrate
        power_across_time += power;
        sq_power_across_time += power*power;
        //lost_sample_counter += LostSamples[i + gy*sk_step];
    }
    //LostSamplesCorrection[gy] = lost_sample_counter;
    //Compute mean of power sum and normalize square power sum
    float4 tmp;
//    if(lost_sample_counter < sk_step){
//        float4 mean = convert_float4(power_across_time)/(sk_step-lost_sample_counter) + (float4)0.00000001;
//        tmp = convert_float4(sq_power_across_time)/(mean*mean);
//    }
//    else{
//        tmp = (float4)(0.0);
//    }      
    float4 mean = convert_float4(power_across_time)/(sk_step) + (float4)0.00000001;
    tmp = convert_float4(sq_power_across_time)/(mean*mean);
    //Compute address in output data and add sum to output array
    address = 4*gx + gy*4*gx_size;
    output[0 + address] = tmp.s0;
    output[1 + address] = tmp.s1;
    output[2 + address] = tmp.s2;
    output[3 + address] = tmp.s3;

    //const uint vector_index = element_index % 4;

    //Output the variance for the selected feed
    //if(gx == element_index / 4) {
    //    switch(vector_index) {
    //        case 0:
    //            output_var[gy] = power_across_time.s0;
    //            break;
    //        case 1:
    //            output_var[gy] = power_across_time.s1;
    //            break;
    //        case 2:
    //            output_var[gy] = power_across_time.s2;
    //            break;
    //        case 3:
    //            output_var[gy] = power_across_time.s3;
    //            break;
    //    }
    //}

}
