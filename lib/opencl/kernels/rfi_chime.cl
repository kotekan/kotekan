/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: August 2017
File Purpose: OpenCL Kernel to flag RFI in CHIME data
Details:
        Sums power and power**2 across time
        Normalizes
        Sums power and power**2 across elements
        Compute Spectral Kurtosis
        Flags RFI
**********************************************************************************/

__kernel void
rfi_chime(
          __global char* input, //Input buffer
          __global uint *count, //Amount of each frequency zeroed
          __global float* in_means, //Normilization means
          float sqrtM, //Square root of M
          int sensitivity, //# of deviations for Kurtosis threshold
          int time_samples, //Time samples in data block
          int zero //Whether or no to zero data
)
{
        short gx = get_global_id(0); //Get Work Id's
        short gy = get_global_id(1);
        short gz = get_global_id(2);
        short lx = get_local_id(0);
        short gx_size = get_global_size(0); //#Elements
        short gy_size = get_global_size(1); //#Frequencies
        short gz_size = get_global_size(2); //#Time Sample/SK_STEP
        short lx_size = get_local_size(0);
        uint SK_STEP = time_samples/gz_size;

        __local float power_across_input[256]; //Local Memory
        __local float sq_power_across_input[256];
        __local bool Zero_Flag;

        uint power_across_time = (uint)(0);
        uint sq_power_across_time = (uint)(0);
        uint precalc_index  = gx + gy*gx_size + SK_STEP*gz*(gx_size*gy_size);

        for(int i =0; i < SK_STEP; i+=4){ //Sum across time
                uint base_location = precalc_index + i*(gx_size*gy_size);
                char4 data_point = (char4) (input[base_location],
                                        input[base_location + 1*(gx_size*gy_size)],
                                        input[base_location + 2*(gx_size*gy_size)],
                                        input[base_location + 3*(gx_size*gy_size)]);
                char4 real = (char4) (((data_point.x >> 4) & 0xF) - 8,
                                        ((data_point.y >> 4) & 0xF) - 8,
                                        ((data_point.z >> 4) & 0xF) - 8,
                                        ((data_point.w >> 4) & 0xF) - 8);
                char4 imag = (char4) ((data_point.x & 0xF) - 8,
                                        (data_point.y & 0xF) - 8,
                                        (data_point.z & 0xF) - 8,
                                        (data_point.w & 0xF) - 8);
                uint4 power = convert_uint4(real*real + imag*imag); //Compute power
                uint4 sq_power = power*power;
                power_across_time += power.x + power.y + power.z + power.w;
                sq_power_across_time += sq_power.x + sq_power.y + sq_power.z + sq_power.w;
        }

        uint collapse_power = power_across_time;
        uint collapse_sq_power = sq_power_across_time;

        if(gz == 0) in_means[gx + gy*gx_size] += 0.1*((float)collapse_power/SK_STEP - in_means[gx + gy*gx_size]);

        barrier(CLK_GLOBAL_MEM_FENCE);
        
        float Median = in_means[gx + gy*gx_size]*(512.0/729.0); //Adjust Mean to Median

        power_across_input[lx] = (collapse_power/Median);//Save Values
        sq_power_across_input[lx] = (collapse_sq_power/(Median*Median));

        barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish

        for(short i = lx_size/2; i>0; i >>= 1){ //Reduce along Elements
                if(lx < i){
                        power_across_input[lx] += power_across_input[lx + i];
                        sq_power_across_input[lx] += sq_power_across_input[lx + i];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(lx == 0){ //Calculate SK and Flag
                uint M = (gx_size-0)*SK_STEP;
                float SK = (((float)M+1)/((float)M-1))*((((float)M*sq_power_across_input[0])/((float)power_across_input[0]*power_across_input[0]))-1);
                Zero_Flag = (SK < (1 - (sensitivity*2.0)/sqrtM) || SK > (1 + (sensitivity*2.0)/sqrtM));
                count[gy+gy_size*gz] = SK_STEP*Zero_Flag;
        }

        /*barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish
        for(short j = 0; j < SK_STEP; j++){ //Zero flagged values
                uint data_location = gx + gy*gx_size + (SK_STEP*gz + j)*(gx_size*gy_size);
                if(Zero_Flag && zero == 1) input[data_location] = 0x88; //Zero
        }*/
}

