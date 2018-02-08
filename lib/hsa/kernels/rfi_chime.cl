/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor
Date: January 2018
File Purpose: OpenCL Kernel to compute Real-Time Spectral Kurtosis of Pathfinder Data
Details:
        Sums power and square power across time
        Normalizes by the mean power
        Sums square power across elements
        Compute Spectral Kurtosis
**********************************************************************************/

__kernel void
rfi_chime(
     __global char *input,
     __global float *output,
     __global float *swap,
     __constant uchar *InputMask,
     const uint num_bad_inputs,
     const uint sk_step
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
        
    //Declare Local Memory
    __local float sq_power_across_input[256];
    uint4 power_across_time = (uint4)(0);
    uint4 sq_power_across_time = (uint4)(0);
    uint precalc_index  = gx + gy*gx_size + sk_step*gz*(gx_size*gy_size);
    uint precalc_stepsize = gx_size*gy_size;

    //Sum Across Time
    for(int i =0; i < sk_step; i+=4){ 
        uint base_location = precalc_index + i*precalc_stepsize;
        char4 data_point = (char4) (input[base_location],
                                    input[base_location + 1*precalc_stepsize],
                                    input[base_location + 2*precalc_stepsize],
                                    input[base_location + 3*precalc_stepsize]);
        char4 real = (char4) (((data_point.x >> 4) & 0xF) - 8,
                              ((data_point.y >> 4) & 0xF) - 8,
                              ((data_point.z >> 4) & 0xF) - 8,
                              ((data_point.w >> 4) & 0xF) - 8);
        char4 imag = (char4) ((data_point.x & 0xF) - 8,
                              (data_point.y & 0xF) - 8,
                              (data_point.z & 0xF) - 8,
                              (data_point.w & 0xF) - 8);
        uint4 power = convert_uint4(real*real + imag*imag); //Compute power
        power_across_time += power;
        sq_power_across_time += power*power;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    uint power_sum = power_across_time.x + power_across_time.y + power_across_time.z + power_across_time.w;
    uint sq_power_sum = sq_power_across_time.x + sq_power_across_time.y + sq_power_across_time.z + sq_power_across_time.w;
	
    //printf("S1: %d, S2: %d, Step: %d\n",power_sum,sq_power_sum,sk_step);
    float mean = (float)power_sum/sk_step + 0.00000001;
    sq_power_across_input[lx] = (1-InputMask[lx])*sq_power_sum/(mean*mean);

    //printf("Normalized S2 %f\n",sq_power_across_input[lx]);
    barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish
    for(int j = lx_size/2; j>0; j >>= 1){ //Reduce along Elements
        //printf("Current S2: %f\n", sq_power_across_input[0]);
        if(lx < j){
            sq_power_across_input[lx] += sq_power_across_input[lx + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //printf("Total S2: %f\n", sq_power_across_input[0]);	

    if(lx == 0){
        swap[(gx + gy*gx_size + gz*precalc_stepsize)/lx_size] = sq_power_across_input[0];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if(gx == 0){ //Calculate SK
        float S2  = 0.0;
        for(int i = 0; i < gx_size/lx_size; i++){
            S2 += swap[(i + gy*gx_size + gz*precalc_stepsize)/lx_size];
        }

        uint M = (gx_size-num_bad_inputs)*sk_step;
        float SK = (((float)M+1)/((float)M-1))*((S2/M) - 1);
        output[gy + gz*gy_size] = SK;
        //printf("GX: %d GY: %d GZ %d M %d S2: %f SK: %f\n",gx,gy,gz,M,S2,SK);
    }
}
