//A beamforming routine currently capable of combining 256 elements
#pragma OPENCL EXTENSION cl_amd_printf : enable
#define TIME_SCL                    32
#define NUM_FREQUENCIES             8
#define NUM_ELEMENTS                256
#define NUM_POL_ELEMENTS            NUM_ELEMENTS/2
#define TIME_OFFSET_DIV_TIME_SCL    get_global_id(2)
#define FREQUENCY_BAND              get_global_id(1)
#define ELEMENT_ID_DIV_4            get_local_id(0) //this kernel works for 256 elements, so global and local are equivalent here

__kernel void gpu_beamforming(__global   unsigned int  *data,
                              __global   unsigned char *output,
                              __constant float         *freq_band_GHz,
                              __constant float         *phases,
                              __constant unsigned int  *element_mask,
                                const    float          scale_factor)
{
    __local float lds_data[NUM_ELEMENTS/4*2];//=128//(NUM_ELEMENTS/4)*2 (Re+Im)
    //Also note that the arrangement of elements means that each work item will only deal with a single polarization in the first stage

    //look up the frequency for this kernel, in GHz
    const float freq = freq_band_GHz[FREQUENCY_BAND];//this currently is a small array (for the subset of frequency bands handled by the kernel invocation)
    float R[4], I[4];
    //float R_0, R_1, R_2, R_3, I_0, I_1, I_2, I_3;//try this?
    float outR, outI;

    //determine the polarization and assign an offset for usage later: groups of 16 elements per slot noting we load 4 values per load.
    //Even numbered slots correspond to one polarization
    //Odd values correspond to the other (the way the cylinders and FPGAs map things out)
    int polarized = (ELEMENT_ID_DIV_4>>2)&0x1; //check if (id/16) even or odd?
    //int polarization_offset = polarized*(NUM_POL_ELEMENTS/4*2);//check odd/even for element_id/16.  Want to offset by 0 or 64 (half the size of lds_data)
    //calculate the phase correction
    float phase_re[4], phase_im[4]; //why arrays?
    int base_element_id = ELEMENT_ID_DIV_4*4;
    //phases is calculated as the true phase. we must apply -phases to remove the geometric delay.
    phase_im[0] = sincos(-phases[base_element_id+0]*freq, &phase_re[0]);
    phase_im[1] = sincos(-phases[base_element_id+1]*freq, &phase_re[1]);
    phase_im[2] = sincos(-phases[base_element_id+2]*freq, &phase_re[2]);
    phase_im[3] = sincos(-phases[base_element_id+3]*freq, &phase_re[3]);

    unsigned int temp = element_mask[ELEMENT_ID_DIV_4];
    int mask[4];
    mask[0] = (temp>> 0)&0x1;
    mask[1] = (temp>> 8)&0x1;
    mask[2] = (temp>>16)&0x1;
    mask[3] = (temp>>24)&0x1;
    for (uint t=0; t<TIME_SCL; t++)//launch less kernels by looping for multiple timesteps (32)--this way the phases calculated with sincos are reused
    {
        // read in the data
        uint input_data = data[get_local_id(0)
                               + (NUM_ELEMENTS/4) * (FREQUENCY_BAND + NUM_FREQUENCIES*(TIME_OFFSET_DIV_TIME_SCL * TIME_SCL + t)) ];



        I[0] = ((float)((input_data>> 0)&0xF) - 8)*mask[0];
        I[1] = ((float)((input_data>> 8)&0xF) - 8)*mask[1];
        I[2] = ((float)((input_data>>16)&0xF) - 8)*mask[2];
        I[3] = ((float)((input_data>>24)&0xF) - 8)*mask[3];

        R[0] = ((float)((input_data>> 4)&0xF) - 8)*mask[0];
        R[1] = ((float)((input_data>>12)&0xF) - 8)*mask[1];
        R[2] = ((float)((input_data>>20)&0xF) - 8)*mask[2];
        R[3] = ((float)((input_data>>28)&0xF) - 8)*mask[3];

        // Actually wanted the sin and cos of the negative angle.  Cos is symmetric, so doesn't change.  Sin needs a sign flip.
        // with numbers as (Re, Im),
        // (A,B) x (C, D) = AC + i^2 BD + i(BC+AD)
        //      (AC - BD, BC + AD)
        // A is R, B is I, C is phase_re, and D is actually -phase_im
        // Thus outR = R**2 + I**2
        // and  outI = 0

        outR = R[0]*R[0] + I[0]*I[0] +
               R[1]*R[1] + I[1]*I[1] +
               R[2]*R[2] + I[2]*I[2] +
               R[3]*R[3] + I[3]*I[3];

        barrier(CLK_LOCAL_MEM_FENCE);
        //reorder the data to group polarizations for the reduction

        int address = get_local_id(0) + (polarized*60) - ((get_local_id(0)>>3)*4); //ELEMENT_ID_DIV_4 + polarized*(NUM_POL_ELEMENTS/4*2-4) - (ELEMENT_ID_DIV_4>>3)*4

        lds_data[address]                    = outR;
        lds_data[address+NUM_POL_ELEMENTS/4] = outI; //offset by 32


        //need to calculate reduction for 2 polarizations, 128 elements each, real and imaginary.
        //do not want to have blocks that need to be calculated separately
        //at worst, calculate an offset to split the job to use the work items efficiently
        //though since there are only a few possible stages, perhaps it doesn't matter if threads are under-utilized
        //easiest is to have a by-2 reduction.
        //the above section reduces it to 32 for each polarization to start
        //by-2: 32 16 8 4 2 done
        //by-4: 32 8 2 (finish outside the loop? forget the loop entirely and unroll manually?)

        //check answers first, then worry about speed by using reductions


        ////////////////////////////////
        // CHECKED THE REDUCTION SECTION: a slow serial summation gives the same output as the tree.  So not the tree.  Could be the output or the above section.
        ////////////////////////////////


        for (uint i = NUM_POL_ELEMENTS/4; i>1; i = i/2){
            barrier(CLK_LOCAL_MEM_FENCE);
            if (get_local_id(0) >= i/2)
                continue;

            //load and accumulate 2 items
            R[0]=lds_data[get_local_id(0)*2+0];
            R[1]=lds_data[get_local_id(0)*2+1];

            I[0]=lds_data[get_local_id(0)*2+i+0];
            I[1]=lds_data[get_local_id(0)*2+i+1];

            outR = R[0]+R[1];
            outI = I[0]+I[1];

            //output
            barrier(CLK_LOCAL_MEM_FENCE);//this barrier is not likely needed, but to be safe...
            lds_data[get_local_id(0)]=outR;//R[0];
            lds_data[get_local_id(0)+i/2]=outI;//I[0];

            //second polarization
            R[2]=lds_data[get_local_id(0)*2+0+64];
            R[3]=lds_data[get_local_id(0)*2+1+64];

            I[2]=lds_data[get_local_id(0)*2+i+0+64];
            I[3]=lds_data[get_local_id(0)*2+i+1+64];

            outR = R[2]+R[3];
            outI = I[2]+I[3];

            //output
            barrier(CLK_LOCAL_MEM_FENCE);//this barrier is not likely needed, but to be safe...
            lds_data[get_local_id(0)+64]=outR;//R[0];
            lds_data[get_local_id(0)+i/2+64]=outI;//I[0];
        }

        // write output to buffer as an int, shift 16 bits up (perhaps to save as a fixed pt floating point number? Max val possible is NUM_ELEMENTS/2 * 5.6*2
        //NUM_ELEMENTS_PER_POLARIZATION = NUM_ELEMENTS/2
        //max_expected = NUM_ELEMENTS_PER_POLARIZATION* 8*(2^(-1/2)) *2
        //=NUM_ELEMENTS/2* 8*(2^(-1/2)) *2
        //=NUM_ELEMENTS/2 * 2^(7/2)
        // scale beamformed result by NUM_ELEMENTS/2 as a first guess, though signals from a single source, compared to the sky, should not be the WHOLE signal,
        // so integer divisions could go to 0 (int divide truncates the result (floor))
        if (get_local_id(0) == 0) {

            ////////////////////////////// Scale and rail method
            lds_data[0]  *= scale_factor;
            lds_data[64] *= scale_factor;

            //convert to integer
            unsigned int tempInt0 = (unsigned int)round(lds_data[0]);
            
            // Adding one to all values so that 0x00 is reserved 
            tempInt0 += 1
            tempInt0 = (tempInt0 >  255 ?  255 : tempInt0);

            unsigned int tempInt64 = (unsigned int)round(lds_data[64]);

            tempInt64 += 1
            tempInt64 = (tempInt64 >  255 ?  255 : tempInt64);

            unsigned char temp1 = tempInt0;
            unsigned char temp2 = tempInt64;

            //switch from two's complement encoding to offset encoding (i.e. swap the sign bit from 1 to 0 or vice versa)
            output[2*(FREQUENCY_BAND + (TIME_OFFSET_DIV_TIME_SCL*TIME_SCL+t)*NUM_FREQUENCIES)] = temp1;
            output[2*(FREQUENCY_BAND + (TIME_OFFSET_DIV_TIME_SCL*TIME_SCL+t)*NUM_FREQUENCIES)+1] = temp2;

        }
    }

    return;
}
