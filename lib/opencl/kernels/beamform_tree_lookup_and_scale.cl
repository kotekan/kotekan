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
                              __constant float         *complex_multipliers,
                              __constant unsigned int  *element_mask,
                                const    float          scale_factor)
{
    __local float lds_data[NUM_ELEMENTS/4*2];//=128//(NUM_ELEMENTS/4)*2 (Re+Im)
    //Also note that the arrangement of elements means that each work item will only deal with a single polarization in the first stage

    float R[4], I[4];
    //float R_0, R_1, R_2, R_3, I_0, I_1, I_2, I_3;//try this?
    float outR, outI;

    //determine the polarization and assign an offset for usage later: groups of 16 elements per slot noting we load 4 values per load.
    //Even numbered slots correspond to one polarization
    //Odd values correspond to the other (the way the cylinders and FPGAs map things out)
    int polarized = (ELEMENT_ID_DIV_4>>2)&0x1; //check if (id/16) even or odd?
    //int polarization_offset = polarized*(NUM_POL_ELEMENTS/4*2);//check odd/even for element_id/16.  Want to offset by 0 or 64 (half the size of lds_data)
    //calculate the phase correction

    float8 complex_val = vload8(FREQUENCY_BAND*NUM_ELEMENTS/4+ELEMENT_ID_DIV_4, complex_multipliers);
    // data is ordered F0: [Re0 Im0 Re1 Im1 ... ReN ImN] F1: [Re0 Im0 Re1 Im1 ... ReN ImN] ... F7: [Re0 Im0 Re1 Im1 ... ReN ImN]

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
        // Thus outR = R*phase_re + I*phase_im
        // and  outI = I*phase_re - R*phase_im
        // summing 4 products gives the following:
        outR = R[0]*complex_val.s0 + I[0]*complex_val.s1 +
               R[1]*complex_val.s2 + I[1]*complex_val.s3 +
               R[2]*complex_val.s4 + I[2]*complex_val.s5 +
               R[3]*complex_val.s6 + I[3]*complex_val.s7;
        outI = I[0]*complex_val.s0 - R[0]*complex_val.s1 +
               I[1]*complex_val.s2 - R[1]*complex_val.s3 +
               I[2]*complex_val.s4 - R[2]*complex_val.s5 +
               I[3]*complex_val.s6 - R[3]*complex_val.s7;

        barrier(CLK_LOCAL_MEM_FENCE);
        //reorder the data to group polarizations for the reduction

        int address = get_local_id(0) + (polarized*60) - ((get_local_id(0)>>3)*4); //ELEMENT_ID_DIV_4 + polarized*(NUM_POL_ELEMENTS/4*2-4) - (ELEMENT_ID_DIV_4>>3)*4
//         if (t == 0 && FREQUENCY_BAND==0)
//             printf("%2i: %3i %3i\n", get_local_id(0), address, address + NUM_POL_ELEMENTS/4);
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

//         if (get_local_id(0) == 0){
//             for (int i = 1; i < 32; i++){
//                 lds_data[0] += lds_data[i];
//                 lds_data[32] += lds_data[i+32];
//                 lds_data[64] += lds_data[i+64];
//                 lds_data[96] += lds_data[i+96];
//             }
//         }
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
            ////////////////////////////// Bit shift method is not the ideal way--leads to other problems

// //            printf("%3i: %3i: lds_data[0]: %10.6f, %3i\n",t+TIME_OFFSET_DIV_TIME_SCL*TIME_SCL,FREQUENCY_BAND,lds_data[0],((((int)lds_data[ 0])>>bit_shift_factor)&0x0f));
//             unsigned char temp1 = (((((int)lds_data[ 0])>>bit_shift_factor)&0x0f)<<4) |
//                   (((((int)lds_data[ 1])>>bit_shift_factor)&0x0f)>>0);
//             unsigned char temp2 = (((((int)lds_data[64])>>bit_shift_factor)&0x0f)<<4) |
//                  (((((int)lds_data[65])>>bit_shift_factor)&0x0f)>>0);
//             output[2*(FREQUENCY_BAND + (TIME_OFFSET_DIV_TIME_SCL*TIME_SCL+t)*NUM_FREQUENCIES)] = temp1^(0x88);//255;
// //                   (((((int)lds_data[ 0])>>bit_shift_factor)&0x0f)<<4) |
// //                   (((((int)lds_data[ 1])>>bit_shift_factor)&0x0f)>>0);
//
//             output[2*(FREQUENCY_BAND + (TIME_OFFSET_DIV_TIME_SCL*TIME_SCL+t)*NUM_FREQUENCIES)+1] =temp2^(0x88);//51;
// //                  (((((int)lds_data[64])>>bit_shift_factor)&0x0f)<<4) |
// //                  (((((int)lds_data[65])>>bit_shift_factor)&0x0f)>>0);
//
            ////////////////////////////// Scale and rail method
            lds_data[0]  *= scale_factor;
            lds_data[1]  *= scale_factor;
            lds_data[64] *= scale_factor;
            lds_data[65] *= scale_factor;
            //convert to integer
            int tempInt0 = (int)round(lds_data[0]);

//             if (tempInt0>7)
//                 tempInt0 = 7;
//             else if (tempInt0 <-7)
//                 tempInt0 = -7;

            tempInt0 = (tempInt0 >  7 ?  7 : tempInt0);
            tempInt0 = (tempInt0 < -7 ? -7 : tempInt0);

            int tempInt1 = (int)round(lds_data[1]);

//             if (tempInt1 >7)
//                 tempInt1 = 7;
//             else if (tempInt1 <-7)
//                 tempInt1 = -7;

            tempInt1 = (tempInt1 >  7 ?  7 : tempInt1);
            tempInt1 = (tempInt1 < -7 ? -7 : tempInt1);

            int tempInt64 = (int)round(lds_data[64]);

//             if (tempInt64>7)
//                 tempInt64 = 7;
//             else if (tempInt64 <-7)
//                 tempInt64 = -7;

            tempInt64 = (tempInt64 >  7 ?  7 : tempInt64);
            tempInt64 = (tempInt64 < -7 ? -7 : tempInt64);

            int tempInt65 = (int)round(lds_data[65]);

//             if (tempInt65 >7)
//                 tempInt65 = 7;
//             else if (tempInt65 <-7)
//                 tempInt65 = -7;

            tempInt65 = (tempInt65 >  7 ?  7 : tempInt65);
            tempInt65 = (tempInt65 < -7 ? -7 : tempInt65);

            unsigned char temp1 = (((tempInt0 )&0x0f)<<4) | (((tempInt1 )&0x0f)>>0);
            unsigned char temp2 = (((tempInt64)&0x0f)<<4) | (((tempInt65)&0x0f)>>0);

            //switch from two's complement encoding to offset encoding (i.e. swap the sign bit from 1 to 0 or vice versa)
            output[2*(FREQUENCY_BAND + (TIME_OFFSET_DIV_TIME_SCL*TIME_SCL+t)*NUM_FREQUENCIES)] = temp1^(0x88);
            output[2*(FREQUENCY_BAND + (TIME_OFFSET_DIV_TIME_SCL*TIME_SCL+t)*NUM_FREQUENCIES)+1] =temp2^(0x88);

        }
    }

    return;
}
