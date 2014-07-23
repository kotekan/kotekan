// 64 element FFT routine--appears to work now: July 17, 2014.  Clean up!!!
//#pragma OPENCL EXTENSION cl_amd_printf : enable
#define PI  3.14159265358979323f
#define REAL    x
#define IMAG    y
//#define OFFSET_B    0

__kernel void FFT64(__global float2 *data, int sign, __global float2 *results_array){
    __local float2 local_data[128];//128 float2 * 2 float/float2 * 4 B/float = 1024 B
    //__local sinLookup[64];
    uint local_address = get_local_id(0);
    uint local_address_shift1;
    float signPI = PI * sign;
    uint offset;// = ((local_address & 0x20)>>5)* OFFSET_B; //have 64 work items, but algorithm uses 32 work items per 64 complex elements
                                                         //this calculates the offset to the second batch of data. If the data sets are contiguous
                                                         //then OFFSET_B = 0

    //load data together into local memory
    //barrier(CLK_LOCAL_MEM_FENCE);
    local_data[local_address]      = data[local_address + get_group_id(0)*128]; //first set of 64 numbers
    local_data[local_address + 64] = data[local_address + get_group_id(0)*128 + 64]; //second set

    //now make 'offset' point to the second set of data
    offset = ((local_address & 0x20)<<1); //64 if the latter half, 0 if not
    //printf("%d ", offset);

    //Bit reversal calculation
    //idea from http://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith32Bits
    local_address = (local_address & 0x1F); //mask so the address only ranges from 0-31
    local_address_shift1 = local_address<<1;
    uint index1 = local_address_shift1;
    uint index2 = index1 + 1;
    index1 = (( ( ((index1 * 0x0802) & 0x22110) | ((index1 * 0x8020)&0x88440) ) * 0x10101 ) >> 18) & 0x3F; //explanation at bottom!
    index1 += offset;
    index2 = (( ( ((index2 * 0x0802) & 0x22110) | ((index2 * 0x8020)&0x88440) ) * 0x10101 ) >> 18) & 0x3F; //explanation at bottom!
    index2 += offset;
    //if (get_group_id(0)==0)
    //    printf("%d \n", local_address);

    //want bit-reversed addresses
    //barrier(CLK_LOCAL_MEM_FENCE);
    //Danielson-Lanczos theorem section, which allows the O(n*log2(n)) calculation count
    //stage 0
    //uint index1 = (local_address << 1) + offset;
    //uint index2 = index1 + 1;
    float2 temp, temp2;
    temp  = local_data[index1]; //load according to bit reversed addresses
    temp2 = local_data[index2];
    //if (get_group_id(0)==0)
    //    printf("id %02d, index1: %02d, Re: %f Im %f ; index2: %02d, Re: %f Im %f\n",get_local_id(0),index1, local_data[index1].REAL, local_data[index1].IMAG, index2, local_data[index2].REAL, local_data[index2].IMAG);
    local_data[(local_address_shift1)+offset]   = temp+temp2; //store according to local addresses
    local_data[(local_address_shift1)+offset+1] = temp-temp2;
    
 
    //barrier(CLK_LOCAL_MEM_FENCE);
    //things are in lock-step so synch points at each stage aren't needed, right?
    //stage 1
    //((index << 1) & 0xfc) + (index & 0x1)
    index1 = (((local_address_shift1) & 0x3c) + (local_address & 0x1)) + offset; //0011 1100
    index2 = index1 + 2;
    temp2 = local_data[index2];
    float2 sincos_temp;
    float theta = (local_address & 0x1)*signPI*.5f;
    sincos_temp.IMAG = native_sin(theta);//((local_address & 0x1)*signPI*.5f);
    sincos_temp.REAL = native_cos(theta);//((local_address & 0x1)*signPI*.5f);
    //sincos_temp.IMAG = sin((local_address & 0x1)*sign*PI*.5f);
    //sincos_temp.REAL = cos((local_address & 0x1)*sign*PI*.5f);
    temp.REAL = sincos_temp.REAL * temp2.REAL - sincos_temp.IMAG * temp2.IMAG;
    temp.IMAG = sincos_temp.REAL * temp2.IMAG + sincos_temp.IMAG * temp2.REAL;
//    if (get_group_id(0)==0){
//        printf("id %02d, local_address: %d, index1: %d, index2: %d, cos: %f, sin: %f\n",get_local_id(0),local_address, index1,index2, sincos_temp.REAL, sincos_temp.IMAG);
//        printf("    local_data[%d].RE = %f, local_data[%d].IM = %f, local_data[%d].RE = %f, local_data[%d].IM = %f\n",index1,local_data[index1].REAL,index1, local_data[index1].IMAG,index2, local_data[index2].REAL,index2,local_data[index2].IMAG);
//        printf("    RE: %f, IM: %f, RE: %f, IM: %f\n", (local_data[index1]-temp).REAL, (local_data[index1]-temp).IMAG, (local_data[index1]+temp).REAL, (local_data[index1]+temp).IMAG);
//    }
    local_data[index2] = local_data[index1]-temp;
    local_data[index1] += temp;


    //barrier(CLK_LOCAL_MEM_FENCE);
    //stage 2
    //((index << 1) & 0xf8) + (index & 0x3)
    index1 = (((local_address_shift1) & 0x38) + (local_address & 0x3)) + offset;
    index2 = index1 + 4;
    temp2 = local_data[index2];
    theta = (local_address & 0x3)*signPI/4.f;
    sincos_temp.IMAG = native_sin(theta);//((local_address & 0x3)*signPI/4.f);
    sincos_temp.REAL = native_cos(theta);//((local_address & 0x3)*signPI/4.f);
    //sincos_temp.IMAG = sin((local_address & 0x3)*sign*PI/4.f);
    //sincos_temp.REAL = cos((local_address & 0x3)*sign*PI/4.f);
    temp.REAL = sincos_temp.REAL * temp2.REAL - sincos_temp.IMAG * temp2.IMAG;
    temp.IMAG = sincos_temp.REAL * temp2.IMAG + sincos_temp.IMAG * temp2.REAL;
    local_data[index2] = local_data[index1]-temp;
    local_data[index1] += temp;


    //barrier(CLK_LOCAL_MEM_FENCE);
    //stage 3
    //((index << 1) & 0xf0) + (index & 0x7)
    index1 = (((local_address_shift1) & 0x30) + (local_address & 0x7)) + offset;
    index2 = index1 + 8;
    temp2 = local_data[index2];
    theta = (local_address & 0x7)*signPI/8.f;
    sincos_temp.IMAG = native_sin(theta);//((local_address & 0x7)*signPI/8.f);
    sincos_temp.REAL = native_cos(theta);//((local_address & 0x7)*signPI/8.f);
    //sincos_temp.IMAG = sin((local_address & 0x7)*sign*PI/8.f);
    //sincos_temp.REAL = cos((local_address & 0x7)*sign*PI/8.f);
    temp.REAL = sincos_temp.REAL * temp2.REAL - sincos_temp.IMAG * temp2.IMAG;
    temp.IMAG = sincos_temp.REAL * temp2.IMAG + sincos_temp.IMAG * temp2.REAL;
    local_data[index2] = local_data[index1]-temp;
    local_data[index1] += temp;


    //barrier(CLK_LOCAL_MEM_FENCE);
    //stage 4
    //((index << 1) & 0xe0) + (index & 0xf)
    index1 = (((local_address_shift1) & 0x20) + (local_address & 0xf)) + offset;
    index2 = index1 + 16;
    temp2 = local_data[index2];
    theta = (local_address & 0xf)*signPI/16.f;
    sincos_temp.IMAG = native_sin(theta);//((local_address & 0xf)*signPI/16.f);
    sincos_temp.REAL = native_cos(theta);//((local_address & 0xf)*signPI/16.f);
    //sincos_temp.IMAG = sin((local_address & 0xf)*sign*PI/16.f);
    //sincos_temp.REAL = cos((local_address & 0xf)*sign*PI/16.f);
    temp.REAL = sincos_temp.REAL * temp2.REAL - sincos_temp.IMAG * temp2.IMAG;
    temp.IMAG = sincos_temp.REAL * temp2.IMAG + sincos_temp.IMAG * temp2.REAL;
    local_data[index2] = local_data[index1]-temp;
    local_data[index1] += temp;


    //barrier(CLK_LOCAL_MEM_FENCE);
    //stage 5
    //((index << 1) & 0xc0) + (index & 0x1f) //1100 0000
    //index1 = (((local_address <<1) & 0x00) + (local_address & 0x1f)) + offset;
    index1 = (local_address & 0x1f) + offset;
    index2 = index1 + 32;
    temp2 = local_data[index2];
    theta = (local_address & 0x1f)*signPI/32.f;
    sincos_temp.IMAG = native_sin(theta);//((local_address & 0x1f)*signPI/32.f);
    sincos_temp.REAL = native_cos(theta);//((local_address & 0x1f)*signPI/32.f);
    //sincos_temp.IMAG = sin((local_address & 0x1f)*sign*PI/32.f);
    //sincos_temp.REAL = cos((local_address & 0x1f)*sign*PI/32.f);
    temp.REAL = sincos_temp.REAL * temp2.REAL - sincos_temp.IMAG * temp2.IMAG;
    temp.IMAG = sincos_temp.REAL * temp2.IMAG + sincos_temp.IMAG * temp2.REAL;
    local_data[index2] = local_data[index1]-temp;
    local_data[index1] += temp;


    //barrier(CLK_LOCAL_MEM_FENCE);
    //write back to memory
    results_array[get_local_id(0)+get_group_id(0)*128] = local_data[get_local_id(0)];
    results_array[get_local_id(0) +get_group_id(0)*128  + 64] = local_data[get_local_id(0) + 64];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// bit reversal explanation:
//
//                                     abcd efgh         local_address
//                         * 0000 1000 0000 0010              * 0x0802   multiply to shift so that a, e, b, f go into correct relative positions
//                 -----------------------------         -------------
//                      0abc defg h00a bcde fgh0             result_1a
//                    & 0010 0010 0001 0001 0000             & 0x22110   mask to keep only those parts
//                 -----------------------------         -------------
//                      00b0 00f0 000a 000e 0000              result_1
//
//                                     abcd efgh         local_address
//                         * 1000 0000 0010 0000              * 0x8020   multiply to shift so that c, g, d, h go into correct relative positions
//                 -----------------------------         -------------
//                 0abc defg h00a bcde fgh0 0000             result_2a
//                    & 1000 1000 0100 0100 0000             & 0x88440   mask to keep only those parts
//                 -----------------------------         -------------
//                      d000 h000 0c00 0g00 0000              result_2
//
//                      d000 h000 0c00 0g00 0000              result_1
//                    | 00b0 00f0 000a 000e 0000            | result_2   combine results
//                 -----------------------------         -------------
//                      d0b0 h0f0 0c0a 0g0e 0000              result_3
//
//                      d0b0 h0f0 0c0a 0g0e 0000              result_3
//                    * 0001 0000 0001 0000 0001             * 0x10101   multiply to combine results into correct order
//  --------------------------------------------
//                      d0b0 h0f0 0c0a 0g0e 0000
//            d0b0 h0f0 0c0a 0g0e 0000 0000 0000
//  d0b0 h0f0 0c0a 0g0e 0000 0000 0000 0000 0000
//  --------------------------------------------         -------------
//       h0f0 dcba hgfe dcba hgfe 0c0a 0g0e 0000              result_4
//
//       h0f0 dcba hgfe dcba hgfe 0c0a 0g0e 0000 >> 19   result_4 >>19   shift result to correct final position
//  --------------------------------------------        --------------       (original version wanted to shift by 16 so abc would be kept)
//                              h 0f0d cbah gfed              result_5
//
//                              h 0f0d cbah gfed              result_5
//                                      & 1 1111                & 0x1F   mask to have correct final result
//                             -----------------             ---------
//                                        h gfed              result_6