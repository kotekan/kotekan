// LWS for this kernel will be {4*64,   1,    1}
// GWS will be ordered         {4*64, pol, time}
#define REAL    x
#define IMAG    y
#define PI2 -1.5707963267948966f
#define PI4 -0.7853981633974483f
#define PI8 -0.39269908169872414f
#define PI16 -0.19634954084936207f
#define PI32 -0.09817477042468103f
#define PI64 -0.04908738521234052f
#define PI128 -0.02454369260617026f
#define PI256 -0.01227184630308513f

#define CUSTOM_BIT_REVERSE_9_BITS(index) ((( ( (((index) * 0x0802) & 0x22110) | (((index) * 0x8020)&0x88440) ) * 0x10101 ) >> 15) & 0x1FE)

__kernel void zero_padded_FFT512( __global uint *data, __global uint *mapped,  __global float2 *Co, __global float *results_array,  __global float2 *Gain){

  __local float2 local_data[2048];//4* 512 float2 * 2 float/float2 * 4 B/float = 16kB
    uint local_address = get_local_id(0);  //0 to 255
    uint data_temp;
    float2 temp_0, temp_1, temp_2, temp_3;
    data_temp = data[get_global_id(2)*512 + get_global_id(1)*256 + local_address];

    //unpack to index, with gaps of 256
    uint index_0 = CUSTOM_BIT_REVERSE_9_BITS(local_address*4) + (((local_address&0x40)<<2)  + ((local_address&0x80)<<2) + ((local_address&0xc0)<<2));
    uint index_1 = CUSTOM_BIT_REVERSE_9_BITS(local_address*4+1) + (((local_address&0x40)<<2)  + ((local_address&0x80)<<2) + ((local_address&0xc0)<<2));
    uint index_2 = CUSTOM_BIT_REVERSE_9_BITS(local_address*4+2) + (((local_address&0x40)<<2)  + ((local_address&0x80)<<2) + ((local_address&0xc0)<<2));
    uint index_3 = CUSTOM_BIT_REVERSE_9_BITS(local_address*4+3) + (((local_address&0x40)<<2)  + ((local_address&0x80)<<2) + ((local_address&0xc0)<<2));

    temp_0.REAL = ((int)((data_temp & 0x000000f0) >>   4u)) - 8;  //240
    temp_0.IMAG = ((int)((data_temp & 0x0000000f) >>   0u)) - 8;   //15
    temp_1.REAL = ((int)((data_temp & 0x0000f000) >>  12u)) - 8;   //61440
    temp_1.IMAG = ((int)((data_temp & 0x00000f00) >>   8u)) - 8;   //3840
    temp_2.REAL = ((int)((data_temp & 0x00f00000) >>  20u)) - 8;  //
    temp_2.IMAG = ((int)((data_temp & 0x000f0000) >>  16u)) - 8;
    temp_3.REAL = ((int)((data_temp & 0xf0000000) >>  28u)) - 8;
    temp_3.IMAG = ((int)((data_temp & 0x0f000000) >>  24u)) - 8;

    uint gain_id = get_global_id(1)*1024+local_address*4;
    local_data[index_0    ].REAL = temp_0.REAL*Gain[gain_id].REAL + temp_0.IMAG*Gain[gain_id].IMAG;
    local_data[index_0    ].IMAG = -temp_0.REAL*Gain[gain_id].IMAG + temp_0.IMAG*Gain[gain_id].REAL;
    local_data[index_0 +1 ]      = local_data[index_0    ];

    gain_id = gain_id + 1;
    local_data[index_1    ].REAL = temp_1.REAL*Gain[gain_id].REAL + temp_1.IMAG*Gain[gain_id].IMAG;
    local_data[index_1    ].IMAG = -temp_1.REAL*Gain[gain_id].IMAG + temp_1.IMAG*Gain[gain_id].REAL;
    local_data[index_1 +1 ]      = local_data[index_1    ];

    gain_id = gain_id + 1;
    local_data[index_2    ].REAL = temp_2.REAL*Gain[gain_id].REAL + temp_2.IMAG*Gain[gain_id].IMAG;
    local_data[index_2    ].IMAG = -temp_2.REAL*Gain[gain_id].IMAG + temp_2.IMAG*Gain[gain_id].REAL;
    local_data[index_2 +1 ]      = local_data[index_2    ];

    gain_id = gain_id + 1;
    local_data[index_3    ].REAL = temp_3.REAL*Gain[gain_id].REAL + temp_3.IMAG*Gain[gain_id].IMAG;
    local_data[index_3    ].IMAG = -temp_3.REAL*Gain[gain_id].IMAG + temp_3.IMAG*Gain[gain_id].REAL;
    local_data[index_3 +1 ]      = local_data[index_3    ];

    barrier(CLK_LOCAL_MEM_FENCE);  //crucial

    //Beamform NS -----
    float theta;
    float2 sincos_temp, temp;

    //stage 1
    index_0 = ((local_address & 0xfe)<<1) | ((local_address&0x1)<<0);
    index_1 = index_0+2;
    theta = PI2*(local_address&0x1);
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    for (int i=0;i<4;i++){
      temp_0 = local_data[index_0 + 512*i];
      temp_1 = local_data[index_1 + 512*i];
      temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
      temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
      local_data[index_0 + i*512] = temp_0+temp;
      local_data[index_1 + i*512] = temp_0-temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 2
    index_0 = ((local_address & 0xfc) << 1) | ((local_address & 0x3)<<0);
    index_1 = index_0 + 4;
    theta = (local_address & 0x3)*PI4;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    for (int i=0;i<4;i++){
      temp_0 = local_data[index_0 + i*512];
      temp_1 = local_data[index_1 + i*512];
      temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
      temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
      local_data[index_0 + i*512] = temp_0+temp;
      local_data[index_1 + i*512] = temp_0-temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 3
    index_0 = ((local_address & 0xf8) << 1) | ((local_address & 0x7)<<0);
    index_1 = index_0 + 8;
    theta = (local_address & 0x7)*PI8;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    for (int i=0;i<4;i++){
      temp_0 = local_data[index_0 + i*512];
      temp_1 = local_data[index_1 + i*512];
      temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
      temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
      local_data[index_0 + i*512] = temp_0+temp;
      local_data[index_1 + i*512] = temp_0-temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 4
    index_0 = ((local_address & 0xf0) << 1) | ((local_address & 0xf)<<0);
    index_1 = index_0 + 16;
    theta = (local_address & 0xf)*PI16;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    for (int i=0;i<4;i++){
      temp_0 = local_data[index_0 + i*512];
      temp_1 = local_data[index_1 + i*512];
      temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
      temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
      local_data[index_0 + i*512] = temp_0+temp;
      local_data[index_1 + i*512] = temp_0-temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 5
    index_0 = ((local_address & 0xe0) << 1) | ((local_address & 0x1f)<<0);
    index_1 = index_0 + 32;
    theta = (local_address & 0x1f)*PI32;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    for (int i=0;i<4;i++){
      temp_0 = local_data[index_0 + i*512];
      temp_1 = local_data[index_1 + i*512];
      temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
      temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
      local_data[index_0 + i*512] = temp_0+temp;
      local_data[index_1 + i*512] = temp_0-temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 6
    index_0 = ((local_address & 0xc0) << 1) | ((local_address & 0x3f)<<0);
    index_1 = index_0 + 64;
    theta = (local_address & 0x3f)*PI64;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    for (int i=0;i<4;i++){
      temp_0 = local_data[index_0 + i*512];
      temp_1 = local_data[index_1 + i*512];
      temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
      temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
      local_data[index_0 + i*512] = temp_0+temp;
      local_data[index_1 + i*512] = temp_0-temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 7
    index_0 = ((local_address & 0x80) << 1) | ((local_address & 0x7f)<<0);
    index_1 = index_0 + 128;
    theta = (local_address & 0x7f)*PI128;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    for (int i=0;i<4;i++){
      temp_0 = local_data[index_0 + i*512];
      temp_1 = local_data[index_1 + i*512];
      temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
      temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
      local_data[index_0 + i*512] = temp_0+temp;
      local_data[index_1 + i*512] = temp_0-temp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 8
    index_0 = local_address;
    index_1 = index_0 + 256;
    theta = (index_0)*PI256;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    for (int i=0;i<4;i++){
      temp_0 = local_data[index_0 + i*512];
      temp_1 = local_data[index_1 + i*512];
      temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
      temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
      local_data[index_0 + i*512] = temp_0+temp;
      local_data[index_1 + i*512] = temp_0-temp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Clamping
    index_0= mapped[local_address];
    index_1 = index_0 + 512;
    index_2 = index_0 + 1024;
    index_3 = index_0 + 1536;

    //FFT BEAMFORM EAST-WEST:
    float2 CoArray[16];
    for (int i=0; i<16;i++){
      CoArray[i] = Co[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float2 Temp;
    //change to 255-localadd in order to flip cylinder NS
    uint address = get_global_id(2)*2048 + get_global_id(1)*1024 + (255-local_address);

    float out;
    temp_0.REAL = 4;
    temp_0.IMAG = 4;
    //Beam0
    Temp.REAL =\
      local_data[index_0].REAL * CoArray[0].REAL + local_data[index_0].IMAG * CoArray[0].IMAG + \
      local_data[index_1].REAL * CoArray[1].REAL + local_data[index_1].IMAG * CoArray[1].IMAG + \
      local_data[index_2].REAL * CoArray[2].REAL + local_data[index_2].IMAG * CoArray[2].IMAG + \
      local_data[index_3].REAL * CoArray[3].REAL + local_data[index_3].IMAG * CoArray[3].IMAG;
    Temp.IMAG =\
      local_data[index_0].REAL * CoArray[0].IMAG - local_data[index_0].IMAG * CoArray[0].REAL + \
      local_data[index_1].REAL * CoArray[1].IMAG - local_data[index_1].IMAG * CoArray[1].REAL + \
      local_data[index_2].REAL * CoArray[2].IMAG - local_data[index_2].IMAG * CoArray[2].REAL + \
      local_data[index_3].REAL * CoArray[3].IMAG - local_data[index_3].IMAG * CoArray[3].REAL ;
    Temp = Temp/temp_0;
    __asm__ ("V_CVT_PKRTZ_F16_F32 %0, %1, %2" : "=v"(out)
                                              : "v"(Temp.x),
                                                "v"(Temp.y));
    results_array[address] = out;
//    results_array[address] = Temp/temp_0;

    //Beam1
    Temp.REAL =\
      local_data[index_0].REAL * CoArray[4].REAL + local_data[index_0].IMAG * CoArray[4].IMAG + \
      local_data[index_1].REAL * CoArray[5].REAL + local_data[index_1].IMAG * CoArray[5].IMAG + \
      local_data[index_2].REAL * CoArray[6].REAL + local_data[index_2].IMAG * CoArray[6].IMAG + \
      local_data[index_3].REAL * CoArray[7].REAL + local_data[index_3].IMAG * CoArray[7].IMAG;
    Temp.IMAG =\
      local_data[index_0].REAL * CoArray[4].IMAG - local_data[index_0].IMAG * CoArray[4].REAL + \
      local_data[index_1].REAL * CoArray[5].IMAG - local_data[index_1].IMAG * CoArray[5].REAL + \
      local_data[index_2].REAL * CoArray[6].IMAG - local_data[index_2].IMAG * CoArray[6].REAL + \
      local_data[index_3].REAL * CoArray[7].IMAG - local_data[index_3].IMAG * CoArray[7].REAL ;
    Temp = Temp/temp_0;
    __asm__ ("V_CVT_PKRTZ_F16_F32 %0, %1, %2" : "=v"(out)
                                              : "v"(Temp.x),
                                                "v"(Temp.y));
    results_array[address + 256] = out;
//    results_array[address+ 256] = Temp/temp_0;

    //Beam2
  Temp.REAL = \
    local_data[index_0].REAL * CoArray[8].REAL + local_data[index_0].IMAG * CoArray[8].IMAG + \
    local_data[index_1].REAL * CoArray[9].REAL + local_data[index_1].IMAG * CoArray[9].IMAG + \
    local_data[index_2].REAL * CoArray[10].REAL + local_data[index_2].IMAG * CoArray[10].IMAG + \
    local_data[index_3].REAL * CoArray[11].REAL + local_data[index_3].IMAG * CoArray[11].IMAG;
  Temp.IMAG = \
    local_data[index_0].REAL * CoArray[8].IMAG - local_data[index_0].IMAG * CoArray[8].REAL + \
    local_data[index_1].REAL * CoArray[9].IMAG - local_data[index_1].IMAG * CoArray[9].REAL + \
    local_data[index_2].REAL * CoArray[10].IMAG - local_data[index_2].IMAG * CoArray[10].REAL +   \
    local_data[index_3].REAL * CoArray[11].IMAG - local_data[index_3].IMAG * CoArray[11].REAL ;
    Temp = Temp/temp_0;
    __asm__ ("V_CVT_PKRTZ_F16_F32 %0, %1, %2" : "=v"(out)
                                              : "v"(Temp.x),
                                                "v"(Temp.y));
    results_array[address + 2*256] = out;
//  results_array[address + 2*256] = Temp/temp_0;

  //Beam3
  Temp.REAL = \
    local_data[index_0].REAL * CoArray[12].REAL + local_data[index_0].IMAG * CoArray[12].IMAG + \
    local_data[index_1].REAL * CoArray[13].REAL + local_data[index_1].IMAG * CoArray[13].IMAG + \
    local_data[index_2].REAL * CoArray[14].REAL + local_data[index_2].IMAG * CoArray[14].IMAG + \
    local_data[index_3].REAL * CoArray[15].REAL + local_data[index_3].IMAG * CoArray[15].IMAG;
  Temp.IMAG = \
    local_data[index_0].REAL * CoArray[12].IMAG - local_data[index_0].IMAG * CoArray[12].REAL + \
    local_data[index_1].REAL * CoArray[13].IMAG - local_data[index_1].IMAG * CoArray[13].REAL + \
    local_data[index_2].REAL * CoArray[14].IMAG - local_data[index_2].IMAG * CoArray[14].REAL +   \
    local_data[index_3].REAL * CoArray[15].IMAG - local_data[index_3].IMAG * CoArray[15].REAL ;
    Temp = Temp/temp_0;
    __asm__ ("V_CVT_PKRTZ_F16_F32 %0, %1, %2" : "=v"(out)
                                              : "v"(Temp.x),
                                                "v"(Temp.y));
    results_array[address + 3*256] = out;
//  results_array[address+ 3*256] = Temp/temp_0;

  //exit

}

