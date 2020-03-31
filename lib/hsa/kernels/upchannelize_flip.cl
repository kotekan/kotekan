#define PI2 1.5707963267948966f
#define PI4 0.7853981633974483f
#define PI8 0.39269908169872414f
#define PI16 0.19634954084936207f
#define PI32 0.09817477042468103f
#define PI64 0.04908738521234052f
#define REAL    x
#define IMAG    y
#define CHUNK_OFFSET 128
#define n_all 256
#define scaling 400.

__constant float BP[16] = { 0.52225748 , 0.58330915 , 0.6868705 , 0.80121821 , 0.89386546 , 0.95477358 , 0.98662733 , 0.99942558 , 0.99988676 , 0.98905127 , 0.95874124 , 0.90094667 , 0.81113021 , 0.6999944 , 0.59367968 , 0.52614263};

#define BIT_REVERSE_7_BITS(index) ((( ( (((index) * 0x0802) & 0x22110) | (((index) * 0x8020)&0x88440) ) * 0x10101 ) >> 17) & 0x7F)
//input data is float2 with beam-pol-time, try to do 3 N=128 at once so that we can sum 3 time samples
//LWS = {     64 ,  1  }
//GWS = {nsamp/6*, 1024}

__kernel void upchannelize(__global float2 *data, __global float *results_array){

  uint nbeam = get_global_size(1);
  uint nsamp = get_global_size(0)*6+32;
  uint nsamp_out = get_global_size(0)*6/128/3;
  __local float2 local_data[384];
  uint local_address = get_local_id(0);
  float outtmp;
  outtmp = 0;

  // Loop over 2 polarisations
  for (int p=0;p<2;p++){
    uint address_in = (p*nbeam*nsamp)+get_global_id(1)*nsamp+get_group_id(0)*384 + local_address;
    //read from global
    local_data[local_address      ] = data[address_in      ];
    local_data[local_address + 64 ] = data[address_in + 64 ];
    local_data[local_address + 128] = data[address_in + 128];
    local_data[local_address + 192] = data[address_in + 192];
    local_data[local_address + 256] = data[address_in + 256];
    local_data[local_address + 320] = data[address_in + 320];

    uint index_0 = local_address * 2;
    uint index_1 = index_0 + 1; //these two indices span 128 entries
    index_0 = BIT_REVERSE_7_BITS(index_0);
    index_1 = BIT_REVERSE_7_BITS(index_1);
    uint index_2 = index_0 + CHUNK_OFFSET; //repeat with CHUNK_OFFSET
    uint index_3 = index_1 + CHUNK_OFFSET;
    uint index_4 = index_2 + CHUNK_OFFSET; //repeat with CHUNK_OFFSET
    uint index_5 = index_3 + CHUNK_OFFSET;

    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 0
    float2 temp, temp_0, temp_1, temp_2, temp_3 ,temp_4, temp_5;
    temp_0 = local_data[index_0]; //load according to bit reversed addresses
    temp_1 = local_data[index_1];
    temp_2 = local_data[index_2]; //load according to bit reversed addresses
    temp_3 = local_data[index_3];
    temp_4 = local_data[index_4]; //load according to bit reversed addresses
    temp_5 = local_data[index_5];

    local_data[local_address * 2                     ] = temp_0+temp_1;
    local_data[local_address * 2 + 1                 ] = temp_0-temp_1;
    local_data[local_address * 2 + CHUNK_OFFSET      ] = temp_2+temp_3; 
    local_data[local_address * 2 + CHUNK_OFFSET + 1  ] = temp_2-temp_3;
    local_data[local_address * 2 + 2*CHUNK_OFFSET    ] = temp_4+temp_5; 
    local_data[local_address * 2 + 2*CHUNK_OFFSET + 1] = temp_4-temp_5;

    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 1
    float2 sincos_temp;
    float theta;
    index_0 = (((local_address<<1) & 0x7c) + (local_address & 0x1));
    index_1 = index_0+2;
    index_2 = index_0 + CHUNK_OFFSET;
    index_3 = index_1 + CHUNK_OFFSET;
    index_4 = index_2 + CHUNK_OFFSET;
    index_5 = index_3 + CHUNK_OFFSET;
    temp_0 = local_data[index_0];
    temp_1 = local_data[index_1];
    temp_2 = local_data[index_2];
    temp_3 = local_data[index_3];
    temp_4 = local_data[index_4];
    temp_5 = local_data[index_5];
    theta = PI2*(local_address&0x1);
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
    local_data[index_0] = temp_0 + temp;
    local_data[index_1] = temp_0 - temp;
    temp.REAL = sincos_temp.REAL * temp_3.REAL - sincos_temp.IMAG * temp_3.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_3.IMAG + sincos_temp.IMAG * temp_3.REAL;
    local_data[index_2] = temp_2+temp;
    local_data[index_3] = temp_2-temp;
    temp.REAL = sincos_temp.REAL * temp_5.REAL - sincos_temp.IMAG * temp_5.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_5.IMAG + sincos_temp.IMAG * temp_5.REAL;
    local_data[index_4] = temp_4+temp;
    local_data[index_5] = temp_4-temp;

    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 2
    index_0 = (((local_address<<1) & 0x78) + (local_address & 0x3));
    index_1 = index_0 + 4;
    index_2 = index_0 + CHUNK_OFFSET;
    index_3 = index_1 + CHUNK_OFFSET;
    index_4 = index_2 + CHUNK_OFFSET;
    index_5 = index_3 + CHUNK_OFFSET;
    temp_0 = local_data[index_0];
    temp_1 = local_data[index_1];
    theta = (local_address & 0x3)*PI4;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
    local_data[index_0] = temp_0+temp;
    local_data[index_1] = temp_0-temp;

    temp_2 = local_data[index_2];
    temp_3 = local_data[index_3];
    temp.REAL = sincos_temp.REAL * temp_3.REAL - sincos_temp.IMAG * temp_3.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_3.IMAG + sincos_temp.IMAG * temp_3.REAL;
    local_data[index_2] = temp_2+temp;
    local_data[index_3] = temp_2-temp;

    temp_4 = local_data[index_4];
    temp_5 = local_data[index_5];
    temp.REAL = sincos_temp.REAL * temp_5.REAL - sincos_temp.IMAG * temp_5.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_5.IMAG + sincos_temp.IMAG * temp_5.REAL;
    local_data[index_4] = temp_4+temp;
    local_data[index_5] = temp_4-temp;

    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 3
    index_0 = (((local_address << 1) & 0x70) + (local_address & 0x7));
    index_1 = index_0 + 8;
    index_2 = index_0 + CHUNK_OFFSET;
    index_3 = index_1 + CHUNK_OFFSET;
    index_4 = index_2 + CHUNK_OFFSET;
    index_5 = index_3 + CHUNK_OFFSET;
    temp_0 = local_data[index_0];
    temp_1 = local_data[index_1];
    theta = (local_address & 0x7)*PI8;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
    local_data[index_0] = temp_0+temp;
    local_data[index_1] = temp_0-temp;

    temp_2 = local_data[index_2];
    temp_3 = local_data[index_3];
    temp.REAL = sincos_temp.REAL * temp_3.REAL - sincos_temp.IMAG * temp_3.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_3.IMAG + sincos_temp.IMAG * temp_3.REAL;
    local_data[index_2] = temp_2+temp;
    local_data[index_3] = temp_2-temp;

    temp_4 = local_data[index_4];
    temp_5 = local_data[index_5];
    temp.REAL = sincos_temp.REAL * temp_5.REAL - sincos_temp.IMAG * temp_5.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_5.IMAG + sincos_temp.IMAG * temp_5.REAL;
    local_data[index_4] = temp_4+temp;
    local_data[index_5] = temp_4-temp;

    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 4
    index_0 = (((local_address << 1) & 0x60) + (local_address & 0xf));
    index_1 = index_0 + 16;
    index_2 = index_0 + CHUNK_OFFSET;
    index_3 = index_1 + CHUNK_OFFSET;
    index_4 = index_2 + CHUNK_OFFSET;
    index_5 = index_3 + CHUNK_OFFSET;
    temp_0 = local_data[index_0];
    temp_1 = local_data[index_1];
    theta = (local_address & 0xf)*PI16;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
    local_data[index_0] = temp_0+temp;
    local_data[index_1] = temp_0-temp;

    temp_2 = local_data[index_2];
    temp_3 = local_data[index_3];
    temp.REAL = sincos_temp.REAL * temp_3.REAL - sincos_temp.IMAG * temp_3.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_3.IMAG + sincos_temp.IMAG * temp_3.REAL;
    local_data[index_2] = temp_2+temp;
    local_data[index_3] = temp_2-temp;

    temp_4 = local_data[index_4];
    temp_5 = local_data[index_5];
    temp.REAL = sincos_temp.REAL * temp_5.REAL - sincos_temp.IMAG * temp_5.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_5.IMAG + sincos_temp.IMAG * temp_5.REAL;
    local_data[index_4] = temp_4+temp;
    local_data[index_5] = temp_4-temp;

    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 5
    index_0 = (((local_address << 1) & 0x40) + (local_address & 0x1f));
    index_1 = index_0 + 32;
    index_2 = index_0 + CHUNK_OFFSET;
    index_3 = index_1 + CHUNK_OFFSET;
    index_4 = index_2 + CHUNK_OFFSET;
    index_5 = index_3 + CHUNK_OFFSET;
    temp_0 = local_data[index_0];
    temp_1 = local_data[index_1];
    theta = (local_address & 0x1f)*PI32;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
    local_data[index_0] = temp_0+temp;
    local_data[index_1] = temp_0-temp;

    temp_2 = local_data[index_2];
    temp_3 = local_data[index_3];
    temp.REAL = sincos_temp.REAL * temp_3.REAL - sincos_temp.IMAG * temp_3.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_3.IMAG + sincos_temp.IMAG * temp_3.REAL;
    local_data[index_2] = temp_2+temp;
    local_data[index_3] = temp_2-temp;

    temp_4 = local_data[index_4];
    temp_5 = local_data[index_5];
    temp.REAL = sincos_temp.REAL * temp_5.REAL - sincos_temp.IMAG * temp_5.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_5.IMAG + sincos_temp.IMAG * temp_5.REAL;
    local_data[index_4] = temp_4+temp;
    local_data[index_5] = temp_4-temp;

    barrier(CLK_LOCAL_MEM_FENCE);

    //stage 6
    index_0 = (local_address & 0x3f);
    index_1 = index_0 + 64;
    index_2 = index_0 + CHUNK_OFFSET;
    index_3 = index_1 + CHUNK_OFFSET;
    index_4 = index_2 + CHUNK_OFFSET;
    index_5 = index_3 + CHUNK_OFFSET;
    temp_0 = local_data[index_0];
    temp_1 = local_data[index_1];
    theta = (local_address & 0x3f)*PI64;
    sincos_temp.IMAG = native_sin(theta);
    sincos_temp.REAL = native_cos(theta);
    temp.REAL = sincos_temp.REAL * temp_1.REAL - sincos_temp.IMAG * temp_1.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_1.IMAG + sincos_temp.IMAG * temp_1.REAL;
    local_data[index_0] = temp_0+temp;
    local_data[index_1] = temp_0-temp;

    temp_2 = local_data[index_2];
    temp_3 = local_data[index_3];
    temp.REAL = sincos_temp.REAL * temp_3.REAL - sincos_temp.IMAG * temp_3.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_3.IMAG + sincos_temp.IMAG * temp_3.REAL;
    local_data[index_2] = temp_2+temp;
    local_data[index_3] = temp_2-temp;

    temp_4 = local_data[index_4];
    temp_5 = local_data[index_5];
    temp.REAL = sincos_temp.REAL * temp_5.REAL - sincos_temp.IMAG * temp_5.IMAG;
    temp.IMAG = sincos_temp.REAL * temp_5.IMAG + sincos_temp.IMAG * temp_5.REAL;
    local_data[index_4] = temp_4+temp;
    local_data[index_5] = temp_4-temp;

    barrier(CLK_LOCAL_MEM_FENCE);


    //Downsample sum every 8 frequencies and 3 time, and sum Re Im
    //so write out 16 numbers only

    if (get_local_id(0) < 16){ //currently only 16 out of 64 has work to do. not ideal
      for (int j=0;j<3;j++){
        for (int i=0;i<8;i++){
          outtmp += local_data[ get_local_id(0)*8 +j*128 +i].REAL*local_data[ get_local_id(0)*8 +j*128 +i].REAL+local_data[ get_local_id(0)*8 +j*128 +i].IMAG*local_data[ get_local_id(0)*8 +j*128 +i].IMAG;
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (p == 1) {
        outtmp = outtmp/48.;
        //FFT shift by (id+8)%16
        results_array[get_global_id(1)*nsamp_out*16+get_group_id(0)*16+ ((get_local_id(0)+8)%16) ] = outtmp/BP[((get_local_id(0)+8)%16)] ;
      }
    }
  } //end loop of 2 pol
}


