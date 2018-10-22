#define TS 32
#define REAL    x
#define IMAG    y

__kernel void pulsarbf( __global uint *data, __global float2 *phase,  __global unsigned char *output){

  __local float2 sum[256];
  uint nsamp = get_global_size(2)*TS;
  //float ph[8]; //need 8x2 phases for these unpacked guys
  float2 ph0 = phase[ (get_group_id(1)*2+get_group_id(0))*1024 + get_local_id(0)*4];
  float2 ph1 = phase[ (get_group_id(1)*2+get_group_id(0))*1024 + get_local_id(0)*4+1];
  float2 ph2 = phase[ (get_group_id(1)*2+get_group_id(0))*1024 + get_local_id(0)*4+2];
  float2 ph3 = phase[ (get_group_id(1)*2+get_group_id(0))*1024 + get_local_id(0)*4+3];

  for (int t=0;t<TS;t++){
  
    uint data_temp = data[(t*nsamp/TS+get_group_id(2))*512 + get_group_id(0)*256 + get_local_id(0) ];
    sum[get_local_id(0)].REAL = ph0.REAL*((float)((data_temp&0x000000f0)>>4u)-8) 
      - ph0.IMAG*((float)((data_temp & 0x0000000f)>> 0u)-8)
      + ph1.REAL*((float)((data_temp & 0x0000f000)>>12u)-8)
      - ph1.IMAG*((float)((data_temp & 0x00000f00)>> 8u)-8)
      + ph2.REAL*((float)((data_temp & 0x00f00000)>>20u)-8)
      - ph2.IMAG*((float)((data_temp & 0x000f0000)>>16u)-8)
      + ph3.REAL*((float)((data_temp & 0xf0000000)>>28u)-8)
      - ph3.IMAG*((float)((data_temp & 0x0f000000)>>24u)-8);

    sum[get_local_id(0)].IMAG = ph0.IMAG*((float)((data_temp&0x000000f0)>>4u)-8) 
      + ph0.REAL*((float)((data_temp & 0x0000000f)>> 0u)-8)
      + ph1.IMAG*((float)((data_temp & 0x0000f000)>>12u)-8)
      + ph1.REAL*((float)((data_temp & 0x00000f00)>> 8u)-8)
      + ph2.IMAG*((float)((data_temp & 0x00f00000)>>20u)-8)
      + ph2.REAL*((float)((data_temp & 0x000f0000)>>16u)-8)
      + ph3.IMAG*((float)((data_temp & 0xf0000000)>>28u)-8)
      + ph3.REAL*((float)((data_temp & 0x0f000000)>>24u)-8);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Reduction of 128, eventually each number comes from the sum of 2048 values
    float factor;
    factor = 1024;
    if (get_local_id(0) < 64) {
      sum[get_local_id(0)] = sum[4*get_local_id(0)]+ sum[4*get_local_id(0)+1] + sum[4*get_local_id(0) +2] + sum[4*get_local_id(0)+3];
    }
    if (get_local_id(0) < 16) {
      sum[get_local_id(0)] = sum[4*get_local_id(0)] + sum[4*get_local_id(0) +1] + sum[4*get_local_id(0) +2] + sum[4*get_local_id(0)+3];
    }
    if (get_local_id(0) < 4) {
      sum[get_local_id(0)] = sum[4*get_local_id(0)] + sum[4*get_local_id(0) +1] + sum[4*get_local_id(0) +2] + sum[4*get_local_id(0)+3];
    }
    if (get_local_id(0) == 0) {
      float2 final = sum[4*get_local_id(0)] + sum[4*get_local_id(0) +1] + sum[4*get_local_id(0) +2] + sum[4*get_local_id(0)+3];

      //Real
      uint out_real = (int) final.REAL/factor+0.5;
      uint out_imag = (int) final.IMAG/factor+0.5;
      output[(t*nsamp/TS+get_group_id(2))*20 + get_group_id(1)*2 + get_group_id(0)] = ((out_real<<4) & 0xF0) + (out_imag & 0x0F);
    }
    barrier(CLK_LOCAL_MEM_FENCE); //crucial if TS > 1 !
  } //end t
  
}

