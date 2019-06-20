// Input data is float with sample-beam-freq, sum each frequency across samples
//LWS = {     1 ,  1  }
//GWS = {1, 1024}

__kernel void sum_hfb(__global float *data, __global float *hfb_sum_output_array){

  uint nbeam = get_global_size(1);
  uint nsamp = get_global_size(0)*6+32;
  uint nsamp_out = get_global_size(0)*6/128/3;
  __local float local_data[128*10];
  uint local_address = get_local_id(0);
  const int work_offset = get_local_id(0) * 2;

  int beam = get_global_id(1); 
  uint address_in = beam * 128 * 10;

  //read from global
  //async_work_group_copy(local_data, &data[address_in], 128*10,0);
  //barrier(CLK_LOCAL_MEM_FENCE);
  
  // Read samples from global
  for(int sample=0; sample<10; sample++) {

      for(int freq=0; freq<128; freq++) {
      
          local_data[10*freq + sample] = data[1024*128*sample + beam*128 + freq];
      }
  }

  // Sum data across samples
  for(int freq=0; freq<128; freq++) {

      float sum = 0.f;

      for(int sample=0; sample<10; sample++) {
          sum += local_data[(freq * 10) + sample];
      }
      
      hfb_sum_output_array[(beam * 128) + freq] = sum;
  }

}
