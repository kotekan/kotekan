// Input data is float with sample-beam-freq, sum each frequency across samples
//LWS = {64,  1  }
//GWS = {64, 1024}

__kernel void sum_hfb(__global float *data, __constant uchar *lost_samples_buf, __global float *hfb_sum_output_array, const uint num_samples){
    
  __local float local_data[128*10];
  uint local_address = get_local_id(0);

  const int beam = get_global_id(1); 
  const int freq = get_local_id(0) * 2;

  // Read samples from global
  for(int sample=0; sample<num_samples; sample++) {

      local_data[num_samples*freq + sample] = data[1024*128*sample + beam*128 + freq];
      local_data[num_samples*(freq + 1) + sample] = data[1024*128*sample + beam*128 + freq + 1];
  }
    
  // Sum data across samples
  float freq_sum_1 = 0.f, freq_sum_2 = 0.f;

  for(int sample=0; sample<num_samples; sample++) {

      if(!lost_samples_buf[((freq * num_samples) + sample) % 30])
          freq_sum_1 += local_data[(freq * num_samples) + sample];
      
      if(!lost_samples_buf[(((freq + 1) * num_samples) + sample) % 30])
          freq_sum_2 += local_data[((freq + 1) * num_samples) + sample];
  }

  hfb_sum_output_array[(beam * 128) + freq] = freq_sum_1;
  hfb_sum_output_array[(beam * 128) + freq + 1] = freq_sum_2;
}
