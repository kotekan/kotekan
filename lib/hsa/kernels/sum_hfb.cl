// Input data is float with sample-beam-freq, sum each frequency across samples
//LWS = {64,  1  }
//GWS = {64, 1024}

#define NUM_SUB_FREQS 128

__kernel void sum_hfb(__global float *data, __constant uchar *compressed_lost_samples_buf, __global float *hfb_sum_output_array, const uint num_samples){
    
  uint local_address = get_local_id(0);

  const int beam = get_global_id(1); 
  const int freq = get_local_id(0) * 2;

  // Sum data across samples from global memory
  float freq_sum_1 = 0.f, freq_sum_2 = 0.f;

  for(int sample=0; sample<num_samples; sample++) {

      if(!compressed_lost_samples_buf[sample]) {
          freq_sum_1 += data[1024*NUM_SUB_FREQS*sample + beam*NUM_SUB_FREQS + freq];
          freq_sum_2 += data[1024*NUM_SUB_FREQS*sample + beam*NUM_SUB_FREQS + freq + 1];
      }
  }

  // Write sums back to global
  hfb_sum_output_array[(beam * NUM_SUB_FREQS) + freq] = freq_sum_1;
  hfb_sum_output_array[(beam * NUM_SUB_FREQS) + freq + 1] = freq_sum_2;
}
