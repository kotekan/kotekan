// Input data is float with sample-beam-freq, sum each frequency across all samples.
// Excluding samples from the sum where there is >=1 in the lost samples. A vector reduction is performed to get the sum.
// Each work group is comprised of 64 work items, where each work item sums samples for 2 frequencies. The grid size is 64 x 1024, so each group works on 1 beam.
//
//LWS = {64,  1  } (num_sub_freq / 2, 1)
//GWS = {64, 1024} (num_sub_freq / 2, num_frb_total_beams)

#define NUM_SUB_FREQS 128

__kernel void sum_hfb(__global float *data, __constant uint *compressed_lost_samples_buf, __global float *hfb_sum_output_array, const uint num_samples){
    
  uint local_address = get_local_id(0);

  const int beam = get_global_id(1); 
  const int freq = get_local_id(0) * 2;

  // Sum data across samples from global memory
  float4 freq_sum_1 = (float4)(0.f, 0.f, 0.f, 0.f), freq_sum_2 = (float4)(0.f, 0.f, 0.f, 0.f);
  float4 data_1, data_2;

  for(int sample=0; sample<num_samples; sample+=4) {

      if(!compressed_lost_samples_buf[sample]) {

          // Load data into vectors
          data_1.s0 = data[1024*NUM_SUB_FREQS*sample + beam*NUM_SUB_FREQS + freq];
          data_1.s1 = data[1024*NUM_SUB_FREQS*(sample + 1) + beam*NUM_SUB_FREQS + freq];
          data_1.s2 = data[1024*NUM_SUB_FREQS*(sample + 2) + beam*NUM_SUB_FREQS + freq];
          data_1.s3 = data[1024*NUM_SUB_FREQS*(sample + 3) + beam*NUM_SUB_FREQS + freq];
          
          data_2.s0 = data[1024*NUM_SUB_FREQS*sample + beam*NUM_SUB_FREQS + freq + 1];
          data_2.s1 = data[1024*NUM_SUB_FREQS*(sample + 1) + beam*NUM_SUB_FREQS + freq + 1];
          data_2.s2 = data[1024*NUM_SUB_FREQS*(sample + 2) + beam*NUM_SUB_FREQS + freq + 1];
          data_2.s3 = data[1024*NUM_SUB_FREQS*(sample + 3) + beam*NUM_SUB_FREQS + freq + 1];
 
          // Add vectors
          freq_sum_1 += data_1;
          freq_sum_2 += data_2;
      }
  }

  // Write sums back to global
  // Perform reduction on vector elements to get total
  hfb_sum_output_array[(beam * NUM_SUB_FREQS) + freq] = freq_sum_1.s0 + freq_sum_1.s1 + freq_sum_1.s2 + freq_sum_1.s3;
  hfb_sum_output_array[(beam * NUM_SUB_FREQS) + freq + 1] = freq_sum_2.s0 + freq_sum_2.s1 + freq_sum_2.s2 + freq_sum_2.s3;
}
