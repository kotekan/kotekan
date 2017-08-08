/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor 
Date: August 2017
File Purpose: OpenCL Kernel to flag RFI in VDIF data
Details:
	Sums power and power**2 across time
	Normalizes
	Sums power and power**2 across elements
	Compute Spectral Kurtosis
	Flags RFI

**********************************************************************************/

__kernel void
rfi_vdif(
	  __global char* input, //Input Vdif data
	  __global char* masked_data, //Output VDIF data w/ RFI mask
	  __global float* in_means, //Input Average Power
	  float sqrtM, //Square root of Integration length
	  int sensitivity, //Number of standard deviations to place threshold
	  int time_samples, //Number of time-samples in data
	  int header_len // Length of VDIF header
)
{
	int gx = get_global_id(0); //Get Work Id's
	int gy = get_global_id(1);
	int gz = get_global_id(2);
	int ly = get_local_id(1);
	int gx_size = get_global_size(0); //#Frequencies 
	int gy_size = get_global_size(1); //#Elements
	int gz_size = get_global_size(2); //#Time Sample/SK_STEP

	int SK_STEP = time_samples/gz_size;

	__local float power_across_input[2]; //Local Memory
	__local float sq_power_across_input[2];
	__local bool Zero_Flag;
	
	__local unsigned int power_across_time[8]; //Local Memory
	__local unsigned int sq_power_across_time[8];

	power_across_time[gz] = 0; //Intialize Local Memory
	sq_power_across_time[gz] = 0;
	
	for(int i =0; i < SK_STEP; i++){ //Sum across time
		char data_point = input[header_len*(gy + 1) + gx + gy*gx_size + SK_STEP*gz*(gy_size*(gx_size+header_len)) + i*(gy_size*(gx_size+header_len))];
	   	char real, imag;
	   	unsigned int power;
		real = ((data_point >> 4) & 0xF)-8;
	   	imag = (data_point & 0xF)-8;
		power = real*real + imag*imag;
		in_means[gx + gy*gx_size]  += 0.01*(power-in_means[gx + gy*gx_size]);
		power_across_time[gz] += power;
	   	sq_power_across_time[gz] += (power*power);
	}

	barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish
	barrier(CLK_GLOBAL_MEM_FENCE);

	float Median = in_means[gx + gy*gx_size]*(512.0/729.0); //Adjust Mean to Median

	power_across_input[ly] = (float)power_across_time[gz]/Median; //Save Values
	sq_power_across_input[ly] = (float)sq_power_across_time[gz]/(Median*Median);

	for(int i = gy_size/2; i>0; i >>= 1){ //Reduce along Elements
        	if(ly < i){
            		power_across_input[ly] += power_across_input[ly + i];
			sq_power_across_input[ly] += sq_power_across_input[ly + i];
       		}
        	barrier(CLK_LOCAL_MEM_FENCE);
    	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(ly == 0){ //Calculate SK and Flag
		unsigned int M = gy_size*SK_STEP;
		float SK = ((((float)M+1)/(M-1))*(M*(sq_power_across_input[0])/(power_across_input[0]*power_across_input[0])-1));
		Zero_Flag = (SK < (1 - (sensitivity*2.0)/sqrtM) || SK > (1 + (sensitivity*2.0)/sqrtM));
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(int j = 0; j < SK_STEP; j++){ //Zero flagged values
		unsigned int data_location = header_len*(gy + 1) + gx + gy*gx_size + SK_STEP*gz*(gy_size*(gx_size+header_len)) + j*(gy_size*(gx_size+header_len));
		if(gx < header_len) masked_data[data_location-header_len] = (char)input[data_location-header_len]; //Copy Header
		if(Zero_Flag) masked_data[data_location] = input[data_location];//0x88; //Zero
		else masked_data[data_location] = input[data_location]; //Copy value
	}
}

