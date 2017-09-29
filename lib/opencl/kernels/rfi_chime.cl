/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor 
Date: August 2017
File Purpose: OpenCL Kernel to flag RFI in CHIME data
Details:
	Sums power and power**2 across time
	Normalizes
	Sums power and power**2 across elements
	Compute Spectral Kurtosis
	Flags RFI

**********************************************************************************/

__kernel void
rfi_chime(
	  __global char* input, //Input buffer
	  __global uint *count, //Amount of each frequency zeroed
	  __global float* in_means, //Normilization means
	  float sqrtM, //Square root of M
	  int sensitivity, //# of deviations for Kurtosis threshold
	  int time_samples, //Time samples in data block
	  int zero //Whether or no to zero data
)
{
	short gx = get_global_id(0); //Get Work Id's
	short gy = get_global_id(1);
	short gz = get_global_id(2);
	short lx = get_local_id(0);
	short gx_size = get_global_size(0); //#Elements
	short gy_size = get_global_size(1); //#Frequencies 
	short gz_size = get_global_size(2); //#Time Sample/SK_STEP
	short lx_size = get_local_size(0);
	uint SK_STEP = time_samples/gz_size;
	//ushort deadChannels[29] = {7,24,29,35,44,51,57,65,67,82,83,93,131,136,146,163,164,188,228,229,230,231,234,240,244,245,246,247,252};
	//bool deadChannels_flag[256];
	//for(short l = 0; l < 256; l++){
	//	deadChannels_flag[l] = 0;	
	//}
	//for(short k = 0; k < 29; k++){
	//	deadChannels_flag[deadChannels[k]] = 1;
	//}

	__local uint power_across_input[256]; //Local Memory
	__local uint sq_power_across_input[256];
	__local bool Zero_Flag;
	
	uint16 power_across_time = (uint16)(0); 
	uint16 sq_power_across_time = (uint16)(0);

	for(int i =0; i < SK_STEP; i+=16){ //Sum across time
		uint base_location = gx + gy*gx_size + (SK_STEP*gz + i)*(gx_size*gy_size);
		char16 data_point = (char16) (input[base_location], 
					input[base_location + 1*(gx_size*gy_size)],
					input[base_location + 2*(gx_size*gy_size)],
					input[base_location + 3*(gx_size*gy_size)],
					input[base_location + 4*(gx_size*gy_size)],
					input[base_location + 5*(gx_size*gy_size)],
					input[base_location + 6*(gx_size*gy_size)],
					input[base_location + 7*(gx_size*gy_size)],
					input[base_location + 8*(gx_size*gy_size)],
					input[base_location + 9*(gx_size*gy_size)],
					input[base_location + 10*(gx_size*gy_size)],
					input[base_location + 11*(gx_size*gy_size)],
					input[base_location + 12*(gx_size*gy_size)],
					input[base_location + 13*(gx_size*gy_size)],
					input[base_location + 14*(gx_size*gy_size)],
					input[base_location + 15*(gx_size*gy_size)]);
	   	short16 real = convert_short16((data_point >> 4) & 0xF) - (short16)(8);
		short16 imag = convert_short16(data_point & 0xF) - (short16)(8);
	   	short16 power = real*real + imag*imag; //Compute power
		power_across_time += convert_uint16(power);
	   	sq_power_across_time += convert_uint16(power)*convert_uint16(power);
	}
	
	uint8 collapse_power8 = power_across_time.odd + power_across_time.even;
	uint8 collapse_sq_power8 = sq_power_across_time.odd + sq_power_across_time.even;
	uint4 collapse_power4 = collapse_power8.odd + collapse_power8.even;
	uint4 collapse_sq_power4 = collapse_sq_power8.odd + collapse_sq_power8.even;
	uint2 collapse_power2 = collapse_power4.odd + collapse_power4.even;
	uint2 collapse_sq_power2 = collapse_sq_power4.odd + collapse_sq_power4.even;
	uint collapse_power = collapse_power2.odd + collapse_power2.even;
	uint collapse_sq_power = collapse_sq_power2.odd + collapse_sq_power2.even;

	if(gz == 0) in_means[gx + gy*gx_size] += 0.1*((float)collapse_power/SK_STEP - in_means[gx + gy*gx_size]);

	float mean = in_means[gx + gy*gx_size];

	barrier(CLK_GLOBAL_MEM_FENCE);

	float Median = mean*(512.0/729.0); //Adjust Mean to Median

	power_across_input[lx] = (collapse_power/Median)*(1-deadChannels_flag[lx]); //Save Values
	sq_power_across_input[lx] = (collapse_sq_power/(Median*Median))*(1-deadChannels_flag[lx]);

	barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish

	for(short i = lx_size/2; i>0; i >>= 1){ //Reduce along Elements
        	if(lx < i){
            		power_across_input[lx] += power_across_input[lx + i];
			sq_power_across_input[lx] += sq_power_across_input[lx + i];
       		}
        	barrier(CLK_LOCAL_MEM_FENCE);
    	}

	if(lx == 0){ //Calculate SK and Flag
		uint M = (gx_size-29)*SK_STEP;
		float SK = (((float)M+1)/((float)M-1))*((((float)M*sq_power_across_input[0])/((float)power_across_input[0]*power_across_input[0]))-1);
		Zero_Flag = (SK < (1 - (sensitivity*2.0)/sqrtM) || SK > (1 + (sensitivity*2.0)/sqrtM));
		count[gy+gy_size*gz] = SK_STEP*Zero_Flag;
	}
	
	//barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish

	/*for(short j = 0; j < SK_STEP; j++){ //Zero flagged values
		uint data_location = gx + gy*gx_size + (SK_STEP*gz + j)*(gx_size*gy_size);
		if(Zero_Flag && zero == 1) input[data_location] = 0x88; //Zero
	}*/
}


/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor 
Date: August 2017
File Purpose: OpenCL Kernel to flag RFI in CHIME data
Details:
	Sums power and power**2 across time
	Normalizes
	Sums power and power**2 across elements
	Compute Spectral Kurtosis
	Flags RFI

**********************************************************************************/
/*
__kernel void
rfi_chime(
	  __global char* input, //Input buffer
	  __global unsigned int *count, //Amount of each frequency zeroed
	  __global float* in_means, //Normilization means
	  float sqrtM, //Square root of M
	  int sensitivity, //# of deviations for Kurtosis threshold
	  int time_samples, //Time samples in data block
	  int zero //Whether or no to zero data
)
{
	int gx = get_global_id(0); //Get Work Id's
	int gy = get_global_id(1);
	int gz = get_global_id(2);
	int lx = get_local_id(0);
	int gx_size = get_global_size(0); //#Elements
	int gy_size = get_global_size(1); //#Frequencies 
	int gz_size = get_global_size(2); //#Time Sample/SK_STEP
	int lx_size = get_local_size(0);
	int SK_STEP = time_samples/gz_size;
	int deadChannels[29] = {7,24,29,35,44,51,57,65,67,82,83,93,131,136,146,163,164,188,228,229,230,231,234,240,244,245,246,247,252};
	int deadChannels_flag[256];
	for( int l = 0; l < 256; l++){
		deadChannels_flag[l] = 0;	
	}
	for(int k = 0; k < 29; k++){
		deadChannels_flag[deadChannels[k]] = 1;
	}

	__local unsigned int power_across_input[256]; //Local Memory
	__local unsigned int sq_power_across_input[256];
	__local bool Zero_Flag;
	
	unsigned int power_across_time = 0; 
	unsigned int sq_power_across_time = 0;
	float mean = in_means[gx + gy*gx_size];

	for(int i =0; i < SK_STEP; i++){ //Sum across time
		char data_point = input[gx + gy*gx_size + (SK_STEP*gz + i)*(gx_size*gy_size)];
	   	char real, imag;
	   	unsigned char power; //Compute power
		real = ((data_point >> 4) & 0xF)-8;
	   	imag = (data_point & 0xF)-8;
		power = real*real + imag*imag;
		mean  += 0.01*(power-mean); //Adjust mean
		power_across_time += power;
	   	sq_power_across_time += (power*power);
	}

	float Median = mean*(512.0/729.0); //Adjust Mean to Median
	power_across_input[lx] = (power_across_time/Median)*(1-deadChannels_flag[lx]); //Save Values
	sq_power_across_input[lx] = (sq_power_across_time/(Median*Median))*(1-deadChannels_flag[lx]);

	barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish

	for(int i = lx_size/2; i>0; i >>= 1){ //Reduce along Elements
        	if(lx < i){
            		power_across_input[lx] += power_across_input[lx + i];
			sq_power_across_input[lx] += sq_power_across_input[lx + i];
       		}
        	barrier(CLK_LOCAL_MEM_FENCE);
    	}

	if(lx == 0){ //Calculate SK and Flag
		unsigned int M = (gx_size-29)*SK_STEP;
		float SK = (((float)M+1)/((float)M-1))*((((float)M*sq_power_across_input[0])/((float)power_across_input[0]*power_across_input[0]))-1);
		Zero_Flag = (SK < (1 - (sensitivity*2.0)/sqrtM) || SK > (1 + (sensitivity*2.0)/sqrtM));
		count[gy+gy_size*gz] = SK_STEP*Zero_Flag;
	}
	in_means[gx + gy*gx_size] = mean;
	//barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish

	for(int j = 0; j < SK_STEP; j++){ //Zero flagged values
		unsigned int data_location = gx + gy*gx_size + (SK_STEP*gz + j)*(gx_size*gy_size);
		if(Zero_Flag && zero == 1) input[data_location] = 0x88; //Zero
	}
}*/

