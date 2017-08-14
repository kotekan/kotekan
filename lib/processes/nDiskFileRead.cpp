/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor 
Date: August 2017
File Purpose: Read VDIF data from N disks. Has the option to remove RFI as it reads
Details: 
	-Constructor: Sets up config parameters as local variables. Registers process as producer.
	-main_thread: Creates and Handles N file reading threads
	-file_read_thread: Reads VDIF data from disk with the option of RFI removal
Notes:
	RFI removal only works for 2 element systems. It is more efficent to do RFI removal
in the computeDualpolPower Process or in the hsaRfiVdif GPU process.

**********************************************************************************/

#include "nDiskFileRead.hpp"
#include <random>
#include "errors.h"
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include "vdif_functions.h"

using std::string;

nDiskFileRead::nDiskFileRead(Config& config, const string& unique_name,
                                bufferContainer &buffer_containter) :
    KotekanProcess(config, unique_name, buffer_containter,
                     std::bind(&nDiskFileRead::main_thread, this))
{   
    buf = get_buffer("out_buf"); //Buffer

    num_disks = config.get_int(unique_name,"num_disks"); //Data paramters
    num_elements = config.get_int(unique_name,"num_elements");
    num_frequencies = config.get_int(unique_name,"num_freq");

    disk_base = config.get_string(unique_name,"disk_base"); //Data location parameters
    disk_set = config.get_string(unique_name,"disk_set");
    capture = config.get_string(unique_name,"capture");

    WITH_RFI = config.get_bool(unique_name,"rfi"); //Rfi paramters
    Normalize = config.get_bool(unique_name,"normalize");
    SK_STEP = config.get_int(unique_name,"sk_step");
    THRESHOLD_SENSITIVITY = config.get_int(unique_name,"rfi_sensitivity");

    register_producer(buf, unique_name.c_str()); //Mark as producer
}

nDiskFileRead::~nDiskFileRead() {

}

void nDiskFileRead::apply_config(uint64_t fpga_seq) {

}

void nDiskFileRead::main_thread() {

    // Create the threads
    file_thread_handles.resize(num_disks);
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i] = std::thread(&nDiskFileRead::file_read_thread, this, i);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        INFO("Setting thread affinity");
        for (auto &i : config.get_int_array(unique_name, "cpu_affinity"))
            CPU_SET(i, &cpuset);

        pthread_setaffinity_np(file_thread_handles[i].native_handle(), sizeof(cpu_set_t), &cpuset);
    }

    // Join the threads
    for (uint32_t i = 0; i < num_disks; ++i) {
        file_thread_handles[i].join();
    }
    mark_producer_done(buf, 0);
}

void nDiskFileRead::file_read_thread(int disk_id) {

    int buf_id = disk_id;
    unsigned int file_index = disk_id; //Starting File index

    unsigned int VDIF_BLOCK_SIZE = num_frequencies + sizeof(VDIFHeader); //Set the VDIF block size
    float MEAN_SENSITIVITY = 0.01; //Sets the amount each power value moves the running mean

    float Mean[num_frequencies*num_elements];//Declare Mean array

    for(unsigned int i = 0; i < num_frequencies*num_elements; i++){
	Mean[i] = 0; //Intialize Mean array
    }

    for (;;) { //Endless loop

        wait_for_empty_buffer(buf, unique_name.c_str(), buf_id);

        unsigned char* buf_ptr = buf->data[buf_id];
        char file_name[100]; //Find current file
        snprintf(file_name, sizeof(file_name), "%s%s/%d/%s/%07d.vdif",disk_base.c_str(),disk_set.c_str(), disk_id,capture.c_str(),file_index);
	INFO("Entering File: %s\n",file_name);
        FILE * in_file = fopen(file_name, "r"); //Open current file for reading
        fseek(in_file, 0L, SEEK_END); //Get length of current file
        long sz = ftell(in_file);
        rewind(in_file);
        assert(sz == buf->buffer_size); 
        
	if(!WITH_RFI){ //Without RFI removal, read whole file into buffer
		fread(buf_ptr,buf->buffer_size,1,in_file);
	}
        else{ //With RFI removal
		//NOTE: Here thread refers to independant elements 
		unsigned int thread_counter = 0; 
		float thread_1_power[num_frequencies]; //Declare power arrays
		float thread_2_power[num_frequencies];
		float thread_1_power_sq[num_frequencies];
		float thread_2_power_sq[num_frequencies];
		float adjust = 512.0/729.0; //Compute mean to median factor

		//Create empty Buffer for whole file
		char *total_buffer_ptr = new char[(num_elements*SK_STEP*VDIF_BLOCK_SIZE)];

		//Initialize Arrays
		for(unsigned int i = 0; i < num_frequencies; i++){
			thread_1_power[i] = 0;
		   	thread_1_power_sq[i] = 0;
			thread_2_power[i] = 0;
		   	thread_2_power_sq[i] = 0;
		}
		//Intialize buffer
		memset(total_buffer_ptr,136,(num_elements*SK_STEP*VDIF_BLOCK_SIZE));

		long ptr_counter = 0;//Current Location in file
		unsigned int invalid_data_counter = 0; //amount of invalid data

		while(ptr_counter < sz){//Loop through file

			thread_counter++; //Update number of threads
			char buffer[VDIF_BLOCK_SIZE]; //Buffer to hold first VDIF Block
			bool mask[num_frequencies];
			fread(&buffer, sizeof(char), VDIF_BLOCK_SIZE, in_file);//Read in first block
			memcpy(total_buffer_ptr, (void*)&buffer, sizeof(buffer));
			total_buffer_ptr += sizeof(buffer);
		
			if((buffer[3] & 0x1) == 0x1){//Is it valid
				invalid_data_counter++; 
				continue;
			}
			//Sum Across Time
			if((int)buffer[14] == 0){ //If thread 0
				for(unsigned int i = 0; i < num_frequencies; i++){
					char data_point = buffer[i + sizeof(VDIFHeader)];
					char real, imag;
				   	unsigned int power;
					real = ((data_point >> 4) & 0xF)-8;
				   	imag = (data_point & 0xF)-8;
					power = real*real + imag*imag; //Compute power
					thread_1_power[i] += power;
				   	thread_1_power_sq[i] += power*power;
					//Update Mean
					if(Mean[i] == 0){
						Mean[i] = thread_1_power[i];
					}
					Mean[i] += (thread_1_power[i] - Mean[i])*MEAN_SENSITIVITY;
				}
			}
			else{//If thread 1
				for(unsigned int i = 0; i < num_frequencies; i++){
					char data_point = buffer[i + sizeof(VDIFHeader)];
					char real, imag;
				   	unsigned int power;
					real = ((data_point >> 4) & 0xF)-8;
				   	imag = (data_point & 0xF)-8;
					power = real*real + imag*imag;
					thread_2_power[i] += power;
				   	thread_2_power_sq[i] += power*power;
					if(Mean[i+num_frequencies] == 0){
						Mean[i+num_frequencies] = thread_2_power[i];
					}
					Mean[i + num_frequencies] += (thread_2_power[i] - Mean[i + num_frequencies])*MEAN_SENSITIVITY;
				}
			}
			//After a certain amount of time
			if(thread_counter == num_elements*SK_STEP){ 
				unsigned int M = (num_elements)*(SK_STEP) - invalid_data_counter;
			    	float LOWER_BOUND =  1 - THRESHOLD_SENSITIVITY*2/sqrt((float)M);
				float UPPER_BOUND = 1 + THRESHOLD_SENSITIVITY*2/sqrt((float)M);
				thread_counter = 0;//Reset Counter
				//Sum across elements
				for(unsigned int i = 0; i < num_frequencies; i++){

					if(Normalize){
						//Normalize
						thread_1_power[i] /= Mean[i]*adjust;
						thread_2_power[i] /= Mean[i+num_frequencies]*adjust;
						thread_1_power_sq[i] /= (Mean[i]*adjust)*(Mean[i]*adjust);
						thread_2_power_sq[i] /= (Mean[i+num_frequencies]*adjust)*(Mean[i+num_frequencies]*adjust);
					}
					
					//Sum across inputs
					thread_1_power[i] += thread_2_power[i];
				   	thread_1_power_sq[i] += thread_2_power_sq[i];
				
					//Compute Kurtosis
					float SK = (float)((((float)M+1)/(M-1))*(M*(thread_1_power_sq[i])/(thread_1_power[i]*thread_1_power[i])-1));
					//Apply Threshold
					if((SK < LOWER_BOUND || SK > UPPER_BOUND)){
						mask[i] = 0;
					}
					else{
						mask[i] = 1;
					}
				
					thread_1_power[i] = 0;
		   			thread_1_power_sq[i] = 0;
					thread_2_power[i] = 0;
		   			thread_2_power_sq[i] = 0;
				}
				total_buffer_ptr -= (num_elements*SK_STEP*VDIF_BLOCK_SIZE);
				for(unsigned int i =0; i < num_elements*SK_STEP; i++){ //Zero data
					total_buffer_ptr += sizeof(VDIFHeader);
					for(unsigned int k = 0; k < num_frequencies; k++){
						if(i%2 == 1){
							if(mask[k] == 0){ //RFI
								memset(total_buffer_ptr,137,1);
							}
							else{ //Not RFI
								memset(total_buffer_ptr,136,1);
							}
						}
						total_buffer_ptr++;
					}
				}
				//Fill in total buffer with RFI removed
				total_buffer_ptr -= (num_elements*SK_STEP*VDIF_BLOCK_SIZE);
				memcpy(buf_ptr, total_buffer_ptr, (num_elements*SK_STEP*VDIF_BLOCK_SIZE));
				buf_ptr += num_elements*SK_STEP*VDIF_BLOCK_SIZE; //Update Buffer locations
				ptr_counter += num_elements*SK_STEP*VDIF_BLOCK_SIZE;
				invalid_data_counter = 0;
			}
		}
	}

        file_index += num_disks;

        fclose(in_file);

        set_data_ID(buf, buf_id, file_index);
        mark_buffer_full(buf, unique_name.c_str(), buf_id);
        buf_id = (buf_id + num_disks) % buf->num_buffers;

        INFO("nDiskFileRead: read %s\n", file_name);
    }
}
