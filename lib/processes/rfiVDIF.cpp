#include "rfiVDIF.hpp"
#include <random>
#include "errors.h"
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include "vdif_functions.h"

using std::string;

REGISTER_KOTEKAN_PROCESS(rfiVDIF);

rfiVDIF::rfiVDIF(Config& config, const string& unique_name,
                                bufferContainer &buffer_containter) :
    KotekanProcess(config, unique_name, buffer_containter,
                     std::bind(&rfiVDIF::main_thread, this))
{
    //Apply kotekan config
    apply_config(0);
    //Register process as consumer and producer
    register_consumer(buf_in, unique_name.c_str());
    register_producer(buf_out,  unique_name.c_str());
}

rfiVDIF::~rfiVDIF() {
}

void rfiVDIF::apply_config(uint64_t fpga_seq) {

    //Buffers
    buf_in  = get_buffer("vdif_in");
    buf_out = get_buffer("rfi_out");

    //Data paramters
    num_elements = config.get_int(unique_name,"num_elements");
    num_frequencies = config.get_int(unique_name,"num_freq");
    num_timesteps = config.get_int(unique_name,"samples_per_data_set");
    
    //Rfi paramters
    COMBINED = config.get_bool(unique_name,"RFI_combined");
    SK_STEP = config.get_int(unique_name,"sk_step");
}

void rfiVDIF::main_thread() {

    //Frame parameters
    int buf_in_id = 0;
    uint8_t * in_frame = NULL;
    int buf_out_id = 0;
    uint8_t * out_frame = NULL;

    //Set the VDIF block size
    unsigned int VDIF_BLOCK_SIZE = num_frequencies + sizeof(VDIFHeader);
 
    //Counters and indices
    unsigned int i, j, k, block_counter, power, RFI_INDEX;
    long ptr_counter;

    //Holders for real/imag componenets	
    char real, imag; 

    //Total integration length
    float M;

    //Declare power arrays
    float power_arr[num_elements][num_frequencies]; 
    float power_sq_arr[num_elements][num_frequencies];

    //Invalid Data array
    unsigned int invalid_data_counter[num_elements];
    unsigned int RFI_Buffer_Size = num_elements*num_frequencies*(num_timesteps/SK_STEP);
    float S2[num_frequencies];

    //Create empty Buffer for RFI Values
    if(COMBINED){
        RFI_Buffer_Size /= num_elements;
    }

    //Buffer to hold kurtosis estimates
    float RFI_Buffer[RFI_Buffer_Size];

    //Value of current block's element index
    int current_element;

    while (!stop_thread) { //Endless loop

        //Get a new frame
        in_frame = wait_for_full_frame(buf_in, unique_name.c_str(), buf_in_id);
        if (in_frame == NULL) break;
        //Reset Counters
        RFI_INDEX = 0;
        block_counter = 0;
        ptr_counter = 0;

        //Loop through frame
        while(ptr_counter < buf_in->frame_size){

            //Initialize Arrays for a single block
            unsigned char block[VDIF_BLOCK_SIZE]; 
            
            if(block_counter == 0){ //Reset after each SK_Step
                for(i = 0; i < num_elements; i++){
                    invalid_data_counter[i] = 0;
                    for (j = 0; j < num_frequencies; j++){
                        power_arr[i][j] = 0;
                        power_sq_arr[i][j] = 0;
                    }
                }
            }

            //Read in first block
            memcpy(block,
                    in_frame+ptr_counter,
                    VDIF_BLOCK_SIZE);

            //Update the buffer location
            ptr_counter += VDIF_BLOCK_SIZE;

            //Find current input number
            current_element = (int)block[14];

            //Check Validity of the block
            //TODO Why doesn't this work?
            //if((block[3] & 0x1) == 0x1){
            //      invalid_data_counter[current_element]++;
            //      continue;		
            //}
            
            //Sum Across Time
            for(i = 0; i < num_frequencies; i++){
                real = ((block[sizeof(VDIFHeader) + i] >> 4) & 0xF)-8;
                imag = (block[sizeof(VDIFHeader) + i] & 0xF)-8;
                power = real*real + imag*imag; //Compute power
                power_arr[current_element][i] += power;
                power_sq_arr[current_element][i] += power*power;
            }
            
            //Update number of blocks read
            block_counter++; 

            //After a certain amount of timesteps
            if(block_counter == num_elements*SK_STEP){

                if(COMBINED){
                    
                    //Compute the correct value for M
                    M = num_elements*SK_STEP;
                    for(i = 0; i < num_elements; i++){

                        M -= invalid_data_counter[i];
                        for (j = 0; j < num_frequencies; j++){
                            //Normalize
                            power_sq_arr[i][j] /= (power_arr[i][j]/SK_STEP)*(power_arr[i][j]/SK_STEP);
                        }
                    }   
                    for(i = 0; i < num_frequencies; i++){

                        S2[i] = 0; //Intialize
                        //Sum Across Input
                        for (j = 0; j < num_elements; j++){
                
                            S2[i] += power_sq_arr[j][i];

                        }

                        //Compute Kurtosis for each frequency
                        RFI_Buffer[RFI_INDEX] = ((M+1)/(M-1))*(S2[i]/M-1);
                        RFI_INDEX++;
                        //INFO("Pre Value %f M %f Second Value %f S2 %f S2/M %f",((M+1)/(M-1)),M, (S2[i]/M-1),S2[i], S2[i]/M)
                    }					        				    
                }
                else{

                    //For each element
                    for(k = 0; k < num_elements; k ++){

                        //Compute the correct value for M
                        M = SK_STEP - (float)invalid_data_counter[k];

                        for(i = 0; i < num_frequencies; i++){

                            //Compute Kurtosis for each frequency
                            RFI_Buffer[RFI_INDEX] = (((M+1)/(M-1))*((M*power_sq_arr[k][i])/(power_arr[k][i]*power_arr[k][i])-1));
                            RFI_INDEX++;
                            
                        }
                    }
                }
                //Reset Block Counter
                block_counter = 0;
            }
        }
        //INFO("FIRST SK Value %f",RFI_Buffer[100])
        out_frame = wait_for_empty_frame(buf_out, unique_name.c_str(), buf_out_id);
        if (out_frame == NULL) break;

        memcpy(out_frame,RFI_Buffer,RFI_Buffer_Size*sizeof(float));

        mark_frame_full(buf_out, unique_name.c_str(), buf_out_id);
        mark_frame_empty(buf_in, unique_name.c_str(), buf_in_id);

        buf_out_id = (buf_out_id + 1) % buf_out->num_frames;
        buf_in_id = (buf_in_id + 1) % buf_in->num_frames;
        INFO("Frame %d Complete Stream ID\n", buf_in_id); 

        //Testing
        /*
        float counter = 0;
        for(k = 0; k < RFI_Buffer_Size; k ++){
            if(RFI_Buffer[k] > 1 + 6/(22.627)|| RFI_Buffer[k] < 1 - 6/(22.627)){
                counter++;
            }
        }
        INFO("Percent RFI %f",counter/RFI_Buffer_Size);*/
    }
}
