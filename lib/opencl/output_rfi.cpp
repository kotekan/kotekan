#include "output_rfi.h"

output_rfi::output_rfi(const char* param_name, Config &param_config, const string &unique_name):
    clCommand(param_name, param_config, unique_name)
{
}

output_rfi::~output_rfi()
{
}

void output_rfi::build(device_interface &param_Device)
{
    apply_config(0);
    clCommand::build(param_Device);
}

cl_event output_rfi::execute(int param_bufferID, const uint64_t& fpga_seq, class device_interface &param_Device, cl_event param_PrecedeEvent)
{
    clCommand::execute(param_bufferID, 0, param_Device, param_PrecedeEvent);
    /*unsigned int count_return_array[8*32768/256];
    clEnqueueReadBuffer (param_Device.getQueue(1),
        param_Device.getRfiCountBuffer(param_bufferID),
        CL_TRUE,
        0,
	sizeof(count_return_array),
        (void *)count_return_array,
        0,
	NULL,
	NULL);
    //FILE *f = fopen("../../../rfi_band.csv","a");
    for(int i = 0; i < _num_local_freq;i++){
	unsigned int counter = 0;
	for(int j = 0; j < 32768/256;j++){
		counter += count_return_array[i + 8*j];
	}
	int freq_bin = 15 + 0*16 + 128*i;
        float freq_mhz = 800 - freq_bin*((float)400/1024);
	//fprintf(f,"%f,%f\n",freq_mhz,(float)counter/_samples_per_data_set);
        INFO("Percentage Masked: %f Frequency %f\n", 100*(float)counter/32768, freq_mhz);
    }
    //fclose(f);
*/
    // Read the results
    CHECK_CL_ERROR( clEnqueueReadBuffer(param_Device.getQueue(0),
                                            param_Device.getRfiCountBuffer(param_bufferID),
                                            CL_FALSE,
                                            0,
                                            param_Device.getRfiBuf()->frame_size,
                                            param_Device.getRfiBuf()->frames[param_bufferID],
                                            1,
                                            &param_PrecedeEvent,
					    &postEvent[param_bufferID]) );
    INFO("Copied RFI to Buffer %d, Size %d",param_bufferID,param_Device.getRfiBuf()->frame_size);
    return postEvent[param_bufferID];
}
