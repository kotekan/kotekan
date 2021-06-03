#include "oneDiskVDIFWrite.hpp"

#include "BeamMetadata.hpp"   // for BeamMetadata
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"           // for allocate_new_metadata_object, copy_metadata, mark_frame_empty
#include "chimeMetadata.hpp"  // for atomic_add_lost_timesamples, zero_lost_samples
#include "kotekanLogging.hpp" // for INFO
#include "tx_utils.hpp"       // for add_nsec
#include "vdif_functions.h"   // for VDIFHeader

#include <stdlib.h>
#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <iostream>    // for cout
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <visUtil.hpp> // for frameID, modulo

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(oneDiskVDIFWrite);

oneDiskVDIFWrite::oneDiskVDIFWrite(Config& config_, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&BeamBufferSort::main_thread, this)) {
        uint32_t nchan;
   
    disk_base = config.get<std::string>(unique_name, "disk_base");
    disk_set = config.get<std::string>(unique_name, "disk_set");
    disk_id = config.get_default<int>(unique_name, "disk_id", 0);     
    file_name = config.get<std::string>(unique_name, "file_name");
    file_ext = config.get_default<std::string>(unique_name, "file_ext", "vdif");
    use_abs_path = config.get_default<bool>(unique_name, "use_abs_path", false);
    abs_path = config.get_default<std::string>(unique_name, "abs_path", "");
    instrument_name =
        config.get_default<std::string>(unique_name, "instrument_name", "no_name_set");
    write_to_disk = config.get_default<bool>(unique_name, "write_to_disk", true);
    nframe_per_payload = config.get(unique_name, "nframe_per_payload");
    // Set up buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

oneDiskVDIFWrite::~oneDiskVDIFWrite(){}


void oneDiskVDIFWrite::main_thread(){
    uint8_t* input = nullptr;
    frameID in_frame_id(in_buf);
    bool isFileOpen = false;
    char data_time[64];
    char data_set_c[150];
    const int full_path_len = 200;
    char full_path[full_path_len];    
    time_t rawtime;
    struct tm* timeinfo;
    size_t file_num;
    uint32_t in_frame_count;
    uint32_t vdif_buf_size;
    uint32_t in_frame_offset = 0;
    uint32_t n_vdif_frame;
    

    time(&rawtime);

    timeinfo = gmtime(&rawtime);
    // The time prescision in the file should never be more than a second. 
    // Since creating the directory is independent for each disk now.
    // To make sure the directory name is the same, the precision of time stamp
    // in the directory name should be limited to seconds.
    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(data_set_c, sizeof(data_set_c), "%s_%s_%s", data_time, instrument_name.c_str(),
             file_ext.c_str());
    // Make output directory
    if (write_to_disk) {
        make_raw_dirs(disk_base.c_str(), disk_set.c_str(), dataset_name.c_str(), num_disks);
    }
    
    // allocate the vdif buffer 
    vdif_frame_header_size = 32;
    vdif_frame_data_size = vdif_samples_per_frame * vdif_freq_per_frame * num_pol;
    vdif_frame_size = vdif_frame_header_size + vdif_frame_data_size;
    vdif_frame = (VDIF_Frame*) malloc(vdif_frame_size);
    // Fill fixed parts of vdif frame header
    vdif_frame -> vdif_header -> invalid = 0;
    vdif_frame -> vdif_header -> legacy = 0;
    vdif_frame -> vdif_header -> ref_epoch = ref_epoch; // Need to tell
    vdif_frame -> vdif_header -> unused = 0;
    vdif_frame -> vdif_header -> log_num_chan = 3;
    vdif_frame -> vdif_header -> frame_len = (vdif_frame_data_size + vdif_frame_header_size) / 8;
    vdif_frame -> vdif_header -> vdif_version = 0;
    vdif_frame -> vdif_header -> data_type = 1;
    vdif_frame -> vdif_header -> bit_depth = 4 - 1;  //  bits per sample minus 1
    vdif_frame -> vdif_header -> station_id = 9999;
    vdif_frame -> vdif_header -> edv = 0;   // for now
    vdif_frame -> vdif_header -> eud1 = 0; 
    vdif_frame -> vdif_header -> eud2 = 0;
    vdif_frame -> vdif_header -> eud3 = 0;
    vdif_frame -> vdif_header -> eud4 = 0;


    while (!stop_thread) {
	if (in_frame_offset == 0){
            input = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
            if (input == nullptr)
                break;
            MergedBeamMetadata* in_metadata =
                (MergedBeamMetadata*)get_metadata(in_buf, in_frame_id);
        
            // Check if input buffer has the right amount of data.
	    // Number of channel in the file should be able to divided by 4
	    assert(in_metadata -> nchan % 4 == 0);
            // Number of time samples should be divided by vdif_samples_per_frame
	    assert(in_metadata -> sub_frame_data_size % vdif_samples_per_frame == 0);
            // Time resolution must be 2560 ns exactly
	    assert(in_metadata -> ctime.tv_nsec % (vdif_samples_per_frame * time_res_nsec) == 0);	
	
            // Compute the number of vdif frames in the input buffer 
	    n_vdif_time_frame = in_metadata -> sub_frame_data_size / vdif_samples_per_frame;
	    n_vdif_freq_frame = in_metadata -> nchan / vdif_freq_per_frame;
        
	    // input buffer frame size
	    in_buf_frame_size = in_metadata -> sub_frame_data_size + in_metadata -> sub_frame_metadata_size;
        }

	if (!isFileOpen) {
	    if (!use_abs_path)
	        snprintf(full_path, full_path_len, "%s/%s/%d/%s/%010zu_%02d.%s", disk_base.c_str(),	
		         disk_set.c_str(), disk_id, date_set_c.c_str(), file_num, disk_id, file_ext.c_str());
	    else{
	        snprintf(full_path, full_path_len, "%s/%010zu_%02d.%s", abs_path.c_str(), file_num, 0, file_ext.c_str());

	    }
	}
	// Copy data by vdif thread
	for (ii = 0; ii < n_vdif_time_frame; ii++){
            /// Fill the VDIF header changing part
            vdif_frame -> vdif_header -> seconds = unix_time_seconds - unix_time_seonds_reference;
            vdif_frame -> vdif_header -> frame_nr = unit_time_ns / (vdif_samples_per_frame * time_res_nsec);

	    for (jj = 0; jj < n_vdif_freq_frame; jj++){
	        // Add frame thread id by frequency.
                vdif_frame -> vdif_header -> thread_id = in_metadata -> freq_start + jj * vdif_freq_per_frame;
		// copy vdif frequency per frame (4) from input buffer
		for (kk = 0; kk < vdif_freq_per_frame; kk++){
		    // Copy data to payload
		    target_freq_frame = jj * vdif_freq_per_frame + kk;
		    src_start = target_freq_frame * in_buf_frame_size + in_frame_offset + in_metadata -> sub_frame_metadata_size;
		    target_start = kk * vdif_samples_per_frame;
		    uint8_t* target = &vdif_frame -> payload[target_start];
		    memcpy(target, input, vdif_samples_per_frame * num_pol);
		}
		if (write(fd, vdif_frame, vdif_frame_size) != vdif_frame_size){
		    ERROR("Failed to write vdif frame for file {:s}", full_path);
		    exit(-1);
		}
	    }
	    in_frame_offset = ii * vdif_samples_per_frame;     
	}

        frame_ctr++;
        // Finish writing the data for one file. Go to next file.
	if (frame_ctr == _num_frames_per_file) {
            if (close(fd) == -1) {
                ERROR("Cannot close file {:s}", full_path);
            }
            isFileOpen = false;
            frame_ctr = 0;
            file_num++;
        }
        
	// If the input data is finished, mark the in buffer empty
	if (in_frame_offset == in_metadata -> sub_frame_data_size){
	    mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
	    in_frame_id++;
	}

    }
    
}



