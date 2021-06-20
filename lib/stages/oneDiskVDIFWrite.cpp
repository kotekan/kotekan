#include "oneDiskVDIFWrite.hpp"

#include "BeamMetadata.hpp"   // for BeamMetadata
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"           // for allocate_new_metadata_object, copy_metadata, mark_frame_empty
#include "chimeMetadata.hpp"  // for atomic_add_lost_timesamples, zero_lost_samples
#include "kotekanLogging.hpp" // for INFO
#include "tx_utils.hpp"       // for add_nsec
#include "vdif_functions.h"   // for VDIFHeader
#include "util.h"             // for make_raw_dirs 

#include <assert.h>    // for assert
#include <stdlib.h>
#include <atomic>      // for atomic_bool
#include <chrono>      // for system_clock, system_clock::time_point
#include <errno.h>     // for errno
#include <math.h>      // for modf
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <iostream>    // for cout
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <fcntl.h>     // for open, O_CREAT, O_WRONLY
#include <unistd.h>    // for write, close, gethostname, ssize_t
#include <visUtil.hpp> // for frameID, modulo

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using std::string;

REGISTER_KOTEKAN_STAGE(oneDiskVDIFWrite);

oneDiskVDIFWrite::oneDiskVDIFWrite(Config& config_, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&oneDiskVDIFWrite::main_thread, this)) {
    double int_year;
    double frac_year;
    tm ref_time{};

    disk_base = config.get<std::string>(unique_name, "disk_base");
    disk_set = config.get<std::string>(unique_name, "disk_set");
    disk_id = config.get<int>(unique_name, "disk_id");     
    file_name = config.get<std::string>(unique_name, "file_name");
    file_ext = config.get_default<std::string>(unique_name, "file_ext", "vdif");
    use_abs_path = config.get_default<bool>(unique_name, "use_abs_path", false);
    abs_path = config.get_default<std::string>(unique_name, "abs_path", "");
    instrument_name =
        config.get_default<std::string>(unique_name, "instrument_name", "no_name_set");
    write_to_disk = config.get_default<bool>(unique_name, "write_to_disk", true);
    nframe_per_payload = config.get<int>(unique_name, "nframe_per_payload");
    nvdif_payload_per_file = config.get<uint32_t>(unique_name, "nvdif_payload_per_file");
    vdif_frame_header_size = config.get<uint32_t>(unique_name, "vdif_frame_header_size");
    num_pol = config.get<uint32_t>(unique_name, "num_pol"); 
    // The refernce epoch would be the last two digit or the year. If it in half of a 
    // year, add 0.5. 
    ref_year = config.get<double>(unique_name, "ref_year");
    frac_year = modf(ref_year, &int_year);
    ref_time.tm_year = (int)int_year - 1900;
    ref_time.tm_mday = 1;
    if (frac_year > 0){ // The reference time is in the second half of the year
        ref_time.tm_mon = 6;
    }
    time_t ref_tt = std::mktime(&ref_time);
    ref_ct = timespec{ref_tt, 0};
    //unix_time_sec_ref = 
    vdif_samples_per_frame = config.get_default<float>(unique_name, "vdif_samples_per_frame", 625);
    vdif_freq_per_frame = config.get_default<float>(unique_name, "vdif_freq_per_frame", 4);
    string note = config.get_default<std::string>(unique_name, "note", "");
    auto& tel = Telescope::instance();
    time_res_nsec = tel.seq_length_nsec();
    // Set up buffers
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
}

oneDiskVDIFWrite::~oneDiskVDIFWrite(){}


void oneDiskVDIFWrite::main_thread(){
    int fd;
    int payload_ctr;
    uint8_t* input = nullptr;
    frameID in_frame_id(in_buf);
    uint8_t* vdif_frame;
    VDIFHeader* vdif_header; // VDIF header
    uint8_t* payload;  // VDIF payload
    bool isFileOpen = false;
    char data_time[64];
    char data_set_c[150];
    char data_dir_name[256];
    const int full_path_len = 512;
    char full_path[full_path_len];    
    time_t rawtime;
    struct tm* timeinfo;
    size_t file_num;
    uint32_t in_frame_count;
    uint32_t vdif_buf_size;
    uint32_t time_offset = 0;
    uint32_t freq_offset = 0;
    uint32_t n_vdif_frame;
    uint32_t vdif_frame_data_size;
    uint32_t vdif_frame_size;
    uint32_t n_vdif_time_frame;
    uint32_t n_vdif_freq_frame;
    uint32_t in_buf_frame_size;
    uint32_t target_freq_frame;
    uint32_t src_start;
    uint32_t target_start;
    uint32_t inframe_ctr = 0;
    long intsec_offset;
    char setting_log_file[full_path_len];
    char data_log_file[full_path_len];
    FILE* setting_file;
    FILE* j_file;

    MergedBeamMetadata* in_metadata = nullptr;
    // telescope object for use later
    auto& tel = Telescope::instance();
    timespec out_frame_unix_time;
    timespec vdif_file_start_time;


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
        if (use_abs_path){
            snprintf(data_dir_name, sizeof(data_dir_name), "%s", abs_path.c_str());
	}
	else{
	    snprintf(data_dir_name, sizeof(data_dir_name), "%s/%s/%d/%s", disk_base.c_str(), disk_set.c_str(), disk_id, data_set_c);
	}
	make_dir(data_dir_name);
	// write setting log
	snprintf(setting_log_file, full_path_len, "%s/%s.txt", data_dir_name, "setting_log");
        setting_file = fopen(setting_log_file, "w");
	snprintf(data_log_file, full_path_len, "%s/%s.txt", data_dir_name, "data_log");
	j_file = fopen(data_log_file, "w");
    }
    
        
    // allocate the vdif buffer 
    vdif_frame_data_size = vdif_samples_per_frame * vdif_freq_per_frame * num_pol;
    vdif_frame_size = vdif_frame_header_size + vdif_frame_data_size;
    INFO("VDIF size {:d} {:d}", vdif_frame_data_size, vdif_frame_size);
    vdif_frame = (uint8_t*) malloc(vdif_frame_size);
    // Cast vdif header
    vdif_header = (VDIFHeader*) &vdif_frame[0];
    // cast vdif payload
    payload = &vdif_frame[vdif_frame_header_size];
    // Fill fixed parts of vdif frame header
    vdif_header -> invalid = 0;
    vdif_header -> legacy = 0;
    // Only take the last two digits of the year.
    // If the time beyond 2100, we should consider change the code here
    vdif_header -> ref_epoch = (uint32_t)((ref_year - 2000) * 2) ; // Cast it to int. 
    vdif_header -> unused = 0;
    vdif_header -> log_num_chan = 3;
    vdif_header -> frame_len = (vdif_frame_data_size + vdif_frame_header_size) / 8;
    vdif_header -> vdif_version = 0;
    vdif_header -> data_type = 1;
    vdif_header -> bits_depth = 4 - 1;  //  bits per sample minus 1
    vdif_header -> station_id = 9999;
    vdif_header -> edv = 0;   // for now
    vdif_header -> eud1 = 0; 
    vdif_header -> eud2 = 0;
    vdif_header -> eud3 = 0;
    vdif_header -> eud4 = 0;

    // Write setting log file 
    const int data_format_version = 3;

    while (!stop_thread) {
	if (time_offset == 0){
            input = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
            if (input == nullptr)
                break;
            in_metadata = (MergedBeamMetadata*)get_metadata(in_buf, in_frame_id);
            INFO("Data is here\n");      
            // Check if input buffer has the right amount of data.
	    // Number of channel in the file should be able to divided by 4
	    assert(in_metadata -> nchan % 4 == 0);
            // Number of time samples should be divided by vdif_samples_per_frame
	    assert(in_metadata -> sub_frame_data_size % vdif_samples_per_frame == 0);
	    // Make sure the vdif file has all the frequency from the input buffer
	    assert(nvdif_payload_per_file % in_metadata -> nchan ==0);
            // Time resolution must be 2560 ns exactly
	    // Check if the start is n times of vdif_samples_per_frame * time_res_nsec
            INFO("Time {:d} TIME RES {:d}", in_metadata -> ctime.tv_nsec, time_res_nsec);
	    //assert(in_metadata -> ctime.tv_nsec % (vdif_samples_per_frame * time_res_nsec) == 0);	
	    if (file_num == 0 && payload_ctr == 0){
	        // Get the frame zero for checking start time offset from an integer second. 
                timespec time0 = in_metadata -> ctime;
	        add_nsec(time0, -1 * (long)(in_metadata -> fpga_seq_start * time_res_nsec));
                INFO("Time {:d} {:d}", time0.tv_sec, time0.tv_nsec);
		// Check with integer second the offset should go
		if (time0.tv_nsec > time_res_nsec){
		    intsec_offset = time0.tv_nsec - 1e9;
		}
		else{
		    intsec_offset = time0.tv_nsec;
		}
		INFO("Intset_off {:d}", intsec_offset);
		if(write_to_disk){
		    fprintf(setting_file, "format_version_number=%02d\n", data_format_version);
                    fprintf(setting_file, "num_freq=%d\n", in_metadata -> nchan);
                    fprintf(setting_file, "num_inputs=%d\n", num_pol);
                    fprintf(setting_file, "num_frames=%d\n", 1); // Not sure what does this one mean
                    fprintf(setting_file, "num_timesamples=%d\n", nvdif_payload_per_file * vdif_samples_per_frame);
                    fprintf(setting_file, "header_len=%d\n", vdif_frame_header_size); // VDIF
                    fprintf(setting_file, "packet_len=%d\n",
                        vdif_frame_header_size + in_metadata -> nchan); // nchan + header_size. TODO: need to confirm the purpose of this.
                    fprintf(setting_file, "offset=%d\n", 0);
                    fprintf(setting_file, "data_bits=%d\n", vdif_header -> bits_depth);
                    fprintf(setting_file, "stride=%d\n", 1);
                    fprintf(setting_file, "stream_id=n/a\n");
                    fprintf(setting_file, "note=\"%s\"\n", note.c_str());
                    fprintf(setting_file, "start_time=%s\n", data_time); // dataset_name.c_str());
                    fprintf(setting_file, "start_sec_offset (ns)=%ld\n", intsec_offset);
		    fprintf(setting_file, "# Warning: The start time is when the program starts it, the time "
                        "recorded in the packets is more accurate\n");
		    fclose(setting_file);
                    INFO("Created setting log file: {:s}\n", setting_log_file);
		}
            }    

            // Compute the number of vdif frames in the input buffer 
	    n_vdif_time_frame = in_metadata -> sub_frame_data_size / vdif_samples_per_frame;
	    n_vdif_freq_frame = in_metadata -> nchan / vdif_freq_per_frame;
        
	    // input buffer frame size
	    in_buf_frame_size = in_metadata -> sub_frame_data_size + in_metadata -> sub_frame_metadata_size;
        }
        // If not a new file, open a new file.
	if (!isFileOpen && write_to_disk) {
	    snprintf(full_path, full_path_len, "%s/%010zu_%02d.%s", data_dir_name,	
		     file_num, disk_id, file_ext.c_str());
	    
	    fd = open(full_path, O_WRONLY | O_CREAT, 0666);

            if (fd == -1) {
                ERROR("Cannot open file");
                ERROR("File name was: {:s}", file_name);
                exit(errno);
            }
	    isFileOpen = true;
	    INFO("Write data to: {:s}", full_path);
	}
	out_frame_unix_time = in_metadata -> ctime;
        add_nsec(out_frame_unix_time, (long)(time_offset * time_res_nsec));	
	// Record the VDIF file start time
	if (payload_ctr == 0){
	    vdif_file_start_time = out_frame_unix_time;
	}
	//Start filling the header in the vdif payload
	assert(out_frame_unix_time.tv_sec > ref_ct.tv_sec && "Reference time is beyond current time.");
	vdif_header -> seconds = out_frame_unix_time.tv_sec - ref_ct.tv_sec;
	//INFO("second {:d} {:d}", vdif_header -> seconds, out_frame_unix_time.tv_sec - ref_ct.tv_sec);
	//assert(out_frame_unix_time.tv_nsec % (vdif_samples_per_frame * time_res_nsec) ==0);
        vdif_header -> data_frame = out_frame_unix_time.tv_nsec / (vdif_samples_per_frame * time_res_nsec);
        vdif_header -> thread_id = in_metadata -> freq_start + freq_offset;
	INFO("Time offset {:d}", time_offset);
	// copy vdif frequency per frame (4) from input buffer
	for (uint32_t ii = 0; ii < vdif_freq_per_frame; ii++){
            // Copy data to payload
            src_start = freq_offset * in_buf_frame_size + time_offset + in_metadata -> sub_frame_metadata_size;
            target_start = ii * vdif_samples_per_frame;
	    uint8_t* target = &payload[target_start];
            memcpy(target, &input[src_start], vdif_samples_per_frame * num_pol);
	    freq_offset ++;
        }
	//INFO("Freq offset {:d}", freq_offset);
	// If all the frequency from one time is finished, move to the next time.
	if (freq_offset >= in_metadata -> nchan){
	    time_offset += vdif_samples_per_frame;
	    freq_offset = 0;
	}
	if (write_to_disk){
	    if (write(fd, vdif_frame, vdif_frame_size) != vdif_frame_size){
	        ERROR("Failed to write vdif frame for file {:s}", full_path);
                exit(-1);
	    }
	}

        // number of payload in file 
        payload_ctr++;
        // Finish writing the data for one file. Go to next file.
	if (payload_ctr == nvdif_payload_per_file) {
            if (write_to_disk){
                if (close(fd) == -1) {
                    ERROR("Cannot close file {:s}", full_path);
                }
                isFileOpen = false;
                fprintf(j_file, "%ld %ld %s\n", vdif_file_start_time.tv_sec, vdif_file_start_time.tv_nsec, full_path);
	    }
	    else{
	        INFO("Data packet ready for at {:d} {:d}", vdif_file_start_time.tv_sec, vdif_file_start_time.tv_nsec);
	    }
            payload_ctr = 0;
            file_num++;
        }
        
	// If the input data is finished, mark the in buffer empty
	if (time_offset == in_metadata -> sub_frame_data_size){
	    mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
	    in_frame_id++;
	    inframe_ctr++;
	    time_offset = 0;
	}
    }
    // free vdif frame 
    free(vdif_frame);
    fclose(j_file);
    
}



