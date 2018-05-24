#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <functional>
#include <string>
#include <errno.h>
#include <time.h>
#include <mutex>

#include "rfiRecord.hpp"
#include "util.h"
#include "errors.h"
#include "chimeMetadata.h"


REGISTER_KOTEKAN_PROCESS(rfiRecord);

rfiRecord::rfiRecord(Config& config,
                     const string& unique_name,
                     bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&rfiRecord::main_thread, this)){
    //Get buffer from framework
    rfi_buf = get_buffer("rfi_in");
    //Register process as consumer
    register_consumer(rfi_buf, unique_name.c_str());
    //Intialize internal config
    apply_config(0);
    using namespace std::placeholders;
    restServer &rest_server = restServer::instance();
    string endpoint = unique_name + "/rfi_record"
    rest_server.register_post_callback(endpoint,
                                        std::bind(&rfiRecord::rest_callback, this, _1, _2));
}

rfiRecord::~rfiRecord() {
}

void rfiRecord::rest_callback(connectionInstance& conn, json& json_request) {
    rest_callback_mutex.lock()

    INFO("RFI Callback Received... Changing Parameters")

    frames_per_packet = json_request["frames_per_packet"];
    write_to = json_request["write_to"];
    write_to_disk = json_request["write_to_disk"];
    filenum = 0;

    conn.send_empty_reply(STATUS_OK);
    rest_callback_mutex.unlock()
}


void rfiRecord::apply_config(uint64_t fpga_seq) {

    _num_freq = config.get_int(unique_name, "num_total_freq");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");

    _sk_step = config.get_int(unique_name, "sk_step");
    COMBINED = config.get_bool(unique_name,"rfi_combined");
    frames_per_packet = config.get_int(unique_name, "frames_per_packet");

    total_links = config.get_int(unique_name, "total_links");
    write_to = config.get_string(unique_name, "write_to");
    write_to_disk = config.get_bool(unique_name, "write_to_disk");
}

void rfiRecord::save_meta_data(uint16_t streamID, int64_t firstSeqNum) {

    //Create Directories
    char data_time[50];
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = gmtime(&rawtime);
    strftime(data_time, sizeof(data_time), "%Y%m%dT%H%M%SZ", timeinfo);
    snprintf(time_dir, sizeof(time_dir), "%s_rfi", data_time);
    DEBUG("Making Directories")
    make_rfi_dirs((int) streamID, write_to.c_str(), time_dir);
    //Create Info File
    char info_file_name[200];
    snprintf(info_file_name, sizeof(info_file_name), "%s/%d/%s/info.txt",
                      write_to.c_str(), streamID, time_dir);
    DEBUG("Creating Info File")
    FILE * info_file = fopen(info_file_name, "w");
    if(!info_file) {
        ERROR("Error creating info file: %s\n", info_file_name);
    }

    fprintf(info_file, "streamID=%d\n", streamID);
    fprintf(info_file, "firstSeqNum=%lld\n",(long long)firstSeqNum);
    fprintf(info_file, "utcTime=%s\n",data_time);
    fprintf(info_file, "num_elements=%d\n", _num_elements);
    fprintf(info_file, "num_total_freq=%d\n", _num_freq);
    fprintf(info_file, "num_local_freq=%d\n", _num_local_freq);
    fprintf(info_file, "samples_per_data_set=%d\n", _samples_per_data_set);
    fprintf(info_file, "sk_step=%d\n", _sk_step);
    fprintf(info_file, "rfi_combined=%d\n", COMBINED);
    fprintf(info_file, "frames_per_packet=%d\n", frames_per_packet);
    fprintf(info_file, "total_links=%d\n", total_links);
    fclose(info_file);

    INFO("Created meta data file: %s\n", info_file_name);
}
void rfiRecord::main_thread() {

    uint32_t frame_id = 0;
    uint8_t * frame = NULL;
    uint32_t link_id = 0;
    uint32_t file_num = 0;
    int fd = -1;
    while (!stop_thread) {

        double start_time = e_time();
        rest_callback_mutex.lock()
        //Get Frame
        frame = wait_for_full_frame(rfi_buf, unique_name.c_str(), frame_id);
        if (frame == NULL) break;
        DEBUG("Got frame %d", frame_id);
        if(file_num < total_links){
            //Make Necessary Directories using timecode
            //Create INFO file with metadata
            DEBUG("Saving Meta Data")
            save_meta_data(get_stream_id(rfi_buf, frame_id), get_fpga_seq_num(rfi_buf, frame_id));
//            save_meta_data((uint16_t)link_id, get_fpga_seq_num(rfi_buf, frame_id));
        }
        if(write_to_disk){
            char file_name[100];
            //Figure out which file
            snprintf(file_name, sizeof(file_name), "%s/%d/%s/%07d.rfi",
                     write_to.c_str(),
                     get_stream_id(rfi_buf, frame_id),
 //                    link_id,
                     time_dir,
                     file_num/1024);
            //Open that file
            fd = open(file_name, O_WRONLY | O_APPEND | O_CREAT, 0666);
            if (fd == -1) {
                ERROR("Cannot open file %s", file_name);
            }
            //Write to that file
            ssize_t bytes_writen = write(fd, frame, rfi_buf->frame_size);
            if (bytes_writen != rfi_buf->frame_size) {
                ERROR("Failed to write buffer to disk");
            }
            //Close that file
            if (close(fd) == -1) {
                ERROR("Cannot close file %s", file_name);
            }
        }
        //Mark Frame Empty
        mark_frame_empty(rfi_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % rfi_buf->num_frames;
//        DEBUG("Stream ID commented out for testing")
        link_id = (link_id + 1) % total_links;
        file_num++;
        DEBUG("Frame ID %d Succesfully Recorded link %d out of %d links in %fms",frame_id, link_id, total_links, (e_time()-start_time)*1000);
        rest_callback_mutex.unlock()
    }
}




