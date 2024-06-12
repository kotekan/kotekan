#include "pulsarSimProcess.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for wait_for_empty_frame, mark_frame_full, register_producer
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for beamCoord
#include "kotekanLogging.hpp"  // for INFO, CHECK_MEM
#include "vdif_functions.h"    // for VDIFHeader

#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <fstream>    // for basic_ostream::operator<<, operator<<, stringstream, basi...
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdlib.h>   // for exit
#include <string.h>   // for memcpy
#include <string>     // for allocator, string, char_traits
#include <sys/time.h> // for timeval
#include <unistd.h>   // for gethostname
#include <vector>     // for vector

using std::string;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

#define samples_in_frame 3125
#define num_packet 16

REGISTER_KOTEKAN_STAGE(pulsarSimProcess);
pulsarSimProcess::pulsarSimProcess(Config& config_, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&pulsarSimProcess::main_thread, this)) {

    // Apply config.
    _num_gpus = config.get<int32_t>(unique_name, "num_gpus");
    _nfreq_coarse = config.get<int32_t>(unique_name, "num_gpus"); // 4
    _num_pulsar = config.get<int32_t>(unique_name, "num_pulsar");
    _num_pol = config.get<int32_t>(unique_name, "num_pol");
    _udp_packet_size = config.get<int32_t>(unique_name, "udp_pulsar_packet_size");
    _udp_header_size = config.get<int32_t>(unique_name, "udp_pulsar_header_size");

    pulsar_buf = get_buffer("pulsar_out_buf");
    pulsar_buf->register_producer(unique_name);
}

pulsarSimProcess::~pulsarSimProcess() {}

void pulsarSimProcess::parse_host_name() {
    int rack = 0, node = 0, nos = 0;
    std::stringstream temp_ip[number_of_subnets];

    gethostname(my_host_name, sizeof(my_host_name));
    CHECK_MEM(my_host_name);

    if (my_host_name[0] != 'c' && my_host_name[3] != 'g') {
        INFO("Not a valid name \n");
        exit(0);
    }

    if (my_host_name[1] == 'n') {
        nos = 0;
        my_node_id = 0;
    } else if (my_host_name[1] == 's') {
        nos = 100;
        my_node_id = 128;
    } else {
        INFO("Not a valid name \n");
        exit(0);
    }

    switch (my_host_name[2]) {
        case '0':
            rack = 0;
            break;
        case '1':
            rack = 1;
            break;
        case '2':
            rack = 2;
            break;
        case '3':
            rack = 3;
            break;
        case '4':
            rack = 4;
            break;
        case '5':
            rack = 5;
            break;
        case '6':
            rack = 6;
            break;
        // case '7': rack=7; break;
        case '8':
            rack = 8;
            break;
        case '9':
            rack = 9;
            break;
        case 'A':
            rack = 10;
            break;
        case 'B':
            rack = 11;
            break;
        case 'C':
            rack = 12;
            break;
        case 'D':
            rack = 13;
            break;
        default:
            INFO("Not a valid name \n");
            exit(0);
    }

    switch (my_host_name[4]) {
        case '0':
            node = 0;
            break;
        case '1':
            node = 1;
            break;
        case '2':
            node = 2;
            break;
        case '3':
            node = 3;
            break;
        case '4':
            node = 4;
            break;
        case '5':
            node = 5;
            break;
        case '6':
            node = 6;
            break;
        case '7':
            node = 7;
            break;
        case '8':
            node = 8;
            break;
        case '9':
            node = 9;
            break;
        default:
            INFO("Not a valid name \n");
            exit(0);
    }

    for (int i = 0; i < number_of_subnets; i++) {
        temp_ip[i] << "10." << i + 15 << "." << nos + rack << ".1" << node;
        my_ip_address[i] = temp_ip[i].str();
        INFO("{:s} ", my_ip_address[i]);
    }

    if (rack < 7)
        my_node_id += rack * 10 + (9 - node); // fix for the arrangment of nodes in the racks
    if (rack > 7)
        my_node_id += (rack - 1) * 10 + (9 - node);
}

void pulsarSimProcess::fill_headers(unsigned char* out_buf, struct VDIFHeader* vdif_header,
                                    const uint64_t fpga_seq_num, struct timeval* time_now,
                                    struct beamCoord* beam_coord, uint16_t* freq_ids) {
    //        assert(sizeof(struct VDIFHeader) == _udp_header_size);
    for (int i = 0; i < num_packet; ++i) { // 16 frames in a stream
        uint64_t fpga_now = (fpga_seq_num + samples_in_frame * i);
        vdif_header->eud2 = (fpga_now & ((0xFFFFFFFFl) << 32)) >> 32;
        vdif_header->eud3 = (fpga_now & ((0xFFFFFFFFl) >> 0));
        vdif_header->seconds = time_now->tv_sec;
        vdif_header->data_frame = (time_now->tv_usec / 1.e6) / (samples_in_frame * 2.56e-6);

        for (int f = 0; f < _num_gpus; ++f) { // 4 freq
            vdif_header->thread_id = freq_ids[f];
            for (int psr = 0; psr < _num_pulsar; ++psr) { // 10 streams
                vdif_header->eud1 = psr;                  // beam id
                uint16_t ra_part = (uint16_t)(beam_coord[f].ra[psr] * 100);
                uint16_t dec_part = (uint16_t)((beam_coord[f].dec[psr] + 90) * 100);
                vdif_header->eud4 = ((ra_part << 16) & 0xFFFF0000) + (dec_part & 0xFFFF);
                memcpy(&out_buf[(f * _num_pulsar + psr) * num_packet * _udp_packet_size
                                + i * _udp_packet_size],
                       vdif_header, sizeof(struct VDIFHeader));
            }
        } // end freq
        // Increment time for the next frame
        time_now->tv_usec += samples_in_frame * 2.56;
        if (time_now->tv_usec > 999999) {
            time_now->tv_usec = time_now->tv_usec % 1000000;
            time_now->tv_sec += 1;
        }
    } // end packet
}

void pulsarSimProcess::main_thread() {
    number_of_subnets = 2;
    int out_buffer_ID = 0;

    parse_host_name();
    uint16_t freq_ids[_num_gpus];

    struct timeval time_now = (struct timeval){0, 0};

    struct VDIFHeader vdif_header;
    vdif_header.seconds = 0; // UD
    vdif_header.legacy = 0;
    vdif_header.invalid = 0;
    vdif_header.data_frame = 0; // UD
    vdif_header.ref_epoch = 36; // First half of 2018.
    vdif_header.unused = 0;
    vdif_header.frame_len = 5000;
    vdif_header.log_num_chan = 0; // Check ln4=2 or ln1=0 ?
    vdif_header.vdif_version = 1;
    char si[2] = {'C', 'H'};
    vdif_header.station_id = (si[0] << 8) + si[1]; // Need to fomrally ask the Vdif community
    vdif_header.thread_id = 0;                     // UD     freq
    vdif_header.bits_depth = 8;                    // 4+4
    vdif_header.data_type = 1;                     // Complex
    vdif_header.edv = 0;
    vdif_header.eud1 = 0; // UD: beam number [0 to 9]
    vdif_header.eud2 = 0; // UD: fpga count high bit
    vdif_header.eud3 = 0; // UD: fpga count low bit
    vdif_header.eud4 = 0; // Ra_int + Ra_dec + Dec_int + Dec_dec ? Source name ? Obs ID?

    uint64_t fpga_seq_num = 0;

    struct beamCoord beam_coord[_num_gpus];
    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* out_frame = pulsar_buf->wait_for_empty_frame(unique_name, out_buffer_ID);
    if (out_frame == nullptr)
        return;

    for (int i = 0; i < _num_gpus; i++)
        freq_ids[i] = my_node_id;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        // If this is the first time wait until we get the start of an interger second period.
        pulsar_buf->mark_frame_full(unique_name, out_buffer_ID);
        out_buffer_ID = (out_buffer_ID + 1) % pulsar_buf->num_frames;
        out_frame = pulsar_buf->wait_for_empty_frame(unique_name, out_buffer_ID);
        if (out_frame == nullptr)
            return;
        // Fill the headers of the new buffer
        fpga_seq_num += samples_in_frame * num_packet;
        fill_headers((unsigned char*)out_frame, &vdif_header, fpga_seq_num, &time_now, beam_coord,
                     (uint16_t*)freq_ids);
    } // end stop thread
}
