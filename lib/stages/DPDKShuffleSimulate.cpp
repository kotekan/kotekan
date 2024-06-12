#include "DPDKShuffleSimulate.hpp"

#include "Config.hpp"          // for Config
#include "ICETelescope.hpp"    // for ice_stream_id_t, ice_encode_stream_id
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"       // for Telescope
#include "buffer.hpp"          // for Buffer, allocate_new_metadata_object, mark_frame_full
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for set_first_packet_recv_time, set_fpga_seq_num, set_stream_id
#include "kotekanLogging.hpp"  // for DEBUG, INFO
#include "visUtil.hpp"         // for frameID, current_time, modulo, ts_to_double

#include "fmt.hpp" // for format

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <regex>      // for match_results<>::_Base_type
#include <stddef.h>   // for size_t
#include <sys/time.h> // for gettimeofday, timeval
#include <unistd.h>   // for sleep, usleep
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(DPDKShuffleSimulate);

STAGE_CONSTRUCTOR(DPDKShuffleSimulate) {

    // Register as consumer on buffer
    lost_samples_buf = get_buffer("lost_samples_buf");
    lost_samples_buf->register_producer(unique_name);

    for (uint32_t i = 0; i < shuffle_size; ++i) {
        voltage_data_buf[i] = get_buffer(fmt::format("voltage_data_buf_{:d}", i));
        voltage_data_buf[i]->register_producer(unique_name);
    }

    // Get some configuration options
    _num_samples_per_dataset = config.get<int32_t>(unique_name, "samples_per_data_set");
}

DPDKShuffleSimulate::~DPDKShuffleSimulate() {}

void DPDKShuffleSimulate::main_thread() {

    frameID lost_samples_frame_id(lost_samples_buf);
    frameID voltage_data_frame_id[shuffle_size] = {voltage_data_buf[0], voltage_data_buf[1],
                                                   voltage_data_buf[2], voltage_data_buf[3]};

    double frame_length =
        _num_samples_per_dataset * ts_to_double(Telescope::instance().seq_length());

    uint8_t* lost_samples_frame;
    uint8_t* voltage_data_frames[shuffle_size];

    uint64_t fpga_seq = 0;

    sleep(20);

    while (!stop_thread) {
        double start_time = current_time();

        lost_samples_frame =
            (uint8_t*)lost_samples_buf->wait_for_empty_frame(unique_name, lost_samples_frame_id);
        if (lost_samples_frame == nullptr)
            break;

        for (uint32_t i = 0; i < shuffle_size; ++i) {
            voltage_data_frames[i] = (uint8_t*)voltage_data_buf[i]->wait_for_empty_frame(
                unique_name, voltage_data_frame_id[i]);
            if (voltage_data_frames[i] == nullptr)
                break;
        }

        // Set contents for lost samples buffer
        for (uint32_t sample = 0; sample < _num_samples_per_dataset; ++sample) {
            assert((size_t)_num_samples_per_dataset == lost_samples_buf->frame_size);
            // TODO add option to have some data lost
            lost_samples_frame[sample] = 0;
        }

        // Set metadata for voltage buffers
        struct timeval now;
        gettimeofday(&now, nullptr);
        for (uint32_t i = 0; i < shuffle_size; ++i) {
            voltage_data_buf[i]->allocate_new_metadata_object(voltage_data_frame_id[i]);

            // StreamID
            ice_stream_id_t ice_stream_id;
            ice_stream_id.crate_id = 0;
            ice_stream_id.link_id = 0;
            ice_stream_id.slot_id = 0;
            ice_stream_id.unused = i;
            set_stream_id(voltage_data_buf[i], voltage_data_frame_id[i],
                          ice_encode_stream_id(ice_stream_id));

            set_fpga_seq_num(voltage_data_buf[i], voltage_data_frame_id[i], fpga_seq);
            set_first_packet_recv_time(voltage_data_buf[i], voltage_data_frame_id[i], now);
        }
        fpga_seq += _num_samples_per_dataset;

        // Set metadata for lost samples buf
        lost_samples_buf->allocate_new_metadata_object(lost_samples_frame_id);
        set_fpga_seq_num(lost_samples_buf, lost_samples_frame_id, fpga_seq);
        set_first_packet_recv_time(lost_samples_buf, lost_samples_frame_id, now);

        // TODO Set the default values for the frames to something.

        // Release frames
        lost_samples_buf->mark_frame_full(unique_name, lost_samples_frame_id++);
        for (uint32_t i = 0; i < shuffle_size; ++i) {
            voltage_data_buf[i]->mark_frame_full(unique_name, voltage_data_frame_id[i]++);
        }

        // Sleep for a period of time to match the FPGA data rate.
        double time = current_time();
        double frame_end_time = start_time + frame_length;
        if (time < frame_end_time) {
            DEBUG("Generated a set of frames, now sleeping for: {:f} seconds",
                  frame_end_time - time);
            usleep((int)(1e6 * (frame_end_time - time)));
        }
        // Log message to confirm the system is alive
        if (fpga_seq / _num_samples_per_dataset % 100 == 0)
            INFO("Generated {:d} test samples.", fpga_seq);
    }
}
