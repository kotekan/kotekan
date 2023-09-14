#include "hdf5FileWrite.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, get_metadata_container, mark_frame_empty, regis...
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for chimeMetadata
#include "errors.h"
#include "kotekanLogging.hpp"    // for ERROR, INFO
#include "metadata.h"            // for metadataContainer
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "visUtil.hpp"           // for current_time

#undef NDEBUG

#include <atomic>     // for atomic_bool
#include <cassert>    // for assert
#include <errno.h>    // for errno
#include <exception>  // for exception
#include <fcntl.h>    // for open, O_CREAT, O_WRONLY
#include <functional> // for _Bind_helper<>::type, bind, function
#include <hdf5.h>
#include <iomanip>   // for setfill, setw
#include <regex>     // for match_results<>::_Base_type
#include <signal.h>  // for raise, SIGHUP
#include <sstream>   // for ostringstream
#include <stdexcept> // for runtime_error
#include <stdint.h>  // for uint32_t, int32_t, uint8_t
#include <stdio.h>   // for snprintf
#include <stdlib.h>  // for exit
#include <string>    // for string
#include <unistd.h>  // for gethostname
#include <vector>    // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(hdf5FileWrite);

std::atomic<uint32_t> hdf5FileWrite::n_finished{0};

hdf5FileWrite::hdf5FileWrite(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&hdf5FileWrite::main_thread, this)) {

    buf = get_buffer("in_buf");
    register_consumer(buf, unique_name.c_str());
    _base_dir = config.get<std::string>(unique_name, "base_dir");
    _file_name = config.get<std::string>(unique_name, "file_name");
    _prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    _exit_after_n_frames = config.get_default<uint32_t>(unique_name, "exit_after_n_frames", 0);
    _exit_with_n_writers = config.get_default<uint32_t>(unique_name, "exit_with_n_writers", 0);
}

hdf5FileWrite::~hdf5FileWrite() {}

void hdf5FileWrite::main_thread() {

    bool isFileOpen = false;
    std::string full_path;
    hid_t fd;

    uint32_t frame_id = 0;
    uint32_t frame_ctr = 0;
    uint8_t* frame = nullptr;

    auto& write_time_metric =
        Metrics::instance().add_gauge("kotekan_hdf5filewrite_write_time_seconds", unique_name);
    while (!stop_thread) {

        // This call is blocking.
        frame = wait_for_full_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        // Start timing the write time
        const double st = current_time();

        if (!isFileOpen) {

            std::ostringstream sbuf;
            sbuf << _base_dir << "/";
            if (_prefix_hostname) {
                char hostname[256];
                gethostname(hostname, sizeof hostname);
                sbuf << hostname << "_";
            }
            sbuf << _file_name << ".h5";
            full_path = sbuf.str();

            fd = H5Fcreate(full_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            if (fd < 0) {
                ERROR("Cannot open file");
                ERROR("File name was: {:s}", full_path.c_str());
                exit(errno);
            }

            isFileOpen = true;
        }

        // Create group for frame
        std::ostringstream sbuf;
        sbuf << std::setfill('0') << std::setw(7) << frame_ctr;
        const std::string frame_ctr_str = sbuf.str();
        const hid_t group =
            H5Gcreate(fd, frame_ctr_str.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (group < 0) {
            ERROR("Cannot create group");
            ERROR("Group name was: {:s}", frame_ctr_str.c_str());
            exit(errno);
        }

        // Write the metadata to file
        struct metadataContainer* mc = get_metadata_container(buf, frame_id);
        if (mc != nullptr) {
            const uint32_t metadata_size = mc->metadata_size;

            // There should be a metadata tag, or this should be a proper C++ class hierarchy.
            if (metadata_size == sizeof(chimeMetadata)) {
                const chimeMetadata* const md = (const chimeMetadata*)mc->metadata;
                {
                    const hid_t space = H5Screate(H5S_SCALAR);
                    assert(space >= 0);
                    const hid_t attr = H5Acreate(group, "fpga_seq_num", H5T_STD_I64LE, space,
                                                 H5P_DEFAULT, H5P_DEFAULT);
                    assert(attr >= 0);
                    herr_t herr = H5Awrite(attr, H5T_NATIVE_INT64, &md->fpga_seq_num);
                    assert(herr >= 0);
                    herr = H5Aclose(attr);
                    assert(herr >= 0);
                    herr = H5Sclose(space);
                    assert(herr >= 0);
                }
                // TODO: Write other attributes
            }
        }

        // Write the contents of the buffer frame to file

        hid_t space = -1;
        if (_file_name == "phases") {
            const int rank = 5;
            // const hsize_t dims[5] =
            //     {num_frequencies, num_polarizations, num_beams, num_dishes, num_components};
            const hsize_t dims[rank] = {16, 2, 96, 512, 2};
            assert(buf->frame_size == 16 * 2 * 96 * 512 * 2);
            space = H5Screate_simple(rank, dims, dims);
        } else if (_file_name == "voltage") {
            const int rank = 4;
            // const hsize_t dims[4] = {num_times, num_polarizations, num_frequencies, num_dishes};
            const hsize_t dims[rank] = {32768, 2, 16, 512};
            assert(buf->frame_size == 32768 * 2 * 16 * 512);
            space = H5Screate_simple(rank, dims, dims);
        } else if (_file_name == "beams" || _file_name == "wanted_beams") {
            const int rank = 4;
            // const hsize_t dims[4] = {num_times, num_polarizations, num_frequencies, num_beams};
            const hsize_t dims[rank] = {32768, 2, 16, 96};
            assert(buf->frame_size == 32768 * 2 * 16 * 96);
            space = H5Screate_simple(rank, dims, dims);
        } else {
            ERROR("Unknown file name {:s}; cannot deduce frame shape", _file_name.c_str());
        }
        assert(space >= 0);

        // TODO: Check buffer_type
        const hid_t dataset = H5Dcreate(group, buf->buffer_name, H5T_STD_U8LE, space, H5P_DEFAULT,
                                        H5P_DEFAULT, H5P_DEFAULT);
        assert(dataset >= 0);

        herr_t herr = H5Dwrite(dataset, H5T_NATIVE_UINT8, space, space, H5P_DEFAULT, frame);
        assert(herr >= 0);

        herr = H5Dclose(dataset);
        assert(herr >= 0);
        herr = H5Sclose(space);
        assert(herr >= 0);

        herr = H5Gclose(group);
        assert(herr >= 0);

        herr = H5Fflush(fd, H5F_SCOPE_GLOBAL);
        assert(herr >= 0);

        INFO("Data file write done for {:s}", full_path.c_str());

        ++frame_ctr;

        if (frame_ctr == _exit_after_n_frames) {
            herr = H5Fclose(fd);
            if (herr < 0) {
                ERROR("Cannot close file {:s}", full_path.c_str());
            }
            isFileOpen = false;
            stop_thread = true;
        }

        double elapsed = current_time() - st;
        write_time_metric.set(elapsed);

        mark_frame_empty(buf, unique_name.c_str(), frame_id);

        frame_id = (frame_id + 1) % buf->num_frames;
    }

    // Check if we should exit after writing out a fixed number of
    // files. Useful for some tests and burst modes. Will hopefully be
    // replaced by frames which can contain a "final" signal.
    if (!_exit_with_n_writers || ++n_finished >= _exit_with_n_writers)
        exit_kotekan(ReturnCode::CLEAN_EXIT);
}
