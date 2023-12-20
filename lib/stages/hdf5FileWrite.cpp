#include "hdf5FileWrite.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, get_metadata_container, mark_frame_empty, regis...
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for chimeMetadata
#include "chordMetadata.hpp"   // for chordMetadata
#include "errors.h"
#include "kotekanLogging.hpp"    // for ERROR, INFO
#include "metadata.hpp"          // for metadataContainer
#include "prometheusMetrics.hpp" // for Metrics, Gauge
#include "visUtil.hpp"           // for current_time

#include <atomic>     // for atomic_bool
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
    buf->register_consumer(unique_name);
    _base_dir = config.get<std::string>(unique_name, "base_dir");
    _file_name = config.get<std::string>(unique_name, "file_name");
    _prefix_hostname = config.get_default<bool>(unique_name, "prefix_hostname", true);
    _exit_after_n_frames = config.get_default<uint32_t>(unique_name, "exit_after_n_frames", 0);
    _exit_with_n_writers = config.get_default<uint32_t>(unique_name, "exit_with_n_writers", 0);
}

hdf5FileWrite::~hdf5FileWrite() {}

void hdf5FileWrite::main_thread() {
    std::string full_path;
    hid_t fd = H5I_UNINIT;

    uint32_t frame_id = 0;
    uint32_t frame_ctr = 0;
    uint8_t* frame = nullptr;

    auto& write_time_metric =
        Metrics::instance().add_gauge("kotekan_hdf5filewrite_write_time_seconds", unique_name);
    while (!stop_thread) {

        // This call is blocking.
        frame = buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        // Start timing the write time
        const double st = current_time();

        if (fd < 0) {
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
        struct metadataContainer* mc = buf->get_metadata_container(frame_id);
        if (mc != nullptr) {
            const uint32_t metadata_size = mc->metadata_size;

            // There should be a metadata tag, or this should be a proper C++ class hierarchy.
            if (metadata_size == sizeof(chimeMetadata)) {
                const chimeMetadata* const md = (const chimeMetadata*)mc->metadata;
                {
                    const hid_t space = H5Screate(H5S_SCALAR);
                    if (space < 0)
                        ERROR("Could not create data space");
                    const hid_t attr = H5Acreate(group, "fpga_seq_num", H5T_NATIVE_INT64, space,
                                                 H5P_DEFAULT, H5P_DEFAULT);
                    if (attr < 0)
                        ERROR("Could not create attribute");
                    herr_t herr = H5Awrite(attr, H5T_NATIVE_INT64, &md->fpga_seq_num);
                    if (herr < 0)
                        ERROR("Could not write attribute");
                    herr = H5Aclose(attr);
                    if (herr < 0)
                        ERROR("Could not close attribute");
                    herr = H5Sclose(space);
                    if (herr < 0)
                        ERROR("Could not close attribute");
                }
                // TODO: Write other attributes
            }
        }

        // Write the contents of the buffer frame to file

        hid_t type = -1;
        std::size_t type_size = 0;
        hid_t space = -1;

        if (metadata_container_is_chord(mc)) {
            // We have proper CHORD metadata and know the buffer type and shape
            const chordMetadata& metadata = *static_cast<const chordMetadata*>(mc->metadata);

            switch (metadata.type) {
                case int4p4:
                    type = H5T_NATIVE_UINT8;
                    type_size = 1;
                    break;
                case int8:
                    type = H5T_NATIVE_INT8;
                    type_size = 1;
                    break;
                case int16:
                    type = H5T_NATIVE_INT16;
                    type_size = 2;
                    break;
                case int32:
                    type = H5T_NATIVE_INT32;
                    type_size = 4;
                    break;
                case int64:
                    type = H5T_NATIVE_INT64;
                    type_size = 8;
                    break;
                case float16:
                    // TODO: Define HDF5 float16 type
                    type = H5T_NATIVE_UINT16;
                    type_size = 2;
                    break;
                case float32:
                    type = H5T_NATIVE_FLOAT;
                    type_size = 4;
                    break;
                case float64:
                    type = H5T_NATIVE_DOUBLE;
                    type_size = 8;
                    break;
                default:
                    ERROR("Unsupported metadata type");
            }

            const int rank = metadata.dims;
            if (rank < 0)
                ERROR("Negative number of metadata dimensions");
            hsize_t dims[CHORD_META_MAX_DIM];
            for (int d = 0; d < rank; ++d) {
                dims[d] = metadata.dim[d];
            }
            hsize_t np = 1;
            for (int d = 0; d < rank; ++d) {
                np *= dims[d];
            }
            if (buf->frame_size != np * type_size)
                ERROR("Buffer frame size is different from total metadata array length");
            space = H5Screate_simple(rank, dims, dims);

        } else {
            // We don't have proper CHORD metadata and don't know the buffer type and shape

            type = H5T_NATIVE_UINT8;
            const int rank = 1;
            hsize_t dims[1];
            dims[0] = buf->frame_size;
            space = H5Screate_simple(rank, dims, dims);
        }

        if (type < 0)
            ERROR("Illegal HDF5 data type");
        if (space < 0)
            ERROR("Illegal HDF5 data space");

        const hid_t dataset = H5Dcreate(group, buf->buffer_name.c_str(), type, space, H5P_DEFAULT,
                                        H5P_DEFAULT, H5P_DEFAULT);
        if (dataset < 0)
            ERROR("Could not create HDF5 dataset");

        if (metadata_container_is_chord(mc)) {
            // Write dimension names
            const chordMetadata& metadata = *static_cast<const chordMetadata*>(mc->metadata);

            hid_t dim_name_type = H5Tcreate(H5T_STRING, CHORD_META_MAX_DIMNAME);
            if (dim_name_type < 0)
                ERROR("Could not create HDF5 datatype for dim_name");
            const hsize_t dim_name_dims[1]{static_cast<hsize_t>(metadata.dims)};
            hid_t dim_name_space = H5Screate_simple(1, dim_name_dims, nullptr);
            if (dim_name_space < 0)
                ERROR("Could not create HDF5 dataspace for dim_name");
            hid_t dim_name_attr = H5Acreate2(dataset, "dim_name", dim_name_type, dim_name_space,
                                             H5P_DEFAULT, H5P_DEFAULT);
            if (dim_name_attr < 0)
                ERROR("Could not create HDF5 attribute for dim_name");
            herr_t herr = H5Awrite(dim_name_attr, dim_name_type, metadata.dim_name);
            if (herr < 0)
                ERROR("Could not write HDF5 attribute for dim_name");
            herr = H5Aclose(dim_name_attr);
            if (herr < 0)
                ERROR("Could not close HDF5 attribute for dim_name");
            herr = H5Sclose(dim_name_space);
            if (herr < 0)
                ERROR("Could not close HDF5 dataspace for dim_name");
            herr = H5Tclose(dim_name_type);
            if (herr < 0)
                ERROR("Could not close HDF5 datatype for dim_name");
        }

        herr_t herr = H5Dwrite(dataset, type, space, space, H5P_DEFAULT, frame);
        if (herr < 0)
            ERROR("Could not write HDF5 dataset");

        herr = H5Dclose(dataset);
        if (herr < 0)
            ERROR("Could not close HDF5 dataset");
        herr = H5Sclose(space);
        if (herr < 0)
            ERROR("Could not close HDF5 dataspace");

        herr = H5Gclose(group);
        if (herr < 0)
            ERROR("Could not close HDF5 group");

        herr = H5Fflush(fd, H5F_SCOPE_GLOBAL);
        if (herr < 0)
            ERROR("Could not flush HDF5 file");

        INFO("Data file write done for {:s} {:d}", full_path.c_str(), frame_ctr);

        ++frame_ctr;

        if (frame_ctr == _exit_after_n_frames) {
            if (fd >= 0) {
                herr = H5Fclose(fd);
                if (herr < 0) {
                    ERROR("Cannot close file {:s}", full_path.c_str());
                }
                fd = -1;
            }
            stop_thread = true;
        }

        double elapsed = current_time() - st;
        write_time_metric.set(elapsed);

        buf->mark_frame_empty(unique_name, frame_id);

        frame_id = (frame_id + 1) % buf->num_frames;
    }

    // Check if we should exit after writing out a fixed number of
    // files. Useful for some tests and burst modes. Will hopefully be
    // replaced by frames which can contain a "final" signal.
    if (!_exit_with_n_writers || ++n_finished >= _exit_with_n_writers) {
        exit_kotekan(ReturnCode::CLEAN_EXIT);
    }
}
