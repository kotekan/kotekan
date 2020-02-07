#include "calibration_io.hpp"

#include "chimeMetadata.h"
#include "fpga_header_functions.h"
#include "visBuffer.hpp"
#include "visFile.hpp"

#include <algorithm>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <sys/stat.h>

using namespace HighFive;


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(eigenWriter);


eigenWriter::eigenWriter(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&eigenWriter::main_thread, this)) {

    // Get parameters from config
    num_ev = config.get<size_t>(unique_name, "num_ev");
    ev_fname = config.get_default<std::string>(unique_name, "ev_file", "./ev.h5");
    // Default value is 1h / 10s cadence
    ev_file_len = config.get_default<size_t>(unique_name, "ev_file_len", 360);
    // Frequencies to include in file
    for (auto f : config.get<std::vector<int>>(unique_name, "freq_ids")) {
        freq_ids.push_back((uint16_t)f);
        freqs.push_back({freq_from_bin(f), (400.0 / 1024)});
    }

    // Fetch the buffer, register it
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Assuming all inputs are included
    inputs = std::get<1>(parse_reorder_default(config, unique_name));

    // open file for writing. delete existing file.
    struct stat check_exists;
    if (stat(ev_fname.c_str(), &check_exists) == 0)
        std::remove(ev_fname.c_str());
    file = std::unique_ptr<eigenFile>(new eigenFile(ev_fname, num_ev, ev_file_len, freqs, inputs));
}

eigenWriter::~eigenWriter() {
    file->flush();
}

void eigenWriter::main_thread() {

    unsigned int frame_id = 0;

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if ((wait_for_full_frame(in_buf, unique_name.c_str(), frame_id)) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        // Find the index of this frequency in the file
        uint16_t freq_ind =
            std::find(freq_ids.begin(), freq_ids.end(), frame.freq_id) - freq_ids.begin();

        // Put the time into correct format
        auto ftime = frame.time;
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};

        // Get data and write to file
        // TODO: once we have a better idea how HDF5 handles writing, could skip this extra copy
        std::vector<cfloat> evec(frame.evec.begin(), frame.evec.end());

        std::vector<float> eval(frame.eval.begin(), frame.eval.end());

        float erms = frame.erms;

        file->write_eigenvectors(t, freq_ind, evec, eval, erms);

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }
}


// TODO: include weights used in decomposition
eigenFile::eigenFile(const std::string& fname, const uint16_t& num_ev, const size_t& file_len,
                     const std::vector<freq_ctype>& freqs, const std::vector<input_ctype>& inputs) {

    ninput = inputs.size();
    nfreq = freqs.size();
    ntimes = file_len;
    nev = num_ev;

    // Create file
    INFO("Creating new eigenvectors file {:s}", fname);
    file = std::unique_ptr<File>(new File(fname, File::SwmrCreate));

    // Create eigenvector dataset
    std::vector<size_t> ev_dims = {ntimes, nfreq, nev, ninput};
    std::vector<std::string> ev_axes = {"time", "freq", "eigenmode", "input"};
    DataSpace ev_space = DataSpace(ev_dims);
    DataSet ev = file->createDataSet("evec", ev_space, create_datatype<cfloat>());
    ev.createAttribute<std::string>("axis", DataSpace::From(ev_axes)).write(ev_axes);

    // Create eigenvalue dataset
    std::vector<size_t> eval_dims = {ntimes, nfreq, nev};
    std::vector<std::string> eval_axes = {"time", "freq", "eigenmode"};
    DataSpace eval_space = DataSpace(eval_dims);
    DataSet eval = file->createDataSet("eval", eval_space, create_datatype<float>());
    eval.createAttribute<std::string>("axis", DataSpace::From(eval_axes)).write(eval_axes);

    // Create rms dataset
    std::vector<size_t> rms_dims = {ntimes, nfreq};
    std::vector<std::string> rms_axes = {"time", "freq"};
    DataSpace rms_space = DataSpace(rms_dims);
    DataSet rms = file->createDataSet("erms", rms_space, create_datatype<float>());
    rms.createAttribute<std::string>("axis", DataSpace::From(rms_axes)).write(rms_axes);

    // Create index map
    Group indexmap = file->createGroup("index_map");

    DataSet time_imap = indexmap.createDataSet<time_ctype>("time", DataSpace(ntimes));

    DataSet freq_imap = indexmap.createDataSet<freq_ctype>("freq", DataSpace(nfreq));
    freq_imap.write(freqs);

    DataSet input_imap = indexmap.createDataSet<input_ctype>("input", DataSpace(ninput));
    input_imap.write(inputs);

    // Start writing at top of file
    eof_ind = 0;
}

eigenFile::~eigenFile() {
    file->flush();
}

void eigenFile::flush() {
    file->flush();
}

void eigenFile::write_eigenvectors(time_ctype new_time, uint32_t freq_ind,
                                   std::vector<cfloat> eigenvectors, std::vector<float> eigenvalues,
                                   float new_rms) {

    // Find position in file
    size_t curr_ind;
    if (curr_times.size() < ntimes) {
        curr_ind = curr_times.size();
        curr_times.push_back(new_time.fpga_count);
    } else {
        uint64_t curr_time = new_time.fpga_count;
        auto time_ind = std::find(curr_times.begin(), curr_times.end(), curr_time);
        // Increment current end of file position
        if (time_ind == curr_times.end()) {
            curr_ind = eof_ind;
            eof_ind = (eof_ind + 1) % ntimes;
            curr_times[curr_ind] = curr_time;
        } else {
            curr_ind = time_ind - curr_times.begin();
        }
    }

    // write eigenvectors
    // need to cast to const ptr to fall into correct template
    evec()
        .select({curr_ind, freq_ind, 0, 0}, {1, 1, nev, ninput})
        .write((const cfloat*)eigenvectors.data());
    // write eigenvalues
    eval().select({curr_ind, freq_ind, 0}, {1, 1, nev}).write((const float*)eigenvalues.data());
    // write rms
    erms().select({curr_ind, freq_ind}, {1, 1}).write(new_rms);
    // write time
    time().select({curr_ind}, {1}).write(new_time);
}

DataSet eigenFile::evec() {
    return file->getDataSet("evec");
}

DataSet eigenFile::eval() {
    return file->getDataSet("eval");
}

DataSet eigenFile::erms() {
    return file->getDataSet("erms");
}

DataSet eigenFile::time() {
    return file->getDataSet("index_map/time");
}

DataSet eigenFile::freq() {
    return file->getDataSet("index_map/freq");
}

DataSet eigenFile::input() {
    return file->getDataSet("index_map/input");
}
