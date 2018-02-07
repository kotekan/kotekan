#include "writeEigenvec.hpp"
#include "visBuffer.hpp"
#include "chimeMetadata.h"
#include "fpga_header_functions.h"
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <algorithm>

using namespace HighFive;

writeEigenvec::writeEigenvec(Config &config,
                             const string& unique_name,
                             bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&writeEigenvec::main_thread, this)) {

    // Get parameters from config
    num_eigenvectors =  config.get_int(unique_name, "num_eigenvectors");
    ev_fname = config.get_string_default(unique_name, "ev_file", "./ev.h5");
    // Default value is 1h / 10s cadence
    ev_file_len = config.get_int_default(unique_name, "ev_file_len", 360);
    freq_half = config.get_int(unique_name, "freq_half");

    // Fetch the buffer, register it
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Assuming half the band is available at each receiver node
    // TODO: use correct scheme
    for (int i = 0; i < 512; i ++) {
        int id = i + 512 * freq_half;
        freqs.push_back({freq_from_bin(id), (400.0 / 1024)});
    }

    // Assuming all inputs are included
    // TODO: this may not be the case. would need to read them from buffer
    inputs = std::get<1>(parse_reorder_default(config, unique_name));

    // open file for writing
    // TODO: need to deal with existing files
    file = std::unique_ptr<evFile>(
            new evFile(ev_fname, num_eigenvectors, ev_file_len,
                       freqs, inputs)
    );

}

writeEigenvec::~writeEigenvec() {

    file->flush();
    // TODO: delete file?
}

void writeEigenvec::apply_config(uint64_t fpga_seq) {

}

void writeEigenvec::main_thread() {

    unsigned int frame_id = 0;

    while (!stop_thread) {

        // Wait for the buffer to be filled with data
        if((wait_for_full_frame(in_buf, unique_name.c_str(),
                                frame_id)) == nullptr) {
            break;
        }

        // Get a view of the current frame
        auto frame = visFrameView(in_buf, frame_id);

        // Find the index of this frequency in the file
        // TODO: use correct scheme
        size_t freq_ind = frame.freq_id() - freq_half * 512;
        
        auto ftime = frame.time();
        time_ctype t = {std::get<0>(ftime), ts_to_double(std::get<1>(ftime))};

        file->write_eigenvectors(t, freq_ind,
                                 frame.eigenvectors(), frame.eigenvalues());

        // Mark the buffer and move on
        mark_frame_empty(in_buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;

    }
}


// TODO: include additional data:
//          - sum of remaining eigenvalues
//          - weights use in decomposition
evFile::evFile(const std::string & fname,
               const uint16_t & num_eigenvectors,
               const size_t & file_len,
               const std::vector<freq_ctype> & freqs,
               const std::vector<input_ctype> & inputs) {

    ninput = inputs.size();
    nfreq = freqs.size();
    ntimes = file_len;
    nev = num_eigenvectors;

    // Create file
    INFO("Creating new eigenvectors file %s", fname.c_str());
    file = std::unique_ptr<File>(
            new File(fname, File::SwmrCreate)
    );

    // Create eigenvector dataset
    std::vector<size_t> ev_dims = {ntimes, nfreq, nev, ninput};
    std::vector<std::string> ev_axes = {"time", "freq", "eigenmode", "input"};
    DataSpace ev_space = DataSpace(ev_dims);
    DataSet ev = file->createDataSet(
            "eigenvector", ev_space, create_datatype<std::complex<float>>()
    );
    ev.createAttribute<std::string>(
            "axis", DataSpace::From(ev_axes)
    ).write(ev_axes);

    // Create eigenvalue dataset
    std::vector<size_t> eval_dims = {ntimes, nfreq, nev};
    std::vector<std::string> eval_axes = {"time", "freq", "eigenmode"};
    DataSpace eval_space = DataSpace(eval_dims);
    DataSet eval = file->createDataSet(
            "eigenvalue", eval_space, create_datatype<float>()
    );
    eval.createAttribute<std::string>(
            "axis", DataSpace::From(eval_axes)
    ).write(eval_axes);

    // Create index map
    Group indexmap = file->createGroup("index_map");

    DataSet time_imap = indexmap.createDataSet(
      "time", DataSpace(ntimes), create_datatype<time_ctype>()
    );

    DataSet freq_imap = indexmap.createDataSet<freq_ctype>(
            "freq", DataSpace(nfreq)
    );
    freq_imap.write(freqs);

    DataSet input_imap = indexmap.createDataSet<input_ctype>(
            "input", DataSpace(ninput)
    );
    input_imap.write(inputs);

    // Start writing at top of file
    eof_ind = 0;

}

evFile::~evFile() {
    file->flush();
}

void evFile::flush() {
    file->flush();
}

void evFile::write_eigenvectors(time_ctype new_time, uint32_t freq_ind,
                          std::complex<float> * eigenvectors,
                          float * eigenvalues) {

    // Find position in file
    uint64_t curr_time = new_time.fpga_count;
    size_t curr_ind;
    auto time_ind = std::find(curr_times.begin(), curr_times.end(), curr_time);
    // Increment current end of file position
    if (time_ind == curr_times.end()) {
        curr_ind = eof_ind;
        eof_ind = (eof_ind + 1) % ntimes;
        curr_times[curr_ind] = curr_time;
    } else {
        curr_ind = *time_ind;
    }

    // write eigenvectors
    // TODO: check dimensions work out
    evec().select(
            {curr_ind, freq_ind, 0, 0}, {1, 1, nev, ninput}
    ).write(eigenvectors);
    // write eigenvalues
    eval().select(
            {curr_ind, freq_ind, 0}, {1, 1, nev}
    ).write(eigenvalues);
    // write time
    time().select({curr_ind}, {1}).write(new_time);

}

DataSet evFile::evec() {
    return file->getDataSet("eigenvector");
}

DataSet evFile::eval() {
    return file->getDataSet("eigenvalue");
}

DataSet evFile::time() {
    return file->getDataSet("index_map/time");
}

DataSet evFile::freq() {
    return file->getDataSet("index_map/freq");
}

DataSet evFile::prod() {
    return file->getDataSet("index_map/prod");
}
