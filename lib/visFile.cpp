#include "visFile.hpp"
#include "errors.h"
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

using namespace HighFive;

visFile::visFile(const std::string& name,
                 const std::string& acq_name,
                 const std::string& inst_name,
                 const std::string& notes,
                 const std::vector<freq_ctype>& freqs,
                 const std::vector<input_ctype>& inputs) {

    // Create the lock file first such that there is no time the file is
    // unlocked
    lock_filename = name + ".lock";
    std::ofstream lock_file(lock_filename);
    lock_file << getpid() << std::endl;
    lock_file.close();

    size_t ninput = inputs.size();

    INFO("Creating new output file %s", name.c_str());

    file = std::unique_ptr<File>(
        new File(name, File::ReadWrite | File::Create | File::Truncate)
    );

    createIndex(freqs, inputs);
    createDatasets(freqs.size(), ninput, ninput * (ninput + 1) / 2);

    // === Set the required attributes for a valid file ===
    std::string version = "NT_2.4.0";
    file->createAttribute<std::string>(
        "archive_version", DataSpace::From(version)).write(version);
    file->createAttribute<std::string>(
        "acquisition_name", DataSpace::From(acq_name)).write(acq_name);
    file->createAttribute<std::string>(
        "instrument_name", DataSpace::From(inst_name)).write(inst_name);

    // TODO: get git version tag somehow
    std::string git_version = "not set";
    file->createAttribute<std::string>(
        "git_version_tag", DataSpace::From(git_version)).write(git_version);

    file->createAttribute<std::string>(
        "notes", DataSpace::From(notes)).write(notes);

    char temp[256];
    std::string username = (getlogin_r(temp, 256) == 0) ? temp : "unknown";
    file->createAttribute<std::string>(
        "system_user", DataSpace::From(username)).write(username);

    gethostname(temp, 256);
    std::string hostname = temp;
    file->createAttribute<std::string>(
        "collection_server", DataSpace::From(hostname)).write(hostname);
}

visFile::~visFile() {

    file->flush();
    file.reset(nullptr);
    std::remove(lock_filename.c_str());
}

void visFile::createIndex(const std::vector<freq_ctype>& freqs,
                          const std::vector<input_ctype>& inputs) {

    Group indexmap = file->createGroup("index_map");

    DataSet time_imap = indexmap.createDataSet(
      "time", DataSpace({0}, {DataSpace::UNLIMITED}),
      create_datatype<time_ctype>(), std::vector<size_t>({1})
    );

    // Create and fill frequency dataset
    DataSet freq_imap = indexmap.createDataSet<freq_ctype>("freq", DataSpace(freqs.size()));
    freq_imap.write(freqs);


    DataSet input_imap = indexmap.createDataSet<input_ctype>("input", DataSpace(inputs.size()));
    input_imap.write(inputs);

    std::vector<prod_ctype> prod_vector;
    for(uint16_t i=0; i < inputs.size(); i++) {
        for(uint16_t j = i; j < inputs.size(); j++) {
            prod_vector.push_back({i, j});
        }
    }
    DataSet prod_imap = indexmap.createDataSet<prod_ctype>(
        "prod", DataSpace(prod_vector.size())
    );
    prod_imap.write(prod_vector);

    file->flush();

}

void visFile::createDatasets(size_t nfreq, size_t ninput, size_t nprod) {

    // Create extensible spaces for the different types of spaces we have
    DataSpace vis_space = DataSpace({0, nfreq, nprod},
                                    {DataSpace::UNLIMITED, nfreq, nprod});
    DataSpace gain_space = DataSpace({0, nfreq, ninput},
                                    {DataSpace::UNLIMITED, nfreq, ninput});
    DataSpace exp_space = DataSpace({0, ninput},
                                    {DataSpace::UNLIMITED, ninput});

    std::vector<std::string> vis_axes = {"time", "freq", "prod"};
    std::vector<std::string> gain_axes = {"time", "freq", "input"};
    std::vector<std::string> exp_axes = {"time", "input"};

    std::vector<size_t> vis_dims = {1, 1, nprod};
    std::vector<size_t> gain_dims = {1, 1, ninput};
    std::vector<size_t> exp_dims = {1, ninput};


    DataSet vis = file->createDataSet(
        "vis", vis_space, create_datatype<complex_int>(), vis_dims
    );
    vis.createAttribute<std::string>(
        "axis", DataSpace::From(vis_axes)).write(vis_axes);


    Group flags = file->createGroup("flags");
    DataSet vis_weight = flags.createDataSet(
        "vis_weight", vis_space, create_datatype<unsigned char>(), vis_dims
    );
    vis_weight.createAttribute<std::string>(
        "axis", DataSpace::From(vis_axes)).write(vis_axes);


    DataSet gain_coeff = file->createDataSet(
        "gain_coeff", gain_space, create_datatype<complex_int>(), gain_dims
    );
    gain_coeff.createAttribute<std::string>(
        "axis", DataSpace::From(gain_axes)).write(gain_axes);


    DataSet gain_exp = file->createDataSet(
        "gain_exp", exp_space, create_datatype<int>(), exp_dims
    );
    gain_exp.createAttribute<std::string>(
        "axis", DataSpace::From(exp_axes)).write(exp_axes);


    file->flush();

}


size_t visFile::addSample(
    time_ctype new_time, uint32_t freq_ind, std::vector<complex_int> new_vis,
    std::vector<uint8_t> new_weight, std::vector<complex_int> new_gcoeff,
    std::vector<int32_t> new_gexp
) {

    // TODO: extend this routine such that it can insert frequencies into
    // previous time samples

    DataSet time_imap = file->getDataSet("index_map/time");
    DataSet vis = file->getDataSet("vis");
    DataSet vis_weight = file->getDataSet("flags/vis_weight");
    DataSet gain_coeff = file->getDataSet("gain_coeff");
    DataSet gain_exp = file->getDataSet("gain_exp");

    // Get size of dimensions
    std::vector<size_t> dims = vis.getSpace().getDimensions();
    size_t ntime = dims[0], nfreq = dims[1], nprod = dims[2];
    dims = gain_coeff.getSpace().getDimensions();
    size_t ninput = dims[2];

    uint32_t time_ind = ntime - 1;

    // Get the latest time in the file
    time_ctype last_time;

    if(ntime > 0) {
        time_imap.select({time_ind}, {1}).read(&last_time);
    }

    // If we haven't seen the new time add it to the time axis and extend the time
    // dependent datasets
    if(ntime == 0 || new_time.fpga_count > last_time.fpga_count) {
        INFO("Current size: %zd; new size: %zd", ntime, ntime + 1);

        // Add a new entry to the time axis
        ntime++; time_ind++;
        time_imap.resize({ntime});
        time_imap.select({time_ind}, {1}).write(&new_time);

        // Extend all other datasets
        vis.resize({ntime, nfreq, nprod});
        vis_weight.resize({ntime, nfreq, nprod});
        gain_coeff.resize({ntime, nfreq, ninput});
        gain_exp.resize({ntime, ninput});

    }

    vis.select({time_ind, freq_ind, 0}, {1, 1, nprod}).write(new_vis);
    vis_weight.select({time_ind, freq_ind, 0}, {1, 1, nprod}).write(new_weight);
    gain_coeff.select({time_ind, freq_ind, 0}, {1, 1, ninput}).write(new_gcoeff);
    gain_exp.select({time_ind, 0}, {1, ninput}).write(new_gexp);

    file->flush();

    return ntime;
}


// Initialise the serial from a std::string
input_ctype::input_ctype(uint16_t id, std::string serial) {
    chan_id = id;
    memset(correlator_input, 0, 32);
    serial.copy(correlator_input, 32);
}


// Add support for all our custom types to HighFive
template <> inline DataType HighFive::create_datatype<freq_ctype>() {
    CompoundType f;
    f.addMember("centre", H5T_IEEE_F64LE);
    f.addMember("width", H5T_IEEE_F64LE);
    f.autoCreate();
    return f;
}

template <> inline DataType HighFive::create_datatype<time_ctype>() {
    CompoundType t;
    t.addMember("fpga_count", H5T_STD_U64LE);
    t.addMember("ctime", H5T_IEEE_F64LE);
    t.autoCreate();
    return t;
}

template <> inline DataType HighFive::create_datatype<input_ctype>() {

    CompoundType i;
    hid_t s32 = H5Tcopy(H5T_C_S1);
    H5Tset_size(s32, 32);
    //AtomicType<char[32]> s32;
    i.addMember("chan_id", H5T_STD_U16LE, 0);
    i.addMember("correlator_input", s32, 2);
    i.manualCreate(34);

    return i;
}

template <> inline DataType HighFive::create_datatype<prod_ctype>() {

    CompoundType p;
    p.addMember("input_a", H5T_STD_U16LE);
    p.addMember("input_b", H5T_STD_U16LE);
    p.autoCreate();
    return p;
}

template <> inline DataType HighFive::create_datatype<complex_int>() {
    CompoundType c;
    c.addMember("r", H5T_STD_I32LE);
    c.addMember("i", H5T_STD_I32LE);
    c.autoCreate();
    return c;
}
