#include "visFile.hpp"
#include "errors.h"
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

using namespace HighFive;

visFile::visFile(const std::string& name,
                 const std::string& acq_name,
                 const std::string& root_path,
                 const std::string& inst_name,
                 const std::string& notes,
                 const std::vector<freq_ctype>& freqs,
                 const std::vector<input_ctype>& inputs) {

    std::string data_filename = root_path + "/" + acq_name + "/" + name;

    // Create the lock file first such that there is no time the file is
    // unlocked
    lock_filename = root_path + "/" + acq_name + "/." + name + ".lock";
    std::ofstream lock_file(lock_filename);
    lock_file << getpid() << std::endl;
    lock_file.close();

    size_t ninput = inputs.size();

    INFO("Creating new output file %s", name.c_str());

    file = std::unique_ptr<File>(
        new File(data_filename, File::ReadWrite | File::Create | File::Truncate)
    );

    createIndex(freqs, inputs);
    createDatasets(freqs.size(), ninput, ninput * (ninput + 1) / 2);

    // === Set the required attributes for a valid file ===
    std::string version = "NT_3.0.0";
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
        "vis", vis_space, create_datatype<cfloat>(), vis_dims
    );
    vis.createAttribute<std::string>(
        "axis", DataSpace::From(vis_axes)).write(vis_axes);


    Group flags = file->createGroup("flags");
    DataSet vis_weight = flags.createDataSet(
        "vis_weight", vis_space, create_datatype<float>(), vis_dims
    );
    vis_weight.createAttribute<std::string>(
        "axis", DataSpace::From(vis_axes)).write(vis_axes);


    DataSet gain_coeff = file->createDataSet(
        "gain_coeff", gain_space, create_datatype<cfloat>(), gain_dims
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

// Quick functions for fetching datasets and dimensions
DataSet visFile::vis() {
    return file->getDataSet("vis");
}

DataSet visFile::vis_weight() {
    return file->getDataSet("flags/vis_weight");
}

DataSet visFile::gain_coeff() {
    return file->getDataSet("gain_coeff");
}

DataSet visFile::gain_exp() {
    return file->getDataSet("gain_exp");
}

DataSet visFile::time() {
    return file->getDataSet("index_map/time");
}

size_t visFile::num_time() {
    return time().getSpace().getDimensions()[0];
}

size_t visFile::num_prod() {
    return vis().getSpace().getDimensions()[2];
}

size_t visFile::num_freq() {
    return vis().getSpace().getDimensions()[1];
}

size_t visFile::num_input() {
    return gain_exp().getSpace().getDimensions()[1];
}


uint32_t visFile::extendTime(time_ctype new_time) {

    // Get the current dimensions
    size_t ntime = num_time(), nprod = num_prod(),
           ninput = num_input(), nfreq = num_freq();

    INFO("Current size: %zd; new size: %zd", ntime, ntime + 1);
    // Add a new entry to the time axis
    ntime++;
    time().resize({ntime});
    time().select({ntime - 1}, {1}).write(&new_time);

    // Extend all other datasets
    vis().resize({ntime, nfreq, nprod});
    vis_weight().resize({ntime, nfreq, nprod});
    gain_coeff().resize({ntime, nfreq, ninput});
    gain_exp().resize({ntime, ninput});

    // Flush the changes
    file->flush();

    return ntime - 1;
}


void visFile::writeSample(
    uint32_t time_ind, uint32_t freq_ind, std::vector<cfloat> new_vis,
    std::vector<float> new_weight, std::vector<cfloat> new_gcoeff,
    std::vector<int32_t> new_gexp
) {

    // Get the current dimensions
    size_t nprod = num_prod(), ninput = num_input();

    vis().select({time_ind, freq_ind, 0}, {1, 1, nprod}).write(new_vis);
    vis_weight().select({time_ind, freq_ind, 0}, {1, 1, nprod}).write(new_weight);
    gain_coeff().select({time_ind, freq_ind, 0}, {1, 1, ninput}).write(new_gcoeff);
    gain_exp().select({time_ind, 0}, {1, ninput}).write(new_gexp);

    file->flush();
}


size_t visFile::addSample(
    time_ctype new_time, uint32_t freq_ind, std::vector<cfloat> new_vis,
    std::vector<float> new_weight, std::vector<cfloat> new_gcoeff,
    std::vector<int32_t> new_gexp
) {

    size_t ntime = num_time();
    uint32_t time_ind = ntime - 1;

    // Get the latest time in the file
    time_ctype last_time;

    if(ntime > 0) {
        time().select({time_ind}, {1}).read(&last_time);
    }

    // If we haven't seen the new time add it to the time axis and extend the time
    // dependent datasets
    if(ntime == 0 || new_time.fpga_count > last_time.fpga_count) {
        time_ind = extendTime(new_time);
        ntime++;
    }

    writeSample(time_ind, freq_ind, new_vis, new_weight, new_gcoeff, new_gexp);
    return ntime;
}


visFileBundle::visFileBundle(const std::string root_path,
                             int freq_chunk,
                             const std::string instrument_name,
                             const std::string notes,
                             const std::vector<freq_ctype>& freqs,
                             const std::vector<input_ctype>& inputs,
                             size_t rollover, size_t window_size) :

    root_path(root_path),
    freq_chunk(freq_chunk),
    instrument_name(instrument_name),
    notes(notes),
    freqs(freqs),
    inputs(inputs),
    rollover(rollover),
    window_size(window_size)

{


}


void visFileBundle::addSample(time_ctype new_time, uint32_t freq_ind,
                              std::vector<cfloat> new_vis,
                              std::vector<float> new_weight,
                              std::vector<cfloat> new_gcoeff,
                              std::vector<int32_t> new_gexp) {

    std::shared_ptr<visFile> file;
    uint32_t ind;

    uint64_t count = new_time.fpga_count;

    if(vis_file_map.size() == 0) {
        // If no files are currently in the map we should create a new one.
        addFile(new_time);
    } else {
        // If there are files in the list we need to figure out whether to
        // insert a new entry or not
        uint64_t max_fpga = vis_file_map.rbegin()->first;
        uint64_t min_fpga = vis_file_map.begin()->first;

        if(count < min_fpga) {
            // This data is older that anything else in the map so we should just drop it
            INFO("Dropping integration as buffer (FPGA count: %" PRIu64
                 ") arrived too late (minimum in pool %" PRIu64 ")",
                 new_time.fpga_count, min_fpga);
            return;
        }

        if(count > max_fpga) {
            // We've got a later time and so we need to add a new time sample,
            // if the current file does not need to rollover register the new
            // sample as being in the last file, otherwise create a new file
            std::tie(file, ind) = vis_file_map.rbegin()->second;  // Unpack the last entry

            if(file->num_time() < rollover) {
                // Extend the time axis and add into the sample map
                ind = file->extendTime(new_time);
                vis_file_map[count] = std::make_tuple(file, ind);
            } else {
                addFile(new_time);
            }

            // As we've added a new sample we need to delete the earliest sample
            if(vis_file_map.size() > window_size) {
                vis_file_map.erase(vis_file_map.begin());
            }
        }
    }

    if(vis_file_map.find(count) == vis_file_map.end()) {
        // This is slightly subtle, but if a sample was not found at this point
        // then it must lie within the range, but not have been saved into the
        // files already. This means that adding it would make the files time
        // axis be out of order, so we just skip it for now.
        INFO("Skipping integration (FPGA count %" PRIu64
             ") as it would be written out of order.", count);
        return;
    }

    // We can now safely add the sample into the file
    std::tie(file, ind) = vis_file_map[count];
    file->writeSample(ind, freq_ind, new_vis, new_weight,
                      new_gcoeff, new_gexp);

}

void visFileBundle::addFile(time_ctype first_time) {

    time_t t = (time_t)first_time.ctime;

    // Start the acq and create the directory if required
    if(acq_name.empty()) {
        // Format the time (annoyingly you still have to use streams for this)
        std::ostringstream s;
        s << std::put_time(std::gmtime(&t), "%Y%m%dT%H%M%SZ");
        // Set the acq name
        acq_name = s.str() + "_" + instrument_name + "_corr";

        // Set the acq fields on the instance
        acq_start_time = first_time.ctime;

        // Create acquisition directory. Don't bother checking if it already exists, just let it transparently fail
        mkdir((root_path + "/" + acq_name).c_str(), 0755);
    }

    // Construct the name of the new file
    char fname_temp[100];
    snprintf(
        fname_temp, sizeof(fname_temp), "%08d_%04d.h5",
        (unsigned int)(first_time.ctime - acq_start_time), freq_chunk
    );
    std::string file_name = fname_temp;

    // Create the file, create room for the first sample and add into the file map
    auto file = std::make_shared<visFile>(
        file_name, acq_name, root_path, instrument_name, "", freqs, inputs
    );
    auto ind = file->extendTime(first_time);
    vis_file_map[first_time.fpga_count] = std::make_tuple(file, ind);
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

template <> inline DataType HighFive::create_datatype<cfloat>() {
    CompoundType c;
    c.addMember("r", H5T_IEEE_F32LE);
    c.addMember("i", H5T_IEEE_F32LE);
    c.autoCreate();
    return c;
}
