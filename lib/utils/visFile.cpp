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
                 const std::string& weights_type,
                 const std::vector<freq_ctype>& freqs,
                 const std::vector<input_ctype>& inputs,
                 const std::vector<prod_ctype>& prods,
                 size_t num_ev, size_t num_time) {

    std::string data_filename = root_path + "/" + acq_name + "/" + name;

    // Create the lock file first such that there is no time the file is
    // unlocked
    lock_filename = root_path + "/" + acq_name + "/." + name + ".lock";
    std::ofstream lock_file(lock_filename);
    lock_file << getpid() << std::endl;
    lock_file.close();

    // Determine whether to write the eigensector or not...
    write_ev = (num_ev > 0);
    size_t ninput = inputs.size();

    INFO("Creating new output file %s", name.c_str());

    file = std::unique_ptr<File>(
        new File(data_filename, File::ReadWrite | File::Create | File::Truncate)
    );

    create_axes(freqs, inputs, prods, num_ev, num_time);
    create_datasets(freqs.size(), ninput, prods.size(), num_ev, weights_type);

    dset("vis_weight").createAttribute<std::string>(
        "type", DataSpace::From(weights_type)).write(weights_type);

    // === Set the required attributes for a valid file ===
    std::string version = "NT_3.1.0";
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

void visFile::create_axes(const std::vector<freq_ctype>& freqs,
                          const std::vector<input_ctype>& inputs,
                          const std::vector<prod_ctype>& prods,
                          size_t num_ev, size_t num_time = 0) {

    create_time_axis(num_time);

    // Create and fill other axes
    create_axis("freq", freqs);
    create_axis("input", inputs);
    create_axis("prod", prods);

    if(write_ev) {
        std::vector<uint32_t> ev_vector(num_ev);
        std::iota(ev_vector.begin(), ev_vector.end(), 0);
        create_axis("ev", ev_vector);
    }
}

template<typename T>
void visFile::create_axis(std::string name, const std::vector<T>& axis) {

    Group indexmap = file->exist("index_map") ?
                     file->getGroup("index_map") :
                     file->createGroup("index_map");
    
    DataSet index = indexmap.createDataSet<T>(name, DataSpace(axis.size()));
    index.write(axis);
}

void visFile::create_time_axis(size_t num_time) {

    Group indexmap = file->exist("index_map") ?
                     file->getGroup("index_map") :
                     file->createGroup("index_map");
    
    DataSet time_axis = indexmap.createDataSet(
      "time", DataSpace({0}, {num_time}),
      create_datatype<time_ctype>(), std::vector<size_t>({1})
    );
}


void visFile::create_datasets(size_t nfreq, size_t ninput, size_t nprod,
                              size_t nev, std::string weights_type) {

    Group flags = file->createGroup("flags");

    create_dataset<cfloat>("vis", {"time", "freq", "prod"});
    create_dataset<float>("flags/vis_weight", {"time", "freq", "prod"});
    create_dataset<cfloat>("gain_coeff", {"time", "freq", "input"});
    create_dataset<int>("gain_exp", {"time", "input"});

    // Only write the eigenvector datasets if there's going to be anything in
    // them
    if(write_ev) {
        create_dataset<float>("eval", {"time", "freq", "ev"});
        create_dataset<cfloat>("evec", {"time", "freq", "ev", "input"});
        create_dataset<float>("erms", {"time", "freq"}); 
    }

    file->flush();

}

template<typename T>
void visFile::create_dataset(const std::string& name, const std::vector<std::string>& axes) {

    size_t max_time = dset("index_map/time").getSpace().getMaxDimensions()[0];

    // Mapping of axis names to sizes (start, max, chunk)
    std::map<std::string, std::tuple<size_t, size_t, size_t>> size_map;
    size_map["freq"] = std::make_tuple(length("freq"), length("freq"), 1);
    size_map["input"] = std::make_tuple(length("input"), length("input"), length("input"));
    size_map["prod"] = std::make_tuple(length("prod"), length("prod"), length("prod"));
    size_map["ev"] = std::make_tuple(length("ev"), length("ev"), length("ev"));
    size_map["time"] = std::make_tuple(0, max_time, 1);

    std::vector<size_t> cur_dims, max_dims, chunk_dims;

    for(auto axis : axes) {
        auto cs = size_map[axis];
        cur_dims.push_back(std::get<0>(cs));
        max_dims.push_back(std::get<1>(cs));
        chunk_dims.push_back(std::get<2>(cs));
    }
    
    DataSpace space = DataSpace(cur_dims, max_dims);
    DataSet dset = file->createDataSet(
        name, space, create_datatype<T>(), max_dims
    );
    dset.createAttribute<std::string>(
        "axis", DataSpace::From(axes)).write(axes);
}

// Quick functions for fetching datasets and dimensions
DataSet visFile::dset(const std::string& name) {
    const std::string dset_name = name == "vis_weight" ? "flags/vis_weight" : name;
    return file->getDataSet(dset_name);
}

size_t visFile::length(const std::string& axis_name) {
    if(!write_ev && axis_name == "ev") return 0;
    return dset("index_map/" + axis_name).getSpace().getDimensions()[0];
}

size_t visFile::num_time() {
    return length("time");
}


uint32_t visFile::extend_time(time_ctype new_time) {

    // Get the current dimensions
    size_t ntime = length("time"), nprod = length("prod"),
           ninput = length("input"), nfreq = length("freq"),
           nev = length("ev");

    INFO("Current size: %zd; new size: %zd", ntime, ntime + 1);
    // Add a new entry to the time axis
    ntime++;
    dset("index_map/time").resize({ntime});
    dset("index_map/time").select({ntime - 1}, {1}).write(&new_time);

    // Extend all other datasets
    dset("vis").resize({ntime, nfreq, nprod});
    dset("vis_weight").resize({ntime, nfreq, nprod});
    dset("gain_coeff").resize({ntime, nfreq, ninput});
    dset("gain_exp").resize({ntime, ninput});

    if(write_ev) {
        dset("eval").resize({ntime, nfreq, nev});
        dset("evec").resize({ntime, nfreq, nev, ninput});
        dset("erms").resize({ntime, nfreq});
    }

    // Flush the changes
    file->flush();

    return ntime - 1;
}


void visFile::write_sample(
    uint32_t time_ind, uint32_t freq_ind, std::vector<cfloat> new_vis,
    std::vector<float> new_weight, std::vector<cfloat> new_gcoeff,
    std::vector<int32_t> new_gexp, std::vector<float> new_eval,
    std::vector<cfloat> new_evec, float new_erms
) {

    // Get the current dimensions
    size_t nprod = length("prod"), ninput = length("input"), nev = length("ev");

    dset("vis").select({time_ind, freq_ind, 0}, {1, 1, nprod}).write(new_vis);
    dset("vis_weight").select({time_ind, freq_ind, 0}, {1, 1, nprod}).write(new_weight);
    dset("gain_coeff").select({time_ind, freq_ind, 0}, {1, 1, ninput}).write(new_gcoeff);
    dset("gain_exp").select({time_ind, 0}, {1, ninput}).write(new_gexp);

    if(write_ev) {
        dset("eval").select({time_ind, freq_ind, 0}, {1, 1, nev}).write(new_eval);
        dset("evec").select({time_ind, freq_ind, 0, 0}, {1, 1, nev, ninput}).write((const cfloat *)new_evec.data());
        dset("erms").select({time_ind, freq_ind}, {1, 1}).write(new_erms);
    }

    file->flush();
}


bool visFileBundle::resolve_sample(time_ctype new_time) {

    uint64_t count = new_time.fpga_count;

    if(vis_file_map.size() == 0) {
        // If no files are currently in the map we should create a new one.
        add_file(new_time);
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
            return false;
        }

        if(count > max_fpga) {
            // We've got a later time and so we need to add a new time sample,
            // if the current file does not need to rollover register the new
            // sample as being in the last file, otherwise create a new file
            std::shared_ptr<visFile> file;
            uint32_t ind;
            std::tie(file, ind) = vis_file_map.rbegin()->second;  // Unpack the last entry

            if(file->num_time() < rollover) {
                // Extend the time axis and add into the sample map
                ind = file->extend_time(new_time);
                vis_file_map[count] = std::make_tuple(file, ind);
            } else {
                add_file(new_time);
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
        return false;
    }

    return true;
}


void visFileBundle::add_file(time_ctype first_time) {

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
    auto file = mkFile(file_name, acq_name, root_path);
    auto ind = file->extend_time(first_time);
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
