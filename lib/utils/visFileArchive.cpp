#include "visFileArchive.hpp"
#include "errors.h"
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <libgen.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

using namespace HighFive;


// Bitshuffle parameters
H5Z_filter_t H5Z_BITSHUFFLE = 32008;
unsigned int BSHUF_H5_COMPRESS_LZ4 = 2;
unsigned int BSHUF_BLOCK = 0;  // let bitshuffle choose
const std::vector<unsigned int> BSHUF_CD = {BSHUF_BLOCK, BSHUF_H5_COMPRESS_LZ4};


// Create an archive file with times as input
visFileArchive::visFileArchive(const std::string& name,
                               const std::map<std::string, std::string>& metadata,
                               const std::vector<time_ctype>& times,
                               const std::vector<freq_ctype>& freqs,
                               const std::vector<input_ctype>& inputs,
                               const std::vector<prod_ctype>& prods,
                               size_t num_ev,
                               std::vector<int> chunk_size) {

    std::string data_filename = name + ".h5";

    lock_filename = create_lockfile(data_filename);

    // Determine whether to write the eigensector or not...
    write_ev = (num_ev > 0);

    // Set HDF5 chunk size
    chunk = chunk_size;
    // Check chunk size
	// Check chunk size
    if (chunk[0] < 1 || chunk[1] < 1 || chunk[2] < 1)
        throw std::invalid_argument("visRawReader: config: Chunk size " \
            "needs to be greater or equal to (1,1,1) (is ("
            + std::to_string(chunk[0]) + ","
            + std::to_string(chunk[1]) + ","
            + std::to_string(chunk[2]) + ")).");
    if (chunk[0] > (int)freqs.size()) {
        chunk[0] = freqs.size();
        INFO("visFileArchive: Chunk frequency dimension greater than axes. Will use a smaller chunk.")
    }
    if (chunk[1] > (int)prods.size()) {
        chunk[1] = prods.size();
        INFO("visFileArchive: Chunk product dimension greater than axes. Will use a smaller chunk.")
    }
    if (chunk[2] > (int)times.size()) {
        chunk[2] = times.size();
        INFO("visFileArchive: Chunk time dimension greater than axes. Will use a smaller chunk.")
    }

    INFO("Creating new archive file %s", name.c_str());

    file = std::unique_ptr<File>(
        new File(data_filename, File::ReadWrite | File::Create | File::Truncate)
    );
    create_axes(times, freqs, inputs, prods, num_ev);
    create_datasets();

    // Write out metadata into flle
    for (auto item : metadata) {
        file->createAttribute<std::string>(
            item.first, DataSpace::From(item.second)
        ).write(item.second);
    }

    // Add weight type flag where gossec expects it
    dset("vis_weight").createAttribute<std::string>(
        "type", DataSpace::From(metadata.at("weight_type"))
    ).write(metadata.at("weight_type"));

}


template<typename T>
void visFileArchive::write_block(std::string name, size_t f_ind, size_t t_ind, size_t chunk_f,
                                 size_t chunk_t, const T* data) {
    //DEBUG("writing %d freq, %d times, at (%d,%d).", chunk_f, chunk_t, f_ind, t_ind);
    if (name == "gain_exp") {
        dset(name).select({0, t_ind}, {length("input"), chunk_t}).write(data);
    } else if (name == "evec") {
        dset(name).select({f_ind, 0, 0, t_ind},
                          {chunk_f, length("ev"), length("input"), chunk_t}).write(data);
    } else if (name == "erms") {
        dset(name).select({f_ind, t_ind}, {chunk_f, chunk_t}).write(data);
    } else {
        size_t last_dim = dset(name).getSpace().getDimensions().at(1);
        dset(name).select({f_ind, 0, t_ind}, {chunk_f, last_dim, chunk_t}).write(data);
    }
}

// Instantiate for types that will get used to satisfy linker
template void visFileArchive::write_block<std::complex<float>>(std::string name, size_t f_ind, size_t t_ind, size_t chunk_f, size_t chunk_t, std::complex<float> const*);
template void visFileArchive::write_block<float>(std::string name, size_t f_ind, size_t t_ind, size_t chunk_f, size_t chunk_t, float const*);
template void visFileArchive::write_block<int>(std::string name, size_t f_ind, size_t t_ind, size_t chunk_f, size_t chunk_t, int const*);


//
// The following was adapted from visFileH5
//

visFileArchive::~visFileArchive() {
    file->flush();
    file.reset(nullptr);
    std::remove(lock_filename.c_str());
}

void visFileArchive::create_axes(const std::vector<time_ctype>& times,
                            const std::vector<freq_ctype>& freqs,
                            const std::vector<input_ctype>& inputs,
                            const std::vector<prod_ctype>& prods,
                            size_t num_ev) {

    create_axis("freq", freqs);
    create_axis("input", inputs);
    create_axis("prod", prods);
    create_axis("time", times);

    if(write_ev) {
        std::vector<uint32_t> ev_vector(num_ev);
        std::iota(ev_vector.begin(), ev_vector.end(), 0);
        create_axis("ev", ev_vector);
    }
}

template<typename T>
void visFileArchive::create_axis(std::string name, const std::vector<T>& axis) {

    Group indexmap = file->exist("index_map") ?
                     file->getGroup("index_map") :
                     file->createGroup("index_map");

    DataSet index = indexmap.createDataSet<T>(name, DataSpace(axis.size()));
    index.write(axis);
}

void visFileArchive::create_datasets() {

    Group flags = file->createGroup("flags");

    bool compress = true;
    bool no_compress = false;

    // Create transposed dataset shapes
    create_dataset("vis", {"freq", "prod", "time"}, create_datatype<cfloat>(), compress);
    create_dataset("flags/vis_weight", {"freq", "prod", "time"},
                   create_datatype<float>(), compress);
    create_dataset("gain_coeff", {"freq", "input", "time"},
                   create_datatype<cfloat>(), compress);
    create_dataset("gain_exp", {"input", "time"}, create_datatype<int>(), no_compress);

    // Only write the eigenvector datasets if there's going to be anything in them
    if(write_ev) {
        create_dataset("eval", {"freq", "ev", "time"},
                       create_datatype<float>(), no_compress);
        create_dataset("evec", {"freq", "ev", "input", "time"},
                       create_datatype<cfloat>(), compress);
        create_dataset("erms", {"freq", "time"},
                       create_datatype<float>(), no_compress); 
    }

    file->flush();

}

void visFileArchive::create_dataset(const std::string& name, const std::vector<std::string>& axes,
                               DataType type, const bool& compress) {

    // Mapping of axis names to sizes (start, chunk)
    std::map<std::string, std::tuple<size_t, size_t>> size_map;
    size_map["freq"] = std::make_tuple(length("freq"), chunk[0]);
    size_map["input"] = std::make_tuple(length("input"),
            std::min((size_t)(chunk[1]), length("input")));
    size_map["prod"] = std::make_tuple(length("prod"), chunk[1]);
    size_map["ev"] = std::make_tuple(length("ev"), length("ev"));
    size_map["time"] = std::make_tuple(length("time"), chunk[2]);

    std::vector<size_t> cur_dims, max_dims, chunk_dims;

    for(auto axis : axes) {
        auto cs = size_map[axis];
        cur_dims.push_back(std::get<0>(cs));
        chunk_dims.push_back(std::get<1>(cs));
    }

    DataSpace space = DataSpace(cur_dims);

    if (compress) {
        // Add chunking and bitshuffle filter to plist
        // Pulled this out of HighFive createDataSet source
        std::vector<hsize_t> real_chunk(chunk_dims.size());
        std::copy(chunk_dims.begin(), chunk_dims.end(), real_chunk.begin());
        // Set dataset creation properties to enable chunking
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        if(H5Pset_chunk(plist, int(chunk_dims.size()), &(real_chunk.at(0))) < 0) {
            HDF5ErrMapper::ToException<DataSpaceException>("Failed trying to create chunk.");
        }
        // Set bitshuffle compression filter
        if(H5Pset_filter(plist, H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY,
                    BSHUF_CD.size(), BSHUF_CD.data()) < 0) {
            HDF5ErrMapper::ToException<DataSpaceException>("Failed trying to set bishuffle filter.");
        }

        DataSet dset = file->createDataSet(name, space, type, plist);
        dset.createAttribute<std::string>(
            "axis", DataSpace::From(axes)).write(axes);
    } else {
        DataSet dset = file->createDataSet(name, space, type, chunk_dims);
        dset.createAttribute<std::string>(
            "axis", DataSpace::From(axes)).write(axes);
    }

}

// Quick functions for fetching datasets and dimensions
DataSet visFileArchive::dset(const std::string& name) {
    const std::string dset_name = name == "vis_weight" ? "flags/vis_weight" : name;
    return file->getDataSet(dset_name);
}

size_t visFileArchive::length(const std::string& axis_name) {
    if(!write_ev && axis_name == "ev") return 0;
    return dset("index_map/" + axis_name).getSpace().getDimensions()[0];
}


// TODO: these should be included from visFileH5
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
