#include "hfbFileArchive.hpp"

#include "visFile.hpp" // for create_lockfile

#include "fmt.hpp" // for format, fmt

#include <algorithm>                   // for copy, min
#include <cstdint>                     // for uint32_t
#include <cstdio>                      // for remove
#include <highfive/H5Attribute.hpp>    // for Attribute, Attribute::write
#include <highfive/H5DataSet.hpp>      // for DataSet, AnnotateTraits::createAttribute, DataSet...
#include <highfive/H5DataSpace.hpp>    // for DataSpace::From, DataSpace, DataSpace::getDimensions
#include <highfive/H5DataType.hpp>     // for CompoundType, create_datatype, CompoundType::addM...
#include <highfive/H5Exception.hpp>    // for DataSpaceException, HDF5ErrMapper
#include <highfive/H5File.hpp>         // for File, NodeTraits::createGroup, File::flush, NodeT...
#include <highfive/H5Group.hpp>        // for Group
#include <highfive/H5Object.hpp>       // for hid_t
#include <highfive/H5PropertyList.hpp> // for H5Pcreate, H5Pset_chunk, H5Pset_filter, H5P_DATAS...
#include <highfive/H5Selection.hpp>    // for SliceTraits::select, Selection, SliceTraits::write
#include <numeric>                     // for iota
#include <stdexcept>                   // for invalid_argument
#include <tuple>                       // for make_tuple, get, tuple
#include <type_traits>                 // for __decay_and_strip<>::__type
#include <utility>                     // for move, pair

using namespace HighFive;


// Create an archive file for uncompressed products
hfbFileArchive::hfbFileArchive(const std::string& name,
                               const std::map<std::string, std::string>& metadata,
                               const std::vector<time_ctype>& times,
                               const std::vector<freq_ctype>& freqs,
                               const std::vector<uint32_t>& beams,
                               const std::vector<uint32_t>& subfreqs, std::vector<int> chunk_size,
                               const kotekan::logLevel log_level) {

    set_log_level(log_level);

    // Check axes and create file
    setup_file(name, metadata, times, freqs, beams, subfreqs, chunk_size);

    // Make datasets
    create_axes(times, freqs, beams, subfreqs);
    create_datasets();
}


// Create an archive file for baseline-stacked products
hfbFileArchive::hfbFileArchive(
    const std::string& name, const std::map<std::string, std::string>& metadata,
    const std::vector<time_ctype>& times, const std::vector<freq_ctype>& freqs,
    const std::vector<uint32_t>& beams, const std::vector<uint32_t>& subfreqs,
    const std::vector<stack_ctype>& stack, std::vector<rstack_ctype>& reverse_stack,
    std::vector<int> chunk_size, const kotekan::logLevel log_level) {

    set_log_level(log_level);

    // Check axes and create file
    setup_file(name, metadata, times, freqs, beams, subfreqs, chunk_size);
    // Different bound check for stacked data
    if (chunk[1] > (int)stack.size()) {
        chunk[1] = stack.size();
        INFO("hfbFileArchive: Chunk stack dimension greater than axes. Will use a smaller chunk.")
    }

    // Make datasets, for stacked data
    stacked = true;
    create_axes(times, freqs, beams, subfreqs, stack);
    create_datasets();

    // Write the reverse map of products to stack
    dset("reverse_map/stack").select({0}, {length("prod")}).write(reverse_stack.data());
}


void hfbFileArchive::setup_file(const std::string& name,
                                const std::map<std::string, std::string>& metadata,
                                const std::vector<time_ctype>& times,
                                const std::vector<freq_ctype>& freqs,
                                const std::vector<uint32_t>& subfreqs,
                                const std::vector<uint32_t>& beams,
                                std::vector<int> chunk_size) {

    std::string data_filename = fmt::format(fmt("{:s}.h5"), name);

    lock_filename = create_lockfile(data_filename);

    // Set HDF5 chunk size
    chunk = chunk_size;
    // Check chunk size
    // Check chunk size
    if (chunk[0] < 1 || chunk[1] < 1 || chunk[2] < 1)
        throw std::invalid_argument(fmt::format(fmt("hfbFileArchive: config: Chunk size needs to "
                                                    "be greater or equal to (1,1,1) (is ({:d},{:d},"
                                                    "{:d}))."),
                                                chunk[0], chunk[1], chunk[2]));
    if (chunk[0] > (int)freqs.size()) {

        INFO("hfbFileArchive: Chunk frequency ({}) dimension greater than axes ({}). Will use a smaller "
             "chunk.", chunk[0], freqs.size())
        chunk[0] = freqs.size();
    }
    if (chunk[2] > (int)times.size()) {

        INFO("hfbFileArchive: Chunk time ({}) dimension greater than axes({}). Will use a smaller chunk.", chunk[2], times.size())
        chunk[2] = times.size();
    }
    if (chunk[3] > (int)beams.size()) {

        INFO("hfbFileArchive: Chunk beam ({}) dimension greater than axes ({}). Will use a smaller chunk.", chunk[3], beams.size())
        chunk[3] = beams.size();
    }
    if (chunk[4] > (int)subfreqs.size()) {

        INFO("hfbFileArchive: Chunk sub-frequency ({}) dimension greater than axes ({}). Will use a smaller "
             "chunk.", chunk[4], subfreqs.size())
        chunk[4] = subfreqs.size();
    }

    INFO("Creating new archive file {:s}", name);

    file = std::unique_ptr<File>(
        new File(data_filename, File::ReadWrite | File::Create | File::Truncate));

    // Write out metadata into flle
    for (auto item : metadata) {
        file->createAttribute<std::string>(item.first, DataSpace::From(item.second))
            .write(item.second);
    }

    // Get weight type flag
    weight_type = metadata.at("weight_type");
}


template<typename T>
void hfbFileArchive::write_block(std::string name, size_t f_ind, size_t t_ind, size_t chunk_f,
                                 size_t chunk_t, const T* data) {
    size_t beam_last_dim = dset(name).getSpace().getDimensions().at(1);
    size_t subfreq_last_dim = dset(name).getSpace().getDimensions().at(2);
    DEBUG("writing {:d} freq, {:d} times, {:d} beams, {:d} sub-freq at ({:d}, 0, 0, {:d}). Data[0]: {}", chunk_f, chunk_t, beam_last_dim, subfreq_last_dim, f_ind, t_ind, data[0]);
    dset(name).select({f_ind, 0, 0, t_ind}, {chunk_f, beam_last_dim, subfreq_last_dim, chunk_t}).write(data);
}

// Instantiate for types that will get used to satisfy linker
template void hfbFileArchive::write_block<std::complex<float>>(std::string name, size_t f_ind,
                                                               size_t t_ind, size_t chunk_f,
                                                               size_t chunk_t,
                                                               std::complex<float> const*);
template void hfbFileArchive::write_block<float>(std::string name, size_t f_ind, size_t t_ind,
                                                 size_t chunk_f, size_t chunk_t, float const*);
template void hfbFileArchive::write_block<int>(std::string name, size_t f_ind, size_t t_ind,
                                               size_t chunk_f, size_t chunk_t, int const*);


//
// The following was adapted from hfbFileH5
//

hfbFileArchive::~hfbFileArchive() {
    file->flush();
    file.reset(nullptr);
    std::remove(lock_filename.c_str());
}

void hfbFileArchive::create_axes(const std::vector<time_ctype>& times,
                                 const std::vector<freq_ctype>& freqs,
                                 const std::vector<uint32_t>& beams,
                                 const std::vector<uint32_t>& subfreqs) {

    create_axis("freq", freqs);
    create_axis("time", times);
    create_axis("beam", beams);
    create_axis("subfreq", subfreqs);
}

void hfbFileArchive::create_axes(const std::vector<time_ctype>& times,
                                 const std::vector<freq_ctype>& freqs,
                                 const std::vector<uint32_t>& beams,
                                 const std::vector<uint32_t>& subfreqs,
                                 const std::vector<stack_ctype>& stack) {

    create_axes(times, freqs, beams, subfreqs);

    create_axis("stack", stack);
}

template<typename T>
void hfbFileArchive::create_axis(std::string name, const std::vector<T>& axis) {

    Group indexmap =
        file->exist("index_map") ? file->getGroup("index_map") : file->createGroup("index_map");

    DataSet index = indexmap.createDataSet<T>(name, DataSpace(axis.size()));
    index.write(axis);
}

void hfbFileArchive::create_datasets() {

    //Group flags = file->createGroup("flags");

    bool compress = true;
    bool no_compress = false;

    // Create transposed dataset shapes
    create_dataset("hfb", {"freq", "beam", "subfreq", "time"}, create_datatype<float>(), compress);

    if (stacked) {
        Group rev_map = file->createGroup("reverse_map");
        create_dataset("reverse_map/stack", {"prod"}, create_datatype<rstack_ctype>(), no_compress);
    }

    // Add weight type flag where gossec expects it
    //dset("hfb_weight")
    //    .createAttribute<std::string>("type", DataSpace::From(weight_type))
    //    .write(weight_type);

    file->flush();
}

void hfbFileArchive::create_dataset(const std::string& name, const std::vector<std::string>& axes,
                                    DataType type, const bool& compress) {

    // Mapping of axis names to sizes (start, chunk)
    std::map<std::string, std::tuple<size_t, size_t>> size_map;
    size_map["freq"] = std::make_tuple(length("freq"), chunk[0]);
    size_map["beam"] = std::make_tuple(length("beam"), chunk[3]);
    size_map["time"] = std::make_tuple(length("time"), chunk[2]);
    size_map["subfreq"] = std::make_tuple(length("subfreq"), chunk[4]);
    if (stacked)
        size_map["stack"] = std::make_tuple(length("stack"), chunk[1]);

    std::vector<size_t> cur_dims, max_dims, chunk_dims;

    for (auto axis : axes) {
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
        if (H5Pset_chunk(plist, int(chunk_dims.size()), &(real_chunk.at(0))) < 0) {
            HDF5ErrMapper::ToException<DataSpaceException>("Failed trying to create chunk.");
        }
        // Set bitshuffle compression filter
        if (H5Pset_filter(plist, H5Z_BITSHUFFLE, H5Z_FLAG_MANDATORY, BSHUF_CD.size(),
                          BSHUF_CD.data())
            < 0) {
            HDF5ErrMapper::ToException<DataSpaceException>(
                "Failed trying to set bishuffle filter.");
        }

        DataSet dset = file->createDataSet(name, space, type, plist);
        dset.createAttribute<std::string>("axis", DataSpace::From(axes)).write(axes);
    } else {
        DataSet dset = file->createDataSet(name, space, type, chunk_dims);
        dset.createAttribute<std::string>("axis", DataSpace::From(axes)).write(axes);
    }
}

// Quick functions for fetching datasets and dimensions
DataSet hfbFileArchive::dset(const std::string& name) {
    const std::string dset_name = name == "hfb_weight" ? "flags/hfb_weight" : name;
    return file->getDataSet(dset_name);
}

size_t hfbFileArchive::length(const std::string& axis_name) {
    return dset(fmt::format(fmt("index_map/{:s}"), axis_name)).getSpace().getDimensions()[0];
}


// TODO: these should be included from hfbFileH5
// Add support for all our custom types to HighFive
template<>
inline DataType HighFive::create_datatype<freq_ctype>() {
    CompoundType f;
    f.addMember("centre", H5T_IEEE_F64LE);
    f.addMember("width", H5T_IEEE_F64LE);
    f.autoCreate();
    return std::move(f);
}

template<>
inline DataType HighFive::create_datatype<time_ctype>() {
    CompoundType t;
    t.addMember("fpga_count", H5T_STD_U64LE);
    t.addMember("ctime", H5T_IEEE_F64LE);
    t.autoCreate();
    return std::move(t);
}

template<>
inline DataType HighFive::create_datatype<input_ctype>() {

    CompoundType i;
    hid_t s32 = H5Tcopy(H5T_C_S1);
    H5Tset_size(s32, 32);
    // AtomicType<char[32]> s32;
    i.addMember("chan_id", H5T_STD_U16LE, 0);
    i.addMember("correlator_input", s32, 2);
    i.manualCreate(34);

    return std::move(i);
}

template<>
inline DataType HighFive::create_datatype<prod_ctype>() {

    CompoundType p;
    p.addMember("input_a", H5T_STD_U16LE);
    p.addMember("input_b", H5T_STD_U16LE);
    p.autoCreate();
    return std::move(p);
}

template<>
inline DataType HighFive::create_datatype<cfloat>() {
    CompoundType c;
    c.addMember("r", H5T_IEEE_F32LE);
    c.addMember("i", H5T_IEEE_F32LE);
    c.autoCreate();
    return std::move(c);
}

template<>
inline DataType HighFive::create_datatype<rstack_ctype>() {
    CompoundType c;
    c.addMember("stack", H5T_STD_U32LE);
    c.addMember("conjugate", H5T_STD_U8LE);
    c.autoCreate();
    return std::move(c);
}

template<>
inline DataType HighFive::create_datatype<stack_ctype>() {
    CompoundType c;
    c.addMember("prod", H5T_STD_U32LE);
    c.addMember("conjugate", H5T_STD_U8LE);
    c.autoCreate();
    return std::move(c);
}
