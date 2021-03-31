#include "visFileArchive.hpp"

#include "H5Support.hpp" // for AtomicType<>::AtomicType, dset_id_str
#include "visFile.hpp"   // for create_lockfile

#include "fmt.hpp" // for format, fmt

#include <algorithm>                   // for copy, max, min
#include <cstdint>                     // for uint32_t
#include <cstdio>                      // for remove
#include <highfive/H5Attribute.hpp>    // for Attribute, Attribute::write
#include <highfive/H5DataSet.hpp>      // for DataSet, AnnotateTraits::createAttribute, DataSet...
#include <highfive/H5DataSpace.hpp>    // for DataSpace::From, DataSpace, DataSpace::DataSpace
#include <highfive/H5DataType.hpp>     // for CompoundType, create_datatype, CompoundType::addM...
#include <highfive/H5Exception.hpp>    // for DataSpaceException, HDF5ErrMapper
#include <highfive/H5File.hpp>         // for File, NodeTraits::createDataSet, NodeTraits::crea...
#include <highfive/H5Group.hpp>        // for Group
#include <highfive/H5Object.hpp>       // for HighFive
#include <highfive/H5PropertyList.hpp> // for H5Pcreate, H5Pset_chunk, H5Pset_filter, H5T_IEEE_...
#include <highfive/H5Selection.hpp>    // for SliceTraits::write, SliceTraits::select, Selection
#include <numeric>                     // for iota
#include <stdexcept>                   // for invalid_argument
#include <tuple>                       // for make_tuple, tuple, get
#include <type_traits>                 // for __decay_and_strip<>::__type, remove_reference<>::...
#include <utility>                     // for move, pair

using namespace HighFive;


// Create an archive file for uncompressed products
visFileArchive::visFileArchive(const std::string& name,
                               const std::map<std::string, std::string>& metadata,
                               const std::vector<time_ctype>& times,
                               const std::vector<freq_ctype>& freqs,
                               const std::vector<input_ctype>& inputs,
                               const std::vector<prod_ctype>& prods, size_t num_ev,
                               std::vector<int> chunk_size, const kotekan::logLevel log_level) {

    set_log_level(log_level);

    // Check axes and create file
    setup_file(name, metadata, times, freqs, prods, num_ev, chunk_size);

    // Make datasets
    create_axes(times, freqs, inputs, prods, num_ev);
    create_datasets();
}


// Create an archive file for baseline-stacked products
visFileArchive::visFileArchive(
    const std::string& name, const std::map<std::string, std::string>& metadata,
    const std::vector<time_ctype>& times, const std::vector<freq_ctype>& freqs,
    const std::vector<input_ctype>& inputs, const std::vector<prod_ctype>& prods,
    const std::vector<stack_ctype>& stack, std::vector<rstack_ctype>& reverse_stack, size_t num_ev,
    std::vector<int> chunk_size, const kotekan::logLevel log_level) {

    set_log_level(log_level);

    // Check axes and create file
    setup_file(name, metadata, times, freqs, prods, num_ev, chunk_size);
    // Different bound check for stacked data
    if (chunk[1] > (int)stack.size()) {
        chunk[1] = stack.size();
        INFO("visFileArchive: Chunk stack dimension greater than axes. Will use a smaller chunk.")
    }

    // Make datasets, for stacked data
    stacked = true;
    create_axes(times, freqs, inputs, prods, stack, num_ev);
    create_datasets();

    // Write the reverse map of products to stack
    dset("reverse_map/stack").select({0}, {length("prod")}).write(reverse_stack.data());
}


void visFileArchive::setup_file(const std::string& name,
                                const std::map<std::string, std::string>& metadata,
                                const std::vector<time_ctype>& times,
                                const std::vector<freq_ctype>& freqs,
                                const std::vector<prod_ctype>& prods, size_t num_ev,
                                std::vector<int> chunk_size) {

    std::string data_filename = fmt::format(fmt("{:s}.h5"), name);

    lock_filename = create_lockfile(data_filename);

    // Determine whether to write the eigensector or not...
    write_ev = (num_ev > 0);

    // Set HDF5 chunk size
    chunk = chunk_size;
    // Check chunk size
    // Check chunk size
    if (chunk[0] < 1 || chunk[1] < 1 || chunk[2] < 1)
        throw std::invalid_argument(fmt::format(fmt("visFileArchive: config: Chunk size needs to "
                                                    "be greater or equal to (1,1,1) (is ({:d},{:d},"
                                                    "{:d}))."),
                                                chunk[0], chunk[1], chunk[2]));
    if (chunk[0] > (int)freqs.size()) {
        chunk[0] = freqs.size();
        INFO("visFileArchive: Chunk frequency dimension greater than axes. Will use a smaller "
             "chunk.")
    }
    if (chunk[1] > (int)prods.size()) {
        chunk[1] = prods.size();
        INFO("visFileArchive: Chunk product dimension greater than axes. Will use a smaller chunk.")
    }
    if (chunk[2] > (int)times.size()) {
        chunk[2] = times.size();
        INFO("visFileArchive: Chunk time dimension greater than axes. Will use a smaller chunk.")
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
void visFileArchive::write_block(std::string name, size_t f_ind, size_t t_ind, size_t chunk_f,
                                 size_t chunk_t, const T* data) {
    // DEBUG("writing {:d} freq, {:d} times, at ({:d},{:d}).", chunk_f, chunk_t, f_ind, t_ind);
    if (name == "flags/inputs") {
        DEBUG2("writing {}...", name);
        dset(name).select({0, t_ind}, {length("input"), chunk_t}).write(data);
    } else if (name == "evec") {
        DEBUG2("writing {}...", name);
        dset(name)
            .select({f_ind, 0, 0, t_ind}, {chunk_f, length("ev"), length("input"), chunk_t})
            .write(data);
    } else if (name == "erms" || name == "flags/frac_lost" || name == "flags/frac_rfi"
               || name == "flags/dataset_id") {
        DEBUG2("writing {}...", name);
        dset(name).select({f_ind, t_ind}, {chunk_f, chunk_t}).write(data);
    } else {
        DEBUG2("writing {}...", name);
        size_t last_dim = dset(name).getSpace().getDimensions().at(1);
        dset(name).select({f_ind, 0, t_ind}, {chunk_f, last_dim, chunk_t}).write(data);
    }
}

// Instantiate for types that will get used to satisfy linker
template void visFileArchive::write_block<std::complex<float>>(std::string name, size_t f_ind,
                                                               size_t t_ind, size_t chunk_f,
                                                               size_t chunk_t,
                                                               std::complex<float> const*);
template void visFileArchive::write_block<float>(std::string name, size_t f_ind, size_t t_ind,
                                                 size_t chunk_f, size_t chunk_t, float const*);
template void visFileArchive::write_block<int>(std::string name, size_t f_ind, size_t t_ind,
                                               size_t chunk_f, size_t chunk_t, int const*);
template void visFileArchive::write_block<dset_id_str>(std::string name, size_t f_ind, size_t t_ind,
                                                       size_t chunk_f, size_t chunk_t,
                                                       dset_id_str const*);


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
                                 const std::vector<prod_ctype>& prods, size_t num_ev) {

    create_axis("freq", freqs);
    create_axis("input", inputs);
    create_axis("prod", prods);
    create_axis("time", times);

    if (write_ev) {
        std::vector<uint32_t> ev_vector(num_ev);
        std::iota(ev_vector.begin(), ev_vector.end(), 0);
        create_axis("ev", ev_vector);
    }
}

void visFileArchive::create_axes(const std::vector<time_ctype>& times,
                                 const std::vector<freq_ctype>& freqs,
                                 const std::vector<input_ctype>& inputs,
                                 const std::vector<prod_ctype>& prods,
                                 const std::vector<stack_ctype>& stack, size_t num_ev) {

    create_axes(times, freqs, inputs, prods, num_ev);

    create_axis("stack", stack);
}

template<typename T>
void visFileArchive::create_axis(std::string name, const std::vector<T>& axis) {

    Group indexmap =
        file->exist("index_map") ? file->getGroup("index_map") : file->createGroup("index_map");

    DataSet index = indexmap.createDataSet<T>(name, DataSpace(axis.size()));
    index.write(axis);
}

void visFileArchive::create_datasets() {

    Group flags = file->createGroup("flags");

    bool compress = true;
    bool no_compress = false;

    // Create transposed dataset shapes
    create_dataset("vis", {"freq", prod_or_stack(), "time"}, create_datatype<cfloat>(), compress);
    create_dataset("flags/vis_weight", {"freq", prod_or_stack(), "time"}, create_datatype<float>(),
                   compress);
    create_dataset("flags/inputs", {"input", "time"}, create_datatype<float>(), no_compress);
    create_dataset("gain", {"freq", "input", "time"}, create_datatype<cfloat>(), compress);
    create_dataset("flags/frac_lost", {"freq", "time"}, create_datatype<float>(), no_compress);
    create_dataset("flags/frac_rfi", {"freq", "time"}, create_datatype<float>(), no_compress);
    create_dataset("flags/dataset_id", {"freq", "time"}, create_datatype<dset_id_str>(),
                   no_compress);

    // Only write the eigenvector datasets if there's going to be anything in them
    if (write_ev) {
        create_dataset("eval", {"freq", "ev", "time"}, create_datatype<float>(), no_compress);
        create_dataset("evec", {"freq", "ev", "input", "time"}, create_datatype<cfloat>(),
                       compress);
        create_dataset("erms", {"freq", "time"}, create_datatype<float>(), no_compress);
    }

    if (stacked) {
        Group rev_map = file->createGroup("reverse_map");
        create_dataset("reverse_map/stack", {"prod"}, create_datatype<rstack_ctype>(), no_compress);
    }

    // Add weight type flag where gossec expects it
    dset("vis_weight")
        .createAttribute<std::string>("type", DataSpace::From(weight_type))
        .write(weight_type);

    file->flush();
}

void visFileArchive::create_dataset(const std::string& name, const std::vector<std::string>& axes,
                                    DataType type, const bool& compress) {

    // Mapping of axis names to sizes (start, chunk)
    std::map<std::string, std::tuple<size_t, size_t>> size_map;
    size_map["freq"] = std::make_tuple(length("freq"), chunk[0]);
    size_map["input"] =
        std::make_tuple(length("input"), std::min((size_t)(chunk[1]), length("input")));
    size_map["prod"] = std::make_tuple(length("prod"), chunk[1]);
    size_map["ev"] = std::make_tuple(length("ev"), length("ev"));
    size_map["time"] = std::make_tuple(length("time"), chunk[2]);
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
DataSet visFileArchive::dset(const std::string& name) {
    const std::string dset_name = name == "vis_weight" ? "flags/vis_weight" : name;
    return file->getDataSet(dset_name);
}

size_t visFileArchive::length(const std::string& axis_name) {
    if (!write_ev && axis_name == "ev")
        return 0;
    return dset(fmt::format(fmt("index_map/{:s}"), axis_name)).getSpace().getDimensions()[0];
}
