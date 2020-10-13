
#include "visFileH5.hpp"

#include "Hash.hpp"           // for Hash
#include "datasetManager.hpp" // for datasetManager, dset_id_t
#include "datasetState.hpp"   // for eigenvalueState, freqState, inputState, prodState
#include "visBuffer.hpp"      // for VisFrameView
#include "visUtil.hpp"        // for cfloat, time_ctype, freq_ctype, input_ctype, prod_ctype

#include "fmt.hpp"      // for format, fmt
#include "gsl-lite.hpp" // for span

#include <complex>                  // for complex
#include <cstdio>                   // for remove
#include <cxxabi.h>                 // for __forced_unwind
#include <errno.h>                  // for errno
#include <exception>                // for exception
#include <fcntl.h>                  // for sync_file_range, posix_fadvise, posix_fallocate, SYN...
#include <future>                   // for async, future
#include <highfive/H5Attribute.hpp> // for Attribute, Attribute::write, Attribute::getSpace
#include <highfive/H5DataSet.hpp>   // for DataSet, DataSet::resize, AnnotateTraits::createAttr...
#include <highfive/H5DataSpace.hpp> // for DataSpace, DataSpace::From, DataSpace::DataSpace
#include <highfive/H5DataType.hpp>  // for CompoundType, create_datatype, CompoundType::addMember
#include <highfive/H5File.hpp>      // for File, NodeTraits::createDataSet, File::flush, NodeTr...
#include <highfive/H5Group.hpp>     // for Group
#include <highfive/H5Object.hpp>    // for Object::getId, HighFive
#include <highfive/H5Selection.hpp> // for Selection, SliceTraits::write, SliceTraits::select
#include <numeric>                  // for iota
#include <stdexcept>                // for runtime_error, out_of_range
#include <string.h>                 // for strerror
#include <sys/stat.h>               // for fstat, stat
#include <system_error>             // for system_error
#include <tuple>                    // for make_tuple, tuple, get
#include <type_traits>              // for remove_reference<>::type
#include <unistd.h>                 // for pwrite, TEMP_FAILURE_RETRY
#include <utility>                  // for move, pair

using namespace HighFive;


// Register the HDF5 file writers
REGISTER_VIS_FILE("hdf5", visFileH5);
REGISTER_VIS_FILE("hdf5fast", visFileH5Fast);


//
// Implementation of standard HDF5 visibility data file
//

visFileH5::visFileH5(const std::string& name, const kotekan::logLevel log_level,
                     const std::map<std::string, std::string>& metadata, dset_id_t dataset,
                     size_t max_time) {
    set_log_level(log_level);

    auto& dm = datasetManager::instance();

    auto istate_fut = std::async(&datasetManager::dataset_state<inputState>, &dm, dataset);
    auto pstate_fut = std::async(&datasetManager::dataset_state<prodState>, &dm, dataset);
    auto fstate_fut = std::async(&datasetManager::dataset_state<freqState>, &dm, dataset);
    auto sstate_fut = std::async(&datasetManager::dataset_state<stackState>, &dm, dataset);
    auto evstate_fut = std::async(&datasetManager::dataset_state<eigenvalueState>, &dm, dataset);

    const stackState* sstate = sstate_fut.get();
    const inputState* istate = istate_fut.get();
    const prodState* pstate = pstate_fut.get();
    const freqState* fstate = fstate_fut.get();
    const eigenvalueState* evstate = evstate_fut.get();

    if (!istate || !pstate || !fstate) {
        ERROR("Required datasetState not found for dataset ID {}\nThe following required states "
              "were found:\ninputState - {:p}\nprodState - {:p}\nfreqState - {:p}\n",
              dataset, (void*)istate, (void*)pstate, (void*)fstate);
        throw std::runtime_error("Could not create file.");
    }


    if (sstate) {
        throw std::runtime_error("H5 writers do not currently worked with "
                                 "stacked data.");
    }

    std::string data_filename = fmt::format(fmt("{:s}.h5"), name);

    lock_filename = create_lockfile(data_filename);

    // Save the number of eigenvalues we are going to get
    num_ev = evstate ? evstate->get_num_ev() : 0;


    INFO("Creating new output file {:s}", name);

    file = std::unique_ptr<File>(
        new File(data_filename, File::ReadWrite | File::Create | File::Truncate));
    create_axes(unzip(fstate->get_freqs()).second, istate->get_inputs(), pstate->get_prods(),
                num_ev);
    _max_time = max_time;

    // Write out metadata into flle
    for (auto item : metadata) {
        file->createAttribute<std::string>(item.first, DataSpace::From(item.second))
            .write(item.second);
    }
}

void visFileH5::deferred_init() {
    create_time_axis();
    create_datasets();
}

visFileH5::~visFileH5() {
    file->flush();
    file.reset(nullptr);
    std::remove(lock_filename.c_str());
}

void visFileH5::create_axes(const std::vector<freq_ctype>& freqs,
                            const std::vector<input_ctype>& inputs,
                            const std::vector<prod_ctype>& prods, size_t num_ev) {

    // Create and fill other axes
    create_axis("freq", freqs);
    create_axis("input", inputs);
    create_axis("prod", prods);

    if (num_ev != 0) {
        std::vector<uint32_t> ev_vector(num_ev);
        std::iota(ev_vector.begin(), ev_vector.end(), 0);
        create_axis("ev", ev_vector);
    }
}

template<typename T>
void visFileH5::create_axis(std::string name, const std::vector<T>& axis) {

    Group indexmap =
        file->exist("index_map") ? file->getGroup("index_map") : file->createGroup("index_map");

    DataSet index = indexmap.createDataSet<T>(name, DataSpace(axis.size()));
    index.write(axis);
}

void visFileH5::create_time_axis() {

    Group indexmap =
        file->exist("index_map") ? file->getGroup("index_map") : file->createGroup("index_map");

    DataSet time_axis =
        indexmap.createDataSet("time", DataSpace({0}, {_max_time}), create_datatype<time_ctype>(),
                               std::vector<size_t>({1}));
}


void visFileH5::create_datasets() {

    Group flags = file->createGroup("flags");

    create_dataset("vis", {"time", "freq", "prod"}, create_datatype<cfloat>());
    create_dataset("flags/vis_weight", {"time", "freq", "prod"}, create_datatype<float>());
    create_dataset("gain_coeff", {"time", "freq", "input"}, create_datatype<cfloat>());
    create_dataset("gain_exp", {"time", "input"}, create_datatype<int>());

    // Only write the eigenvector datasets if there's going to be anything in
    // them
    if (num_ev != 0) {
        create_dataset("eval", {"time", "freq", "ev"}, create_datatype<float>());
        create_dataset("evec", {"time", "freq", "ev", "input"}, create_datatype<cfloat>());
        create_dataset("erms", {"time", "freq"}, create_datatype<float>());
    }

    // Copy the weight type flag to where gossec expects it
    std::string wt;
    file->getAttribute("weight_type").read(wt);
    auto ds = file->getAttribute("weight_type").getSpace();
    dset("vis_weight").createAttribute<std::string>("type", ds).write(wt);

    file->flush();
}

void visFileH5::create_dataset(const std::string& name, const std::vector<std::string>& axes,
                               DataType type) {

    size_t max_time = dset("index_map/time").getSpace().getMaxDimensions()[0];

    // Mapping of axis names to sizes (start, max, chunk)
    std::map<std::string, std::tuple<size_t, size_t, size_t>> size_map;
    size_map["freq"] = std::make_tuple(length("freq"), length("freq"), 1);
    size_map["input"] = std::make_tuple(length("input"), length("input"), length("input"));
    size_map["prod"] = std::make_tuple(length("prod"), length("prod"), length("prod"));
    size_map["ev"] = std::make_tuple(length("ev"), length("ev"), length("ev"));
    size_map["time"] = std::make_tuple(0, max_time, 1);

    std::vector<size_t> cur_dims, max_dims, chunk_dims;

    for (auto axis : axes) {
        auto cs = size_map[axis];
        cur_dims.push_back(std::get<0>(cs));
        max_dims.push_back(std::get<1>(cs));
        chunk_dims.push_back(std::get<2>(cs));
    }

    DataSpace space = DataSpace(cur_dims, max_dims);
    DataSet dset = file->createDataSet(name, space, type, max_dims);
    dset.createAttribute<std::string>("axis", DataSpace::From(axes)).write(axes);
}

// Quick functions for fetching datasets and dimensions
DataSet visFileH5::dset(const std::string& name) {
    const std::string dset_name = name == "vis_weight" ? "flags/vis_weight" : name;
    return file->getDataSet(dset_name);
}

size_t visFileH5::length(const std::string& axis_name) {
    if (axis_name == "ev" && num_ev == 0)
        return 0;
    return dset(fmt::format(fmt("index_map/{:s}"), axis_name)).getSpace().getDimensions()[0];
}

size_t visFileH5::num_time() {
    return length("time");
}


uint32_t visFileH5::extend_time(time_ctype new_time) {

    // If we haven't create all the datasets, we need to do that now.
    if (!file->exist("vis")) {
        deferred_init();
    }

    // Get the current dimensions
    size_t ntime = length("time"), nprod = length("prod"), ninput = length("input"),
           nfreq = length("freq"), nev = length("ev");

    INFO("Current size: {:d}; new size: {:d}", ntime, ntime + 1);
    // Add a new entry to the time axis
    ntime++;
    dset("index_map/time").resize({ntime});
    dset("index_map/time").select({ntime - 1}, {1}).write(&new_time);

    // Extend all other datasets
    dset("vis").resize({ntime, nfreq, nprod});
    dset("vis_weight").resize({ntime, nfreq, nprod});
    dset("gain_coeff").resize({ntime, nfreq, ninput});
    dset("gain_exp").resize({ntime, ninput});

    if (num_ev != 0) {
        dset("eval").resize({ntime, nfreq, nev});
        dset("evec").resize({ntime, nfreq, nev, ninput});
        dset("erms").resize({ntime, nfreq});
    }

    // Flush the changes
    file->flush();

    return ntime - 1;
}


void visFileH5::write_sample(uint32_t time_ind, uint32_t freq_ind, const FrameView& frame_view) {

    const VisFrameView& frame = static_cast<const VisFrameView&>(frame_view);

    // TODO: consider adding checks for all dims
    if (frame.num_ev != num_ev) {
        throw std::runtime_error(fmt::format(
            fmt("Number of eigenvalues don't match for write (got {:d}, expected {:d})"),
            frame.num_ev, num_ev));
    }

    // Get the current dimensions
    size_t nprod = length("prod"), ninput = length("input"), nev = length("ev");

    std::vector<cfloat> gain_coeff(ninput, {1, 0});
    std::vector<int32_t> gain_exp(ninput, 0);

    dset("vis").select({time_ind, freq_ind, 0}, {1, 1, nprod}).write(frame.vis.data());

    dset("vis_weight").select({time_ind, freq_ind, 0}, {1, 1, nprod}).write(frame.weight.data());
    dset("gain_coeff").select({time_ind, freq_ind, 0}, {1, 1, ninput}).write(gain_coeff);
    dset("gain_exp").select({time_ind, 0}, {1, ninput}).write(gain_exp);

    if (num_ev != 0) {
        dset("eval").select({time_ind, freq_ind, 0}, {1, 1, nev}).write(frame.eval.data());
        dset("evec")
            .select({time_ind, freq_ind, 0, 0}, {1, 1, nev, ninput})
            .write(frame.evec.data());
        dset("erms").select({time_ind, freq_ind}, {1, 1}).write(&frame.erms);
    }

    file->flush();
}


//
// Implementation of the fast HDF5 visibility data file
//

visFileH5Fast::visFileH5Fast(const std::string& name, const kotekan::logLevel log_level,
                             const std::map<std::string, std::string>& metadata, dset_id_t dataset,
                             size_t max_time) :
    visFileH5(name, log_level, metadata, dataset, max_time) {}

void visFileH5Fast::deferred_init() {
    create_time_axis();
    create_datasets();
    setup_raw();
}

visFileH5Fast::~visFileH5Fast() {
    // Save the number of samples added into the `num_time` attribute.
    int nt = (size_t)num_time();
    file->createAttribute<int>("num_time", DataSpace::From(nt)).write(nt);
}


void visFileH5Fast::create_time_axis() {
    // Fill the time axis with zeros
    std::vector<time_ctype> times(_max_time, {0, 0.0});
    create_axis("time", times);
}


void visFileH5Fast::create_dataset(const std::string& name, const std::vector<std::string>& axes,
                                   HighFive::DataType type) {

    std::vector<size_t> dims;

    for (auto axis : axes) {
        dims.push_back(length(axis));
    }

    hid_t create_p = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(create_p, H5D_CONTIGUOUS);
    H5Pset_alloc_time(create_p, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(create_p, H5D_FILL_TIME_NEVER);

    DataSpace space = DataSpace(dims);
    DataSet dset = file->createDataSet(name, space, type, create_p);
    dset.createAttribute<std::string>("axis", DataSpace::From(axes)).write(axes);
}


void visFileH5Fast::setup_raw() {

    // Get all the dataset lengths
    ntime = 0;
    nfreq = length("freq");
    nprod = length("prod");
    ninput = length("input");
    nev = length("ev");

    // Calculate all the dataset file offsets
    time_offset = H5Dget_offset(dset("index_map/time").getId());
    vis_offset = H5Dget_offset(dset("vis").getId());
    weight_offset = H5Dget_offset(dset("vis_weight").getId());
    gcoeff_offset = H5Dget_offset(dset("gain_coeff").getId());
    gexp_offset = H5Dget_offset(dset("gain_exp").getId());

    if (num_ev != 0) {
        eval_offset = H5Dget_offset(dset("eval").getId());
        evec_offset = H5Dget_offset(dset("evec").getId());
        erms_offset = H5Dget_offset(dset("erms").getId());
    }

    // WARNING: this is very much discouraged by the HDF5 folks. Only really
    // works for the sec2 driver.
    int* fhandle;
    H5Fget_vfd_handle(file->getId(), H5P_DEFAULT, (void**)(&fhandle));
    fd = *fhandle;

#ifndef __APPLE__
    struct stat st;
    if ((fstat(fd, &st) != 0) || (posix_fallocate(fd, 0, st.st_size) != 0)) {
        ERROR("Couldn't preallocate file: {:s}", strerror(errno));
    }
#endif
}

template<typename T>
bool visFileH5Fast::write_raw(off_t dset_base, int ind, size_t n, const std::vector<T>& vec) {


    if (vec.size() < n) {
        ERROR("Expected size of write ({:d}) exceeds vector length ({:d})", n, vec.size());
        return false;
    }

    return write_raw(dset_base, ind, n, vec.data());
}

template<typename T>
bool visFileH5Fast::write_raw(off_t dset_base, int ind, size_t n, const T* data) {


    size_t nb = n * sizeof(T);
    off_t offset = dset_base + ind * nb;

    // Write in a retry macro loop incase the write was interrupted by a signal
    int nbytes = TEMP_FAILURE_RETRY(pwrite(fd, (const void*)data, nb, offset));

    if (nbytes < 0) {
        ERROR("Write error attempting to write {:d} bytes at offset {:d}: {:s}", nb, offset,
              strerror(errno));
        return false;
    }

    return true;
}

void visFileH5Fast::flush_raw_async(off_t dset_base, int ind, size_t n) {
#ifdef __linux__
    sync_file_range(fd, dset_base + ind * n, n, SYNC_FILE_RANGE_WRITE);
#else
    (void)dset_base;
    (void)ind;
    (void)n;
#endif
}

void visFileH5Fast::flush_raw_sync(off_t dset_base, int ind, size_t n) {
#ifdef __linux__
    sync_file_range(fd, dset_base + ind * n, n,
                    SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE
                        | SYNC_FILE_RANGE_WAIT_AFTER);
    posix_fadvise(fd, dset_base + ind * n, n, POSIX_FADV_DONTNEED);
#else
    (void)dset_base;
    (void)ind;
    (void)n;
#endif
}

uint32_t visFileH5Fast::extend_time(time_ctype new_time) {

    // If we haven't create the datasets, we need to do that now.
    if (!file->exist("vis")) {
        deferred_init();
    }

    // Perform a raw write of the new time sample
    write_raw(time_offset, ntime, 1, &new_time);

    // Start to flush out older dataset regions
    uint delta_async = 2;
    if (ntime > delta_async) {
        flush_raw_async(vis_offset, ntime - delta_async, nfreq * nprod * sizeof(cfloat));
        flush_raw_async(weight_offset, ntime - delta_async, nfreq * nprod * sizeof(float));
        flush_raw_async(gcoeff_offset, ntime - delta_async, nfreq * ninput * sizeof(cfloat));
        flush_raw_async(gexp_offset, ntime - delta_async, ninput * sizeof(int32_t));

        if (num_ev != 0) {
            flush_raw_async(eval_offset, ntime - delta_async, nfreq * nev * sizeof(float));
            flush_raw_async(evec_offset, ntime - delta_async,
                            nfreq * nev * ninput * sizeof(cfloat));
            flush_raw_async(evec_offset, ntime - delta_async, nfreq * sizeof(float));
        }
    }

    // Flush and clear out any really old parts of the datasets
    uint delta_sync = 4;
    if (ntime > delta_sync) {
        flush_raw_sync(vis_offset, ntime - delta_sync, nfreq * nprod * sizeof(cfloat));
        flush_raw_sync(weight_offset, ntime - delta_sync, nfreq * nprod * sizeof(float));
        flush_raw_sync(gcoeff_offset, ntime - delta_sync, nfreq * ninput * sizeof(cfloat));
        flush_raw_sync(gexp_offset, ntime - delta_sync, ninput * sizeof(int32_t));

        if (num_ev != 0) {
            flush_raw_sync(eval_offset, ntime - delta_sync, nfreq * nev * sizeof(float));
            flush_raw_sync(evec_offset, ntime - delta_sync, nfreq * nev * ninput * sizeof(cfloat));
            flush_raw_sync(evec_offset, ntime - delta_sync, nfreq * sizeof(float));
        }
    }

    // Increment the time count and return the index of the added sample
    return ntime++;
}

void visFileH5Fast::deactivate_time(uint32_t time_ind) {
    flush_raw_sync(vis_offset, time_ind, nfreq * nprod * sizeof(cfloat));
    flush_raw_sync(weight_offset, time_ind, nfreq * nprod * sizeof(float));
    flush_raw_sync(gcoeff_offset, time_ind, nfreq * ninput * sizeof(cfloat));
    flush_raw_sync(gexp_offset, time_ind, ninput * sizeof(int32_t));

    if (num_ev != 0) {
        flush_raw_sync(eval_offset, time_ind, nfreq * nev * sizeof(float));
        flush_raw_sync(evec_offset, time_ind, nfreq * nev * ninput * sizeof(cfloat));
        flush_raw_sync(evec_offset, time_ind, nfreq * sizeof(float));
    }
}

void visFileH5Fast::write_sample(uint32_t time_ind, uint32_t freq_ind,
                                 const FrameView& frame_view) {

    const VisFrameView& frame = static_cast<const VisFrameView&>(frame_view);

    // TODO: consider adding checks for all dims
    if (frame.num_ev != num_ev) {
        throw std::runtime_error(fmt::format(
            fmt("Number of eigenvalues don't match for write (got {:d}, expected {:d})"),
            frame.num_ev, num_ev));
    }

    std::vector<cfloat> gain_coeff(ninput, {1, 0});
    std::vector<int32_t> gain_exp(ninput, 0);

    write_raw(vis_offset, time_ind * nfreq + freq_ind, nprod, frame.vis.data());
    write_raw(weight_offset, time_ind * nfreq + freq_ind, nprod, frame.weight.data());
    write_raw(gcoeff_offset, time_ind * nfreq + freq_ind, ninput, gain_coeff);
    write_raw(gexp_offset, time_ind, ninput, gain_exp);

    if (num_ev != 0) {
        write_raw(eval_offset, time_ind * nfreq + freq_ind, nev, frame.eval.data());
        write_raw(evec_offset, time_ind * nfreq + freq_ind, nev * ninput, frame.evec.data());
        write_raw(erms_offset, time_ind * nfreq + freq_ind, 1, (const float*)&frame.erms);
    }

    // Figure out what (if any) flushing or advising to do here.
}


size_t visFileH5Fast::num_time() {
    return ntime;
}


// Add support for all our custom types to HighFive
template<>
DataType HighFive::create_datatype<freq_ctype>() {
    CompoundType f;
    f.addMember("centre", H5T_IEEE_F64LE);
    f.addMember("width", H5T_IEEE_F64LE);
    f.autoCreate();
    return std::move(f);
}

template<>
DataType HighFive::create_datatype<time_ctype>() {
    CompoundType t;
    t.addMember("fpga_count", H5T_STD_U64LE);
    t.addMember("ctime", H5T_IEEE_F64LE);
    t.autoCreate();
    return std::move(t);
}

template<>
DataType HighFive::create_datatype<input_ctype>() {

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
DataType HighFive::create_datatype<prod_ctype>() {

    CompoundType p;
    p.addMember("input_a", H5T_STD_U16LE);
    p.addMember("input_b", H5T_STD_U16LE);
    p.autoCreate();
    return std::move(p);
}

template<>
DataType HighFive::create_datatype<cfloat>() {
    CompoundType c;
    c.addMember("r", H5T_IEEE_F32LE);
    c.addMember("i", H5T_IEEE_F32LE);
    c.autoCreate();
    return std::move(c);
}
