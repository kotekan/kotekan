#ifndef H5_SUPPORT_HPP
#define H5_SUPPORT_HPP

#include "N2Util.hpp"  // for N2::freq_ctype, N2::prod_ctype
#include "visUtil.hpp" // for freq_ctype, prod_ctype, time_ctype, input_ctype

#include <cstddef>                 // for offsetof
#include <highfive/H5DataType.hpp> // for DataType, AtomicType, DataType::DataType

using namespace HighFive;

const size_t DSET_ID_LEN = 33; // Length of the string used to represent dataset IDs
struct dset_id_str {
    char hash[DSET_ID_LEN];
};

namespace HighFive {
// \cond NO_DOC
// Fixed length string to store dataset ID
template<>
inline AtomicType<dset_id_str>::AtomicType() {
    _hid = H5Tcopy(H5T_C_S1);
    H5Tset_size(_hid, DSET_ID_LEN);
}
// \endcond
}; // namespace HighFive

// These templated functions are needed in order to tell HighFive how the
// various structs are converted into HDF5 datatypes
// \cond NO_DOC
template<>
inline DataType HighFive::create_datatype<freq_ctype>() {
    return CompoundType{
        {"centre", AtomicType<double>{}},
        {"width", AtomicType<double>{}},
    };
}

template<>
inline DataType HighFive::create_datatype<time_ctype>() {
    return CompoundType{
        {"fpga_count", AtomicType<uint64_t>{}},
        {"ctime", AtomicType<double>{}},
    };
}

template<>
inline DataType HighFive::create_datatype<input_ctype>() {
    return CompoundType{
        {"chan_id", AtomicType<uint16_t>{}},
        {"correlator_input", AtomicType<char[32]>{}},
    };
}

template<>
inline DataType HighFive::create_datatype<prod_ctype>() {
    return CompoundType{
        {"input_a", AtomicType<uint16_t>{}},
        {"input_b", AtomicType<uint16_t>{}},
    };
}

template<>
inline DataType HighFive::create_datatype<N2::freq_ctype>() {
    return CompoundType({{"centre", AtomicType<double>{}, offsetof(N2::freq_ctype, centre)},
                         {"width", AtomicType<double>{}, offsetof(N2::freq_ctype, width)}},
                        sizeof(N2::freq_ctype));
}

template<>
inline DataType HighFive::create_datatype<N2::prod_ctype>() {
    return CompoundType({{"input_a", AtomicType<uint16_t>{}, offsetof(N2::prod_ctype, input_a)},
                         {"input_b", AtomicType<uint16_t>{}, offsetof(N2::prod_ctype, input_b)}},
                        sizeof(N2::prod_ctype));
}

template<>
inline DataType HighFive::create_datatype<cfloat>() {
    return CompoundType{
        {"r", AtomicType<float>{}},
        {"i", AtomicType<float>{}},
    };
}

template<>
inline DataType HighFive::create_datatype<rstack_ctype>() {
    return CompoundType{
        {"stack", AtomicType<uint32_t>{}},
        {"conjugate", AtomicType<uint8_t>{}},
    };
}

template<>
inline DataType HighFive::create_datatype<stack_ctype>() {
    return CompoundType{
        {"prod", AtomicType<uint32_t>{}},
        {"conjugate", AtomicType<uint8_t>{}},
    };
}
// \endcond

#endif
