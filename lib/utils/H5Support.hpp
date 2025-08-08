#ifndef H5_SUPPORT_HPP
#define H5_SUPPORT_HPP

#include "visUtil.hpp" // for freq_ctype, prod_ctype, time_ctype, input_ctype

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
    CompoundType f;
    f.addMember("centre", H5T_IEEE_F64LE);
    f.addMember("width", H5T_IEEE_F64LE);
    f.autoCreate();
    return f;
}

template<>
inline DataType HighFive::create_datatype<time_ctype>() {
    CompoundType t;
    t.addMember("fpga_count", H5T_STD_U64LE);
    t.addMember("ctime", H5T_IEEE_F64LE);
    t.autoCreate();
    return t;
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

    return i;
}

template<>
inline DataType HighFive::create_datatype<prod_ctype>() {

    CompoundType p;
    p.addMember("input_a", H5T_STD_U16LE);
    p.addMember("input_b", H5T_STD_U16LE);
    p.autoCreate();
    return p;
}

template<>
inline DataType HighFive::create_datatype<cfloat>() {
    CompoundType c;
    c.addMember("r", H5T_IEEE_F32LE);
    c.addMember("i", H5T_IEEE_F32LE);
    c.autoCreate();
    return c;
}

template<>
inline DataType HighFive::create_datatype<rstack_ctype>() {
    CompoundType c;
    c.addMember("stack", H5T_STD_U32LE);
    c.addMember("conjugate", H5T_STD_U8LE);
    c.autoCreate();
    return c;
}

template<>
inline DataType HighFive::create_datatype<stack_ctype>() {
    CompoundType c;
    c.addMember("prod", H5T_STD_U32LE);
    c.addMember("conjugate", H5T_STD_U8LE);
    c.autoCreate();
    return c;
}
// \endcond

#endif
