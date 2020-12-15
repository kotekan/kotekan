#ifndef DSET_ID_HPP
#define DSET_ID_HPP

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

#endif
