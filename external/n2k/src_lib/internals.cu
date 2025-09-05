#include "../include/n2k/internals/internals.hpp"
#include <gputils/cuda_utils.hpp>

using namespace std;
using namespace gputils;


namespace n2k {
#if 0
}  // editor auto-indent
#endif


Array<complex<int>> make_random_unpacked_e_array(int T, int F, int S)
{
    std::mt19937 &rng = gputils::default_rng;
    auto dist = std::uniform_int_distribution<int> (-7, 7);

    Array<complex<int>> ret({T,F,S}, af_uhost);
    for (long i = 0; i < ret.size; i++)
	ret.data[i] = { dist(rng), dist(rng) };

    return ret;
}


Array<complex<int>> unpack_e_array(const Array<uint8_t> &E_in, bool offset_encoded)
{
    const uint8_t to_offset_encoded = offset_encoded ? 0 : 0x88;
    
    assert(E_in.on_host());
    assert(E_in.is_fully_contiguous());

    Array<complex<int>> E_out(E_in.ndim, E_in.shape, af_uhost);

    for (long i = 0; i < E_in.size; i++) {
	uint8_t e = E_in.data[i] ^ to_offset_encoded;
	int e_re = int(e & 0xf) - 8;
	int e_im = int((e >> 4) & 0xf) - 8;       
	E_out.data[i] = { e_re, e_im };
    }

    return E_out;
}


Array<uint8_t> pack_e_array(const Array<complex<int>> &E_in, bool offset_encoded)
{
    const uint8_t from_twos_complement = offset_encoded ? 0x88 : 0;
    
    assert(E_in.on_host());
    assert(E_in.is_fully_contiguous());

    Array<uint8_t> E_out(E_in.ndim, E_in.shape, af_uhost);

    for (long i = 0; i < E_in.size; i++) {
	uint8_t e_re = E_in.data[i].real() & 0xf;
	uint8_t e_im = E_in.data[i].imag() & 0xf;
	E_out.data[i] = ((e_re) | (e_im << 4)) ^ from_twos_complement;
    }

    return E_out;
}


void _check_array(int ndim, const ssize_t *shape, const ssize_t *strides, ssize_t size, int aflags,
		  const char *func_name, const char *arr_name, int expected_ndim, bool contiguous)
{
    const char *err;

    if (size == 0)
	err = "is empty";
    else if (expected_ndim != ndim)
	err = "does not have correct dimension";
    else if (!gputils::af_on_gpu(aflags))
	err = "is not on GPU";
    else if (contiguous && (gputils::compute_ncontig(ndim,shape,strides) != ndim))
	err = "is not contiguous";
    else
	return;

    stringstream ss;
    ss << func_name << "(): '" << arr_name << "' array " << err;
    throw runtime_error(ss.str());
}


}  // namespace n2k
