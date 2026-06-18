#include "gpsL2CCode.hpp"

#include <stdexcept> // for out_of_range
#include <string>    // for to_string

namespace gps {

namespace {

// 27-stage Galois (modular) shift register feedback mask, octal -- the form
// used by GNSS-SDR and PocketSDR for IS-GPS-200 L2C (equivalent to polynomial
// 1112225171 octal). One step: R = (R >> 1) XOR (MASK if LSB set).
constexpr uint32_t L2C_FEEDBACK = 0445112474U;

// Per-PRN initial register states (octal), PRN 1..32, from IS-GPS-200 Table
// 3-II; transcribed from PocketSDR / GNSS-SDR (the two agree on CM).
constexpr uint32_t L2CM_INIT[L2C_PRN_MAX] = {
    0742417664, 0756014035, 0002747144, 0066265724, 0601403471, 0703232733,
    0124510070, 0617316361, 0047541621, 0733031046, 0713512145, 0024437606,
    0021264003, 0230655351, 0001314400, 0222021506, 0540264026, 0205521705,
    0064022144, 0120161274, 0044023533, 0724744327, 0045743577, 0741201660,
    0700274134, 0010247261, 0713433445, 0737324162, 0311627434, 0710452007,
    0722462133, 0050172213,
};

constexpr uint32_t L2CL_INIT[L2C_PRN_MAX] = {
    0624145772, 0506610362, 0220360016, 0710406104, 0001143345, 0053023326,
    0652521276, 0206124777, 0015563374, 0561522076, 0023163525, 0117776450,
    0606516355, 0003037343, 0046515565, 0671511621, 0605402220, 0002576207,
    0525163451, 0266527765, 0006760703, 0501474556, 0743747443, 0615534726,
    0763621420, 0720727474, 0700521043, 0222567263, 0132765304, 0746332245,
    0102300466, 0255231716,
};

void check_prn(int prn) {
    if (prn < L2C_PRN_MIN || prn > L2C_PRN_MAX)
        throw std::out_of_range("GPS L2C PRN out of range (1..32): " + std::to_string(prn));
}

std::vector<int8_t> generate(uint32_t reg, int length) {
    std::vector<int8_t> code(length);
    for (int i = 0; i < length; ++i) {
        code[i] = (reg & 1U) ? -1 : +1;                  // LSB 0 -> +1, 1 -> -1
        reg = (reg >> 1U) ^ (L2C_FEEDBACK * (reg & 1U)); // Galois step
    }
    return code;
}

} // namespace

std::vector<int8_t> generate_l2cm_code(int prn) {
    check_prn(prn);
    return generate(L2CM_INIT[prn - 1], L2CM_CODE_LENGTH);
}

std::vector<int8_t> generate_l2cl_code(int prn) {
    check_prn(prn);
    return generate(L2CL_INIT[prn - 1], L2CL_CODE_LENGTH);
}

} // namespace gps
