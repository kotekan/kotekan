#include <cstdint>
#include <cstring>

// 2**31 + 2**30 will be used to check for overflow
const uint32_t HIGH_BITS = 3221225472;

/**
 *  @brief Truncate the precision of *val* by rounding to a multiple of a power of
 *        two, keeping error less than or equal to *err*.
 *
 *  @warning Undefined results for err < 0 and err > 2**30.
 */
inline int32_t bit_truncate(int32_t val, int32_t err) {
    // *gran* is the granularity. It is the power of 2 that is *larger than* the
    // maximum error *err*.
    int32_t gran = err;
    gran |= gran >> 1;
    gran |= gran >> 2;
    gran |= gran >> 4;
    gran |= gran >> 8;
    gran |= gran >> 16;
    gran += 1;

    // Bitmask selects bits to be rounded.
    int32_t bitmask = gran - 1;

    // Determine if there is a round-up/round-down tie.
    // This operation gets the `gran = 1` case correct (non tie).
    int32_t tie = ((val & bitmask) << 1) == gran;

    // The acctual rounding.
    int32_t val_t = (val - (gran >> 1)) | bitmask;
    val_t += 1;
    // There is a bit of extra bit twiddling for the err == 0.
    val_t -= (err == 0);

    // Break any tie by rounding to even.
    val_t -= val_t & (tie * gran);

    return val_t;
}


/**
 *  @brief Count the number of leading zeros in a binary number.
 *         Taken from https://stackoverflow.com/a/23857066
 */
inline int32_t count_zeros(int32_t x) {
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return __builtin_popcount(~x);
}


/**
 * @brief Fast power of two float.
 *
 * Result is undefined for e < -126.
 *
 * @param   e   Exponent
 *
 * @returns The result of 2^e
 */
inline float fast_pow(int8_t e) {
    // Construct float bitwise
    uint32_t out_i = ((uint32_t)(127 + e) << 23);
    // Cast into float
    float out_f;
    memcpy(&out_f, &out_i, sizeof(float));
    return out_f;
}


/**
 *  @brief Truncate precision of a floating point number by applying the algorithm of
 *         `bit_truncate` to the mantissa.
 *
 *  Note that NaN and inf are not explicitly checked for. According to the IEEE spec, it is
 *  impossible for the truncation to turn an inf into a NaN. However, if the truncation
 *  happens to remove all of the non-zero bits in the mantissa, a NaN can become inf.
 *
 */
inline float bit_truncate_float(float val, float err) {
    // cast float memory into an int
    int32_t* cast_val_ptr = (int32_t*)&val;
    // extract the exponent and sign
    int32_t val_pre = cast_val_ptr[0] >> 23;
    // strip sign
    int32_t val_pow = val_pre & 255;
    int32_t val_s = val_pre >> 8;
    // extract mantissa. mask is 2**23 - 1. Add back the implicit 24th bit
    int32_t val_man = (cast_val_ptr[0] & 8388607) + 8388608;
    // scale the error to the integer representation of the mantissa
    // scale by 2**(23 + 127 - pow)
    int32_t int_err = (int32_t)(err * fast_pow(150 - val_pow));
    // make sure hasn't overflowed. if set to 2**30-1, will surely round to 0.
    // must keep err < 2**30 for bit_truncate to work
    int_err = (int_err & HIGH_BITS) ? 1073741823 : int_err;

    // truncate
    int32_t tr_man = bit_truncate(val_man, int_err);

    // count leading zeros
    int32_t z_count = count_zeros(tr_man);
    // adjust power after truncation to account for loss of implicit bit
    val_pow -= z_count - 8;
    // shift mantissa by same amount, remove implicit bit
    tr_man = (tr_man << (z_count - 8)) & 8388607;
    // round to zero case
    val_pow = ((z_count != 32) ? val_pow : 0);
    // restore sign and exponent
    int32_t tr_val = tr_man | ((val_pow | (val_s << 8)) << 23);
    // cast back to float
    float* tr_val_ptr = (float*)&tr_val;

    return tr_val_ptr[0];
}
