#ifndef AIRSPY_FRAME_DESC_HPP
#define AIRSPY_FRAME_DESC_HPP

#include "DataType.hpp" // for DataType
#include "NDArray.hpp"  // for GenericNDArray

#include <cstddef> // for ptrdiff_t
#include <memory>  // for shared_ptr

/**
 * @file
 * @brief Frame-descriptor builders for the airspy autocorr / crosscorr pipelines.
 *
 * Uses the *buffer-level* FrameDesc mechanism (Buffer::set_frame_desc /
 * get_frame_description) -- independent of the metadata pool, which these
 * pipelines run with set to "none". The convention: every airspy
 * producer/consumer stage calls @c buf->set_frame_desc on each buffer it
 * touches via one of these helpers. The first call records the descriptor;
 * subsequent calls compare via @c FrameDesc::operator== and ERROR on
 * mismatch -- so producer/consumer construction order is irrelevant, and a
 * future rewiring that conflicts with the expected layout is caught at
 * framework init rather than as silent garbage downstream.
 *
 * @note GenericNDArray models a single dtype, so the @c post_corr_buf
 * layout @c [spectrum_length float32 bins, 1 uint32 integration count] per
 * element is described as a pure-float32 array of the right total size
 * (the trailing count word is type-tagged float32 in the descriptor). The
 * byte-size and dtype of the spectra are accurately validated; only the
 * count slot's reported dtype is cosmetically wrong. A future
 * @c MixedNDArray descriptor could model the heterogeneous record
 * faithfully -- filed as a separate framework follow-up.
 */
namespace kotekan_airspy {

/// Raw airspy samples: int16 1-D buffer of length @p n_samples.
inline std::shared_ptr<kotekan::GenericNDArray> make_input_desc(std::ptrdiff_t n_samples) {
    return kotekan::GenericNDArray::create(kotekan::DataType::int16, "airspy_samples", {n_samples},
                                           {"sample"}, nullptr);
}

/// fftwEngine output spectra: cfloat32 1-D buffer of length @p n_complex.
/// Same shape regardless of how many FFTs are packed per frame -- the
/// descriptor records total complex elements, the consumers' indexing
/// arithmetic comes from their own config.
inline std::shared_ptr<kotekan::GenericNDArray> make_fengine_desc(std::ptrdiff_t n_complex) {
    return kotekan::GenericNDArray::create(kotekan::DataType::cfloat32, "fengine_spectra",
                                           {n_complex}, {"bin"}, nullptr);
}

/// SimpleCrosscorr / simpleAutocorr -> networkPowerStream layout. Per
/// element: @p spectrum_length float32 bins followed by a uint32
/// integration-count word (see file note above). Shape is 1-D for
/// single-element (autocorr) and 2-D [num_elements, spectrum_length + 1]
/// for multi-element (crosscorr), to match what the producers actually
/// emit and what networkPowerStream walks for each element/time index.
///
/// @note Only networkPowerStream (the consumer) calls set_frame_desc with
/// this; the producers (simpleAutocorr / SimpleCrosscorr) deliberately
/// leave their out_buf untagged so the corr-buffer round-trip tests that
/// use rawFileWrite still work (rawFileWrite refuses NDArray-tagged
/// buffers). The consumer-side assertion still documents the expected
/// layout in code; cross-checks are preserved on input_buf and
/// post_fengine_buf where they don't collide with the test infrastructure.
inline std::shared_ptr<kotekan::GenericNDArray>
make_power_corr_desc(std::ptrdiff_t num_elements, std::ptrdiff_t spectrum_length) {
    if (num_elements == 1) {
        return kotekan::GenericNDArray::create(kotekan::DataType::float32, "power_corr",
                                               {spectrum_length + 1}, {"bin"}, nullptr);
    }
    return kotekan::GenericNDArray::create(kotekan::DataType::float32, "power_corr",
                                           {num_elements, spectrum_length + 1}, {"elem", "bin"},
                                           nullptr);
}

/// GpsReplicaCorrelator output: one fixed-width record per tracked PRN per
/// emitted integration. The record is @p record_floats float32 fields
/// (see GpsReplicaCorrelator.hpp for the field order), and a frame holds
/// @p num_prns of them, so the descriptor is a 2-D float32 array
/// [num_prns, record_floats]. The PRN id is stored as a float field (PRNs
/// 1..32 are exactly representable), keeping the record a homogeneous
/// float32 block -- same single-dtype convention as make_power_corr_desc's
/// trailing count word.
inline std::shared_ptr<kotekan::GenericNDArray>
make_gps_record_desc(std::ptrdiff_t num_prns, std::ptrdiff_t record_floats) {
    return kotekan::GenericNDArray::create(kotekan::DataType::float32, "gps_records",
                                           {num_prns, record_floats}, {"prn", "field"}, nullptr);
}

} // namespace kotekan_airspy

#endif // AIRSPY_FRAME_DESC_HPP
