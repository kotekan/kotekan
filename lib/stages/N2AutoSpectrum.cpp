#include "N2AutoSpectrum.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for float32
#include "FrameDesc.hpp"       // for FrameDesc
#include "N2FrameView.hpp"     // for N2FrameView
#include "N2Util.hpp"          // for frameID, prod_ctype
#include "NDArray.hpp"         // for GenericNDArray
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"  // for FATAL_ERROR, WARN, DEBUG

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>  // for fill
#include <complex>    // for complex
#include <cstddef>    // for size_t, ptrdiff_t
#include <functional> // for bind, function
#include <limits>     // for numeric_limits

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using N2::frameID;

REGISTER_KOTEKAN_STAGE(N2AutoSpectrum);

namespace {
// An output frame held open while its bin's frequencies arrive.
struct PendingBin {
    uint64_t abs_time_idx;
    int frame_id;
    float* data;
};
} // namespace

N2AutoSpectrum::N2AutoSpectrum(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&N2AutoSpectrum::main_thread, this)),
    _num_local_freq(config.get<int64_t>(unique_name, "num_local_freq")),
    _freq_id_base(config.get_default<int64_t>(unique_name, "freq_id_base", 0)),
    _num_pending_bins(config.get_default<uint32_t>(unique_name, "num_pending_bins", 2)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    if (in_buf->buffer_type != "N2")
        FATAL_ERROR("N2AutoSpectrum: in_buf must be of type 'N2', got '{:s}'", in_buf->buffer_type);
    if (out_buf->buffer_type != "ndarray")
        FATAL_ERROR("N2AutoSpectrum: out_buf must be of type 'ndarray', got '{:s}'",
                    out_buf->buffer_type);

    if (_num_local_freq <= 0)
        FATAL_ERROR("num_local_freq is not positive: {:d}", _num_local_freq);
    if (_num_pending_bins < 1)
        FATAL_ERROR("num_pending_bins is not positive: {:d}", _num_pending_bins);
    // Pending bins plus one peek-held (or in-flight) frame must fit in the ring
    if ((int)_num_pending_bins + 1 > out_buf->num_frames)
        FATAL_ERROR("num_pending_bins ({:d}) needs at least {:d} frames in out_buf ({:s}), "
                    "which has {:d}",
                    _num_pending_bins, _num_pending_bins + 1, out_buf->buffer_name,
                    out_buf->num_frames);

    in_desc = in_buf->require_frame_desc<kotekan::N2FrameDesc>();
    _num_elements = in_desc->get_num_elements();

    out_buf->require_frame_desc(kotekan::GenericNDArray::describe(
        kotekan::float32, "autospectrum", {_num_local_freq, (std::ptrdiff_t)_num_elements},
        {"F", "E"}, {1, 1}));

    // Locate each element's autocorrelation in the input product list
    const auto& prods = in_desc->get_product_list();
    auto_prod_index.resize(_num_elements, UINT32_MAX);
    for (size_t k = 0; k < prods.size(); ++k) {
        if (prods[k].input_a == prods[k].input_b)
            auto_prod_index[prods[k].input_a] = k;
    }
    for (uint32_t e = 0; e < _num_elements; ++e) {
        if (auto_prod_index[e] == UINT32_MAX)
            FATAL_ERROR("N2AutoSpectrum: in_buf ({:s}) has no autocorrelation for element {:d}",
                        in_buf->buffer_name, e);
    }
}

void N2AutoSpectrum::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);

    // Bins held open, oldest first; flushed in order so newer output frames
    // are always filled later (peek returns the newest filled frame).
    std::vector<PendingBin> pending;

    const size_t out_frame_len = (size_t)_num_local_freq * _num_elements;
    uint64_t num_late = 0;
    uint64_t num_out_of_range = 0;

    while (!stop_thread) {

        if (in_buf->wait_for_full_frame(unique_name, in_frame_id) == nullptr)
            break;

        N2FrameView input_vis(in_buf, in_frame_id);
        const uint64_t bin = input_vis.abs_time_idx;

        PendingBin* slab = nullptr;
        for (size_t i = 0; i < pending.size(); ++i) {
            if (pending[i].abs_time_idx == bin)
                slab = &pending[i];
        }

        if (slab == nullptr && !pending.empty() && bin < pending.back().abs_time_idx) {
            num_late++;
            WARN("Dropping frame for bin {:d} (freq_id {:d}): newest open bin is {:d} "
                 "({:d} dropped so far).",
                 bin, input_vis.freq_id, pending.back().abs_time_idx, num_late);
            in_buf->mark_frame_empty(unique_name, in_frame_id++);
            continue;
        }

        if (slab == nullptr) {
            if (pending.size() >= _num_pending_bins) {
                DEBUG("Flushing bin {:d} to {:s}[{:d}].", pending.front().abs_time_idx,
                      out_buf->buffer_name, pending.front().frame_id);
                out_buf->mark_frame_full(unique_name, pending.front().frame_id);
                pending.erase(pending.begin());
            }

            float* data = (float*)out_buf->wait_for_empty_frame(unique_name, out_frame_id);
            if (data == nullptr)
                break;
            std::fill(data, data + out_frame_len, std::numeric_limits<float>::quiet_NaN());

            out_buf->allocate_new_metadata_object(out_frame_id);
            const std::shared_ptr<chordMetadata> out_meta =
                get_chord_metadata(out_buf, out_frame_id);
            out_meta->set_from_frame_desc(out_buf->get_frame_desc<kotekan::GenericNDArray>());
            out_meta->set_fpga_seq_num(input_vis.fpga_start_tick);

            pending.push_back({bin, (int)out_frame_id, data});
            slab = &pending.back();
            out_frame_id++;
        }

        const int64_t row = (int64_t)input_vis.freq_id - _freq_id_base;
        if (row < 0 || row >= _num_local_freq) {
            num_out_of_range++;
            if (num_out_of_range == 1)
                WARN("freq_id {:d} outside output rows [{:d}, {:d}), dropping; further "
                     "occurrences logged at DEBUG.",
                     input_vis.freq_id, _freq_id_base, _freq_id_base + _num_local_freq);
            else
                DEBUG("freq_id {:d} outside output rows, dropping ({:d} so far).",
                      input_vis.freq_id, num_out_of_range);
        } else {
            float* out_row = slab->data + row * _num_elements;
            for (uint32_t e = 0; e < _num_elements; ++e)
                out_row[e] = input_vis.vis[auto_prod_index[e]].real();
        }

        in_buf->mark_frame_empty(unique_name, in_frame_id++);
    }
}
