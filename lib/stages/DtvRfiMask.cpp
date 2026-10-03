#include "Config.hpp"
#include "DataType.hpp"
#include "N2Util.hpp"
#include "NDArray.hpp"
#include "Stage.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <set>
#include <vector>

/**
 * @class DtvRfiMask
 * @brief Fold one CHORD 8192-sample DTV decision block into the aligned RFImask frame.
 *
 * DTV int8[F] uses 1 = reject; RFImask uint1x8[8, F, 128] uses 1 = keep. The output is their
 * intersection: a rejected frequency has every time bit cleared, an accepted one keeps the
 * existing mask. Both the visibility correlator and its packet-loss counter must consume this
 * output. One detector block per RFImask frame; mismatched time or frequency identity stops the
 * run. Optional fine-support rows distinguish invalid rank support from a valid keep.
 * Invalid rows clear every time bit. The permanent-mask limit also clears every bit,
 * independently of the detector score; the raw detector products retain their meaning.
 *
 * @par Buffers
 * @buffer  rfi_buf     RFImask frames from n2k.
 *         @buffer_format   NDArray uint1x8 [8, num_local_freq, 128]
 *         @buffer_metadata chordMetadata
 * @buffer  dtv_buf     DTV decisions from cudaPilotProxyDetector.
 *         @buffer_format   NDArray int8 [num_local_freq]
 *         @buffer_metadata chordMetadata
 * @buffer  out_buf     The combined mask, same format as rfi_buf.
 * @buffer  fine_support_buf Optional int32 [F,2] rows [rank_valid,n_bulk].
 *         (-1,-1) means no fine test; 0/1 means invalid/valid rank support.
 *
 * @conf    num_local_freq  int  Frequencies per frame.
 * @conf    num_times       int  Samples per detector block; must be 8192.
 * @conf    permanent_mask_freq_ids Receiver coarse-frequency IDs at the reject-all limit.
 * @conf    require_fine_freq_ids Receiver IDs that must have an evaluated fine decision.
 */
class DtvRfiMask : public kotekan::Stage {
public:
    DtvRfiMask(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffers) :
        Stage(config, unique_name, buffers, std::bind(&DtvRfiMask::main_thread, this)),
        frequencies(config.get<int>(unique_name, "num_local_freq")), rfi(get_buffer("rfi_buf")),
        dtv(get_buffer("dtv_buf")), output(get_buffer("out_buf")),
        support(config.exists(unique_name, "fine_support_buf") ? get_buffer("fine_support_buf")
                                                               : nullptr) {
        if (frequencies <= 0 || config.get<int>(unique_name, "num_times") != 8192)
            FATAL_ERROR("DtvRfiMask requires positive num_local_freq and num_times=8192");
        rfi->register_consumer(unique_name);
        dtv->register_consumer(unique_name);
        output->register_producer(unique_name);
        const auto read_ids = [&](const char* field, std::vector<int>& result) {
            const auto ids =
                config.get_default<nlohmann::json>(unique_name, field, nlohmann::json::array());
            if (!ids.is_array())
                FATAL_ERROR("DtvRfiMask {:s} must be an array", field);
            for (const auto& id : ids) {
                if (!id.is_number_integer() || id < 0 || id >= 12288)
                    FATAL_ERROR(
                        "DtvRfiMask threshold IDs must be exact nonnegative integers below 12288");
                result.push_back(id.get<int>());
            }
            if (std::set<int>(result.begin(), result.end()).size() != result.size())
                FATAL_ERROR("DtvRfiMask threshold frequency IDs must be unique");
        };
        read_ids("permanent_mask_freq_ids", permanent);
        read_ids("require_fine_freq_ids", required);
        for (const int id : required)
            if (std::find(permanent.begin(), permanent.end(), id) != permanent.end())
                FATAL_ERROR("DtvRfiMask finite and permanent-mask thresholds overlap");
        if (!required.empty() && !support)
            FATAL_ERROR("DtvRfiMask required fine thresholds need fine_support_buf");
        if (rfi->frame_size != size_t(1024 * frequencies) || dtv->frame_size != size_t(frequencies)
            || output->frame_size != rfi->frame_size)
            FATAL_ERROR("DtvRfiMask buffer sizes do not match one detector block");
        const auto desc =
            kotekan::GenericNDArray::describe(kotekan::uint1x8, "RFImask", {8, frequencies, 128},
                                              {"T8hi128", "F", "T8lo128"}, {1024, 1, 8});
        rfi->require_frame_desc(desc);
        output->require_frame_desc(desc);
        dtv->require_frame_desc(kotekan::GenericNDArray::describe(kotekan::int8, "dtv_mask",
                                                                  {frequencies}, {"F"}, {1}));
        if (support) {
            support->register_consumer(unique_name);
            if (support->frame_size != size_t(frequencies) * 2 * sizeof(int32_t))
                FATAL_ERROR("DtvRfiMask fine support buffer size mismatch");
            support->require_frame_desc(kotekan::GenericNDArray::describe(
                kotekan::int32, "dtv_fine_support", {frequencies, 2}, {"F", "S"}, {1, 1}));
        }
    }

    void main_thread() override {
        N2::frameID rfi_id(rfi), dtv_id(dtv), out_id(output);
        N2::frameID support_id(support ? support : dtv);
        bool started = false;
        int64_t next_seq = 0;
        int period = 0;
        std::vector<int> bound_freq;
        while (!stop_thread) {
            const auto* keep = rfi->wait_for_full_frame(unique_name, rfi_id);
            if (!keep)
                break;
            const auto* reject = dtv->wait_for_full_frame(unique_name, dtv_id);
            if (!reject)
                break;
            const int32_t* fine = nullptr;
            if (support) {
                fine = reinterpret_cast<const int32_t*>(
                    support->wait_for_full_frame(unique_name, support_id));
                if (!fine)
                    break;
            }
            const auto rm = get_chord_metadata(rfi, rfi_id);
            const auto dm = get_chord_metadata(dtv, dtv_id);
            rm->check_frame_desc(rfi->get_frame_desc<kotekan::GenericNDArray>());
            dm->check_frame_desc(dtv->get_frame_desc<kotekan::GenericNDArray>());
            const auto seq = rm->get_fpga_seq_num();
            const auto rp = rm->get_time_downsampling_fpga();
            const auto dp = dm->get_time_downsampling_fpga();
            if (seq < 0 || dp <= 0 || rp <= 0 || rp % 1024 || int64_t(rp) * 8 != dp
                || seq != dm->get_fpga_seq_num())
                FATAL_ERROR("DtvRfiMask time alignment mismatch");
            if (!rm->has_coarse_freq() || !dm->has_coarse_freq()
                || rm->get_coarse_freq().size() != size_t(frequencies)
                || rm->get_coarse_freq() != dm->get_coarse_freq())
                FATAL_ERROR("DtvRfiMask frequency identity mismatch");
            const auto freq = rm->get_coarse_freq();
            if (std::set<int>(freq.begin(), freq.end()).size() != freq.size())
                FATAL_ERROR("DtvRfiMask duplicate frequency identity");
            for (const auto& meta : {rm, dm}) {
                if (!meta->has_freq_upchan_factor() || !meta->has_freq_upchan_index()
                    || meta->get_freq_upchan_factor() != std::vector<int>(frequencies, 1)
                    || meta->get_freq_upchan_index() != std::vector<int>(frequencies, 0))
                    FATAL_ERROR("DtvRfiMask requires un-upchannelized coarse frequencies");
            }
            if (started && (seq != next_seq || dp != period || freq != bound_freq))
                FATAL_ERROR("DtvRfiMask discontinuous time or frequency identity");
            if (!started)
                for (const auto& ids : {permanent, required})
                    for (const int id : ids)
                        if (std::find(freq.begin(), freq.end(), id) == freq.end())
                            FATAL_ERROR("DtvRfiMask threshold frequency ID {:d} is absent", id);
            if (support) {
                const auto sm = get_chord_metadata(support, support_id);
                sm->check_frame_desc(support->get_frame_desc<kotekan::GenericNDArray>());
                if (sm->get_fpga_seq_num() != seq || sm->get_time_downsampling_fpga() != dp
                    || !sm->has_coarse_freq() || sm->get_coarse_freq() != freq
                    || !sm->has_freq_upchan_factor() || !sm->has_freq_upchan_index()
                    || sm->get_freq_upchan_factor() != std::vector<int>(frequencies, 1)
                    || sm->get_freq_upchan_index() != std::vector<int>(frequencies, 0))
                    FATAL_ERROR("DtvRfiMask fine support identity mismatch");
            }
            std::vector<bool> excluded(frequencies);
            for (int f = 0; f < frequencies; ++f) {
                if (reject[f] != 0 && reject[f] != 1)
                    FATAL_ERROR("DtvRfiMask detector decision must be zero or one");
                bool invalid = false;
                if (fine) {
                    const int32_t valid = fine[2 * f], count = fine[2 * f + 1];
                    if (!((valid == -1 && count == -1)
                          || ((valid == 0 || valid == 1) && count >= 0 && count <= 256
                              && (valid == 0 || count > 0))))
                        FATAL_ERROR("DtvRfiMask malformed fine support row");
                    if (valid == -1
                        && std::find(required.begin(), required.end(), freq[f]) != required.end())
                        FATAL_ERROR("DtvRfiMask required fine threshold was not evaluated");
                    invalid = valid == 0;
                }
                excluded[f] =
                    reject[f] || invalid
                    || std::find(permanent.begin(), permanent.end(), freq[f]) != permanent.end();
            }

            auto* combined = output->wait_for_empty_frame(unique_name, out_id);
            if (!combined)
                break;
            std::memcpy(combined, keep, rfi->frame_size);
            for (int t = 0; t < 8; ++t)
                for (int f = 0; f < frequencies; ++f)
                    if (excluded[f])
                        std::memset(combined + (t * frequencies + f) * 128, 0, 128);
            output->allocate_new_metadata_object(out_id);
            const auto om = get_chord_metadata(output, out_id);
            om->deepCopy(rm);
            om->check_frame_desc(output->get_frame_desc<kotekan::GenericNDArray>());
            started = true;
            next_seq = seq + dp;
            period = dp;
            bound_freq = freq;
            rfi->mark_frame_empty(unique_name, rfi_id++);
            dtv->mark_frame_empty(unique_name, dtv_id++);
            if (support)
                support->mark_frame_empty(unique_name, support_id++);
            output->mark_frame_full(unique_name, out_id++);
        }
    }

private:
    const int frequencies;
    Buffer *rfi, *dtv, *output, *support;
    std::vector<int> permanent, required;
};

REGISTER_KOTEKAN_STAGE(DtvRfiMask);
