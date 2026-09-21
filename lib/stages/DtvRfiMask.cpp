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

#include <cstring>
#include <functional>
#include <vector>

/**
 * @class DtvRfiMask
 * @brief Fold one CHORD 8192-sample DTV decision block into the aligned RFImask frame.
 *
 * DTV int8[F] uses 1 = reject; RFImask uint1x8[8, F, 128] uses 1 = keep. The output is their
 * intersection: a rejected frequency has every time bit cleared, an accepted one keeps the
 * existing mask. Both the visibility correlator and its packet-loss counter must consume this
 * output. One detector block per RFImask frame; mismatched time or frequency identity stops the
 * run.
 *
 * @par Buffers
 * @buffer  rfi_buf     RFImask frames from n2k.
 *         @buffer_format   NDArray uint1x8 [8, num_local_freq, 128]
 *         @buffer_metadata chordMetadata
 * @buffer  dtv_buf     DTV decisions from cudaPilotProxyDetector.
 *         @buffer_format   NDArray int8 [num_local_freq]
 *         @buffer_metadata chordMetadata
 * @buffer  out_buf     The combined mask, same format as rfi_buf.
 *
 * @conf    num_local_freq  int  Frequencies per frame.
 * @conf    num_times       int  Samples per detector block; must be 8192.
 */
class DtvRfiMask : public kotekan::Stage {
public:
    DtvRfiMask(kotekan::Config& config, const std::string& unique_name,
               kotekan::bufferContainer& buffers) :
        Stage(config, unique_name, buffers, std::bind(&DtvRfiMask::main_thread, this)),
        frequencies(config.get<int>(unique_name, "num_local_freq")), rfi(get_buffer("rfi_buf")),
        dtv(get_buffer("dtv_buf")), output(get_buffer("out_buf")) {
        if (frequencies <= 0 || config.get<int>(unique_name, "num_times") != 8192)
            FATAL_ERROR("DtvRfiMask requires positive num_local_freq and num_times=8192");
        rfi->register_consumer(unique_name);
        dtv->register_consumer(unique_name);
        output->register_producer(unique_name);
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
    }

    void main_thread() override {
        N2::frameID rfi_id(rfi), dtv_id(dtv), out_id(output);
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
            for (const auto& meta : {rm, dm}) {
                if (!meta->has_freq_upchan_factor() || !meta->has_freq_upchan_index()
                    || meta->get_freq_upchan_factor() != std::vector<int>(frequencies, 1)
                    || meta->get_freq_upchan_index() != std::vector<int>(frequencies, 0))
                    FATAL_ERROR("DtvRfiMask requires un-upchannelized coarse frequencies");
            }
            if (started && (seq != next_seq || dp != period || freq != bound_freq))
                FATAL_ERROR("DtvRfiMask discontinuous time or frequency identity");

            auto* combined = output->wait_for_empty_frame(unique_name, out_id);
            if (!combined)
                break;
            std::memcpy(combined, keep, rfi->frame_size);
            for (int t = 0; t < 8; ++t)
                for (int f = 0; f < frequencies; ++f)
                    if (reject[f])
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
            output->mark_frame_full(unique_name, out_id++);
        }
    }

private:
    const int frequencies;
    Buffer *rfi, *dtv, *output;
};

REGISTER_KOTEKAN_STAGE(DtvRfiMask);
