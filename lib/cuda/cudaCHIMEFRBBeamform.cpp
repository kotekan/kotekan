#include "Config.hpp"                 // for Config
#include "DataType.hpp"               // for int4x2_swapped_withoffset_t, uint1x8_t
#include "NDArray.hpp"                // for NDArray
#include "NDArrayBuffer.hpp"          // for NDArrayBuffer
#include "NDArrayRingBuffer.hpp"      // for NDArrayRingBuffer, read_descriptor_t, extent_t
#include "Telescope.hpp"              // for Telescope
#include "bufferContainer.hpp"        // for bufferContainer
#include "chordMetadata.hpp"          // for chordMetadata
#include "cudaCommand.hpp"            // for cudaCommand, cudaPipelineState, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp"    // for cudaDeviceInterface
#include "cudaUtils.hpp"              // for CHECK_CUDA_ERROR
#include "gpuCommand.hpp"             // for gpuCommandType
#include "kotekanLogging.hpp"         // for DEBUG, FATAL_ERROR
#include "pirate/ChimeBeamformer.hpp" // for launch_chime_frb_beamform

#include <cassert>            // for assert
#include <cstddef>            // for ptrdiff_t, size_t
#include <cstdint>            // for uint64_t, uint8_t
#include <cuda_runtime_api.h> // for cudaStreamSynchronize
#include <driver_types.h>     // for CUstream_st, cudaEvent_t, CUevent_st
#include <memory>             // for allocator, shared_ptr, __shared_ptr_access
#include <string>             // for basic_string, string
#include <type_traits>        // for is_same
#include <vector>             // for vector

using kotekan::bufferContainer;

class cudaCHIMEFRBBeamform : public cudaCommand {
public:
    cudaCHIMEFRBBeamform(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                         const int instance_num);
    virtual ~cudaCHIMEFRBBeamform();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    constexpr static const int num_cylinders = 4;
    constexpr static const int num_dishes_per_cylinder = 256;

    // Parameters
    const int buffer_depth;
    const int num_times;
    const int num_frequencies;
    const int num_polarizations;
    const int num_dishes;
    const bool poison_buffers;

    // Kotekan buffer names
    const std::string voltage_name;
    const std::string frb1_beams_name;

    // Buffers
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> voltage; // (time, freq, pol, ew*ns)
    NDArrayRingBuffer<kotekan::GetType_t<kotekan::cfloat16>, 5>
        frb1_beams; // (time, freq, pol, ew, ns)

    // pre-computed non-buffer data
    NDArrayBuffer<uint, 2> map;
    NDArrayBuffer<float, 4> co;
    NDArrayBuffer<float, 4> gains; // these actually come from a buffer

    const Telescope& tel;

    // helpers
    void calculate_ew_co(std::vector<float>& co, const std::vector<int>& coarse_freq);

    bool first_time; // keep track if whether we have already initialized constant data
};

REGISTER_CUDA_COMMAND(cudaCHIMEFRBBeamform);

void cudaCHIMEFRBBeamform::calculate_ew_co(std::vector<float>& coff,
                                           const std::vector<int>& coarse_freq) {
    // copied from lib/testing/gpuBeamformSimulate.cpp git hash bdbe7b3
    // then re-formatted

    assert(num_cylinders == 4);
    const double ew_spacing[4] = {// same as production CHIME
                                  -0.4, 0, 0.4, 0.8};

    // these are defined gpuBeamformSimulate.cpp, let's hope they are correct
    const double PI = 3.14159265;
    const double light = 299792458.;

    // Work out the EW phase coefficients from freq_MHz
    for (int f = 0; f < num_frequencies; ++f) {
        const double freq_MHz = tel.to_freq_MHz(freq_id_t(coarse_freq.at(f)));
        for (int angle_iter = 0; angle_iter < num_cylinders; angle_iter++) {
            double anglefrac = sin(ew_spacing[angle_iter] * PI / 180.);
            for (int cylinder = 0; cylinder < num_cylinders; cylinder++) {
                coff.at(angle_iter * num_cylinders * 2 + cylinder * 2) =
                    cos(2 * PI * anglefrac * cylinder * 22 * freq_MHz * 1.e6 / light);
                coff.at(angle_iter * num_cylinders * 2 + cylinder * 2 + 1) =
                    sin(2 * PI * anglefrac * cylinder * 22 * freq_MHz * 1.e6 / light);
            }
        }
    }
}

cudaCHIMEFRBBeamform::cudaCHIMEFRBBeamform(kotekan::Config& config, const std::string& unique_name,
                                           kotekan::bufferContainer& host_buffers,
                                           cudaDeviceInterface& device, const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num, no_cuda_command_state,
                "cudaCHIMEFRBBeamform"),
    // Parameters
    buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    num_times(config.get<int>(unique_name, "num_times")),
    num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    num_dishes(config.get<int>(unique_name, "num_dishes")),
    poison_buffers(config.get_default<bool>(unique_name, "poison_buffers", false)),
    // Buffer names
    voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    frb1_beams_name(config.get<std::string>(unique_name, "frb1_beams_name")),
    // Buffers
    voltage(voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{buffer_depth * num_times, num_frequencies,
                                          num_polarizations, num_dishes},
            std::array<std::string, 4>{"T", "F", "P", "D"}, *this),
    frb1_beams(frb1_beams_name, "frb1",
               std::array<std::ptrdiff_t, 5>{buffer_depth * num_times, num_frequencies,
                                             num_polarizations, num_cylinders,
                                             num_dishes_per_cylinder},
               std::array<std::string, 5>{"T", "F", "P", "EW", "NS"}, *this),
    map(unique_name + "/gpu_mem_map", "index",
        std::array<ptrdiff_t, 2>{num_frequencies, num_dishes_per_cylinder},
        std::array<kotekan::Symbol, 2>{"F", "NS"}, *this),
    co(unique_name + "/gpu_mem_co", "coeff",
       std::array<ptrdiff_t, 4>{num_frequencies, num_cylinders, num_cylinders, 2},
       std::array<kotekan::Symbol, 4>{"F", "EWout", "EWin", "ReIm"}, *this),
    gains(unique_name + "/gpu_mem_gains", "G",
          std::array<ptrdiff_t, 4>{num_frequencies, num_polarizations, num_cylinders,
                                   num_dishes_per_cylinder},
          std::array<kotekan::Symbol, 4>{"F", "P", "EW", "NS"}, *this),
    tel(Telescope::instance()), first_time(true)
//
{
    if (num_times % 128 != 0)
        FATAL_ERROR("num_times % 128 != 0");

    if (num_polarizations != 2)
        FATAL_ERROR("num_polarizations != 2");

    if (num_dishes != num_cylinders * num_dishes_per_cylinder)
        FATAL_ERROR("num_dishes != 1024");

    static_assert(
        sizeof(uint8_t)
        == kotekan::type_total_bytes(
            kotekan::int4x2_swapped_withoffset)); // pirate does not declare the correct 4bit type
    static_assert(std::is_same_v<__half, kotekan::GetType_t<kotekan::cfloat16>::value_type>);

    voltage.register_consumer();
    frb1_beams.register_producer();

    set_command_type(gpuCommandType::KERNEL);
}

cudaCHIMEFRBBeamform::~cudaCHIMEFRBBeamform() {}

int cudaCHIMEFRBBeamform::wait_on_precondition() {
    DEBUG("Waiting for voltage input ringbuffer data for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t voltage_read = num_times; // get one full frame
    const int voltage_errcode =
        voltage.wait_and_claim_readable([&](const std::ptrdiff_t available_elements) {
            if (available_elements < voltage_read)
                return read_descriptor_t{.claimed = 0, .read = 0};
            else
                return read_descriptor_t{.claimed = voltage_read, .read = voltage_read};
        });
    if (voltage_errcode < 0)
        return voltage_errcode;
    DEBUG("Done waiting for voltage input ringbuffer data for frame {:d}; will read {:d} elements",
          gpu_frame_id, voltage_read);

    DEBUG("Waiting for frb1_beams output ringbuffer space for frame {:d}...", gpu_frame_id);
    const std::ptrdiff_t frb1_beams_written =
        voltage_read; // no downsampling in time when forming beams
    const int frb1_beams_errcode = frb1_beams.wait_for_writable(frb1_beams_written);
    if (frb1_beams_errcode < 0)
        return frb1_beams_errcode;
    DEBUG("Done waiting for frb1_beams ouput ringbuffer data for frame {:d}; will write {:d} "
          "elements",
          gpu_frame_id, frb1_beams_written);

    return 0;
}

cudaEvent_t cudaCHIMEFRBBeamform::execute(cudaPipelineState& /*pipestate*/,
                                          const std::vector<cudaEvent_t>& /*pre_events*/) {
    pre_execute();
    record_start_event();

    voltage.check_metadata();

    // TODO: Set these metadata only once
    frb1_beams.set_metadata(voltage.get_metadata());

    // There is no poison value
    // if (poison_buffers)
    //    frb1_beams.set_to_poison(0xff);

    // needs metadata and thus access to an actual frame
    if (first_time && instance_num == 0) { // first time and we handle frame 0 of the buffer depth
        DEBUG("Setting up constant data");

        const auto meta = voltage.get_metadata();
        const auto coarse_freq = meta->get_coarse_freq();

        // indices for clamping
        std::vector<uint> host_map(map.get_ndarray().size());
        for (int f = 0; f < num_frequencies; ++f) {
            const double northmost_beam = 60.0; // same as production CHIME
            const double freq_in_MHz = tel.to_freq_MHz(freq_id_t(coarse_freq.at(f)));
            uint* const host_map_memory = &host_map.at(f * map.get_ndarray().stride(0));
            pirate::calculate_cl_index(host_map_memory, freq_in_MHz, northmost_beam);
        }
        CHECK_CUDA_ERROR(
            cudaMemcpyAsync(host_map.data(), map.get_ndarray().get_data(),
                            map.get_ndarray().size() * map.get_ndarray().get_value_type_size(),
                            cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));

        // this could be done in the constructor, though keeping it here keeps
        // things in one location
        std::vector<float> host_co(co.get_ndarray().size());
        calculate_ew_co(host_co, coarse_freq);
        CHECK_CUDA_ERROR(
            cudaMemcpyAsync(host_co.data(), co.get_ndarray().get_data(),
                            co.get_ndarray().size() * co.get_ndarray().get_value_type_size(),
                            cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));

        // TODO: this should actually come out of a buffer, but is hard-coded for now
        std::vector<float> host_gains(gains.get_ndarray().size(), 1.0);
        CHECK_CUDA_ERROR(
            cudaMemcpyAsync(host_gains.data(), gains.get_ndarray().get_data(),
                            gains.get_ndarray().size() * gains.get_ndarray().get_value_type_size(),
                            cudaMemcpyHostToDevice, device.getStream(cuda_stream_id)));

        // only almost thread save in that we must ensure that the compiler does
        // to re-order execution too aggressively
        first_time = true;
    }


    const kotekan::int4x2_swapped_withoffset_t* const voltage_memory = voltage.get_ndarray().data();
    kotekan::GetType_t<kotekan::cfloat16>* const frb1_beams_memory =
        frb1_beams.get_ndarray().data();
    const uint* const map_memory = map.get_ndarray().data();
    const float* const co_memory = co.get_ndarray().data();
    const float* const gains_memory = gains.get_ndarray().data();

    const std::ptrdiff_t Tsize = voltage.get_ndarray().extent(0);
    assert(Tsize % 128 == 0);
    // Tmin wraps around into actual array index to avoid overflows
    const std::ptrdiff_t Tmin = voltage.get_read_valid().begin() % Tsize;
    assert(Tmin % 128 == 0);
    const std::ptrdiff_t T = voltage.get_read_valid().size();
    if (Tmin + T > Tsize) {
        FATAL_ERROR("Chunk starting at Tmin={:d} of size T={:d} runs past end of ringbuffer {:s} "
                    "of size Tsize={:d}",
                    Tmin, T, voltage.get_buffer_name(), Tsize);
    }
    assert(Tmin + T <= Tsize);
    const std::ptrdiff_t T_offset = Tmin * voltage.get_ndarray().stride(0);

    const std::ptrdiff_t Tfrb_size = frb1_beams.get_ndarray().extent(0);
    assert(Tfrb_size % 128 == 0);
    // Tmin wraps around into actual array index to avoid overflows
    const std::ptrdiff_t Tfrb_min = frb1_beams.get_read_valid().begin() % Tfrb_size;
    assert(Tfrb_min % 128 == 0);
    const std::ptrdiff_t Tfrb = frb1_beams.get_read_valid().size();
    const std::ptrdiff_t Tfrb_offset = Tmin * frb1_beams.get_ndarray().stride(0);

    assert(T == Tfrb);
    // check compatibility of arrays
    assert(voltage.get_ndarray().extent(1) == frb1_beams.get_ndarray().extent(1));           // freq
    assert(voltage.get_ndarray().extent(2) == 2 && frb1_beams.get_ndarray().extent(2) == 2); // pol
    assert(voltage.get_ndarray().extent(3) == num_dishes_per_cylinder * num_cylinders
           && frb1_beams.get_ndarray().extent(3) == num_cylinders
           && frb1_beams.get_ndarray().extent(4) == num_dishes_per_cylinder); // dishes

    const auto& voltage_meta = voltage.get_metadata();
    const auto& frb1_beams_meta = frb1_beams.get_metadata();

    // pirate uses a uint8_t pointer for pairs of 4bit ints
    // pirate uses a float16 pointer for pairs of float16 that form a complex<float16>
    pirate::launch_chime_frb_beamform(
        reinterpret_cast<const uint8_t*>(voltage_memory + T_offset), map_memory, co_memory,
        reinterpret_cast<__half*>(frb1_beams_memory + Tfrb_offset), gains_memory,
        static_cast<long>(T), static_cast<long>(num_frequencies), device.getStream(cuda_stream_id));
#ifdef DEBUGGING
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
#endif
#if 0
    n2k::launch_s12_kernel((ulong*)(rfi_S012_memory + S_stride + Trfi_offset),
                           (const uint8_t*)(voltage_memory + T_offset), T, 0, Tsize,
                           num_frequencies, num_dishes * num_polarizations, rfi_downsampling_factor,
                           F_stride, offset_encoded, device.getStream(cuda_stream_id));
#endif
#ifdef DEBUGGING
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));
#endif

    // There is no poison value
    // if (poison_buffers)
    //     frb1_beams.check_for_poison(0xff);

    return record_end_event();
}

void cudaCHIMEFRBBeamform::finalize_frame() {
    // Advance the ring buffers
    voltage.finish_read();
    frb1_beams.finish_write();

    cudaCommand::finalize_frame();
}
